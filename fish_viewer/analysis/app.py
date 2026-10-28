"""A focused scan → cells → signals → A/B distances → Imaris desktop app."""
import json
from pathlib import Path
import time

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..app import ChannelPanel, Job, STYLE
from ..models import LibraryEntry
from ..reader import ImarisReader, entry_for_path, load_dataset, scan_library
from ..channel_hints import membrane_stain_channels
from ..signals.overlays import read_signals
from ..views import SliceView, VolumeView
from ..display import auto_adjust
from ..runtime_profile import configure_device_selector
from .controller import AnalysisController, default_models, default_refinement_model, default_proposal_model, migrate_bundled_defaults, sensitive_stardist_model, STAGES


class AnalysisWindow(QtWidgets.QMainWindow):
    def __init__(self, off_screen=False, settings=None):
        super().__init__()
        self.setWindowTitle('FISH Analysis — cells, signals and distances')
        self.resize(1180, 780)
        self.setStyleSheet(STYLE)
        self.off_screen = off_screen
        self.settings = settings or QtCore.QSettings('Cell NN', 'FISH Analysis')
        try:
            self.channel_roles=json.loads(self.settings.value('channel_roles','{}'))
            if not isinstance(self.channel_roles,dict):self.channel_roles={}
        except (TypeError,ValueError):
            self.channel_roles={}
        self._source_image=None
        self.controller = AnalysisController(self)
        self.controller.progress.connect(self.processing_progress)
        self.controller.completed.connect(self.completed)
        self.controller.failed.connect(self.failed)
        self.controller.cancelled.connect(self.cancelled)
        self.documents = {}
        self._document_key = None
        self.dataset = None
        self.result_path = None
        self.result_report = None
        self.volume_view = None
        self.channel_dialog = None
        self.full_detail = False
        self.interpolation = self.settings.value('display_interpolation', True, type=bool)
        self.loading = False
        self.running = False
        self._generation = 0
        self._jobs = set()
        self.queue = []
        self.scan_number = 0
        self._build()
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.tick)
        self.refresh_timer = QtCore.QTimer(self)
        self.refresh_timer.setSingleShot(True)
        self.refresh_timer.setInterval(100)
        self.refresh_timer.timeout.connect(self.refresh_images)
        self.sync_buttons()

    def _button(self, text, callback, layout):
        button = QtWidgets.QPushButton(text)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    def _build(self):
        splitter = QtWidgets.QSplitter()
        self.setCentralWidget(splitter)
        self.sidebar = QtWidgets.QScrollArea()
        self.sidebar.setWidgetResizable(True)
        self.sidebar.setMinimumWidth(320)
        self.sidebar.setMaximumWidth(410)
        body = QtWidgets.QWidget()
        left = QtWidgets.QVBoxLayout(body)
        title = QtWidgets.QLabel('FISH Analysis')
        title.setStyleSheet('font-size:22px;font-weight:600;padding:5px;')
        left.addWidget(title)
        description = QtWidgets.QLabel('Cell contours, signals and shortest 3D distances to the cell border.')
        description.setWordWrap(True)
        left.addWidget(description)
        row = QtWidgets.QHBoxLayout()
        self.add_button = self._button('Add scans…', self.open_files, row)
        self.folder_button = self._button('Add folder…', self.open_folder, row)
        left.addLayout(row)
        self.scans = QtWidgets.QListWidget()
        self.scans.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.scans.setFixedHeight(135)
        self.scans.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scans.currentItemChanged.connect(self.scan_selected)
        self.scans.setToolTip('Select one scan, or use Ctrl/Shift to select several for sequential processing.')
        left.addWidget(self.scans)
        row = QtWidgets.QHBoxLayout()
        self.remove_button = self._button('Remove selected', self.remove_selected, row)
        self.select_button = self._button('Select all', self.scans.selectAll, row)
        left.addLayout(row)
        self.retry_button = self._button('Select unfinished scans', self.select_unfinished, left)
        left.addWidget(QtWidgets.QLabel('Save results in'))
        row = QtWidgets.QHBoxLayout()
        default_output = Path(QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.StandardLocation.DocumentsLocation)) / 'FISH Analysis'
        self.output = QtWidgets.QLineEdit(self.settings.value('output', str(default_output)))
        row.addWidget(self.output, 1)
        self.output_button = self._button('Browse…', lambda: self.choose_folder(self.output), row)
        left.addLayout(row)
        self.options_button = QtWidgets.QToolButton()
        self.options_button.setText('Pipeline and settings')
        self.options_button.setCheckable(True)
        self.options_button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.options_button.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        left.addWidget(self.options_button)
        self.options = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(self.options)
        migrate_bundled_defaults(self.settings)
        cells, signals = default_models()
        # Fixed model folders belong to presets; they are not model pickers.
        self.cell_model = QtWidgets.QLineEdit(str(cells))
        self.signal_model = QtWidgets.QLineEdit(str(signals))
        self.boundary_model = QtWidgets.QLineEdit('')
        self.boundary_model.setPlaceholderText('Optional 3D U-Net model folder')
        self.pipeline = QtWidgets.QComboBox()
        self.pipeline.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.pipeline.setMinimumContentsLength(12)
        self.pipeline.setMinimumWidth(0)
        self.pipeline.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,QtWidgets.QSizePolicy.Policy.Fixed)
        for label, key in (('Both-detector consensus (default)', 'consensus'), ('Fine-tuned StarDist 2D', 'stardist2d')):
            self.pipeline.addItem(label, key)
        self.pipeline.setToolTip('Each pipeline uses its current bundled models and the signal-center U-Net. No training is performed.')
        form.addRow('Pipeline', self.pipeline)
        self.pipeline.currentIndexChanged.connect(self.select_pipeline)
        selected = self.pipeline.findData(self.settings.value('pipeline', 'consensus'))
        self.pipeline.setCurrentIndex(max(0, selected))
        self.select_pipeline()
        self.roundness = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.roundness.setRange(75, 100)
        saved_roundness = int(self.settings.value('minimum_roundness_percent', 82))
        self.roundness.setValue(saved_roundness if saved_roundness >= 75 else 82)
        self.roundness.setToolTip('Minimum physical 3D sphericity (75–100%). Higher values reject less round cells. Accepted contours are not reshaped.')
        self.roundness_value = QtWidgets.QLabel()
        self.roundness.valueChanged.connect(lambda v: self.roundness_value.setText(f'{v}%'))
        self.roundness_value.setText(f'{self.roundness.value()}%')
        row = QtWidgets.QHBoxLayout(); row.addWidget(self.roundness, 1); row.addWidget(self.roundness_value)
        form.addRow('Minimum roundness', row)
        self.pair_distance = QtWidgets.QDoubleSpinBox()
        self.pair_distance.setRange(.001, 10000.)
        self.pair_distance.setDecimals(3)
        self.pair_distance.setSingleStep(.1)
        self.pair_distance.setSuffix(' µm')
        self.pair_distance.setValue(float(self.settings.value('signal_pair_max_distance_um', 1.)))
        self.pair_distance.setToolTip('Within each cell and time point, pair closest signals once per channel pair. Omit pairs farther apart than this limit.')
        self.pair_distance.valueChanged.connect(self.pairing_changed)
        form.addRow('Signal-pair limit', self.pair_distance)
        self.device = QtWidgets.QComboBox()
        configure_device_selector(self.device, self.settings.value('device', 'auto'))
        form.addRow('Device', self.device)
        left.addWidget(self.options)
        self.options.hide()
        self.options_button.toggled.connect(lambda visible: (
            self.options.setVisible(visible), self.options_button.setArrowType(
                QtCore.Qt.ArrowType.DownArrow if visible else QtCore.Qt.ArrowType.RightArrow)))
        self.roles_button = self._button('Cell / signal channels…', self.show_channels, left)
        self.processing_summary = QtWidgets.QLabel()
        self.processing_summary.setWordWrap(True)
        left.addWidget(self.processing_summary)
        self.pipeline.currentIndexChanged.connect(self.update_processing_summary)
        self.device.currentIndexChanged.connect(self.update_processing_summary)
        row = QtWidgets.QHBoxLayout()
        self.run_button = self._button('Analyze selected', self.start, row)
        self.run_button.setStyleSheet('background:#205e58;font-weight:bold;padding:10px;')
        self.cancel_button = self._button('Cancel', self.cancel, row)
        left.addLayout(row)
        self.phase = QtWidgets.QLabel('Add a scan to begin.')
        self.phase.setWordWrap(True)
        self.phase.setStyleSheet('font-weight:600;')
        left.addWidget(self.phase)
        self.overall = QtWidgets.QProgressBar()
        self.overall.setRange(0, 5000)
        self.overall.setValue(0)
        self.overall.setFormat('Ready')
        left.addWidget(self.overall)
        self.stage_progress = QtWidgets.QProgressBar()
        self.stage_progress.setRange(0, 1000)
        self.stage_progress.setValue(0)
        left.addWidget(self.stage_progress)
        self.elapsed = QtWidgets.QLabel('')
        left.addWidget(self.elapsed)
        self.summary = QtWidgets.QLabel('Results are saved as an annotated Imaris image and Excel workbook with cell volumes and distances.')
        self.summary.setWordWrap(True)
        left.addWidget(self.summary)
        self.saved_button = self._button('Open saved analysis…', self.open_saved, left)
        self.results_button = self._button('Open result folder', self.open_result_folder, left)
        self.excel_button = self._button('Export Excel…', self.export_excel, left)
        note = QtWidgets.QLabel('Processes time point 1. Cells without signals are removed. A = signal; B = nearest cell border.')
        note.setWordWrap(True)
        note.setObjectName('muted')
        left.addWidget(note)
        left.addStretch()
        for label in body.findChildren(QtWidgets.QLabel):
            if label.wordWrap():
                label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        self.sidebar.setWidget(body)
        splitter.addWidget(self.sidebar)

        right = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(right)
        self.heading = QtWidgets.QLabel('No scan loaded')
        self.heading.setWordWrap(True)
        self.heading.setObjectName('heading')
        self.heading.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        layout.addWidget(self.heading)
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs, 1)
        image_page = QtWidgets.QWidget()
        image_layout = QtWidgets.QVBoxLayout(image_page)
        row = QtWidgets.QHBoxLayout()
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(['2D layers', '3D volume'])
        self.mode.currentIndexChanged.connect(self.change_mode)
        row.addWidget(self.mode)
        self.fit_button = self._button('Fit', self.fit_view, row)
        self.channels_button = self._button('Channels…', self.show_channels, row)
        self.auto_display_button = self._button('Auto display', self.auto_display, row)
        self.smooth_display = QtWidgets.QCheckBox('Smooth')
        self.smooth_display.setToolTip('Interpolate image display: bilinear in 2D and trilinear in 3D.')
        self.smooth_display.setChecked(self.interpolation)
        self.smooth_display.toggled.connect(self.set_interpolation)
        row.addWidget(self.smooth_display)
        row.addStretch()
        self.show_cells, self.show_points, self.show_lines = [QtWidgets.QCheckBox(name) for name in ('Cells', 'A/B', 'Lines')]
        for box in (self.show_cells, self.show_points, self.show_lines):
            box.setChecked(True)
            box.toggled.connect(self.refresh_annotations)
            row.addWidget(box)
        image_layout.addLayout(row)
        self.views = QtWidgets.QStackedWidget()
        self.slice_view = SliceView()
        self.slice_view.set_interpolation(self.interpolation)
        self.slice_view.stepped.connect(lambda delta: self.slice.setValue(self.slice.value()+delta))
        self.views.addWidget(self.slice_view)
        image_layout.addWidget(self.views, 1)
        self.slice_row = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(self.slice_row)
        row.setContentsMargins(0, 0, 0, 0)
        self.axis = QtWidgets.QComboBox()
        self.axis.addItems(['XY / Z', 'XZ / Y', 'YZ / X'])
        self.axis.currentIndexChanged.connect(self.change_axis)
        row.addWidget(self.axis)
        self.slice = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slice.valueChanged.connect(self.change_slice)
        row.addWidget(self.slice, 1)
        self.slice_label = QtWidgets.QLabel('')
        row.addWidget(self.slice_label)
        image_layout.addWidget(self.slice_row)
        self.selection_info = QtWidgets.QLabel('Select a row in Distances to locate its A/B pair.')
        self.selection_info.setWordWrap(True)
        image_layout.addWidget(self.selection_info)
        self.tabs.addTab(image_page, 'Image')
        from .editor import AnnotationEditor
        self.editor = AnnotationEditor(self)
        annotation_panel = QtWidgets.QScrollArea()
        annotation_panel.setWidgetResizable(True)
        annotation_panel.setMinimumWidth(265)
        annotation_panel.setWidget(self.editor)
        self.annotation_dock = QtWidgets.QDockWidget('Annotations and editing', self)
        self.annotation_dock.setWidget(annotation_panel)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.annotation_dock)
        self.annotation_dock.hide()
        image_layout.insertWidget(1, self.editor.toolbar)
        annotation_toggle = self.annotation_dock.toggleViewAction()
        annotation_toggle.setText('Annotations')
        self.editor.toolbar.addAction(annotation_toggle)
        self.distances = self._table(['Signal', 'Cell', 'Channel', 'Distance (µm)', 'Score',
                                      'A X', 'A Y', 'A Z', 'B X', 'B Y', 'B Z', 'Note'])
        self.distances.itemSelectionChanged.connect(self.focus_measurement)
        self.distances.itemDoubleClicked.connect(lambda _: self.tabs.setCurrentIndex(0))
        self.tabs.addTab(self.distances, 'Distances')
        self.counts = self._table(['Cell', 'Channel', 'Channel name', 'Signals', 'Volume (µm³)'])
        self.counts.itemSelectionChanged.connect(self.focus_cell)
        self.counts.itemDoubleClicked.connect(lambda _: self.tabs.setCurrentIndex(0))
        self.tabs.addTab(self.counts, 'Cell counts')
        overview = QtWidgets.QWidget()
        overview_layout = QtWidgets.QVBoxLayout(overview)
        self.results_overview = QtWidgets.QLabel('Analyze a scan or open a saved analysis to see results.')
        self.results_overview.setWordWrap(True)
        self.results_overview.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        overview_layout.addWidget(self.results_overview)
        self.overview_folder = self._button('Open exports folder', self.open_result_folder, overview_layout)
        self.overview_excel = self._button('Export current measurements to Excel…', self.export_excel, overview_layout)
        self.overview_imaris = self._button('Export edited Imaris…', self.editor.export_scene, overview_layout)
        overview_layout.addStretch()
        self.tabs.addTab(overview, 'Results')
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 840])
        self.statusBar().showMessage('Original scans are read-only; export saves edited copies.')

    @staticmethod
    def _table(columns):
        table = QtWidgets.QTableWidget(0, len(columns))
        table.setHorizontalHeaderLabels(columns)
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        table.verticalHeader().hide()
        table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setStretchLastSection(True)
        return table

    def sync_buttons(self):
        self.update_processing_summary()
        self.overview_folder.setEnabled(self.result_path is not None)
        editable = self.dataset is not None and not self.loading and not self.running
        self.overview_excel.setEnabled(editable)
        self.overview_imaris.setEnabled(editable)
        for widget in (self.add_button, self.folder_button, self.remove_button, self.select_button,
                       self.scans, self.retry_button, self.output, self.output_button, self.options, self.saved_button):
            widget.setEnabled(not self.running)
        self.run_button.setEnabled(not self.running and not self.loading and self.scans.count() > 0)
        self.cancel_button.setEnabled(self.running)
        self.results_button.setEnabled(self.result_path is not None)
        self.excel_button.setEnabled(self.dataset is not None and not self.loading and not self.running)
        self.editor.setEnabled(editable)
        self.editor.toolbar.setEnabled(editable)
        self.auto_display_button.setEnabled(editable)
        for widget in (self.roles_button, self.channels_button, self.fit_button, self.mode, self.axis, self.slice):
            widget.setEnabled(self.dataset is not None and not self.running and not self.loading)

    def channel_summary(self, record):
        if record.get('metadata_error'):
            return 'Cannot read channel metadata', True
        channels = record.get('channels', [])
        roles = self.channel_roles.get(str(Path(record['image']).resolve()), {})
        cell = roles.get('cell')
        signals = roles.get('signals')
        candidates = membrane_stain_channels(record['image'], channels)
        warning = None
        if cell is None:
            cell_text = 'automatic' + (f' (declared channel {candidates[0]+1})' if len(candidates) == 1 else '')
            if len(candidates) != 1:
                warning = 'Check channels: automatic cell assignment needs review'
        else:
            cell_text = str(cell + 1)
            if cell >= len(channels):
                warning = 'Check channels: selected cell channel is missing'
        signal_text = 'automatic (non-cell channels)' if signals is None else ', '.join(str(i+1) for i in signals)
        if signals is not None and (not signals or any(i >= len(channels) or i == cell for i in signals)):
            warning = 'Check channels: choose valid signal channels'
        if len(channels) < 2:
            warning = 'Check channels: at least two channels are needed'
        return warning or f'Cell: {cell_text}; signals: {signal_text}', bool(warning)

    def refresh_scan_item(self, item):
        record = item.data(QtCore.Qt.ItemDataRole.UserRole)
        roles, warning = self.channel_summary(record)
        state = record.get('status', 'pending').capitalize()
        badge = ' · Check channels' if warning else ''
        item.setText(f'{state}{badge} · {Path(record["image"]).name}')
        item.setToolTip(record['image'] + '\n' + roles + ('\n' + record['error'] if record.get('error') else ''))
        item.setForeground(QtGui.QBrush(QtGui.QColor('#e9b85f' if warning or state == 'Failed' else '#d8e3ed')))

    def set_scan_status(self, item, status, error=None):
        record = item.data(QtCore.Qt.ItemDataRole.UserRole)
        record['status'] = status
        record['error'] = error
        item.setData(QtCore.Qt.ItemDataRole.UserRole, record)
        self.refresh_scan_item(item)

    def update_processing_summary(self, *_):
        roles = 'Add a scan to review channel roles.'
        item = self.scans.currentItem()
        if item:
            roles = self.channel_summary(item.data(QtCore.Qt.ItemDataRole.UserRole))[0]
        self.processing_summary.setText(f'{self.pipeline.currentText()}\nDevice: {self.device.currentText()}\n{roles}')

    def update_results_overview(self, report):
        sections = []
        for name, points, cells in (
            ('This run', report.get('points', []), report.get('cell_volumes', [])),
            ('Pre-existing', report.get('manual_measurements', {}).get('points', []), report.get('manual_measurements', {}).get('cells', [])),
        ):
            distances = [p['distance_um'] for p in points if p.get('distance_um') is not None and np.isfinite(p['distance_um'])]
            text = f'{name}: {len(cells)} cells · {len(points)} signals · {len(distances)} measured distances'
            if distances:
                text += f'\nA → B distance (µm): median {np.median(distances):.4f}; mean {np.mean(distances):.4f}; range {min(distances):.4f}–{max(distances):.4f}'
            sections.append(text)
        filtering = report.get('signal_filter', {})
        if report.get('signal_pairs'):
            sections.append(f"{len(report['signal_pairs'])} signal-to-signal pairs within "
                            f"{report['signal_pairing']['maximum_distance_um']:g} µm.")
        if 'removed_cell_count' in filtering:
            sections.append(f'{filtering["removed_cell_count"]} cells without signals removed.')
        if self.result_path:
            sections.append(f'Saved analysis: {self.result_path.parent}\nExports: {self.result_path.parent / "export"}')
        else:
            sections.append('Current annotations; export to save a copy.')
        self.results_overview.setText('\n\n'.join(sections))

    def choose_folder(self, field):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Choose folder', field.text())
        if path:
            field.setText(path)

    def open_files(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, 'Add Imaris scans', '', 'Imaris (*.ims *.imx *.IMS *.IMX)')
        if paths:
            self.add_paths(paths)

    def open_folder(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Add scans from folder and subfolders')
        if path:
            self.add_paths([path])

    def background(self, function, completed, failed=None):
        job = Job(function)
        self._jobs.add(job)
        job.signals.result.connect(completed)
        job.signals.error.connect(failed or self.error)
        job.signals.finished.connect(lambda: self._jobs.discard(job))
        QtCore.QThreadPool.globalInstance().start(job)

    def add_paths(self, paths):
        if self.running:
            return
        self.phase.setText('Reading scan list…')
        def read(_):
            entries = []
            for path in paths:
                entries.extend(scan_library(path)[0] if Path(path).is_dir() else [entry_for_path(path)])
            discovered = sorted({(str(e.image.resolve()), str(e.scene.resolve()) if e.scene else None) for e in entries}, key=lambda e: e[0])
            checked = []
            for image, scene in discovered:
                try:
                    channels = ImarisReader(image).metadata.channels
                    checked.append((image, scene, channels, None))
                except Exception as exc:
                    checked.append((image, scene, [], str(exc)))
            return checked
        def done(paths):
            known = {(r['image'],r.get('scene')) for r in (self.scans.item(i).data(QtCore.Qt.ItemDataRole.UserRole) for i in range(self.scans.count()))}
            first = None
            for path, scene, channels, problem in paths:
                if (path,scene) in known:
                    continue
                known.add((path,scene))
                label=Path(path).name
                if scene and Path(scene).stem!=Path(path).stem: label+=' · '+Path(scene).name
                item = QtWidgets.QListWidgetItem(label)
                item.setToolTip(path)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, dict(image=path, scene=scene, channels=channels, status='pending', metadata_error=problem))
                self.refresh_scan_item(item)
                self.scans.addItem(item)
                if first is None:
                    first = item
            if first:
                self.scans.setCurrentItem(first)
            self.phase.setText('Select scans, then click Analyze selected.')
            self.sync_buttons()
        self.background(read, done)

    def remove_selected(self):
        for item in self.scans.selectedItems():
            self.scans.takeItem(self.scans.row(item))
        self.sync_buttons()

    def select_unfinished(self):
        for row in range(self.scans.count()):
            item = self.scans.item(row)
            item.setSelected(item.data(QtCore.Qt.ItemDataRole.UserRole).get('status') != 'completed')

    def scan_selected(self, item, previous=None):
        if item is None or self.running:
            return
        record = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if record.get('result'):
            self.load_result(record['result'])
        else:
            self._load(record.get('scene') or record['image'])

    def _load(self, image, result_path=None):
        self._source_image=str(entry_for_path(image).image.resolve())
        if result_path:
            saved=json.loads(Path(result_path).read_text())
            self._source_image=saved.get('source',self._source_image)
        self._generation += 1
        generation = self._generation
        self.loading = True
        self.sync_buttons()
        self.heading.setText('Loading ' + Path(image).name + '…')
        def read(_):
            if result_path:
                manifest = json.loads(Path(result_path).read_text(encoding='utf-8'))
                root = Path(result_path).parent
                dataset = load_dataset(LibraryEntry(root / manifest['image'], root / manifest['scene'] if manifest.get('scene') else None))
                _, report = read_signals(root / manifest['signal_report'])
                from ..annotation_state import activate_run, annotation_report
                saved = annotation_report(dataset)
                activate_run(dataset.scene, manifest.get('annotation_run_id') or saved.get('annotation_run_id'))
                for key in ('manual_measurements', 'signal_pairing'):
                    if key in saved:
                        report[key] = saved[key]
            else:
                dataset = load_dataset(entry_for_path(image))
                report = None
            for channel, volume in zip(dataset.metadata.channels, dataset.volumes):
                auto_adjust(channel, volume)
            return dataset, report
        def done(value):
            if generation != self._generation:
                return
            self.loading = False
            self.result_path = Path(result_path) if result_path else None
            self.set_dataset(*value)
            if not self.running:
                self.phase.setText('Saved analysis ready' if result_path else 'Select scans, then click Analyze selected.')
                self.overall.setValue(self.overall.maximum() if result_path else 0)
                self.overall.setFormat('Complete' if result_path else 'Ready')
            self.sync_buttons()
        def failed(message):
            if generation == self._generation:
                self.loading = False
                self.sync_buttons()
                self.error(message)
        self.background(read, done, failed)

    @staticmethod
    def document_key(dataset):
        active_run = next((obj.run_id for objects in (dataset.scene.surfaces, dataset.scene.measurements)
                           for obj in objects if obj.source != 'manual' and obj.run_id), None)
        return (str(dataset.metadata.path), str(dataset.scene.source or dataset.metadata.path), active_run)

    def set_dataset(self, dataset, report=None):
        # Slice/axis updates emit redraws immediately. Old selections and queued
        # gestures must be detached before any view sees the replacement scene.
        self.editor.reset_interaction()
        if self.editor.document is not None and self.dataset is not None:
            self.documents[self._document_key] = self.editor.document
            from dataclasses import replace
            # Keep annotations/undo, not every full scan volume in memory.
            self.editor.document.dataset = replace(self.dataset, volumes=[])
        self._document_key = self.document_key(dataset)
        cached=self.documents.get(self._document_key)
        if cached is not None:
            dataset.scene=cached.dataset.scene
        self.dataset, self.result_report = dataset, report
        if self.channel_dialog:
            self.channel_dialog.close()
            self.channel_dialog.deleteLater()
            self.channel_dialog = None
        self.heading.setText(dataset.metadata.path.name)
        self.slice_view.set_dataset(dataset)
        self._volume_dataset = None
        self.change_axis(self.axis.currentIndex())
        self.change_mode(self.mode.currentIndex())
        self.distances.blockSignals(True)
        self.distances.setRowCount(0)
        self.counts.setRowCount(0)
        self.selection_info.setText('Select a row in Distances to locate its A/B pair.')
        if report:
            for i, point in enumerate(report['points']):
                self.distances.insertRow(i)
                values = [point['signal_id'], point['cell_id'], point['channel']+1,
                          f'{point["distance_um"]:.4f}', f'{point["score"]:.4f}',
                          *[f'{v:.4f}' for v in point['xyz']], *[f'{v:.4f}' for v in point['border_xyz']],
                          'Image-edge contact' if point.get('border_at_image_edge') else '']
                for j, value in enumerate(values):
                    item = QtWidgets.QTableWidgetItem(str(value))
                    if j == 0:
                        item.setData(QtCore.Qt.ItemDataRole.UserRole, point)
                    self.distances.setItem(i, j, item)
            for i, row in enumerate(report['counts']):
                self.counts.insertRow(i)
                volume = next((v['volume_um3'] for v in report.get('cell_volumes', []) if v['cell_id'] == row['cell_id']), None)
                for j, value in enumerate([row['cell_id'], row['channel']+1, row['channel_name'], row['count'],
                                          f'{volume:.3f}' if volume is not None else 'Unavailable']):
                    self.counts.setItem(i, j, QtWidgets.QTableWidgetItem(str(value)))
            filtering = report['signal_filter']
            self.summary.setText(f'{filtering["kept_cell_count"]} cells kept · {len(report["points"])} signals and distances\n'
                                 f'{filtering["removed_cell_count"]} cells without signals removed.')
        else:
            self.summary.setText('Ready for automatic cell contours, signals and A/B measurements.')
        self.distances.blockSignals(False)
        self.refresh_annotations()
        self.editor.set_dataset(dataset, report, cached)
        self.update_processing_summary()

    def start(self):
        if self.running or self.loading:
            return
        if not self.output.text().strip():
            self.error('Choose a folder for analysis results.')
            return
        items = self.scans.selectedItems()
        if not items:
            self.error('Select at least one scan.')
            return
        self.queue = [self.scans.row(item) for item in items]
        for row in self.queue:
            self.set_scan_status(self.scans.item(row), 'pending')
        self.batch_failures = 0
        self.total_scans = len(self.queue)
        self.scan_number = 0
        self.running = True
        self.started = time.monotonic()
        self.cancel_button.setText('Cancel')
        self.timer.start()
        self.overall.setRange(0, self.total_scans*5000)
        self.overall.setValue(0)
        for key, widget in [('output', self.output), ('cell_model', self.cell_model),
                            ('boundary_model', self.boundary_model), ('signal_model', self.signal_model)]:
            self.settings.setValue(key, widget.text())
        self.settings.setValue('pipeline', self.pipeline.currentData())
        self.settings.setValue('minimum_roundness_percent', self.roundness.value())
        self.settings.setValue('device', self.device.currentData())

        self.sync_buttons()
        self.next_scan()

    def select_pipeline(self, *_):
        key = self.pipeline.currentData()
        root = Path(__file__).resolve().parents[2] / 'models'
        self.cell_model.setText(str(root / ('cells-stardist2d' if key == 'stardist2d' else 'cells-sensitive')))
        self.signal_model.setText(str(default_models()[1]))
        self.boundary_model.setText(default_refinement_model() if key == 'consensus' else '')


    def next_scan(self):
        if not self.queue:
            self.running = False
            self.timer.stop()
            self.phase.setText(f'Analysis finished · {self.batch_failures} failed' if self.batch_failures else 'Analysis complete')
            self.overall.setValue(self.overall.maximum())
            self.overall.setFormat('Complete')
            self.sync_buttons()
            return
        self.active_row = self.queue.pop(0)
        item = self.scans.item(self.active_row)
        self.scan_number += 1
        self.active_image = item.data(QtCore.Qt.ItemDataRole.UserRole)['image']
        self.set_scan_status(item, 'running')
        try:
            roles=self.channel_roles.get(str(Path(self.active_image).resolve()),{})
            self.controller.start(self.active_image, self.output.text(), self.cell_model.text(),
                                  self.signal_model.text(), self.device.currentData(), self.boundary_model.text().strip() or None,
                                  roundness=self.roundness.value()/100,
                                  maximum_pair_distance=self.pair_distance.value(),
                                  pipeline=self.pipeline.currentData(),
                                  cell_channel=roles.get('cell'),signal_channels=roles.get('signals'),source_scene=item.data(QtCore.Qt.ItemDataRole.UserRole).get('scene'))
        except Exception as exc:
            self.failed(str(exc))

    def processing_progress(self, event):
        stage = event['stage']
        stages = event.get('stages', 5)
        total, current = event.get('total', 0), event.get('current', 0)
        name = event.get('stage_name', STAGES[min(stage-1,len(STAGES)-1)][1])
        self.phase.setText(f'{stage} / {stages} · {name}\n{event["phase"]}')
        self.overall.setFormat(f'Scan {self.scan_number} / {self.total_scans} · Step {stage} / {stages}')
        fraction = min(1., max(0., current/total)) if total > 0 else 0.
        self.overall.setValue((self.scan_number-1)*5000 + round((stage-1+fraction)/stages*5000))
        self.stage_progress.setRange(0, 1000 if total > 0 else 0)
        if total > 0:
            self.stage_progress.setValue(round(fraction*1000))
            self.stage_progress.setFormat(f'{current} / {total}')
        self.statusBar().showMessage(f'{Path(self.active_image).name} · {event["phase"]}')

    def completed(self, report):
        item = self.scans.item(self.active_row)
        record = item.data(QtCore.Qt.ItemDataRole.UserRole)
        record['result'] = report['report_path']
        item.setData(QtCore.Qt.ItemDataRole.UserRole, record)
        self.set_scan_status(item, 'completed')
        self.stage_progress.setRange(0, 1000)
        self.stage_progress.setValue(1000)
        self.stage_progress.setFormat('Saved')
        self.load_result(report['report_path'])
        QtCore.QTimer.singleShot(0, self.next_scan)

    def failed(self, message):
        if not self.running:
            self.error(message)
            return
        self.batch_failures += 1
        self.set_scan_status(self.scans.item(self.active_row), 'failed', message)
        self.phase.setText('Scan failed; continuing with the remaining scans.')
        self.stage_progress.setRange(0, 1000)
        self.statusBar().showMessage(str(message)[-1000:])
        QtCore.QTimer.singleShot(0, self.next_scan)

    def cancelled(self):
        self.queue.clear()
        self.running = False
        self.timer.stop()
        self.phase.setText('Cancelled — completed results were kept.')
        self.stage_progress.setRange(0, 1000)
        if hasattr(self, 'active_row'):
            self.set_scan_status(self.scans.item(self.active_row), 'cancelled')
        self.sync_buttons()

    def cancel(self):
        self.queue.clear()
        self.controller.cancel()
        self.cancel_button.setText('Stop now')
        self.phase.setText('Stopping after the current operation…')

    def tick(self):
        seconds = int(time.monotonic()-self.started)
        self.elapsed.setText(f'Elapsed {seconds//60:02d}:{seconds%60:02d}')

    def open_saved(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open saved analysis', self.output.text(), 'Analysis result (analysis-result.json)')
        if path:
            try:
                self.load_result(path)
            except Exception as exc:
                self.error(str(exc))

    def load_result(self, path):
        path = Path(path).resolve()
        report = json.loads(path.read_text(encoding='utf-8'))
        if report.get('engine') != 'fish-analysis' or report.get('status') != 'complete':
            raise ValueError('Choose an analysis-result.json from a completed analysis.')
        self._load(path.parent / report['image'], path)

    def export_excel(self):
        if self.dataset is None:
            return
        default = str(self.result_path.parent / (self.dataset.metadata.path.stem + '.xlsx')) if self.result_path else 'analysis.xlsx'
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Export volumes and distances', default, 'Excel workbook (*.xlsx)')
        if path:
            try:
                from ..workflow.excel import write_excel
                write_excel(Path(path).with_suffix('.xlsx'), self.editor.document.report())
                self.phase.setText('Excel workbook saved: ' + str(path))
            except Exception as exc:
                self.error(str(exc))

    def pairing_changed(self, value):
        self.settings.setValue('signal_pair_max_distance_um', value)
        if hasattr(self, 'editor') and self.editor.document:
            self.editor.document.set_pairing_distance(value)

    def open_result_folder(self):
        if self.result_path:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.result_path.parent / 'export')))

    def change_axis(self, axis):
        if self.dataset is None:
            return
        self.slice_view.axis = axis
        self.slice.blockSignals(True)
        self.slice.setRange(0, self.dataset.volumes[0].shape[axis]-1)
        self.slice.setValue(self.slice.maximum()//2)
        self.slice.blockSignals(False)
        self.change_slice(self.slice.value())
        self.slice_view.fit()

    def change_slice(self, index):
        if self.dataset is not None:
            self.slice_view.index = index
            self.slice_view.update_image()
            self.slice_label.setText(f'{index+1} / {self.slice.maximum()+1}')

    def change_mode(self, index):
        if self.dataset is None:
            return
        if index:
            if self.volume_view is None:
                self.volume_view = VolumeView(off_screen=self.off_screen)
                self.volume_view.full_detail = self.full_detail
                self.volume_view.interpolation = self.interpolation
                self.views.addWidget(self.volume_view)
            if self._volume_dataset is not self.dataset:
                self.volume_view.set_dataset(self.dataset)
                self._volume_dataset = self.dataset
            self.views.setCurrentWidget(self.volume_view)
        else:
            self.views.setCurrentWidget(self.slice_view)
        self.slice_row.setVisible(index == 0)
        self.refresh_annotations()

    def show_channels(self):
        if not self.dataset:
            return
        if self.channel_dialog:
            self.channel_dialog.show()
            self.channel_dialog.raise_()
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle('Channel roles and display')
        dialog.resize(370, 620)
        layout = QtWidgets.QVBoxLayout(dialog)
        roles_box=QtWidgets.QGroupBox('Processing roles for this scan')
        roles_layout=QtWidgets.QVBoxLayout(roles_box)
        note=QtWidgets.QLabel('Choose one cell channel and the signal channels. Other channels are ignored for signal detection. These choices do not change display colors.')
        note.setWordWrap(True);roles_layout.addWidget(note)
        self.cell_channel_choice=QtWidgets.QComboBox()
        self.cell_channel_choice.addItem('Cell channel: automatic',None)
        labels=[f'{c.name}' + (f' · {c.excitation_nm:g} nm' if c.excitation_nm is not None else '') for c in self.dataset.metadata.channels]
        for i,c in enumerate(self.dataset.metadata.channels):
            swatch=QtGui.QPixmap(12,12);swatch.fill(QtGui.QColor.fromRgbF(*c.color))
            self.cell_channel_choice.addItem(QtGui.QIcon(swatch),f'Cell channel: {i+1} · {labels[i]}',i)
        source=self._source_image or str(self.dataset.metadata.path.resolve())
        saved=self.channel_roles.get(source,{})
        self.cell_channel_choice.setCurrentIndex(max(0,self.cell_channel_choice.findData(saved.get('cell'))))
        roles_layout.addWidget(self.cell_channel_choice)
        self.signal_channel_checks=[]
        names=[c.name for c in self.dataset.metadata.channels]
        chosen=saved.get('signals')
        if chosen is None:chosen=list(range(len(names)))
        for i,name in enumerate(names):
            check=QtWidgets.QCheckBox(f'Signal channel: {i+1} · {labels[i]}')
            swatch=QtGui.QPixmap(12,12);swatch.fill(QtGui.QColor.fromRgbF(*self.dataset.metadata.channels[i].color))
            check.setIcon(QtGui.QIcon(swatch))
            check.setChecked(i in chosen);roles_layout.addWidget(check)
            self.signal_channel_checks.append(check)
        def save_roles(*_):
            cell=self.cell_channel_choice.currentData()
            for i,check in enumerate(self.signal_channel_checks):
                check.setEnabled(i!=cell)
                if i==cell:check.setChecked(False)
            signals=[i for i,c in enumerate(self.signal_channel_checks) if c.isChecked() and i!=cell]
            self.channel_roles[source]=dict(cell=cell,signals=signals,names=names)
            self.settings.setValue('channel_roles',json.dumps(self.channel_roles))
            for row in range(self.scans.count()):
                self.refresh_scan_item(self.scans.item(row))
            self.update_processing_summary()
        self.cell_channel_choice.currentIndexChanged.connect(save_roles)
        for check in self.signal_channel_checks:check.toggled.connect(save_roles)
        for i,check in enumerate(self.signal_channel_checks):
            if i==self.cell_channel_choice.currentData():check.setChecked(False);check.setEnabled(False)
        roles_box.setEnabled(not self.running)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        body = QtWidgets.QWidget()
        channels = QtWidgets.QVBoxLayout(body)
        channels.addWidget(roles_box)
        for i, (channel, volume) in enumerate(zip(self.dataset.metadata.channels, self.dataset.volumes)):
            panel = ChannelPanel(channel, volume, i)
            panel.changed.connect(lambda: self.refresh_timer.start())
            channels.addWidget(panel)
        scroll.setWidget(body)
        layout.addWidget(scroll)
        detail = QtWidgets.QCheckBox('Full volume detail in 3D')
        detail.setChecked(self.full_detail)
        detail.setToolTip('Use all voxels instead of a downsampled 3D preview. Measurement coordinates are always full precision.')
        def changed(value):
            self.full_detail = value
            if self.volume_view:
                self.volume_view.full_detail = value
                self.refresh_images()
        detail.toggled.connect(changed)
        layout.addWidget(detail)
        self.channel_dialog = dialog
        dialog.show()

    def auto_display(self):
        if self.dataset is None:
            return
        for channel, volume in zip(self.dataset.metadata.channels, self.dataset.volumes):
            auto_adjust(channel, volume)
        if self.channel_dialog:
            for panel in self.channel_dialog.findChildren(ChannelPanel):
                panel.sync()
        self.refresh_images()

    def set_interpolation(self, enabled):
        self.interpolation = bool(enabled)
        self.settings.setValue('display_interpolation', self.interpolation)
        self.slice_view.set_interpolation(self.interpolation)
        if self.volume_view:
            self.volume_view.set_interpolation(self.interpolation)

    def refresh_images(self):
        if self.dataset is not None:
            self.slice_view.update_image()
            if self.volume_view and self._volume_dataset is self.dataset:
                self.volume_view.update_volume()

    def refresh_annotations(self):
        for view in (self.slice_view, self.volume_view):
            if view is None or view.dataset is not self.dataset:
                continue
            view.surfaces_visible = self.show_cells.isChecked()
            view.markers_visible = self.show_points.isChecked()
            view.lines_visible = self.show_lines.isChecked()
            if view is self.slice_view:
                view.update_overlays()
            else:
                view.update_annotations()

    def fit_view(self):
        if self.dataset is not None:
            (self.volume_view if self.mode.currentIndex() else self.slice_view).fit()

    def focus_cell(self):
        item = self.counts.item(self.counts.currentRow(), 0)
        if item is None or self.dataset is None:
            return
        uid = item.data(QtCore.Qt.ItemDataRole.UserRole)
        index = next((i for i, surface in enumerate(self.dataset.scene.surfaces) if surface.uid == uid), None)
        if index is not None:
            self.editor.select_annotation(('surface', index))

    def focus_measurement(self):
        row = self.distances.currentRow()
        if row < 0 or self.dataset is None:
            return
        point = self.distances.item(row, 0).data(QtCore.Qt.ItemDataRole.UserRole)
        if not point:
            return
        index = next((i for i, m in enumerate(self.dataset.scene.measurements)
                      if m.uid == point.get('annotation_id')), None)
        if index is None:
            return
        self.slice_view.selected = ('measurement', index)
        self.editor.select_annotation(('measurement', index))
        if self.volume_view:
            self.volume_view.selected = ('measurement', index)
        xyz = np.asarray(point['xyz'])
        axis = 2-self.slice_view.axis
        self.slice.setValue(int(round((xyz[axis]-self.dataset.center_origin[axis])/self.dataset.spacing[axis])))
        surface = next((s for s in self.dataset.scene.surfaces if s.uid == point.get('cell_id')), None)
        coordinates = surface.vertices if surface is not None else np.asarray([point['xyz'], point.get('border_xyz', point['xyz'])])
        low, high = coordinates.min(0)-.8, coordinates.max(0)+.8
        u, v = self.slice_view.projection_axes
        self.slice_view.fitInView(QtCore.QRectF(low[u], low[v], high[u]-low[u], high[v]-low[v]), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        if self.volume_view and self._volume_dataset is self.dataset:
            self.volume_view.plotter.reset_camera(bounds=tuple(v for lo, hi in zip(low, high) for v in (lo, hi)))
        distance=point.get('distance_um')
        label = point.get('signal_name', point['signal_id'])
        self.selection_info.setText(f'{label} · A → B: {distance:.4f} µm' if distance is not None else str(label))
        self.refresh_annotations()

    def error(self, message):
        QtWidgets.QMessageBox.warning(self, 'FISH Analysis', str(message)[-3000:])

    def closeEvent(self, event):
        self._generation += 1
        self.timer.stop()
        self.refresh_timer.stop()
        self.controller.shutdown()
        if self.volume_view:
            self.volume_view.close()
        super().closeEvent(event)
