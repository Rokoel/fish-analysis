from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import traceback

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .models import Dataset, LibraryEntry, Scene
from .reader import entry_for_path, load_dataset, scan_library
from .views import SliceView, VolumeView
from .display import auto_adjust


STYLE = """
QMainWindow, QWidget { background: #111821; color: #e0e8f1; font-size: 12px; }
QToolBar { background: #18222e; border: 0; spacing: 8px; padding: 9px; }
QToolButton, QPushButton { background: #243243; border: 1px solid #35475c; border-radius: 5px; padding: 7px 10px; }
QToolButton:hover, QPushButton:hover { background: #30445a; border-color: #6a839b; }
QToolButton:checked, QPushButton:checked { background: #205e58; border-color: #4adbb7; color: #ffffff; }
QPushButton:disabled, QToolButton:disabled { color: #63758a; background: #192431; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background: #0e151e; border: 1px solid #35475c; border-radius: 4px; padding: 5px; }
QTreeWidget { background: #111821; border: 0; outline: 0; }
QTreeWidget::item { padding: 5px 3px; }
QTreeWidget::item:hover { background: #1e2c3b; }
QTreeWidget::item:selected, QTreeWidget::item:selected:!active { background: #fff06a; color: #111821; }
QTableWidget, QListWidget { selection-background-color: #fff06a; selection-color: #111821; }
QTableWidget::item:selected, QTableWidget::item:selected:!active,
QListWidget::item:selected, QListWidget::item:selected:!active { background: #fff06a; color: #111821; }
QGroupBox { border: 1px solid #304050; border-radius: 6px; margin-top: 12px; padding: 12px 8px 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #b9cadb; }
QTabWidget::pane { border: 1px solid #304050; }
QTabBar::tab { background: #1c2836; padding: 9px 14px; }
QTabBar::tab:selected { background: #2b454a; color: #64dfc4; }
QScrollArea { border: 0; }
QSlider::groove:horizontal { background: #2d3d50; height: 4px; border-radius: 2px; }
QSlider::handle:horizontal { background: #60d9bd; width: 13px; margin: -5px 0; border-radius: 6px; }
QSplitter::handle { background: #283646; width: 2px; }
QStatusBar { background: #18222e; color: #aebed0; }
QLabel#muted { color: #8fa3b9; }
QLabel#heading { font-size: 16px; font-weight: 600; }
QCheckBox { spacing: 7px; }
QToolTip { color: #e0e8f1; background: #26374b; border: 1px solid #536b85; }
"""


class JobSignals(QtCore.QObject):
    result = QtCore.Signal(object)
    error = QtCore.Signal(str)
    progress = QtCore.Signal(str)
    finished = QtCore.Signal()


class Job(QtCore.QRunnable):
    def __init__(self, function):
        super().__init__()
        self.function = function
        self.signals = JobSignals()

    def run(self):
        try:
            self.signals.result.emit(self.function(self.signals.progress.emit))
        except Exception:
            self.signals.error.emit(traceback.format_exc())
        finally:
            self.signals.finished.emit()


def label(text, muted=False):
    widget = QtWidgets.QLabel(text)
    if muted:
        widget.setObjectName('muted')
    return widget


def slider(minimum, maximum, value):
    control = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
    control.setRange(minimum, maximum)
    control.setValue(value)
    return control


class ChannelPanel(QtWidgets.QGroupBox):
    changed = QtCore.Signal()

    def __init__(self, channel, volume, index):
        super().__init__(f'{index + 1}  ·  {channel.name}')
        self.channel = channel
        self.volume = volume
        layout = QtWidgets.QFormLayout(self)
        row = QtWidgets.QHBoxLayout()
        self.visible = QtWidgets.QCheckBox('Visible')
        self.visible.setChecked(channel.visible)
        self.color = QtWidgets.QPushButton('Color')
        row.addWidget(self.visible)
        row.addStretch()
        row.addWidget(self.color)
        layout.addRow(row)
        self.low, self.high = QtWidgets.QDoubleSpinBox(), QtWidgets.QDoubleSpinBox()
        minimum, maximum = (float(np.iinfo(volume.dtype).min), float(np.iinfo(volume.dtype).max)) if np.issubdtype(volume.dtype, np.integer) else (float(np.nanmin(volume)), float(np.nanmax(volume)))
        self.maximum = max(maximum, minimum + 1)
        for control in (self.low, self.high):
            control.setRange(minimum, self.maximum)
            control.setDecimals(2)
            control.setKeyboardTracking(False)
        self.low.setValue(channel.low)
        self.high.setValue(channel.high)
        layout.addRow('Black level', self.low)
        layout.addRow('White level', self.high)
        self.brightness = slider(-100, 100, round(channel.brightness * 100))
        self.contrast = slider(10, 400, round(channel.contrast * 100))
        self.gamma = QtWidgets.QDoubleSpinBox()
        self.gamma.setRange(0.1, 5)
        self.gamma.setSingleStep(0.1)
        self.gamma.setValue(channel.gamma)
        self.gamma.setKeyboardTracking(False)
        self.opacity = slider(0, 100, round(channel.opacity * 100))
        layout.addRow('Brightness', self.brightness)
        layout.addRow('Contrast', self.contrast)
        layout.addRow('Gamma', self.gamma)
        layout.addRow('Opacity', self.opacity)
        row = QtWidgets.QHBoxLayout()
        auto = QtWidgets.QPushButton('Auto contrast')
        reset = QtWidgets.QPushButton('Reset')
        auto.setToolTip('Estimate background and foreground separately; adjust range and gamma without changing pixels.')
        row.addWidget(auto)
        row.addWidget(reset)
        layout.addRow(row)
        self.defaults = asdict(channel)
        self._update_color()
        self.visible.toggled.connect(self._changed)
        for control in (self.low, self.high, self.gamma, self.brightness, self.contrast, self.opacity):
            control.valueChanged.connect(self._changed)
        self.color.clicked.connect(self.pick_color)
        auto.clicked.connect(self.auto)
        reset.clicked.connect(self.reset)

    def _update_color(self):
        color = QtGui.QColor.fromRgbF(*self.channel.color).name()
        self.color.setStyleSheet(f'border: 2px solid {color};')

    def _changed(self):
        if self.high.value() <= self.low.value():
            self.low.setStyleSheet('border: 1px solid #eaa660;')
            self.high.setStyleSheet('border: 1px solid #eaa660;')
            return
        self.low.setStyleSheet('')
        self.high.setStyleSheet('')
        c = self.channel
        c.visible = self.visible.isChecked()
        c.low, c.high = self.low.value(), self.high.value()
        c.brightness = self.brightness.value() / 100
        c.contrast = self.contrast.value() / 100
        c.gamma = self.gamma.value()
        c.opacity = self.opacity.value() / 100
        self.brightness.setToolTip(f'{c.brightness:+.2f}')
        self.contrast.setToolTip(f'{c.contrast:.2f}×')
        self.changed.emit()

    def pick_color(self):
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor.fromRgbF(*self.channel.color), self, 'Channel color')
        if color.isValid():
            self.channel.color = color.getRgbF()[:3]
            self._update_color()
            self.changed.emit()

    def sync(self):
        c = self.channel
        controls = [self.visible, self.low, self.high, self.brightness, self.contrast, self.gamma, self.opacity]
        for control in controls:
            control.blockSignals(True)
        self.visible.setChecked(c.visible)
        self.low.setValue(c.low)
        self.high.setValue(c.high)
        self.brightness.setValue(round(c.brightness * 100))
        self.contrast.setValue(round(c.contrast * 100))
        self.gamma.setValue(c.gamma)
        self.opacity.setValue(round(c.opacity * 100))
        for control in controls:
            control.blockSignals(False)
        self._update_color()

    def auto(self):
        auto_adjust(self.channel, self.volume)
        self.sync()
        self.changed.emit()

    def reset(self):
        for k, value in self.defaults.items():
            setattr(self.channel, k, value)
        self.sync()
        self.changed.emit()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, off_screen=False):
        super().__init__()
        self.off_screen = off_screen
        self.setWindowTitle('FISH Viewer')
        self.resize(1450, 920)
        self.setMinimumSize(760, 420)
        self.setStyleSheet(STYLE)
        self.dataset: Dataset | None = None
        self.entry: LibraryEntry | None = None
        self.entries = []
        self.channel_panels = []
        self.volume_view = None
        self._volume_dirty = True
        self._volume_dataset = None
        self._generation = 0
        self._jobs = set()
        self._saved_states = {}
        self._pending_load = False
        self.neural_panel = None
        self.neural_dock = None
        self.signal_panel = None
        self.signal_dock = None
        self.workflow_panel = None
        self.workflow_dock = None
        self._pipeline_archives = {}
        self._signal_archives = {}
        self._source_visibility = {'manual': True, 'prediction': True}
        self.thread_pool = QtCore.QThreadPool(self)
        self.thread_pool.setMaxThreadCount(2)
        self.refresh_timer = QtCore.QTimer(self)
        self.refresh_timer.setSingleShot(True)
        self.refresh_timer.setInterval(120)
        self.refresh_timer.timeout.connect(self.refresh_images)
        self._build_toolbar()
        self._build_body()
        self.statusBar().showMessage('Open an Imaris image or choose a folder to browse its subfolders.')
        self.busy = QtWidgets.QProgressBar()
        self.busy.setMaximumWidth(140)
        self.busy.setRange(0, 0)
        self.busy.hide()
        self.statusBar().addPermanentWidget(self.busy)

    def _build_toolbar(self):
        toolbar = self.addToolBar('Viewer')
        toolbar.setMovable(False)
        self.open_action = toolbar.addAction('Open files…', self.open_files)
        self.open_action.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        toolbar.addAction('Open folder…', self.open_folder)
        toolbar.addSeparator()
        self.mode_group = QtGui.QActionGroup(self)
        self.mode_2d = toolbar.addAction('2D slices')
        self.mode_3d = toolbar.addAction('3D volume')
        for action in (self.mode_2d, self.mode_3d):
            action.setCheckable(True)
            self.mode_group.addAction(action)
        self.mode_2d.setChecked(True)
        self.mode_2d.triggered.connect(lambda: self.switch_mode(False))
        self.mode_3d.triggered.connect(lambda: self.switch_mode(True))
        toolbar.addAction('Fit view', self.fit_view).setShortcut('F')
        toolbar.addSeparator()
        toolbar.addAction('Screenshot…', self.screenshot)
        toolbar.addAction('Save view…', self.save_view)
        toolbar.addAction('Load view…', self.load_view)
        toolbar.addSeparator()
        toolbar.addAction('Neural training…', self.show_neural)
        toolbar.addAction('Help', self.show_help)
        QtGui.QShortcut(QtGui.QKeySequence('Tab'), self, activated=lambda: self.switch_mode(not self.mode_3d.isChecked()))
        QtGui.QShortcut(QtGui.QKeySequence('Right'), self, activated=lambda: self.step_slice(1))
        QtGui.QShortcut(QtGui.QKeySequence('Left'), self, activated=lambda: self.step_slice(-1))

    def _build_body(self):
        splitter = QtWidgets.QSplitter()
        splitter.setMinimumHeight(640)
        self.viewer_scroll = QtWidgets.QScrollArea()
        self.viewer_scroll.setWidgetResizable(True)
        self.viewer_scroll.setWidget(splitter)
        self.setCentralWidget(self.viewer_scroll)
        library = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(library)
        title = label('DATASETS', muted=True)
        layout.addWidget(title)
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText('Filter files or folders…')
        self.search.textChanged.connect(self.filter_library)
        layout.addWidget(self.search)
        self.library = QtWidgets.QTreeWidget()
        self.library.setHeaderHidden(True)
        self.library.itemActivated.connect(self.activate_entry)
        self.library.setToolTip('Double-click a dataset, or select it and press Enter, to open it.')
        layout.addWidget(self.library, 1)
        self.library_count = label('No datasets', muted=True)
        layout.addWidget(self.library_count)
        layout.addWidget(label('Double-click a file to open', muted=True))
        splitter.addWidget(library)
        center = QtWidgets.QWidget()
        center_layout = QtWidgets.QVBoxLayout(center)
        self.heading = label('FISH Viewer')
        self.heading.setObjectName('heading')
        self.heading.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.description = label('Explore image stacks and saved cell annotations.', muted=True)
        self.description.setWordWrap(True)
        center_layout.addWidget(self.heading)
        center_layout.addWidget(self.description)
        actions = QtWidgets.QHBoxLayout()
        self.auto_contours_button = QtWidgets.QPushButton('Auto-select contours…')
        self.auto_contours_button.setEnabled(False)
        self.auto_contours_button.clicked.connect(self.auto_select_contours)
        self.neural_report_button = QtWidgets.QPushButton('Open saved run…')
        self.neural_report_button.clicked.connect(self.open_neural_report)
        actions.addWidget(self.auto_contours_button)
        actions.addWidget(self.neural_report_button)
        actions.addStretch()
        center_layout.addLayout(actions)
        self.signal_button = QtWidgets.QPushButton('Find signals / signal training…')
        self.signal_button.clicked.connect(self.show_signals)
        center_layout.addWidget(self.signal_button)
        self.workflow_button = QtWidgets.QPushButton('Scan workflow / Export Imaris…')
        self.workflow_button.clicked.connect(self.show_workflow)
        center_layout.addWidget(self.workflow_button)
        self.warning = label('')
        self.warning.setWordWrap(True)
        self.warning.setStyleSheet('color: #edbd7c; padding: 6px; background: #30291f;')
        self.warning.hide()
        center_layout.addWidget(self.warning)
        self.stack = QtWidgets.QStackedWidget()
        self.empty = label('Open an .ims image or its .imx scene\n\nChoose a folder to browse all its subfolders.')
        self.empty.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.empty.setStyleSheet('font-size: 17px; color: #91a8bf; background: #080c13;')
        self.stack.addWidget(self.empty)
        self.slice_view = SliceView()
        self.stack.addWidget(self.slice_view)
        self.slice_view.stepped.connect(self.step_slice)
        center_layout.addWidget(self.stack, 1)
        self.slice_bar = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(self.slice_bar)
        row.setContentsMargins(0, 5, 0, 0)
        self.orientation = QtWidgets.QComboBox()
        self.orientation.addItems(['XY · Z slices', 'XZ · Y slices', 'YZ · X slices'])
        self.orientation.currentIndexChanged.connect(self.change_axis)
        self.slice_slider = slider(0, 0, 0)
        self.slice_slider.valueChanged.connect(self.change_slice)
        self.slice_number = QtWidgets.QSpinBox()
        self.slice_number.setRange(1, 1)
        self.slice_number.setKeyboardTracking(False)
        self.slice_number.valueChanged.connect(lambda n: self.slice_slider.setValue(n - 1))
        row.addWidget(self.orientation)
        row.addWidget(self.slice_slider, 1)
        row.addWidget(self.slice_number)
        center_layout.addWidget(self.slice_bar)
        self.hint = label('Scroll: change slice  ·  Ctrl + scroll: zoom  ·  Drag: pan  ·  F: fit  ·  Tab: 2D / 3D', muted=True)
        self.hint.setWordWrap(True)
        center_layout.addWidget(self.hint)
        splitter.addWidget(center)
        controls = QtWidgets.QWidget()
        controls_layout = QtWidgets.QVBoxLayout(controls)
        time_row = QtWidgets.QFormLayout()
        self.time_combo = QtWidgets.QComboBox()
        self.resolution_combo = QtWidgets.QComboBox()
        time_row.addRow('Time point', self.time_combo)
        time_row.addRow('Resolution', self.resolution_combo)
        controls_layout.addLayout(time_row)
        self.time_combo.currentIndexChanged.connect(self.reload_selection)
        self.resolution_combo.currentIndexChanged.connect(self.reload_selection)
        tabs = QtWidgets.QTabWidget()
        self.channel_scroll = QtWidgets.QScrollArea()
        self.channel_scroll.setWidgetResizable(True)
        tabs.addTab(self.channel_scroll, 'Channels')
        annotation_page = QtWidgets.QWidget()
        ann_layout = QtWidgets.QVBoxLayout(annotation_page)
        self.show_surfaces = QtWidgets.QCheckBox('Cell surfaces / borders')
        self.show_markers = QtWidgets.QCheckBox('Measurement endpoints')
        self.show_lines = QtWidgets.QCheckBox('Measurement lines')
        for control in (self.show_surfaces, self.show_markers, self.show_lines):
            control.setChecked(True)
            control.toggled.connect(self.refresh_annotations)
            ann_layout.addWidget(control)
        self.show_predicted_signals = QtWidgets.QCheckBox('Predicted signal centers')
        self.show_predicted_signals.setChecked(True)
        self.show_predicted_signals.toggled.connect(self.signal_visibility)
        ann_layout.addWidget(self.show_predicted_signals)
        self.objects = QtWidgets.QTreeWidget()
        self.objects.setHeaderLabels(['Saved objects'])
        self.objects.itemChanged.connect(self.object_visibility)
        self.objects.currentItemChanged.connect(self.select_object)
        ann_layout.addWidget(self.objects, 1)
        self.object_info = label('Select an object to inspect it.', muted=True)
        self.object_info.setWordWrap(True)
        self.object_info.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        ann_layout.addWidget(self.object_info)
        focus = QtWidgets.QPushButton('Focus selected object')
        focus.clicked.connect(self.focus_object)
        ann_layout.addWidget(focus)
        tabs.addTab(annotation_page, 'Annotations')
        controls_layout.addWidget(tabs, 1)
        volume_group = QtWidgets.QGroupBox('3D display')
        form = QtWidgets.QFormLayout(volume_group)
        self.surface_opacity = slider(0, 100, 32)
        self.volume_opacity = slider(1, 100, 25)
        self.wireframe = QtWidgets.QCheckBox('Wireframe cell surfaces')
        self.full_detail = QtWidgets.QCheckBox('Full volume detail')
        self.full_detail.setToolTip('Use every voxel in 3D. The default preview samples a maximum of 256 voxels per axis. 2D uses the selected image resolution.')
        form.addRow('Cell opacity', self.surface_opacity)
        form.addRow('Volume opacity', self.volume_opacity)
        form.addRow(self.wireframe)
        form.addRow(self.full_detail)
        controls_layout.addWidget(volume_group)
        self.surface_opacity.valueChanged.connect(self.refresh_annotations)
        self.wireframe.toggled.connect(self.refresh_annotations)
        self.volume_opacity.valueChanged.connect(self.schedule_refresh)
        self.full_detail.toggled.connect(self.schedule_refresh)
        splitter.addWidget(controls)
        splitter.setSizes([265, 845, 340])
        splitter.setStretchFactor(1, 1)
        controls.setMinimumWidth(295)
        library.setMinimumWidth(180)

    def run_job(self, function, result, error=None, progress=None):
        job = Job(function)
        self._jobs.add(job)
        job.signals.result.connect(result)
        job.signals.error.connect(error or self.error)
        job.signals.progress.connect(progress or self.statusBar().showMessage)
        job.signals.finished.connect(lambda: self._jobs.discard(job))
        self.thread_pool.start(job)

    def error(self, details):
        self.busy.hide()
        self._pending_load = False
        self.statusBar().showMessage('Could not complete the operation.')
        dialog = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Icon.Warning, 'FISH Viewer', details.splitlines()[-1], parent=self)
        dialog.setDetailedText(details)
        dialog.exec()

    def open_files(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, 'Open Imaris files', '', 'Imaris files (*.ims *.imx *.IMS *.IMX)')
        if files:
            self.open_paths(files)

    def open_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, 'Browse Imaris folder and all subfolders')
        if folder:
            self.add_folder(folder)

    def open_paths(self, paths):
        entries = []
        for path in paths:
            if Path(path).is_dir():
                self.add_folder(path)
            else:
                try:
                    entries.append(entry_for_path(path))
                except Exception:
                    self.error(traceback.format_exc())
        if entries:
            self.add_entries(entries)
            self.load_entry(entries[0])

    def add_folder(self, folder):
        self.busy.show()
        self.statusBar().showMessage(f'Indexing {folder}…')
        def complete(result):
            entries, warnings = result
            self.add_entries(entries)
            self.busy.hide()
            self.statusBar().showMessage(f'{len(entries)} datasets indexed. Double-click a file to open it.' + (f' {len(warnings)} indexing errors.' if warnings else ''))
            if warnings:
                self.warning.setText('\n'.join(warnings[:5]))
                self.warning.show()
        self.run_job(lambda progress: scan_library(folder), complete)

    def add_entries(self, entries):
        combined = set(self.entries) | set(entries)
        # Replace an unpaired image entry when an explicit scene is added later.
        annotated = {e.image for e in combined if e.scene}
        self.entries = sorted((e for e in combined if e.scene or e.image not in annotated), key=lambda e: (str(e.image).casefold(), str(e.scene)))
        self.library.clear()
        groups = {}
        for entry in self.entries:
            parent = entry.image.parent
            if parent not in groups:
                item = QtWidgets.QTreeWidgetItem([parent.name])
                item.setToolTip(0, str(parent))
                self.library.addTopLevelItem(item)
                groups[parent] = item
            suffix = '  ◦' if entry.scene else ''
            name = entry.label + suffix
            if not entry.image.exists():
                name += '  (missing image)'
            item = QtWidgets.QTreeWidgetItem([name])
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, entry)
            item.setToolTip(0, str(entry.image) + (f'\nScene: {entry.scene.name}' if entry.scene else '\nNo companion scene found'))
            groups[parent].addChild(item)
        self.library_count.setText(f'{len(self.entries)} datasets · ◦ companion scene')
        if len(groups) <= 8:
            self.library.expandAll()
        self.filter_library(self.search.text())

    def filter_library(self, text):
        query = text.casefold()
        for i in range(self.library.topLevelItemCount()):
            parent = self.library.topLevelItem(i)
            count = 0
            for j in range(parent.childCount()):
                child = parent.child(j)
                visible = query in (parent.toolTip(0) + '/' + child.text(0)).casefold()
                child.setHidden(not visible)
                count += visible
            parent.setHidden(not count)
            if query and count:
                parent.setExpanded(True)

    def activate_entry(self, item, column=0):
        entry = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if entry:
            self.load_entry(entry)

    def load_entry(self, entry, time=0, level=0, state_override=None):
        if self.dataset and self.entry:
            self._saved_states[self.entry] = self.view_state()
        state = state_override if state_override is not None else self._saved_states.get(entry)
        self._generation += 1
        generation = self._generation
        self._pending_load = True
        self.busy.show()
        self.statusBar().showMessage(f'Opening {entry.image.name}…')
        def complete(dataset):
            if generation != self._generation:
                return
            self._pending_load = False
            self.busy.hide()
            self.entry = entry
            self.set_dataset(dataset, state)
        def failed(details):
            if generation == self._generation:
                self.error(details)
        def progress(text):
            if generation == self._generation:
                self.statusBar().showMessage(text)
        archive = self._pipeline_archives.get((str(entry.image.resolve()), time))
        signal_archive = self._signal_archives.get((str(entry.image.resolve()), time))
        def read_with_predictions(report):
            dataset = load_dataset(entry, time, level, report)
            if archive:
                try:
                    from .pipeline.runner import read_prediction
                    surfaces, metadata = read_prediction(archive)
                    image_name=metadata.get('image_name') or Path(str(metadata.get('image','')).replace('\\','/')).name
                    if image_name != entry.image.name:
                        raise ValueError('The predictions belong to another image.')
                    if level == 0:
                        import hashlib
                        fingerprint = hashlib.sha256()
                        fingerprint.update(str(dataset.volumes[0].shape).encode())
                        for volume in dataset.volumes:
                            fingerprint.update(memoryview(np.ascontiguousarray(volume)))
                        if fingerprint.hexdigest() != metadata['fingerprint']:
                            raise ValueError('Image pixels differ from the input used to generate these predictions.')
                    if metadata.get('signal_filter'):
                        # A filtered result is a new scene. Source/manual annotations
                        # remain available in the original file, not in this result.
                        dataset.scene = Scene(source=dataset.scene.source, warnings=dataset.scene.warnings)
                    dataset.scene.surfaces.extend(surfaces)
                    if signal_archive:
                        from .signals.overlays import read_signals
                        points, signal_report = read_signals(signal_archive, metadata['fingerprint'])
                        if Path(signal_report['cell_predictions']).resolve() == Path(archive).resolve():
                            dataset.scene.measurements.extend(points)
                except Exception as exc:
                    dataset.scene.warnings.append(f'Cannot load pipeline contours: {exc}')
            return dataset
        self.run_job(read_with_predictions, complete, failed, progress)

    def set_dataset(self, dataset, state=None):
        self.dataset = dataset
        if self.workflow_panel:self.workflow_panel.refresh_scan()
        self.auto_contours_button.setEnabled(True)
        for measure in dataset.scene.measurements:
            if measure.source == 'signal_prediction':measure.visible = self.show_predicted_signals.isChecked()
        for surface in dataset.scene.surfaces:
            if not self._source_visibility.get(surface.source, True):
                surface.visible = False
        self._volume_dirty = True
        self._volume_dataset = None
        self.heading.setText(dataset.metadata.path.name)
        m = dataset.metadata
        self.description.setText(f'{" × ".join(map(str, dataset.volumes[0].shape[::-1]))} voxels · {len(m.channels)} channels · '
            f'{len(dataset.scene.surfaces)} surfaces · {len(dataset.scene.measurements)} measurement groups'
            + (f' · {len(dataset.scene.contours)} unmeshed contours' if dataset.scene.contours else '') + '\n'
            f'Voxel size: {" × ".join(f"{s:.4g}" for s in dataset.spacing)} {m.unit} · {m.format}')
        warnings = m.warnings + dataset.scene.warnings
        self.warning.setVisible(bool(warnings))
        self.warning.setText('\n'.join(warnings[:6]))
        for control in (self.time_combo, self.resolution_combo):
            control.blockSignals(True)
            control.clear()
        for time in m.times:
            self.time_combo.addItem(str(time + 1), time)
        self.time_combo.setCurrentIndex(self.time_combo.findData(dataset.time))
        self.time_combo.setEnabled(len(m.times) > 1)
        for level, shape in sorted(m.levels.items()):
            self.resolution_combo.addItem(('Full' if level == 0 else f'Level {level}') + f' · {shape[2]} × {shape[1]} × {shape[0]}', level)
        self.resolution_combo.setCurrentIndex(self.resolution_combo.findData(dataset.level))
        for control in (self.time_combo, self.resolution_combo):
            control.blockSignals(False)
        channel_body = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(channel_body)
        layout.setContentsMargins(7, 3, 7, 8)
        self.channel_panels = []
        for i, (channel, volume) in enumerate(zip(m.channels, dataset.volumes)):
            panel = ChannelPanel(channel, volume, i)
            panel.changed.connect(self.schedule_refresh)
            layout.addWidget(panel)
            self.channel_panels.append(panel)
        layout.addStretch()
        self.channel_scroll.setWidget(channel_body)
        self.objects.blockSignals(True)
        self.objects.clear()
        for kind, title, objects in [('surface', 'Cell surfaces', dataset.scene.surfaces), ('measurement', 'Measurements', dataset.scene.measurements), ('contour', 'Unmeshed contours', dataset.scene.contours)]:
            group = QtWidgets.QTreeWidgetItem([f'{title} ({len(objects)})'])
            self.objects.addTopLevelItem(group)
            for i, obj in enumerate(objects):
                item = QtWidgets.QTreeWidgetItem([obj.name])
                item.setData(0, QtCore.Qt.ItemDataRole.UserRole, (kind, i))
                item.setCheckState(0, QtCore.Qt.CheckState.Checked if obj.visible else QtCore.Qt.CheckState.Unchecked)
                item.setToolTip(0, obj.name)
                group.addChild(item)
            group.setExpanded(True)
        self.objects.blockSignals(False)
        self.object_info.setText('Select an object to inspect it. A/B are saved endpoint names; their biological roles are not inferred.')
        self.slice_view.set_dataset(dataset)
        self._sync_slice_controls()
        if state:
            try:
                self.apply_state(state, change_mode=False)
            except (ValueError, TypeError, KeyError):
                pass
        self.switch_mode(state.get('mode') == '3d' if state else self.mode_3d.isChecked())
        if state and state.get('camera') and self.volume_view and self._volume_dataset is dataset:
            self.volume_view.plotter.camera_position = state['camera']
        self.refresh_annotations()
        self.statusBar().showMessage(f'Opened {m.path.name}. Source files are read-only.')

    def _sync_slice_controls(self):
        if not self.dataset:
            return
        self.slice_slider.blockSignals(True)
        self.slice_number.blockSignals(True)
        count = self.dataset.volumes[0].shape[self.slice_view.axis]
        self.slice_slider.setRange(0, count - 1)
        self.slice_number.setRange(1, count)
        self.slice_slider.setValue(self.slice_view.index)
        self.slice_number.setValue(self.slice_view.index + 1)
        self.slice_slider.blockSignals(False)
        self.slice_number.blockSignals(False)

    def change_axis(self, axis):
        self.slice_view.axis = axis
        if self.dataset:
            self.slice_view.index = self.dataset.volumes[0].shape[axis] // 2
            self._sync_slice_controls()
            self.slice_view.update_image()
            self.slice_view.fit()

    def change_slice(self, index):
        if self.dataset:
            self.slice_view.index = index
            self.slice_number.blockSignals(True)
            self.slice_number.setValue(index + 1)
            self.slice_number.blockSignals(False)
            self.slice_view.update_image()

    def step_slice(self, delta):
        self.slice_slider.setValue(self.slice_slider.value() + delta)

    def reload_selection(self):
        if self.entry and not self._pending_load:
            self.load_entry(self.entry, self.time_combo.currentData(), self.resolution_combo.currentData())

    def switch_mode(self, volume):
        self.mode_3d.setChecked(volume)
        self.mode_2d.setChecked(not volume)
        self.slice_bar.setVisible(not volume)
        if self.dataset is None:
            return
        if volume:
            self.statusBar().showMessage('Preparing 3D volume…')
            if self.volume_view is None:
                self.volume_view = VolumeView(self.stack, off_screen=self.off_screen)
                self.stack.addWidget(self.volume_view)
            self.stack.setCurrentWidget(self.volume_view)
            self._configure_volume()
            if self._volume_dataset is not self.dataset:
                self.volume_view.set_dataset(self.dataset)
                self._volume_dataset = self.dataset
                self._volume_dirty = False
            elif self._volume_dirty:
                self.volume_view.update_volume()
                self._volume_dirty = False
            self.refresh_annotations()
            self.hint.setText('Drag: rotate · Scroll: zoom · Shift + drag: pan · F: fit · Tab: 2D / 3D')
            self.statusBar().showMessage('3D volume ready.' + (' Full detail.' if self.full_detail.isChecked() else ' Preview sampling; enable Full volume detail for every voxel.'))
        else:
            self.stack.setCurrentWidget(self.slice_view)
            self.slice_view.update_image()
            self.hint.setText('Scroll: change slice · Ctrl + scroll: zoom · Drag: pan · F: fit · Tab: 2D / 3D')

    def _configure_volume(self):
        if self.volume_view:
            self.volume_view.volume_opacity = self.volume_opacity.value() / 100
            self.volume_view.surface_opacity = self.surface_opacity.value() / 100
            self.volume_view.full_detail = self.full_detail.isChecked()
            self.volume_view.wireframe = self.wireframe.isChecked()

    def schedule_refresh(self):
        self._volume_dirty = True
        self.refresh_timer.start()

    def refresh_images(self):
        if not self.dataset:
            return
        self.slice_view.update_image()
        if self.mode_3d.isChecked() and self.volume_view and self._volume_dataset is self.dataset:
            self._configure_volume()
            self.volume_view.update_volume()
            self._volume_dirty = False

    def refresh_annotations(self):
        for view in (self.slice_view, self.volume_view):
            if view:
                view.surfaces_visible = self.show_surfaces.isChecked()
                view.markers_visible = self.show_markers.isChecked()
                view.lines_visible = self.show_lines.isChecked()
        self.slice_view.update_overlays()
        self._configure_volume()
        if self.volume_view and self.mode_3d.isChecked() and self._volume_dataset is self.dataset:
            self.volume_view.update_annotations()

    def object_visibility(self, item):
        key = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if key and self.dataset:
            kind, index = key
            collection = self.object_collection(kind)
            collection[index].visible = item.checkState(0) == QtCore.Qt.CheckState.Checked
            self.refresh_annotations()

    def select_object(self, item, previous=None):
        key = item.data(0, QtCore.Qt.ItemDataRole.UserRole) if item else None
        self.slice_view.selected = key
        if self.volume_view:
            self.volume_view.selected = key
        if key and self.dataset:
            kind, index = key
            if kind == 'surface':
                obj = self.dataset.scene.surfaces[index]
                text = f'{obj.name}\n{len(obj.vertices):,} vertices · {len(obj.triangles):,} triangles\nTime point {obj.time + 1}'
            elif kind == 'measurement':
                obj = self.dataset.scene.measurements[index]
                if obj.source == 'signal_prediction':
                    text = f'{obj.name}\n' + '\n'.join(obj.labels[:12])
                else:
                    text = f'{obj.name}\n{len(obj.points)} endpoints · {len(obj.edges)} segments\nTotal distance: {obj.length:.4f} {self.dataset.metadata.unit}'
                    text += '\n' + '\n'.join(f'{obj.labels[i]} → {obj.labels[j]}: {np.linalg.norm(obj.points[j] - obj.points[i]):.4f} {self.dataset.metadata.unit}' for i, j in obj.edges[:6])
            else:
                obj = self.dataset.scene.contours[index]
                text = f'{obj.name}\n{len(obj.points)} vertices · saved contour curve\nThis outline has no saved mesh.'
            self.object_info.setText(text)
        self.refresh_annotations()

    def focus_object(self):
        key = self.slice_view.selected
        if not key or not self.dataset:
            return
        kind, index = key
        obj = self.object_collection(kind)[index]
        points = obj.vertices if kind == 'surface' else obj.vertices(self.dataset.metadata) if kind == 'contour' else obj.points
        lower, upper = points.min(axis=0), points.max(axis=0)
        center = (lower + upper) / 2
        if kind == 'measurement' and obj.source == 'signal_prediction':
            # Keep surrounding cell context when focusing a single punctum.
            lower = np.minimum(lower, center - 1.5)
            upper = np.maximum(upper, center + 1.5)
        if self.mode_3d.isChecked() and self.volume_view:
            padding = max(float((upper - lower).max()) * 0.2, max(self.dataset.spacing) * 2)
            bounds = tuple(v for low, high in zip(lower, upper) for v in (low - padding, high + padding))
            self.volume_view.plotter.reset_camera(bounds=bounds)
        else:
            axis = 2 - self.slice_view.axis
            self.slice_slider.setValue(round((center[axis] - self.dataset.center_origin[axis]) / self.dataset.spacing[axis]))
            u, v = self.slice_view.projection_axes
            padding = max(float((upper - lower).max()) * 0.15, max(self.dataset.spacing) * 3)
            rect = QtCore.QRectF(float(lower[u] - padding), float(lower[v] - padding), float(upper[u] - lower[u] + 2 * padding), float(upper[v] - lower[v] + 2 * padding))
            self.slice_view.fitInView(rect, QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def object_collection(self, kind):
        return {'surface': self.dataset.scene.surfaces, 'measurement': self.dataset.scene.measurements,
            'contour': self.dataset.scene.contours}[kind]

    def fit_view(self):
        if self.mode_3d.isChecked() and self.volume_view:
            self.volume_view.fit()
        else:
            self.slice_view.fit()

    def view_state(self):
        if not self.dataset:
            return {}
        return dict(version=1, image=str(self.dataset.metadata.path),
            time=self.dataset.time, level=self.dataset.level,
            channels=[asdict(c) for c in self.dataset.metadata.channels],
            axis=self.slice_view.axis, index=self.slice_view.index, mode='3d' if self.mode_3d.isChecked() else '2d',
            surfaces=self.show_surfaces.isChecked(), markers=self.show_markers.isChecked(), lines=self.show_lines.isChecked(),
            surface_visibility=[s.visible for s in self.dataset.scene.surfaces],
            measurement_visibility=[m.visible for m in self.dataset.scene.measurements],
            contour_visibility=[c.visible for c in self.dataset.scene.contours],
            surface_opacity=self.surface_opacity.value(), volume_opacity=self.volume_opacity.value(),
            wireframe=self.wireframe.isChecked(), full_detail=self.full_detail.isChecked(),
            camera=[list(v) for v in self.volume_view.plotter.camera_position] if self.volume_view and self._volume_dataset is self.dataset else None)

    def apply_state(self, state, change_mode=True):
        if state.get('version') != 1 or len(state.get('channels', [])) != len(self.channel_panels):
            raise ValueError('Incompatible display settings file.')
        # Validate all intensity controls before applying any of them.
        for values in state['channels']:
            color = values.get('color')
            if not isinstance(color, (list, tuple)) or len(color) != 3 or not all(np.isfinite(v) and 0 <= v <= 1 for v in color):
                raise ValueError('Invalid channel color.')
            if not all(np.isfinite(float(values[k])) for k in ('low', 'high', 'gamma', 'opacity', 'brightness', 'contrast')):
                raise ValueError('Invalid channel display value.')
            if values['low'] >= values['high'] or not 0.1 <= values['gamma'] <= 5 or not 0.1 <= values['contrast'] <= 4 or not -1 <= values['brightness'] <= 1 or not 0 <= values['opacity'] <= 1:
                raise ValueError('Channel display values are out of range.')
        for panel, values in zip(self.channel_panels, state['channels']):
            for key in ('color', 'low', 'high', 'gamma', 'opacity', 'visible', 'brightness', 'contrast'):
                setattr(panel.channel, key, values[key])
            panel.sync()
        self.orientation.setCurrentIndex(max(0, min(2, int(state.get('axis', 0)))))
        self.slice_slider.setValue(max(0, min(self.slice_slider.maximum(), int(state.get('index', 0)))))
        for key, control in [('surfaces', self.show_surfaces), ('markers', self.show_markers), ('lines', self.show_lines), ('wireframe', self.wireframe), ('full_detail', self.full_detail)]:
            control.setChecked(bool(state.get(key, control.isChecked())))
        self.surface_opacity.setValue(int(state.get('surface_opacity', 32)))
        self.volume_opacity.setValue(int(state.get('volume_opacity', 25)))
        self.objects.blockSignals(True)
        for group_index, key, collection in [(0, 'surface_visibility', self.dataset.scene.surfaces), (1, 'measurement_visibility', self.dataset.scene.measurements), (2, 'contour_visibility', self.dataset.scene.contours)]:
            for i, (obj, visible) in enumerate(zip(collection, state.get(key, []))):
                obj.visible = bool(visible)
                self.objects.topLevelItem(group_index).child(i).setCheckState(0, QtCore.Qt.CheckState.Checked if visible else QtCore.Qt.CheckState.Unchecked)
        self.objects.blockSignals(False)
        self._volume_dirty = True
        self.refresh_images()
        if change_mode:
            self.switch_mode(state.get('mode') == '3d')
        if self.volume_view and self._volume_dataset is self.dataset and state.get('camera'):
            self.volume_view.plotter.camera_position = state['camera']
        self.refresh_annotations()

    def save_view(self):
        if not self.dataset:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save display settings', self.dataset.metadata.path.stem + '.fish-view.json', 'Display settings (*.fish-view.json)')
        if path:
            try:
                target = Path(path)
                if not target.name.endswith('.fish-view.json'):
                    target = Path(str(target) + '.fish-view.json')
                target.write_text(json.dumps(self.view_state(), indent=2), encoding='utf-8')
                self.statusBar().showMessage(f'Display settings saved to {target}')
            except Exception:
                self.error(traceback.format_exc())

    def load_view(self):
        if not self.dataset:
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Load display settings for the current image', '', 'Display settings (*.json)')
        if path:
            try:
                state = json.loads(Path(path).read_text(encoding='utf-8'))
                if Path(state.get('image', '')).name != self.dataset.metadata.path.name:
                    raise ValueError('These display settings belong to a different image. Open that image first.')
                time, level = int(state.get('time', self.dataset.time)), int(state.get('level', self.dataset.level))
                if time not in self.dataset.metadata.times or level not in self.dataset.metadata.levels:
                    raise ValueError('The saved time point or resolution is unavailable in this image.')
                if (time, level) != (self.dataset.time, self.dataset.level):
                    self.load_entry(self.entry or entry_for_path(self.dataset.metadata.path), time, level, state_override=state)
                else:
                    self.apply_state(state)
                    self.statusBar().showMessage('Display settings restored.')
            except Exception:
                self.error(traceback.format_exc())

    def screenshot(self):
        if not self.dataset:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save current view', self.dataset.metadata.path.stem + '.png', 'PNG image (*.png)')
        if path:
            try:
                if not path.lower().endswith('.png'):
                    path += '.png'
                if self.mode_3d.isChecked() and self.volume_view:
                    self.volume_view.plotter.screenshot(path)
                elif not self.slice_view.viewport().grab().save(path):
                    raise OSError('Could not save screenshot.')
                self.statusBar().showMessage(f'Screenshot saved to {path}')
            except Exception:
                self.error(traceback.format_exc())

    def show_help(self):
        QtWidgets.QMessageBox.information(self, 'Using FISH Viewer',
            'Open files or recursively index a folder. Double-click a dataset to load it.\n\n'
            '2D: scroll or use arrow keys for slices; Ctrl + scroll to zoom; drag to pan. '
            'Choose XY, XZ, or YZ for another slicing direction.\n\n'
            '3D: drag to rotate; scroll to zoom; Shift + drag to pan. '
            'Full volume detail uses all voxels at the selected resolution.\n\n'
            'F fits the view. Tab switches 2D/3D.\n\n'
            'Channels have independent color, visibility, brightness, contrast, gamma and opacity. '
            'The Annotations tab controls saved meshes and A/B measurement endpoints and lines. '
            'Select a measurement for distances; Focus jumps to it. In 2D, only annotations intersecting the current slice are shown.\n\n'
            'Save view / Load view store display settings separately. Screenshots export PNG. '
            'Original images and annotation files are never modified.')


    def show_neural(self):
        if self.workflow_dock:self.workflow_dock.hide()
        if self.signal_dock:self.signal_dock.hide()
        if self.neural_panel is None:
            from .neural.gui import NeuralPanel
            self.neural_dock = QtWidgets.QDockWidget('StarDist 3D neural training', self)
            self.neural_dock.setObjectName('neuralTraining')
            self.neural_panel = NeuralPanel(self)
            self.neural_scroll = self.scroll_panel(self.neural_panel)
            self.neural_dock.setWidget(self.neural_scroll)
            self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.neural_dock)
            self.resizeDocks([self.neural_dock], [560], QtCore.Qt.Orientation.Vertical)
            if self.screen().availableGeometry().height() < 900:
                self.neural_dock.setFloating(True)
                self.fit_floating_dock(self.neural_dock)
        self.neural_dock.show()
        self.neural_dock.raise_()

    @staticmethod
    def scroll_panel(panel):
        from .scrolling import PanelScrollArea
        return PanelScrollArea(panel)

    def fit_floating_dock(self, dock):
        screen = self.screen().availableGeometry()
        dock.resize(min(1200, screen.width() - 40), min(700, screen.height() - 80))
        dock.move(screen.topLeft() + QtCore.QPoint(20, 30))

    def open_neural_report(self):
        self.show_neural()
        self.neural_panel.load_report()

    def auto_select_contours(self):
        from .analysis.app import AnalysisWindow
        self.analysis_window = AnalysisWindow()
        self.analysis_window.show()
        if self.entry is not None:
            self.analysis_window.add_paths([self.entry.image])

    def open_pipeline_prediction(self, row):
        try:
            image = Path(row['image']).resolve()
            scene = Path(row['scene']) if row.get('scene') else None
            entry = LibraryEntry(image, scene) if scene and scene.exists() else entry_for_path(image)
            archive = Path(row['predictions'])
            if not archive.is_file():
                raise FileNotFoundError(f'Prediction archive is missing: {archive}')
            key = (str(image), 0)
            if self._pipeline_archives.get(key) != archive:self._signal_archives.pop(key, None)
            self._pipeline_archives[key] = archive
            self.add_entries([entry])
            self.load_entry(entry)
        except Exception:
            self.error(traceback.format_exc())

    def show_signals(self):
        if self.workflow_dock:self.workflow_dock.hide()
        if self.signal_panel is None:
            from .signals.gui import SignalPanel
            self.signal_dock = QtWidgets.QDockWidget('Signal detection and training', self)
            self.signal_dock.setObjectName('signalDetection')
            self.signal_panel = SignalPanel(self)
            self.signal_scroll = self.scroll_panel(self.signal_panel)
            self.signal_dock.setWidget(self.signal_scroll)
            self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.signal_dock)
            self.resizeDocks([self.signal_dock], [600], QtCore.Qt.Orientation.Vertical)
            if self.screen().availableGeometry().height() < 900:
                self.signal_dock.setFloating(True);self.fit_floating_dock(self.signal_dock)
        for dock in (self.neural_dock,):
            if dock:dock.hide()
        self.signal_dock.show();self.signal_dock.raise_()

    def show_workflow(self):
        if self.workflow_panel is None:
            from .workflow.gui import WorkflowPanel
            self.workflow_dock = QtWidgets.QDockWidget('Scan workflow and Imaris export', self)
            self.workflow_dock.setObjectName('scanWorkflow')
            self.workflow_panel = WorkflowPanel(self)
            self.workflow_scroll = self.scroll_panel(self.workflow_panel)
            self.workflow_dock.setWidget(self.workflow_scroll)
            self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.workflow_dock)
            self.resizeDocks([self.workflow_dock], [560], QtCore.Qt.Orientation.Vertical)
            if self.screen().availableGeometry().height() < 900:
                self.workflow_dock.setFloating(True)
                self.fit_floating_dock(self.workflow_dock)
        for dock in (self.neural_dock, self.signal_dock):
            if dock:dock.hide()
        self.workflow_panel.refresh_scan()
        self.workflow_dock.show()
        self.workflow_dock.raise_()

    def open_signal_prediction(self, path):
        try:
            from .signals.overlays import read_signals
            _, report = read_signals(path)
            image = Path(report['image']).resolve()
            scene = Path(report['scene']) if report.get('scene') else None
            entry = LibraryEntry(image, scene) if scene and scene.exists() else entry_for_path(image)
            key = (str(image), 0)
            self._pipeline_archives[key] = Path(report['cell_predictions'])
            self._signal_archives[key] = Path(path)
            self.show_predicted_signals.setChecked(True)
            self.show_markers.setChecked(True)
            self.show_surfaces.setChecked(True)
            self._source_visibility['prediction'] = True
            # Object lists change when predictions are replaced/filtered. Do not
            # apply the old per-index visibility flags to unrelated new objects.
            state = dict(self.view_state() if entry == self.entry else self._saved_states.get(entry, {}))
            for name in ('surface_visibility', 'measurement_visibility', 'contour_visibility'):
                state.pop(name, None)
            state.update(surfaces=True, markers=True)
            self.add_entries([entry]);self.load_entry(entry, state_override=state)
            if self.signal_dock:self.signal_dock.hide()
        except Exception:self.error(traceback.format_exc())

    def signal_visibility(self,visible):
        if not self.dataset:return
        self.objects.blockSignals(True)
        group = self.objects.topLevelItem(1)
        for i,measure in enumerate(self.dataset.scene.measurements):
            if measure.source == 'signal_prediction':
                measure.visible = visible
                if group:group.child(i).setCheckState(0, QtCore.Qt.CheckState.Checked if visible else QtCore.Qt.CheckState.Unchecked)
        self.objects.blockSignals(False);self.refresh_annotations()

    def set_surface_source_visible(self, source, visible):
        self._source_visibility[source] = visible
        if not self.dataset:
            return
        self.objects.blockSignals(True)
        group = self.objects.topLevelItem(0)
        for i, surface in enumerate(self.dataset.scene.surfaces):
            if surface.source == source:
                surface.visible = visible
                group.child(i).setCheckState(0, QtCore.Qt.CheckState.Checked if visible else QtCore.Qt.CheckState.Unchecked)
        self.objects.blockSignals(False)
        self.refresh_annotations()

    def closeEvent(self, event):
        self._generation += 1
        if self.neural_panel:
            self.neural_panel.shutdown()
        if self.signal_panel:
            self.signal_panel.shutdown()
        if self.workflow_panel:
            self.workflow_panel.shutdown()
        self.refresh_timer.stop()
        self.thread_pool.clear()
        self.thread_pool.waitForDone()
        if self.volume_view:
            self.volume_view.close()
        event.accept()
