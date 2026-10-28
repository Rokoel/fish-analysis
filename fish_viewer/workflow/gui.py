"""A sequential scan workflow, with one cancellable worker process per stage."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time
import uuid

from ..worker_runtime import process_environment
from ..project_paths import bundled_model

from PySide6 import QtCore, QtWidgets

from ..models import LibraryEntry
from ..neural.config import NeuralConfig
from ..neural.model_picker import discover_models, read_model
from ..neural.precision import evidence_preset


class WorkflowPanel(QtWidgets.QWidget):
    def __init__(self, host):
        super().__init__(host)
        self.host = host
        self.job = None
        self.busy = False
        self.result_path = None
        self.exported = None
        self._shutdown = False
        self._cancel_requested = False
        root = Path(__file__).resolve().parents[2]
        layout = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel('Detect contours → detect signals inside them → remove cells without signals → export Imaris')
        intro.setWordWrap(True)
        intro.setStyleSheet('font-size:16px;font-weight:bold;')
        layout.addWidget(intro)
        self.scan = QtWidgets.QLabel()
        self.scan.setWordWrap(True)
        layout.addWidget(self.scan)
        form = QtWidgets.QFormLayout()
        self.cell_model = QtWidgets.QComboBox()
        self.cell_model.setMinimumContentsLength(30)
        self.cell_model.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        for path, title in discover_models():
            self.cell_model.addItem(title, str(path))
        self.signal_model = QtWidgets.QLineEdit()
        bundled_signals=bundled_model('signals-center-unet')
        if (bundled_signals/'signal-model.json').is_file():
            self.signal_model.setText(str(bundled_signals))
        self.output = QtWidgets.QLineEdit(str(root / 'workflow_runs'))
        self.settings = []
        for name, control, action in [
            ('Cell model', self.cell_model, self.choose_cell_model),
            ('Signal model', self.signal_model, lambda: self.choose_folder(self.signal_model)),
            ('Run output folder', self.output, lambda: self.choose_folder(self.output)),
        ]:
            row = QtWidgets.QHBoxLayout()
            row.addWidget(control, 1)
            button = QtWidgets.QPushButton('Browse…')
            button.clicked.connect(action)
            row.addWidget(button)
            form.addRow(name, row)
            self.settings.extend([control, button])
        self.device = QtWidgets.QComboBox()
        self.device.addItems(['auto', 'gpu', 'cpu'])
        form.addRow('Device', self.device)
        self.settings.append(self.device)
        layout.addLayout(form)
        note = QtWidgets.QLabel('Roundness limits are off; visible cell boundaries are required. A cell is kept if at least one signal center is inside it, including its boundary voxels. Processing uses time point 1; exports preserve all image time points.')
        note.setWordWrap(True)
        layout.addWidget(note)
        row = QtWidgets.QHBoxLayout()
        self.run_button = QtWidgets.QPushButton('Detect cells + signals and filter')
        self.run_button.clicked.connect(self.start)
        self.filter_button = QtWidgets.QPushButton('Remove cells without signals')
        self.filter_button.clicked.connect(self.filter_current)
        row.addWidget(self.run_button)
        row.addWidget(self.filter_button)
        layout.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        self.open_button = QtWidgets.QPushButton('Open saved result…')
        self.open_button.clicked.connect(self.open_result)
        self.report_button = QtWidgets.QPushButton('Signal report / counts')
        self.report_button.clicked.connect(self.show_report)
        self.cancel_button = QtWidgets.QPushButton('Cancel')
        self.cancel_button.clicked.connect(self.cancel)
        self.cancel_button.setEnabled(False)
        for button in (self.open_button, self.report_button, self.cancel_button):
            row.addWidget(button)
        layout.addLayout(row)
        self.stage_label = QtWidgets.QLabel('Ready')
        self.stage_label.setStyleSheet('font-size:16px;font-weight:bold;')
        self.stage_label.setWordWrap(True)
        layout.addWidget(self.stage_label)
        self.overall = QtWidgets.QProgressBar()
        self.overall.setRange(0, 3)
        self.overall.setValue(0)
        self.overall.setFormat('%v / 3 detection stages complete')
        layout.addWidget(self.overall)
        self.phase = QtWidgets.QLabel('Open a scan, select the models, then start.')
        self.phase.setWordWrap(True)
        layout.addWidget(self.phase)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        self.elapsed = QtWidgets.QLabel()
        layout.addWidget(self.elapsed)
        self.summary = QtWidgets.QLabel('No filtered result yet.')
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.summary)
        self.result_location = QtWidgets.QLineEdit()
        self.result_location.setReadOnly(True)
        self.result_location.setPlaceholderText('Saved result location')
        layout.addWidget(self.result_location)
        self.export_button = QtWidgets.QPushButton('Export current result as Imaris…')
        self.export_button.clicked.connect(self.export_current)
        layout.addWidget(self.export_button)
        export_note = QtWidgets.QLabel('Export creates a standalone .ims containing the image and all annotations, plus an Excel workbook of cell volumes and distances.')
        export_note.setWordWrap(True)
        layout.addWidget(export_note)
        layout.addStretch()
        # Long scan names and error paths must not force the panel wider than a
        # small screen. Full result paths have their own horizontally scrollable field.
        for control in self.findChildren(QtWidgets.QLabel):
            if control.wordWrap():
                control.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.update_elapsed)
        self.refresh_scan()

    def refresh_scan(self):
        if not self.busy:
            self.scan.setText(f'Scan: {self.host.entry.image.name}' if self.host.entry else 'No scan loaded')

    def choose_folder(self, control):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Choose folder', control.text())
        if path:
            control.setText(path)

    def choose_cell_model(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Choose cell model')
        if path:
            try:
                read_model(path)
                self.cell_model.addItem(str(Path(path).parent.name), str(Path(path).resolve()))
                self.cell_model.setCurrentIndex(self.cell_model.count() - 1)
            except Exception as exc:
                self.error(str(exc))

    def error(self, message):
        self.phase.setText(message)
        QtWidgets.QMessageBox.warning(self, 'Scan workflow', message)

    def check_ready(self):
        if self.busy:
            raise ValueError('The scan workflow is already running.')
        if any(panel and panel.job for panel in (self.host.neural_panel, self.host.signal_panel)):
            raise ValueError('Wait for the current processing job to finish, or cancel it first.')
        if self.host.entry is None or self.host.dataset is None or self.host._pending_load:
            raise ValueError('Open a scan and wait for it to load first.')
        if self.host.dataset.time != 0:
            raise ValueError('Select time point 1 before processing this scan.')

    def current_report(self):
        self.check_ready()
        key = (str(self.host.entry.image.resolve()), self.host.dataset.time)
        path = self.host._signal_archives.get(key)
        if not path:
            raise ValueError('Detect signals on this scan, or open a saved signal result first.')
        return Path(path)

    def begin(self):
        self.job_dir = Path(self.output.text()).expanduser().resolve() / '.jobs' / uuid.uuid4().hex
        self.job_dir.mkdir(parents=True)
        self.cancel_file = self.job_dir / 'cancel'
        self._cancel_requested = False
        self.started = time.monotonic()
        self.set_busy(True)
        self.timer.start()
        self.update_elapsed()

    def set_busy(self, value):
        self.busy = value
        for control in self.settings + [self.run_button, self.filter_button, self.open_button, self.report_button, self.export_button]:
            control.setEnabled(not value)
        self.cancel_button.setEnabled(value)
        self.cancel_button.setText('Cancel')
        if not value:
            self.timer.stop()

    def start(self):
        try:
            self.check_ready()
            cell_model = self.cell_model.currentData()
            manifest = read_model(cell_model or '')
            self.selected_signal_model = str(Path(self.signal_model.text()).expanduser().resolve())
            from ..signals.model import read_model as read_signal_model
            read_signal_model(self.selected_signal_model)
            self.selected_entry = self.host.entry
            self.selected_device = self.device.currentText()
            self.begin()
            self.scan.setText(f'Scan: {self.selected_entry.image.name}')
            self.overall.setValue(0)
            self.result_path = None
            self.summary.setText('Processing selected scan…')
            config = replace(evidence_preset(NeuralConfig.from_dict(manifest['config'])), device=self.selected_device)
            request = dict(config=asdict(config), mode='apply', model=cell_model, output=self.output.text(),
                           cancel_file=str(self.cancel_file), entries=[dict(image=str(self.selected_entry.image),
                           scene=str(self.selected_entry.scene) if self.selected_entry.scene else None)])
            request_path = self.job_dir / 'cells-request.json'
            request_path.write_text(json.dumps(request), encoding='utf-8')
            self.launch('cells', ['-m', 'fish_viewer.neural', '--request', str(request_path)])
        except Exception as exc:
            if not self.job:
                self.set_busy(False)
            self.error(str(exc))

    def filter_current(self):
        try:
            path = self.current_report()
            self.begin()
            self.overall.setValue(2)
            self.launch_filter(path)
        except Exception as exc:
            if not self.job:
                self.set_busy(False)
            self.error(str(exc))

    def launch_filter(self, path):
        self.launch('filter', ['-m', 'fish_viewer.workflow', 'filter', '--report', str(path), '--output', self.output.text(),
                               '--cancel-file', str(self.cancel_file)])

    def launch(self, stage, arguments):
        self.stage = stage
        self.stage_label.setText(dict(cells='1 / 3 · Detecting cell contours', signals='2 / 3 · Detecting signals inside cells',
                                     filter='3 / 3 · Keeping cells with signals', export='Exporting Imaris scan')[stage])
        self.phase.setText('Starting worker…')
        self.progress.setRange(0, 0)
        self.buffer = b''
        self.stderr = ''
        self.finished_report = None
        self.job = QtCore.QProcess(self)
        self.job.setProcessEnvironment(process_environment(self.job_dir))
        self.job.readyReadStandardOutput.connect(self.read_events)
        self.job.readyReadStandardError.connect(self.read_errors)
        self.job.finished.connect(self.process_finished)
        self.job.errorOccurred.connect(lambda error: self.process_finished(-1) if error == QtCore.QProcess.ProcessError.FailedToStart else None)
        self.job.start(sys.executable, arguments)

    def read_events(self):
        if not self.job:
            return
        self.buffer += bytes(self.job.readAllStandardOutput())
        while b'\n' in self.buffer:
            line, self.buffer = self.buffer.split(b'\n', 1)
            try:
                event = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                continue
            kind = event.get('kind')
            if kind == 'finished':
                self.finished_report = event['report']
            elif kind in ('progress', 'tile', 'file'):
                self.phase.setText(event.get('phase', 'Predicting image tiles') + (f' · {event["current"]} / {event["total"]}' if 'total' in event else ''))
                total = event.get('total', 0)
                self.progress.setRange(0, 1000 if total > 0 else 0)
                if total > 0:
                    self.progress.setValue(round(1000 * event.get('current', 0) / total))
            elif kind == 'warning':
                self.summary.setText(event['message'])

    def read_errors(self):
        if self.job:
            text = bytes(self.job.readAllStandardError()).decode(errors='replace')
            self.stderr = (self.stderr + text)[-6000:]
            with (self.job_dir / f'{self.stage}.log').open('a', encoding='utf-8') as stream:
                stream.write(text)

    def process_finished(self, exit_code=0, *_):
        if not self.job:
            return
        self.read_events()
        self.read_errors()
        old, self.job = self.job, None
        old.deleteLater()
        if self._shutdown:
            return
        report = self.finished_report or {}
        if (self._cancel_requested and not (self.stage == 'export' and report.get('status') == 'complete')) or report.get('status') == 'cancelled':
            self.set_busy(False)
            self.stage_label.setText('Cancelled')
            self.phase.setText('Processing stopped. Completed stage reports remain in the output folder.')
            return
        if exit_code != 0 or report.get('status') != 'complete':
            self.set_busy(False)
            self.stage_label.setText('Processing failed')
            self.error(report.get('error', self.stderr or 'The worker did not return a completed result.')[-2000:])
            return
        try:
            if self.stage == 'cells':
                row = report['files'][0]
                self.overall.setValue(1)
                arguments = ['-m', 'fish_viewer.signals', '--mode', 'apply', '--image', str(self.selected_entry.image),
                             '--cell-predictions', row['predictions'], '--model', self.selected_signal_model,
                             '--output', self.output.text(), '--device', self.selected_device, '--cancel-file', str(self.cancel_file)]
                if self.selected_entry.scene:
                    arguments += ['--scene', str(self.selected_entry.scene)]
                self.launch('signals', arguments)
                return
            if self.stage == 'signals':
                self.overall.setValue(2)
                self.launch_filter(Path(report['directory']) / 'report.json')
                return
            self.set_busy(False)
            self.progress.setRange(0, 1000)
            self.progress.setValue(1000)
            if self.stage == 'filter':
                self.overall.setValue(3)
                self.show_result(Path(report['directory']) / 'report.json')
            else:
                self.exported = report
                self.stage_label.setText('Export complete')
                self.summary.setText(f'{report["cells"]} cells · {report["signals"]} signals\nImaris image, scene, count tables and annotations exported.')
                self.result_location.setText(report['image'])
                entry = LibraryEntry(Path(report['image']), Path(report['scene']) if report.get('scene') else None)
                self.host.add_entries([entry])
                self.host.load_entry(entry)
                self.host.workflow_dock.hide()
        except Exception as exc:
            self.set_busy(False)
            self.error(str(exc))

    def show_result(self, path):
        from ..signals.overlays import read_signals
        _, report = read_signals(path)
        self.result_path = Path(path)
        self.result_location.setText(str(path))
        filtering = report.get('signal_filter')
        if filtering:
            self.stage_label.setText('Ready to export')
            self.summary.setText(f'{filtering["kept_cell_count"]} cells kept · {filtering["removed_cell_count"]} cells without signals removed · {len(report["points"])} signals')
            self.overall.setValue(3)
        else:
            self.stage_label.setText('Signals loaded — remove cells without signals to prepare export')
            self.summary.setText(f'{len(report["points"])} signals')
            self.overall.setValue(2)
        self.host.open_signal_prediction(path)

    def open_result(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open saved signal or filtered scan report', self.output.text(), 'Run report (*.json)')
        if path:
            try:
                self.show_result(path)
            except Exception as exc:
                self.error(str(exc))

    def show_report(self):
        try:
            path = self.current_report()
            self.host.show_signals()
            self.host.signal_panel.display_report(json.loads(path.read_text()))
            self.host.signal_panel.tabs.setCurrentIndex(1)
        except Exception as exc:
            self.error(str(exc))

    def export_current(self, checked=False, target=None):
        try:
            path = self.current_report()
            report = json.loads(path.read_text())
            if not report.get('signal_filter'):
                raise ValueError('Click Remove cells without signals before exporting.')
            if target is None:
                target, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Export Imaris image + scene pair',
                    str(Path(self.output.text()) / (self.host.entry.image.stem + '_with_signals.ims')), 'Imaris image (*.ims)')
            if not target:
                return
            target = Path(target)
            if not target.suffix:
                target = target.with_suffix('.ims')
            self.begin()
            self.launch('export', ['-m', 'fish_viewer.workflow', 'export', '--report', str(path),
                                  '--output', str(target), '--cancel-file', str(self.cancel_file)])
        except Exception as exc:
            if not self.job:
                self.set_busy(False)
            self.error(str(exc))

    def update_elapsed(self):
        self.elapsed.setText(f'Elapsed: {int(time.monotonic() - self.started)} seconds')

    def cancel(self):
        if self.busy:
            if self._cancel_requested:
                if self.job:self.job.kill()
                return
            self._cancel_requested = True
            self.cancel_file.touch()
            self.phase.setText('Stopping after the current operation…')
            self.cancel_button.setText('Stop now')

    def shutdown(self):
        self._shutdown = True
        self.timer.stop()
        if self.job:
            process = self.job
            self.cancel_file.touch()
            if not process.waitForFinished(5000):
                process.kill()
                process.waitForFinished(1500)
