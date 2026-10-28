"""Signal training, reports and per-cell detection in a separate worker process."""
import json
from pathlib import Path
import sys
import uuid

from ..worker_runtime import process_environment
from ..project_paths import bundled_model

from PySide6 import QtCore,QtGui,QtWidgets

from ..training_gui import PlotCanvas


class SignalPanel(QtWidgets.QWidget):
    def __init__(self,host):
        super().__init__(host);self.host=host;self.job=None;self.report=None;self.history=[];self.buffer=b'';self.finished_report=None
        self.run_directory=None;self.watched=None;self._shutdown=False
        root=Path(__file__).resolve().parents[2]
        layout=QtWidgets.QVBoxLayout(self)
        self.mode=QtWidgets.QComboBox();self.mode.addItems(['Detect signals in current scan','Train signal detector from audit'])
        self.run_button=QtWidgets.QPushButton('Run');self.run_button.clicked.connect(self.start)
        self.cancel_button=QtWidgets.QPushButton('Cancel');self.cancel_button.clicked.connect(self.cancel);self.cancel_button.setEnabled(False)
        open_button=QtWidgets.QPushButton('Open saved signal run…');open_button.clicked.connect(self.open_report)
        row=QtWidgets.QHBoxLayout()
        for control in (self.mode,self.run_button,self.cancel_button,open_button):row.addWidget(control)
        layout.addLayout(row)
        form=QtWidgets.QFormLayout()
        self.model_path=QtWidgets.QLineEdit()
        self.audit_path=QtWidgets.QLineEdit()
        self.output=QtWidgets.QLineEdit(str(root/'signal_runs'))
        bundled=bundled_model('signals-center-unet')
        if (bundled/'signal-model.json').is_file():self.model_path.setText(str(bundled))
        for title,edit,folder in [('Signal model',self.model_path,True),('Signal audit',self.audit_path,False),('Output folder',self.output,True)]:
            button=QtWidgets.QPushButton('Browse…')
            button.clicked.connect(lambda checked=False,e=edit,f=folder:self.browse(e,f))
            row=QtWidgets.QHBoxLayout();row.addWidget(edit);row.addWidget(button);form.addRow(title,row)
        self.device=QtWidgets.QComboBox();self.device.addItems(['auto','gpu','cpu']);form.addRow('Device',self.device)
        self.epochs=QtWidgets.QSpinBox();self.epochs.setRange(1,1000);self.epochs.setValue(100);form.addRow('Maximum epochs',self.epochs)
        layout.addLayout(form)
        self.phase=QtWidgets.QLabel('Ready. First load neural cell contours using Auto-select contours or a saved cell run.')
        self.phase.setWordWrap(True);self.phase.setStyleSheet('font-size:16px;font-weight:bold;');layout.addWidget(self.phase)
        self.epoch_label=QtWidgets.QLabel('Epoch —');layout.addWidget(self.epoch_label)
        self.progress=QtWidgets.QProgressBar();layout.addWidget(self.progress)
        self.detail=QtWidgets.QLabel('Only signal centers inside neural cell contours are accepted.');self.detail.setWordWrap(True);layout.addWidget(self.detail)
        self.tabs=QtWidgets.QTabWidget();layout.addWidget(self.tabs,1)
        self.canvas=PlotCanvas();self.tabs.addTab(self.canvas,'Training graphs')
        results=QtWidgets.QWidget();result_layout=QtWidgets.QVBoxLayout(results)
        self.summary=QtWidgets.QLabel('No results yet.');self.summary.setWordWrap(True);self.summary.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse);result_layout.addWidget(self.summary)
        self.counts=QtWidgets.QTableWidget(0,4);self.counts.setHorizontalHeaderLabels(['Cell','Channel','Channel name','Signals']);result_layout.addWidget(self.counts)
        self.points=QtWidgets.QTableWidget(0,7);self.points.setHorizontalHeaderLabels(['Signal','Cell','Channel','X (µm)','Y (µm)','Z (µm)','Score']);result_layout.addWidget(self.points)
        for table in (self.counts,self.points):
            table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
            table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        actions=QtWidgets.QHBoxLayout()
        self.show_button=QtWidgets.QPushButton('Show signals in viewer');self.show_button.clicked.connect(self.show_result);self.show_button.setEnabled(False)
        folder_button=QtWidgets.QPushButton('Open output folder');folder_button.clicked.connect(self.open_output)
        actions.addWidget(self.show_button);actions.addWidget(folder_button);result_layout.addLayout(actions)
        workflow_button=QtWidgets.QPushButton('Filter cells / Export Imaris…')
        workflow_button.clicked.connect(self.open_workflow);result_layout.addWidget(workflow_button)
        note=QtWidgets.QLabel('Signal channels are inferred from A markers during training. Detection scores are learned heatmap scores, not calibrated confidence. Training precision/recall cover annotated cell regions only.')
        note.setWordWrap(True);result_layout.addWidget(note);self.tabs.addTab(results,'Results / counts')
        self.timer=QtCore.QTimer(self);self.timer.setInterval(2000);self.timer.timeout.connect(self.refresh_report)
        self.draw()

    def browse(self,edit,folder):
        path=QtWidgets.QFileDialog.getExistingDirectory(self,'Choose folder',edit.text()) if folder else QtWidgets.QFileDialog.getOpenFileName(self,'Choose audit',edit.text(),'JSON (*.json)')[0]
        if path:edit.setText(path)

    def error(self,message):QtWidgets.QMessageBox.warning(self,'Signal detection',message)

    def start(self):
        if self.job:return
        root=Path(__file__).resolve().parents[2]
        if self.host.workflow_panel and self.host.workflow_panel.busy:
            self.error('Wait for the scan workflow to finish, or cancel it first.');return
        self.timer.stop();self.watched=None
        mode='apply' if self.mode.currentIndex()==0 else 'train'
        arguments=['-m','fish_viewer.signals','--mode',mode,'--output',self.output.text(),'--device',self.device.currentText()]
        if mode=='train':
            if not Path(self.audit_path.text()).is_file():self.error('Choose a completed signal audit.');return
            arguments=[str(root / 'scripts/train_signal_centers.py'), '--index', self.audit_path.text(), '--output', str(Path(self.output.text()) / ('unet-training-'+uuid.uuid4().hex[:10])), '--epochs', str(self.epochs.value()), '--device', 'cuda' if self.device.currentText() == 'gpu' else self.device.currentText()]
        else:
            entry=self.host.entry
            if entry is None or self.host._pending_load:self.error('Open a scan first.');return
            archive=self.host._pipeline_archives.get((str(entry.image.resolve()),0))
            if not archive:self.error('Load neural cell contours first using Auto-select contours or a saved neural cell run.');return
            if not (Path(self.model_path.text())/'signal-model.json').is_file():self.error('Choose a trained signal model folder.');return
            arguments+=['--image',str(entry.image),'--cell-predictions',str(archive),'--model',self.model_path.text()]
            if entry.scene:arguments+=['--scene',str(entry.scene)]
        job_dir=Path(self.output.text()).resolve()/'.jobs'/uuid.uuid4().hex;job_dir.mkdir(parents=True)
        self.cancel_file=job_dir/'cancel'
        arguments+=['--cancel-file',str(self.cancel_file)]
        self.buffer=b'';self.finished_report=None;self.run_directory=None;self.stderr='';self.history=[];self.report=None
        self.draw();self.show_button.setEnabled(False);self.summary.setText('Processing…')
        self.job=QtCore.QProcess(self)
        self.job.setProcessEnvironment(process_environment(job_dir))
        self.job.readyReadStandardOutput.connect(self.read_events);self.job.readyReadStandardError.connect(self.read_errors)
        self.job.finished.connect(self.process_finished)
        self.job.errorOccurred.connect(lambda error:self.process_finished(-1) if error==QtCore.QProcess.ProcessError.FailedToStart else None)
        self.run_button.setEnabled(False);self.cancel_button.setEnabled(True);self.mode.setEnabled(False)
        self.phase.setText('Starting signal worker…');self.tabs.setCurrentIndex(0)
        self.epoch_label.setText('Applying saved signal model' if mode=='apply' else f'Epoch — / {self.epochs.value()}')
        from ..spots.runtime import interpreter
        self.job.start(interpreter() if mode == 'train' else sys.executable,arguments)

    def read_events(self):
        if not self.job:return
        self.buffer+=bytes(self.job.readAllStandardOutput())
        while b'\n' in self.buffer:
            line,self.buffer=self.buffer.split(b'\n',1)
            try:event=json.loads(line)
            except (ValueError,UnicodeDecodeError):continue
            if event['kind']=='run_started':self.run_directory=event['directory']
            elif event['kind']=='finished':self.finished_report=event['report']
            elif event['kind']=='epoch':self.history.append(event);self.draw()
            elif event['kind']=='progress':self.update_progress(event)

    def read_errors(self):
        if self.job:
            text=bytes(self.job.readAllStandardError()).decode(errors='replace');self.stderr=(self.stderr+text)[-6000:]
            with self.cancel_file.with_name('worker.log').open('a') as stream:stream.write(text)

    def update_progress(self,event):
        self.phase.setText(event['phase']);self.progress.setRange(0,event['total']);self.progress.setValue(event['current'])
        if 'epoch' in event:self.epoch_label.setText(f'Epoch {event["epoch"]} / {event["epochs"]}')
        if 'loss' in event:self.detail.setText(f'Current training loss: {event["loss"]:.4f}')
        elif event.get('file'):self.detail.setText(Path(event['file']).name)

    def process_finished(self,*args):
        if not self.job:return
        self.read_events();self.read_errors();old=self.job;self.job=None;old.deleteLater()
        self.run_button.setEnabled(True);self.cancel_button.setEnabled(False);self.mode.setEnabled(True)
        if self._shutdown:return
        if self.finished_report:
            self.display_report(self.finished_report)
            if self.report['mode']=='apply' and self.report['status']=='complete':self.show_result()
        else:
            self.phase.setText('Signal worker failed');self.error(self.stderr[-1500:] or 'The signal worker failed to start.')

    def display_report(self,report):
        if report.get('engine') not in ('signal-center-unet3d', 'signal-cnn3d', 'signal-spotiflow3d') or report.get('mode') not in ('train', 'apply'):raise ValueError('Choose a signal training or single-scan application report.')
        self.report=report;self.history=report.get('epochs',[]);self.draw()
        if report.get('progress'):self.update_progress(report['progress'])
        if report.get('model_directory'):self.model_path.setText(report['model_directory'])
        text=f'{report["status"]} · {report["directory"]}'
        if report['mode']=='train':
            if report['status'] != 'running' or 'epoch' not in report.get('progress', {}):
                self.epoch_label.setText(f'{len(self.history)} completed epochs · best epoch {report.get("best_epoch")}')
            for split in ('validation','test'):
                metric=report.get(split)
                if metric:
                    text+=f'\n{split.capitalize()}: {metric["tp"]} matched / {metric["references"]} signals; {metric["fp"]} false detections in scored regions.'
        else:
            self.epoch_label.setText('Signal detection complete' if report['status']=='complete' else 'Applying saved signal model')
            text+=f'\n{len(report.get("points",[]))} signals · {len({r["cell_id"] for r in report.get("counts",[])})} neural cells. CSV exports are in the output folder.'
        if report.get('error'):text+='\n'+report['error'][-1000:]
        self.summary.setText(text)
        self.counts.setRowCount(len(report.get('counts',[])))
        for i,row in enumerate(report.get('counts',[])):
            for j,value in enumerate((row['cell_id'],row['channel']+1,row['channel_name'],row['count'])):self.counts.setItem(i,j,QtWidgets.QTableWidgetItem(str(value)))
        self.points.setRowCount(len(report.get('points',[])))
        for i,row in enumerate(report.get('points',[])):
            for j,value in enumerate((row['signal_id'],row['cell_id'],row['channel']+1,*[f'{p:.3f}' for p in row['xyz']],f'{row["score"]:.3f}')):self.points.setItem(i,j,QtWidgets.QTableWidgetItem(str(value)))
        self.show_button.setEnabled(report['mode']=='apply' and report['status']=='complete')
        if report['status']!='running':self.tabs.setCurrentIndex(1 if report['mode']=='apply' else 0)

    def draw(self):
        axes=self.canvas.axes
        for ax in axes:ax.clear();ax.grid(alpha=.15);ax.set_xlabel('Epoch')
        x=[r['epoch'] for r in self.history]
        for key,label,color in [('loss','Training','#56dcc1'),('val_loss','Validation','#84bafa')]:
            if all(key in r for r in self.history):axes[0].plot(x,[r[key] for r in self.history],label=label,color=color)
        axes[0].set_title('Training loss');axes[0].legend()
        for ax,key in zip(axes[1:],('precision','recall','f05')):
            ax.plot(x,[(r.get('validation_selected') or r.get('validation') or r).get(key) for r in self.history],color='#56dcc1');ax.set_title('Validation '+key+(' · selected score' if any('validation_selected' in r or 'threshold' in r for r in self.history) else ' · score 0.5'));ax.set_ylim(0,1)
        self.canvas.style();self.canvas.draw_idle()

    def open_report(self):
        if self.job:return
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,'Open signal run',self.output.text(),'JSON (*.json)')
        if path:
            try:
                report=json.loads(Path(path).read_text());self.display_report(report)
                self.timer.stop();self.watched=Path(path) if report['status']=='running' else None
                if self.watched:self.timer.start()
            except Exception as exc:self.error(str(exc))

    def refresh_report(self):
        if self.watched and not self.job:
            try:self.display_report(json.loads(self.watched.read_text()))
            except (OSError,ValueError):return
            if self.report['status']!='running':self.timer.stop();self.watched=None

    def show_result(self):
        if self.report and self.report['mode']=='apply' and self.report['status']=='complete':
            self.host.open_signal_prediction(Path(self.report['directory'])/'report.json')

    def open_workflow(self):
        self.show_result()
        self.host.show_workflow()

    def open_output(self):
        if self.report:QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(self.report['directory']))

    def cancel(self):
        if self.job:self.cancel_file.touch();self.phase.setText('Stopping after the current operation…')

    def shutdown(self):
        self._shutdown=True;self.timer.stop()
        if self.job:
            process=self.job;self.cancel_file.touch()
            if not process.waitForFinished(5000):process.kill();process.waitForFinished(1500)
