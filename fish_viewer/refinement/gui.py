"""Live preprocessing, epoch and loss progress for the boundary worker."""
import json
from pathlib import Path
import sys
import uuid

from ..worker_runtime import process_environment

from PySide6 import QtCore, QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg


class TrainingDialog(QtWidgets.QDialog):
    model_ready = QtCore.Signal(str)

    def __init__(self, parent=None, output='', device='auto', kind='boundaries'):
        super().__init__(parent)
        self.kind=kind
        self.setWindowTitle('Train cell centers and boxes' if kind=='proposals' else 'Train 3D U-Net boundaries');self.resize(850,680)
        self.process=None;self.rows=[];self.buffer=b'';self.report=None
        layout=QtWidgets.QVBoxLayout(self)
        description=QtWidgets.QLabel('Train on complete manual cell boundaries without roundness filtering. '
            'Uses the acquisition split from an existing collection audit. Original scans remain unchanged.')
        description.setWordWrap(True);layout.addWidget(description)
        if kind=='proposals':
            description.setText('Train a tiny 3D center and box detector. Center loss uses complete marked cells only; '
                'unmarked cells and regions are ignored. Choose a calibrated cell collection cache.')
        form=QtWidgets.QFormLayout();layout.addLayout(form)
        self.audit=QtWidgets.QLineEdit()
        row=QtWidgets.QHBoxLayout();row.addWidget(self.audit)
        browse=QtWidgets.QPushButton('Browse…');browse.clicked.connect(self.choose_audit);row.addWidget(browse)
        form.addRow('Collection audit',row)
        self.output=QtWidgets.QLineEdit(output);form.addRow('Save training in',self.output)
        self.device=QtWidgets.QComboBox();self.device.addItems(['auto','cpu','gpu'])
        self.device.setCurrentText(device);form.addRow('Device',self.device)
        self.phase=QtWidgets.QLabel('Ready');self.phase.setWordWrap(True);layout.addWidget(self.phase)
        self.epoch=QtWidgets.QLabel('Epoch —');layout.addWidget(self.epoch)
        self.progress=QtWidgets.QProgressBar();layout.addWidget(self.progress)
        self.figure=Figure(figsize=(8,4),layout='constrained');self.canvas=FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas,1);self.axes=self.figure.subplots(1,2);self.draw()
        row=QtWidgets.QHBoxLayout();layout.addLayout(row)
        self.run=QtWidgets.QPushButton('Train');self.run.clicked.connect(self.start);row.addWidget(self.run)
        self.stop=QtWidgets.QPushButton('Cancel');self.stop.setEnabled(False)
        self.stop.clicked.connect(self.cancel);row.addWidget(self.stop)
        self.use=QtWidgets.QPushButton('Use trained model');self.use.setEnabled(False)
        self.use.clicked.connect(self.choose_model);row.addWidget(self.use)

    def choose_audit(self):
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,'Choose collection audit',self.audit.text(),'JSON (*.json)')
        if path:self.audit.setText(path)

    def draw(self):
        keys=('loss','box_mae_um') if self.kind=='proposals' else ('loss','dice')
        titles=('Masked center + box loss','Box face MAE (µm)') if self.kind=='proposals' else ('Boundary-weighted loss','Target-cell Dice')
        for ax,key,title in zip(self.axes,keys,titles):
            ax.clear()
            for split in ('training','validation'):
                ax.plot([r['epoch'] for r in self.rows],[r[split][key] for r in self.rows],label=split)
            ax.set(xlabel='Epoch',title=title);ax.grid(alpha=.2);ax.legend()
        self.canvas.draw_idle()

    def start(self):
        if not Path(self.audit.text()).is_file() or not self.output.text().strip():
            QtWidgets.QMessageBox.warning(self,'Training','Choose an existing audit and output folder.');return
        self.root=Path(self.output.text()).expanduser().resolve()/('boundary-training-'+uuid.uuid4().hex[:8])
        self.root.mkdir(parents=True);self.cancel_file=self.root/'cancel'
        self.rows=[];self.report=None;self.draw();self.errors='';self.buffer=b''
        self.run.setEnabled(False);self.stop.setEnabled(True);self.use.setEnabled(False)
        self.audit.setEnabled(False);self.output.setEnabled(False);self.device.setEnabled(False)
        process=QtCore.QProcess(self);self.process=process
        process.setProcessEnvironment(process_environment(quiet_tensorflow=True, preserve_pythonpath=True))
        process.readyReadStandardOutput.connect(self.read)
        process.readyReadStandardError.connect(self.read_errors)
        process.finished.connect(self.finished)
        process.errorOccurred.connect(lambda e:self.finished(-1) if e==QtCore.QProcess.ProcessError.FailedToStart else None)
        arguments=['-m','fish_viewer.proposals' if self.kind=='proposals' else 'fish_viewer.refinement',
            '--mode','train','--audit',self.audit.text(),'--patches',str(self.root/'patches'),
            '--output',str(self.root/'runs'),'--device',self.device.currentText(),'--cancel-file',str(self.cancel_file)]
        if self.kind!='proposals':arguments+=['--cache',str(self.root/'cache')]
        process.start(sys.executable,arguments)

    def read(self):
        if self.process is None:return
        self.buffer+=bytes(self.process.readAllStandardOutput())
        while b'\n' in self.buffer:
            line,self.buffer=self.buffer.split(b'\n',1)
            try:event=json.loads(line)
            except (ValueError,UnicodeDecodeError):continue
            self.handle_event(event)

    def handle_event(self,event):
        if event.get('kind')=='progress':
            self.phase.setText(event['phase']+(('\n'+Path(event['file']).name) if event.get('file') else ''))
            total,current=event.get('total',0),event.get('current',0)
            self.progress.setRange(0,total or 0);self.progress.setValue(current)
            self.progress.setFormat(f'{current} / {total}' if total else 'Working…')
            if event.get('epoch'):
                detail=''
                if 'loss' in event:
                    metric=f"box MAE {event['box_mae_um']:.3f} µm" if self.kind=='proposals' else f"Dice {event['dice']:.4f}"
                    detail=f" · loss {event['loss']:.4f} · {metric}"
                self.epoch.setText(f"Epoch {event['epoch']} / {event['epochs']}"+detail)
        elif event.get('kind')=='epoch':
            self.rows.append(event);self.draw()
        elif event.get('kind')=='finished':self.report=event['report']

    def read_errors(self):
        if self.process is None:return
        text=bytes(self.process.readAllStandardError()).decode(errors='replace')
        self.errors=(self.errors+text)[-8000:]
        with (self.root/'worker.log').open('a',encoding='utf-8') as stream:stream.write(text)

    def finished(self,code=0,*_):
        if self.process is None:return
        self.read();self.read_errors();process,self.process=self.process,None;process.deleteLater()
        report=self.report or {}
        complete=code==0 and report.get('status')=='complete'
        self.phase.setText('Training complete · best epoch '+str(report['best_epoch']) if complete else
                           ('Training cancelled' if report.get('status')=='cancelled' else 'Training failed: '+report.get('error',self.errors)))
        self.run.setEnabled(True);self.stop.setEnabled(False);self.use.setEnabled(complete)
        self.audit.setEnabled(True);self.output.setEnabled(True);self.device.setEnabled(True)

    def cancel(self):
        if self.process is not None:self.cancel_file.touch();self.phase.setText('Cancelling after the current operation…')

    def choose_model(self):
        if self.report and self.report.get('status')=='complete':
            self.model_ready.emit(self.report['model_directory']);self.accept()

    def closeEvent(self,event):
        if self.process is not None:
            self.cancel_file.touch()
            process=self.process
            if not process.waitForFinished(1000):
                process.kill();process.waitForFinished(1000)
        super().closeEvent(event)
