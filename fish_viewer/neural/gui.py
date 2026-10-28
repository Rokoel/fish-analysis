"""Qt controls and JSON event transport; no TensorFlow import in the viewer."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time
import uuid

import numpy as np
from ..worker_runtime import process_environment

from PySide6 import QtCore, QtWidgets

from ..training_gui import TrainingPanel
from ..pipeline.runner import _json
from ..models import LibraryEntry
from .config import NeuralConfig
from .precision import evidence_preset


class NeuralPanel(TrainingPanel):
    def __init__(self, host):
        self.neural_controls = {}
        self.neural_flags = {}
        self._collection_audit = None
        self._base_config = evidence_preset(NeuralConfig())
        self._auto_open_result=False
        self._buffer = b''
        self._finished_report = None
        self._run_directory = None
        self._cancel_requested = False
        super().__init__(host)
        self.mode.clear()
        self.mode.addItems(['Train StarDist 3D + validate + test', 'Apply a saved StarDist model'])
        self.mode.currentIndexChanged.connect(self.update_audit_controls)
        self.layout().itemAt(0).layout().itemAt(1).widget().setText('Epochs')
        self.trials.setRange(1,10000)
        self.trials.setValue(self._base_config.epochs)
        self.output.setText(str(Path.cwd()/'neural_runs'))
        self.status.setText('Ready. StarDist 3D learns from manual cells; unannotated voxels are ignored.')
        self.results.horizontalHeaderItem(4).setText('Dice: all refs')
        self.set_config(self._base_config)
        self.update_audit_controls()
        progress_box=QtWidgets.QGroupBox('Run progress')
        progress_layout=QtWidgets.QVBoxLayout(progress_box)
        self.phase_label=QtWidgets.QLabel('Ready')
        self.phase_label.setStyleSheet('font-size: 17px; font-weight: bold;')
        self.epoch_label=QtWidgets.QLabel('Epoch —')
        self.epoch_progress=QtWidgets.QProgressBar();self.epoch_progress.setFormat('Epochs completed: %v / %m')
        self.operation_progress=QtWidgets.QProgressBar()
        self.operation_label=QtWidgets.QLabel('Choose a collection and run training or prediction.')
        self.operation_label.setWordWrap(True)
        self.loss_label=QtWidgets.QLabel('Loss —')
        for widget in (self.phase_label,self.epoch_label,self.epoch_progress,self.operation_progress,self.operation_label,self.loss_label):
            progress_layout.addWidget(widget)
        self.layout().insertWidget(1,progress_box)
        self.watch_timer=QtCore.QTimer(self);self.watch_timer.setInterval(2000)
        self.watch_timer.timeout.connect(self.refresh_watched_run)
        self._watched_report=None

    def _setup_page(self):
        super()._setup_page()
        form = self.controls['voxel_um'].parentWidget().layout()
        self.exclude_border.setChecked(False)
        self.model_path = QtWidgets.QLineEdit()
        self.model_path.editingFinished.connect(self.restore_chosen_model)
        choose = QtWidgets.QPushButton('Choose model…')
        choose.clicked.connect(self.choose_model)
        row = QtWidgets.QHBoxLayout();row.addWidget(self.model_path);row.addWidget(choose)
        form.insertRow(0,'Saved model for application',row)
        self.device = QtWidgets.QComboBox();self.device.addItems(['auto','gpu','cpu'])
        form.insertRow(1,'Training / inference device',self.device)
        self.audit_path=QtWidgets.QLineEdit();self.audit_path.setReadOnly(True)
        choose_audit=QtWidgets.QPushButton('Choose audit…');choose_audit.clicked.connect(self.choose_audit)
        clear_audit=QtWidgets.QPushButton('Clear');clear_audit.clicked.connect(lambda:self.set_audit(None))
        audit_row=QtWidgets.QHBoxLayout();audit_row.addWidget(self.audit_path);audit_row.addWidget(choose_audit);audit_row.addWidget(clear_audit)
        form.insertRow(2,'Audited collection for training',audit_row)
        evidence=QtWidgets.QPushButton('Use visible boundaries · ignore roundness')
        evidence.clicked.connect(self.use_evidence_preset);form.addRow(evidence)
        for key,label in [('native_sampling','Use source XY resolution when coarser'),
                          ('streaming_validation','Validate every annotated crop each epoch'),
                          ('precision_mode','Require membrane evidence around every accepted cell'),
                          ('use_shape_filter','Limit roundness and elongation (optional)'),
                          ('curate_training_shapes','Train only on complete cells within shape limits'),
                          ('select_balanced_acceptance','Select acceptance on validation with negative controls'),
                          ('calibrate_probability','Choose probability threshold by reference Dice (recall oriented)')]:
            control=QtWidgets.QCheckBox(label);self.neural_flags[key]=control;form.addRow(control)
        self.neural_flags['precision_mode'].toggled.connect(self.precision_changed)
        self.neural_flags['use_shape_filter'].toggled.connect(self.shape_filter_changed)
        self.overlap_policy=QtWidgets.QComboBox()
        self.overlap_policy.addItem('Discard overlapping annotations','discard')
        self.overlap_policy.addItem('Retain complete annotations in separate crops','isolate')
        form.addRow('Overlapping cells',self.overlap_policy)
        for key,label,low,high,step in [
            ('steps_per_epoch','Batches per epoch',1,10000,1),('batch_size','Batch size',1,16,1),
            ('patch_z','Patch depth (voxels)',16,128,16),('patch_xy','Patch width / height',16,256,16),
            ('patience','Early-stop patience (epochs)',1,1000,1),
            ('validate_every','Measure contours every N epochs',1,1000,1)]:
            control=QtWidgets.QSpinBox();control.setRange(low,high);control.setSingleStep(step)
            self.neural_controls[key]=control;form.addRow(label,control)
        preview_count=QtWidgets.QSpinBox();preview_count.setRange(0,10000)
        self.neural_controls['validation_images']=preview_count
        form.addRow('Graph preview images (0 = all)',preview_count)
        for key,label,low,high,step in [
            ('learning_rate','Learning rate',1e-7,.01,.0001),
            ('probability_threshold','Cell probability threshold',.05,.95,.05),
            ('nms_threshold','Object overlap threshold',.05,.9,.05),
            ('precision_max_diameter_um','Conservative maximum cell diameter (µm)',1,500,.5),
            ('membrane_support','Required supported surface fraction',.1,1,.05),
            ('membrane_contrast','Minimum membrane contrast',0,1,.01),
            ('membrane_intensity','Minimum membrane intensity',0,1,.01),
            ('synthetic_background_fraction','Synthetic blank crops per real crop',0,1,.05)]:
            control=QtWidgets.QDoubleSpinBox();control.setDecimals(7);control.setRange(low,high);control.setSingleStep(step)
            self.neural_controls[key]=control;form.addRow(label,control)
        info=QtWidgets.QLabel('Unknown image regions are ignored during training. Synthetic blank controls do not replace reviewed false detections. '
            'Conservative selection favors fewer contours; its scores are not calibrated confidence. '
            'Apply uses saved preprocessing and permits conservative acceptance overrides.')
        info.setWordWrap(True);form.addRow(info)

    def precision_changed(self,enabled):
        calibration=self.neural_flags['calibrate_probability']
        if enabled:calibration.setChecked(False)
        calibration.setEnabled(not enabled)
        if not enabled and 'select_balanced_acceptance' in self.neural_flags:
            self.neural_flags['select_balanced_acceptance'].setChecked(False)



    def use_evidence_preset(self):
        self.set_config(evidence_preset(self.config()))

    def shape_filter_changed(self,enabled):
        for key in ('min_sphericity','max_elongation'):self.controls[key].setEnabled(enabled)

    def config(self):
        keys=('voxel_um','min_diameter_um','max_diameter_um','min_sphericity','max_elongation')
        values={k:self.controls[k].value() for k in keys}
        values.update({k:v.value() for k,v in self.neural_controls.items()})
        values.update({k:v.isChecked() for k,v in self.neural_flags.items()})
        values['overlap_policy']=self.overlap_policy.currentData()
        return replace(self._base_config,**values,exclude_border=self.exclude_border.isChecked(),
            epochs=self.trials.value(),device=self.device.currentText()).validate()

    def set_config(self, config):
        if not isinstance(config,NeuralConfig):
            config=evidence_preset(NeuralConfig())
        self._base_config=config
        for key,control in self.controls.items():
            if hasattr(config,key):control.setValue(getattr(config,key))
        for key,control in self.neural_controls.items():control.setValue(getattr(config,key))
        for key,control in self.neural_flags.items():control.setChecked(getattr(config,key))
        self.precision_changed(config.precision_mode)
        self.shape_filter_changed(config.use_shape_filter)
        self.overlap_policy.setCurrentIndex(self.overlap_policy.findData(config.overlap_policy))
        self.exclude_border.setChecked(config.exclude_border)
        self.trials.setValue(config.epochs);self.device.setCurrentText(config.device)

    def choose_model(self):
        path=QtWidgets.QFileDialog.getExistingDirectory(self,'Select StarDist model folder',self.model_path.text())
        if path:
            self.model_path.setText(path);self.restore_chosen_model()

    def restore_chosen_model(self):
        path=Path(self.model_path.text())/'fish-model.json'
        if not path.is_file():return
        try:
            manifest=json.loads(path.read_text())
            if manifest.get('engine')!='stardist3d':raise ValueError('Not a supported StarDist model.')
            self.set_config(replace(NeuralConfig.from_dict(manifest['config']),device=self.device.currentText()))
            self.status.setText('Loaded the chosen model’s saved preprocessing and acceptance settings.')
        except Exception as exc:self.show_error(str(exc))

    def apply_to_scan(self,entry,model_path):
        if self.job:raise ValueError('A neural operation is already running. Wait for it or cancel it first.')
        manifest=json.loads((Path(model_path)/'fish-model.json').read_text())
        config=replace(evidence_preset(NeuralConfig.from_dict(manifest['config'])),device=self.device.currentText())
        self.mode.setCurrentIndex(1);self.set_entries([entry]);self.model_path.setText(str(model_path))
        self.set_config(config);self._auto_open_result=True
        self.start()
        if not self.job:self._auto_open_result=False

    def _results_page(self):
        super()._results_page()
        button=QtWidgets.QPushButton('Review selected cell contours…')
        button.clicked.connect(self.review_selected)
        self.tabs.widget(3).layout().insertWidget(2,button)

    def review_selected(self):
        index=self.results.currentRow()
        if index<0:return
        row=self.results.item(index,0).data(QtCore.Qt.ItemDataRole.UserRole)
        try:
            image_path=Path(row['predictions']).parent/'image.npy'
            if not image_path.exists() and (self.report or {}).get('audit_path'):
                audit=json.loads(Path(self.report['audit_path']).read_text())
                case=next(c for c in audit['cases'] if c['image']==row['image'])
                image_path=Path(case['directory'])/'image.npy'
            from .review import ReviewDialog
            dialog=ReviewDialog(row['predictions'],image_path,self,
                on_export=lambda path:self.host.open_pipeline_prediction(dict(row,predictions=str(path))))
            dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
            dialog.show()
        except Exception as exc:self.show_error(str(exc))

    def choose_audit(self):
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,'Choose completed collection audit','','JSON (*.json)')
        if path:
            try:self.set_audit(path,restore_config=True)
            except Exception as exc:self.show_error(str(exc))

    def set_audit(self,path,restore_config=False):
        audit=json.loads(Path(path).read_text()) if path else None
        if audit and (audit.get('engine')!='stardist3d-audit' or audit.get('status')!='complete'):
            raise ValueError('Select a completed collection audit.json.')
        self._collection_audit=audit;self.audit_path.setText(str(Path(path).resolve()) if path else '')
        if audit:
            self.set_entries([LibraryEntry(Path(c['image']),Path(c['scene']) if c['scene'] else None) for c in audit['cases']])
            if restore_config:self.set_config(NeuralConfig.from_dict(audit['config']))
        self.update_audit_controls()

    def update_audit_controls(self):
        active=self.mode.currentIndex()==0 and self._collection_audit is not None
        self.files.setEnabled(not active)
        layout=self.setup_page.layout().itemAt(0).layout()
        for index in (0,2):
            buttons=layout.itemAt(index).layout()
            for i in range(buttons.count()):
                widget=buttons.itemAt(i).widget()
                if isinstance(widget,QtWidgets.QPushButton):widget.setEnabled(not active)
        if active:self.file_count.setText(f'{len(self._collection_audit["cases"])} audited scans · fixed split')
        else:self.file_count.setText(f'{len(self.entries)} images')

    def load_settings(self):
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,'Load StarDist settings','','JSON (*.json)')
        if path:
            try:
                settings=json.loads(Path(path).read_text())
                if settings.get('engine')!='stardist3d':raise ValueError('Not StarDist settings.')
                self.set_config(NeuralConfig.from_dict(settings['config']))
            except Exception as exc:self.show_error(str(exc))

    def save_settings(self):
        path,_=QtWidgets.QFileDialog.getSaveFileName(self,'Save StarDist settings','neural-settings.json','JSON (*.json)')
        if path:
            try:_json(Path(path),dict(version=1,engine='stardist3d',config=asdict(self.config())))
            except Exception as exc:self.show_error(str(exc))

    def start(self):
        if self.job:return
        if self.host.workflow_panel and self.host.workflow_panel.busy:
            self.show_error('Wait for the scan workflow to finish, or cancel it first.');return
        self.watch_timer.stop();self._watched_report=None
        try:
            config=self.config();entries=self.checked_entries()
            if not entries:raise ValueError('Select input images.')
            if not self.output.text().strip():raise ValueError('Choose an output folder.')
            mode='train' if self.mode.currentIndex()==0 else 'apply'
            audit=self.audit_path.text() if mode=='train' and self._collection_audit else None
            if audit:
                entries=[LibraryEntry(Path(c['image']),Path(c['scene']) if c['scene'] else None) for c in self._collection_audit['cases']]
            if mode=='apply' and not (Path(self.model_path.text())/'fish-model.json').is_file():
                raise ValueError('Select a trained StarDist model folder containing fish-model.json.')
            job_dir=Path(self.output.text()).resolve()/'.jobs'/uuid.uuid4().hex
            job_dir.mkdir(parents=True)
            self.cancel_file=job_dir/'cancel'
            request=dict(config=asdict(config),mode=mode,output=str(Path(self.output.text()).resolve()),
                audit=audit,
                model=self.model_path.text() or None,cancel_file=str(self.cancel_file),
                entries=[dict(image=str(e.image.resolve()),scene=str(e.scene.resolve()) if e.scene else None) for e in entries])
            _json(job_dir/'request.json',request)
        except Exception as exc:
            self.show_error(str(exc));return
        self.history=[];self.final_scores=None;self.report=None;self.previews={};self.channel_scores=[];self.timings={}
        self.results.setRowCount(0);self.log.clear();self.summary.setText('Neural run in progress.')
        self._buffer=b'';self._finished_report=None;self._run_directory=None;self._cancel_requested=False
        self._last_stderr=''
        self.draw_scores();self.draw_preprocessing()
        self.job=QtCore.QProcess(self)
        self.job.setProcessEnvironment(process_environment(job_dir))
        self.job.readyReadStandardOutput.connect(self.read_events)
        self.job.readyReadStandardError.connect(self.read_errors)
        self.job.finished.connect(self.process_finished)
        self.job.errorOccurred.connect(self.process_error)
        self.set_running(True);self.cancel_button.setText('Cancel')
        self.started=time.monotonic();self.timer.start();self.tabs.setCurrentIndex(1)
        self.status.setText('Starting neural worker…')
        self.update_progress(dict(phase='Starting worker',current=0,total=0))
        self.epoch_label.setText(f'Epoch — / {config.epochs}' if mode=='train' else 'Applying saved model')
        self.epoch_progress.setRange(0,config.epochs);self.epoch_progress.setValue(0)
        self.loss_label.setText('Loss —')
        self.job.start(sys.executable,['-m','fish_viewer.neural','--request',str(job_dir/'request.json')])

    def read_events(self):
        if not self.job:return
        self._buffer+=bytes(self.job.readAllStandardOutput())
        while b'\n' in self._buffer:
            line,self._buffer=self._buffer.split(b'\n',1)
            try:
                event=json.loads(line)
            except (ValueError,UnicodeDecodeError):
                if line.strip():self.log.appendPlainText(line.decode(errors='replace')[-500:])
                continue
            if event['kind']=='finished':self._finished_report=event['report']
            else:self.handle_event(event)

    def read_errors(self):
        if self.job:
            text=bytes(self.job.readAllStandardError()).decode(errors='replace')
            self._last_stderr=(self._last_stderr+text)[-6000:]
            # Keep full library diagnostics on disk without flooding live graphs.
            if hasattr(self,'cancel_file'):
                with self.cancel_file.with_name('worker.log').open('a',encoding='utf-8') as stream:stream.write(text)

    def process_error(self, error):
        if error==QtCore.QProcess.ProcessError.FailedToStart:
            self.process_finished(-1,QtCore.QProcess.ExitStatus.CrashExit)

    def process_finished(self, exit_code, status):
        if not self.job:return
        self.read_events();self.read_errors()
        report=self._finished_report
        if report is None and self._run_directory:
            path=Path(self._run_directory)/'report.json'
            if path.exists():
                report=json.loads(path.read_text())
                report['status']='cancelled' if self._cancel_requested else 'failed'
                report['elapsed_s']=time.monotonic()-self.started
                report['error']=f'Worker exited before completion (code {exit_code}). '+self._last_stderr[-1200:]
                _json(path,report)
        old=self.job;self.job=None;old.deleteLater()
        self.cancel_button.setText('Cancel')
        if self._shutdown:return
        if report is not None:self.completed(report)
        else:self.failed('Neural worker could not start or finish. '+self._last_stderr[-1200:])

    def cancel(self):
        if self.job:
            if self._cancel_requested:
                self.job.kill()
            else:
                self._cancel_requested=True;self.cancel_file.touch()
                self.status.setText('Stopping after the current operation; saved checkpoints will remain.')
                self.cancel_button.setText('Stop now')

    def handle_event(self,event):
        if self._shutdown:return
        kind=event['kind']
        if kind=='run_started':self._run_directory=event['directory']
        elif kind=='progress':self.update_progress(event)
        elif kind=='tile':
            self.operation_label.setText(f'{Path(event["file"]).name} · volume tile {event["current"]} / {event["total"]}')
        elif kind=='runtime':self.log.appendPlainText('TensorFlow device: '+event['device'])
        elif kind=='neural_split':
            for name,paths in event['split'].items():self.log.appendPlainText(name.capitalize()+':\n'+'\n'.join(paths))
        elif kind=='audit':
            a=event['audit']
            self.log.appendPlainText(f'{Path(event["file"]).name}: {a["training_objects"]} training cells, '
                f'{a["cut_references"]} cut references, {a["background_voxels"]} reviewed background voxels.')
            self.previews['Candidate interiors']=np.maximum(np.asarray(event['preview']),0)
            self.draw_preprocessing()
        elif kind=='batch':
            self.progress.setValue(round(100*event['batch']/event['total']))
            self.status.setText(f'Epoch {len(self.history)+1}: batch {event["batch"]}/{event["total"]}, loss {event["loss"]:.4f}')
            self.loss_label.setText(f'Current training loss: {event["loss"]:.4f}')
        elif kind=='validation_batch':
            self.status.setText(f'Epoch {len(self.history)+1}: validating batch {event["batch"]}')
        elif kind=='epoch':
            self.history.append(event);self.draw_scores()
            self.epoch_progress.setValue(event['epoch'])
            self.loss_label.setText(f'Completed epoch {event["epoch"]}: training loss {event["losses"]["loss"]:.4f} · '
                f'validation loss {event["losses"]["val_loss"]:.4f} · best validation loss {event["best_val_loss"]:.4f}')
            self.status.setText(f'Epoch {event["epoch"]}: training loss {event["losses"]["loss"]:.4f}, validation {event["losses"]["val_loss"]:.4f}')
        elif kind=='calibration':
            m=event['validation']['complete_cells']
            self.log.appendPlainText(f'Validation threshold {event["probability_threshold"]:.2f}: complete-cell Dice {m["mean_dice"]:.3f}, '
                f'{m["predicted_count"]} candidates. Unmatched predictions remain unscored.')
        elif kind=='balanced_candidate':
            row=event['candidate'];settings=row['config']
            self.log.appendPlainText(f'Acceptance p={settings["probability_threshold"]:.2f}, membrane={settings["membrane_support"]:.2f}: '
                + (f'validation detection {row["eligible_detection_rate"]:.1%}, {row["predicted_count"]} contours; negative controls passed.'
                   if row['controls']['passed'] else 'rejected by negative controls.'))
        else:super().handle_event(event)

    def update_progress(self,event):
        phase=event['phase'];current=int(event.get('current',0));total=int(event.get('total',0))
        split=event.get('split')
        self.phase_label.setText(phase+(f' · {split}' if split else ''))
        self.operation_progress.setRange(0,total);self.operation_progress.setValue(current)
        self.operation_progress.setFormat(f'{phase}: %v / %m' if total else phase+'…')
        if 'epoch' in event:
            epoch=event['epoch'];epochs=event['epochs']
            self.epoch_label.setText(f'Epoch {epoch} / {epochs}')
            self.epoch_progress.setRange(0,epochs);self.epoch_progress.setValue(epoch-1)
        if 'loss' in event:self.loss_label.setText(f'Current training loss: {event["loss"]:.4f}')
        if event.get('file'):
            self.operation_label.setText(f'Image {min(current+1,total)} / {total}: {Path(event["file"]).name}')
            self.operation_label.setToolTip(event['file'])
        else:self.operation_label.setText(f'{current} / {total}' if total else 'Initializing…')
        if phase=='Complete':
            count=event.get('completed_epochs',len(self.history))
            self.epoch_progress.setRange(0,max(count,(self.report or {}).get('config',{}).get('epochs',count),1))
            self.epoch_progress.setValue(count)
            self.epoch_label.setText(f'Completed {count} epochs · best epoch {event.get("best_epoch")}' if count else 'Prediction complete')
        self.status.setText(self.phase_label.text())

    def draw_scores(self):
        axes=self.score_canvas.axes
        for ax in axes:ax.clear();ax.set_xlabel('Epoch');ax.grid(alpha=.15)
        rows=self.history
        for key,label,color in [('loss','Training','#56dcc1'),('val_loss','Validation','#84bafa')]:
            axes[0].plot([r['epoch'] for r in rows],[r['losses'].get(key,np.nan) for r in rows],label=label,color=color)
        axes[0].set_title('StarDist total loss: probability + shape')
        specs=[('mean_dice','Validation overlap: complete cells'),('detection_rate','Detected complete cells (IoU ≥ 0.5)'),
               ('boundary_rmse_um','Matched complete-cell boundary RMSE (µm)')]
        scored=[r for r in rows if 'validation' in r]
        preview=self.neural_controls['validation_images'].value()>0
        for ax,(key,title) in zip(axes[1:],specs):
            ax.set_title(title)
            ax.plot([r['epoch'] for r in scored],[r['validation']['complete_cells'].get(key) for r in scored],
                'o-',color='#56dcc1',label='Validation preview' if preview else 'Validation',markersize=3)
            if key=='mean_dice':
                ax.plot([r['epoch'] for r in scored],[r['validation']['complete_cells'].get('mean_iou') for r in scored],
                    color='#84bafa',label='IoU')
            selected=((self.report or {}).get('calibrated_validation') or (self.report or {}).get('selected_validation',{})).get('complete_cells',{}).get(key)
            if selected is not None:
                ax.scatter([(self.report or {}).get('best_epoch',0)],[selected],marker='D',s=45,
                    color='#d4a0ee',label='Selected: full validation')
            final=(self.final_scores or {}).get('complete_cells',{}).get(key)
            if final is not None:ax.scatter([max([r['epoch'] for r in rows],default=0)+1],[final],
                marker='*',s=90,color='#edb867',label='Test: final only')
            if key!='boundary_rmse_um':ax.set_ylim(-.02,1.02)
        for ax in axes:ax.legend(fontsize=7,facecolor='#17232f',labelcolor='#e0e8f1',edgecolor='#405366')
        self.score_canvas.style();self.score_canvas.draw_idle()

    def draw_preprocessing(self):
        super().draw_preprocessing()
        self.preprocess_canvas.axes[1].set_title('Normalized network input')
        self.preprocess_canvas.axes[2].set_title('Manual training cells; unknown ignored')
        self.preprocess_canvas.draw_idle()

    def completed(self,report):
        self.history=report.get('epochs',[])
        super().completed(report)
        if report.get('progress'):self.update_progress(report['progress'])
        if self.history:
            last=self.history[-1]
            self.loss_label.setText(f'Completed epoch {last["epoch"]}: training loss {last["losses"]["loss"]:.4f} · validation loss {last["losses"]["val_loss"]:.4f}')
            if 'epoch' not in report.get('progress',{}):
                maximum=report['config']['epochs']
                self.epoch_progress.setRange(0,maximum);self.epoch_progress.setValue(last['epoch'])
                self.epoch_label.setText(f'Completed {last["epoch"]} / {maximum} epochs · best epoch {report.get("best_epoch")}')
        if report['status'] in ('failed','cancelled'):self.phase_label.setText(report['status'].capitalize())
        if report.get('model_directory'):self.model_path.setText(report['model_directory'])
        if report.get('config'):self.set_config(NeuralConfig.from_dict(report['config']))
        if report.get('audit_path'):
            try:self.set_audit(report['audit_path'])
            except (OSError,ValueError) as exc:self.log.appendPlainText(f'Collection audit unavailable: {exc}')
        complete=report.get('evaluation',{}).get('complete_cells',{})
        if complete.get('mean_dice') is not None:
            self.summary.setText(self.summary.text()+f'\nComplete test cells: Dice {complete["mean_dice"]:.3f}, '
                f'detected {complete["detection_rate"]:.1%}; best epoch {report.get("best_epoch")}.')
        auto_open=self._auto_open_result;self._auto_open_result=False
        if auto_open and report['status']=='complete' and report['files']:
            self.predicted_visible.setChecked(True)
            self.host.set_surface_source_visible('prediction',True)
            self.host.open_pipeline_prediction(report['files'][0])
            self.host.neural_dock.hide()

    def failed(self,details):
        self._auto_open_result=False
        super().failed(details)

    def load_report(self):
        if self.job:return
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,'Open StarDist run',self.output.text(),'JSON (*.json)')
        if path:
            try:
                report=json.loads(Path(path).read_text());self.display_report(report)
                self.watch_timer.stop();self._watched_report=None
                if report['status']=='running':
                    self._watched_report=Path(path);self.watch_timer.start()
            except Exception as exc:self.show_error(str(exc))

    def refresh_watched_run(self):
        if not self._watched_report or self.job:return
        try:report=json.loads(self._watched_report.read_text())
        except (OSError,ValueError):return
        changed=len(report.get('epochs',[]))!=len(self.history)
        self.report=report;self.history=report.get('epochs',[])
        if changed:
            self.draw_scores()
            last=self.history[-1]
            self.loss_label.setText(f'Completed epoch {last["epoch"]}: training loss {last["losses"]["loss"]:.4f} · validation loss {last["losses"]["val_loss"]:.4f}')
        if report.get('progress'):self.update_progress(report['progress'])
        self.elapsed.setText(f'{int(report.get("elapsed_s",0))} s')
        self.summary.setText(f'Watching running job · {len(report["files"])} output datasets\n{report["directory"]}')
        for row in report['files'][self.results.rowCount():]:self.add_result(row)
        if report['status']!='running':
            self.watch_timer.stop();self._watched_report=None;self.completed(report)

    def display_report(self,report):
        if report.get('version')!=1 or report.get('engine')!='stardist3d':raise ValueError('Not a supported StarDist run.')
        self.results.setRowCount(0);self.log.clear()
        for row in report['files']:self.add_result(row)
        self.completed(report)

    def shutdown(self):
        self._shutdown=True;self.timer.stop()
        self.watch_timer.stop()
        if self.job:
            process=self.job
            self._cancel_requested=True;self.cancel_file.touch()
            if not process.waitForFinished(5000):
                process.kill();process.waitForFinished(1500)
