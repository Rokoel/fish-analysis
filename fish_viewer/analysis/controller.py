"""Run the existing inference workers in order without loading TensorFlow into Qt."""
from dataclasses import asdict, replace
from datetime import datetime
import json
from pathlib import Path
import sys
import uuid

from ..worker_runtime import process_environment
from ..project_paths import bundled_model

from PySide6 import QtCore

from ..neural.config import NeuralConfig
from ..neural.model_picker import read_model
from ..neural.precision import evidence_preset


STAGES = [('cells', 'Detecting cell contours'), ('signals', 'Finding signals inside cells'),
          ('filter', 'Removing cells without signals'), ('measure', 'Measuring border and signal-pair distances'),
          ('export', 'Saving annotated Imaris image and Excel workbook')]


def default_models():
    cells, signals = bundled_model('cells-sensitive'), bundled_model('signals-center-unet')
    if not (cells / 'fish-model.json').exists():
        raise FileNotFoundError('The bundled sensitive StarDist model for consensus is missing.')
    if not (signals / 'signal-model.json').is_file():
        raise FileNotFoundError('The bundled signal-center U-Net model is missing.')
    return cells, signals


def sensitive_stardist_model():
    path = bundled_model('cells-sensitive')
    return str(path) if (path / 'fish-model.json').is_file() else ''


def default_refinement_model():
    path = bundled_model('refinement-bias-candidate')
    return str(path) if (path / 'refinement-model.json').is_file() else ''


def default_proposal_model():
    path = bundled_model('proposals')
    return str(path) if (path / 'proposal-model.json').is_file() else ''


def migrate_bundled_defaults(settings):
    """Keep a supported explicit choice; migrate all retired choices to consensus."""
    pipeline = settings.value('pipeline', 'consensus')
    if pipeline not in ('consensus', 'stardist2d'):
        pipeline = 'consensus'
    settings.setValue('pipeline', pipeline)
    settings.setValue('cell_model', str(bundled_model('cells-stardist2d' if pipeline == 'stardist2d' else 'cells-sensitive')))
    settings.setValue('signal_model', str(default_models()[1]))
    settings.setValue('boundary_model', default_refinement_model() if pipeline == 'consensus' else '')
    settings.setValue('default_pipeline_version', 'two-pipelines-1')


def check_models(cells, signals):
    manifest = json.loads((Path(cells) / 'fish-model.json').read_text())
    if manifest.get('engine') == 'stardist2d-slice':
        from ..neural.slice_inference import read_model as read_slice_model
        manifest = read_slice_model(cells)
    elif manifest.get('engine') == 'stardist3d':
        manifest = read_model(cells)
    else:
        raise ValueError('Choose the consensus StarDist detector or fine-tuned StarDist 2D.')
    from ..signals.model import read_model as read_signal_model
    read_signal_model(signals)
    return manifest


class AnalysisController(QtCore.QObject):
    progress = QtCore.Signal(object)
    completed = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    cancelled = QtCore.Signal()
    busy_changed = QtCore.Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.job = None
        self.busy = False
        self.stopping = False
        self.closing = False

    def start(self, image, output, cells=None, signals=None, device='auto', refinement=None,
              stardist_proposals=None, consensus_model=None, roundness=.82, cell_channel=None,
              signal_channels=None, source_scene=None, maximum_pair_distance=1., pipeline=None):
        if self.busy:
            raise ValueError('A scan is already being processed.')
        import math
        if not math.isfinite(maximum_pair_distance) or maximum_pair_distance <= 0:
            raise ValueError('The maximum signal-pair distance must be positive and finite.')
        if pipeline not in (None, 'consensus', 'stardist2d'):
            raise ValueError('Only both-detector consensus and fine-tuned StarDist 2D are supported.')
        cells = cells or bundled_model('cells-stardist2d' if pipeline == 'stardist2d' else 'cells-sensitive')
        signals = signals or default_models()[1]
        manifest = check_models(cells, signals)
        inferred = 'stardist2d' if manifest.get('engine') == 'stardist2d-slice' else 'consensus'
        if pipeline is not None and pipeline != inferred:
            raise ValueError('The cell checkpoint does not match the chosen pipeline.')
        self.pipeline = inferred
        self.roundness = float(roundness) if roundness is not None else None
        if self.roundness is not None and not 0 <= self.roundness <= 1:
            raise ValueError('Minimum roundness must lie between 0 and 1.')
        if cell_channel is not None and (type(cell_channel) is not int or cell_channel < 0):
            raise ValueError('Cell channel must be a nonnegative channel index.')
        if signal_channels is not None and (not signal_channels or any(type(i) is not int or i < 0 for i in signal_channels) or len(set(signal_channels)) != len(signal_channels)):
            raise ValueError('Select distinct nonnegative signal channels.')
        if cell_channel is not None and signal_channels is not None and cell_channel in signal_channels:
            raise ValueError('Cell and signal channels must be different.')
        self.image = Path(image).expanduser().resolve()
        if not self.image.is_file():
            raise FileNotFoundError(self.image)
        if device not in ('auto', 'cpu', 'gpu'):
            raise ValueError('Unknown processing device.')
        self.device, self.maximum_pair_distance = device, maximum_pair_distance
        self.source_scene = str(Path(source_scene).resolve()) if source_scene else None
        self.cell_channel, self.signal_channels = cell_channel, signal_channels
        self.cell_channel_arguments = ['--cell-channel', str(cell_channel)] if cell_channel is not None else []
        self.signal_channel_arguments = ['--signal-channels', *map(str, signal_channels)] if signal_channels is not None else []
        self.signal_model = str(Path(signals).resolve())
        self.refinement_model = str(Path(refinement or default_refinement_model()).resolve()) if inferred == 'consensus' else None
        self.consensus_model = str(Path(consensus_model or default_proposal_model()).resolve()) if inferred == 'consensus' else None
        self.stardist_proposals = inferred == 'consensus'
        if inferred == 'consensus':
            from ..refinement.config import read_model as read_refinement
            from ..proposals.config import read_model as read_proposals
            read_refinement(self.refinement_model)
            read_proposals(self.consensus_model)
        self.stages = list(STAGES)
        if inferred == 'consensus':
            self.stages[1:1] = [('refine', 'Redrawing and checking StarDist proposals'),
                                ('consensus_cells', 'Finding cells with the center/box model'),
                                ('consensus', 'Keeping cells detected by both approaches')]
        if self.roundness is not None and self.roundness > 0:
            self.stages.insert(next(i for i,s in enumerate(self.stages) if s[0] == 'signals'), ('shape', 'Checking 3D cell roundness'))
        self.directory = Path(output).expanduser().resolve() / f'analysis-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}'
        self.directory.mkdir(parents=True)
        self.cancel_file = self.directory / 'cancel'
        self.stopping = False
        self.state = dict(version=1, engine='fish-analysis', status='running', source=str(self.image),
                          source_scene=self.source_scene, pipeline=inferred, cell_model=str(Path(cells).resolve()),
                          signal_model=self.signal_model, device=device, processed_time=0, stages={},
                          refinement_model=self.refinement_model, consensus_model=self.consensus_model,
                          cell_detector_engine=manifest.get('engine', 'stardist3d'), stardist_proposals=self.stardist_proposals,
                          minimum_roundness=self.roundness, maximum_pair_distance_um=maximum_pair_distance,
                          channel_roles=dict(cell=cell_channel, signals=signal_channels))
        self._persist()
        self.busy = True
        self.busy_changed.emit(True)
        if inferred == 'stardist2d':
            self._launch(0, ['-m', 'fish_viewer.neural.slice_inference', '--image', str(self.image),
                            '--model', str(Path(cells).resolve()), '--output', str(self.directory/'work'),
                            '--device', device, '--cancel-file', str(self.cancel_file)] + self.cell_channel_arguments)
        else:
            config = replace(NeuralConfig.from_dict(manifest['config']), device=device, cell_channel=cell_channel)
            request = dict(config=asdict(config), mode='apply', model=str(Path(cells).resolve()),
                           output=str(self.directory/'work'), cancel_file=str(self.cancel_file),
                           entries=[dict(image=str(self.image), scene=None)])
            target = self.directory/'cells-request.json'
            target.write_text(json.dumps(request, indent=2), encoding='utf-8')
            self._launch(0, ['-m', 'fish_viewer.neural', '--request', str(target)])

    def _persist(self):
        target = self.directory / 'analysis-result.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.state, indent=2, allow_nan=False), encoding='utf-8')
        temporary.replace(target)

    def _launch(self, index, arguments):
        self.stage_index = index
        self.stage = self.stages[index][0]
        self.buffer, self.errors, self.finished_report = b'', '', None
        self.state['current_stage'] = self.stage
        self._persist()
        self.progress.emit(dict(stage=index+1, stages=len(self.stages), stage_name=self.stages[index][1],
                                phase=self.stages[index][1], current=0, total=0))
        process = QtCore.QProcess(self)
        self.job = process
        process.setProcessEnvironment(process_environment(self.directory, quiet_tensorflow=True))
        process.readyReadStandardOutput.connect(self._read)
        process.readyReadStandardError.connect(self._read_errors)
        process.finished.connect(self._finished)
        process.errorOccurred.connect(lambda error: self._finished(-1) if error == QtCore.QProcess.ProcessError.FailedToStart else None)
        process.start(sys.executable, arguments)

    def _read(self):
        if self.job is None:
            return
        self.buffer += bytes(self.job.readAllStandardOutput())
        while b'\n' in self.buffer:
            line, self.buffer = self.buffer.split(b'\n', 1)
            try:
                event = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                continue
            if event.get('kind') == 'finished':
                self.finished_report = event['report']
            elif event.get('kind') in ('progress', 'tile', 'file', 'stage', 'warning'):
                phase = event.get('phase') or event.get('message') or (
                    'Predicting image tiles' if event['kind'] == 'tile' else self.stages[self.stage_index][1])
                self.progress.emit(dict(stage=self.stage_index+1, stages=len(self.stages), phase=phase,
                                        stage_name=self.stages[self.stage_index][1],
                                        current=event.get('current', 0), total=event.get('total', 0)))

    def _read_errors(self):
        if self.job is not None:
            text = bytes(self.job.readAllStandardError()).decode(errors='replace')
            self.errors = (self.errors + text)[-8000:]
            with (self.directory / f'{self.stage}.log').open('a', encoding='utf-8') as stream:
                stream.write(text)

    def _finished(self, code=0, *_):
        if self.job is None:
            return
        self._read()
        self._read_errors()
        process, self.job = self.job, None
        process.deleteLater()
        report = self.finished_report or {}
        if self.closing:
            return
        # If publication finished just before cancellation, preserve and display
        # that complete result instead of falsely reporting it as cancelled.
        published = self.stage == 'export' and code == 0 and report.get('status') == 'complete'
        if (self.stopping and not published) or report.get('status') == 'cancelled':
            self.state['status'] = 'cancelled'
            self._stop()
            self.cancelled.emit()
            return
        if code != 0 or report.get('status') != 'complete':
            self.state.update(status='failed', error=report.get('error') or self.errors or 'The processing worker failed to start or exited unexpectedly.')
            self._stop()
            self.failed.emit(self.state['error'])
            return
        try:
            self.state['stages'][self.stage] = report.get('directory', report.get('image'))
            def launch(stage, args):
                self._launch(next(i for i,s in enumerate(self.stages) if s[0] == stage), args)
            work, cancel = str(self.directory/'work'), str(self.cancel_file)
            if self.stage in ('cells','refine','consensus_cells','consensus','shape'):
                if len(report.get('files', [])) != 1 or report.get('failures'):
                    raise ValueError('Cell detection did not produce a complete scan result.')
                if self.stage == 'cells' and self.pipeline == 'consensus':
                    launch('refine', ['-m','fish_viewer.refinement','--mode','apply','--cells-report',
                        str(Path(report['directory'])/'report.json'),'--model',self.refinement_model,
                        '--output',work,'--device',self.device,'--cancel-file',cancel,'--redraw-proposals'])
                elif self.stage == 'refine':
                    self.state['refinement_summary'] = report['summary']
                    launch('consensus_cells', ['-m','fish_viewer.proposals','--mode','apply','--image',str(self.image),
                        '--model',self.consensus_model,'--boundary-model',self.refinement_model,
                        '--output',work,'--device',self.device,'--cancel-file',cancel] + self.cell_channel_arguments)
                elif self.stage == 'consensus_cells':
                    launch('consensus', ['-m','fish_viewer.proposals.consensus','--primary-report',
                        str(Path(self.state['stages']['refine'])/'report.json'),'--secondary-report',
                        str(Path(report['directory'])/'report.json'),'--output',work,'--cancel-file',cancel])
                elif self.stage != 'shape' and self.roundness is not None and self.roundness > 0:
                    if self.stage == 'consensus': self.state['consensus_summary'] = report['summary']
                    launch('shape', ['-m','fish_viewer.analysis.shape','--cells-report',str(Path(report['directory'])/'report.json'),
                        '--output',work,'--minimum',str(self.roundness),'--cancel-file',cancel])
                else:
                    launch('signals', ['-m','fish_viewer.signals','--mode','apply','--image',str(self.image),
                        '--cell-predictions',report['files'][0]['predictions'],'--model',self.signal_model,
                        '--output',work,'--device',self.device,'--cancel-file',cancel] + self.signal_channel_arguments)
            elif self.stage in ('signals','filter'):
                operation = 'filter' if self.stage == 'signals' else 'measure'
                launch(operation, ['-m','fish_viewer.workflow',operation,'--report',str(Path(report['directory'])/'report.json'),
                    '--output',work,'--cancel-file',cancel] + (['--max-pair-distance',str(self.maximum_pair_distance)] if operation == 'measure' else []))
            elif self.stage == 'measure':
                self.state['signal_report'] = str((Path(report['directory'])/'report.json').relative_to(self.directory))
                launch('export', ['-m','fish_viewer.workflow','export',*(['--scene',self.source_scene] if self.source_scene else []),
                    '--report',str(Path(report['directory'])/'report.json'),'--output',str(self.directory/'export'/(self.image.stem+'_measured.ims')),
                    '--cancel-file',cancel])
            else:
                self.state.update(status='complete',cells=report['cells'],signals=report['signals'],
                    annotation_run_id=report.get('annotation_run_id'),measurements=report['measurements'],
                    image_sha256=report['image_sha256'],image=str(Path(report['image']).relative_to(self.directory)),
                    scene=str(Path(report['scene']).relative_to(self.directory)) if report.get('scene') else None,
                    files=[str(Path(p).relative_to(self.directory)) for p in report['files']])
                self._stop()
                self.completed.emit(dict(self.state,report_path=str(self.directory/'analysis-result.json')))
        except Exception as exc:
            self.state.update(status='failed',error=str(exc))
            self._stop()
            self.failed.emit(str(exc))

    def _stop(self):
        self._persist()
        self.busy = False
        self.busy_changed.emit(False)

    def cancel(self):
        if self.busy:
            if self.stopping and self.job is not None:
                self.job.kill()
            else:
                self.stopping = True
                self.cancel_file.touch()

    def shutdown(self):
        if self.job is not None:
            self.closing = True
            self.cancel_file.touch()
            process = self.job
            if not process.waitForFinished(5000):
                process.kill()
                process.waitForFinished(1500)
            self.state['status'] = 'cancelled'
            self._persist()
            self.busy = False
