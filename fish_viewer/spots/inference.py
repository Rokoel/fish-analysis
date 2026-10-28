"""Dense cell-crop inference with the existing signal/workflow report schema."""
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
from scipy import ndimage as ndi
import torch

from ..models import LibraryEntry
from ..reader import load_dataset
from ..run_paths import new_run_directory
from ..pipeline.runner import _json
from ..pipeline.segmentation import Cancelled, checkpoint
from ..signals.data import normalized, at_labels
from ..signals.engine import image_fingerprint
from ..signals.evaluation import suppress
from ..workflow.results import write_tables
from .config import CenterConfig, read_model
from .data import crop_inputs, resample_crop
from .detection import local_centers
from .network import create
from .training import predict_volume


def load(directory, device_name='auto'):
    manifest = read_model(directory)
    config = CenterConfig(**manifest['config'])
    device = torch.device('cuda' if device_name in ('gpu', 'auto', 'cuda') and torch.cuda.is_available() else 'cpu')
    if device_name in ('gpu', 'cuda') and device.type != 'cuda':
        raise ValueError('GPU requested but the PyTorch runtime has no CUDA device.')
    torch.set_num_threads(4)
    model = create(manifest['engine'], config).to(device)
    weights = torch.load(Path(directory) / manifest['checkpoint'], map_location=device, weights_only=True)
    model.load_state_dict(weights['state_dict']); model.eval()
    return model, manifest, config, device


def apply(image, scene, cell_predictions, model_directory, output, device_name='auto', emit=lambda e: None, cancel=None, signal_channels=None):
    directory = new_run_directory(output, 'signal-centers')
    report = dict(version=1, mode='apply', status='running', directory=str(directory),
        image=str(Path(image).resolve()), scene=str(scene) if scene else None,
        cell_predictions=str(Path(cell_predictions).resolve()), model_directory=str(Path(model_directory).resolve()),
        channel_index_base=0, unit='µm', time=0, points=[], counts=[])
    started = time.monotonic()
    def progress(phase, current, total):
        report['progress'] = dict(phase=phase, current=current, total=total)
        _json(directory / 'report.json', report); emit(dict(kind='progress', **report['progress']))
    emit(dict(kind='run_started', directory=str(directory)))
    try:
        model, manifest, config, device = load(model_directory, device_name)
        report.update(engine=manifest['engine'], threshold=manifest['threshold'], config=manifest['config'],
            checkpoint_sha256=hashlib.sha256((Path(model_directory) / manifest['checkpoint']).read_bytes()).hexdigest(),
            runtime=dict(torch=torch.__version__, device=str(device)), candidate_policy='Learned dense center heatmap peaks only; no image-based candidate finder.')
        with np.load(cell_predictions, allow_pickle=False) as f:
            labels = f['labels']; metadata = json.loads(str(f['metadata']))
        if 'rays' not in metadata.get('config', {}) and metadata.get('detector', {}).get('engine') != 'cell-center-box3d':
            raise ValueError('Neural-network cell contours are required.')
        progress('Loading signal channels', 0, 1)
        dataset = load_dataset(LibraryEntry(Path(image), Path(scene) if scene else None))
        if image_fingerprint(dataset) != metadata['fingerprint']:
            raise ValueError('Cell contours belong to different image pixels.')
        report['fingerprint'] = metadata['fingerprint']
        report['channels'] = [dict(name=c.name, color=list(c.color)) for c in dataset.metadata.channels]
        membrane = int(metadata['channel']); report['membrane_channel'] = membrane
        from ..channel_roles import signal_indices
        selected_channels=signal_indices(len(dataset.volumes),membrane,signal_channels)
        report['signal_channels']=selected_channels
        for field, name in [('detector', 'cell_detector'), ('refinement', 'boundary_refinement'), ('consensus', 'cell_consensus'), ('roundness_filter', 'roundness_filter')]:
            if metadata.get(field): report[name] = metadata[field]
        images = [normalized(v) for v in dataset.volumes]
        origin = np.asarray(dataset.center_origin); spacing = np.asarray(dataset.spacing[::-1])
        boxes = ndi.find_objects(labels); ids = [i for i, box in enumerate(boxes, 1) if box is not None]
        proposed = []
        for index, cell in enumerate(ids):
            checkpoint(cancel)
            progress('Predicting signal-center heatmaps inside cells', index, len(ids))
            crop_origin, shape, mask, membrane_image = crop_inputs(images, membrane, labels, metadata, origin, spacing, cell, boxes[cell - 1], config)
            for channel in selected_channels:
                volume=images[channel]
                checkpoint(cancel)
                signal = resample_crop(volume, spacing, origin, crop_origin, shape, config.spacing_um)
                inputs = np.stack([signal, membrane_image, mask]).astype(np.float32)
                heatmap, flow = predict_volume(model, inputs, device)
                rows = local_centers(heatmap, mask, crop_origin, config, cell, channel, floor=min(.01, manifest['threshold']))
                membership = at_labels(labels, [p['xyz'] for p in rows], metadata['origin_xyz'], metadata['spacing_zyx'])
                proposed.extend(p for p, owner in zip(rows, membership) if owner == cell)
        points = suppress(proposed, [p['score'] for p in proposed], manifest['threshold'], config.detection_separation_um)
        report['candidate_count'] = len(proposed)
        report['points'] = [dict(signal_id=i + 1, **p) for i, p in enumerate(points)]
        report['counts'] = [dict(cell_id=cell, channel=channel, channel_name=info['name'],
            count=sum(p['cell_id'] == cell and p['channel'] == channel for p in points))
            for cell in ids for channel, info in enumerate(report['channels']) if channel in selected_channels]
        write_tables(directory, report)
        report['status'] = 'complete'; progress('Complete', len(points), len(points))
    except Cancelled:
        report['status'] = 'cancelled'
    except Exception:
        report.update(status='failed', error=traceback.format_exc())
    report['elapsed_s'] = time.monotonic() - started
    _json(directory / 'report.json', report); emit(dict(kind='finished', report=report))
    return report
