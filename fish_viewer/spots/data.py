"""Cell crops; unknown pixels never become training negatives."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy import ndimage as ndi

from ..pipeline.runner import _json
from ..signals.config import SignalConfig
from ..signals.data import prepare_supervision
from .config import CenterConfig


def crop_geometry(box, metadata, shape, config):
    spacing = np.asarray(metadata['spacing_zyx'])
    pad = np.ceil(config.margin_um / spacing).astype(int)
    low = np.maximum(0, [s.start for s in box] - pad)
    high = np.minimum(shape, [s.stop for s in box] + pad)
    origin = np.asarray(metadata['origin_xyz']) + (low * spacing)[::-1]
    size = np.maximum(2, np.ceil((high - low - 1) * spacing / config.spacing_um).astype(int) + 1)
    return origin, tuple(map(int, size))


def resample_crop(volume, spacing, origin, crop_origin, shape, step, order=1):
    spacing = np.asarray(spacing)
    scale = step / spacing
    offset = (np.asarray(crop_origin) - origin)[::-1] / spacing
    return ndi.affine_transform(volume, np.diag(scale), offset, output_shape=shape,
                                order=order, mode='constant', cval=0, prefilter=False)


def crop_inputs(images, membrane, labels, metadata, native_origin, native_spacing, cell, box, config):
    origin, shape = crop_geometry(box, metadata, labels.shape, config)
    ids = resample_crop(labels, metadata['spacing_zyx'], np.asarray(metadata['origin_xyz']), origin, shape, config.spacing_um, 0)
    target = ids == cell
    # Signal channel supplied by caller: no fluorescence identity is pooled.
    cell_image = resample_crop(images[membrane], native_spacing, native_origin, origin, shape, config.spacing_um)
    return origin, shape, target, cell_image


def gaussian_targets(points, shape, sigma):
    """Continuous-coordinate Gaussian maxima, without snapping the labels."""
    heatmap = np.zeros(shape, np.float32)
    for point in np.asarray(points).reshape(-1, 3):
        low = np.maximum(0, np.floor(point - 4 * sigma).astype(int))
        high = np.minimum(shape, np.ceil(point + 4 * sigma).astype(int) + 1)
        if np.any(high <= low):
            continue
        box = tuple(slice(int(a), int(b)) for a, b in zip(low, high))
        grid = np.indices(tuple(high - low), dtype=np.float32) + low[:, None, None, None]
        value = np.exp(-np.sum((grid - point[:, None, None, None]) ** 2, axis=0) / (2 * sigma ** 2))
        heatmap[box] = np.maximum(heatmap[box], value)
    return heatmap


def exclusion_mask(valid, markers, truth, channel, origin, config):
    """Ignore neighborhoods of every A that is not an eligible target here."""
    keys = {tuple(p['xyz']) for p in truth}
    for point in markers:
        if point['channel'] not in (None, channel) or tuple(point['xyz']) in keys:
            continue
        at = (np.asarray(point['xyz']) - origin)[::-1] / config.spacing_um
        radius = .9 / config.spacing_um
        low = np.maximum(0, np.floor(at - radius).astype(int))
        high = np.minimum(valid.shape, np.ceil(at + radius).astype(int) + 1)
        if np.any(high <= low):
            continue
        box = tuple(slice(int(a), int(b)) for a, b in zip(low, high))
        grid = np.indices(tuple(high - low), dtype=np.float32) + low[:, None, None, None]
        unsafe = np.sum((grid - at[:, None, None, None]) ** 2, axis=0) < radius ** 2
        valid[box][unsafe] = False
    return valid


def build(signal_audit, output, config=None, emit=print):
    config = (config or CenterConfig()).validate()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    previous = json.loads(Path(signal_audit).read_text())
    collection = json.loads(Path(previous['collection_audit']).read_text())
    if previous['status'] != 'complete' or collection['status'] != 'complete':
        raise ValueError('Completed split-safe cell/signal audits are required.')
    sources = {s['image']: s for s in collection['sources']}
    cases = {c['image']: c for c in collection['cases']}
    groups = {s: {c['group'] for c in previous['cases'] if c['split'] == s} for s in ('training', 'validation', 'test')}
    if any(groups[a] & groups[b] for a, b in [('training', 'validation'), ('training', 'test'), ('validation', 'test')]):
        raise ValueError('Acquisition leakage between folds.')
    report = dict(version=1, engine='signal-center-data', status='running', directory=str(output),
        config=asdict(config), parent_audit=str(Path(signal_audit).resolve()),
        parent_audit_sha256=hashlib.sha256(Path(signal_audit).read_bytes()).hexdigest(),
        cell_checkpoint_sha256=previous['cell_checkpoint_sha256'],
        boundary_checkpoint_sha256=previous['boundary_checkpoint_sha256'], cases=[], records=[])
    started = time.monotonic()
    for index, case in enumerate(previous['cases']):
        emit(dict(kind='progress', phase='Building direct signal-center volumes', current=index, total=len(previous['cases']), file=case['image']))
        prepared = prepare_supervision(cases[case['image']], sources, dict(predictions=case['cell_predictions']), SignalConfig())
        expected = {(p['cell_id'], p['channel'], tuple(p['xyz'])) for p in case['truth']}
        if expected != {(p['cell_id'], p['channel'], tuple(p['xyz'])) for p in prepared['truth']}:
            raise ValueError('Reference eligibility changed from the frozen comparison.')
        labels = prepared['labels']
        metadata = prepared['metadata']
        roi = prepared['roi']
        boxes = ndi.find_objects(labels)
        case_id = len(report['cases'])
        cell_ids = [int(c) for c in np.unique(labels[roi > 0]) if c > 0]
        # Do not omit an eligible manual center lying just beyond its manual mask.
        cell_ids = sorted(set(cell_ids) | {p['cell_id'] for p in case['truth']})
        report['cases'].append({k: case[k] for k in ('image', 'split', 'group', 'fingerprint', 'truth', 'markers', 'channels', 'membrane_channel', 'cell_predictions')})
        folder = output / f'{case_id + 1:04d}-{Path(case["image"]).stem}'
        folder.mkdir()
        for cell in cell_ids:
            origin, shape, mask, membrane_image = crop_inputs(prepared['images'], prepared['membrane'], labels, metadata,
                prepared['origin'], prepared['spacing'], cell, boxes[cell - 1], config)
            known = resample_crop(roi, metadata['spacing_zyx'], np.asarray(metadata['origin_xyz']), origin, shape, config.spacing_um, 0) > 0
            for channel, image in enumerate(prepared['images']):
                if channel == prepared['membrane']:
                    continue
                truth = [p for p in case['truth'] if p['cell_id'] == cell and p['channel'] == channel]
                points = np.asarray([(np.asarray(p['xyz']) - origin)[::-1] / config.spacing_um for p in truth], np.float32).reshape(-1, 3)
                heat = gaussian_targets(points, shape, config.sigma_um / config.spacing_um)
                valid = exclusion_mask(known.copy() & mask, prepared['points'], truth, channel, origin, config)
                # Positive labels are known even if a human contour narrowly misses A.
                valid |= (heat > .01) & mask
                if not valid.any():
                    continue
                signal = resample_crop(image, prepared['spacing'], prepared['origin'], origin, shape, config.spacing_um)
                inputs = np.stack([signal, membrane_image, mask], axis=0).astype(np.float16)
                path = folder / f'cell-{cell}-channel-{channel}.npz'
                np.savez_compressed(path, inputs=inputs, valid=valid.astype(np.uint8), points=points)
                report['records'].append(dict(case=case_id, split=case['split'], group=case['group'], cell_id=cell, channel=channel,
                    path=str(path), origin_xyz=origin.tolist(), shape=list(shape), points=len(points), valid_voxels=int(valid.sum())))
        report['progress'] = dict(current=index + 1, total=len(previous['cases']), phase='Building direct signal-center volumes')
        report['elapsed_s'] = time.monotonic() - started
        _json(output / 'index.json', report)
        del prepared
    report['summary'] = {s: dict(scans=sum(c['split'] == s for c in report['cases']),
        cell_channels=sum(r['split'] == s for r in report['records']),
        eligible_signals=sum(len(c['truth']) for c in report['cases'] if c['split'] == s),
        supervised_signals=sum(r['points'] for r in report['records'] if r['split'] == s)) for s in groups}
    for s, values in report['summary'].items():
        if values['eligible_signals'] != values['supervised_signals']:
            raise ValueError(f'Signal target loss during crop creation: {s}.')
    report.update(status='complete', supervision='Fixed predicted cells intersect fully marked unambiguous manual cells. '
        'Unknown regions have zero loss. Unsafe excluded-A neighborhoods ignored. Positive Gaussian labels are explicitly known. '
        'Inputs: signal channel, membrane channel, target-cell mask. No hand-designed signal candidates.')
    _json(output / 'index.json', report)
    emit(dict(kind='finished', summary=report['summary']))
    return report
