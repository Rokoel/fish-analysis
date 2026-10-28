"""Masked center training, validation-only checkpoint/threshold selection."""
from collections import OrderedDict
from dataclasses import asdict
import csv
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..signals.config import SignalConfig
from ..signals.evaluation import select_threshold
from ..signals.data import at_labels
from ..neural.audit import read_references
from ..pipeline.segmentation import checkpoint
from .config import CenterConfig
from .data import gaussian_targets
from .detection import local_centers
from .network import create, loss


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


class Crops(Dataset):
    def __init__(self, records, config, engine, training=True):
        self.records = records; self.config = config; self.engine = engine; self.training = training
        self.cache = OrderedDict()
        # Every eligible A receives an explicitly centered sample each epoch.
        # Every cell/channel also receives a random known-region sample.
        self.samples = [(i, p) for i, record in enumerate(records) for p in range(record['points'])]
        self.samples += [(i, None) for i in range(len(records))]

    def __len__(self):
        return len(self.samples)

    def arrays(self, index):
        if index not in self.cache:
            with np.load(self.records[index]['path'], allow_pickle=False) as f:
                self.cache[index] = tuple(f[name] for name in ('inputs', 'valid', 'points'))
            if len(self.cache) > 8:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]

    def __getitem__(self, index):
        record, positive = self.samples[index]
        inputs, valid, points = self.arrays(record)
        size = self.config.patch_size
        if positive is not None:
            center = points[positive].copy()
            center += np.random.uniform(-size / 4, size / 4, 3) if self.training else 0
        else:
            coords = np.argwhere(valid > 0)
            center = coords[np.random.randint(len(coords))]
        low = np.floor(center - (size - 1) / 2).astype(int)
        x = np.zeros((3, size, size, size), np.float32)
        v = np.zeros((size,) * 3, np.float32)
        source_lo = np.maximum(0, low)
        source_hi = np.minimum(valid.shape, low + size)
        target_lo = source_lo - low
        src = tuple(slice(int(a), int(b)) for a, b in zip(source_lo, source_hi))
        dest = tuple(slice(int(a), int(a + b - c)) for a, b, c in zip(target_lo, source_hi, source_lo))
        x[(slice(None), *dest)] = inputs[(slice(None), *src)]
        v[dest] = valid[src]
        p = points - low
        # Include near-crop points for consistent Gaussian targets.
        p = p[np.all((p >= -4) & (p <= size + 3), axis=1)]
        if self.training:
            for axis in range(3):
                if np.random.random() < .5:
                    x = np.flip(x, axis=axis + 1); v = np.flip(v, axis=axis)
                    p[:, axis] = size - 1 - p[:, axis]
            k = np.random.randint(4)
            x = np.rot90(x, k, axes=(2, 3)); v = np.rot90(v, k, axes=(1, 2))
            for _ in range(k):
                p[:, 1], p[:, 2] = size - 1 - p[:, 2].copy(), p[:, 1].copy()
            x = x.copy()
            x[:2] = np.clip(x[:2] * np.random.uniform(.8, 1.2, (2, 1, 1, 1)) + np.random.normal(0, .005, x[:2].shape), 0, 1)
        target = gaussian_targets(p, (size,) * 3, self.config.sigma_um / self.config.spacing_um)
        result = dict(inputs=torch.from_numpy(x.copy()), valid=torch.from_numpy(v.copy()[None]),
                      heatmap=torch.from_numpy(target[None]))
        return result


def autocast_dtype(model):
    return torch.float16


def autocast_enabled(model, device):
    return device.type == 'cuda' and (autocast_dtype(model) != torch.bfloat16 or torch.cuda.is_bf16_supported())


def predict_volume(model, inputs, device):
    shape = inputs.shape[-3:]
    padded = tuple(int(np.ceil(s / 8) * 8) for s in shape)
    x = np.zeros((1, 3, *padded), np.float32)
    x[(0, slice(None), *tuple(slice(0, s) for s in shape))] = inputs
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=autocast_dtype(model), enabled=autocast_enabled(model, device)):
        out = model(torch.as_tensor(x, device=device))
    box = tuple(slice(0, s) for s in shape)
    heat = torch.sigmoid(out['heatmaps'][0][0, 0].float()).cpu().numpy()[box]
    if not np.isfinite(heat).all():
        raise ValueError('Non-finite signal-center inference output.')
    return heat, None


def scoring_regions(index, split):
    """Reconstruct the original benchmark ROI; training exclusions do not score."""
    parent = json.loads(Path(index['parent_audit']).read_text())
    collection = json.loads(Path(parent['collection_audit']).read_text())
    sources = {c['image']: c for c in collection['cases']}
    regions = {}
    for case in index['cases']:
        if case['split'] != split:
            continue
        with np.load(case['cell_predictions'], allow_pickle=False) as f:
            labels = f['labels']; metadata = json.loads(str(f['metadata']))
        marked = {m for p in case['markers'] for m in p['manual_cells']}
        ambiguous = {m for p in case['markers'] if p['channel'] is None for m in p['manual_cells']}
        refs, _ = read_references(Path(sources[case['image']]['directory']) / 'references.npz')
        roi = np.zeros(labels.shape, np.uint8)
        for ref in marked - ambiguous:
            box, mask = refs[ref]; roi[box][mask] = 1
        for ref in ambiguous:
            box, mask = refs[ref]; roi[box][mask] = 0
        roi *= labels > 0
        regions[case['image']] = (roi, labels, metadata)
    return regions


def predict_records(model, index, split, config, device, emit=lambda e: None, regions=None):
    cases = [dict(image=c['image'], group=c['group'], truth=c['truth'], candidates=[]) for c in index['cases'] if c['split'] == split]
    lookup = {c['image']: c for c in cases}
    records = [r for r in index['records'] if r['split'] == split]
    regions = regions if regions is not None else scoring_regions(index, split)
    model.eval()
    for i, record in enumerate(records):
        if i % 50 == 0:
            emit(dict(kind='progress', phase=f'Predicting {split} center heatmaps', current=i, total=len(records)))
        with np.load(record['path'], allow_pickle=False) as f:
            inputs = f['inputs'].astype(np.float32)
        heat, flow = predict_volume(model, inputs, device)
        rows = local_centers(heat, inputs[2] > .5, record['origin_xyz'], config, record['cell_id'], record['channel'])
        image = index['cases'][record['case']]['image']
        roi, labels, metadata = regions[image]
        known = at_labels(roi, [p['xyz'] for p in rows], metadata['origin_xyz'], metadata['spacing_zyx'])
        owners = at_labels(labels, [p['xyz'] for p in rows], metadata['origin_xyz'], metadata['spacing_zyx'])
        lookup[image]['candidates'].extend(p for p, valid, owner in zip(rows, known, owners) if valid and owner == record['cell_id'])
    return cases, [np.asarray([p['score'] for p in c['candidates']]) for c in cases]


def train(index_path, output, engine, config=None, device_name='auto', emit=print, initial_model=None, cancel=None):
    checkpoint(cancel)
    config = (config or CenterConfig()).validate()
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    index = json.loads(Path(index_path).read_text())
    if index['status'] != 'complete':
        raise ValueError('Training requires a complete center corpus.')
    if index['config']['spacing_um'] != config.spacing_um or index['config']['sigma_um'] != config.sigma_um:
        raise ValueError('Training sampling differs from the frozen crop corpus.')
    torch.set_num_threads(4)
    torch.manual_seed(config.seed); random.seed(config.seed); np.random.seed(config.seed)
    device = torch.device('cuda' if device_name=='auto' and torch.cuda.is_available() else 'cpu' if device_name=='auto' else device_name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable.')
    previous = None
    if initial_model:
        from .config import read_model
        previous = read_model(initial_model)
        if Path(previous['data_index']).resolve() != Path(index_path).resolve() or any(previous[k] != index[k] for k in ('cell_checkpoint_sha256', 'boundary_checkpoint_sha256')):
            raise ValueError('Recovery requires the original corpus and frozen cell models.')
        if previous['engine'] != engine or any(previous['config'][k] != asdict(config)[k] for k in ('spacing_um', 'sigma_um', 'filters', 'patch_size')):
            raise ValueError('Recovered checkpoint architecture/sampling must match.')
    model = create(engine, config).to(device)
    if previous:
        weights = torch.load(Path(initial_model) / previous['checkpoint'], map_location=device, weights_only=True)
        model.load_state_dict(weights['state_dict'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == 'cuda' and autocast_dtype(model) == torch.float16)
    records = [r for r in index['records'] if r['split'] == 'training']
    dataset = Crops(records, config, engine)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=2, pin_memory=device.type == 'cuda', persistent_workers=True)
    scoring = SignalConfig(match_radius_um=config.match_radius_um, detection_separation_um=config.detection_separation_um,
                           target_precision=config.target_precision)
    model_dir = output / 'model'; model_dir.mkdir()
    validation_regions = scoring_regions(index, 'validation')
    report = dict(version=1, engine=engine, status='running', mode='train', directory=str(output), model_directory=str(model_dir),
        config=asdict(config), index=str(Path(index_path).resolve()), data_summary=index['summary'], epochs=[],
        runtime=dict(torch=torch.__version__, device=str(device), autocast=str(autocast_dtype(model)) if autocast_enabled(model, device) else 'torch.float32', gpu=torch.cuda.get_device_name() if device.type == 'cuda' else None),
        parameters=sum(p.numel() for p in model.parameters()), samples_per_epoch=len(dataset))
    best = (-1., -1., -1.); last_best = 0; started = time.monotonic()
    start_epoch = 0
    if previous:
        start_epoch = int(previous['best_epoch'])
        if config.epochs <= start_epoch:
            raise ValueError('Maximum epoch must exceed the recovered checkpoint epoch.')
        parent_report = json.loads((Path(initial_model).parent / 'report.json').read_text())
        report['epochs'] = [r for r in parent_report['epochs'] if r['epoch'] <= start_epoch]
        report['resume'] = dict(initial_model=str(Path(initial_model).resolve()), initial_epoch=start_epoch,
            parent_report=str((Path(initial_model).parent / 'report.json').resolve()),
            optimizer_restarted=True, reason='Resume selected checkpoint with a fresh optimizer.',
            parent_config=previous['config'], parent_runtime=parent_report['runtime'])
        cases, probabilities = predict_records(model, index, 'validation', config, device, emit, validation_regions)
        selection = select_threshold(cases, probabilities, scoring)
        metrics = selection['selected']['metrics']
        best = (float(selection['target_met']), metrics['recall'] or 0., metrics['precision'] or 0.) if selection['target_met'] else (0., metrics['f05'], metrics['precision'] or 0.)
        last_best = start_epoch; report['best_epoch'] = start_epoch
        torch.save(dict(state_dict=model.state_dict()), model_dir / 'best.pt')
        previous.update(config=asdict(config), threshold=selection['selected']['threshold'], validation=metrics,
                        selection=selection, training_resume=report['resume'])
        write(model_dir / 'signal-model.json', previous)
        write(output / 'best-validation.json', dict(cases=cases, threshold=previous['threshold'], metrics=metrics))
    write(output / 'report.json', report)
    for epoch in range(start_epoch + 1, config.epochs + 1):
        checkpoint(cancel)
        model.train(); totals = np.zeros(3)
        for batch_index, batch in enumerate(loader):
            checkpoint(cancel)
            x, target, valid = (batch[n].to(device, non_blocking=True) for n in ('inputs', 'heatmap', 'valid'))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype(model), enabled=autocast_enabled(model, device)):
                value, heat, auxiliary = loss(model(x), target, valid, config)
            if not torch.isfinite(value):
                raise ValueError('Non-finite direct center loss.')
            scaler.scale(value).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            scaler.step(optimizer); scaler.update()
            totals += [value.item(), heat.item(), auxiliary.item()]
            if batch_index % 25 == 0:
                report['progress'] = dict(phase='Training direct 3D signal centers', epoch=epoch, epochs=config.epochs,
                    current=batch_index, total=len(loader), loss=value.item())
                write(output / 'report.json', report); emit(dict(kind='progress', **report['progress']))
        cases, probabilities = predict_records(model, index, 'validation', config, device, emit, validation_regions)
        selection = select_threshold(cases, probabilities, scoring)
        metrics = selection['selected']['metrics']
        key = (float(selection['target_met']), metrics['recall'] or 0., metrics['precision'] or 0.) if selection['target_met'] else (0., metrics['f05'], metrics['precision'] or 0.)
        improved = key > best
        if improved:
            best = key; last_best = epoch
            torch.save(dict(state_dict=model.state_dict()), model_dir / 'best.pt')
            manifest = dict(version=1, engine=engine, config=asdict(config), checkpoint='best.pt', threshold=selection['selected']['threshold'],
                best_epoch=epoch, validation=metrics, selection=selection, data_index=str(Path(index_path).resolve()),
                inputs=['signal', 'membrane', 'target_cell_mask'], training_resume=report.get('resume'), score_note='Learned heatmap peak score; validation-calibrated, not a probability of correctness.',
                cell_checkpoint_sha256=index['cell_checkpoint_sha256'], boundary_checkpoint_sha256=index['boundary_checkpoint_sha256'])
            write(model_dir / 'signal-model.json', manifest)
            write(output / 'best-validation.json', dict(cases=cases, threshold=manifest['threshold'], metrics=metrics))
        row = dict(epoch=epoch, loss=float(totals[0] / len(loader)), heatmap_loss=float(totals[1] / len(loader)),
            precision=metrics['precision'], recall=metrics['recall'], f05=metrics['f05'],
            threshold=selection['selected']['threshold'], saved=improved, elapsed_s=time.monotonic() - started)
        report['epochs'].append(row); report['best_epoch'] = last_best; report['elapsed_s'] = row['elapsed_s']
        write(output / 'report.json', report); emit(dict(kind='epoch', **row))
        with (output / 'epochs.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row)); writer.writeheader(); writer.writerows(report['epochs'])
        graphs(output, report['epochs'])
        if epoch - last_best >= config.patience:
            break
    report.update(status='complete', elapsed_s=time.monotonic() - started,
                  validation=json.loads((model_dir / 'signal-model.json').read_text())['validation'],
                  progress=dict(phase='Training complete', current=len(report['epochs']), total=len(report['epochs'])))
    write(output / 'report.json', report); emit(dict(kind='finished', report=report))
    return report


def graphs(output, rows):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(11, 4), layout='constrained'); FigureCanvasAgg(figure)
    a, b = figure.subplots(1, 2); epochs = [r['epoch'] for r in rows]
    a.plot(epochs, [r['loss'] for r in rows], label='total loss'); a.plot(epochs, [r['heatmap_loss'] for r in rows], label='center loss')
    b.plot(epochs, [r['precision'] or 0 for r in rows], label='validation precision'); b.plot(epochs, [r['recall'] or 0 for r in rows], label='validation recall')
    for ax in (a, b): ax.set_xlabel('Epoch'); ax.grid(alpha=.2); ax.legend()
    b.set_ylim(0, 1.02); a.set_title('Masked training loss'); b.set_title('Validation-calibrated detections')
    figure.savefig(Path(output) / 'training-progress.png', dpi=150)
