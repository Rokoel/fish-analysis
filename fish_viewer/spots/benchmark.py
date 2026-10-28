"""Frozen-checkpoint evaluation and synthetic empty-signal controls."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from ..signals.evaluation import suppress, evaluate_points, summarize
from .inference import load
from .training import predict_records, predict_volume, write
from .detection import local_centers


def evaluate(index_path, model_directory, split, output, device='auto', emit=print):
    index=json.loads(Path(index_path).read_text())
    if index['status']!='complete':raise ValueError('Complete center corpus required.')
    model,manifest,config,runtime_device=load(model_directory,device)
    cases,scores=predict_records(model,index,split,config,runtime_device,emit)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for case,probabilities in zip(cases,scores):
        points=suppress(case['candidates'],probabilities,manifest['threshold'],config.detection_separation_um)
        rows.append(dict(image=case['image'],group=case['group'],points=points,
                         metrics=evaluate_points(points,case['truth'],config.match_radius_um)))
    result=dict(version=1,status='complete',engine=manifest['engine'],split=split,
        model_directory=str(Path(model_directory).resolve()),threshold=manifest['threshold'],
        checkpoint_sha256=hashlib.sha256((Path(model_directory)/manifest['checkpoint']).read_bytes()).hexdigest(),
        index=str(Path(index_path).resolve()),summary=summarize([r['metrics'] for r in rows]),scans=rows,
        policy='One-to-one physical matches within the same cell and channel; unknown regions unscored. Checkpoint and threshold fixed before evaluating held-out data.')
    write(output/'report.json',result)
    write(output/'raw-predictions.json',dict(cases=cases))
    emit(dict(kind='finished',report=result))
    return result


def controls(model_directory, output, device='auto'):
    model, manifest, config, runtime_device = load(model_directory, device)
    shape = (48,) * 3
    radius = np.linalg.norm((np.indices(shape) - 23.5) * config.spacing_um, axis=0)
    mask = radius < 2.5
    membrane = np.exp(-((radius - 2.5) / .18) ** 2).astype(np.float32)
    rng = np.random.default_rng(3141)
    images = dict(blank=np.zeros(shape, np.float32), constant=np.zeros(shape, np.float32),
        noise=np.clip(rng.normal(.1, .025, shape), 0, 1).astype(np.float32), membrane_only=membrane.copy())
    rows = []
    for name, image in images.items():
        heat, flow = predict_volume(model, np.stack([image, membrane, mask]).astype(np.float32), runtime_device)
        proposed = local_centers(heat, mask, np.zeros(3), config, 1, 1)
        accepted = suppress(proposed, [p['score'] for p in proposed], manifest['threshold'], config.detection_separation_um)
        rows.append(dict(name=name, learned_peaks=len(proposed), accepted=len(accepted), max_score=float(heat[mask].max())))
    result = dict(status='complete', engine=manifest['engine'], threshold=manifest['threshold'], controls=rows,
                  passed=all(r['accepted'] == 0 for r in rows), note='Synthetic controls with a cell mask and membrane channel; not field precision.')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    write(output / 'controls.json', result)
    return result
