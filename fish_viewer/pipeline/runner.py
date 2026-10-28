"""Prediction archives and acquisition-safe data splits shared by the neural pipelines."""
from dataclasses import asdict,dataclass
import csv,hashlib,json,re
from pathlib import Path
import numpy as np
from ..models import LibraryEntry
from .config import PipelineConfig
from .segmentation import Prepared,Prediction,prediction_surfaces
@dataclass(eq=False)
class Case:
    entry: LibraryEntry
    prepared: Prepared
    references: list
    names: list[str]
    skipped: list[str]
    group: str
    directory: Path

def acquisition_group(entry):
    match = re.match(r'^(\d{6,8})[_ -]', entry.image.name)
    return str(entry.image.parent) + '/' + (match.group(1) if match else entry.image.stem)

def split_cases(cases, fraction=0.2):
    """Group by acquisition prefix and exact pixel fingerprint before splitting."""
    if not 0 < fraction < 1:
        raise ValueError('Evaluation fraction must be between 0 and 1.')
    # Union groups when identical pixels were saved under different filenames.
    parent = list(range(len(cases)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    groups, fingerprints = {}, {}
    for i, case in enumerate(cases):
        for key, mapping in ((case.group, groups), (case.prepared.fingerprint, fingerprints)):
            if key in mapping:
                parent[find(i)] = find(mapping[key])
            else:
                mapping[key] = i
    buckets = {}
    for i, case in enumerate(cases):
        buckets.setdefault(find(i), []).append(case)
    grouped = sorted(buckets.values(), key=lambda bucket: min(c.group for c in bucket))
    if len(grouped) < 2:
        raise ValueError('Tuning with independent evaluation needs at least two acquisition groups. '
            'Add images from another acquisition, or use Apply settings for a single group.')
    count = max(1, min(len(grouped)-1, int(round(len(grouped) * fraction))))
    # Stable chronological/name order, independent of annotation scores.
    return [c for group in grouped[:-count] for c in group], [c for group in grouped[-count:] for c in group]

def _json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)

def save_prediction(path, prediction, prepared, entry, config, metrics):
    metadata = dict(version=1, image=str(entry.image), image_name=entry.image.name,
        scene=str(entry.scene) if entry.scene else None, time=0, channel=prepared.channel,
        channel_scores=prepared.channel_scores, fingerprint=prepared.fingerprint, origin_xyz=prepared.origin,
        spacing_zyx=prepared.spacing, unit='µm', config=asdict(config), metrics=metrics, cells=prediction.cells)
    np.savez_compressed(path, labels=prediction.labels, metadata=np.array(json.dumps(metadata, allow_nan=False)))
    with Path(path).with_suffix('.cells.csv').open('w', newline='', encoding='utf-8') as stream:
        fields = ['label', 'volume_um3', 'diameter_um', 'sphericity', 'elongation', 'center_xyz']
        fields += sorted({key for cell in prediction.cells for key in cell}-set(fields))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(prediction.cells)
    return metadata

def read_prediction(path):
    with np.load(path, allow_pickle=False) as archive:
        labels = archive['labels'].copy()
        metadata = json.loads(str(archive['metadata']))
    if metadata.get('version') != 1 or labels.ndim != 3 or not labels.size or not np.issubdtype(labels.dtype, np.integer) or labels.min() < 0:
        raise ValueError('Invalid contour prediction archive.')
    spacing = tuple(metadata['spacing_zyx'])
    origin = tuple(metadata['origin_xyz'])
    if len(spacing) != 3 or len(origin) != 3 or not all(np.isfinite(spacing)) or not all(s > 0 for s in spacing) or not all(np.isfinite(origin)):
        raise ValueError('Invalid prediction calibration.')
    if labels.max() > 100_000:
        raise ValueError('Prediction archive has too many object labels.')
    prepared = Prepared(np.empty((0, 0, 0), np.float32), spacing, origin,
        metadata['channel'], metadata['channel_scores'], metadata['fingerprint'], {}, {})
    prediction = Prediction(labels, metadata.get('cells', []), {}, {}, {})
    return prediction_surfaces(prediction, prepared, metadata.get('time', 0)), metadata
