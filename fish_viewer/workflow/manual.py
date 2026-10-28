"""Original manual volumes and A-B distances for the combined Excel measurement export."""
from pathlib import Path

import numpy as np

from ..annotation_state import split_measurements
from ..models import LibraryEntry
from ..reader import entry_for_path, load_dataset
from ..pipeline.segmentation import checkpoint
from ..signals.config import SignalConfig
from ..signals.data import normalized, infer_channels
from .volumes import surface_volume
from .geometry import containing_surface
from .distances import SurfaceLocator


def manual_measurements(report, cancel=None):
    if 'manual_measurements' in report:
        return report['manual_measurements']
    source = Path(report.get('image', ''))
    if not source.is_file():
        return dict(cells=[], points=[])
    entry = entry_for_path(source)
    scene_path = report.get('scene')
    if scene_path and Path(scene_path).is_file():
        entry = LibraryEntry(source, Path(scene_path))
    dataset = load_dataset(entry, time=report.get('time', 0))
    if report.get('fingerprint'):
        from ..signals.engine import image_fingerprint
        if image_fingerprint(dataset) != report['fingerprint']:
            raise ValueError('Original scan pixels differ from the analysis.')
    time = report.get('time', 0)
    surfaces = [s for s in dataset.scene.surfaces if s.time == time and s.source == 'manual']
    split_measurements(dataset.scene)
    groups = [g for g in dataset.scene.measurements if g.source == 'manual']
    by_signal_id = {g.uid: g for g in groups}
    by_cell_id = {s.uid: s for s in surfaces}
    if not surfaces and not groups:
        return dict(cells=[], points=[])
    cells = []
    for surface in surfaces:
        checkpoint(cancel)
        cells.append(dict(cell_id=surface.uid, cell_name=surface.name, volume_um3=surface_volume(surface)))
    points = []
    for group in groups:
        linked = set()
        for left, right in group.edges:
            if group.times[left] != time or group.times[right] != time:
                continue
            if {group.labels[left].strip().upper(), group.labels[right].strip().upper()} != {'A', 'B'}:
                continue
            a, b = (left, right) if group.labels[left].strip().upper() == 'A' else (right, left)
            linked.add(a)
            points.append(dict(signal_id=group.uid, signal_name=group.name, xyz=group.points[a].tolist(),
                               distance_um=float(np.linalg.norm(group.points[a] - group.points[b]))))
        for index, (point, label, point_time) in enumerate(zip(group.points, group.labels, group.times)):
            if label.strip().upper() == 'A' and point_time == time and index not in linked:
                points.append(dict(signal_id=group.uid, signal_name=group.name, xyz=point.tolist(), distance_um=None))
    if points:
        spacing, origin = dataset.spacing[::-1], dataset.center_origin
        points = infer_channels([normalized(v) for v in dataset.volumes], points,
                                report.get('membrane_channel', 0), origin, spacing, SignalConfig())
        locators = {}
        for point in points:
            checkpoint(cancel)
            group = by_signal_id[point['signal_id']]
            if group.channel is not None:
                point['channel'] = group.channel
            assigned = by_cell_id.get(group.cell_uid)
            if assigned is None:
                index = containing_surface(surfaces, point['xyz'])
                assigned = surfaces[index] if index is not None else None
            point['cell_id'] = assigned.uid if assigned is not None else None
            point['cell_name'] = assigned.name if assigned is not None else None
            if point['distance_um'] is None and assigned is not None:
                if assigned.uid not in locators:
                    locators[assigned.uid] = SurfaceLocator(assigned)
                point['distance_um'] = float(locators[assigned.uid].closest(point['xyz'])[1])
            channel = point['channel']
            point['channel_name'] = dataset.metadata.channels[channel].name if channel is not None else None
    return dict(cells=cells, points=points, source_scene=str(dataset.scene.source) if dataset.scene.source else None)
