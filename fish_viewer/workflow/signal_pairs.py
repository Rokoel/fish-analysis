"""Closest-first, one-to-one signal pairs within each cell and time point."""

from collections import defaultdict
from itertools import combinations
from uuid import NAMESPACE_URL, uuid5

import numpy as np


DEFAULT_MAX_DISTANCE_UM = 1.0


def pair_signals(points, maximum=DEFAULT_MAX_DISTANCE_UM):
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError('The maximum signal-pair distance must be positive and finite.')
    groups = defaultdict(lambda: defaultdict(list))
    for point in points:
        if point.get('cell_id') is None or point.get('channel') is None:
            continue
        xyz = np.asarray(point['xyz'], float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError('Signal pairing requires finite XYZ coordinates in micrometres.')
        groups[(point['cell_id'], point.get('time', 0))][point['channel']].append(point)
    pairs = []
    for (cell, time), channels in groups.items():
        for a, b in combinations(sorted(channels), 2):
            candidates = []
            for left in channels[a]:
                for right in channels[b]:
                    distance = float(np.linalg.norm(np.asarray(left['xyz']) - right['xyz']))
                    if distance <= maximum:
                        candidates.append((distance, str(left['signal_id']), str(right['signal_id']), left, right))
            used_a, used_b = set(), set()
            for distance, _, _, left, right in sorted(candidates, key=lambda row: row[:3]):
                la, rb = left['signal_id'], right['signal_id']
                if la in used_a or rb in used_b:
                    continue
                used_a.add(la)
                used_b.add(rb)
                uid = str(uuid5(NAMESPACE_URL, f'fish-pair:{cell}:{time}:{a}:{la}:{b}:{rb}'))
                pairs.append(dict(pair_id=uid, cell_id=cell, cell_name=left.get('cell_name', cell),
                                  time=time, signal_id_a=la, signal_id_b=rb,
                                  signal_name_a=left.get('signal_name', la), signal_name_b=right.get('signal_name', rb),
                                  channel_a=a, channel_b=b, channel_name_a=left.get('channel_name'),
                                  channel_name_b=right.get('channel_name'), xyz_a=list(left['xyz']),
                                  xyz_b=list(right['xyz']), distance_um=distance))
    return pairs


def add_signal_pairs(report, maximum=None):
    maximum = (report.get('signal_pairing', {}).get('maximum_distance_um', DEFAULT_MAX_DISTANCE_UM)
               if maximum is None else maximum)
    report['signal_pairing'] = dict(method='closest_first_one_to_one_per_channel_pair',
                                    maximum_distance_um=maximum, unit='um')
    points = [dict(point, time=point.get('time', report.get('time', 0)))
              for point in [*report.get('manual_measurements', {}).get('points', []), *report.get('points', [])]]
    report['signal_pairs'] = pair_signals(points, maximum)
    return report


def update_scene_pairs(scene, metadata, maximum, run_id):
    """Rebuild derived lines, retaining existing pair identities and provenance."""
    import re
    from ..annotation_state import timestamp
    from ..models import Measurement
    import pyvista as pv
    from .distances import SurfaceLocator
    old = {m.uid: m for m in scene.measurements if m.kind == 'signal_pair'}
    base = [m for m in scene.measurements if m.kind != 'signal_pair']
    cells = {s.uid: (i, s) for i, s in enumerate(scene.surfaces)}
    points = []
    for m in base:
        a = next((i for i, label in enumerate(m.labels) if re.fullmatch(r'A\d*', label.strip(), re.I)),
                 0 if len(m.points) == 1 and m.labels[0].strip().upper() != 'B' else None)
        cell = cells.get(m.cell_uid)
        if a is None or cell is None or m.channel is None or m.times[a] != cell[1].time:
            continue
        points.append(dict(signal_id=m.uid, signal_name=m.name, xyz=m.points[a].tolist(), time=m.times[a],
                           cell_id=m.cell_uid, cell_name=cell[1].name, channel=m.channel,
                           channel_name=metadata.channels[m.channel].name))
    # Test all centers against a cell together; rebuilding the same containment
    # mesh for every individual signal makes live contour editing sluggish.
    enclosed = []
    by_cell = defaultdict(list)
    for point in points:
        by_cell[point['cell_id']].append(point)
    for uid, candidates in by_cell.items():
        if len({p['channel'] for p in candidates}) < 2:
            continue
        surface = cells[uid][1]
        locator = SurfaceLocator(surface)
        mesh = pv.wrap(locator.mesh).clean()
        if mesh.n_open_edges:
            continue
        xyz = np.asarray([p['xyz'] for p in candidates])
        bounded = np.all((xyz >= surface.vertices.min(0)) & (xyz <= surface.vertices.max(0)), axis=1)
        inside = pv.PolyData(xyz).select_enclosed_points(mesh, tolerance=1e-7, check_surface=False)['SelectedPoints'].astype(bool)
        for i, point in enumerate(candidates):
            if bounded[i] and (inside[i] or locator.closest(xyz[i])[1] <= 1e-7):
                enclosed.append(point)
    points = enclosed
    pairs = pair_signals(points, maximum) if metadata.unit in ('um', 'µm') else []
    derived = []
    for pair in pairs:
        m = old.get(pair['pair_id'])
        if m is None:
            m = Measurement(f"Signal pair · channels {pair['channel_a'] + 1} ↔ {pair['channel_b'] + 1}",
                            np.empty((2, 3)), [], [], uid=pair['pair_id'], source='program', run_id=run_id,
                            created_at=timestamp(), kind='signal_pair', all_times=False,
                            color=(0.2, 1., 1.))
        m.points = np.asarray([pair['xyz_a'], pair['xyz_b']], float)
        m.labels = [f"Channel {pair['channel_a'] + 1}: {pair['channel_name_a']}",
                    f"Channel {pair['channel_b'] + 1}: {pair['channel_name_b']}"]
        m.times, m.edges = [pair['time']] * 2, [(0, 1)]
        m.cell_uid = pair['cell_id']
        m.signal_uids = [pair['signal_id_a'], pair['signal_id_b']]
        derived.append(m)
        pair['source'] = 'Manual' if m.source == 'manual' else 'Program'
    scene.measurements = base + derived
    return pairs
