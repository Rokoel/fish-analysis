"""Read saved signal predictions as viewer point annotations."""
import json
from pathlib import Path

import numpy as np

from ..models import Measurement


def read_signals(path,expected_fingerprint=None):
    report=json.loads(Path(path).read_text())
    if report.get('version')!=1 or report.get('engine') not in ('signal-center-unet3d', 'signal-cnn3d', 'signal-spotiflow3d') or report.get('mode')!='apply' or report.get('status')!='complete':
        raise ValueError('Choose a completed signal application report.')
    if expected_fingerprint is not None and report.get('fingerprint')!=expected_fingerprint:
        raise ValueError('Signals belong to different image pixels.')
    groups={}
    for point in report['points']:
        if len(point['xyz'])!=3 or not np.isfinite(point['xyz']).all() or not 0<=point['score']<=1:
            raise ValueError('Invalid signal coordinates or score.')
        groups.setdefault((point['cell_id'],point['channel']),[]).append(point)
    result=[]
    for (cell,channel),points in sorted(groups.items()):
        if not 0<=channel<len(report['channels']):raise ValueError('Invalid signal channel.')
        color=tuple(report['channels'][channel]['color'])
        if report.get('measurement'):
            for point in points:
                border = np.asarray(point.get('border_xyz'), dtype=float)
                distance = point.get('distance_um', float('nan'))
                if (border.shape != (3,) or not np.isfinite(border).all() or not np.isfinite(distance)
                        or distance < 0 or not np.isclose(np.linalg.norm(border-point['xyz']), distance, atol=1e-5)):
                    raise ValueError('Invalid saved A/B border measurement.')
                name = f'Signal {point["signal_id"]} · cell {cell} · channel {channel+1} · {distance:.3f} µm'
                result.append(Measurement(name, np.array([point['xyz'], border]), ['A', 'B'],
                    [report.get('time', 0)]*2, edges=[(0, 1)], color=color,
                    all_times=False, source='signal_prediction'))
            continue
        labels=[f'Signal {p["signal_id"]} · score {p["score"]:.3f}' for p in points]
        result.append(Measurement(f'NN signals · cell {cell} · channel {channel+1} · {len(points)} signals',
            np.array([p['xyz'] for p in points]),labels,[0]*len(points),color=color,all_times=False,source='signal_prediction'))
    return result,report
