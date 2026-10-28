"""Decode learned center heatmaps; no image-based candidate finder."""
import numpy as np
from scipy import ndimage as ndi

from ..signals.evaluation import suppress


def local_centers(heatmap, mask, origin, config, cell, channel, floor=.01):
    # Only learned heatmap peaks are proposed. Physical NMS is shared with the
    # physical suppression. Centers can lie arbitrarily close to the boundary.
    peaks = (heatmap == ndi.maximum_filter(heatmap, size=3, mode='nearest')) & (heatmap >= floor) & mask
    if float(heatmap.max()) <= float(heatmap.min()):
        return []
    points = np.argwhere(peaks)
    rows = []
    for point in points:
        center = point.astype(float)
        # Quadratic log-heatmap localization in each physical grid axis.
        for axis in range(3):
            if 0 < point[axis] < heatmap.shape[axis] - 1:
                lower = point.copy(); lower[axis] -= 1
                upper = point.copy(); upper[axis] += 1
                a, b, c = np.log(np.maximum(1e-8, [heatmap[tuple(lower)], heatmap[tuple(point)], heatmap[tuple(upper)]]))
                denominator = a - 2 * b + c
                if denominator < -1e-8:
                    center[axis] += np.clip(.5 * (a - c) / denominator, -.5, .5)
        if not ndi.map_coordinates(mask.astype(np.uint8), center[:, None], order=0, mode='constant', cval=0)[0]:
            continue
        xyz = (np.asarray(origin) + (center * config.spacing_um)[::-1]).tolist()
        rows.append(dict(cell_id=int(cell), channel=int(channel), xyz=xyz, score=float(heatmap[tuple(point)])))
    return suppress(rows, [p['score'] for p in rows], floor, config.detection_separation_um)
