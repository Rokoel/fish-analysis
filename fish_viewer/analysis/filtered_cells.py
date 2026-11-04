"""Restore signal-filtered contours for review without adding export annotations."""
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np

from ..annotation_state import timestamp
from ..pipeline.runner import read_prediction
from ..signals.engine import image_fingerprint


def review_metadata(report, active_run=None):
    review = report.get('filtered_cell_review')
    if review is None:
        ids = report.get('signal_filter', {}).get('removed_cell_ids', [])
        archive = report.get('unfiltered_cell_predictions')
        if not ids or not archive:
            return None
        review = dict(archive=archive, cell_ids=ids,
                      run_id=active_run or report.get('annotation_run_id'))
        report['filtered_cell_review'] = review
    return review


def restore_filtered_cells(dataset, report, active_run=None):
    """Use only this run's verified pre-signal-filter archive, never rejected proposals."""
    review = review_metadata(report, active_run)
    if review is None:
        return
    path = Path(review['archive'])
    run_id = review.get('run_id') or active_run
    if not path.is_file() or not run_id or not review['cell_ids']:
        return  # Older/moved exports still open when internal worker files are absent.
    surfaces, metadata = read_prediction(path)
    if metadata['fingerprint'] != report.get('fingerprint') or metadata['fingerprint'] != image_fingerprint(dataset):
        raise ValueError('Removed-cell review contours belong to different image pixels.')
    spacing, origin = np.asarray(metadata['spacing_zyx']), np.asarray(metadata['origin_xyz'])
    with np.load(path, allow_pickle=False) as archive:
        shape = archive['labels'].shape
    if (not np.allclose(origin - spacing[::-1] / 2, dataset.metadata.origin, rtol=1e-6, atol=1e-6)
            or not np.allclose(dataset.metadata.extent[::-1], np.asarray(shape) * spacing, rtol=1e-6, atol=1e-6)):
        raise ValueError('Removed-cell review contours have different physical calibration.')
    present = {s.uid for s in dataset.scene.surfaces}
    ids = set(review['cell_ids'])
    created_at = next((s.created_at for s in dataset.scene.surfaces if s.run_id == run_id and s.created_at), None) or timestamp()
    for surface in surfaces:
        cell = int(surface.name.rsplit(' ', 1)[-1])
        if cell not in ids:
            continue
        uid = str(uuid5(NAMESPACE_URL, f'fish-analysis:{run_id}:signal-filtered:{cell}'))
        if uid in present:
            continue
        surface.uid, surface.run_id, surface.created_at = uid, run_id, created_at
        surface.source = 'program' if active_run == run_id else 'manual'
        surface.filtered_cell_id = cell
        surface.export_included = surface.visible = False
        dataset.scene.surfaces.append(surface)
        present.add(uid)
