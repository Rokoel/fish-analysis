from dataclasses import replace
import json
from pathlib import Path
import threading
import time

import h5py
import numpy as np
import pytest

from fish_viewer.models import Channel, Dataset, LibraryEntry, Metadata, Scene
from fish_viewer.pipeline.config import PipelineConfig
from fish_viewer.pipeline.evaluation import aggregate, evaluate, rasterize_reference
from fish_viewer.pipeline import runner
from fish_viewer.pipeline.segmentation import (
    Cancelled, Prediction, Prepared, mask_properties, prediction_surfaces, prepare,
)


@pytest.fixture
def synthetic(tmp_path):
    """Two hollow spheres and a misleadingly named punctum channel, in µm."""
    shape = (48, 80, 80)
    z, y, x = np.indices(shape) * 0.3
    r = np.sqrt((z-7.2)**2 + (y-7.2)**2 + (x-7.2)**2)
    s = np.sqrt((z-7.2)**2 + (y-16)**2 + (x-16)**2)
    rings = np.maximum(np.exp(-0.5*((r-2)/0.18)**2), np.exp(-0.5*((s-2.2)/0.18)**2))
    spots = np.maximum(np.exp(-r*r/0.12), np.exp(-s*s/0.12))
    channels = [Channel('Lamin 633', (1, 0, 0), excitation_nm=633), Channel('Unknown', (0, 0, 1))]
    metadata = Metadata(tmp_path / '101212_a.ims', shape, (10, 20, 30), (0.3, 0.3, 0.3), channels, [0], {0:shape})
    volumes = [(v*60000).astype(np.uint16) for v in (spots, rings)]
    truth = (r < 2).astype(np.uint32) + (s < 2.2).astype(np.uint32)*2
    prepared = Prepared(rings, (0.3,)*3, metadata.center_origin, 1, [], 'synthetic', {}, {})
    surfaces = prediction_surfaces(Prediction(truth, [], {}, {}, {}), prepared)
    for surface in surfaces:
        surface.source = 'manual'
    return Dataset(metadata, Scene(surfaces=surfaces), volumes), truth

def references(labels):
    box = tuple(slice(0, n) for n in labels.shape)
    return [(box, labels == i) for i in np.unique(labels) if i], ['Cell 1', 'Cell 2'][:int(labels.max())]

def write_ims(dataset, path):
    metadata = dataset.metadata
    with h5py.File(path, 'w') as f:
        group = f.create_group('DataSetInfo/Image')
        for axis in range(3):
            group.attrs[f'ExtMin{axis}'] = str(metadata.origin[axis]).encode()
            group.attrs[f'ExtMax{axis}'] = str(metadata.origin[axis] + metadata.extent[axis]).encode()
        group.attrs['Unit'] = b'um'
        for i, (channel, volume) in enumerate(zip(metadata.channels, dataset.volumes)):
            group = f.create_group(f'DataSetInfo/Channel {i}')
            group.attrs['Name'] = channel.name.encode()
            group.attrs['Color'] = ' '.join(map(str, channel.color)).encode()
            group.attrs['ColorRange'] = b'0 60000'
            group = f.create_group(f'DataSet/ResolutionLevel 0/TimePoint 0/Channel {i}')
            for axis, n in zip('ZYX', volume.shape):
                group.attrs[f'ImageSize{axis}'] = str(n).encode()
            group.create_dataset('Data', data=volume)

def test_shape_metrics_use_physical_units_and_reject_elongation():
    z, y, x = np.indices((25, 65, 65))
    ellipsoid = ((z-12)/5)**2 + ((y-32)/18)**2 + ((x-32)/18)**2 <= 1
    physical_sphere = mask_properties(ellipsoid, np.array([0.72, 0.2, 0.2]))
    voxel_ellipsoid = mask_properties(ellipsoid, np.array([0.2, 0.2, 0.2]))
    assert physical_sphere['elongation'] < 1.1
    assert voxel_ellipsoid['elongation'] > 3

def test_evaluation_identical_missing_unknown_and_merged_cells():
    truth = np.zeros((16, 16, 24), np.uint32)
    truth[4:10, 4:10, 3:9] = 1
    truth[4:10, 4:10, 13:19] = 2
    refs, names = references(truth)
    exact = evaluate(truth, refs, names, (1, 1, 1))
    assert exact['mean_dice'] == exact['mean_iou'] == exact['detection_rate'] == 1
    assert exact['boundary_rmse_um'] == exact['pair_mask_mse'] == 0
    extra = truth.copy()
    extra[12:14, 12:14, 12:14] = 3
    unknown = evaluate(extra, refs, names, (1, 1, 1))
    assert unknown['mean_dice'] == 1
    assert unknown['unscored_predictions'] == 1
    missing = evaluate(np.zeros_like(truth), refs, names, (1, 1, 1))
    assert missing['mean_dice'] == missing['detection_rate'] == 0
    assert missing['objective'] == 1
    assert missing['boundary_rmse_um'] is None
    merged = evaluate((truth > 0).astype(np.uint32), refs, names, (1, 1, 1))
    assert merged['matched_count'] == 1  # One predicted object cannot explain two cells.
    assert merged['mean_dice'] == pytest.approx(1/3)
    assert merged['detection_rate'] == 0.5
    empty = evaluate(extra, [], [], (1, 1, 1))
    assert empty['mean_dice'] is None and empty['unscored_predictions'] == 3
    assert aggregate([exact, missing, empty])['mean_dice'] == 0.5

def test_boundary_error_scales_in_micrometers():
    truth = np.zeros((12, 12, 12), np.uint32)
    truth[3:8, 3:8, 3:8] = 1
    refs, names = references(truth)
    shifted = np.roll(truth, 1, axis=0)
    small = evaluate(shifted, refs, names, (0.3,)*3)
    large = evaluate(shifted, refs, names, (0.6,)*3)
    assert 0 < small['boundary_rmse_um'] < 0.3
    assert large['boundary_rmse_um'] == pytest.approx(small['boundary_rmse_um']*2)
    assert small['pair_mask_mse'] == large['pair_mask_mse'] > 0

def test_mesh_rasterization_and_archive_preserve_calibration(synthetic, tmp_path):
    dataset, truth = synthetic
    prepared = prepare(dataset, PipelineConfig())
    refs, names, skipped = rasterize_reference(dataset.scene.surfaces, prepared)
    assert not skipped and len(refs) == 2
    assert evaluate(truth, refs, names, prepared.spacing)['mean_dice'] == 1
    prediction = Prediction(truth, [], {}, {}, {})
    path = tmp_path / 'contours.npz'
    runner.save_prediction(path, prediction, prepared, LibraryEntry(dataset.metadata.path), PipelineConfig(), {})
    surfaces, metadata = runner.read_prediction(path)
    assert all(s.source == 'prediction' for s in surfaces)
    roundtrip, names, skipped = rasterize_reference(surfaces, prepared)
    assert not skipped
    assert evaluate(prediction.labels, roundtrip, names, prepared.spacing)['mean_dice'] == 1
    assert metadata['spacing_zyx'] == list(prepared.spacing)
    assert path.with_suffix('.cells.csv').is_file()
    assert metadata['fingerprint'] == prepared.fingerprint

def test_split_keeps_acquisitions_and_duplicate_pixels_together(tmp_path):
    cases = []
    for group, fingerprint in [('101212', 'a'), ('101212', 'b'), ('131212', 'c'), ('141212', 'c'), ('151212', 'd')]:
        entry = LibraryEntry(tmp_path / f'{group}_{len(cases)}.ims')
        prepared = Prepared(np.zeros((2,)*3), (1,)*3, (0,)*3, 0, [], fingerprint, {}, {})
        cases.append(runner.Case(entry, prepared, [], [], [], runner.acquisition_group(entry), tmp_path))
    tuning, evaluation = runner.split_cases(cases, 0.5)
    assert set(c.group for c in tuning).isdisjoint(c.group for c in evaluation)
    assert set(c.prepared.fingerprint for c in tuning).isdisjoint(c.prepared.fingerprint for c in evaluation)
    with pytest.raises(ValueError, match='two acquisition groups'):
        runner.split_cases(cases[:2])
