from dataclasses import asdict
import json
from pathlib import Path
import shutil
import threading
import time

import numpy as np
import pytest

from fish_viewer.models import LibraryEntry
from fish_viewer.neural.config import NeuralConfig
from fish_viewer.reader import entry_for_path, load_dataset
from fish_viewer.signals.engine import image_fingerprint
from fish_viewer.workflow.export import export_result
from fish_viewer.workflow.results import filter_result


@pytest.fixture
def signal_result(ims_file, tmp_path):
    dataset = load_dataset(LibraryEntry(ims_file))
    labels = np.zeros(dataset.metadata.shape, np.uint32)
    labels[:, :, 0] = 1
    labels[:, :, 1] = 7
    labels[:, :, 3] = 9
    directory = tmp_path / 'detections'
    directory.mkdir()
    metadata = dict(version=1, image=str(ims_file), image_name=ims_file.name, time=0,
        origin_xyz=dataset.center_origin, spacing_zyx=dataset.spacing[::-1], channel=0, channel_scores={},
        fingerprint=image_fingerprint(dataset), config=asdict(NeuralConfig()),
        cells=[dict(label=i) for i in (1, 7, 9)], metrics=dict(mean_dice=.9))
    np.savez_compressed(directory / 'contours.npz', labels=labels, metadata=np.array(json.dumps(metadata)))
    points = [dict(signal_id=1, cell_id=7, channel=1, xyz=[13., 21., 31.], score=.99901, contrast=.5),
              dict(signal_id=2, cell_id=9, channel=1, xyz=[17., 25., 33.], score=.99902, contrast=.6)]
    report = dict(version=1, engine='signal-cnn3d', mode='apply', status='complete', time=0,
                  directory=str(directory), image=str(ims_file), scene=None, cell_predictions=str(directory / 'contours.npz'),
                  fingerprint=metadata['fingerprint'], channels=[dict(name=c.name, color=c.color) for c in dataset.metadata.channels],
                  channel_index_base=0, unit='µm', membrane_channel=0, points=points,
                  counts=[dict(cell_id=1, channel=1, channel_name='Fluor 1', count=99)])
    path = directory / 'report.json'
    path.write_text(json.dumps(report))
    return path


def test_filter_uses_real_centers_preserves_ids_and_sources(signal_result, tmp_path):
    before = signal_result.read_bytes()
    source = json.loads(before)
    archive_before = Path(source['cell_predictions']).read_bytes()
    report = filter_result(signal_result, tmp_path / 'runs')
    assert report['signal_filter']['kept_cell_ids'] == [7, 9]
    assert report['signal_filter']['removed_cell_ids'] == [1]
    assert [(row['cell_id'], row['count']) for row in report['counts']] == [(7, 1), (9, 1)]
    with np.load(report['cell_predictions']) as archive:
        assert set(np.unique(archive['labels'])) == {0, 7, 9}
        metadata = json.loads(str(archive['metadata']))
        assert metadata['cells'] == [dict(label=7), dict(label=9)]
        assert 'metrics' not in metadata
    assert signal_result.read_bytes() == before
    assert Path(source['cell_predictions']).read_bytes() == archive_before
    assert report['points'] == source['points']  # Includes centers on the image/cell boundary.


def test_provisional_recovery_masks_cannot_enter_final_workflow(signal_result,tmp_path):
    report=json.loads(signal_result.read_text())
    path=Path(report['cell_predictions'])
    with np.load(path) as data:
        labels=data['labels'];metadata=json.loads(str(data['metadata']))
    metadata['prediction_role']='candidate_only'
    np.savez_compressed(path,labels=labels,metadata=np.array(json.dumps(metadata)))
    with pytest.raises(ValueError,match='Provisional recovery masks'):
        filter_result(signal_result,tmp_path/'recovery-export')
    assert not (tmp_path/'recovery-export').exists()


@pytest.mark.parametrize('damage', ['fingerprint', 'image', 'calibration', 'outside', 'wrong_cell', 'duplicate', 'membrane'])
def test_rejects_mismatched_annotations_before_writing(signal_result, tmp_path, damage):
    report = json.loads(signal_result.read_text())
    if damage == 'fingerprint':report['fingerprint'] = 'bad'
    elif damage == 'image':
        import h5py
        with h5py.File(report['image'], 'r+') as image:
            image['DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data'][0, 0, 0] = 50
    elif damage == 'calibration':
        import h5py
        with h5py.File(report['image'], 'r+') as image:
            image['DataSetInfo/Image'].attrs['ExtMax0'] = np.frombuffer(b'26', dtype='S1')
    elif damage == 'outside':report['points'][0]['xyz'] = [15., 21., 31.]
    elif damage == 'wrong_cell':report['points'][0]['cell_id'] = 1
    elif damage == 'duplicate':report['points'][1]['signal_id'] = 1
    elif damage == 'membrane':report['points'][0]['channel'] = 0
    signal_result.write_text(json.dumps(report))
    with pytest.raises(ValueError):filter_result(signal_result, tmp_path / 'runs')
    assert not (tmp_path / 'runs').exists()


def test_imaris_export_is_portable_and_preserves_all_image_bytes(signal_result, tmp_path):
    from fish_viewer.pipeline.runner import read_prediction
    report = filter_result(signal_result, tmp_path / 'runs')
    original_image = Path(report['image'])
    original_bytes = original_image.read_bytes()
    source_surfaces, _ = read_prediction(report['cell_predictions'])
    target = tmp_path / 'export' / 'cells & signals.ims'
    output = export_result(Path(report['directory']) / 'report.json', target)
    assert output['cells'] == output['signals'] == 2
    assert original_bytes == original_image.read_bytes()
    assert {p.name for p in target.parent.iterdir()} == {target.name, target.with_suffix('.xlsx').name}
    exported = load_dataset(entry_for_path(target))
    assert len(exported.scene.surfaces) == 2
    assert len(exported.scene.measurements) == 2
    assert not exported.scene.warnings
    for original, copied in zip(source_surfaces, exported.scene.surfaces):
        np.testing.assert_allclose(original.vertices, copied.vertices)
        assert copied.source == 'manual'
        assert copied.run_id == output['annotation_run_id']
    for group, point in zip(exported.scene.measurements, report['points']):
        assert group.source == 'manual' and group.edges == []
        assert group.run_id == output['annotation_run_id']
        assert f'cell {point["cell_id"]}' in group.labels[0]
        assert str(point['score']) in group.labels[0]
        np.testing.assert_allclose(group.points[0], point['xyz'])
    # Every channel/time/resolution is untouched, not just the active view.
    for selected_time in (0, 1):
        for level in (0, 1):
            original = load_dataset(LibraryEntry(original_image), selected_time, level)
            reopened = load_dataset(entry_for_path(target), selected_time, level)
            for a, b in zip(original.volumes, reopened.volumes):np.testing.assert_array_equal(a, b)
    from fish_viewer.annotation_state import annotation_report
    assert sum(row['count'] for row in annotation_report(exported)['counts']) == 2
    relocated = tmp_path / 'moved'
    shutil.move(str(target.parent), relocated)
    shutil.rmtree(Path(report['directory']).parent)
    shutil.rmtree(signal_result.parent)
    original_image.unlink()
    for suffix in ('.ims',):
        entry = entry_for_path(relocated / target.with_suffix(suffix).name)
        standalone = load_dataset(entry)
        assert len(standalone.scene.surfaces) == 2
        assert sum(len(group.points) for group in standalone.scene.measurements) == 2


def test_empty_result_removes_every_contour_and_exports(signal_result, tmp_path):
    report = json.loads(signal_result.read_text())
    report['points'] = []
    signal_result.write_text(json.dumps(report))
    filtered = filter_result(signal_result, tmp_path / 'runs')
    assert filtered['signal_filter']['removed_cell_ids'] == [1, 7, 9]
    assert filtered['counts'] == []
    target = tmp_path / 'empty.ims'
    export_result(Path(filtered['directory']) / 'report.json', target)
    reopened = load_dataset(entry_for_path(target))
    assert reopened.scene.surfaces == reopened.scene.measurements == []


def test_export_refuses_overwrite_and_cleans_up_on_cancel(signal_result, tmp_path):
    from fish_viewer.pipeline.segmentation import Cancelled
    report = filter_result(signal_result, tmp_path / 'runs')
    path = Path(report['directory']) / 'report.json'
    existing = tmp_path / 'existing.xlsx'
    existing.write_text('Keep this file')
    with pytest.raises(FileExistsError):export_result(path, existing.with_suffix('.ims'))
    assert existing.read_text() == 'Keep this file'
    assert not existing.with_suffix('.ims').exists()
    cancellation = threading.Event()
    def cancel_during_copy(event):
        if event['phase'].startswith('Copying'):cancellation.set()
    with pytest.raises(Cancelled):
        export_result(path, tmp_path / 'cancelled.ims', cancel_during_copy, cancellation)
    assert not list(tmp_path.glob('cancelled.*'))
    assert not list(tmp_path.glob('.fish-export-*'))


@pytest.mark.gui
def test_filter_and_export_from_gui_workers(qapp, signal_result, tmp_path):
    from fish_viewer.app import MainWindow
    window = MainWindow(off_screen=True)
    errors = []
    window.error = errors.append
    def wait(predicate):
        deadline = time.monotonic() + 40
        while not predicate() and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(.01)
        assert predicate(), 'GUI worker timed out'
        assert not errors, errors
    try:
        window.open_signal_prediction(signal_result)
        wait(lambda: not window._pending_load)
        window.show_workflow()
        panel = window.workflow_panel
        window.workflow_dock.setFloating(True)
        window.workflow_dock.resize(800, 400)
        panel.error = errors.append
        panel.output.setText(str(tmp_path / 'runs'))
        panel.filter_button.click()
        wait(lambda: not panel.busy and not window._pending_load)
        assert panel.overall.value() == 3
        for _ in range(4):qapp.processEvents()
        assert panel.width() <= window.workflow_scroll.viewport().width()
        assert window.workflow_scroll.verticalScrollBar().maximum() > 0
        window.workflow_scroll.ensureWidgetVisible(panel.export_button)
        qapp.processEvents()
        position = panel.export_button.mapTo(window.workflow_scroll.viewport(), panel.export_button.rect().center())
        assert 0 <= position.y() < window.workflow_scroll.viewport().height()
        assert len(window.dataset.scene.surfaces) == 2
        assert all(surface.source == 'prediction' for surface in window.dataset.scene.surfaces)
        panel.export_current(target=tmp_path / 'from-gui.ims')
        wait(lambda: not panel.busy and not window._pending_load)
        assert panel.exported['signals'] == 2
        assert window.entry.image.name == 'from-gui.ims'
        assert len(window.dataset.scene.surfaces) == 2
        assert all(surface.visible for surface in window.dataset.scene.surfaces)
    finally:
        window.close()
        qapp.processEvents()
