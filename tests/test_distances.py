import csv
import json
from pathlib import Path
from types import SimpleNamespace
import time
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from fish_viewer.models import Surface
from fish_viewer.reader import entry_for_path, load_dataset
from fish_viewer.workflow.distances import SurfaceLocator, measure_points, measure_result
from fish_viewer.workflow.export import export_result
from fish_viewer.workflow.results import filter_result
from test_workflow import signal_result  # Reuse the calibrated two-signal scan fixture.


def test_closest_point_is_on_triangle_interior_edge_or_vertex():
    surface = Surface('Predicted cell 1', np.array([[0., 0., 0.], [4., 0., 0.], [0., 4., 0.]]), np.array([[0, 1, 2]]))
    locator = SurfaceLocator(surface)
    for point, border, distance in [([1, 1, 3], [1, 1, 0], 3),
                                     ([3, 3, 0], [2, 2, 0], np.sqrt(2)),
                                     ([-1, -2, 2], [0, 0, 0], 3),
                                     ([1, 1, 0], [1, 1, 0], 0)]:
        found, length, triangle = locator.closest(point)
        np.testing.assert_allclose(found, border, atol=1e-12)
        assert length == pytest.approx(distance) and triangle == 0
    # Rigid transformations preserve the distance, and the B point follows them.
    rotated = surface.vertices[:, [2, 0, 1]] + [10, 20, 30]
    found, length, _ = SurfaceLocator(Surface('rotated', rotated, surface.triangles)).closest([13, 21, 31])
    np.testing.assert_allclose(found, [10, 21, 31])
    assert length == pytest.approx(3)


def test_distance_uses_assigned_cell_not_a_nearer_other_cell_and_physical_units():
    # Translated, anisotropically scaled coordinates are already physical XYZ.
    vertices = np.array([[0, 0, 0], [8, 0, 0], [0, 12, 0]], dtype=float) + [10, 20, 30]
    own = Surface('Predicted cell 7', vertices, np.array([[0, 1, 2]]))
    other = Surface('Predicted cell 9', vertices + [0, 0, 2.9], own.triangles)
    report = dict(points=[dict(signal_id=1, cell_id=7, xyz=[11, 21, 33])])
    metadata = SimpleNamespace(unit='µm', origin=(0, 0, 0), extent=(100, 100, 100))
    point = measure_points(report, [own, other], metadata)[0]
    np.testing.assert_allclose(point['border_xyz'], [11, 21, 30])
    assert point['distance_um'] == 3 and not point['border_at_image_edge']
    metadata.unit = 'voxel'
    with pytest.raises(ValueError, match='micrometres'):
        measure_points(report, [own], metadata)


def test_measured_export_preserves_native_ab_lines_counts_and_csv(signal_result, tmp_path):
    filtered = filter_result(signal_result, tmp_path / 'runs')
    source = Path(filtered['directory']) / 'report.json'
    before = source.read_bytes()
    measured = measure_result(source, tmp_path / 'runs')
    assert source.read_bytes() == before
    assert measured['measurement']['measured_signals'] == 2
    assert measured['counts'] == filtered['counts']
    from fish_viewer.signals.overlays import read_signals
    path = Path(measured['directory']) / 'report.json'
    overlays, _ = read_signals(path)
    assert len(overlays) == 2
    assert all(m.labels == ['A', 'B'] and m.edges == [(0, 1)] for m in overlays)
    target = tmp_path / 'export/measured.ims'
    output = export_result(path, target)
    assert output['signals'] == output['measurements'] == 2
    reopened = load_dataset(entry_for_path(target))
    assert len(reopened.scene.surfaces) == 2
    assert len(reopened.scene.measurements) == 2
    for m, point in zip(reopened.scene.measurements, measured['points']):
        assert m.labels == ['A', 'B'] and m.edges == [(0, 1)]
        np.testing.assert_allclose(m.points[0], point['xyz'])
        np.testing.assert_allclose(m.points[1], point['border_xyz'])
        assert m.length == pytest.approx(point['distance_um'])
    import h5py
    from openpyxl import load_workbook
    with h5py.File(target) as image:
        root = ET.fromstring(image['Scene/Data'][0].tobytes())
    assert all(m.find('LineLabels').get('Distance') == 'true' for m in root.iter('bpMeasurementPoints'))
    workbook = load_workbook(target.with_suffix('.xlsx'))
    rows = list(workbook['Measurements'].values)[1:]
    assert len(rows) == 2 and rows[0][-1] == pytest.approx(measured['points'][0]['distance_um'])
    assert rows[0][4] == 2 and rows[0][2] > 0
    workbook.close()
    assert not target.with_suffix('.signals.csv').exists()
    again = filter_result(path, tmp_path / 'again')
    assert 'measurement' not in again and all('border_xyz' not in p for p in again['points'])


@pytest.mark.parametrize('damage', ['border', 'length', 'archive'])
def test_export_rejects_stale_or_wrong_border_measurements(signal_result, tmp_path, damage):
    filtered = filter_result(signal_result, tmp_path / 'runs')
    report = measure_result(Path(filtered['directory']) / 'report.json', tmp_path / 'runs')
    if damage == 'border':
        report['points'][0]['border_xyz'] = report['points'][0]['xyz']
        report['points'][0]['distance_um'] = 0
    elif damage == 'length':
        report['points'][0]['distance_um'] = 50
    else:
        report['measurement']['contours_sha256'] = 'stale'
    path = Path(report['directory']) / 'report.json'
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        export_result(path, tmp_path / 'bad.ims')
    assert not (tmp_path / 'bad.ims').exists()


def test_empty_and_cancelled_measurement_runs(signal_result, tmp_path):
    import threading
    from fish_viewer.pipeline.segmentation import Cancelled
    report = json.loads(signal_result.read_text())
    report['points'] = []
    signal_result.write_text(json.dumps(report))
    filtered = filter_result(signal_result, tmp_path / 'runs')
    path = Path(filtered['directory']) / 'report.json'
    measured = measure_result(path, tmp_path / 'runs')
    assert measured['points'] == [] and measured['measurement']['measured_signals'] == 0
    result = export_result(Path(measured['directory']) / 'report.json', tmp_path / 'empty.ims')
    assert result['measurements'] == 0
    cancel = threading.Event(); cancel.set()
    with pytest.raises(Cancelled):
        measure_result(path, tmp_path / 'cancelled', cancel=cancel)
    assert not (tmp_path / 'cancelled').exists()


@pytest.mark.gui
def test_focused_app_opens_relocated_measurements_and_small_window(qapp, signal_result, tmp_path):
    from PySide6 import QtCore
    from fish_viewer.analysis.app import AnalysisWindow
    import shutil
    run = tmp_path / 'analysis'
    filtered = filter_result(signal_result, run / 'work')
    measured = measure_result(Path(filtered['directory']) / 'report.json', run / 'work')
    export = export_result(Path(measured['directory']) / 'report.json', run / 'export/result.ims')
    manifest = dict(engine='fish-analysis', status='complete', image='export/result.ims', scene=None,
                    signal_report=str((Path(measured['directory']) / 'report.json').relative_to(run)))
    (run / 'analysis-result.json').write_text(json.dumps(manifest))
    moved = tmp_path / 'relocated'
    shutil.move(run, moved)
    Path(measured['image']).unlink()  # Display must use the exported copy.
    settings = QtCore.QSettings(str(tmp_path / 'settings.ini'), QtCore.QSettings.Format.IniFormat)
    window = AnalysisWindow(off_screen=True, settings=settings)
    errors = []; window.error = errors.append
    try:
        window.resize(900, 550)
        window.show()
        window.load_result(moved / 'analysis-result.json')
        deadline = time.monotonic()+20
        while window.loading and time.monotonic()<deadline:
            qapp.processEvents(); time.sleep(.01)
        assert not window.loading and not errors
        assert window.distances.rowCount() == 2
        assert window.excel_button.isEnabled()
        assert float(window.counts.item(0, 4).text()) > 0
        assert sum(int(window.counts.item(i, 3).text()) for i in range(window.counts.rowCount())) == 2
        window.distances.selectRow(0)
        assert 'A → B:' in window.selection_info.text()
        assert window.slice_view.selected == ('measurement', 0)
        assert all(m.labels == ['A', 'B'] for m in window.dataset.scene.measurements)
        window.options_button.setChecked(True)
        for _ in range(5):qapp.processEvents()
        assert window.sidebar.verticalScrollBar().maximum() > 0
        window.sidebar.ensureWidgetVisible(window.run_button)
        qapp.processEvents()
        assert window.sidebar.widget().width() <= window.sidebar.viewport().width()
        assert not window.findChildren(type(window.run_button), 'Neural training')
    finally:
        window.close()
        qapp.processEvents()
