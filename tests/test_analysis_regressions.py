"""Native IMS geometry, annotation identity and current-run display regressions."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pytest

from test_annotation_editor import cell, document
from test_workflow import signal_result  # noqa: F401
from fish_viewer.analysis.annotations import AnnotationDocument
from fish_viewer.annotation_state import activate_run
from fish_viewer.models import LibraryEntry, Measurement
from fish_viewer.reader import ImarisReader, entry_for_path, load_dataset
from fish_viewer.scenes import parse_scene
from fish_viewer.workflow.edit_export import export_document, validate_roundtrip


def test_native_xyz_normal_records_and_object_local_counts(ims_file):
    first = cell()
    second = cell()
    second.vertices += [20, 0, 0]
    vertices = np.vstack([first.vertices, second.vertices])
    normals = np.tile([0., 0., 1.], (len(vertices), 1))
    with h5py.File(ims_file, 'r+') as image:
        group = image.create_group('Scene/Content/Surfaces0')
        group.attrs['Name'] = 'Native cells'
        group.create_dataset('Vertices', data=np.column_stack([vertices, normals]).astype(np.float32))
        group.create_dataset('Triangles', data=np.vstack([first.triangles, second.triangles]).astype(np.uint64))
        group.create_dataset('TimeNVerticesNTriangles', data=[[0, 8, 12], [1, 8, 12]])
    scene = ImarisReader(ims_file).embedded_scene()
    assert not scene.warnings
    assert len(scene.surfaces) == 2
    for expected, actual in zip((first, second), scene.surfaces):
        np.testing.assert_array_equal(expected.vertices, actual.vertices)
        np.testing.assert_array_equal(expected.triangles, actual.triangles)
    assert [s.time for s in scene.surfaces] == [0, 1]


def test_pairwise_xml_never_connects_unrelated_ab_pairs(qapp, ims_file, tmp_path):
    scene = parse_scene('''<root><bpMeasurementPoints>
      <bpSurfaceComponent><name>Program · custom manual name</name></bpSurfaceComponent>
      <Point Name="A" Position="14 23 31"/><Point Name="B" Position="14 23 30"/>
      <Point Name="A" Position="16 24 32"/><Point Name="B" Position="16 24 34"/>
      <LineMode>Pairwise</LineMode></bpMeasurementPoints></root>''')
    assert scene.measurements[0].edges == [(0, 1), (2, 3)]
    assert scene.measurements[0].source == 'manual'
    data = load_dataset(entry_for_path(ims_file))
    data.scene = scene
    doc = AnnotationDocument(data)
    assert len(scene.measurements) == 2
    before = deepcopy(scene)
    exported = export_document(doc, tmp_path / 'pairs.ims')
    after = load_dataset(entry_for_path(exported)).scene
    validate_roundtrip(before, after)
    assert [m.edges for m in after.measurements] == [[(0, 1)], [(0, 1)]]


def test_duplicate_names_use_distinct_ids_and_preserve_imported_endpoints(qapp, ims_file, tmp_path):
    doc = document(ims_file)
    doc.dataset.scene.surfaces[0].name = 'Predicted cell 1'
    first = doc.dataset.scene.surfaces[0]
    second = cell()
    second.name = first.name
    second.vertices += [20, 0, 0]
    doc.dataset.scene.surfaces.append(second)
    doc.dataset.scene.measurements[0].name = 'Signal 1; cell 1'
    # Deliberately preserve a manual B that is not the nearest surface point.
    doc.dataset.scene.measurements[0].points[1] = [13.8, 23, 30]
    other = Measurement('Signal 1; cell 1', np.array([[34, 23, 31], [34.2, 23, 30.5]]),
                        ['A', 'B'], [0, 0], [(0, 1)])
    doc.dataset.scene.measurements.append(other)
    doc = AnnotationDocument(doc.dataset)
    assert doc.bindings == {0: 0, 1: 1}
    before = deepcopy(doc.dataset.scene)
    exported = export_document(doc, tmp_path / 'duplicate.ims')
    reopened = AnnotationDocument(load_dataset(entry_for_path(exported)))
    validate_roundtrip(before, reopened.dataset.scene)
    assert reopened.bindings == {0: 0, 1: 1}
    rows = reopened.report()['manual_measurements']
    assert len({c['cell_id'] for c in rows['cells']}) == 2
    assert len({m['signal_id'] for m in rows['points']}) == 2
    reopened.delete('surface', 0)
    assert [m.uid for m in reopened.dataset.scene.measurements] == [other.uid]
    assert reopened.bindings == {0: 0}
    reopened.undo.undo()
    validate_roundtrip(before, reopened.dataset.scene)
    assert reopened.bindings == {0: 0, 1: 1}


@pytest.mark.parametrize('keep_sidecar', [True, False])
def test_repeated_pipeline_export_import_has_separate_runs(qapp, signal_result, tmp_path, keep_sidecar):
    from fish_viewer.workflow.results import filter_result
    from fish_viewer.workflow.distances import measure_result
    from fish_viewer.workflow.export import export_result

    first_report = filter_result(signal_result, tmp_path / 'work')
    first_report = measure_result(Path(first_report['directory']) / 'report.json', tmp_path / 'work')
    first = export_result(Path(first_report['directory']) / 'report.json', tmp_path / 'first.ims')
    if not keep_sidecar:
        with h5py.File(first['image'], 'r+') as image:
            del image['FISHAnalysis']
    imported = AnnotationDocument(load_dataset(entry_for_path(first['image'])))
    assert all(m.source == 'manual' for m in imported.dataset.scene.measurements)
    before = deepcopy(imported.dataset.scene)

    # Simulate the next detector run on the same pixels with reused local labels.
    report = json.loads(signal_result.read_text())
    report['image'] = first['image']
    next_input = tmp_path / 'next-input.json'
    next_input.write_text(json.dumps(report))
    second_report = filter_result(next_input, tmp_path / 'work2')
    second_report = measure_result(Path(second_report['directory']) / 'report.json', tmp_path / 'work2')
    second = export_result(Path(second_report['directory']) / 'report.json', tmp_path / 'second.ims')
    assert first['annotation_run_id'] != second['annotation_run_id']
    result = load_dataset(entry_for_path(second['image']))
    assert len(result.scene.surfaces) == len(result.scene.measurements) == 4
    assert all(m.source == 'manual' for m in result.scene.measurements)
    assert len({o.uid for objects in (result.scene.surfaces, result.scene.measurements) for o in objects}) == 8
    validate_roundtrip(before, type(before)(surfaces=result.scene.surfaces[:2], measurements=result.scene.measurements[:2]))
    activate_run(result.scene, second['annotation_run_id'])
    doc = AnnotationDocument(result)
    assert [m.source for m in result.scene.measurements] == ['manual', 'manual', 'program', 'program']
    assert doc.bindings == {0: 0, 1: 1, 2: 2, 3: 3}
    for i, m in enumerate(result.scene.measurements):
        assert m.cell_uid == result.scene.surfaces[doc.bindings[i]].uid
        if m.created_at:
            assert m.created_at in m.name
    # Exact imported names/timestamps survive another export, including when
    # their original sidecar was omitted during import.
    assert [m.name for m in result.scene.measurements[:2]] == [m.name for m in before.measurements]
    with h5py.File(second['image']) as image:
        root = ET.fromstring(image['Scene/Data'][0].tobytes())
    ids = [c.findtext('id') for c in root.iter('bpSurfaceComponent')]
    assert len(ids) == len(set(ids))
    # Overlapping old/new contours must not make deletion cross their explicit
    # signal assignments. Remove only the deleted cell's own signals.
    previous_signals = [m.uid for m in before.measurements]
    doc.delete('surface', 2)
    assert [m.uid for m in result.scene.measurements[:2]] == previous_signals
    assert len(result.scene.measurements) == 3
    assert doc.bindings == {0: 0, 1: 1, 2: 2}
    doc.undo.undo()
    assert len(doc.dataset.scene.measurements) == 4
    report['image'] = second['image']
    next_input.write_text(json.dumps(report))
    third_report = filter_result(next_input, tmp_path / 'work3')
    third_report = measure_result(Path(third_report['directory']) / 'report.json', tmp_path / 'work3')
    third = export_result(Path(third_report['directory']) / 'report.json', tmp_path / 'third.ims')
    with h5py.File(third['image']) as image:
        saved = json.loads(image['FISHAnalysis/Annotations'][()].tobytes())
    assert len(saved['manual_measurements']['cells']) == 4
    assert len(saved['manual_measurements']['points']) == 4
    assert all(' | ' not in p['cell_id'] for p in saved['manual_measurements']['points'])
    if keep_sidecar:
        assert all(p['channel'] == 1 for p in saved['manual_measurements']['points'])


@pytest.mark.gui
@pytest.mark.parametrize("volume", [False, True])
def test_interpolation_default_toggle_and_duplicate_row_selection(qapp, ims_file, tmp_path, volume):
    from PySide6 import QtCore
    from fish_viewer.analysis.app import AnalysisWindow

    if volume and qapp.platformName() == 'offscreen':
        pytest.skip('The native VTK widget needs an X/GL context; run this check with Qt xcb')
    settings = QtCore.QSettings(str(tmp_path / 'display.ini'), QtCore.QSettings.Format.IniFormat)
    window = AnalysisWindow(off_screen=True, settings=settings)
    try:
        doc = document(ims_file)
        duplicate = Measurement(doc.dataset.scene.measurements[0].name,
                                np.array([[14, 23, 32], [14, 23, 34]], float), ['A', 'B'], [0, 0], [(0, 1)])
        doc.dataset.scene.measurements.append(duplicate)
        window.set_dataset(doc.dataset)
        raw = [v.copy() for v in window.dataset.volumes]
        assert window.smooth_display.isChecked()
        assert window.slice_view._image_item.transformationMode() == QtCore.Qt.TransformationMode.SmoothTransformation
        if volume:
            window.change_mode(1)
            assert window.volume_view._volume.GetProperty().GetInterpolationTypeAsString() == 'Linear'
        window.smooth_display.setChecked(False)
        assert window.slice_view._image_item.transformationMode() == QtCore.Qt.TransformationMode.FastTransformation
        if volume:
            assert window.volume_view._volume.GetProperty().GetInterpolationTypeAsString() == 'Nearest Neighbor'
        window.smooth_display.setChecked(True)
        if volume:
            assert window.volume_view._volume.GetProperty().GetInterpolationTypeAsString() == 'Linear'
        window.distances.selectRow(1)
        assert window.editor.selection == ('measurement', 1)
        assert window.slice_view.selected == ('measurement', 1)
        if volume:
            assert window.volume_view.selected == ('measurement', 1)
        for before, after in zip(raw, window.dataset.volumes):
            np.testing.assert_array_equal(before, after)
    finally:
        window.close()


def test_native_ims_and_empty_companion_keep_separate_editor_documents(qapp, ims_file, tmp_path):
    from PySide6 import QtCore
    from fish_viewer.analysis.app import AnalysisWindow

    surface = cell()
    with h5py.File(ims_file, 'r+') as image:
        group = image.create_group('Scene/Content/Surfaces0')
        group.create_dataset('Vertices', data=np.column_stack([surface.vertices, np.ones_like(surface.vertices)]))
        group.create_dataset('Triangles', data=surface.triangles)
    companion = tmp_path / 'empty-scene.imx'
    companion.write_text(f'<bpImarisApplication><bpCommandLoadDataSet file="{ims_file.name}"/></bpImarisApplication>')
    assert entry_for_path(ims_file).scene is None
    window = AnalysisWindow(off_screen=True, settings=QtCore.QSettings(str(tmp_path / 'cache.ini'), QtCore.QSettings.Format.IniFormat))
    try:
        window.set_dataset(load_dataset(entry_for_path(ims_file)))
        assert len(window.dataset.scene.surfaces) == 1
        window.set_dataset(load_dataset(entry_for_path(companion)))
        assert len(window.dataset.scene.surfaces) == 0
        window.set_dataset(load_dataset(entry_for_path(ims_file)))
        assert len(window.dataset.scene.surfaces) == 1
        window.editor.document.delete('surface', 0)
        window.set_dataset(load_dataset(entry_for_path(companion)))
        assert len(window.dataset.scene.surfaces) == 0
        window.set_dataset(load_dataset(entry_for_path(ims_file)))
        assert len(window.dataset.scene.surfaces) == 0
        window.editor.document.undo.undo()
        assert len(window.dataset.scene.surfaces) == 1
    finally:
        window.close()


SUPPLIED = [
    ('11.2 S2 60D_546 GHL_488 lam_633 processed airyskan_image_only.ims', 0, 0),
    ('11.2 S2 60D_546 GHL_488 lam_633 processed airyskan.ims', 10, 13),
    ('11.3 S2 60D_546 GHL_488 lam_633 processed airyskan_image only.imx', 0, 0),
    ('11.3 S2 60D_546 GHL_488 lam_633 processed airyskan_image_only.ims', 0, 0),
    ('11.3 S2 60D_546 GHL_488 lam_633 processed airyskan.ims', 5, 6),
]


@pytest.mark.integration
@pytest.mark.parametrize('filename,cells,signals', SUPPLIED)
def test_supplied_native_files_load_and_export_without_geometry_changes(qapp, tmp_path, filename, cells, signals):
    path = Path.home() / 'Downloads' / filename
    if not path.is_file():
        pytest.skip('Supplied native regression scan is not installed')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    entry = entry_for_path(path)
    reader = ImarisReader(entry.image)
    data = load_dataset(entry, level=max(reader.metadata.levels))
    doc = AnnotationDocument(data)
    assert len(data.scene.surfaces) == cells
    assert len(data.scene.measurements) == signals
    assert not [w for w in data.scene.warnings if 'No saved annotations' not in w]
    if cells:
        assert all(v and v > 0 for v in doc.volumes)
        with h5py.File(entry.image) as image:
            for i, surface in enumerate(data.scene.surfaces):
                native = image[f'Scene/Content/Surfaces{i}']
                np.testing.assert_array_equal(surface.vertices, native['Vertices'][()][:, :3])
                np.testing.assert_array_equal(surface.triangles, native['Triangles'][()])
    before = deepcopy(data.scene)
    exported = export_document(doc, tmp_path / 'verified.ims')
    reopened = load_dataset(entry_for_path(exported), level=max(reader.metadata.levels))
    validate_roundtrip(before, reopened.scene)
    assert [s.name for s in before.surfaces] == [s.name for s in reopened.scene.surfaces]
    assert [m.name for m in before.measurements] == [m.name for m in reopened.scene.measurements]
    for expected, actual in zip(before.surfaces, reopened.scene.surfaces):
        if expected.native_xml:
            old = ET.fromstring(expected.native_xml)
            new = ET.fromstring(actual.native_xml)
            assert old.find('bpSurfaces/bpSurfacesCreationParameters').attrib == new.find('bpSurfaces/bpSurfacesCreationParameters').attrib
            assert old.findtext('bpSurfaces/historyLog') == new.findtext('bpSurfaces/historyLog')
    for expected, actual in zip(before.measurements, reopened.scene.measurements):
        old, new = ET.fromstring(expected.native_xml), ET.fromstring(actual.native_xml)
        for tag in ('PointShape', 'PointDiameter', 'LineWidth', 'LineLabels', 'LabelsColor', 'LabelsFont', 'AutodepthMode'):
            assert old.findtext(tag) == new.findtext(tag)
            assert old.find(tag).attrib == new.find(tag).attrib
    for expected, actual in zip(data.volumes, reopened.volumes):
        np.testing.assert_array_equal(expected, actual)
    with path.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == digest
