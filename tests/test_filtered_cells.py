"""Review-only cells remain grey and unexported until they contain an added signal."""
from copy import deepcopy
import json
from pathlib import Path
import time
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
from openpyxl import load_workbook
from PySide6 import QtCore, QtGui
import pytest

from test_workflow import signal_result  # noqa: F401
from fish_viewer.analysis.annotations import AnnotationDocument
from fish_viewer.analysis.app import AnalysisWindow
from fish_viewer.analysis.filtered_cells import restore_filtered_cells
from fish_viewer.annotation_state import activate_run, annotation_report
from fish_viewer.models import Measurement
from fish_viewer.reader import entry_for_path, load_dataset
from fish_viewer.workflow.distances import measure_result
from fish_viewer.workflow.edit_export import export_document
from fish_viewer.workflow.export import export_result
from fish_viewer.workflow.results import filter_result
from fish_viewer.views import VolumeView


@pytest.fixture
def filtered_analysis(signal_result, tmp_path):
    filtered = filter_result(signal_result, tmp_path / 'work')
    report = measure_result(Path(filtered['directory']) / 'report.json', tmp_path / 'work')
    output = export_result(Path(report['directory']) / 'report.json', tmp_path / 'automatic.ims')
    dataset = load_dataset(entry_for_path(output['image']))
    activate_run(dataset.scene, output['annotation_run_id'])
    return dataset, report, output


def workbook_cells(path):
    book = load_workbook(Path(path).with_suffix('.xlsx'), data_only=True)
    try:
        return {row[1] for row in list(book['Measurements'].values)[1:]}
    finally:
        book.close()


def test_removed_cell_export_promotion_undo_and_reopening(qapp, filtered_analysis, tmp_path):
    dataset, report, output = filtered_analysis
    assert len(dataset.scene.surfaces) == 2  # Automatic export still omits the empty cell.
    restore_filtered_cells(dataset, report, output['annotation_run_id'])
    assert len(dataset.scene.surfaces) == 3
    candidate = dataset.scene.surfaces.pop()
    dataset.scene.surfaces.insert(0, candidate)  # Exercise binding-index remapping on export.
    doc = AnnotationDocument(dataset, report)
    assert candidate.filtered_cell_id == 1 and not candidate.export_included
    assert candidate.display_color == (.55, .55, .55)
    assert len(doc.report()['cell_volumes']) == 2
    before = Path(export_document(doc, tmp_path / 'before.ims'))
    reopened = AnnotationDocument(load_dataset(entry_for_path(before)))
    assert len(reopened.dataset.scene.surfaces) == 2
    assert candidate.uid not in {s.uid for s in reopened.dataset.scene.surfaces}
    assert len(workbook_cells(before)) == 2
    assert {m.cell_uid for m in reopened.dataset.scene.measurements} == {s.uid for s in reopened.dataset.scene.surfaces}

    doc.add_signal([11., 23., 31.], 1)
    assert doc.bindings[len(dataset.scene.measurements) - 1] == 0
    assert candidate.export_included and candidate.display_color == candidate.color
    added = next(p for p in doc.report()['points'] if p['cell_id'] == candidate.uid)
    assert added['distance_um'] >= 0 and len(added['border_xyz']) == 3
    after = Path(export_document(doc, tmp_path / 'after.ims'))
    restored = load_dataset(entry_for_path(after))
    assert len(restored.scene.surfaces) == len(workbook_cells(after)) == 3
    retained = next(s for s in restored.scene.surfaces if s.uid == candidate.uid)
    assert retained.filtered_cell_id == 1
    restore_filtered_cells(restored, annotation_report(restored))
    assert len(restored.scene.surfaces) == 3  # No duplicate of the recovered cell.

    doc.undo.undo()
    assert not dataset.scene.surfaces[0].export_included
    assert len(doc.report()['cell_volumes']) == 2
    doc.undo.redo()
    assert dataset.scene.surfaces[0].export_included
    added_index = next(i for i, m in enumerate(dataset.scene.measurements) if m.uid == added['signal_id'])
    doc.move('measurement', added_index, 0, [100., 100., 100.])
    assert not dataset.scene.surfaces[0].export_included
    doc.undo.undo()
    assert dataset.scene.surfaces[0].export_included
    doc.delete('measurement', added_index)
    assert not dataset.scene.surfaces[0].export_included
    assert dataset.scene.surfaces[0].display_color == (.55, .55, .55)


def test_ui_load_show_hide_colour_and_promotion(qapp, filtered_analysis, tmp_path):
    _, report, output = filtered_analysis
    path = tmp_path / 'analysis-result.json'
    path.write_text(json.dumps(dict(engine='fish-analysis', status='complete',
        source=report['image'], image=output['image'], signal_report=str(Path(report['directory']) / 'report.json'),
        annotation_run_id=output['annotation_run_id'])))
    settings = QtCore.QSettings(str(tmp_path / 'ui.ini'), QtCore.QSettings.Format.IniFormat)
    window = AnalysisWindow(off_screen=True, settings=settings)
    errors = []
    window.error = errors.append
    try:
        window.load_result(path)
        deadline = time.monotonic() + 5
        while (window.loading or window._jobs) and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(.01)
        assert not window.loading and not errors
        doc = window.editor.document
        cell = next(s for s in window.dataset.scene.surfaces if s.filtered_cell_id == 1)
        index = next(i for i, s in enumerate(window.dataset.scene.surfaces) if s.uid == cell.uid)
        toggle = window.editor.show_removed_cells
        assert not toggle.isChecked() and not cell.visible
        assert not any(item.data(0) == ('surface', index) for item in window.slice_view._overlays)
        toggle.setChecked(True)
        assert cell.visible
        paths = [item for item in window.slice_view._overlays if item.data(0) == ('surface', index)]
        assert paths and paths[0].pen().color() == QtGui.QColor.fromRgbF(.55, .55, .55)
        assert len(doc.report()['cell_volumes']) == 2
        toggle.setChecked(False)
        assert not cell.visible
        toggle.setChecked(True)
        doc.add_signal([11., 23., 31.], 1)
        assert cell.export_included and cell.visible
        assert window.counts.rowCount() == 3 and not toggle.isEnabled()
        paths = [item for item in window.slice_view._overlays if item.data(0) == ('surface', index)]
        assert paths[0].pen().color() == QtGui.QColor.fromRgbF(*cell.color)
        doc.undo.undo()
        assert toggle.isEnabled() and window.counts.rowCount() == 2
        assert not window.dataset.scene.surfaces[index].export_included
    finally:
        window.close()


def test_imported_signal_does_not_automatically_promote_review_cell(qapp, filtered_analysis):
    dataset, report, output = filtered_analysis
    restore_filtered_cells(dataset, report, output['annotation_run_id'])
    dataset.scene.measurements.append(Measurement('pre-existing unassociated signal',
        np.array([[11., 23., 31.]]), ['A'], [0], channel=1))
    doc = AnnotationDocument(dataset, report)
    assert not dataset.scene.surfaces[-1].export_included
    assert doc.bindings[len(dataset.scene.measurements) - 1] is None


def test_added_signal_prefers_review_cell_over_overlapping_imported_cell(qapp, filtered_analysis):
    dataset, report, output = filtered_analysis
    restore_filtered_cells(dataset, report, output['annotation_run_id'])
    imported = deepcopy(dataset.scene.surfaces[-1])
    imported.uid, imported.source, imported.filtered_cell_id = str(uuid4()), 'manual', None
    dataset.scene.surfaces.insert(0, imported)
    doc = AnnotationDocument(dataset, report)
    candidate = dataset.scene.surfaces[-1]
    doc.add_signal([11., 23., 31.], 1)
    signal = next(i for i, m in enumerate(dataset.scene.measurements) if m.cell_uid == candidate.uid)
    doc.translate({('measurement', signal)}, [0., .1, 0.])
    assert dataset.scene.measurements[signal].cell_uid == candidate.uid
    assert candidate.export_included
    doc.undo.undo()
    doc.undo.undo()
    assert not dataset.scene.surfaces[-1].export_included
    doc.add_signal([11., 23., 31.], 1, preferred_cell=0)
    assert not dataset.scene.surfaces[-1].export_included
    assert dataset.scene.measurements[-1].cell_uid == imported.uid


def test_volume_actor_uses_grey_until_export_inclusion(filtered_analysis):
    from vtkmodules.vtkRenderingCore import vtkActor

    dataset, report, output = filtered_analysis
    restore_filtered_cells(dataset, report, output['annotation_run_id'])
    cell = dataset.scene.surfaces[-1]
    actor = vtkActor()
    view = SimpleNamespace(dataset=SimpleNamespace(scene=SimpleNamespace(surfaces=[cell], contours=[], measurements=[]), time=0),
        _mesh_actors=[actor], _annotation_actors=[], surfaces_visible=True, surface_opacity=.32,
        wireframe=False, is_selected=lambda *args: True, plotter=SimpleNamespace(render=lambda: None))
    cell.visible = True
    VolumeView.update_annotations(view)
    assert actor.GetVisibility() and actor.GetProperty().GetColor() == (.55, .55, .55)
    cell.visible = False
    VolumeView.update_annotations(view)
    assert not actor.GetVisibility()
    cell.export_included = cell.visible = True
    view.is_selected = lambda *args: False
    VolumeView.update_annotations(view)
    assert actor.GetVisibility() and actor.GetProperty().GetColor() == cell.color


def test_missing_review_archive_keeps_exported_analysis_readable(filtered_analysis):
    dataset, report, output = filtered_analysis
    Path(report['unfiltered_cell_predictions']).unlink()
    restore_filtered_cells(dataset, report, output['annotation_run_id'])
    assert len(dataset.scene.surfaces) == 2


def test_review_archive_checks_pixel_identity_before_restoring(filtered_analysis):
    dataset, report, output = filtered_analysis
    report['fingerprint'] = 'different-source'
    with pytest.raises(ValueError, match='different image pixels'):
        restore_filtered_cells(dataset, report, output['annotation_run_id'])
    assert len(dataset.scene.surfaces) == 2


def test_all_cells_filtered_can_be_reviewed_and_recovered(qapp, signal_result, tmp_path):
    source = json.loads(signal_result.read_text())
    source['points'] = []
    signal_result.write_text(json.dumps(source))
    filtered = filter_result(signal_result, tmp_path / 'work')
    measured = measure_result(Path(filtered['directory']) / 'report.json', tmp_path / 'work')
    exported = export_result(Path(measured['directory']) / 'report.json', tmp_path / 'empty.ims')
    dataset = load_dataset(entry_for_path(exported['image']))
    assert not dataset.scene.surfaces
    restore_filtered_cells(dataset, measured, exported['annotation_run_id'])
    doc = AnnotationDocument(dataset, measured)
    assert len(dataset.scene.surfaces) == 3
    assert all(not s.export_included for s in dataset.scene.surfaces)
    assert not doc.report()['cell_volumes']
    doc.add_signal([11., 23., 31.], 1)
    target = export_document(doc, tmp_path / 'recovered.ims')
    assert len(load_dataset(entry_for_path(target)).scene.surfaces) == len(workbook_cells(target)) == 1
