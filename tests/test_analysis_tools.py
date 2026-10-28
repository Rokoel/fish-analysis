"""Undoable tool operations, sparse display adjustment and visible-layer picking."""

from copy import deepcopy
import sys
import numpy as np
import pyvista as pv
import pytest
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

from test_annotation_editor import cell, document
from test_analysis_cell_navigation import navigation_window
from fish_viewer.display import automatic_levels, auto_adjust
from fish_viewer.models import Channel, Contour, Surface
from fish_viewer.reader import entry_for_path, load_dataset
from fish_viewer.analysis.tools import mesh_from_layers
from fish_viewer.views import mesh_for


def test_auto_display_reveals_sparse_puncta_without_hot_pixel_scaling():
    volume = np.zeros(100_000, np.uint16)
    volume[100:200] = np.linspace(100, 200, 100).astype(np.uint16)
    volume[-1] = 65535
    original = volume.copy()
    low, high, gamma = automatic_levels(volume)
    assert low == 0 and 180 <= high <= 205 and 1 <= gamma <= 1.6
    channel = Channel('signal', (0, 1, 0), brightness=.5, contrast=3, gamma=3)
    auto_adjust(channel, volume)
    assert channel.normalize(np.array([150]))[0] > .6
    assert channel.brightness == 0 and channel.contrast == 1
    np.testing.assert_array_equal(volume, original)


@pytest.mark.parametrize('values', [[0, 0, 0], [55, 55], [np.nan, np.inf], [-5, -4, -3, -2, 0]])
def test_auto_display_handles_uniform_negative_and_nonfinite_data(values):
    low, high, gamma = automatic_levels(np.asarray(values))
    assert np.isfinite([low, high, gamma]).all() and high > low


def test_proportional_edit_moves_nearby_points_in_3d_and_undo_restores(qapp, ims_file):
    doc = document(ims_file)
    before = deepcopy(doc.dataset.scene.surfaces[0])
    doc.move_section_points(0, [(2, 11., 21., 30.)], np.array([.3, 0, 0]), radius=5)
    after = doc.dataset.scene.surfaces[0]
    np.testing.assert_allclose(after.vertices[0], before.vertices[0] + [.3, 0, 0])
    # This point is on another layer, four physical units from the anchor.
    np.testing.assert_allclose(after.vertices[4], before.vertices[4] + [.3 * .104, 0, 0])
    np.testing.assert_array_equal(after.vertices[1], before.vertices[1])
    assert mesh_for(after).n_open_edges == 0 and doc.volumes[0] is not None
    assert after.uid == before.uid and after.source == before.source
    assert doc.undo.count() == 1
    doc.undo.undo()
    np.testing.assert_array_equal(doc.dataset.scene.surfaces[0].vertices, before.vertices)


def test_group_move_and_delete_are_atomic_and_do_not_double_move_signals(qapp, ims_file):
    doc = document(ims_file)
    original = doc.dataset.scene.measurements[0].points.copy()
    doc.translate({('surface', 0), ('measurement', 0)}, [1, 2, 0])
    np.testing.assert_allclose(doc.dataset.scene.measurements[0].points, original + [1, 2, 0])
    assert doc.volumes[0] == pytest.approx(96)
    assert doc.distances[0] == pytest.approx(1)
    assert doc.undo.count() == 1
    doc.delete_many({('surface', 0), ('measurement', 0)})
    assert not doc.dataset.scene.surfaces and not doc.dataset.scene.measurements
    assert doc.undo.count() == 2
    doc.undo.undo()
    assert doc.bindings == {0: 0}
    doc.undo.undo()
    np.testing.assert_array_equal(doc.dataset.scene.measurements[0].points, original)


def test_simplification_reduces_closed_mesh_with_volume_bound_and_undo(qapp, ims_file):
    doc = document(ims_file)
    sphere = pv.Sphere(radius=1, center=(14, 23, 32), theta_resolution=30, phi_resolution=30)
    surface = doc.dataset.scene.surfaces[0]
    surface.vertices = sphere.points.astype(float)
    surface.triangles = sphere.faces.reshape(-1, 4)[:, 1:].copy()
    doc.recalculate()
    before, volume = deepcopy(surface), doc.volumes[0]
    doc.simplify({('surface', 0)}, .5, .05)
    actual = doc.dataset.scene.surfaces[0]
    assert len(actual.vertices) < len(before.vertices)
    assert abs(doc.volumes[0] / volume - 1) <= .05
    assert mesh_for(actual).n_open_edges == 0
    assert doc.distances[0] > 0
    doc.undo.undo()
    np.testing.assert_array_equal(doc.dataset.scene.surfaces[0].vertices, before.vertices)


def simplification_cells(doc):
    sphere = pv.Sphere(radius=1, center=(14, 23, 32), theta_resolution=30, phi_resolution=30)
    reducible = Surface('Reducible cell', sphere.points.astype(float), sphere.faces.reshape(-1, 4)[:, 1:].copy())
    tetrahedron = Surface('Minimum tetrahedron',
                          np.array([[20, 20, 30], [22, 20, 30], [20, 22, 30], [20, 20, 32]], float),
                          np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]))
    another = deepcopy(reducible)
    another.name, another.uid = 'Another reducible cell', 'another-cell'
    another.vertices += [10, 0, 0]
    doc.dataset.scene.surfaces = [reducible, tetrahedron, another]
    doc.recalculate()


def test_simplification_skips_cells_outside_limits_and_continues(qapp, ims_file):
    doc = document(ims_file)
    simplification_cells(doc)
    before = deepcopy(doc.dataset.scene)
    volumes = doc.volumes.copy()
    result = doc.simplify({('surface', i) for i in range(3)}, .5, .05)
    assert [entry['index'] for entry in result['simplified']] == [0, 2]
    assert [entry['index'] for entry in result['skipped']] == [1]
    assert result['skipped'][0]['name'] == 'Minimum tetrahedron'
    assert result['skipped'][0]['reason']
    for index in (0, 2):
        actual = doc.dataset.scene.surfaces[index]
        assert len(actual.vertices) < len(before.surfaces[index].vertices)
        assert abs(doc.volumes[index] / volumes[index] - 1) <= .05
        assert mesh_for(actual).n_open_edges == 0
    skipped = doc.dataset.scene.surfaces[1]
    np.testing.assert_array_equal(skipped.vertices, before.surfaces[1].vertices)
    np.testing.assert_array_equal(skipped.triangles, before.surfaces[1].triangles)
    assert not skipped.edited and doc.volumes[1] == volumes[1]
    assert doc.undo.count() == 1
    doc.undo.undo()
    for actual, original in zip(doc.dataset.scene.surfaces, before.surfaces):
        np.testing.assert_array_equal(actual.vertices, original.vertices)
        np.testing.assert_array_equal(actual.triangles, original.triangles)
    doc.undo.redo()
    assert len(doc.dataset.scene.surfaces[2].vertices) < len(before.surfaces[2].vertices)


def test_simplification_all_skipped_preserves_geometry_and_undo(qapp, ims_file):
    doc = document(ims_file)
    simplification_cells(doc)
    before = deepcopy(doc.dataset.scene.surfaces[1])
    result = doc.simplify({('surface', 1)}, .5, .05)
    assert not result['simplified'] and len(result['skipped']) == 1
    assert doc.undo.count() == 0
    actual = doc.dataset.scene.surfaces[1]
    np.testing.assert_array_equal(actual.vertices, before.vertices)
    np.testing.assert_array_equal(actual.triangles, before.triangles)
    assert not actual.edited


@pytest.mark.parametrize('axis', [0, 1, 2])
def test_layered_contours_form_closed_calibrated_mesh(axis):
    polygon = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], float)
    surface = mesh_from_layers({2.: polygon, 4.: polygon + [.5, .5]}, axis, [1, 1, 1], 'new cell')
    assert mesh_for(surface).n_open_edges == 0
    assert len(surface.vertices) > 4 and np.isfinite(surface.vertices).all()
    assert surface.vertices[:, axis].min() < 2 and surface.vertices[:, axis].max() > 4


def test_new_cell_deformation_and_simplification_survive_export(qapp, ims_file, tmp_path):
    from fish_viewer.analysis.annotations import AnnotationDocument
    from fish_viewer.reader import entry_for_path, load_dataset
    from fish_viewer.workflow.edit_export import export_document, validate_roundtrip
    doc = document(ims_file)
    doc.delete('surface', 0)
    square = np.array([[11, 20], [17, 20], [17, 26], [11, 26]], float)
    doc.add_layered_cell({30.: square, 34.: square}, 2)
    surface = doc.dataset.scene.surfaces[0]
    doc.move_section_points(0, [(2, *surface.vertices[0])], np.array([.1, 0, 0]), radius=3)
    doc.simplify({('surface', 0)}, .2, .1)
    doc.add_signal([14, 23, 32], 0)
    before = deepcopy(doc.dataset.scene)
    volume, distance = doc.volumes[0], doc.distances[0]
    assert volume > 0 and distance > 0 and doc.bindings == {0: 0}
    exported = export_document(doc, tmp_path / 'layered-edited.ims')
    reopened = AnnotationDocument(load_dataset(entry_for_path(exported)))
    validate_roundtrip(before, reopened.dataset.scene)
    assert reopened.volumes[0] == pytest.approx(volume, rel=1e-6)
    assert reopened.distances[0] == pytest.approx(distance, rel=1e-6)
    assert reopened.dataset.scene.measurements[0].cell_uid == before.surfaces[0].uid


def drag(widget, start, end, modifiers=QtCore.Qt.KeyboardModifier.NoModifier):
    QtTest.QTest.mousePress(widget, QtCore.Qt.MouseButton.LeftButton, modifiers, start)
    event = QtGui.QMouseEvent(QtCore.QEvent.Type.MouseMove, QtCore.QPointF(end),
        QtCore.QPointF(widget.mapToGlobal(end)), QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.MouseButton.LeftButton, modifiers)
    from PySide6.QtWidgets import QApplication
    QApplication.sendEvent(widget, event)
    QtTest.QTest.mouseRelease(widget, QtCore.Qt.MouseButton.LeftButton, modifiers, end)


def visible_window(qapp, ims_file, tmp_path):
    window = navigation_window(ims_file, tmp_path, 0)
    second = cell()
    second.vertices = window.dataset.scene.surfaces[0].vertices - np.array([10, 0, 0])
    hidden = cell()
    hidden.vertices = window.dataset.scene.surfaces[0].vertices - np.array([0, 0, 10])
    window.dataset.scene.surfaces.extend([second, hidden])
    window.editor.document.recalculate()
    window.editor.changed()
    window.resize(1400, 900)
    window.show()
    window.slice.setValue(7)
    window.slice_view.fit()
    qapp.processEvents()
    return window


@pytest.mark.gui
def test_toolbar_and_sidebar_context(qapp, ims_file, tmp_path):
    window = visible_window(qapp, ims_file, tmp_path)
    try:
        assert not window.annotation_dock.isVisible()
        assert window.editor.mode == 'select'
        buttons = window.editor.toolbar.findChildren(QtWidgets.QToolButton)
        labels = [button.text() for button in buttons]
        assert labels.index('Proportional editing') == labels.index('Precise editing') + 1
        assert not window.editor.toolbar.isAncestorOf(window.editor.selection_count)
        assert window.editor.isAncestorOf(window.editor.selection_count)
        window.editor.select_annotation(('surface', 0))
        window.editor.set_tool_mode('proportional')
        assert window.annotation_dock.isVisible()
        assert window.editor.point_controls.isVisible()
        assert window.editor.proportional_controls.isVisible()
        assert not window.editor.add_controls.isVisible()
        assert window.editor.selection_count.isVisible()
        assert window.editor.selection_count.text() == '1 objects · 0 points'
        strip = window.editor.toolbar
        strip.layout().setGeometry(strip.rect())
        buttons = strip.findChildren(QtWidgets.QToolButton)
        assert len(buttons) == 7
        assert all(strip.rect().contains(button.geometry()) for button in buttons)
        window.editor.set_tool_mode('select')
        assert not window.annotation_dock.isVisible()
    finally:
        window.close()


@pytest.mark.gui
def test_simplification_sidebar_reports_successes_and_skips(qapp, ims_file, tmp_path):
    window = visible_window(qapp, ims_file, tmp_path)
    errors = []
    window.error = errors.append
    try:
        editor = window.editor
        simplification_cells(editor.document)
        editor.changed()
        editor.set_selection({('surface', i) for i in range(3)})
        editor.set_tool_mode('simplify')
        editor.simplify_selected()
        assert not errors and editor.document.undo.count() == 1
        report = editor.simplification_report.toPlainText()
        assert editor.simplification_report.isVisible()
        assert 'Simplified 2; skipped 1.' in report
        assert 'Cell 1 · Reducible cell:' in report
        assert 'Cell 3 · Another reducible cell:' in report
        assert 'Cell 2 · Minimum tetrahedron:' in report
        assert 'volume-change limit' in report
        assert editor.selection_count.text() == '3 objects · 0 points'
        assert not editor.toolbar.isAncestorOf(editor.selection_count)
        # Outcome details stay available when switching tools.
        editor.set_tool_mode('move')
        editor.set_tool_mode('simplify')
        assert editor.simplification_report.toPlainText() == report
        editor.set_selection({('surface', 1)})
        editor.simplify_selected()
        assert 'Simplified 0; skipped 1.' in editor.simplification_report.toPlainText()
        assert editor.document.undo.count() == 1
        assert not errors
    finally:
        window.close()


@pytest.mark.gui
def test_rectangle_selects_only_visible_layer_and_shift_adds(qapp, ims_file, tmp_path):
    window = visible_window(qapp, ims_file, tmp_path)
    try:
        view, editor = window.slice_view, window.editor
        original = [s.vertices.copy() for s in window.dataset.scene.surfaces]
        # A box around both visible cells must exclude the overlapping hidden cell.
        start = view.mapFromScene(QtCore.QPointF(12, 32))
        end = view.mapFromScene(QtCore.QPointF(30, 40))
        drag(view.viewport(), start, end)
        assert editor.selections == {('surface', 0), ('surface', 1)}
        assert view.selected_objects == editor.selections
        window.slice.setValue(2)
        assert editor.selections == {('surface', 0), ('surface', 1)}
        window.slice.setValue(7)
        editor.set_selection({('surface', 0)})
        start = view.mapFromScene(QtCore.QPointF(12, 32))
        end = view.mapFromScene(QtCore.QPointF(20, 40))
        drag(view.viewport(), start, end, QtCore.Qt.KeyboardModifier.ShiftModifier)
        assert editor.selections == {('surface', 0), ('surface', 1)}
        for expected, actual in zip(original, window.dataset.scene.surfaces):
            np.testing.assert_array_equal(expected, actual.vertices)
        assert editor.document.undo.count() == 0
    finally:
        window.close()


@pytest.mark.gui
def test_select_point_is_not_a_drag_edit(qapp, ims_file, tmp_path):
    window = visible_window(qapp, ims_file, tmp_path)
    try:
        editor, view = window.editor, window.slice_view
        editor.select_annotation(('surface', 0))
        original = window.dataset.scene.surfaces[0].vertices.copy()
        start = view.mapFromScene(editor.handles[0].pos())
        drag(view.viewport(), start, start + QtCore.QPoint(15, 10))
        assert editor.point_selection
        np.testing.assert_array_equal(window.dataset.scene.surfaces[0].vertices, original)
        assert editor.document.undo.count() == 0
    finally:
        window.close()


@pytest.mark.gui
def test_add_new_layered_cell_can_be_refined_and_undone(qapp, ims_file, tmp_path):
    window = visible_window(qapp, ims_file, tmp_path)
    try:
        editor, view = window.editor, window.slice_view
        editor.set_tool_mode('add')
        editor.tool.setCurrentIndex(0)
        for layer in (4, 6):
            window.slice.setValue(layer)
            for x, y in [(20, 23), (26, 23), (26, 29), (20, 29)]:
                point = view.mapFromScene(QtCore.QPointF(x, y))
                QtTest.QTest.mouseClick(view.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=point)
            editor.finish_layer()
        assert len(editor.draft_layers) == 2
        editor.finish_cell()
        assert len(window.dataset.scene.surfaces) == 4
        assert editor.selection == ('surface', 3) and editor.mode == 'precise'
        assert editor.document.volumes[3] > 0
        assert window.dataset.scene.surfaces[3].run_id == editor.document.run_id
        editor.document.undo.undo()
        assert len(window.dataset.scene.surfaces) == 3
    finally:
        window.close()


@pytest.mark.gui
def test_selected_new_cell_is_cleared_before_switching_to_fewer_cells(qapp, ims_file, tmp_path, monkeypatch):
    from fish_viewer.models import Scene
    window = visible_window(qapp, ims_file, tmp_path)
    exceptions = []
    monkeypatch.setattr(sys, 'excepthook', lambda *exc: exceptions.append(exc))
    try:
        editor = window.editor
        square = np.array([[20, 23], [26, 23], [26, 29], [20, 29]], float)
        editor.document.add_layered_cell({window.slice_view.coordinate: square}, 2)
        editor.select_annotation(('surface', 3))
        editor.set_tool_mode('precise')
        assert editor.handles
        other = deepcopy(window.dataset)
        other.metadata.path = tmp_path / 'another-scan.ims'
        other.scene = Scene()
        window.set_dataset(other)
        qapp.processEvents()
        assert not exceptions, [(kind.__name__, str(exc)) for kind, exc, _ in exceptions]
        assert editor.selection is None and not editor.selections
        assert not editor.handles and not editor.point_selection
        # The original cell and its undo history still belong to the first scan.
        window.set_dataset(load_dataset(entry_for_path(ims_file)))
        assert len(window.dataset.scene.surfaces) == 4
        editor.select_annotation(('surface', 3))
        editor.document.undo.undo()
        qapp.processEvents()
        assert len(window.dataset.scene.surfaces) == 3
        assert editor.selection is None and not editor.handles
        assert not exceptions
    finally:
        window.close()


@pytest.mark.gui
def test_delayed_selection_of_removed_cell_is_ignored(qapp, ims_file, tmp_path):
    window = visible_window(qapp, ims_file, tmp_path)
    try:
        editor = window.editor
        removed = ('surface', 2)
        editor.document.delete(*removed)
        # A redraw can arrive before the stale selection notification is cleared.
        editor.selection, editor.selections = removed, {removed}
        assert editor.points().shape == (0, 3)
        editor.draw_handles()
        editor.select_annotation(removed)
        editor.draw_handles()
        assert editor.selection is None and not editor.selections
        assert not editor.handles
    finally:
        window.close()


@pytest.mark.gui
def test_new_cell_selection_delete_undo_redo_does_not_leave_handles(qapp, ims_file, tmp_path, monkeypatch):
    window = visible_window(qapp, ims_file, tmp_path)
    exceptions = []
    monkeypatch.setattr(sys, 'excepthook', lambda *exc: exceptions.append(exc))
    try:
        editor = window.editor
        original_bytes = ims_file.read_bytes()
        square = np.array([[20, 33], [26, 33], [26, 39], [20, 39]], float)
        editor.document.add_layered_cell({window.slice_view.coordinate: square}, 2)
        uid = window.dataset.scene.surfaces[3].uid
        editor.select_annotation(('surface', 3))
        editor.set_tool_mode('precise')
        editor.choose_point(editor.handles[0])
        editor.delete_objects()
        qapp.processEvents()
        assert len(window.dataset.scene.surfaces) == 3
        assert editor.selection is None and not editor.handles
        assert not editor.point_selection and editor.active_point is None
        editor.document.undo.undo()
        assert window.dataset.scene.surfaces[3].uid == uid
        editor.select_annotation(('surface', 3))
        assert editor.handles
        editor.document.undo.redo()
        qapp.processEvents()
        assert len(window.dataset.scene.surfaces) == 3
        assert editor.selection is None and not editor.handles
        assert not exceptions
        assert ims_file.read_bytes() == original_bytes
    finally:
        window.close()


@pytest.mark.gui
def test_queued_point_drag_cannot_edit_replacement_cell(qapp, ims_file, tmp_path):
    window = visible_window(qapp, ims_file, tmp_path)
    errors = []
    window.error = errors.append
    try:
        editor, view = window.editor, window.slice_view
        square = np.array([[20, 33], [26, 33], [26, 39], [20, 39]], float)
        editor.document.add_layered_cell({view.coordinate: square}, 2)
        editor.select_annotation(('surface', 3))
        editor.set_tool_mode('precise')
        start = view.mapFromScene(editor.handles[0].pos())
        drag(view.viewport(), start, start + QtCore.QPoint(10, 5))
        # The point edit is deferred until the next Qt event-loop turn.
        assert editor.document.undo.index() == 1
        editor.document.undo.undo()
        editor.document.add_layered_cell({view.coordinate: square + [1, 0]}, 2)
        before = deepcopy(window.dataset.scene.surfaces[3])
        qapp.processEvents()
        actual = window.dataset.scene.surfaces[3]
        assert not errors
        np.testing.assert_array_equal(actual.vertices, before.vertices)
        np.testing.assert_array_equal(actual.triangles, before.triangles)
        assert editor.document.undo.index() == 1
    finally:
        window.close()


@pytest.mark.gui
def test_shift_click_adds_objects_and_points_without_selecting_other_layers(qapp, ims_file, tmp_path):
    window = visible_window(qapp, ims_file, tmp_path)
    try:
        view, editor = window.slice_view, window.editor
        def border(index):
            a, b = view._cuts[(index, view.axis, view.index)][0]
            u, v = view.projection_axes
            return view.mapFromScene(QtCore.QPointF(float((a[u] + b[u]) / 2), float((a[v] + b[v]) / 2)))
        QtTest.QTest.mouseClick(view.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=border(0))
        assert editor.selections == {('surface', 0)}
        QtTest.QTest.mouseClick(view.viewport(), QtCore.Qt.MouseButton.LeftButton,
                              QtCore.Qt.KeyboardModifier.ShiftModifier, border(1))
        assert editor.selections == {('surface', 0), ('surface', 1)}
        editor.set_selection({('surface', 0)})
        points = [view.mapFromScene(h.pos()) for h in editor.handles]
        first = points[0]
        second = next(p for p in points if (p - first).manhattanLength() > 40)
        QtTest.QTest.mouseClick(view.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=first)
        QtTest.QTest.mouseClick(view.viewport(), QtCore.Qt.MouseButton.LeftButton,
                              QtCore.Qt.KeyboardModifier.ShiftModifier, second)
        assert len(editor.point_selection) == 2
        assert editor.selections == {('surface', 0)}
    finally:
        window.close()


@pytest.mark.gui
def test_group_drag_moves_cells_and_has_one_undo(qapp, ims_file, tmp_path):
    window = visible_window(qapp, ims_file, tmp_path)
    errors = []
    window.error = errors.append
    try:
        editor, view = window.editor, window.slice_view
        editor.set_selection({('surface', 0), ('surface', 1)})
        editor.set_tool_mode('move')
        before = [s.vertices.copy() for s in window.dataset.scene.surfaces]
        a, b = view._cuts[(0, view.axis, view.index)][0]
        start = view.mapFromScene(QtCore.QPointF(float((a[0] + b[0]) / 2), float((a[1] + b[1]) / 2)))
        end = start + QtCore.QPoint(20, 10)
        offset = view.mapToScene(end) - view.mapToScene(start)
        drag(view.viewport(), start, end)
        assert not errors
        delta = [offset.x(), offset.y(), 0]
        np.testing.assert_allclose(window.dataset.scene.surfaces[0].vertices, before[0] + delta)
        np.testing.assert_allclose(window.dataset.scene.surfaces[1].vertices, before[1] + delta)
        np.testing.assert_array_equal(window.dataset.scene.surfaces[2].vertices, before[2])
        assert editor.document.undo.count() == 1
        editor.document.undo.undo()
        np.testing.assert_array_equal(window.dataset.scene.surfaces[0].vertices, before[0])
    finally:
        window.close()


@pytest.mark.gui
def test_proportional_handle_drag_preview_and_commit_match(qapp, ims_file, tmp_path):
    window = visible_window(qapp, ims_file, tmp_path)
    errors = []
    window.error = errors.append
    try:
        editor, view = window.editor, window.slice_view
        editor.select_annotation(('surface', 0))
        editor.set_tool_mode('proportional')
        editor.radius.setValue(5)
        qapp.processEvents()
        before = window.dataset.scene.surfaces[0].vertices.copy()
        handle = editor.handles[0]
        start = view.mapFromScene(handle.pos())
        drag(view.viewport(), start, start + QtCore.QPoint(-10, -5))
        qapp.processEvents()
        assert not errors and editor.document.undo.count() == 1
        after = window.dataset.scene.surfaces[0].vertices
        # Neighboring original mesh vertices move too, not only a new slice point.
        assert np.any(np.linalg.norm(after[:len(before)] - before, axis=1) > 1e-7)
        assert mesh_for(window.dataset.scene.surfaces[0]).n_open_edges == 0
        editor.document.undo.undo()
        np.testing.assert_array_equal(window.dataset.scene.surfaces[0].vertices, before)
    finally:
        window.close()


@pytest.mark.gui
def test_add_signal_and_current_layer_contour_point(qapp, ims_file, tmp_path):
    window = visible_window(qapp, ims_file, tmp_path)
    errors = []
    window.error = errors.append
    try:
        editor, view = window.editor, window.slice_view
        editor.set_tool_mode('add')
        editor.tool.setCurrentIndex(2)
        position = view.mapFromScene(QtCore.QPointF(26, 36))
        QtTest.QTest.mouseClick(view.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=position)
        assert len(window.dataset.scene.measurements) == 1
        assert editor.document.bindings[0] == 0 and editor.document.distances[0] > 0
        editor.set_selection({('surface', 0)})
        editor.tool.setCurrentIndex(1)
        before = len(window.dataset.scene.surfaces[0].vertices)
        a, b = view._cuts[(0, view.axis, view.index)][0]
        point = (a + b) / 2
        position = view.mapFromScene(QtCore.QPointF(float(point[0]), float(point[1])))
        QtTest.QTest.mouseClick(view.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=position)
        assert not errors
        assert len(window.dataset.scene.surfaces[0].vertices) == before + 1
        assert np.min(abs(window.dataset.scene.surfaces[0].vertices[:, 2] - view.coordinate)) < 1e-7
    finally:
        window.close()


@pytest.mark.gui
def test_multiple_drawn_polygon_points_move_and_delete_together(qapp, ims_file, tmp_path):
    window = visible_window(qapp, ims_file, tmp_path)
    errors = []
    window.error = errors.append
    try:
        editor = window.editor
        angles = np.arange(6) * np.pi / 3
        polygon = np.column_stack([24 + 2 * np.cos(angles), 36 + 2 * np.sin(angles)])
        window.dataset.scene.contours.append(Contour('drawn polygon', polygon.copy(), 2, 7))
        editor.changed()
        editor.select_annotation(('contour', 0))
        editor.set_tool_mode('precise')
        before = editor.points().copy()
        anchors = [(2, *before[i]) for i in (0, 1)]
        delta = np.array([.25, .15, 0])
        editor.commit_handle(('contour', 0), before[0], before[0] + delta, 2, anchors)
        assert not errors and editor.document.undo.count() == 1
        np.testing.assert_allclose(editor.points()[:2], before[:2] + delta)
        np.testing.assert_allclose(editor.points()[2:], before[2:])
        editor.delete_point()
        assert not errors and len(editor.points()) == 4
        assert editor.document.undo.count() == 2
        editor.document.undo.undo()
        np.testing.assert_allclose(editor.points()[:2], before[:2] + delta)
        editor.document.undo.undo()
        np.testing.assert_allclose(editor.points(), before)
    finally:
        window.close()
