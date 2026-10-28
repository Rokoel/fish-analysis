"""Selecting annotations reveals them in the current 2D orientation."""

import numpy as np
import pytest
from PySide6 import QtCore, QtTest

from test_annotation_editor import cell
from fish_viewer.analysis.app import AnalysisWindow
from fish_viewer.models import Measurement, Scene
from fish_viewer.reader import entry_for_path, load_dataset


def navigation_window(ims_file, tmp_path, axis, reduced=False, gap=False):
    dataset = load_dataset(entry_for_path(ims_file))
    dataset.metadata.shape = (24, 24, 24) if reduced else (12, 12, 12)
    dataset.level = 1 if reduced else 0
    dataset.volumes = [np.zeros((12, 12, 12), dtype=v.dtype) for v in dataset.volumes]
    surface = cell()
    xyz_axis = 2 - axis
    unit = (surface.vertices - surface.vertices.min(0)) / np.ptp(surface.vertices, axis=0)
    voxels = 6.2 + unit * 2.6
    if gap:
        voxels[:, xyz_axis] = 3.2 + unit[:, xyz_axis] * 1.6
    surface.vertices = np.asarray(dataset.center_origin) + voxels * dataset.spacing
    if gap:
        other = surface.vertices.copy()
        other[:, xyz_axis] += 5 * dataset.spacing[xyz_axis]
        surface.triangles = np.vstack([surface.triangles, surface.triangles + len(other)])
        surface.vertices = np.vstack([surface.vertices, other])
    dataset.scene = Scene(surfaces=[surface])
    window = AnalysisWindow(off_screen=True, settings=QtCore.QSettings(
        str(tmp_path / 'navigation.ini'), QtCore.QSettings.Format.IniFormat))
    window.set_dataset(dataset)
    window.axis.setCurrentIndex(axis)
    window.sync_buttons()
    return window


@pytest.mark.gui
@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('reduced', [False, True])
@pytest.mark.parametrize('current,expected', [(0, 7), (11, 8), (7, 7)])
def test_cell_selection_reveals_nearest_visible_slice(qapp, ims_file, tmp_path, axis, reduced, current, expected):
    window = navigation_window(ims_file, tmp_path, axis, reduced)
    try:
        window.slice.setValue(current)
        transform = window.slice_view.transform()
        window.editor.select_annotation(('surface', 0))
        assert window.slice.value() == window.slice_view.index == expected
        assert window.slice_view._cuts[(0, axis, expected)].size
        assert window.editor.handles
        assert window.slice_view.transform() == transform
    finally:
        window.close()


@pytest.mark.gui
@pytest.mark.parametrize('axis', [0, 1, 2])
def test_cell_selection_checks_real_geometry_inside_bounding_box_gaps(qapp, ims_file, tmp_path, axis):
    window = navigation_window(ims_file, tmp_path, axis, gap=True)
    try:
        window.slice.setValue(6)
        window.editor.select_annotation(('surface', 0))
        assert window.slice.value() == 4
        assert window.slice_view._cuts[(0, axis, 4)].size
    finally:
        window.close()


@pytest.mark.gui
def test_clicking_already_selected_cell_reveals_it_again(qapp, ims_file, tmp_path):
    window = navigation_window(ims_file, tmp_path, 0)
    try:
        window.show()
        window.editor.select_annotation(('surface', 0))
        window.slice.setValue(0)
        qapp.processEvents()
        item = window.editor.tree.currentItem()
        rect = window.editor.tree.visualItemRect(item)
        QtTest.QTest.mouseClick(window.editor.tree.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=rect.center())
        assert window.slice.value() == 7
    finally:
        window.close()


@pytest.mark.gui
def test_cell_count_row_reveals_selected_cell(qapp, ims_file, tmp_path):
    window = navigation_window(ims_file, tmp_path, 0)
    try:
        window.slice.setValue(0)
        window.counts.selectRow(0)
        assert window.editor.selection == ('surface', 0)
        assert window.slice.value() == 7
    finally:
        window.close()


@pytest.mark.gui
def test_cell_count_selection_uses_identity_with_duplicate_cell_names(qapp, ims_file, tmp_path):
    window = navigation_window(ims_file, tmp_path, 0)
    try:
        second = cell()
        second.vertices = window.dataset.scene.surfaces[0].vertices - 4 * np.asarray(window.dataset.spacing)
        window.dataset.scene.surfaces.append(second)
        window.editor.document.recalculate()
        window.editor.changed()
        window.slice.setValue(0)
        window.counts.selectRow(1)
        assert window.editor.selection == ('surface', 1)
        assert window.slice.value() == 3
    finally:
        window.close()


@pytest.mark.gui
def test_cell_selection_does_not_change_inactive_2d_slice(qapp, ims_file, tmp_path):
    window = navigation_window(ims_file, tmp_path, 0)
    try:
        window.slice.setValue(0)
        # Exercise the selection's mode guard without creating a native GL view.
        window.mode.blockSignals(True)
        window.mode.setCurrentIndex(1)
        window.mode.blockSignals(False)
        window.editor.select_annotation(('surface', 0))
        assert window.slice.value() == 0
    finally:
        window.close()


def add_measurement(window, a_slice=7.2, reversed_points=False, single=False):
    center = np.asarray(window.dataset.center_origin)
    spacing = np.asarray(window.dataset.spacing)
    a = center + np.full(3, a_slice) * spacing
    b = a.copy()
    axis = 2 - window.slice_view.axis
    b[axis] = center[axis] + 2.2 * spacing[axis]
    points, labels = ([a], ['A']) if single else ([a, b], ['A', 'B'])
    if reversed_points:
        points, labels = points[::-1], labels[::-1]
    window.dataset.scene.measurements.append(Measurement(
        'Measurement Points 1', np.asarray(points), labels, [0] * len(points),
        [] if single else [(0, 1)]))
    window.editor.set_dataset(window.dataset, None)


@pytest.mark.gui
@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('reduced', [False, True])
@pytest.mark.parametrize('current', [0, 2, 6, 7])
def test_measurement_selection_reveals_signal_a(qapp, ims_file, tmp_path, axis, reduced, current):
    window = navigation_window(ims_file, tmp_path, axis, reduced)
    try:
        add_measurement(window)
        window.slice.setValue(current)
        transform = window.slice_view.transform()
        before = window.dataset.scene.measurements[0].points.copy()
        window.editor.select_annotation(('measurement', 0))
        assert window.slice.value() == window.slice_view.index == 7
        assert window.editor.handles
        assert window.slice_view.transform() == transform
        np.testing.assert_array_equal(window.dataset.scene.measurements[0].points, before)
    finally:
        window.close()


@pytest.mark.gui
@pytest.mark.parametrize('reversed_points,single', [(True, False), (False, True)])
def test_measurement_navigation_handles_reversed_ab_and_single_spots(qapp, ims_file, tmp_path, reversed_points, single):
    window = navigation_window(ims_file, tmp_path, 0)
    try:
        add_measurement(window, reversed_points=reversed_points, single=single)
        window.slice.setValue(0)
        window.editor.select_annotation(('measurement', 0))
        assert window.slice.value() == 7
    finally:
        window.close()


@pytest.mark.gui
def test_measurement_keeps_either_slice_showing_a_boundary_signal(qapp, ims_file, tmp_path):
    window = navigation_window(ims_file, tmp_path, 0)
    try:
        add_measurement(window, a_slice=7.5)
        window.slice.setValue(7)
        window.editor.select_annotation(('measurement', 0))
        assert window.slice.value() == 7
        window.slice.setValue(8)
        window.editor.select_annotation(('measurement', 0))
        assert window.slice.value() == 8
    finally:
        window.close()


@pytest.mark.gui
def test_clicking_selected_measurement_again_reveals_signal(qapp, ims_file, tmp_path):
    window = navigation_window(ims_file, tmp_path, 0)
    try:
        add_measurement(window)
        window.show()
        window.editor.select_annotation(('measurement', 0))
        window.slice.setValue(0)
        qapp.processEvents()
        item = window.editor.tree.currentItem()
        rect = window.editor.tree.visualItemRect(item)
        QtTest.QTest.mouseClick(window.editor.tree.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=rect.center())
        assert window.slice.value() == 7
    finally:
        window.close()
