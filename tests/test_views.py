import copy

import numpy as np
import pyvista as pv
import pytest

from fish_viewer.app import MainWindow
from fish_viewer.reader import entry_for_path, load_dataset
from fish_viewer.views import clip_segment, slice_segments


@pytest.mark.gui
@pytest.mark.parametrize('unit_scale', [1., 1e-6])
def test_extreme_zoom_keeps_location_and_stays_within_scrollbar_range(qapp, ims_file, unit_scale):
    from PySide6 import QtCore, QtGui, QtTest
    from fish_viewer.models import Scene
    from fish_viewer.views import SliceView
    data = load_dataset(entry_for_path(ims_file))
    data.metadata.shape = (2, 512, 512)
    data.metadata.spacing = tuple(unit_scale * value for value in (.04, .04, .32))
    data.metadata.origin = tuple(unit_scale * value for value in (.5, .5, 0))
    data.scene = Scene()
    data.volumes = [np.zeros(data.metadata.shape, np.uint8) for _ in data.volumes]
    data.volumes[0].fill(80)
    view = SliceView()
    view.resize(900, 700)
    view.show()
    try:
        view.set_dataset(data)
        qapp.processEvents()
        position = QtCore.QPoint(430, 310)
        QtTest.QTest.mouseMove(view.viewport(), position)
        qapp.processEvents()
        jumps = []
        for index in range(130):
            before = view.mapToScene(position)
            event = QtGui.QWheelEvent(QtCore.QPointF(position),
                QtCore.QPointF(view.viewport().mapToGlobal(position)), QtCore.QPoint(), QtCore.QPoint(0, 120),
                QtCore.Qt.MouseButton.NoButton, QtCore.Qt.KeyboardModifier.ControlModifier,
                QtCore.Qt.ScrollPhase.ScrollUpdate, False)
            view.wheelEvent(event)
            after = view.mapToScene(position)
            if index > 15:
                jumps.append(np.linalg.norm([after.x() - before.x(), after.y() - before.y()]))
        assert max(jumps) < .01 * unit_scale
        assert view.transform().m11() * data.spacing[0] <= 256 * (1 + 1e-9)
        assert view.horizontalScrollBar().maximum() < 2**30
        # Scrolling after the limit does not move the view or change slice.
        before, transform = view.mapToScene(position), view.transform()
        view.wheelEvent(event)
        assert view.mapToScene(position) == before and view.transform() == transform
        assert view.index == 1
        color = view.viewport().grab().toImage().pixelColor(position)
        assert color.green() >= 190 and color.red() < 20
    finally:
        view.close()


def test_mesh_slice_is_geometric_intersection():
    sphere = pv.Sphere(radius=3, center=(10, 20, 30), theta_resolution=60, phi_resolution=60)
    segments = slice_segments(sphere, 2, 30)
    assert len(segments) > 30
    np.testing.assert_allclose(segments[:, :, 2], 30)
    radii = np.linalg.norm(segments[:, :, :2] - [10, 20], axis=-1)
    np.testing.assert_allclose(radii, 3, atol=0.01)
    assert slice_segments(sphere, 2, 34).size == 0


def test_measurement_line_clipped_to_slice_slab():
    a, b = np.array([0, 0, 0.0]), np.array([10, 0, 10.0])
    clipped = clip_segment(a, b, 2, 5, 0.5)
    np.testing.assert_allclose(clipped, [[4.5, 0, 4.5], [5.5, 0, 5.5]])
    assert clip_segment(a, b, 2, 15, 0.5) is None


@pytest.mark.gui
def test_2d_controls_and_view_settings(qapp, ims_file):
    window = MainWindow(off_screen=True)
    try:
        window.show()
        data = load_dataset(entry_for_path(ims_file))
        window.set_dataset(data)
        qapp.processEvents()
        window.orientation.setCurrentIndex(1)
        assert window.slice_slider.maximum() == 2
        window.orientation.setCurrentIndex(2)
        assert window.slice_slider.maximum() == 3
        window.slice_slider.setValue(2)
        assert window.slice_number.value() == 3
        assert window.slice_view.coordinate == 15
        panel = window.channel_panels[0]
        panel.brightness.setValue(20)
        panel.visible.setChecked(False)
        state = window.view_state()
        panel.brightness.setValue(-30)
        panel.visible.setChecked(True)
        window.apply_state(state)
        assert not data.metadata.channels[0].visible
        assert data.metadata.channels[0].brightness == 0.2
        assert window.slice_view.index == 2
        assert not window.slice_view.viewport().grab().isNull()
        invalid = copy.deepcopy(state)
        invalid['channels'][0]['gamma'] = float('nan')
        with pytest.raises(ValueError):
            window.apply_state(invalid)
    finally:
        window.close()
        qapp.processEvents()


@pytest.mark.gui
@pytest.mark.integration
def test_reference_2d_borders_and_measurement_focus(qapp, reference):
    window = MainWindow(off_screen=True)
    try:
        window.show()
        data = load_dataset(entry_for_path(reference))
        window.set_dataset(data)
        qapp.processEvents()
        assert len(window.slice_view._overlays) >= 10
        window.show_surfaces.setChecked(False)
        window.show_lines.setChecked(False)
        window.show_markers.setChecked(False)
        assert not window.slice_view._overlays
        window.show_surfaces.setChecked(True)
        window.show_lines.setChecked(True)
        window.show_markers.setChecked(True)
        item = window.objects.topLevelItem(1).child(0)
        window.objects.setCurrentItem(item)
        window.focus_object()
        assert 'Total distance' in window.object_info.text()
        assert window.slice_view.selected == ('measurement', 0)
    finally:
        window.close()
        qapp.processEvents()


@pytest.mark.gui
def test_background_load_and_file_switch(qapp, ims_file):
    import time
    window = MainWindow(off_screen=True)
    errors = []
    window.error = errors.append
    try:
        entry = entry_for_path(ims_file)
        window.load_entry(entry)
        deadline = time.monotonic() + 10
        while window._pending_load and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.01)
        assert not errors
        assert window.dataset is not None
        window.channel_panels[0].brightness.setValue(40)
        window.load_entry(entry, time=1, level=1)
        deadline = time.monotonic() + 10
        while window._pending_load and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.01)
        assert not errors
        assert window.dataset.time == 1
        assert window.dataset.level == 1
        assert window.dataset.metadata.channels[0].brightness == 0.4
    finally:
        window.close()
        qapp.processEvents()
