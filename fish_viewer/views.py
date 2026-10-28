from __future__ import annotations

import numpy as np
import pyvista as pv
from PySide6 import QtCore, QtGui, QtWidgets
from pyvistaqt import QtInteractor

from .models import Dataset, composite_slice

SELECTED_COLOR = '#fff06a'

def mesh_for(surface):
    faces = np.column_stack((np.full(len(surface.triangles), 3), surface.triangles))
    return pv.PolyData(surface.vertices, faces)


def slice_segments(mesh: pv.PolyData, axis: int, position: float) -> np.ndarray:
    """Intersect actual mesh triangles with a physical coordinate plane."""
    normal = np.eye(3)[axis]
    origin = normal * position
    cut = mesh.slice(normal=normal, origin=origin)
    lines, pairs, cursor = cut.lines, [], 0
    while cursor < len(lines):
        count = int(lines[cursor])
        ids = lines[cursor + 1:cursor + count + 1]
        pairs.extend((cut.points[a], cut.points[b]) for a, b in zip(ids[:-1], ids[1:]))
        cursor += count + 1
    return np.asarray(pairs).reshape(-1, 2, 3)


def clip_segment(a, b, axis, center, half_width):
    delta = b[axis] - a[axis]
    if abs(delta) < 1e-12:
        return (a, b) if abs(a[axis] - center) <= half_width else None
    t1 = (center - half_width - a[axis]) / delta
    t2 = (center + half_width - a[axis]) / delta
    start, end = max(0, min(t1, t2)), min(1, max(t1, t2))
    return (a + start * (b - a), a + end * (b - a)) if start <= end else None


class SliceView(QtWidgets.QGraphicsView):
    stepped = QtCore.Signal(int)
    overlays_updated = QtCore.Signal()
    MAX_VOXEL_PIXELS = 256

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.setBackgroundBrush(QtGui.QColor('#080c13'))
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        self.interpolation = True
        self.setDragMode(self.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(self.ViewportAnchor.AnchorUnderMouse)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.dataset = None
        self.meshes = []
        self.axis = 0  # numpy ZYX axis
        self.index = 0
        self.selected = None
        self.selected_objects = set()
        self.surfaces_visible = self.markers_visible = self.lines_visible = True
        self._image_item = None
        self._overlays = []
        self._cuts = {}
        self._rect = QtCore.QRectF()

    def is_selected(self, kind, index):
        return self.selected == (kind, index) or (kind, index) in self.selected_objects

    def set_dataset(self, dataset: Dataset):
        self.scene().clear()
        self._image_item = None
        self._overlays = []
        self.dataset = dataset
        self.meshes = [mesh_for(s) for s in dataset.scene.surfaces]
        self._cuts.clear()
        self.index = dataset.volumes[0].shape[self.axis] // 2
        self.selected = None
        self.selected_objects = set()
        self.update_image()
        self.fit()

    @property
    def coordinate(self):
        axis_xyz = 2 - self.axis
        return self.dataset.center_origin[axis_xyz] + self.index * self.dataset.spacing[axis_xyz]

    @property
    def projection_axes(self):
        return {0: (0, 1), 1: (0, 2), 2: (1, 2)}[self.axis]

    def visible_slice_for_surface(self, index):
        """Find the nearest slice intersecting this cell's actual triangles."""
        surface = self.dataset.scene.surfaces[index]
        if not self.surfaces_visible or not surface.visible or surface.time != self.dataset.time:
            return None
        axis = 2 - self.axis
        origin, spacing = self.dataset.center_origin[axis], self.dataset.spacing[axis]
        positions = (surface.vertices[:, axis] - origin) / spacing
        first = max(0, int(np.ceil(positions.min())))
        last = min(self.dataset.volumes[0].shape[self.axis] - 1, int(np.floor(positions.max())))
        for candidate in sorted(range(first, last + 1), key=lambda value: abs(value - self.index)):
            key = (index, self.axis, candidate)
            segments = self._cuts.get(key)
            if segments is None:
                segments = slice_segments(self.meshes[index], axis, origin + candidate * spacing)
            if segments.size:
                self._cuts[key] = segments
                return candidate
        return None

    def visible_slice_for_point(self, point):
        """Keep a visible signal in place, or find the slice containing it."""
        axis = 2 - self.axis
        spacing = self.dataset.spacing[axis]
        coordinate = point[axis]
        if abs(coordinate - self.coordinate) <= spacing / 2:
            return self.index
        position = (coordinate - self.dataset.center_origin[axis]) / spacing
        last = self.dataset.volumes[0].shape[self.axis] - 1
        candidate = int(np.clip(round(position), 0, last))
        if abs(position - candidate) <= .5:
            return candidate
        return None

    def update_image(self):
        if self.dataset is None:
            return
        rgb = np.ascontiguousarray(composite_slice(self.dataset, self.axis, self.index))
        height, width = rgb.shape[:2]
        image = QtGui.QImage(rgb.data, width, height, rgb.strides[0], QtGui.QImage.Format.Format_RGB888).copy()
        if self._image_item is None:
            self._image_item = self.scene().addPixmap(QtGui.QPixmap.fromImage(image))
            self._image_item.setZValue(-10)
        else:
            self._image_item.setPixmap(QtGui.QPixmap.fromImage(image))
        u, v = self.projection_axes
        sx, sy = self.dataset.spacing[u], self.dataset.spacing[v]
        ox, oy = self.dataset.metadata.origin[u], self.dataset.metadata.origin[v]
        self._image_item.setTransform(QtGui.QTransform.fromScale(sx, sy))
        self._image_item.setTransformationMode(QtCore.Qt.TransformationMode.SmoothTransformation
                                              if self.interpolation else QtCore.Qt.TransformationMode.FastTransformation)
        self._image_item.setPos(ox, oy)
        self._rect = QtCore.QRectF(ox, oy, width * sx, height * sy)
        self.setSceneRect(self._rect.adjusted(-sx, -sy, sx, sy))
        self.update_overlays()

    def update_overlays(self):
        if self.dataset is None:
            return
        for item in self._overlays:
            self.scene().removeItem(item)
        self._overlays.clear()
        u, v = self.projection_axes
        xyz_axis = 2 - self.axis
        coordinate = self.coordinate
        half = self.dataset.spacing[xyz_axis] / 2
        if self.surfaces_visible:
            for i, (surface, mesh) in enumerate(zip(self.dataset.scene.surfaces, self.meshes)):
                if not surface.visible or surface.time != self.dataset.time:
                    continue
                key = (i, self.axis, self.index)
                if key not in self._cuts:
                    if len(self._cuts) > 500:
                        self._cuts.clear()
                    self._cuts[key] = slice_segments(mesh, xyz_axis, coordinate)
                path = QtGui.QPainterPath()
                for a, b in self._cuts[key]:
                    path.moveTo(float(a[u]), float(a[v]))
                    path.lineTo(float(b[u]), float(b[v]))
                selected = self.is_selected('surface', i)
                color = SELECTED_COLOR if selected else QtGui.QColor.fromRgbF(*surface.color)
                pen = QtGui.QPen(color)
                pen.setWidthF(4 if selected else 1.4)
                pen.setCosmetic(True)
                if selected:
                    halo = QtGui.QPen(QtGui.QColor('#000000'), 8)
                    halo.setCosmetic(True)
                    backing = self.scene().addPath(path, halo)
                    backing.setData(1, ('surface', i))
                    self._overlays.append(backing)
                item = self.scene().addPath(path, pen)
                item.setData(0, ('surface', i))
                self._overlays.append(item)
            for i, contour in enumerate(self.dataset.scene.contours):
                if not contour.visible or self.dataset.time != 0:
                    continue
                vertices = contour.vertices(self.dataset.metadata)
                path = QtGui.QPainterPath()
                for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
                    clipped = clip_segment(a, b, xyz_axis, coordinate, half)
                    if clipped is not None:
                        p, q = clipped
                        path.moveTo(float(p[u]), float(p[v]))
                        path.lineTo(float(q[u]), float(q[v]))
                selected = self.is_selected('contour', i)
                pen = QtGui.QPen(QtGui.QColor(SELECTED_COLOR) if selected else QtGui.QColor.fromRgbF(*contour.color))
                pen.setWidthF(4 if selected else 2)
                pen.setCosmetic(True)
                if selected:
                    halo = QtGui.QPen(QtGui.QColor('#000000'), 8)
                    halo.setCosmetic(True)
                    backing = self.scene().addPath(path, halo)
                    backing.setData(1, ('contour', i))
                    self._overlays.append(backing)
                else:
                    pen.setStyle(QtCore.Qt.PenStyle.DashLine)
                item = self.scene().addPath(path, pen)
                item.setData(0, ('contour', i))
                self._overlays.append(item)
        for i, measure in enumerate(self.dataset.scene.measurements):
            if not measure.visible:
                continue
            visible = [measure.all_times or t == self.dataset.time for t in measure.times]
            selected = self.is_selected('measurement', i)
            color = QtGui.QColor(SELECTED_COLOR) if selected else QtGui.QColor.fromRgbF(*measure.color)
            pen = QtGui.QPen(color)
            pen.setWidthF(4 if selected else 1.6)
            pen.setCosmetic(True)
            if self.lines_visible:
                for a, b in measure.edges:
                    if not visible[a] or not visible[b]:
                        continue
                    clipped = clip_segment(measure.points[a], measure.points[b], xyz_axis, coordinate, half)
                    if clipped is not None:
                        p, q = clipped
                        if selected:
                            halo = QtGui.QPen(QtGui.QColor('#000000'), 8)
                            halo.setCosmetic(True)
                            self._overlays.append(self.scene().addLine(float(p[u]), float(p[v]), float(q[u]), float(q[v]), halo))
                        line = self.scene().addLine(float(p[u]), float(p[v]), float(q[u]), float(q[v]), pen)
                        line.setData(0, ('measurement', i))
                        self._overlays.append(line)
            if self.markers_visible:
                for j, point in enumerate(measure.points):
                    if not visible[j] or abs(point[xyz_axis] - coordinate) > half:
                        continue
                    radius = 6 if selected else 4
                    marker = QtWidgets.QGraphicsEllipseItem(-radius, -radius, 2 * radius, 2 * radius)
                    marker.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
                    marker.setPen(pen)
                    if selected:
                        marker.setBrush(QtGui.QColor('#000000'))
                        marker.setZValue(5)
                    marker.setPos(float(point[u]), float(point[v]))
                    marker.setData(0, ('measurement', i))
                    self.scene().addItem(marker)
                    self._overlays.append(marker)
                    if self.is_selected('measurement', i):
                        from html import escape
                        label = QtWidgets.QGraphicsTextItem()
                        label.setHtml('<span style="color:#ffffff;background-color:#000000;font-weight:bold;">'
                                      + escape(measure.labels[j]) + '</span>')
                        label.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
                        label.setPos(float(point[u]), float(point[v]))
                        label.setZValue(6)
                        self.scene().addItem(label)
                        self._overlays.append(label)
        self.viewport().update()
        self.overlays_updated.emit()

    def set_interpolation(self, enabled):
        self.interpolation = bool(enabled)
        self.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, self.interpolation)
        if self._image_item is not None:
            self._image_item.setTransformationMode(QtCore.Qt.TransformationMode.SmoothTransformation
                                                  if enabled else QtCore.Qt.TransformationMode.FastTransformation)
        self.viewport().update()

    def fit(self):
        if not self._rect.isEmpty():
            self.fitInView(self._rect, QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if delta and self.dataset is not None:
                self.zoom_at(event.position().toPoint(), 1.2 if delta > 0 else 1 / 1.2)
        else:
            self.stepped.emit(1 if event.angleDelta().y() > 0 else -1)
        event.accept()

    def zoom_at(self, position, factor):
        """Keep the cursor anchored and avoid overflowing Qt's integer scrollbars."""
        current = abs(self.transform().m11())
        if current <= 0 or not np.isfinite(current):
            return
        u, v = self.projection_axes
        fit = min(self.viewport().width() / self._rect.width(),
                  self.viewport().height() / self._rect.height())
        maximum = max(fit, self.MAX_VOXEL_PIXELS / max(self.dataset.spacing[u], self.dataset.spacing[v]))
        # Qt caps scrollbar positions at INT_MAX; include absolute coordinates
        # because stage coordinates may be far from zero, even for a small scan.
        bounds = self.sceneRect()
        coordinate = max(abs(bounds.left()), abs(bounds.right()), abs(bounds.top()), abs(bounds.bottom()))
        if coordinate:
            maximum = min(maximum, 2**29 / coordinate)
        target = max(current, min(current * factor, maximum)) if factor > 1 else min(current, max(current * factor, fit * .02))
        if np.isclose(target, current, rtol=1e-12, atol=0):
            return
        before = self.mapToScene(position)
        anchor = self.transformationAnchor()
        self.setTransformationAnchor(self.ViewportAnchor.NoAnchor)
        try:
            self.scale(target / current, target / current)
            after = self.mapToScene(position)
            self.translate(after.x() - before.x(), after.y() - before.y())
        finally:
            self.setTransformationAnchor(anchor)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.dataset is None:
            return
        painter = QtGui.QPainter(self.viewport())
        painter.setPen(QtGui.QColor('#b1bfce'))
        painter.drawText(18, 28, f'{"ZYX"[self.axis]} = {self.coordinate:.3f} {self.dataset.metadata.unit}   ·   Slice {self.index + 1}')
        # Fixed-screen overlay; physical bar length follows the current zoom.
        pixels_per_unit = abs(self.transform().m11())
        if pixels_per_unit > 0:
            target = 100 / pixels_per_unit
            power = 10 ** np.floor(np.log10(max(target, 1e-12)))
            length = max(v * power for v in (0.1, 0.2, 0.5, 1, 2, 5, 10) if v * power <= target)
            width = int(length * pixels_per_unit)
            x, y = 22, self.viewport().height() - 30
            painter.setPen(QtGui.QPen(QtGui.QColor('#e6edf5'), 2))
            painter.drawLine(x, y, x + width, y)
            painter.drawLine(x, y - 4, x, y + 4)
            painter.drawLine(x + width, y - 4, x + width, y + 4)
            painter.drawText(x, y - 9, f'{length:g} {self.dataset.metadata.unit}')
        painter.end()


class VolumeView(QtWidgets.QWidget):
    def __init__(self, parent=None, off_screen=False):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(self, off_screen=off_screen, auto_update=False)
        layout.addWidget(self.plotter.interactor)
        self.plotter.set_background('#080c13')
        self.dataset = None
        self.selected = None
        self.selected_objects = set()
        self.surfaces_visible = self.markers_visible = self.lines_visible = True
        self.surface_opacity = 0.32
        self.volume_opacity = 0.25
        self.full_detail = False
        self.interpolation = True
        self.wireframe = False
        self._volume = None
        self._grid = None
        self._mesh_actors = []
        self._annotation_actors = []
        self._stride = 1

    def is_selected(self, kind, index):
        return self.selected == (kind, index) or (kind, index) in self.selected_objects

    def set_dataset(self, dataset: Dataset):
        self.dataset = dataset
        self.selected = None
        self.selected_objects = set()
        self.plotter.clear()
        self._mesh_actors = []
        self._annotation_actors = []
        self._volume = None
        self._grid = None
        self.update_volume()
        for surface in dataset.scene.surfaces:
            display_mesh = mesh_for(surface).clean()
            actor = self.plotter.add_mesh(display_mesh, color=surface.color,
                opacity=self.surface_opacity, smooth_shading=True, render=False,
                show_scalar_bar=False, pickable=True)
            self._mesh_actors.append(actor)
        self.update_annotations()
        self.plotter.add_axes()
        self.plotter.show_bounds(grid=False, location='outer', all_edges=False,
            xtitle=f'X ({dataset.metadata.unit})', ytitle=f'Y ({dataset.metadata.unit})',
            ztitle=f'Z ({dataset.metadata.unit})', font_size=9, color='#a6b5c8')
        self.plotter.view_isometric()
        self.fit()

    def update_volume(self):
        if self.dataset is None:
            return
        dataset = self.dataset
        strides = tuple(1 if self.full_detail else max(1, int(np.ceil(n / 256))) for n in dataset.volumes[0].shape)
        self._stride = strides
        selection = tuple(slice(None, None, stride) for stride in strides)
        shape = dataset.volumes[0][selection].shape
        rgb = np.zeros((*shape, 3), dtype=np.float32)
        for channel, volume in zip(dataset.metadata.channels, dataset.volumes):
            if channel.visible:
                intensity = channel.normalize(volume[selection])
                rgb += intensity[..., None] * np.asarray(channel.color, dtype=np.float32)
        np.clip(rgb, 0, 1, out=rgb)
        rgba = np.empty((*shape, 4), dtype=np.uint8)
        rgba[..., :3] = rgb * 255
        rgba[..., 3] = rgb.max(axis=-1) * 255
        # A single RGBA volume makes all channels participate in one ray integration.
        if self._grid is None or tuple(self._grid.dimensions) != shape[::-1]:
            if self._volume is not None:
                self.plotter.remove_actor(self._volume, render=False)
            self._grid = pv.ImageData(dimensions=shape[::-1],
                spacing=tuple(s * stride for s, stride in zip(dataset.spacing, strides[::-1])), origin=dataset.center_origin)
            self._grid.point_data['rgba'] = rgba.reshape(-1, 4)
            self._volume = self.plotter.add_volume(self._grid, scalars='rgba',
                opacity='linear', clim=(0, 255),
                shade=False, mapper='smart', blending='composite',
                opacity_unit_distance=min(dataset.spacing), show_scalar_bar=False,
                pickable=False, render=False, name='Image volume')
        else:
            self._grid.point_data['rgba'][:] = rgba.reshape(-1, 4)
            self._grid.GetPointData().GetScalars().Modified()
            self._grid.Modified()
        transfer = self._volume.GetProperty().GetScalarOpacity()
        transfer.RemoveAllPoints()
        transfer.AddPoint(0, 0)
        transfer.AddPoint(255, self.volume_opacity)
        self.set_interpolation(self.interpolation)
        self.plotter.render()

    def set_interpolation(self, enabled):
        self.interpolation = bool(enabled)
        if self._volume is not None:
            prop = self._volume.GetProperty()
            if enabled:
                prop.SetInterpolationTypeToLinear()
            else:
                prop.SetInterpolationTypeToNearest()
            self.plotter.render()

    def update_annotations(self):
        if self.dataset is None:
            return
        for i, (actor, surface) in enumerate(zip(self._mesh_actors, self.dataset.scene.surfaces)):
            actor.SetVisibility(self.surfaces_visible and surface.visible and surface.time == self.dataset.time)
            prop = actor.GetProperty()
            selected = self.is_selected('surface', i)
            prop.SetOpacity(max(.6, self.surface_opacity) if selected else self.surface_opacity)
            prop.SetColor((1., .94, .42) if selected else surface.color)
            if self.wireframe:
                prop.SetRepresentationToWireframe()
            else:
                prop.SetRepresentationToSurface()
        for actor in self._annotation_actors:
            self.plotter.remove_actor(actor, render=False)
        self._annotation_actors = []
        if self.surfaces_visible and self.dataset.time == 0:
            for i, contour in enumerate(self.dataset.scene.contours):
                if contour.visible:
                    curve = pv.lines_from_points(contour.vertices(self.dataset.metadata), close=True)
                    actor = self.plotter.add_mesh(curve, color=SELECTED_COLOR if self.is_selected('contour', i) else contour.color,
                        line_width=5 if self.is_selected('contour', i) else 2, render=False, reset_camera=False)
                    self._annotation_actors.append(actor)
        for i, measure in enumerate(self.dataset.scene.measurements):
            if not measure.visible:
                continue
            selected = self.is_selected('measurement', i)
            color = SELECTED_COLOR if selected else measure.color
            visible = np.asarray([measure.all_times or t == self.dataset.time for t in measure.times])
            if self.markers_visible and visible.any():
                actor = self.plotter.add_points(measure.points[visible], color=color, point_size=16 if selected else 9,
                    render_points_as_spheres=True, render=False, show_scalar_bar=False, reset_camera=False)
                self._annotation_actors.append(actor)
                if self.is_selected('measurement', i) and measure.labels == ['A', 'B']:
                    actor = self.plotter.add_point_labels(measure.points[visible],
                        [text for text, shown in zip(measure.labels, visible) if shown],
                        font_size=15, text_color='#e6fff8', show_points=False,
                        shape_color='#102521', shape_opacity=.8, always_visible=True,
                        reset_camera=False, render=False)
                    self._annotation_actors.append(actor)
            if self.lines_visible:
                pairs = [(a, b) for a, b in measure.edges if visible[a] and visible[b]]
                if pairs:
                    segments = np.asarray([[measure.points[a], measure.points[b]] for a, b in pairs]).reshape(-1, 3)
                    actor = self.plotter.add_lines(segments, color=color, width=5 if selected else 2, connected=False)
                    self._annotation_actors.append(actor)
                    if self.is_selected('measurement', i) and measure.labels == ['A', 'B']:
                        # A/B can be closer together on screen than the width of
                        # a distance label. Keep the value away from their markers.
                        actor = self.plotter.add_text(f'A-B: {measure.length:.3f} {self.dataset.metadata.unit}',
                            position='upper_left', font_size=11, color='#e6fff8',
                            name='selected-ab-distance', render=False)
                        self._annotation_actors.append(actor)
        self.plotter.render()

    def fit(self):
        if self.dataset:
            origin = self.dataset.metadata.origin
            extent = self.dataset.metadata.extent
            self.plotter.reset_camera(bounds=tuple(v for o, e in zip(origin, extent) for v in (o, o + e)))

    def closeEvent(self, event):
        self.plotter.close()
        super().closeEvent(event)
