"""Annotation selection and calibrated contour-editing tools."""

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from shiboken6 import isValid

from ..views import mesh_for, slice_segments
from .editor_controls import EditingControls
from .editor_interaction import EditingInteraction
from .annotations import AnnotationDocument, origin


class Handle(QtWidgets.QGraphicsEllipseItem):
    HIT_RADIUS = 12  # Screen pixels; the visible dot remains 8 pixels across.

    def __init__(self, editor, vertex, xyz):
        super().__init__(-4, -4, 8, 8)
        self.editor, self.vertex, self.xyz = editor, vertex, np.array(xyz)
        self.setFlags(
            self.GraphicsItemFlag.ItemIsMovable
            | self.GraphicsItemFlag.ItemIgnoresTransformations
        )
        anchor = (2 - editor.window.slice_view.axis, *np.round(xyz, 7))
        self.setBrush(QtGui.QColor("#fff06a" if anchor in editor.point_selection else "#ffffff"))
        self.setPen(QtGui.QPen(QtGui.QColor("#205e58")))
        self.setZValue(20)
        self.setAcceptedMouseButtons(QtCore.Qt.MouseButton.LeftButton)
        self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        u, v = editor.window.slice_view.projection_axes
        self.setPos(float(xyz[u]), float(xyz[v]))
        self.setToolTip(
            f"Point {vertex + 1}: drag to move; select then Delete point"
            if vertex is not None
            else "Slice contour: drag to move this border point"
        )

    def boundingRect(self):
        radius = self.HIT_RADIUS
        return QtCore.QRectF(-radius, -radius, 2 * radius, 2 * radius)

    def shape(self):
        path = QtGui.QPainterPath()
        path.addEllipse(self.boundingRect())
        return path

    def mousePressEvent(self, event):
        self.selection = self.editor.selection
        self.document = self.editor.document
        self.scene_state = self.document.dataset.scene
        self.axis = 2 - self.editor.window.slice_view.axis
        self.drag_start = event.scenePos()
        self.initial_position = self.pos()
        anchor = (self.axis, *np.round(self.xyz, 7))
        self.editor.choose_point(self, anchor in self.editor.point_selection or bool(
            event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier), redraw=False)
        self.anchors = list(self.editor.point_selection)
        self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
        event.accept()

    def mouseMoveEvent(self, event):
        if self.editor.mode == 'select' or self.editor.mode == 'add':
            event.accept()
            return
        # Use scene coordinates explicitly so zoom and image panning cannot
        # turn a point drag into a view drag. The slice coordinate stays fixed.
        self.setPos(self.initial_position + event.scenePos() - self.drag_start)
        xyz = self.xyz.copy()
        u, v = self.editor.window.slice_view.projection_axes
        xyz[u], xyz[v] = self.pos().x(), self.pos().y()
        self.editor.preview_point(self.selection, self.vertex, self.xyz, xyz, self.anchors)
        event.accept()

    def mouseReleaseEvent(self, event):
        self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        event.accept()
        u, v = self.editor.window.slice_view.projection_axes
        xyz = self.xyz.copy()
        xyz[u] = self.pos().x()
        xyz[v] = self.pos().y()
        if not np.array_equal(xyz, self.xyz):
            # Rebuilding overlays must not destroy this item during its own event.
            editor, selection = self.editor, self.selection
            original, axis, anchors = self.xyz.copy(), self.axis, list(self.anchors)
            document, scene = self.document, self.scene_state
            def commit():
                # Undo, deletion or switching scans can replace this scene before
                # the queued edit runs; its old index may now identify another cell.
                if (editor.document is document and document.dataset.scene is scene
                        and editor.selection == selection):
                    editor.commit_handle(selection, original, xyz, axis, anchors)
            QtCore.QTimer.singleShot(0, commit)


class AnnotationEditor(EditingControls, EditingInteraction, QtWidgets.QWidget):
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.document = None
        self.selection = None
        self.handles = []
        self.initialize_interaction()
        self.influence_item = None
        self.build_controls()
        self.shortcut = QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Undo, window)
        self.shortcut.activated.connect(
            lambda: (
                self.document.undo.undo()
                if self.document and not window.running and not window.loading
                else None
            )
        )
        self.delete_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key.Key_Delete), window
        )
        self.delete_shortcut.setParent(self.tree)
        self.delete_shortcut.setContext(
            QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut
        )
        self.delete_shortcut.activated.connect(self.delete)
        self.image_delete_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key.Key_Delete), window.slice_view
        )
        self.image_delete_shortcut.setContext(
            QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut
        )
        self.image_delete_shortcut.activated.connect(self.delete)
        window.slice_view.viewport().installEventFilter(self)
        window.slice_view.installEventFilter(self)
        self.redo_shortcut = QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Redo, window)
        self.redo_shortcut.activated.connect(lambda: self.document.undo.redo() if self.document and not window.running and not window.loading else None)
        self.update_controls()
        window.slice_view.overlays_updated.connect(self.draw_handles)

    def set_dataset(self, dataset, report, cached=None):
        self.reset_interaction()
        self.simplification_report.clear()
        self.simplification_report.hide()
        self.warning.setText(
            "\n".join(w for w in dataset.scene.warnings if "no saved" not in w.lower())
        )
        self.warning.setVisible(bool(self.warning.text()))
        self.document = cached or AnnotationDocument(dataset, report, self.changed)
        self.document.dataset = dataset
        self.document.changed = self.changed
        if not cached and 'signal_pairing' not in self.document.report_template:
            self.document.pairing_max_distance = self.window.pair_distance.value()
            self.document.recalculate()
        self.document.undo.canUndoChanged.connect(self.undo_button.setEnabled)
        self.signal_channel.clear()
        for i, c in enumerate(dataset.metadata.channels):
            self.signal_channel.addItem(f"New signal: channel {i + 1} · {c.name}", i)
        roles = self.window.channel_roles.get(str(self.window._source_image), {})
        signals = roles.get("signals")
        if signals:
            self.signal_channel.setCurrentIndex(
                self.signal_channel.findData(signals[0])
            )
        self.changed()

    def changed(self):
        w = self.window
        view = w.slice_view
        w.pair_distance.blockSignals(True)
        w.pair_distance.setValue(self.document.pairing_max_distance)
        w.pair_distance.blockSignals(False)
        self.selections = {key for key in self.selections if self.annotation_object(key) is not None}
        if self.selection not in self.selections:
            self.selection = next(iter(sorted(self.selections)), None)
            self.point_selection.clear()
            self.active_point = None
        if self.mode in ('precise', 'proportional') and not self.single_contour():
            self.set_tool_mode('move' if self.selections else 'select')
        view.meshes = [mesh_for(s) for s in w.dataset.scene.surfaces]
        view._cuts.clear()
        self.tree.blockSignals(True)
        self.tree.clear()
        for category in ("Manual", "Program"):
            parent = QtWidgets.QTreeWidgetItem(["Pre-existing" if category == "Manual" else "This run"])
            self.tree.addTopLevelItem(parent)
            parent.setExpanded(True)
            for kind, objects in [
                ("surface", w.dataset.scene.surfaces),
                ("contour", w.dataset.scene.contours),
                ("measurement", w.dataset.scene.measurements),
            ]:
                for i, obj in enumerate(objects):
                    if origin(obj) != category:
                        continue
                    value = (
                        self.document.volumes[i]
                        if kind == "surface"
                        else self.document.distances.get(i)
                        if kind == "measurement"
                        else None
                    )
                    label = (
                        f"{value:.4f} " + ("µm³" if kind == "surface" else "µm")
                        if value is not None
                        else ""
                    )
                    item = QtWidgets.QTreeWidgetItem([obj.name, label])
                    item.setData(0, QtCore.Qt.ItemDataRole.UserRole, (kind, i))
                    parent.addChild(item)
                    if (kind, i) in self.selections:
                        item.setSelected(True)
                    if (kind, i) == self.selection:
                        self.tree.setCurrentItem(item, 0, QtCore.QItemSelectionModel.SelectionFlag.NoUpdate)
        self.tree.blockSignals(False)
        view.selected = self.selection
        view.selected_objects = self.selections.copy()
        if w.volume_view and w._volume_dataset is w.dataset:
            camera = w.volume_view.plotter.camera_position
            w.volume_view.set_dataset(w.dataset)
            w.volume_view.plotter.camera_position = camera
            w.volume_view.selected = self.selection
            w.volume_view.selected_objects = self.selections.copy()
        self.visibility()
        self.coordinates()
        self.undo_button.setEnabled(self.document.undo.canUndo())
        w.excel_button.setEnabled(not w.loading and not w.running)
        self.update_tables()
        self.update_controls()

    def update_tables(self):
        w = self.window
        report = self.document.report()
        w.distances.blockSignals(True)
        w.counts.blockSignals(True)
        w.distances.setRowCount(0)
        w.counts.setRowCount(0)
        for source, points in [
            ("Pre-existing", report["manual_measurements"]["points"]),
            ("This run", report["points"]),
        ]:
            for p in points:
                row = w.distances.rowCount()
                w.distances.insertRow(row)
                values = [
                    source + " · " + p.get("signal_name", p["signal_id"]),
                    p.get("cell_name", p.get("cell_id")),
                    p["channel"] + 1 if p["channel"] is not None else "",
                    p.get("distance_um"),
                    p.get("score"),
                    *p["xyz"],
                    *p.get("border_xyz", ["", "", ""]),
                    "",
                ]
                for j, value in enumerate(values):
                    item = QtWidgets.QTableWidgetItem(
                        ""
                        if value is None
                        else f"{value:.4f}"
                        if isinstance(value, float)
                        else str(value)
                    )
                    if j == 0:
                        item.setData(
                            QtCore.Qt.ItemDataRole.UserRole,
                            p,
                        )
                    w.distances.setItem(row, j, item)
        for pair in report['signal_pairs']:
            row = w.distances.rowCount()
            w.distances.insertRow(row)
            values = [f"Signal pair · {pair['signal_name_a']} ↔ {pair['signal_name_b']}",
                      pair['cell_name'], f"{pair['channel_a'] + 1} ↔ {pair['channel_b'] + 1}",
                      pair['distance_um'], '', *pair['xyz_a'], *pair['xyz_b'], 'Signal ↔ signal']
            for j, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(f'{value:.4f}' if isinstance(value, float) else str(value))
                if j == 0:
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, dict(
                        annotation_id=pair['pair_id'], signal_id=pair['pair_id'], signal_name=values[0],
                        cell_id=pair['cell_id'], xyz=pair['xyz_a'], border_xyz=pair['xyz_b'], distance_um=pair['distance_um']))
                w.distances.setItem(row, j, item)
        for source, cells in [
            ("Pre-existing", report["manual_measurements"]["cells"]),
            ("This run", report["cell_volumes"]),
        ]:
            for cell in cells:
                row = w.counts.rowCount()
                w.counts.insertRow(row)
                count = sum(
                    p.get("cell_id") == cell["cell_id"]
                    for points in (
                        report["points"],
                        report["manual_measurements"]["points"],
                    )
                    for p in points
                )
                for j, value in enumerate(
                    [
                        source + " · " + cell.get("cell_name", cell["cell_id"]),
                        "",
                        "",
                        count,
                        cell["volume_um3"],
                    ]
                ):
                    w.counts.setItem(
                        row,
                        j,
                        QtWidgets.QTableWidgetItem(
                            ""
                            if value is None
                            else f"{value:.4f}"
                            if isinstance(value, float)
                            else str(value)
                        ),
                    )
                    if j == 0:
                        w.counts.item(row, j).setData(QtCore.Qt.ItemDataRole.UserRole, cell['cell_id'])
        w.distances.blockSignals(False)
        w.counts.blockSignals(False)
        w.update_results_overview(report)

    def visibility(self):
        if not self.document:
            return
        for objects in (
            self.window.dataset.scene.surfaces,
            self.window.dataset.scene.measurements,
            self.window.dataset.scene.contours,
        ):
            for obj in objects:
                obj.visible = (
                    self.manual.isChecked()
                    if origin(obj) == "Manual"
                    else self.program.isChecked()
                )
        self.window.refresh_annotations()

    def select_annotation(self, selection):
        self.set_selection({selection}, selection, reveal=True)

    def select(self, item, previous=None):
        selection = item.data(0, QtCore.Qt.ItemDataRole.UserRole) if item else None
        self.set_selection({selection} if selection else set(), selection, reveal=True)

    def reveal_selected_annotation(self):
        obj = self.annotation_object(self.selection)
        if obj is None or self.window.mode.currentIndex() != 0:
            return
        kind, index = self.selection
        view = self.window.slice_view
        if kind == 'surface':
            target = view.visible_slice_for_surface(index)
        elif kind == 'measurement':
            measurement = obj
            if not measurement.visible or not (view.markers_visible or view.lines_visible and measurement.edges):
                return
            signal = self.document.signal_index(measurement)
            if signal is None:
                signal = next((i for i, time in enumerate(measurement.times)
                               if measurement.all_times or time == self.window.dataset.time), None)
            if signal is None or not (measurement.all_times or measurement.times[signal] == self.window.dataset.time):
                return
            target = view.visible_slice_for_point(measurement.points[signal])
        else:
            return
        if target is not None:
            self.window.slice.setValue(target)

    def annotation_object(self, selection):
        """Resolve only objects in the currently active document and dataset."""
        if (selection is None or self.document is None
                or self.document.dataset is not self.window.dataset):
            return None
        kind, index = selection
        name = {'surface': 'surfaces', 'contour': 'contours', 'measurement': 'measurements'}.get(kind)
        if name is None:
            return None
        objects = getattr(self.document.dataset.scene, name)
        return objects[index] if 0 <= index < len(objects) else None

    def points(self):
        obj = self.annotation_object(self.selection)
        if obj is None or getattr(obj, 'kind', None) == 'signal_pair':
            return np.empty((0, 3))
        kind, _ = self.selection
        return (
            obj.vertices
            if kind == "surface"
            else obj.vertices(self.window.dataset.metadata)
            if kind == "contour"
            else obj.points
        )

    def coordinates(self):
        points = self.points()
        self.vertex.blockSignals(True)
        self.vertex.setMaximum(max(1, len(points)))
        self.vertex.blockSignals(False)
        if len(points):
            xyz = self.active_point[1:] if self.active_point is not None else points[self.vertex.value() - 1]
            for spin, value in zip(self.xyz, xyz):
                spin.setValue(float(value))

    def draw_handles(self):
        self.draw_draft()
        if self.influence_item is not None and isValid(self.influence_item) and self.influence_item.scene() is not None:
            self.influence_item.scene().removeItem(self.influence_item)
        self.influence_item = None
        if self.mode == 'proportional' and self.active_point is not None:
            view = self.window.slice_view
            axis = 2 - view.axis
            point = np.asarray(self.active_point[1:])
            squared = self.radius.value() ** 2 - (point[axis] - view.coordinate) ** 2
            if squared > 0:
                radius = np.sqrt(squared)
                u, v = view.projection_axes
                pen = QtGui.QPen(QtGui.QColor('#fff06a'), 1, QtCore.Qt.PenStyle.DashLine)
                pen.setCosmetic(True)
                self.influence_item = view.scene().addEllipse(float(point[u] - radius), float(point[v] - radius),
                    float(2 * radius), float(2 * radius), pen)
                self.influence_item.setZValue(8)
        for item in self.handles:
            if isValid(item) and item.scene() is not None:
                item.scene().removeItem(item)
        self.handles = []
        if (not self.document or not self.edit.isChecked() or not self.selection or len(self.selections) != 1
                or not (self.mode == 'select' or self.precise_active() or self.mode == 'move'
                        or self.mode == 'add' and self.tool.currentIndex() == 1)):
            return
        view = self.window.slice_view
        obj = self.annotation_object(self.selection)
        if obj is None:
            return
        points = self.points()
        axis = 2 - view.axis
        kind, index = self.selection
        if not obj.visible:
            return
        if kind != "measurement" and not view.surfaces_visible:
            return
        if kind == "surface":
            if obj.time != self.window.dataset.time:
                return
            segments = view._cuts.get(
                (self.selection[1], view.axis, view.index), np.empty((0, 2, 3))
            )
            section = np.unique(np.round(segments.reshape(-1, 3), 7), axis=0)
            for xyz in section[:: max(1, int(np.ceil(len(section) / 400)))]:
                # Only the actual intersection with this plane is editable.
                # Nearby mesh vertices project to the same screen locations
                # but belong to other layers and must not cover these handles.
                distances = np.linalg.norm(points - xyz, axis=1)
                nearest = int(np.argmin(distances)) if len(distances) else None
                vertex = (
                    nearest
                    if nearest is not None and distances[nearest] < 1e-7
                    else None
                )
                xyz[axis] = view.coordinate
                item = Handle(self, vertex, xyz)
                view.scene().addItem(item)
                self.handles.append(item)
            return
        if kind == "contour":
            if obj.axis != axis or self.window.dataset.time != 0:
                return
            ids = np.flatnonzero(abs(points[:, axis] - view.coordinate) < 1e-7)
        else:
            if not view.markers_visible:
                return
            ids = np.flatnonzero(
                abs(points[:, axis] - view.coordinate)
                <= self.window.dataset.spacing[axis] / 2
            )
        for i in ids:
            if kind == "measurement" and (
                obj.labels[i].upper() == "B"
                or (not obj.all_times and obj.times[i] != self.window.dataset.time)
            ):
                continue
            item = Handle(self, int(i), points[i])
            view.scene().addItem(item)
            self.handles.append(item)

    def perform(self, callback):
        if not self.document or self.window.running or self.window.loading:
            return False
        try:
            callback()
            return True
        except (ValueError, IndexError, RuntimeError) as exc:
            self.changed()
            self.window.error(str(exc))
            return False

    def preview_point(self, selection, vertex, original, xyz, anchors=None):
        obj = self.annotation_object(selection)
        if obj is None:
            return
        view = self.window.slice_view
        kind, index = selection
        if kind == "surface":
            from .tools import deform_surface
            anchors = anchors or [(2 - view.axis, *original)]
            preview = deform_surface(self.document, obj, anchors,
                                     np.asarray(xyz) - original,
                                     self.radius.value() if self.mode == 'proportional' else None,
                                     self.falloff.currentText().lower())
            segments = slice_segments(mesh_for(preview), 2 - view.axis, view.coordinate)
        elif kind == "contour":
            points = self.points().copy()
            anchors = np.asarray([a[1:] for a in anchors]) if anchors else np.asarray([original])
            ids = np.unique(np.argmin(np.linalg.norm(points[:, None] - anchors, axis=2), axis=0))
            points[ids] += xyz - original
            segments = np.stack([points, np.roll(points, -1, axis=0)], axis=1)
        else:
            return
        path = QtGui.QPainterPath()
        u, v = view.projection_axes
        for a, b in segments:
            path.moveTo(float(a[u]), float(a[v]))
            path.lineTo(float(b[u]), float(b[v]))
        for item in view._overlays:
            if (item.data(0) == selection or item.data(1) == selection) and isinstance(item, QtWidgets.QGraphicsPathItem):
                item.setPath(path)

    def delete(self):
        if self.point_selection and self.single_contour():
            self.delete_point()
        else:
            self.delete_objects()

    def delete_point(self):
        if not self.selection:
            return
        if self.selection[0] == 'measurement':
            self.delete_objects()
        elif self.selection[0] == 'surface' and self.point_selection:
            selected = list(self.point_selection)
            self.perform(lambda: self.document.delete_section_points(self.selection[1], selected))
            self.point_selection.clear()
            self.active_point = None
            self.window.refresh_annotations()
        elif self.selection[0] == 'contour' and self.point_selection:
            points = self.points()
            ids = {int(np.argmin(np.linalg.norm(points - a[1:], axis=1))) for a in self.point_selection}
            def change():
                for index in sorted(ids, reverse=True):
                    self.document.delete_vertex(*self.selection, index)
            if self.perform(lambda: self.document.batch('Delete contour points', change)):
                self.point_selection.clear()
                self.active_point = None
                self.window.refresh_annotations()
        else:
            self.perform(lambda: self.document.delete_vertex(*self.selection, self.vertex.value() - 1))

    def move_point(self, index, xyz):
        if self.selection:
            self.perform(lambda: self.document.move(*self.selection, index, xyz))

    def eventFilter(self, obj, event):
        if self.interaction_event(obj, event):
            return True
        return super().eventFilter(obj, event)

    def export_scene(self):
        if not self.document:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.window, "Export current annotations", "", "Imaris image (*.ims)"
        )
        if path:
            from ..workflow.edit_export import export_document

            self.window.loading = True
            self.window.sync_buttons()

            def done(result):
                self.window.loading = False
                self.window.sync_buttons()
                self.window.phase.setText("Edited annotations exported: " + result)

            def failed(message):
                self.window.loading = False
                self.window.sync_buttons()
                self.window.error(message)

            document = self.document
            self.window.background(
                lambda _: export_document(document, path), done, failed
            )
