"""Visible-layer picking, multi-selection and drawing gestures."""

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from shiboken6 import isValid


class EditingInteraction:
    def initialize_interaction(self):
        self.selections = set()
        self.point_selection = set()
        self.active_point = None
        self.gesture = None
        self.space_pan = False
        self.rubber = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Shape.Rectangle, self.window.slice_view.viewport())
        self.draft_layers, self.draft_points, self.draft_items = {}, [], []
        self.draft_axis = self.draft_coordinate = self.draft_time = None

    def reset_interaction(self):
        """Detach the old scene's tools before a dataset change can redraw."""
        self.selection = None
        self.selections.clear()
        self.point_selection.clear()
        self.active_point = None
        self.gesture = None
        self.space_pan = False
        self.rubber.hide()
        for view in (self.window.slice_view, self.window.volume_view):
            if view is not None:
                view.selected, view.selected_objects = None, set()
        self.cancel_draft()
        self.set_tool_mode('select')

    def tree_selection(self):
        keys = {item.data(0, QtCore.Qt.ItemDataRole.UserRole) for item in self.tree.selectedItems()}
        keys.discard(None)
        item = self.tree.currentItem()
        primary = item.data(0, QtCore.Qt.ItemDataRole.UserRole) if item else None
        self.set_selection(keys, primary, reveal=True)

    def set_selection(self, keys, primary=None, reveal=False):
        keys = {key for key in keys if self.annotation_object(key) is not None}
        if keys != self.selections:
            self.point_selection.clear()
            self.active_point = None
        self.selections = keys
        self.selection = primary if primary in keys else next(iter(sorted(keys)), None)
        self.tree.blockSignals(True)
        self.tree.clearSelection()
        for n in range(self.tree.topLevelItemCount()):
            parent = self.tree.topLevelItem(n)
            for j in range(parent.childCount()):
                item = parent.child(j)
                key = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
                item.setSelected(key in keys)
                if key == self.selection:
                    self.tree.setCurrentItem(item, 0, QtCore.QItemSelectionModel.SelectionFlag.NoUpdate)
        self.tree.blockSignals(False)
        for view in (self.window.slice_view, self.window.volume_view):
            if view is not None:
                view.selected, view.selected_objects = self.selection, keys.copy()
        if self.mode in ('precise', 'proportional') and not self.single_contour():
            self.set_tool_mode('move' if keys else 'select')
        self.update_controls()
        if reveal:
            self.reveal_selected_annotation()
        self.coordinates()
        self.window.refresh_annotations()

    def choose_point(self, handle, additive=False, redraw=True):
        axis = 2 - self.window.slice_view.axis
        anchor = (axis, *np.round(handle.xyz, 7))
        self.point_selection = self.point_selection | {anchor} if additive else {anchor}
        self.active_point = anchor
        if handle.vertex is not None:
            self.vertex.setValue(handle.vertex + 1)
        for item in self.handles:
            chosen = (axis, *np.round(item.xyz, 7)) in self.point_selection
            item.setBrush(QtGui.QColor('#fff06a' if chosen else '#ffffff'))
        self.update_controls()
        self.coordinates()
        if redraw:
            self.window.refresh_annotations()

    def commit_handle(self, selection, original, xyz, axis, anchors):
        if self.annotation_object(selection) is None:
            return
        if selection[0] == 'surface':
            radius = self.radius.value() if self.mode == 'proportional' else None
            if not self.perform(lambda: self.document.move_section_points(selection[1], anchors,
                np.asarray(xyz) - original, radius, self.falloff.currentText().lower())):
                return
        elif selection[0] == 'measurement':
            obj = self.window.dataset.scene.measurements[selection[1]]
            index = self.document.signal_index(obj)
            if index is not None:
                self.perform(lambda: self.document.move('measurement', selection[1], index, xyz))
        else:
            points = self.window.dataset.scene.contours[selection[1]].vertices(self.window.dataset.metadata)
            ids = {int(np.argmin(np.linalg.norm(points - a[1:], axis=1))) for a in anchors}
            positions = {i: points[i] + xyz - original for i in ids}
            def change():
                for i, position in positions.items():
                    self.document.move('contour', selection[1], i, position)
            if not self.perform(lambda: self.document.batch('Move contour points', change)):
                return
        self.point_selection = {(a[0], *np.round(np.asarray(a[1:]) + xyz - original, 7)) for a in anchors}
        self.active_point = (axis, *np.round(xyz, 7))
        self.window.refresh_annotations()

    def apply_point_position(self, xyz):
        if not self.selection:
            return
        if self.active_point is not None:
            old = np.asarray(self.active_point[1:])
            self.commit_handle(self.selection, old, np.asarray(xyz), self.active_point[0], list(self.point_selection))
        else:
            self.move_point(self.vertex.value() - 1, xyz)

    def delete_objects(self):
        if self.selections:
            selected = self.selections.copy()
            self.set_selection(set())
            self.perform(lambda: self.document.delete_many(selected))

    def simplify_selected(self):
        selected = {(kind, i) for kind, i in self.selections if kind in ('surface', 'contour')}
        if not selected:
            return
        result = {}
        if not self.perform(lambda: result.update(self.document.simplify(
                selected, self.reduction.value() / 100, self.volume_tolerance.value() / 100))):
            return
        summary = f"Simplified {len(result['simplified'])}; skipped {len(result['skipped'])}."
        lines = [summary]
        for category, title in [('simplified', 'Simplified'), ('skipped', 'Skipped')]:
            if not result[category]:
                continue
            lines.extend(['', title + ':'])
            for entry in result[category]:
                label = 'Cell' if entry['kind'] == 'surface' else 'Contour'
                detail = (f"{entry['before_points']} → {entry['after_points']} points"
                          if category == 'simplified' else entry['reason'])
                lines.append(f"{label} {entry['index'] + 1} · {entry['name']}: {detail}")
        self.simplification_report.setPlainText('\n'.join(lines))
        self.simplification_report.show()
        if result['simplified']:
            self.point_selection.clear()
            self.active_point = None
            self.info.setText(summary + ' Volumes and distances recalculated. Undo restores their original geometry.')
        else:
            self.info.setText(summary + ' No geometry changed.')
        self.draw_handles()

    def draw_draft(self):
        view = self.window.slice_view
        for item in self.draft_items:
            if isValid(item) and item.scene() is not None:
                item.scene().removeItem(item)
        self.draft_items = []
        if not self.document or self.draft_axis != 2 - view.axis or self.draft_time != self.window.dataset.time:
            return
        paths = [(points, True) for coordinate, points in self.draft_layers.items()
                 if abs(coordinate - view.coordinate) < 1e-6]
        if self.draft_points and abs(self.draft_coordinate - view.coordinate) < 1e-6:
            paths.append((self.draft_points, False))
        for points, closed in paths:
            path = QtGui.QPainterPath()
            path.moveTo(*map(float, points[0]))
            for point in points[1:]:
                path.lineTo(*map(float, point))
            if closed:
                path.closeSubpath()
            pen = QtGui.QPen(QtGui.QColor('#75dfff'), 2)
            pen.setCosmetic(True)
            item = view.scene().addPath(path, pen)
            item.setZValue(15)
            self.draft_items.append(item)

    def add_draft_point(self, xyz):
        view = self.window.slice_view
        axis = 2 - view.axis
        if self.draft_axis is not None and (self.draft_axis != axis or self.draft_time != self.window.dataset.time):
            raise ValueError('Finish or discard this cell before changing drawing orientation or time point.')
        self.draft_axis, self.draft_time = axis, self.window.dataset.time
        if self.draft_points and abs(self.draft_coordinate - view.coordinate) > 1e-6:
            self.finish_layer()
            if self.draft_points:
                raise ValueError('Close or discard the unfinished layer before drawing on another slice.')
        self.draft_coordinate = view.coordinate
        point = np.asarray(xyz)[list(view.projection_axes)]
        if not self.draft_points or np.linalg.norm(point - self.draft_points[-1]) > 1e-7:
            self.draft_points.append(point)
        self.draft_info.setText(f'{len(self.draft_layers)} closed layers · {len(self.draft_points)} points on this layer')
        self.draw_draft()

    def finish_layer(self):
        if not self.draft_points:
            return
        if len(self.draft_points) < 3:
            self.info.setText('A layer needs at least three points.')
            return
        points = np.asarray(self.draft_points)
        area = abs(np.dot(points[:, 0], np.roll(points[:, 1], 1)) - np.dot(points[:, 1], np.roll(points[:, 0], 1))) / 2
        if area <= 1e-10:
            self.info.setText('Draw a polygon with a nonzero area.')
            return
        self.draft_layers[self.draft_coordinate] = points.copy()
        self.draft_points = []
        self.draft_info.setText(f'{len(self.draft_layers)} closed layers')
        self.draw_draft()

    def finish_cell(self):
        self.finish_layer()
        if self.draft_points or not self.draft_layers:
            self.info.setText('Close a valid polygon layer first.')
            return
        before = len(self.window.dataset.scene.surfaces)
        self.perform(lambda: self.document.add_layered_cell(self.draft_layers, self.draft_axis))
        if len(self.window.dataset.scene.surfaces) > before:
            self.cancel_draft()
            self.set_selection({('surface', before)}, ('surface', before))
            self.set_tool_mode('precise')

    def cancel_draft(self):
        self.draft_layers, self.draft_points = {}, []
        self.draft_axis = self.draft_coordinate = self.draft_time = None
        self.draft_info.setText('No layers drawn')
        self.draw_draft()

    def object_keys_at(self, area):
        keys = set()
        for item in self.window.slice_view.items(area):
            key = item.data(0)
            if isinstance(key, tuple) and len(key) == 2 and key[0] in ('surface', 'measurement', 'contour'):
                keys.add(key)
        return keys

    def interaction_event(self, obj, event):
        from .editor import Handle
        view = self.window.slice_view
        kind = event.type()
        if kind == QtCore.QEvent.Type.KeyPress and event.key() == QtCore.Qt.Key.Key_Space:
            self.space_pan = True
            return True
        if kind == QtCore.QEvent.Type.KeyRelease and event.key() == QtCore.Qt.Key.Key_Space:
            self.space_pan = False
            return True
        if kind == QtCore.QEvent.Type.KeyPress and event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
            if self.mode == 'add' and self.tool.currentIndex() == 0:
                self.finish_layer()
                return True
        if kind == QtCore.QEvent.Type.KeyPress and event.key() == QtCore.Qt.Key.Key_Escape:
            self.cancel_draft()
            self.rubber.hide()
            self.gesture = None
            return True
        if not self.document or self.window.running or self.window.loading:
            return False
        if kind not in (QtCore.QEvent.Type.MouseButtonPress, QtCore.QEvent.Type.MouseMove,
                        QtCore.QEvent.Type.MouseButtonRelease, QtCore.QEvent.Type.MouseButtonDblClick):
            return False
        position = event.position().toPoint()
        scene_pos = view.mapToScene(position)
        u, v = view.projection_axes
        xyz = np.zeros(3)
        xyz[u], xyz[v], xyz[2 - view.axis] = scene_pos.x(), scene_pos.y(), view.coordinate
        if kind == QtCore.QEvent.Type.MouseButtonPress:
            view.setFocus()
            if event.button() == QtCore.Qt.MouseButton.MiddleButton or self.space_pan:
                self.gesture = ('pan', position, view.horizontalScrollBar().value(), view.verticalScrollBar().value())
                return True
            if event.button() != QtCore.Qt.MouseButton.LeftButton:
                return False
            additive = bool(event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier)
            handles = [item for item in view.items(position) if isinstance(item, Handle)]
            if handles and (self.mode == 'select' or self.precise_active() or self.mode == 'move'):
                closest = min(handles, key=lambda item: (view.mapFromScene(item.pos()) - position).manhattanLength())
                if self.mode == 'select':
                    self.choose_point(closest, additive)
                    return True
                for handle in self.handles:
                    handle.setZValue(21 if handle is closest else 20)
                return False
            keys = self.object_keys_at(position)
            if self.mode == 'add':
                if self.tool.currentIndex() == 2:
                    self.perform(lambda: self.document.add_signal(xyz, self.signal_channel.currentData()))
                elif self.tool.currentIndex() == 1:
                    if self.single_contour():
                        self.add_contour_point(xyz, position)
                    else:
                        self.info.setText('Select one cell contour before adding a point.')
                else:
                    self.perform(lambda: self.add_draft_point(xyz))
                return True
            if self.mode == 'move' and not self.single_contour() and keys:
                if not (keys & self.selections):
                    self.set_selection(keys | self.selections if additive else {next(iter(keys))})
                moved = self.selections.copy()
                cells = {i for k, i in moved if k == 'surface'}
                moved |= {('measurement', i) for i, c in self.document.bindings.items() if c in cells}
                items = [(item, item.pos()) for item in view._overlays if item.data(0) in moved or item.data(1) in moved]
                self.gesture = ('move', xyz, self.selections.copy(), items)
                return True
            self.gesture = ('select', position, additive, self.selections.copy())
            self.rubber.setGeometry(QtCore.QRect(position, QtCore.QSize()))
            return True
        if kind == QtCore.QEvent.Type.MouseButtonDblClick and self.mode == 'add' and self.tool.currentIndex() == 0:
            self.finish_layer()
            return True
        if self.gesture is None:
            return False
        gesture = self.gesture
        if kind == QtCore.QEvent.Type.MouseMove:
            if gesture[0] == 'pan':
                delta = position - gesture[1]
                view.horizontalScrollBar().setValue(gesture[2] - delta.x())
                view.verticalScrollBar().setValue(gesture[3] - delta.y())
            elif gesture[0] == 'select':
                self.rubber.setGeometry(QtCore.QRect(gesture[1], position).normalized())
                self.rubber.show()
            else:
                delta = xyz - gesture[1]
                for item, start in gesture[3]:
                    if isValid(item):
                        item.setPos(start + QtCore.QPointF(float(delta[u]), float(delta[v])))
            return True
        if kind == QtCore.QEvent.Type.MouseButtonRelease:
            self.gesture = None
            self.rubber.hide()
            if gesture[0] == 'move':
                delta = xyz - gesture[1]
                if np.linalg.norm(delta) > 1e-10:
                    self.perform(lambda: self.document.translate(gesture[2], delta))
                else:
                    self.window.refresh_annotations()
            elif gesture[0] == 'select':
                rectangle = QtCore.QRect(gesture[1], position).normalized()
                area = rectangle if rectangle.width() + rectangle.height() > 6 else position
                keys = self.object_keys_at(area)
                points = [item for item in view.items(area) if isinstance(item, Handle)]
                if points and self.single_contour() and not (keys - self.selections):
                    anchors = {(2 - view.axis, *np.round(item.xyz, 7)) for item in points}
                    self.point_selection = self.point_selection | anchors if gesture[2] else anchors
                    self.active_point = next(iter(sorted(anchors)))
                    self.update_controls()
                    self.window.refresh_annotations()
                else:
                    if isinstance(area, QtCore.QPoint) and keys:
                        # Point picking selects the uppermost visible object.
                        ordered = [item.data(0) for item in view.items(area) if item.data(0) in keys]
                        keys = {ordered[0]}
                    self.set_selection(gesture[3] | keys if gesture[2] else keys)
            return True
        return False

    def add_contour_point(self, xyz, position):
        view = self.window.slice_view
        if self.selection[0] == 'surface':
            segments = view._cuts.get((self.selection[1], view.axis, view.index), np.empty((0, 2, 3)))
            if not len(segments):
                self.info.setText('The selected cell has no contour on this layer.')
                return
            a, b = segments[:, 0], segments[:, 1]
            delta = b - a
            fraction = np.clip(np.sum((xyz - a) * delta, axis=1) / np.maximum(np.sum(delta * delta, axis=1), 1e-20), 0, 1)
            candidates = a + fraction[:, None] * delta
            nearest = candidates[np.argmin(np.linalg.norm(candidates - xyz, axis=1))]
            u, v = view.projection_axes
            if (view.mapFromScene(QtCore.QPointF(float(nearest[u]), float(nearest[v]))) - position).manhattanLength() > 16:
                self.info.setText('Click close to the selected cell border to add a point.')
                return
            xyz = nearest
        self.perform(lambda: self.document.add_vertex(*self.selection, xyz))
