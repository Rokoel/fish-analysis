"""Editable calibrated annotations, independent of inference's immutable voxel archives."""

import hashlib
import json
import re
from copy import deepcopy
from uuid import uuid4

from ..annotation_state import annotation_report, split_measurements, timestamp
from ..workflow.signal_pairs import DEFAULT_MAX_DISTANCE_UM, update_scene_pairs

import numpy as np
import pyvista as pv
from PySide6 import QtGui

from ..models import Measurement
from ..views import mesh_for
from ..workflow.distances import SurfaceLocator
from ..workflow.geometry import containing_surface
from ..workflow.volumes import surface_volume


def origin(obj):
    return "Manual" if getattr(obj, "source", "manual") == "manual" else "Program"


class Change(QtGui.QUndoCommand):
    def __init__(self, document, before, after, text):
        super().__init__(text)
        self.document, self.before, self.after = document, before, after

    def undo(self):
        self.document.restore(self.before)

    def redo(self):
        self.document.restore(self.after)


class AnnotationDocument:
    def __init__(self, dataset, report=None, changed=lambda: None):
        self.dataset, self.changed = dataset, changed
        self.report_template = deepcopy(report or annotation_report(dataset))
        self.pairing_max_distance = self.report_template.get('signal_pairing', {}).get('maximum_distance_um', DEFAULT_MAX_DISTANCE_UM)
        self.undo = QtGui.QUndoStack()
        self.bindings = {}
        self.channels = {}
        self.scores = {}
        surfaces = dataset.scene.surfaces
        self.run_id = next((obj.run_id for objects in (dataset.scene.surfaces, dataset.scene.measurements)
                            for obj in objects if origin(obj) == "Program" and obj.run_id), str(uuid4()))
        split_measurements(dataset.scene)
        for objects in (dataset.scene.surfaces, dataset.scene.measurements, dataset.scene.contours):
            for obj in objects:
                if origin(obj) == "Program":
                    obj.run_id = obj.run_id or self.run_id
                    obj.created_at = obj.created_at or timestamp()
        surface_ids = {s.uid: i for i, s in enumerate(surfaces)}
        for i, m in enumerate(dataset.scene.measurements):
            self.channels[i] = m.channel
            self.scores[i] = m.score
            self.bindings[i] = surface_ids.get(m.cell_uid)
            a = self.signal_index(m)
            if a is not None and self.bindings[i] is None:
                candidates = [j for j, s in enumerate(surfaces)
                              if origin(s) == origin(m) and s.time == m.times[a]]
                self.bindings[i] = self.containing(m.points[a], candidates)
                if self.bindings[i] is None:
                    self.bindings[i] = self.containing(m.points[a], [j for j, s in enumerate(surfaces)
                                                                   if s.time == m.times[a]])
        manual_points = self.report_template.get("manual_measurements", {}).get(
            "points", []
        )
        for i, m in enumerate(dataset.scene.measurements):
            row = next((p for p in manual_points if p.get("signal_id") == m.uid), None)
            if row is not None and self.channels[i] is None:
                self.channels[i] = row.get("channel")
        # Restore exact editor associations only for the verified companion.
        sidecar = dataset.metadata.path.with_suffix(".annotations.json")
        if (
            sidecar.is_file()
            and dataset.scene.source
            and dataset.scene.source.suffix.lower() == ".imx"
        ):
            try:
                state = json.loads(sidecar.read_text(encoding="utf-8")).get(
                    "editor_state", {}
                )
                digest = hashlib.sha256(dataset.scene.source.read_bytes()).hexdigest()
                n = len(dataset.scene.measurements)
                bindings = {int(k): v for k, v in state.get("bindings", {}).items()}
                channels = {int(k): v for k, v in state.get("channels", {}).items()}
                scores = {int(k): v for k, v in state.get("scores", {}).items()}
                valid = (
                    state.get("scene_sha256") == digest
                    and set(bindings) == set(channels) == set(scores) == set(range(n))
                    and all(
                        v is None or type(v) is int and 0 <= v < len(surfaces)
                        for v in bindings.values()
                    )
                    and all(
                        v is None
                        or type(v) is int
                        and 0 <= v < len(dataset.metadata.channels)
                        for v in channels.values()
                    )
                    and all(
                        v is None or isinstance(v, (int, float)) and np.isfinite(v)
                        for v in scores.values()
                    )
                )
                if valid:
                    self.bindings, self.channels, self.scores = (
                        bindings,
                        channels,
                        scores,
                    )
            except (OSError, ValueError, TypeError, AttributeError):
                pass
        self.recalculate()

    @staticmethod
    def signal_index(m):
        if m.kind == 'signal_pair':
            return None
        return next(
            (
                i
                for i, label in enumerate(m.labels)
                if bool(re.fullmatch(r"A\d*", label.strip(), re.IGNORECASE))
            ),
            0 if len(m.points) == 1 and m.labels[0].strip().upper() != "B" else None,
        )

    def containing(self, point, candidates=None):
        return containing_surface(self.dataset.scene.surfaces, point, candidates)

    def snapshot(self):
        return deepcopy((self.dataset.scene, self.bindings, self.channels, self.scores, self.pairing_max_distance))

    def restore(self, state):
        self.dataset.scene, self.bindings, self.channels, self.scores, self.pairing_max_distance = deepcopy(state)
        self.recalculate()
        self.changed()

    def apply(self, text, function):
        if getattr(self, '_batch_depth', 0):
            function()
            return
        before = self.snapshot()
        try:
            function()
            self.recalculate()
            after = self.snapshot()
        except BaseException:
            self.dataset.scene, self.bindings, self.channels, self.scores, self.pairing_max_distance = before
            raise
        self.undo.push(Change(self, before, after, text))

    def batch(self, text, function):
        def change():
            self._batch_depth = getattr(self, '_batch_depth', 0) + 1
            try:
                function()
            finally:
                self._batch_depth -= 1
        self.apply(text, change)

    def delete_many(self, selections):
        names = {'surface': 'surfaces', 'measurement': 'measurements', 'contour': 'contours'}
        ids = [(kind, getattr(self.dataset.scene, names[kind])[index].uid) for kind, index in selections]
        def change():
            for kind, uid in sorted(ids, key=lambda item: item[0] != 'surface'):
                objects = getattr(self.dataset.scene, names[kind])
                index = next((i for i, obj in enumerate(objects) if obj.uid == uid), None)
                if index is not None:
                    self.delete(kind, index)
        self.batch('Delete selected objects', change)

    def translate(self, selections, delta):
        delta = np.asarray(delta, float)
        if delta.shape != (3,) or not np.isfinite(delta).all():
            raise ValueError('Use finite XYZ offsets.')
        if selections and all(kind == 'measurement' and self.dataset.scene.measurements[index].kind == 'signal_pair'
                              for kind, index in selections):
            raise ValueError('Signal-pair lines are calculated from their signals. Move the signals instead.')
        def change():
            cells = {i for kind, i in selections if kind == 'surface'}
            for kind, index in selections:
                if kind == 'surface':
                    obj = self.dataset.scene.surfaces[index]
                    obj.vertices += delta
                    obj.edited = True
                elif kind == 'contour':
                    obj = self.dataset.scene.contours[index]
                    shift = delta[obj.axis] / self.dataset.metadata.spacing[obj.axis]
                    if not np.isclose(shift, round(shift)):
                        raise ValueError('A section contour must stay on a scan layer.')
                    obj.plane += int(round(shift))
                    obj.points += delta[[a for a in range(3) if a != obj.axis]]
                    obj.edited = True
            for i, obj in enumerate(self.dataset.scene.measurements):
                if obj.kind == 'signal_pair':
                    continue
                if ('measurement', i) in selections or self.bindings.get(i) in cells:
                    obj.points += delta
                    obj.edited = True
                    self.scores[i] = None
                    if self.bindings.get(i) not in cells:
                        previous = self.bindings.get(i)
                        a = self.signal_index(obj)
                        self.bindings[i] = self.containing(obj.points[a], [j for j, s in enumerate(self.dataset.scene.surfaces)
                                                                        if s.time == obj.times[a]]) if a is not None else None
                        if previous is not None and self.bindings[i] is None and a is not None:
                            obj.points, obj.labels, obj.times, obj.edges = obj.points[[a]], ['A'], [obj.times[a]], []
        self.apply('Move selected objects', change)

    def move_section_points(self, index, anchors, delta, radius=None, falloff='smooth'):
        from .tools import deform_surface
        delta = np.asarray(delta, float)
        if delta.shape != (3,) or not np.isfinite(delta).all():
            raise ValueError('Use finite XYZ offsets.')
        def change():
            self.dataset.scene.surfaces[index] = deform_surface(
                self, self.dataset.scene.surfaces[index], anchors, delta, radius, falloff)
        self.apply('Proportional contour edit' if radius else 'Move contour points', change)

    def delete_section_points(self, index, anchors):
        def change():
            obj = self.dataset.scene.surfaces[index]
            for axis in sorted({a[0] for a in anchors}):
                for coordinate in sorted({a[1 + axis] for a in anchors if a[0] == axis}):
                    self.split_surface_plane(obj, axis, coordinate)
            ids = {self.insert_surface_vertex(obj, np.asarray(a[1:], float)) for a in anchors}
            for vertex in sorted(ids, reverse=True):
                self.delete_vertex('surface', index, vertex)
        self.batch('Delete contour points', change)

    def simplify(self, selections, reduction=.4, volume_tolerance=.05):
        """Simplify eligible objects together, reporting each unmodified skip."""
        from .tools import simplify_surface, simplify_polygon
        if not 0 < reduction < 1 or not 0 <= volume_tolerance < 1:
            raise ValueError('Use a reduction between 0 and 100% and a finite volume limit.')
        result = {'simplified': [], 'skipped': []}
        replacements = []
        for kind, index in sorted(selections):
            if kind not in ('surface', 'contour'):
                continue
            objects = self.dataset.scene.surfaces if kind == 'surface' else self.dataset.scene.contours
            obj = objects[index]
            before = len(obj.vertices) if kind == 'surface' else len(obj.points)
            entry = {'kind': kind, 'index': index, 'uid': obj.uid, 'name': obj.name,
                     'before_points': before}
            try:
                if kind == 'surface':
                    simplified = simplify_surface(obj, reduction, volume_tolerance)
                    after = len(simplified.vertices)
                else:
                    simplified = deepcopy(obj)
                    simplified.points = simplify_polygon(obj.points, reduction)
                    after = len(simplified.points)
                if after >= before:
                    raise ValueError('No points can be removed at the requested reduction.')
            except ValueError as exc:
                result['skipped'].append({**entry, 'reason': str(exc)})
                continue
            simplified.edited = True
            replacements.append((kind, index, simplified))
            result['simplified'].append({**entry, 'after_points': after})

        def change():
            for kind, index, obj in replacements:
                objects = self.dataset.scene.surfaces if kind == 'surface' else self.dataset.scene.contours
                objects[index] = obj
        if replacements:
            self.apply('Simplify selected contours', change)
        return result

    def add_layered_cell(self, layers, axis):
        from .tools import mesh_from_layers
        def change():
            obj = mesh_from_layers(layers, axis, self.dataset.spacing,
                                   f'Cell added {len(self.dataset.scene.surfaces) + 1}')
            obj.time, obj.run_id, obj.created_at = self.dataset.time, self.run_id, timestamp()
            obj.edited = True
            self.dataset.scene.surfaces.append(obj)
            index = len(self.dataset.scene.surfaces) - 1
            for i, measurement in enumerate(self.dataset.scene.measurements):
                a = self.signal_index(measurement)
                if (self.bindings.get(i) is None and a is not None and measurement.times[a] == obj.time
                        and self.containing(measurement.points[a], [index]) is not None):
                    self.bindings[i] = index
        self.apply('Add layered cell contour', change)

    def recalculate(self):
        for i, m in enumerate(self.dataset.scene.measurements):
            cell = self.bindings.get(i)
            m.cell_uid = self.dataset.scene.surfaces[cell].uid if cell is not None else None
            m.channel, m.score = self.channels.get(i), self.scores.get(i)
        self.volumes = [surface_volume(s) for s in self.dataset.scene.surfaces]
        self.distances = {}
        locators = {}
        for i, m in enumerate(self.dataset.scene.measurements):
            a = self.signal_index(m)
            cell = self.bindings.get(i)
            existing_b = next(
                (j for j, label in enumerate(m.labels) if label.upper() == "B"), None
            )
            if a is not None and existing_b is not None:
                self.distances[i] = float(
                    np.linalg.norm(m.points[existing_b] - m.points[a])
                )
            if a is None or cell is None or cell >= len(self.dataset.scene.surfaces):
                continue
            if cell not in locators:
                locators[cell] = SurfaceLocator(self.dataset.scene.surfaces[cell])
            border, distance, _ = locators[cell].closest(m.points[a])
            b = next(
                (j for j, label in enumerate(m.labels) if label.upper() == "B"), None
            )
            surface = self.dataset.scene.surfaces[cell]
            if origin(m) == "Manual" and not m.edited and not surface.edited:
                continue
            if b is None:
                b = len(m.points)
                m.points = np.vstack([m.points, border])
                m.labels.append("B")
                m.times.append(m.times[a])
            else:
                m.points[b] = border
            m.edges = [(a, b)]
            self.distances[i] = distance
            if origin(m) == "Program":
                m.name = re.sub(
                    r"; distance [\d.eE+-]+ um", f"; distance {distance:.6g} um", m.name
                )
        previous = {m.uid: (self.bindings.get(i), self.channels.get(i), self.scores.get(i), self.distances.get(i))
                    for i, m in enumerate(self.dataset.scene.measurements)}
        self.pairs = update_scene_pairs(self.dataset.scene, self.dataset.metadata, self.pairing_max_distance, self.run_id)
        self.bindings, self.channels, self.scores, self.distances = {}, {}, {}, {}
        cells = {s.uid: i for i, s in enumerate(self.dataset.scene.surfaces)}
        for i, m in enumerate(self.dataset.scene.measurements):
            binding, channel, score, distance = previous.get(m.uid, (cells.get(m.cell_uid), m.channel, m.score, None))
            self.bindings[i], self.channels[i], self.scores[i] = binding, channel, score
            if distance is not None:
                self.distances[i] = distance
            if m.kind == 'signal_pair':
                self.distances[i] = m.length

    def set_pairing_distance(self, maximum):
        if not np.isfinite(maximum) or maximum <= 0:
            raise ValueError('The maximum signal-pair distance must be positive and finite.')
        self.apply('Change signal-pair distance limit', lambda: setattr(self, 'pairing_max_distance', float(maximum)))

    def delete(self, kind, index):
        if kind == 'measurement' and self.dataset.scene.measurements[index].kind == 'signal_pair':
            raise ValueError('This distance is calculated automatically. Edit or delete its signals, or change the pairing limit.')
        def change():
            if kind == "surface":
                removed = {i for i, c in self.bindings.items() if c == index}
                surface = self.dataset.scene.surfaces[index]
                for i, m in enumerate(self.dataset.scene.measurements):
                    a = self.signal_index(m)
                    if (self.bindings.get(i) is None and a is not None
                            and m.times[a] == surface.time
                            and self.containing(m.points[a], [index]) is not None):
                        removed.add(i)
                self.dataset.scene.surfaces.pop(index)
                keep = [i for i in range(len(self.dataset.scene.measurements)) if i not in removed]
                self.dataset.scene.measurements = [self.dataset.scene.measurements[i] for i in keep]
                self.bindings = {new: c - 1 if c is not None and c > index else c
                                 for new, old in enumerate(keep) for c in [self.bindings.get(old)]}
                for attribute in ("channels", "scores"):
                    mapping = getattr(self, attribute)
                    setattr(self, attribute, {new: mapping.get(old) for new, old in enumerate(keep)})
            elif kind == "measurement":
                self.dataset.scene.measurements.pop(index)
                for attribute in ("bindings", "channels", "scores"):
                    mapping = getattr(self, attribute)
                    setattr(
                        self,
                        attribute,
                        {i - (i > index): v for i, v in mapping.items() if i != index},
                    )
            else:
                self.dataset.scene.contours.pop(index)

        self.apply("Delete " + kind, change)

    def move(self, kind, index, vertex, xyz):
        xyz = np.asarray(xyz, float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError("Use finite XYZ coordinates.")

        def change():
            obj = getattr(
                self.dataset.scene,
                {
                    "surface": "surfaces",
                    "measurement": "measurements",
                    "contour": "contours",
                }[kind],
            )[index]
            obj.edited = True
            if kind == "surface":
                obj.vertices[vertex] = xyz
            elif kind == "contour":
                obj.points[vertex] = xyz[[a for a in range(3) if a != obj.axis]]
            else:
                if obj.kind == 'signal_pair':
                    raise ValueError('Signal-pair lines are calculated from their signals. Edit the signals instead.')
                if obj.labels[vertex].upper() == "B":
                    raise ValueError(
                        "B is calculated from A and the cell border. Edit A or the contour instead."
                    )
                obj.points[vertex] = xyz
                self.scores[index] = None
                previous = self.bindings.get(index)
                self.bindings[index] = (
                    self.containing(xyz, [previous]) if previous is not None else None
                )
                if self.bindings[index] is None:
                    self.bindings[index] = self.containing(
                        xyz,
                        [
                            j
                            for j, s in enumerate(self.dataset.scene.surfaces)
                            if origin(s) == "Program"
                        ],
                    )
                if self.bindings[index] is None:
                    self.bindings[index] = self.containing(xyz)
                if self.bindings[index] is None:
                    obj.points = obj.points[[vertex]]
                    obj.labels = ["A"]
                    obj.times = [obj.times[vertex]]
                    obj.edges = []

        self.apply("Move point", change)

    @staticmethod
    def insert_surface_vertex(obj, xyz):
        border, _, face = SurfaceLocator(obj).closest(xyz)
        distances = np.linalg.norm(obj.vertices - border, axis=1)
        nearest = int(np.argmin(distances))
        if distances[nearest] < 1e-7:
            return nearest
        triangle = obj.triangles[face]
        n = len(obj.vertices)
        edge = None
        for j in range(3):
            a, b = int(triangle[j]), int(triangle[(j + 1) % 3])
            delta = obj.vertices[b] - obj.vertices[a]
            fraction = np.clip(
                np.dot(border - obj.vertices[a], delta)
                / max(np.dot(delta, delta), 1e-20),
                0,
                1,
            )
            if np.linalg.norm(obj.vertices[a] + fraction * delta - border) < 1e-7:
                edge = (a, b)
                break
        obj.vertices = np.vstack([obj.vertices, border])
        if edge is None:
            replacement = [
                [triangle[0], triangle[1], n],
                [triangle[1], triangle[2], n],
                [triangle[2], triangle[0], n],
            ]
            obj.triangles = np.vstack([np.delete(obj.triangles, face, 0), replacement])
        else:
            affected = np.array([all(v in t for v in edge) for t in obj.triangles])
            replacement = []
            for t in obj.triangles[affected]:
                j = next(
                    j for j in range(3) if {int(t[j]), int(t[(j + 1) % 3])} == set(edge)
                )
                a, b, c = t[j], t[(j + 1) % 3], t[(j + 2) % 3]
                replacement.extend([[a, n, c], [n, b, c]])
            obj.triangles = np.vstack([obj.triangles[~affected], replacement])
        return n

    @staticmethod
    def split_surface_plane(obj, axis, coordinate):
        """Make slice intersections explicit mesh vertices before editing one.

        Shared crossing edges receive one vertex used by both adjoining faces.
        This keeps the other slice points fixed when a border point is moved.
        """
        signed = obj.vertices[:, axis] - coordinate
        distances = signed[obj.triangles]
        crossing = (distances.min(axis=1) < -1e-7) & (distances.max(axis=1) > 1e-7)
        vertices = list(obj.vertices.copy())
        edge_vertices = {}
        replacement = []
        for triangle in obj.triangles[crossing]:
            for side in (-1, 1):
                polygon = []
                for a, b in zip(triangle, np.roll(triangle, -1)):
                    if side * signed[a] >= -1e-7:
                        polygon.append(int(a))
                    if (signed[a] < -1e-7 and signed[b] > 1e-7) or (
                        signed[b] < -1e-7 and signed[a] > 1e-7
                    ):
                        edge = tuple(sorted((int(a), int(b))))
                        if edge not in edge_vertices:
                            fraction = signed[a] / (signed[a] - signed[b])
                            xyz = obj.vertices[a] + fraction * (
                                obj.vertices[b] - obj.vertices[a]
                            )
                            xyz[axis] = coordinate
                            edge_vertices[edge] = len(vertices)
                            vertices.append(xyz)
                        polygon.append(edge_vertices[edge])
                replacement.extend(
                    [polygon[0], polygon[j], polygon[j + 1]]
                    for j in range(1, len(polygon) - 1)
                )
        if replacement:
            obj.vertices = np.asarray(vertices)
            obj.triangles = np.vstack([obj.triangles[~crossing], replacement])

    def move_section(self, index, old_xyz, new_xyz, axis=None):
        def change():
            obj = self.dataset.scene.surfaces[index]
            if axis is not None:
                self.split_surface_plane(obj, axis, old_xyz[axis])
            vertex = self.insert_surface_vertex(obj, old_xyz)
            obj.vertices[vertex] = new_xyz
            obj.edited = True

        self.apply("Move slice contour point", change)

    def add_vertex(self, kind, index, xyz):
        def change():
            obj = getattr(
                self.dataset.scene, "surfaces" if kind == "surface" else "contours"
            )[index]
            obj.edited = True
            if kind == "surface":
                self.insert_surface_vertex(obj, xyz)
            else:
                p = np.asarray(xyz)[[a for a in range(3) if a != obj.axis]]
                a = obj.points
                b = np.roll(a, -1, axis=0)
                delta = b - a
                t = np.clip(
                    np.sum((p - a) * delta, axis=1)
                    / np.maximum(np.sum(delta * delta, axis=1), 1e-20),
                    0,
                    1,
                )
                edge = np.argmin(np.linalg.norm(a + t[:, None] * delta - p, axis=1))
                obj.points = np.insert(a, edge + 1, p, axis=0)

        self.apply("Add contour point", change)

    def delete_vertex(self, kind, index, vertex):
        def change():
            obj = getattr(
                self.dataset.scene, "surfaces" if kind == "surface" else "contours"
            )[index]
            obj.edited = True
            if kind == "contour":
                if len(obj.points) <= 3:
                    raise ValueError("A contour needs at least three points.")
                obj.points = np.delete(obj.points, vertex, 0)
                return
            if len(obj.vertices) <= 4:
                raise ValueError("A volume needs at least four vertices.")
            affected = np.any(obj.triangles == vertex, axis=1)
            edges = [tuple(t[t != vertex]) for t in obj.triangles[affected]]
            neighbors = {v for e in edges for v in e}
            if len(edges) != len(neighbors) or len(neighbors) < 3:
                raise ValueError("This point is not a removable manifold vertex.")
            chain = [edges[0][0]]
            while len(chain) < len(neighbors):
                choices = [
                    v for e in edges if chain[-1] in e for v in e if v not in chain
                ]
                if not choices:
                    raise ValueError(
                        "Cannot safely close this contour after removing the point."
                    )
                chain.append(choices[0])
            patch = pv.PolyData(
                obj.vertices[chain], np.array([len(chain), *range(len(chain))])
            ).triangulate()
            faces = np.asarray(chain)[patch.faces.reshape(-1, 4)[:, 1:]]
            triangles = np.vstack([obj.triangles[~affected], faces])
            obj.vertices = np.delete(obj.vertices, vertex, 0)
            obj.triangles = triangles - (triangles > vertex)
            if mesh_for(obj).clean().n_open_edges:
                raise ValueError("Removing this point would open the volume.")
            oriented = mesh_for(obj).compute_normals(
                consistent_normals=True, auto_orient_normals=True, split_vertices=False
            )
            obj.triangles = oriented.faces.reshape(-1, 4)[:, 1:].copy()

        self.apply("Delete contour point", change)

    def add_signal(self, xyz, channel):
        if channel is None:
            raise ValueError("Select a signal channel first.")

        def change():
            i = len(self.dataset.scene.measurements)
            m = Measurement(
                f"Signal added {i + 1}; channel {channel + 1}",
                np.asarray([xyz], float),
                ["A"],
                [self.dataset.time],
                source="program",
                run_id=self.run_id,
                created_at=timestamp(),
                channel=channel,
                color=self.dataset.metadata.channels[channel].color,
            )
            self.dataset.scene.measurements.append(m)
            self.channels[i] = channel
            self.scores[i] = None
            self.bindings[i] = self.containing(
                np.asarray(xyz),
                [
                    j
                    for j, s in enumerate(self.dataset.scene.surfaces)
                    if s.time == self.dataset.time
                ],
            )

        self.apply("Add signal", change)

    def report(self):
        result = deepcopy(self.report_template)
        for key in (
            "cell_predictions",
            "measurement",
            "signal_filter",
            "directory",
            "progress",
            "counts",
        ):
            result.pop(key, None)
        result["engine"] = "fish-edited-annotations"
        result["geometry_source"] = (
            "current editable scene; original manual A-B values preserved until editing"
        )
        result.update(
            image=str(self.dataset.metadata.path),
            channels=[{"name": c.name} for c in self.dataset.metadata.channels],
            points=[],
            cell_volumes=[],
            manual_measurements={"cells": [], "points": []},
            signal_pairs=deepcopy(self.pairs),
            signal_pairing=dict(method='closest_first_one_to_one_per_channel_pair',
                                maximum_distance_um=self.pairing_max_distance, unit='um'),
        )
        for i, s in enumerate(self.dataset.scene.surfaces):
            row = {"cell_id": s.uid, "cell_name": s.name, "volume_um3": self.volumes[i]}
            (
                result["manual_measurements"]["cells"]
                if origin(s) == "Manual"
                else result["cell_volumes"]
            ).append(row)
        for i, m in enumerate(self.dataset.scene.measurements):
            a = self.signal_index(m)
            if a is None:
                continue
            cell = self.bindings.get(i)
            channel = self.channels.get(i)
            row = {
                "signal_id": m.uid,
                "signal_name": m.name,
                "annotation_id": m.uid,
                "cell_id": self.dataset.scene.surfaces[cell].uid
                if cell is not None
                else None,
                "cell_name": self.dataset.scene.surfaces[cell].name if cell is not None else None,
                "channel": channel,
                "channel_name": self.dataset.metadata.channels[channel].name
                if channel is not None
                else None,
                "xyz": m.points[a].tolist(),
                "time": m.times[a],
                "score": self.scores.get(i),
                "distance_um": self.distances.get(i),
            }
            b = next((j for j, l in enumerate(m.labels) if l.upper() == "B"), None)
            if b is not None:
                row["border_xyz"] = m.points[b].tolist()
            (
                result["manual_measurements"]["points"]
                if origin(m) == "Manual"
                else result["points"]
            ).append(row)
        return result
