"""Read calibrated scene geometry stored as HDF5 objects under Scene/Content.

The scene schema varies between Imaris versions. Recognized mesh/spot arrays
are validated; unknown object layouts are reported rather than silently lost.
"""

import re
import xml.etree.ElementTree as ET
from uuid import NAMESPACE_URL, uuid5
import h5py
import numpy as np
from .models import Scene, Surface, Measurement


def _array(group, names, width, *, normals=False):
    for name in names:
        if name not in group or not isinstance(group[name], h5py.Dataset):
            continue
        value = group[name][()]
        if value.dtype.names:
            fields = value.dtype.names
            if len(fields) < width:
                continue
            value = np.column_stack([value[f] for f in fields[:width]])
        # Native IMS meshes store XYZ + normal XYZ in each vertex record.
        # Flattening these records into triples turns normals into vertices.
        if normals and value.ndim == 2 and value.shape[1] == 6:
            value = value[:, :3]
        if value.ndim == 2 and value.shape[1] != width:
            raise ValueError(f"Invalid {name} record width.")
        if value.size % width:
            raise ValueError(f"Invalid {name} array shape.")
        return np.asarray(value).reshape(-1, width)
    return None


def content_scene(image, decode):
    scene = Scene()
    # Scene8 supersedes Scene when both contain the same object identifiers.
    content = next(
        (
            image[p]
            for p in ("Scene8/Content", "Scene/Content")
            if p in image and isinstance(image[p], h5py.Group) and len(image[p])
        ),
        None,
    )
    if content is None:
        return scene

    def visit(path, group):
        if not isinstance(group, h5py.Group):
            return
        name = decode(group.attrs.get("Name", path.rsplit("/", 1)[-1]))
        if re.match(r"Filaments?\d+$", path.rsplit("/", 1)[-1]):
            scene.warnings.append(f"Unsupported embedded filament object: {path}")
            return
        color = (0.75, 0.85, 0.95)
        if "Material" in group.attrs:
            try:
                material = ET.fromstring(decode(group.attrs["Material"]))
                rgb = np.fromstring(material.get("mDiffusion", ""), sep=" ")
                if len(rgb) == 3:
                    color = tuple(np.clip(rgb, 0, 1))
            except ET.ParseError:
                pass
        try:
            coords = _array(
                group,
                ("CoordsXYZR", "CoordsXYZ", "CoordinatesXYZRadii"),
                4 if "CoordsXYZR" in group or "CoordinatesXYZRadii" in group else 3,
            )
            if coords is not None and not any(
                k in group
                for k in ("Triangles", "TriangleIndices", "TriangleVertexIndices")
            ):
                if not np.isfinite(coords[:, :3]).all():
                    raise ValueError("Nonfinite spot coordinates.")
                times = (
                    np.asarray(group["Time"][()]).reshape(-1)
                    if "Time" in group
                    else np.zeros(len(coords), int)
                )
                if len(times) != len(coords):
                    raise ValueError("Spot time count differs from coordinates.")
                for i, (point, time) in enumerate(zip(coords, times)):
                    scene.measurements.append(
                        Measurement(
                            f"{name} · spot {i + 1}",
                            point[:3].astype(float).reshape(1, 3),
                            ["A"],
                            [int(time)],
                            color=color,
                            all_times=False,
                        )
                    )
                return
            vertices = _array(group, ("Vertices", "VertexPositions", "CoordsXYZ"), 3, normals=True)
            faces = _array(
                group, ("Triangles", "TriangleIndices", "TriangleVertexIndices"), 3
            )
            if vertices is None or faces is None:
                if re.match(r"(?:Surfaces?|Points|Spots)\d+$", path.rsplit("/", 1)[-1]):
                    scene.warnings.append(f"Unsupported embedded object layout: {path}")
                return
            if not np.isfinite(vertices).all() or not np.issubdtype(
                faces.dtype, np.integer
            ):
                raise ValueError("Invalid embedded mesh arrays.")
            records = group["SurfaceData"][()] if "SurfaceData" in group else None
            if "TimeNVerticesNTriangles" in group:
                counts = np.asarray(group["TimeNVerticesNTriangles"][()])
                if counts.ndim != 2 or counts.shape[1] != 3:
                    raise ValueError("Invalid TimeNVerticesNTriangles records.")
                records = np.rec.fromarrays(counts.T, names="TimeIndex,NumberOfVertices,NumberOfTriangles")
            fields = (
                {
                    re.sub("[^a-z]", "", f.lower()).removeprefix("m"): f
                    for f in records.dtype.names
                }
                if records is not None and records.dtype.names
                else {}
            )
            nv = fields.get("numberofvertices")
            nt = fields.get("numberoftriangles")
            time_field = fields.get("timeindex")
            if records is not None and not (nv and nt):
                raise ValueError(
                    "Unsupported SurfaceData object counts; cannot separate meshes safely."
                )
            if nv and nt:
                start_v = start_f = 0
                for i, record in enumerate(records.reshape(-1)):
                    count_v, count_f = int(record[nv]), int(record[nt])
                    v = vertices[start_v : start_v + count_v]
                    f = faces[start_f : start_f + count_f].astype(np.int64)
                    # Imaris stores object-local triangle indices in concatenated arrays.
                    if len(f) and (f.min() < 0 or f.max() >= len(v)):
                        raise ValueError("Invalid surface-local triangle indices.")
                    scene.surfaces.append(
                        Surface(
                            name if len(records) == 1 else f"{name} [{i + 1}]",
                            v.astype(float).copy(),
                            f.copy(),
                            color,
                            time=int(record[time_field]) if time_field else 0,
                            native_id=decode(group.attrs['Id']) if 'Id' in group.attrs else None,
                        )
                    )
                    start_v += count_v
                    start_f += count_f
                if start_v != len(vertices) or start_f != len(faces):
                    raise ValueError("Surface counts do not cover mesh arrays.")
            else:
                if len(faces) and (faces.min() < 0 or faces.max() >= len(vertices)):
                    raise ValueError("Invalid triangle indices.")
                scene.surfaces.append(
                    Surface(
                        name,
                        vertices.astype(float),
                        faces.astype(np.int64),
                        color,
                        time=int(group.attrs.get("TimeIndex", 0)),
                        native_id=decode(group.attrs['Id']) if 'Id' in group.attrs else None,
                    )
                )
        except (ValueError, TypeError, KeyError) as exc:
            scene.warnings.append(f"Embedded object {name}: {exc}")

    groups = []
    content.visititems(lambda path, group: groups.append((path, group)) if isinstance(group, h5py.Group) else None)
    for path, group in sorted(groups, key=lambda item: [int(p) if p.isdigit() else p for p in re.split(r'(\d+)', item[0])]):
        visit(path, group)
    for key in ('surfaces', 'measurements'):
        for index, obj in enumerate(getattr(scene, key)):
            obj.uid = str(uuid5(NAMESPACE_URL, f'{image.filename}#embedded/{key}/{index}'))
    return scene
