"""Export the live annotation document, including imported/manual geometry."""

from copy import deepcopy
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
import numpy as np
from .export import _copy_image, _publish
from .imaris_scene import scene_xml, template, _component, _set
from .excel import write_excel
from ..annotation_state import export_name, split_measurements
from ..models import LibraryEntry
from ..reader import load_dataset


def _complete_settings(obj, defaults):
    """Fill absent settings in older/minimal XML while preserving native values."""
    for child in defaults:
        actual = obj.find(child.tag)
        if actual is None:
            obj.append(deepcopy(child))
        else:
            _complete_settings(actual, child)


def editable_scene_xml(image_name, scene, metadata):
    # Build all native image/camera settings through the established exporter.
    root = ET.fromstring(
        scene_xml(image_name, [], metadata, {"points": [], "channels": []})
    )
    target = root.find("bpSurfaceApplication/bpComponentGroup/bpSurfaceComponent")
    prototype = template().find(
        "bpSurfaceApplication/bpComponentGroup/bpSurfaceComponent/bpComponentGroup"
    )
    for i, s in enumerate(scene.surfaces):
        copy = deepcopy(s)
        copy.name = f"Predicted cell {i + 1}"
        part = ET.fromstring(
            scene_xml(image_name, [copy], metadata, {"points": [], "channels": []})
        )
        group = part.find(
            "bpSurfaceApplication/bpComponentGroup/bpSurfaceComponent/bpComponentGroup"
        )
        label = export_name(s)
        _component(group, label)
        viewer = group.find("bpSurfaceComponent/bpSurfacesViewer")
        if s.native_xml is not None:
            native = ET.fromstring(s.native_xml)
            _complete_settings(native, viewer)
            objects = native.find('bpSurfaces')
            for child in list(objects):
                if child.tag in ('surface', 'bpSurfacesId', 'timeInfo'):
                    objects.remove(child)
            for child in viewer.find('bpSurfaces'):
                if child.tag in ('surface', 'timeInfo'):
                    objects.append(deepcopy(child))
            group.find('bpSurfaceComponent').remove(viewer)
            group.find('bpSurfaceComponent').append(native)
            viewer = native
        _component(viewer, label, s.color, s.opacity)
        _set(viewer, "bpSurfaces/name", label)
        if s.source != 'manual':
            _set(viewer, "bpSurfaces/historyLog", "FISH annotation created: " + (s.created_at or ""))
        _set(viewer, "bpSurfaceComponent/isHidden", "false" if s.visible else "true")
        target.append(group)
    for m in scene.measurements:
        group = deepcopy(prototype)
        comp = group.find("bpSurfaceComponent")
        _component(group, export_name(m))
        viewer = comp.find("bpSurfacesViewer")
        comp.remove(viewer)
        marker = comp.find("bpMeasurementPoints")
        if m.native_xml is not None:
            comp.remove(marker)
            native = ET.fromstring(m.native_xml)
            _complete_settings(native, marker)
            marker = native
            comp.append(marker)
        _component(marker, export_name(m), m.color)
        _set(marker, "bpSurfaceComponent/isHidden", "false" if m.visible else "true")
        for point in marker.findall("Point"):
            marker.remove(point)
        # Native XML measurement groups represent a polyline. Split disjoint
        # edge pairs into independent groups rather than joining unrelated spots.
        if m.edges == [(i, i + 1) for i in range(len(m.points) - 1)] or not m.edges:
            pairs = [tuple(range(len(m.points)))]
        else:
            pairs = m.edges
        used = {i for edge in pairs for i in edge}
        pairs = list(pairs) + [(i,) for i in range(len(m.points)) if i not in used]
        for edge in pairs:
            item = deepcopy(group)
            points = item.find("bpSurfaceComponent/bpMeasurementPoints")
            _set(points, "LineMode", "Polygon" if len(edge) == 1 or m.edges else "None")
            _set(points, "VisibleForAllTimepoints", "true" if m.all_times else "false")
            for offset, index in enumerate(edge):
                points.insert(
                    1 + offset,
                    ET.Element(
                        "Point",
                        Name=m.labels[index],
                        Position=" ".join(f"{v:.12g}" for v in m.points[index]),
                        TimeIndex=str(m.times[index]),
                    ),
                )
            target.append(item)
    for c in scene.contours:
        if c.native_xml is not None:
            manual = ET.fromstring(c.native_xml)
            component = manual.find("bpSurfaceComponent")
            name = component.find("name")
            name.text = c.native_name if c.source == 'manual' and c.native_name else export_name(c)
            store = manual.find("store")
            for element in list(store):
                store.remove(element)
        else:
            manual = ET.Element("bpManualSurface")
            component = ET.SubElement(manual, "bpSurfaceComponent")
            ET.SubElement(component, "name").text = export_name(c)
            material = ET.SubElement(component, "material")
            ET.SubElement(material, "diffuseColor").text = " ".join(
                str(v) for v in c.color
            )
            ET.SubElement(manual, "direction").text = str(c.axis)
            store = ET.SubElement(manual, "store")
        for index in range(c.plane + 1):
            plane = ET.SubElement(store, "plane")
            if index == c.plane:
                line = ET.SubElement(plane, "line")
                for x, y in c.points:
                    ET.SubElement(line, "vertex", x=f"{x:.12g}", y=f"{y:.12g}")
        target.append(manual)
    for index, component in enumerate(root.iter("bpSurfaceComponent")):
        ident = component.find("id")
        if ident is not None:
            ident.text = str(index + 1)
    ET.indent(root)
    return (
        ET.tostring(root, encoding="utf-8", short_empty_elements=False).replace(
            b"\n", b"\r\n"
        )
        + b"\r\n"
    )


def export_document(document, path):
    target = Path(path).expanduser().resolve().with_suffix(".ims")
    source = document.dataset.metadata.path.resolve()
    if source == target:
        raise ValueError("Export to a new filename; the original scan is read-only.")
    destinations = [
        target.with_suffix(s) for s in (".ims", ".xlsx")
    ]
    if any(p.exists() for p in destinations):
        raise FileExistsError(
            "Choose a new filename; existing files are never overwritten."
        )
    # Snapshot before export so later UI edits cannot change half an export.
    scene = deepcopy(document.dataset.scene)
    split_measurements(scene)
    report = document.report()
    metadata = deepcopy(document.dataset.metadata)
    if any(not w.startswith("No saved annotations found.") for w in scene.warnings):
        raise ValueError(
            "Some source annotations could not be read; export cannot safely discard them: "
            + "; ".join(scene.warnings)
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".fish-edit-", dir=target.parent
    ) as temporary:
        staging = Path(temporary)
        _copy_image(source, staging / target.name, lambda e: None, None)
        report["editor_state"] = dict(
            bindings=document.bindings,
            channels=document.channels,
            scores=document.scores,
        )
        report["annotation_run_id"] = document.run_id
        write_excel(staging / target.with_suffix(".xlsx").name, report)
        from .embedded_export import embed_scene
        embed_scene(staging / target.name, scene, metadata, report)
        loaded = load_dataset(LibraryEntry(staging / target.name))
        validate_roundtrip(scene, loaded.scene)
        _publish(staging, destinations)
    return str(target)


def validate_roundtrip(expected, actual):
    """Reject changed coordinates, connectivity, times or missing annotations."""
    for key in ("surfaces", "measurements", "contours"):
        left, right = getattr(expected, key), getattr(actual, key)
        if len(left) != len(right):
            raise ValueError(f"Edited {key} failed export reload validation.")
        for before, after in zip(left, right):
            if key == "surfaces":
                same = (np.allclose(before.vertices, after.vertices, rtol=1e-8, atol=1e-7)
                        and np.array_equal(before.triangles, after.triangles)
                        and before.time == after.time)
                # The native writer may reverse triangle winding for outward normals.
                if not same and np.allclose(before.vertices, after.vertices, rtol=1e-8, atol=1e-7):
                    same = (np.array_equal(np.sort(before.triangles, axis=1), np.sort(after.triangles, axis=1))
                            and before.time == after.time)
            elif key == "measurements":
                same = (np.allclose(before.points, after.points, rtol=1e-10, atol=1e-9)
                        and before.edges == after.edges and before.labels == after.labels
                        and before.times == after.times and before.all_times == after.all_times)
            else:
                same = (np.allclose(before.points, after.points, rtol=1e-10, atol=1e-9)
                        and before.axis == after.axis and before.plane == after.plane)
            if not same:
                raise ValueError(f"Annotation geometry or connections changed during export: {before.name}")
