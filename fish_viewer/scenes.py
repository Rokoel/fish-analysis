"""Read saved Imaris 5/6 XML scenes. Geometry is kept in physical XYZ units."""
from __future__ import annotations

from pathlib import Path, PureWindowsPath
import re
import xml.etree.ElementTree as ET
from uuid import NAMESPACE_URL, uuid5

import numpy as np

from .models import Contour, Measurement, Scene, Surface


def numbers(text: str | None, dtype=np.float64) -> np.ndarray:
    cleaned = re.sub(r'["(),]', ' ', text or '')
    return np.fromstring(cleaned, sep=' ', dtype=dtype)


def scene_reference(path: Path) -> str | None:
    # The load command precedes the potentially large thumbnail and geometry.
    with path.open('rb') as stream:
        for _, el in ET.iterparse(stream, events=('start',)):
            if el.tag == 'bpCommandLoadDataSet':
                return el.get('file')
            if el.tag == 'Thumbnail':
                break
    return None


def resolve_reference(scene_path: Path, reference: str | None) -> Path:
    """Prefer relocated siblings over obsolete absolute acquisition paths."""
    names = []
    if reference:
        names.append(PureWindowsPath(reference.replace('/', '\\')).name)
    names.append(scene_path.with_suffix('.ims').name)
    siblings = {p.name.casefold(): p for p in scene_path.parent.iterdir() if p.is_file()}
    for name in names:
        if name.casefold() in siblings:
            return siblings[name.casefold()].resolve()
    if reference:
        candidate = Path(reference)
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f'Image referenced by {scene_path.name} is missing: {reference or names[-1]}')


def component(el: ET.Element) -> ET.Element:
    found = el.find('bpSurfaceComponent')
    if found is None:
        found = el.find('bpSurfaceObject/bpSurfaceComponent')
    return found if found is not None else el


def appearance(el: ET.Element) -> tuple[str, tuple[float, float, float], bool]:
    comp = component(el)
    name = comp.findtext('name', default='Unnamed object').strip()
    color = numbers(comp.findtext('material/diffuseColor'))
    return name, tuple(np.clip(color[:3], 0, 1)) if color.size >= 3 else (0.8, 0.85, 0.95), comp.findtext('isHidden') != 'true'


def _surface(name, vertex_text, face_text, color, opacity, time, visible) -> Surface:
    values = numbers(vertex_text)
    faces = numbers(face_text, np.int64)
    if not values.size or values.size % 6 or not faces.size or faces.size % 3:
        raise ValueError('incomplete vertex/normal or triangle arrays')
    vertices = values.reshape(-1, 6)[:, :3].copy()
    triangles = faces.reshape(-1, 3)
    if not np.isfinite(vertices).all() or triangles.min() < 0 or triangles.max() >= len(vertices):
        raise ValueError('invalid coordinates or triangle indices')
    return Surface(name, vertices, triangles, color, float(np.clip(opacity, 0, 1)), time, visible)


def parse_scene(data: bytes | str, source: Path | None = None) -> Scene:
    result = Scene(source=source)
    root = ET.fromstring(data, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
    for viewer in root.iter('bpSurfacesViewer'):
        name, color, visible = appearance(viewer)
        props = viewer.find('bpSurfacesProperties')
        opacity = 1 - float(props.get('mTransparency', '0.68')) if props is not None else 0.32
        for index, surface in enumerate(viewer.findall('bpSurfaces/surface')):
            try:
                result.surfaces.append(_surface(
                    name if index == 0 else f'{name} [{index + 1}]',
                    ' '.join(e.text or '' for e in surface.findall('vertices')),
                    ' '.join(e.text or '' for e in surface.findall('triangles')),
                    color, opacity, int(surface.findtext('timeIndex', '0')), visible,
                ))
                result.surfaces[-1].native_xml = ET.tostring(viewer)
            except (ValueError, TypeError) as exc:
                result.warnings.append(f'Surface {name}: {exc}')
    # Imaris 5.0 stores meshes directly in bpSurfaceObject.
    for obj in root.iter('bpSurfaceObject'):
        if obj.find('vertexPositionNormal') is None:
            continue
        name, color, visible = appearance(obj)
        try:
            result.surfaces.append(_surface(name, obj.findtext('vertexPositionNormal'),
                obj.findtext('triangleVertexIndex'), color, 0.32,
                int(obj.findtext('timeIndex', '0')), visible))
        except (ValueError, TypeError) as exc:
            result.warnings.append(f'Surface {name}: {exc}')
    for obj in root.iter('bpMeasurementPoints'):
        name, color, visible = appearance(obj)
        try:
            points = np.array([numbers(e.get('Position')) for e in obj.findall('Point')])
            if not points.size:
                continue
            if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
                raise ValueError('invalid point coordinates')
            labels = [e.get('Name', str(i + 1)) for i, e in enumerate(obj.findall('Point'))]
            times = [int(e.get('TimeIndex', '0')) for e in obj.findall('Point')]
            mode = obj.findtext('LineMode', 'Polygon')
            if mode.lower() not in ('polygon', 'pairwise', 'none', 'off'):
                raise ValueError(f'unsupported measurement line mode: {mode}')
            step = 2 if mode.lower() == 'pairwise' else 1
            edges = [(i, i + 1) for i in range(0, len(points) - 1, step)] if mode.lower() not in ('none', 'off') else []
            result.measurements.append(Measurement(name, points, labels, times, edges, color,
                visible, obj.findtext('VisibleForAllTimepoints', 'true') == 'true', native_xml=ET.tostring(obj)))
        except (ValueError, TypeError) as exc:
            result.warnings.append(f'Measurement {name}: {exc}')
    # Earlier scenes connect direct child points within each component group.
    # A pairwise group with P1/P2/P3/P4 means P1-P2 and P3-P4, not a polyline.
    old_groups = []
    grouped = set()
    for group in root.iter('bpComponentGroup'):
        comp = group.find('bpSurfaceComponent')
        if comp is not None:
            points = comp.findall('bpMeasurementPoint')
            if points:
                conn = group.find('LineConnection')
                old_groups.append((appearance(group)[0], points, conn.get('value', 'None') if conn is not None else 'None'))
                grouped.update(points)
    ungrouped = [p for p in root.iter('bpMeasurementPoint') if p not in grouped]
    if ungrouped:
        old_groups.append(('Measurement points', ungrouped, 'None'))
    for name, old_points, mode in old_groups:
        labels, points, times = [], [], []
        for obj in old_points:
            pos = obj.find('Position')
            if pos is None:
                continue
            labels.append(appearance(obj)[0])
            points.append([float(pos.get(axis, '0')) for axis in 'xyz'])
            times.append(int(obj.findtext('TimeIndex', '0')))
        step = 2 if mode == 'Pairwise' else 1
        edges = [(i, i + 1) for i in range(0, len(points) - 1, step)] if mode in ('Pairwise', 'Polygon') else []
        if points:
            result.measurements.append(Measurement(name, np.asarray(points), labels, times, edges))
    manuals = list(root.iter('bpManualSurface'))
    for manual in manuals:
        name, color, visible = appearance(manual)
        try:
            axis = int(manual.findtext('direction', '2'))
            if axis not in (0, 1, 2):
                raise ValueError('unsupported contour drawing direction')
            for plane, element in enumerate(manual.findall('store/plane')):
                for line in element.findall('line'):
                    points = np.asarray([[float(v.get('x')), float(v.get('y'))] for v in line.findall('vertex')])
                    if len(points) >= 3 and np.isfinite(points).all():
                        result.contours.append(Contour(f'{name} · {"XYZ"[axis]} slice {plane + 1}', points, axis, plane, color, visible, native_xml=ET.tostring(manual), native_name=name))
        except (ValueError, TypeError) as exc:
            result.warnings.append(f'Unmeshed contour {name}: {exc}')
    for tag in ('bpSpots', 'bpFilaments'):
        if root.find('.//' + tag) is not None:
            result.warnings.append(f'{tag} annotations are not supported by this XML reader.')
    # Every imported object is pre-existing. Display names and history text do
    # not confer ownership of an object to a new analysis run.
    if source is not None:
        for key in ('surfaces', 'measurements', 'contours'):
            for index, obj in enumerate(getattr(result, key)):
                obj.uid = str(uuid5(NAMESPACE_URL, f'{source.resolve()}#{key}/{index}'))
    return result


def read_scene(path: Path) -> Scene:
    return parse_scene(path.read_bytes(), path)
