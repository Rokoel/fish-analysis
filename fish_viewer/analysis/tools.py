"""Physical geometry operations used by the annotation tools."""

from copy import deepcopy
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from skimage.draw import polygon
from skimage.measure import approximate_polygon, marching_cubes

from ..models import Surface
from ..views import mesh_for
from ..workflow.volumes import surface_volume


def deform_surface(document, surface, anchors, delta, radius=None, falloff='smooth'):
    result = deepcopy(surface)
    for axis in sorted({a[0] for a in anchors}):
        for coordinate in sorted({a[1 + axis] for a in anchors if a[0] == axis}):
            document.split_surface_plane(result, axis, coordinate)
    ids = [document.insert_surface_vertex(result, np.asarray(a[1:], float)) for a in anchors]
    if radius is None:
        result.vertices[np.unique(ids)] += delta
    else:
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError('Influence radius must be positive.')
        distance = cKDTree(result.vertices[ids]).query(result.vertices)[0]
        t = np.clip(1 - distance / radius, 0, 1)
        weight = t if falloff == 'linear' else t * t * (3 - 2 * t)
        result.vertices += weight[:, None] * delta
    result.edited = True
    return result


def simplify_surface(surface, reduction, volume_tolerance):
    original = surface_volume(surface)
    if original is None:
        raise ValueError('Simplification requires a closed cell mesh.')
    mesh = mesh_for(surface).clean().triangulate()
    for _ in range(6):
        simple = mesh.decimate_pro(reduction, preserve_topology=True, splitting=False,
                                   boundary_vertex_deletion=False)
        result = deepcopy(surface)
        result.vertices = simple.points.copy()
        result.triangles = simple.faces.reshape(-1, 4)[:, 1:].copy()
        volume = surface_volume(result)
        if (len(result.vertices) < len(surface.vertices) and volume is not None
                and abs(volume / original - 1) <= volume_tolerance):
            result.edited = True
            return result
        reduction /= 2
    raise ValueError('This cell cannot be simplified within the volume-change limit. Increase the limit or use less reduction.')


def simplify_polygon(points, reduction):
    target = max(3, int(np.ceil(len(points) * (1 - reduction))))
    low, high = 0., float(np.linalg.norm(np.ptp(points, axis=0)))
    best = points.copy()
    closed = np.vstack([points, points[0]])
    for _ in range(32):
        tolerance = (low + high) / 2
        reduced = approximate_polygon(closed, tolerance=tolerance)[:-1]
        if len(reduced) >= target:
            best, low = reduced, tolerance
        else:
            high = tolerance
    return best


def mesh_from_layers(layers, axis, spacing, name):
    """Interpolate signed-distance polygons between physical drawing planes."""
    if not layers or any(len(points) < 3 for points in layers.values()):
        raise ValueError('Draw at least three points and close a layer first.')
    uv = [a for a in range(3) if a != axis]
    step = np.asarray(spacing, float)
    all_points = np.vstack(list(layers.values()))
    origin_uv = all_points.min(0) - 2 * step[uv]
    shape_uv = np.ceil((all_points.max(0) - origin_uv) / step[uv]).astype(int) + 3
    planes = sorted(layers)
    coordinates = np.arange(planes[0] - step[axis], planes[-1] + 1.5 * step[axis], step[axis])
    if len(coordinates) * np.prod(shape_uv) > 16_000_000:
        raise ValueError('The drawn cell is too large; use a coarser preview resolution or a smaller contour.')
    fields = []
    for coordinate in planes:
        mask = np.zeros(tuple(shape_uv), bool)
        points = (np.asarray(layers[coordinate]) - origin_uv) / step[uv]
        rr, cc = polygon(points[:, 0], points[:, 1], shape=mask.shape)
        mask[rr, cc] = True
        if not mask.any():
            raise ValueError('The contour is smaller than the current voxel spacing.')
        fields.append(distance_transform_edt(mask, sampling=step[uv])
                      - distance_transform_edt(~mask, sampling=step[uv]))
    field = np.empty((len(coordinates), *shape_uv), np.float32)
    for i, coordinate in enumerate(coordinates):
        if coordinate < planes[0] - 1e-7 or coordinate > planes[-1] + 1e-7:
            field[i] = -max(step)
        else:
            right = min(np.searchsorted(planes, coordinate), len(planes) - 1)
            left = max(0, right - 1)
            fraction = 0 if left == right else (coordinate - planes[left]) / (planes[right] - planes[left])
            field[i] = fields[left] * (1 - fraction) + fields[right] * fraction
    vertices, faces, _, _ = marching_cubes(field, level=0, spacing=step[[axis, *uv]], allow_degenerate=False)
    xyz = np.empty_like(vertices)
    xyz[:, axis] = vertices[:, 0] + coordinates[0]
    xyz[:, uv] = vertices[:, 1:] + origin_uv
    result = Surface(name, xyz.astype(float), faces.astype(np.int64), color=(.2, .92, .65), source='program')
    if surface_volume(result) is None:
        raise ValueError('The drawn layers did not produce a closed cell.')
    return result
