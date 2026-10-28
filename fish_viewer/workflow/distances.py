"""Signal A to closest point B on its assigned cell's calibrated triangle mesh."""
import hashlib
from pathlib import Path
import shutil

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
from vtkmodules.vtkCommonCore import vtkPoints, reference
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData, vtkStaticCellLocator

from ..pipeline.runner import read_prediction, _json
from ..pipeline.segmentation import checkpoint
from ..run_paths import new_run_directory
from .results import validated_result, write_tables


class SurfaceLocator:
    """Find closest points on triangle interiors/edges, not just mesh vertices."""
    def __init__(self, surface):
        vertices = np.asarray(surface.vertices, dtype=np.float64)
        faces = np.asarray(surface.triangles)
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
                or not np.isfinite(vertices).all() or faces.ndim != 2 or faces.shape[1] != 3
                or not len(faces) or not np.issubdtype(faces.dtype, np.integer)
                or faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError('A finite triangle mesh is required for border measurements.')
        points = vtkPoints()
        points.SetData(numpy_to_vtk(vertices, deep=True))
        cells = vtkCellArray()
        cells.SetCells(len(faces), numpy_to_vtkIdTypeArray(
            np.column_stack((np.full(len(faces), 3), faces)).astype(np.int64).ravel(), deep=True))
        self.mesh = vtkPolyData()
        self.mesh.SetPoints(points)
        self.mesh.SetPolys(cells)
        self.locator = vtkStaticCellLocator()
        self.locator.SetDataSet(self.mesh)
        self.locator.BuildLocator()

    def closest(self, point):
        point = np.asarray(point, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError('A finite XYZ signal position is required.')
        nearest = [0., 0., 0.]
        triangle, sub_id, squared = reference(0), reference(0), reference(0.)
        self.locator.FindClosestPoint(point, nearest, triangle, sub_id, squared)
        return np.asarray(nearest), float(np.sqrt(max(float(squared), 0.))), int(triangle)


def archive_digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def measure_points(report, surfaces, metadata, emit=lambda event: None, cancel=None):
    if metadata.unit not in ('µm', 'um'):
        raise ValueError('Signal distances require a scan calibrated in micrometres.')
    meshes = {int(s.name.rsplit(' ', 1)[-1]): s for s in surfaces}
    locators, points = {}, []
    low = np.asarray(metadata.origin)
    high = low + metadata.extent
    for index, point in enumerate(report['points']):
        checkpoint(cancel)
        cell = point['cell_id']
        if cell not in meshes:
            raise ValueError(f'Signal {point["signal_id"]} has no assigned cell surface.')
        if cell not in locators:
            locators[cell] = SurfaceLocator(meshes[cell])
        border, distance, triangle = locators[cell].closest(point['xyz'])
        # Marching cubes closes truncated objects at the image edge. Flag these
        # contacts rather than describing a synthetic cap as observed membrane.
        edge = bool(np.any(np.isclose(border, low, atol=1e-5, rtol=0)
                           | np.isclose(border, high, atol=1e-5, rtol=0)))
        points.append(dict(point, border_xyz=border.tolist(), distance_um=distance,
                           border_triangle=triangle, border_at_image_edge=edge))
        emit(dict(kind='progress', phase='Measuring signal-to-cell border distances',
                  current=index + 1, total=len(report['points'])))
    return points


def validate_measurements(report, surfaces, metadata):
    """Recheck saved distances against the exact meshes being exported."""
    measured = report.get('measurement')
    if not measured:
        if any('border_xyz' in p or 'distance_um' in p for p in report['points']):
            raise ValueError('Incomplete border measurement metadata. Rerun measurements.')
        return
    if (measured.get('method') != 'closest_point_on_assigned_cell_triangle_mesh'
            or measured.get('unit') != 'um'
            or measured.get('contours_sha256') != archive_digest(report['cell_predictions'])):
        raise ValueError('Border measurements are stale or belong to different contours.')
    locators = {int(s.name.rsplit(' ', 1)[-1]): SurfaceLocator(s) for s in surfaces}
    for point in report['points']:
        border = np.asarray(point.get('border_xyz'), dtype=float)
        distance = point.get('distance_um', float('nan'))
        if border.shape != (3,) or not np.isfinite(border).all() or not np.isfinite(distance) or distance < 0:
            raise ValueError('Invalid signal-to-border measurement.')
        locator = locators[point['cell_id']]
        _, expected, _ = locator.closest(point['xyz'])
        _, residual, _ = locator.closest(border)
        length = float(np.linalg.norm(border - point['xyz']))
        if residual > 1e-5 or not np.isclose(length, expected, atol=1e-5, rtol=1e-6) or not np.isclose(distance, length, atol=1e-5, rtol=1e-6):
            raise ValueError('A saved B point is not a closest point on its assigned cell surface.')


def measure_result(path, output, emit=lambda event: None, cancel=None, maximum_pair_distance=1.):
    emit(dict(kind='progress', phase='Checking cells and signals before measurement', current=0, total=0))
    report, labels, cell_metadata, metadata = validated_result(path, cancel)
    if not report.get('signal_filter') or report['signal_filter'] != cell_metadata.get('signal_filter'):
        raise ValueError('Remove cells without signals before measuring border distances.')
    surfaces, _ = read_prediction(report['cell_predictions'])
    points = measure_points(report, surfaces, metadata, emit, cancel)
    from .volumes import add_cell_volumes
    add_cell_volumes(report, surfaces, labels, cell_metadata['spacing_zyx'])
    directory = new_run_directory(output, 'measured-scan')
    try:
        checkpoint(cancel)
        # Keep this result independently usable if intermediate stages are moved.
        shutil.copyfile(report['cell_predictions'], directory / 'contours.npz')
        report = dict(report, points=points, directory=str(directory),
                      cell_predictions=str(directory / 'contours.npz'),
                      parent_measurement_report=str(Path(path).resolve()),
                      measurement=dict(method='closest_point_on_assigned_cell_triangle_mesh', unit='um',
                          signal_endpoint='A', border_endpoint='B',
                          contours_sha256=archive_digest(directory / 'contours.npz'),
                          measured_signals=len(points), image_edge_contacts=sum(p['border_at_image_edge'] for p in points)),
                      progress=dict(phase='Measurements complete', current=len(points), total=len(points)))
        from .signal_pairs import add_signal_pairs
        add_signal_pairs(report, maximum_pair_distance)
        write_tables(directory, report)
        _json(directory / 'report.json', report)
        checkpoint(cancel)
    except BaseException:
        shutil.rmtree(directory)
        raise
    return report
