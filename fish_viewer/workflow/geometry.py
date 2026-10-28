"""Physical mesh membership shared by editing and exported associations."""

import numpy as np
import pyvista as pv

from .distances import SurfaceLocator


def containing_surface(surfaces, point, candidates=None):
    point = np.asarray(point, float)
    query = pv.PolyData(point.reshape(1, 3))
    for i in range(len(surfaces)) if candidates is None else candidates:
        surface = surfaces[i]
        if not np.all((point >= surface.vertices.min(0)) & (point <= surface.vertices.max(0))):
            continue
        locator = SurfaceLocator(surface)
        mesh = pv.wrap(locator.mesh).clean()
        if mesh.n_open_edges:
            continue
        if locator.closest(point)[1] <= 1e-7:
            return i
        if query.select_enclosed_points(mesh, tolerance=1e-7, check_surface=False)['SelectedPoints'][0]:
            return i
    return None
