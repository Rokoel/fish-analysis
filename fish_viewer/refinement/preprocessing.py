"""Crop-local membrane correction shared by training and both contour branches.

Prepared images and patch caches retain normalized raw intensities. Enhanced
pixels are model inputs only; surface acceptance still uses the raw membrane.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.exposure import equalize_adapthist


def preprocess_membrane(image, spacing_um, config):
    image = np.asarray(image, dtype=np.float32)
    if config.preprocessing == 'none':
        return image
    spacing = np.broadcast_to(np.asarray(spacing_um, dtype=float), (3,))
    if image.ndim != 3 or not np.isfinite(image).all() or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError('Membrane correction requires a finite 3D crop and positive spacing.')
    if image.max() <= image.min():
        return np.zeros_like(image)
    # A broad Gaussian estimates diffuse fluorescence without using target labels.
    background = ndi.gaussian_filter(image, config.background_sigma_um / spacing, mode='reflect')
    corrected = np.maximum(image - background, 0)
    high = float(np.percentile(corrected, 99.7))
    if high <= 1e-6:
        return np.zeros_like(image)
    corrected = np.clip(corrected / high, 0, 1).astype(np.float32)
    if config.preprocessing == 'background':
        return corrected
    # Explicit 3D tiles avoid slice-by-slice contrast discontinuities along Z.
    kernel = tuple(np.maximum(2, np.rint(config.clahe_tile_um / spacing).astype(int)))
    local = equalize_adapthist(corrected, kernel_size=kernel,
                              clip_limit=config.clahe_clip_limit).astype(np.float32)
    enhanced = np.clip((1-config.clahe_fraction)*corrected + config.clahe_fraction*local, 0, 1).astype(np.float32)
    enhanced[image <= 0] = 0  # Retain zero padding outside the acquired volume.
    return enhanced


def augment_membrane(image, spacing_um, config, rng):
    """Vary acquisition physics in ZYX; never alter a seed or a target mask."""
    image = np.asarray(image, dtype=np.float32).copy()
    axes = [np.linspace(-1, 1, n, dtype=np.float32) for n in image.shape]
    direction = rng.normal(size=3)
    direction /= max(float(np.sum(np.abs(direction))), 1e-6)
    gradient = sum(d*v for d, v in zip(direction, np.ix_(*axes)))
    illumination = 1 + rng.uniform(0, config.illumination_strength)*gradient
    background = ndi.gaussian_filter(image, sigma=1./spacing_um, mode='reflect')
    contrast = rng.uniform(1-config.membrane_contrast_range, 1+config.membrane_contrast_range)
    image = np.maximum((background + contrast*(image-background))*illumination, 0)
    # Axial blur is broader than lateral blur, in physical rather than voxel units.
    fraction = rng.uniform(0, 1)
    sigma = np.array([config.psf_axial_um, config.psf_lateral_um, config.psf_lateral_um])*fraction/spacing_um
    image = ndi.gaussian_filter(image, sigma=sigma, mode='reflect')
    photons = np.exp(rng.uniform(np.log(config.photon_count_min), np.log(config.photon_count_max)))
    return np.clip(rng.poisson(np.clip(image, 0, 1)*photons)/photons, 0, 1).astype(np.float32)
