from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import time

import numpy as np
from scipy import ndimage as ndi
from skimage import feature, measure, transform

from ..models import Dataset, Surface
from .config import PipelineConfig


class Cancelled(Exception):
    pass


def checkpoint(cancel=None):
    if cancel is not None and cancel.is_set():
        raise Cancelled('Cancelled by user.')


@dataclass
class Prepared:
    image: np.ndarray
    spacing: tuple[float, float, float]  # ZYX in µm
    origin: tuple[float, float, float]  # XYZ voxel center
    channel: int
    channel_scores: list[dict]
    fingerprint: str
    previews: dict[str, np.ndarray]
    timings: dict[str, float]
    warnings: list[str] = field(default_factory=list)
    evidence_image: np.ndarray | None = None


@dataclass
class Prediction:
    labels: np.ndarray
    cells: list[dict]
    stages: dict[str, np.ndarray]
    timings: dict[str, float]
    rejected: dict[str, int]


def normalize(image):
    image = np.asarray(image, dtype=np.float32)
    sample = image.ravel()[::max(1, image.size // 500_000)]
    finite = sample[np.isfinite(sample)]
    if finite.size == 0:
        return np.zeros_like(image)
    low, high = np.percentile(finite, (1, 99.7))
    if high <= low:
        high = finite.max()
    if high <= low:
        return np.zeros_like(image)
    return np.clip(np.nan_to_num((image - low) / (high - low), nan=0, posinf=1, neginf=0), 0, 1)


def ring_score(volume, spacing, config, cancel=None):
    """Hough rings plus a bright-annulus/dark-interior test, using only pixels."""
    depth, height, width = volume.shape
    step = max(1, int(np.ceil(max(height, width) / 192)))
    sy, sx = spacing[1] * step, spacing[2] * step
    plane_scale = (sx + sy) / 2
    radii = np.unique(np.linspace(max(3, config.min_diameter_um / (2 * plane_scale)),
        max(4, config.max_diameter_um / (2 * plane_scale)), 9).astype(int))
    radii = radii[radii < min(height, width) / (step * 3)]
    if not len(radii):
        return dict(score=0.0, rings=0, ring_contrast=0.0)
    angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    contrast, strengths = [], []
    for z in np.unique(np.linspace(depth * 0.2, max(0, depth * 0.8 - 1), 5).astype(int)):
        checkpoint(cancel)
        plane = normalize(volume[z, ::step, ::step])
        edges = feature.canny(plane, sigma=1, low_threshold=0.08, high_threshold=0.18)
        if edges.sum() < 20:
            continue
        hough = transform.hough_circle(edges, radii)
        accum, xs, ys, rs = transform.hough_circle_peaks(hough, radii,
            min_xdistance=int(radii.min()), min_ydistance=int(radii.min()),
            total_num_peaks=50, threshold=0.35, normalize=True)
        for a, x, y, r in zip(accum, xs, ys, rs):
            if x < r + 1 or y < r + 1 or x + r + 1 >= plane.shape[1] or y + r + 1 >= plane.shape[0]:
                continue
            profiles = []
            for fraction in (0.35, 0.55, 0.85, 1.0, 1.15, 1.4):
                profiles.append(ndi.map_coordinates(plane, [y + fraction * r * np.sin(angles), x + fraction * r * np.cos(angles)], order=1, mode='nearest'))
            shell = np.max(profiles[2:5], axis=0)
            interior = np.mean(profiles[:2], axis=0)
            gain = float(np.mean(shell - interior))
            completeness = float(np.mean(shell > interior + 0.04))
            if gain > 0.065 and completeness > 0.62:
                strengths.append(float(a * completeness * gain))
                contrast.append(gain)
    return dict(score=float(sum(strengths)), rings=len(strengths),
        ring_contrast=float(np.mean(contrast)) if contrast else 0.0)


def prepare(dataset: Dataset, config: PipelineConfig, emit=None, cancel=None, *, background_correct=True, max_voxels=3_000_000) -> Prepared:
    config.validate()
    if dataset.metadata.unit not in ('µm', 'um'):
        raise ValueError(f'Pipeline requires calibrated micrometer units; file reports {dataset.metadata.unit!r}.')
    start = time.perf_counter()
    def report(stage, **kwargs):
        checkpoint(cancel)
        if emit:
            emit(dict(kind='stage', stage=stage, **kwargs))
    report('Using selected cell channel' if config.cell_channel is not None else 'Automatic channel selection', progress=5)
    fingerprint = hashlib.sha256()
    fingerprint.update(str(dataset.volumes[0].shape).encode())
    scores = []
    for index, (volume, channel) in enumerate(zip(dataset.volumes, dataset.metadata.channels)):
        fingerprint.update(memoryview(np.ascontiguousarray(volume)))
        score = ring_score(volume, dataset.spacing[::-1], config, cancel)
        # A small tie-breaker only; display color alone cannot select a channel.
        metadata_hint = any(term in channel.name.casefold() for term in ('lamin', 'lam', '633', '647', 'envelope')) or channel.excitation_nm in (633, 635, 640, 647)
        score.update(channel=index, name=channel.name, excitation_nm=channel.excitation_nm, metadata_hint=metadata_hint)
        score['score'] *= 1.05 if metadata_hint else 1.0
        scores.append(score)
        report('Using selected cell channel' if config.cell_channel is not None else 'Automatic channel selection', progress=5 + 25 * (index + 1) / len(dataset.volumes), channel_scores=scores.copy())
    if config.cell_channel is not None and config.cell_channel >= len(scores):
        raise ValueError('Selected cell channel does not exist in this scan.')
    from ..channel_hints import membrane_stain_channels
    candidates=membrane_stain_channels(dataset.metadata.path,dataset.metadata.channels) or list(range(len(scores)))
    for row in scores:
        row['declared_membrane_candidate']=row['channel'] in candidates
    best = config.cell_channel if config.cell_channel is not None else max(candidates, key=lambda i: scores[i]['score'])
    warnings = []
    ranked = sorted([scores[i]['score'] for i in candidates], reverse=True)
    if config.cell_channel is None and (ranked[0] <= 0 or (len(ranked) > 1 and ranked[0] < ranked[1] * 1.15)):
        warnings.append('Automatic channel choice has weak or ambiguous ring evidence; inspect the channel-score graph and preview.')
    report('Calibrated resampling', progress=40)
    raw = normalize(dataset.volumes[best])
    physical = np.asarray(dataset.metadata.extent[::-1])
    target = max(config.voxel_um, float((np.prod(physical) / max_voxels) ** (1/3))) if max_voxels else config.voxel_um
    shape = np.maximum(2, np.rint(physical / target).astype(int))
    spacing = physical / shape
    resized = transform.resize(raw, tuple(shape), order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
    origin = tuple(o + s / 2 for o, s in zip(dataset.metadata.origin, spacing[::-1]))
    report('Background correction' if background_correct else 'Network intensity normalization', progress=65)
    # Estimate illumination within each optical section. Averaging background
    # through Z can erase faint deep cells next to bright shallow sections.
    if background_correct:
        background = ndi.gaussian_filter(resized, sigma=(0, max(1, config.max_diameter_um / spacing[1]), max(1, config.max_diameter_um / spacing[2])))
        corrected = np.maximum(resized - background, 0)
        image = np.stack([normalize(plane) for plane in corrected])
    else:
        image = resized
    z = image.shape[0] // 2
    previews = {'Raw selected channel': resized[z].copy(), 'Background corrected': image[z].copy()}
    elapsed = time.perf_counter() - start
    report('Preprocessing ready', progress=100, previews=previews, seconds=elapsed,
        timings={'preprocessing': elapsed})
    return Prepared(image, tuple(spacing), origin, best, scores, fingerprint.hexdigest(), previews, {'preprocessing': elapsed}, warnings)


def mask_properties(mask, spacing):
    coords = np.argwhere(mask).astype(np.float64) * spacing
    volume = len(coords) * float(np.prod(spacing))
    if len(coords) < 4:
        return dict(volume_um3=volume, diameter_um=0.0, sphericity=0.0, elongation=float('inf'))
    values = np.linalg.eigvalsh(np.cov(coords, rowvar=False))
    elongation = float(np.sqrt(max(values[-1], 1e-12) / max(values[0], 1e-12)))
    verts, faces, _, _ = measure.marching_cubes(np.pad(mask.astype(np.uint8), 1), level=0.5, spacing=spacing)
    area = measure.mesh_surface_area(verts, faces)
    sphericity = float(np.pi ** (1/3) * (6 * volume) ** (2/3) / area) if area else 0
    return dict(volume_um3=volume, diameter_um=float((6 * volume / np.pi) ** (1/3)),
        sphericity=min(1.0, sphericity), elongation=elongation)




def prediction_surfaces(prediction: Prediction, prepared: Prepared, time_index=0) -> list[Surface]:
    surfaces = []
    for i, slices in enumerate(ndi.find_objects(prediction.labels), 1):
        if slices is None:
            continue
        mask = prediction.labels[slices] == i
        verts, faces, _, _ = measure.marching_cubes(np.pad(mask.astype(np.uint8), 1), level=0.5, spacing=prepared.spacing)
        offset = (np.array([s.start for s in slices]) - 1) * prepared.spacing
        xyz = (verts + offset)[:, ::-1] + prepared.origin
        surfaces.append(Surface(f'Predicted cell {i}', xyz.copy(), faces.astype(np.int64), (0.20, 0.92, 0.65), 0.35, time_index, source='prediction'))
    return surfaces
