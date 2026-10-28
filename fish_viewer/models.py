from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import numpy as np


@dataclass
class Channel:
    name: str
    color: tuple[float, float, float]
    low: float = 0.0
    high: float = 255.0
    gamma: float = 1.0
    opacity: float = 1.0
    visible: bool = True
    brightness: float = 0.0
    contrast: float = 1.0
    excitation_nm: float | None = None

    def normalize(self, data: np.ndarray) -> np.ndarray:
        result = (data.astype(np.float32) - self.low) / max(self.high - self.low, 1e-9)
        result = (result - 0.5) * self.contrast + 0.5 + self.brightness
        np.clip(result, 0, 1, out=result)
        np.power(result, 1.0 / max(self.gamma, 0.05), out=result)
        return result * self.opacity


@dataclass
class Surface:
    name: str
    vertices: np.ndarray  # N x 3, physical XYZ coordinates
    triangles: np.ndarray  # M x 3, zero-based vertex indices
    color: tuple[float, float, float] = (0.75, 0.85, 0.95)
    opacity: float = 0.32
    time: int = 0
    visible: bool = True
    source: str = 'manual'
    uid: str = field(default_factory=lambda: str(uuid4()))
    run_id: str | None = None
    created_at: str | None = None
    edited: bool = False
    native_id: str | None = None
    native_xml: bytes | None = None


@dataclass
class Measurement:
    name: str
    points: np.ndarray  # N x 3, physical XYZ coordinates
    labels: list[str]
    times: list[int]
    edges: list[tuple[int, int]] = field(default_factory=list)
    color: tuple[float, float, float] = (1.0, 0.85, 0.3)
    visible: bool = True
    all_times: bool = True
    source: str = 'manual'
    uid: str = field(default_factory=lambda: str(uuid4()))
    cell_uid: str | None = None
    run_id: str | None = None
    created_at: str | None = None
    edited: bool = False
    channel: int | None = None
    score: float | None = None
    native_xml: bytes | None = None
    kind: str = 'signal'
    signal_uids: list[str] = field(default_factory=list)

    @property
    def length(self) -> float:
        return sum(float(np.linalg.norm(self.points[b] - self.points[a])) for a, b in self.edges)


@dataclass
class Contour:
    name: str
    points: np.ndarray  # physical coordinates within the stored drawing plane
    axis: int  # XYZ plane normal
    plane: int  # zero-based plane index
    color: tuple[float, float, float] = (1.0, 0.7, 0.3)
    visible: bool = True

    source: str = 'manual'

    native_xml: bytes | None = None
    uid: str = field(default_factory=lambda: str(uuid4()))
    run_id: str | None = None
    created_at: str | None = None
    edited: bool = False
    native_name: str | None = None

    def vertices(self, metadata) -> np.ndarray:
        result = np.empty((len(self.points), 3), dtype=np.float64)
        remaining = [i for i in range(3) if i != self.axis]
        result[:, remaining] = self.points
        result[:, self.axis] = metadata.center_origin[self.axis] + self.plane * metadata.spacing[self.axis]
        return result


@dataclass
class Scene:
    surfaces: list[Surface] = field(default_factory=list)
    measurements: list[Measurement] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    source: Path | None = None
    contours: list[Contour] = field(default_factory=list)


@dataclass
class Metadata:
    path: Path
    shape: tuple[int, int, int]  # ZYX, valid image dimensions (without HDF5 padding)
    origin: tuple[float, float, float]  # lower voxel edge, XYZ
    spacing: tuple[float, float, float]  # XYZ, calibrated voxel size
    channels: list[Channel]
    times: list[int]
    levels: dict[int, tuple[int, int, int]]
    unit: str = "µm"
    format: str = "Imaris HDF5"
    warnings: list[str] = field(default_factory=list)

    @property
    def extent(self) -> tuple[float, float, float]:
        return tuple(n * s for n, s in zip(self.shape[::-1], self.spacing))

    @property
    def center_origin(self) -> tuple[float, float, float]:
        return tuple(o + s / 2 for o, s in zip(self.origin, self.spacing))


@dataclass
class Dataset:
    metadata: Metadata
    scene: Scene
    volumes: list[np.ndarray]  # one ZYX array per channel, selected time and resolution
    time: int = 0
    level: int = 0

    @property
    def spacing(self) -> tuple[float, float, float]:
        return tuple(e / n for e, n in zip(self.metadata.extent, self.volumes[0].shape[::-1]))

    @property
    def center_origin(self) -> tuple[float, float, float]:
        return tuple(o + s / 2 for o, s in zip(self.metadata.origin, self.spacing))


@dataclass(frozen=True)
class LibraryEntry:
    image: Path
    scene: Path | None = None

    @property
    def label(self) -> str:
        return self.image.name


def composite_slice(dataset: Dataset, axis: int, index: int) -> np.ndarray:
    """RGB image, with additive fluorescence blending and no changes to source arrays."""
    shape = np.take(dataset.volumes[0], index, axis=axis).shape
    rgb = np.zeros((*shape, 3), dtype=np.float32)
    for volume, channel in zip(dataset.volumes, dataset.metadata.channels):
        if channel.visible:
            intensity = channel.normalize(np.take(volume, index, axis=axis))
            rgb += intensity[..., None] * np.asarray(channel.color, dtype=np.float32)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
