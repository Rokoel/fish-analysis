from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class PipelineConfig:
    voxel_um: float = 0.30
    min_diameter_um: float = 2.5
    max_diameter_um: float = 8.0
    min_sphericity: float = 0.62
    max_elongation: float = 1.8
    exclude_border: bool = True
    cell_channel: int | None = None

    def validate(self):
        if self.cell_channel is not None and (type(self.cell_channel) is not int or self.cell_channel < 0):
            raise ValueError('Cell channel must be a nonnegative channel index or automatic.')
        values = asdict(self)
        for key, value in values.items():
            if isinstance(value, (float, int)) and not isinstance(value, bool) and not math.isfinite(value):
                raise ValueError(f'{key} must be finite.')
        if not 0.05 <= self.voxel_um <= 5:
            raise ValueError('Processing voxel size must be 0.05–5 µm.')
        if not 0 < self.min_diameter_um < self.max_diameter_um <= 500:
            raise ValueError('Cell diameter limits must satisfy 0 < minimum < maximum ≤ 500 µm.')
        if not 0 <= self.min_sphericity <= 1:
            raise ValueError('Invalid sphericity.')
        if not 1 <= self.max_elongation <= 5:
            raise ValueError('Invalid elongation.')
        return self

    @classmethod
    def from_dict(cls, values):
        return cls(**values).validate()
