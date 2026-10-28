"""Shared physical matching and manual-channel inference settings."""
from dataclasses import dataclass, asdict
import math

@dataclass(frozen=True)
class SignalConfig:
    candidate_separation_um: float = .35  # Legacy name used by saved evaluation configurations.
    detection_separation_um: float | None = None
    candidate_contrast: float = .025  # Backward-compatible manual-channel contrast fallback.
    channel_contrast_min: float | None = None
    match_radius_um: float = .6
    assignment_tolerance_um: float = .35
    channel_score_min: float = 1.5
    channel_margin: float = 1.25
    target_precision: float = .90

    def validate(self):
        for key, value in asdict(self).items():
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f'{key} must be positive and finite.')
        if self.target_precision > 1:
            raise ValueError('Target precision must be at most one.')
        return self
