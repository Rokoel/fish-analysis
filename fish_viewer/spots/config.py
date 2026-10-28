from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class CenterConfig:
    spacing_um: float = .15
    margin_um: float = 1.2
    sigma_um: float = .20
    patch_size: int = 48
    filters: int = 12
    epochs: int = 60
    batch_size: int = 4
    learning_rate: float = .0005
    patience: int = 15
    match_radius_um: float = .6
    detection_separation_um: float = .35
    target_precision: float = .9
    positive_weight: float = 30.
    seed: int = 42

    def validate(self):
        if any(not math.isfinite(v) for v in asdict(self).values()):
            raise ValueError('Center settings must be finite.')
        if not .1 <= self.spacing_um <= .3 or not .1 <= self.sigma_um <= .4:
            raise ValueError('Invalid signal-center sampling scale.')
        if self.patch_size < 32 or self.patch_size > 96 or self.patch_size % 8:
            raise ValueError('Center patch size must be a multiple of eight, 32–96.')
        if not 0 < self.target_precision <= 1 or min(self.epochs, self.batch_size, self.patience) < 1:
            raise ValueError('Invalid center training settings.')
        if not .5 <= self.margin_um <= 3 or not 4 <= self.filters <= 32 or self.filters % 4:
            raise ValueError('Invalid center model size or margin.')
        if not 0 < self.learning_rate <= .01 or not 0 < self.detection_separation_um < self.match_radius_um <= 1:
            raise ValueError('Invalid center learning or matching settings.')
        if not 1 <= self.positive_weight <= 100:
            raise ValueError('Invalid center positive weighting.')
        return self


def read_model(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'signal-model.json').read_text())
    if manifest.get('version') != 1 or manifest.get('engine') not in ('signal-center-unet3d',):
        raise ValueError('Choose a direct 3D signal-center model.')
    CenterConfig(**manifest['config']).validate()
    if manifest.get('checkpoint') != 'best.pt' or not (directory / 'best.pt').is_file():
        raise ValueError('Missing direct signal-center checkpoint.')
    if not 0 <= manifest['threshold'] <= 1:
        raise ValueError('Invalid center acceptance cutoff.')
    return manifest
