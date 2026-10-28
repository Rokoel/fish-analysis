from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class RefinementConfig:
    voxel_um: float = .15
    patch_size: int = 64
    filters: int = 8
    depth: int = 3
    margin_um: float = 2.
    seed_sigma_um: float = 1.
    boundary_band_um: float = .6
    boundary_weight: float = 5.
    learning_rate: float = .0003
    batch_size: int = 2
    epochs: int = 60
    patience: int = 10
    negative_fraction: float = .15
    negative_noise_max: float = .015
    seed_jitter_um: float = .3
    crop_scale_jitter: float = 0.
    crop_shift_um: float = 0.
    threshold: float = .5
    min_overlap: float = .25
    min_volume_ratio: float = .35
    max_volume_ratio: float = 2.5
    surface_loss_weight: float = 0.
    surface_distance_clip_um: float = 2.
    tversky_beta: float = .5
    residual_correction: bool = False
    seed: int = 42

    def validate(self):
        for name, value in asdict(self).items():
            if not math.isfinite(value):
                raise ValueError(f'{name} must be finite.')
        for name, low, high in [('patch_size',32,128),('filters',4,32),('depth',2,4),
                                ('epochs',1,1000),('patience',1,100),('batch_size',1,8),('seed',0,2**31-1)]:
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f'Invalid {name}.')
        if self.patch_size % (2**self.depth):
            raise ValueError('Patch size must be divisible by 2**depth.')
        if not .05 <= self.voxel_um <= .5 or not .5 <= self.margin_um <= 5:
            raise ValueError('Invalid physical sampling or margin.')
        if not .1 <= self.seed_sigma_um <= 3 or not .1 <= self.boundary_band_um <= 3:
            raise ValueError('Invalid seed or boundary band.')
        if not 1 <= self.boundary_weight <= 20 or not 1e-6 <= self.learning_rate <= .01:
            raise ValueError('Invalid loss weight or learning rate.')
        if not 0 <= self.negative_fraction <= .5 or not .05 <= self.threshold <= .95:
            raise ValueError('Invalid negative fraction or threshold.')
        if not .001<=self.negative_noise_max<=.3 or not 0<=self.seed_jitter_um<=2 or not 0<=self.crop_scale_jitter<=.2 or not 0<=self.crop_shift_um<=1:
            raise ValueError('Invalid crop robustness augmentation.')
        if not 0 < self.min_overlap <= 1 or not 0 < self.min_volume_ratio < 1 < self.max_volume_ratio <= 5:
            raise ValueError('Invalid refinement safeguards.')
        if not 0 <= self.surface_loss_weight <= 1 or not .2 <= self.surface_distance_clip_um <= 5:
            raise ValueError('Invalid surface-distance loss settings.')
        if self.residual_correction:
            raise ValueError('Residual correction models are no longer supported.')
        if not .5 <= self.tversky_beta <= .8 or type(self.residual_correction) is not bool:
            raise ValueError('Invalid under-segmentation/correction settings.')
        return self


def read_model(directory):
    directory = Path(directory).expanduser().resolve()
    manifest = json.loads((directory/'refinement-model.json').read_text(encoding='utf-8'))
    if manifest.get('version') != 1 or manifest.get('engine') != 'seeded-unet3d':
        raise ValueError('Choose a FISH 3D U-Net boundary model folder.')
    RefinementConfig(**manifest['config']).validate()
    if manifest.get('checkpoint') != 'best.weights.h5' or not (directory/manifest['checkpoint']).is_file():
        raise ValueError('The boundary model checkpoint is missing or unsupported.')
    return manifest
