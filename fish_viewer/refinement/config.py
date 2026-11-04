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
    batch_size: int = 1
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
    checkpoint_metric: str = 'boundary'
    microscopy_augmentation: bool = False
    illumination_strength: float = .35
    membrane_contrast_range: float = .4
    photon_count_min: float = 80.
    photon_count_max: float = 800.
    psf_lateral_um: float = .18
    psf_axial_um: float = .55
    preprocessing: str = 'none'
    background_sigma_um: float = 3.
    clahe_tile_um: float = 3.
    clahe_clip_limit: float = .01
    clahe_fraction: float = .35
    seed: int = 42

    def validate(self):
        for name, value in asdict(self).items():
            if name not in ('preprocessing', 'checkpoint_metric') and not math.isfinite(value):
                raise ValueError(f'{name} must be finite.')
        if self.checkpoint_metric not in ('loss', 'boundary'):
            raise ValueError('Invalid checkpoint metric.')
        if self.preprocessing not in ('none', 'background', 'clahe'):
            raise ValueError('Invalid membrane preprocessing.')
        if type(self.microscopy_augmentation) is not bool:
            raise ValueError('Invalid microscopy augmentation switch.')
        if not 0 <= self.illumination_strength <= .8 or not 0 <= self.membrane_contrast_range <= .9:
            raise ValueError('Invalid illumination or membrane contrast augmentation.')
        if not 1 <= self.photon_count_min <= self.photon_count_max <= 1e6:
            raise ValueError('Invalid effective photon counts.')
        if not 0 <= self.psf_lateral_um <= self.psf_axial_um <= 2:
            raise ValueError('Invalid anisotropic PSF augmentation.')
        if not .5 <= self.background_sigma_um <= 10 or not .5 <= self.clahe_tile_um <= 10:
            raise ValueError('Invalid physical preprocessing scale.')
        if not 0 < self.clahe_clip_limit <= .1 or not 0 <= self.clahe_fraction <= 1:
            raise ValueError('Invalid CLAHE settings.')
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
