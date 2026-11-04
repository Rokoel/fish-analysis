import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import signal

from ..worker_runtime import Cancellation, restart_with_cuda
from .config import RefinementConfig


def main():
    restart_with_cuda('fish_viewer.refinement')
    parser=argparse.ArgumentParser(description='Train or apply a seeded 3D U-Net boundary model.')
    parser.add_argument('--redraw-proposals',action='store_true',help='Use detector boxes/seeds only; reject failed contours without fallback')
    parser.add_argument('--mode',choices=['train','apply'],required=True)
    parser.add_argument('--audit',type=Path)
    parser.add_argument('--cache',type=Path)
    parser.add_argument('--patches',type=Path)
    parser.add_argument('--cells-report',type=Path)
    parser.add_argument('--model',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--initial-model',type=Path)
    parser.add_argument('--patch-size',type=int,help='Cubic contour input size; inherits the initial model, or 64 for fresh training (96 is experimental)')
    parser.add_argument('--batch-size',type=int,default=1,help='96-cubed crops use 3.375 times the voxels of 64-cubed crops')
    parser.add_argument('--learning-rate',type=float,default=.0003)
    parser.add_argument('--threshold',type=float,default=None,help='Contour threshold; inherited from --initial-model when omitted')
    parser.add_argument('--checkpoint-metric',choices=['loss','boundary'],default='boundary')
    parser.add_argument('--surface-loss-weight',type=float,default=0.)
    parser.add_argument('--microscopy-augmentation',action=argparse.BooleanOptionalAction,default=None,
                        help='Spatial illumination, membrane contrast, effective photon noise and axial PSF blur')
    parser.add_argument('--preprocessing',choices=['none','background','clahe'],default=None,
                        help='Raw normalized crops, background subtraction, or subtraction plus 3D CLAHE; inherited from --initial-model when omitted')
    parser.add_argument('--robust-crops',action='store_true',help='Augment box extent, position and seed; broader synthetic noise controls')
    parser.add_argument('--device',choices=['auto','cpu','gpu'],default='auto')
    parser.add_argument('--cancel-file',type=Path)
    args=parser.parse_args()
    os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
    cancel=Cancellation(args.cancel_file)
    signal.signal(signal.SIGINT,lambda *_:cancel.event.set())
    emit=lambda e:print(json.dumps(e,allow_nan=False),flush=True)
    if args.mode=='train':
        if not all([args.audit,args.cache,args.patches]):
            parser.error('Training requires --audit, --cache and --patches.')
        config=RefinementConfig()
        if args.initial_model:
            from .config import read_model
            saved=RefinementConfig(**read_model(args.initial_model)['config'])
            # Keep inference/validation choices unless an experiment overrides
            # them. Crop size can change without changing convolutional weights.
            keys=('filters','depth','patch_size','threshold','seed_sigma_um','margin_um','voxel_um',
                  'microscopy_augmentation','illumination_strength','membrane_contrast_range',
                  'photon_count_min','photon_count_max','psf_lateral_um','psf_axial_um',
                  'preprocessing','background_sigma_um','clahe_tile_um','clahe_clip_limit','clahe_fraction')
            config=replace(config,**{key:getattr(saved,key) for key in keys})
        config=replace(config,batch_size=args.batch_size,
                       learning_rate=args.learning_rate,checkpoint_metric=args.checkpoint_metric,
                       surface_loss_weight=args.surface_loss_weight)
        if args.patch_size is not None:config=replace(config,patch_size=args.patch_size)
        if args.microscopy_augmentation is not None:config=replace(config,microscopy_augmentation=args.microscopy_augmentation)
        if args.threshold is not None:config=replace(config,threshold=args.threshold)
        if args.preprocessing is not None:config=replace(config,preprocessing=args.preprocessing)
        if args.epochs is not None:config=replace(config,epochs=args.epochs)
        if args.robust_crops:config=replace(config,seed_jitter_um=.8,crop_scale_jitter=.1,crop_shift_um=.4,negative_noise_max=.12)
        from .training import train
        report=train(args.audit,args.cache,args.patches,args.output,config,args.device,emit,cancel,args.initial_model)
    else:
        if not args.cells_report or not args.model:
            parser.error('Application requires --cells-report and --model.')
        from .inference import apply
        report=apply(args.cells_report,args.model,args.output,args.device,emit,cancel,args.audit,redraw=args.redraw_proposals)
    return 0 if report['status']=='complete' else 1


if __name__=='__main__':
    raise SystemExit(main())
