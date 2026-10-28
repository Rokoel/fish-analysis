import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import signal

from ..worker_runtime import Cancellation, restart_with_cuda
from .config import ProposalConfig


def main():
    restart_with_cuda('fish_viewer.proposals')
    p=argparse.ArgumentParser(description='Tiny 3D cell-center/box detector and crop contour pipeline.')
    p.add_argument('--mode',choices=['train','detect','apply'],required=True);p.add_argument('--audit',type=Path)
    p.add_argument('--cell-channel',type=int)
    p.add_argument('--patches',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',type=Path);p.add_argument('--boundary-model',type=Path)
    p.add_argument('--image',type=Path);p.add_argument('--split',choices=['training','validation','test'])
    p.add_argument('--epochs',type=int);p.add_argument('--device',choices=['gpu','cpu','auto'],default='auto')
    p.add_argument('--cancel-file',type=Path);p.add_argument('--threshold',type=float)
    args=p.parse_args();os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
    cancel=Cancellation(args.cancel_file);signal.signal(signal.SIGINT,lambda *_:cancel.event.set())
    emit=lambda e:print(json.dumps(e,allow_nan=False),flush=True)
    if args.mode=='train':
        if not args.audit or not args.patches:p.error('Training requires --audit and --patches.')
        config=ProposalConfig()
        if args.epochs:config=replace(config,epochs=args.epochs)
        from .training import train
        r=train(args.audit,args.patches,args.output,config,args.device,emit,cancel)
    elif args.mode=='detect':
        if not args.model or not (args.image or (args.audit and args.split)):
            p.error('Detection requires --model and --image or --audit with --split.')
        from .inference import detect
        r=detect(args.model,args.output,args.device,emit,cancel,image=args.image,audit=args.audit,split=args.split,threshold=args.threshold,cell_channel=args.cell_channel)
    else:
        if not args.model or not args.boundary_model or not (args.image or (args.audit and args.split)):
            p.error('Application requires both models and --image or --audit with --split.')
        from .inference import apply
        r=apply(args.model,args.boundary_model,args.output,args.device,emit,cancel,
                image=args.image,audit=args.audit,split=args.split,threshold=args.threshold,
                cell_channel=args.cell_channel)
    return 0 if r['status']=='complete' else 1


if __name__=='__main__':raise SystemExit(main())
