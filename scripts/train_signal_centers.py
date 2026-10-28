"""Train direct-center U-Net on the same masked corpus."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import traceback
import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fish_viewer.spots.config import CenterConfig
from fish_viewer.spots.training import train
from fish_viewer.worker_runtime import Cancellation
from fish_viewer.pipeline.segmentation import Cancelled

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--index', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--engine', choices=['signal-center-unet3d',], default='signal-center-unet3d')
parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--initial-model', type=Path, help='Recover a finite selected checkpoint, restarting the optimizer.')
parser.add_argument('--learning-rate', type=float, default=.0005)
parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
parser.add_argument('--cancel-file', type=Path)
args = parser.parse_args()
cancel=Cancellation(args.cancel_file)
signal.signal(signal.SIGINT,lambda *_:cancel.event.set())
try:
    train(args.index, args.output, args.engine, replace(CenterConfig(), epochs=args.epochs, learning_rate=args.learning_rate), args.device,
          lambda e: print(json.dumps(e, allow_nan=False), flush=True), initial_model=args.initial_model,cancel=cancel)
except Cancelled:
    from fish_viewer.spots.training import write
    path=args.output/'report.json'
    report=json.loads(path.read_text()) if path.exists() else dict(engine='signal-center-unet3d',mode='train',directory=str(args.output),epochs=[])
    report['status']='cancelled'
    if path.parent.exists():write(path,report)
    print(json.dumps(dict(kind='finished',report=report)),flush=True)
except Exception:
    path = args.output / 'report.json'
    if path.is_file():
        from fish_viewer.spots.training import write
        report = json.loads(path.read_text())
        report.update(status='failed', error=traceback.format_exc())
        write(path, report)
    raise
