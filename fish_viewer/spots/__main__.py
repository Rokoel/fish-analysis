import argparse
import json
from pathlib import Path
import signal
import sys

from ..worker_runtime import Cancellation


def main():
    parser = argparse.ArgumentParser(description='Direct 3D signal-center model worker.')
    parser.add_argument('--mode', choices=['apply', 'controls'], required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--image', type=Path)
    parser.add_argument('--scene', type=Path)
    parser.add_argument('--cell-predictions', type=Path)
    parser.add_argument('--signal-channels',nargs='+',type=int)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['auto', 'cpu', 'gpu'], default='auto')
    parser.add_argument('--cancel-file', type=Path)
    args = parser.parse_args()
    cancel = Cancellation(args.cancel_file)
    signal.signal(signal.SIGINT, lambda *_: cancel.event.set())
    stdout = sys.stdout; sys.stdout = sys.stderr
    emit = lambda e: print(json.dumps(e, allow_nan=False), file=stdout, flush=True)
    if args.mode == 'controls':
        from .benchmark import controls
        report = controls(args.model, args.output, args.device)
        emit(dict(kind='finished', report=report))
    else:
        if not args.image or not args.cell_predictions:
            parser.error('Application requires --image and --cell-predictions.')
        from .inference import apply
        report = apply(args.image, args.scene, args.cell_predictions, args.model, args.output, args.device, emit, cancel,args.signal_channels)
    return 0 if report['status'] == 'complete' and report.get('passed', True) else 1


if __name__ == '__main__':
    raise SystemExit(main())
