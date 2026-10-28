"""Lightweight filtering/export worker; neural stages use their existing workers."""
import argparse
import json
import signal
import traceback

from ..worker_runtime import Cancellation
from ..pipeline.segmentation import Cancelled


def main():
    parser = argparse.ArgumentParser(description='Filter and export cell/signal predictions.')
    parser.add_argument('operation', choices=['filter', 'measure', 'export', 'unannotated'])
    parser.add_argument('--report')
    parser.add_argument('--scene', help='Original IMX annotation companion for export')
    parser.add_argument('--image', help='Source IMS for an unannotated copy')
    parser.add_argument('--output', required=True)
    parser.add_argument('--cancel-file')
    parser.add_argument('--max-pair-distance', type=float, default=1., help='Maximum within-cell cross-channel signal distance in micrometres')
    args = parser.parse_args()
    if args.operation == 'unannotated' and not args.image:
        parser.error('unannotated requires --image')
    if args.operation != 'unannotated' and not args.report:
        parser.error('filter/measure/export requires --report')
    cancel = Cancellation(args.cancel_file)
    signal.signal(signal.SIGINT, lambda *_: cancel.event.set())
    def emit(event):
        print(json.dumps(event, allow_nan=False), flush=True)
    try:
        if args.operation == 'filter':
            from .results import filter_result
            report = filter_result(args.report, args.output, emit, cancel)
        elif args.operation == 'measure':
            from .distances import measure_result
            report = measure_result(args.report, args.output, emit, cancel, maximum_pair_distance=args.max_pair_distance)
        elif args.operation == 'export':
            from .export import export_result
            report = export_result(args.report, args.output, emit, cancel, source_scene=args.scene)
        else:
            from .export import export_unannotated
            report = export_unannotated(args.image, args.output, emit, cancel)
    except Cancelled:
        report = dict(status='cancelled')
    except Exception:
        report = dict(status='failed', error=traceback.format_exc())
    emit(dict(kind='finished', report=report))
    return 0 if report['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
