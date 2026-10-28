import argparse
import os
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description='Paint training-only FISH crops. Zero/unpainted is unknown, never background.')
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--crop-xy', type=int, default=128, help='Native pixels in X and Y (default: 128)')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--fish-root', type=Path, help='Relocate audited scans to this FISH directory, for example on another machine')
    args = parser.parse_args()
    from .data import TrainingSampler
    try:
        sampler = TrainingSampler(args.audit, (args.crop_xy, args.crop_xy), args.seed, args.fish_root)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    os.environ['QT_API'] = 'pyside6'
    if sys.platform.startswith('linux') and os.environ.get('QT_QPA_PLATFORM') not in ('offscreen', 'minimal'):
        os.environ['QT_QPA_PLATFORM'] = 'xcb'
    from PySide6 import QtWidgets
    from .app import AnnotationWindow
    app = QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName('FISH training annotations')
    app.setStyle('Fusion')
    window = AnnotationWindow(sampler, args.output.resolve())
    window.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
