"""Capture the actual Analysis window with a read-only scan and isolated settings."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan', type=Path, default=ROOT / 'FISH/3abx3ad_101212/141212_3ab3ad_Lam633m_dsRed543_CG488b.imx')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--analysis-result', type=Path, help='Open a completed real analysis instead of the original scan.')
    parser.add_argument('--gallery', action='store_true', help='Also capture contour review and measured distances.')
    args = parser.parse_args()
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PySide6 import QtCore, QtWidgets
    from fish_viewer.analysis.app import AnalysisWindow

    app = QtWidgets.QApplication([])
    app.setStyle('Fusion')
    with tempfile.TemporaryDirectory(prefix='fish-docs-') as temporary:
        settings = QtCore.QSettings(str(Path(temporary) / 'settings.ini'), QtCore.QSettings.Format.IniFormat)
        window = AnalysisWindow(off_screen=True, settings=settings)
        window.resize(1600, 1000)
        window.output.setText(str(Path(temporary) / 'results'))
        errors = []
        window.error = errors.append
        window.show()
        window.resize(1600, 1000)
        if args.analysis_result:
            manifest = json.loads(args.analysis_result.read_text())
            window.add_paths([Path(manifest.get('source_scene') or manifest['source'])])
        else:
            window.add_paths([args.scan.resolve()])
        deadline = time.monotonic() + 60
        while (window.dataset is None or window.loading or window._jobs) and not errors:
            app.processEvents()
            if time.monotonic() >= deadline:
                raise TimeoutError('The scan did not load within 60 seconds.')
            time.sleep(.01)
        if errors:
            raise RuntimeError('\n'.join(errors))
        if args.analysis_result:
            window.load_result(args.analysis_result)
            while window.loading or window._jobs:
                app.processEvents()
                if errors:
                    raise RuntimeError('\n'.join(errors))
                if time.monotonic() >= deadline:
                    raise TimeoutError('The saved analysis did not load within 60 seconds.')
                time.sleep(.01)
            window.set_scan_status(window.scans.item(0), 'completed')
            window.output.setText(str(args.analysis_result.parent))
            window.editor.manual.setChecked(False)
        window.annotation_dock.hide()
        app.processEvents()
        window.slice_view.fit()
        app.processEvents()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not window.grab().save(str(args.output)):
            raise OSError(f'Could not save screenshot: {args.output}')
        if args.gallery:
            if not args.analysis_result:
                raise ValueError('--gallery requires --analysis-result for actual measurements.')
            window.annotation_dock.show()
            candidates = [i for i in range(window.distances.rowCount()) if window.distances.item(i, 0).text().startswith('Program')]
            # Prefer an A-B pair lying in the displayed XY slice.
            def slice_offset(row):
                point = window.distances.item(row, 0).data(QtCore.Qt.ItemDataRole.UserRole)
                return abs(point['xyz'][2]-point['border_xyz'][2])
            row = min(candidates, key=slice_offset)
            window.distances.selectRow(row)
            window.focus_measurement()
            window.slice_view.scale(.25, .25)
            selection = window.slice_view.selected
            tree = window.editor.tree
            tree.topLevelItem(0).setExpanded(False)
            program = tree.topLevelItem(1)
            for index in range(program.childCount()):
                item = program.child(index)
                if item.data(0, QtCore.Qt.ItemDataRole.UserRole) == selection:
                    tree.setCurrentItem(item)
                    tree.scrollToItem(item)
                    break
            app.processEvents()
            if not window.grab().save(str(args.output.with_name('editing.png'))):
                raise OSError('Could not save editing screenshot.')
            window.annotation_dock.hide()
            window.tabs.setCurrentIndex(1)
            app.processEvents()
            if not window.grab().save(str(args.output.with_name('distances.png'))):
                raise OSError('Could not save distances screenshot.')
        window.close()
        app.processEvents()
    print(args.output)


if __name__ == '__main__':
    main()
