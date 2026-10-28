"""Processing choices, editable overview and batch recovery in the Analysis UI."""
import time

from PySide6 import QtCore, QtWidgets

from fish_viewer.analysis.app import AnalysisWindow
from fish_viewer.reader import ImarisReader, entry_for_path, load_dataset


def test_add_scans_preflights_metadata_without_loading_all_volumes(qapp, tmp_path, ims_file):
    settings = QtCore.QSettings(str(tmp_path / 'add.ini'), QtCore.QSettings.Format.IniFormat)
    window = AnalysisWindow(settings=settings)
    errors = []
    window.error = errors.append
    try:
        window.add_paths([ims_file])
        deadline = time.monotonic() + 5
        while (window._jobs or window.loading) and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(.01)
        assert not errors
        assert window.scans.count() == 1 and window.dataset is not None
        assert len(window.scans.item(0).data(QtCore.Qt.ItemDataRole.UserRole)['channels']) == 2
        assert 'Check channels' in window.scans.item(0).text()
    finally:
        window.close()


def window_with_scan(qapp, tmp_path, ims_file):
    settings = QtCore.QSettings(str(tmp_path / 'ui.ini'), QtCore.QSettings.Format.IniFormat)
    window = AnalysisWindow(settings=settings)
    item = QtWidgets.QListWidgetItem()
    item.setData(QtCore.Qt.ItemDataRole.UserRole, dict(image=str(ims_file), scene=None, channels=ImarisReader(ims_file).metadata.channels, status='pending'))
    window.scans.blockSignals(True)
    window.scans.addItem(item)
    window.scans.setCurrentItem(item)
    window.scans.blockSignals(False)
    window._source_image = str(ims_file)
    window.set_dataset(load_dataset(entry_for_path(ims_file)))
    window.sync_buttons()
    return window, item


def test_channel_review_and_processing_summary(qapp, tmp_path, ims_file):
    window, item = window_with_scan(qapp, tmp_path, ims_file)
    try:
        window.refresh_scan_item(item)
        assert 'Check channels' in item.text()
        window.show_channels()
        window.cell_channel_choice.setCurrentIndex(window.cell_channel_choice.findData(0))
        assert 'Check channels' not in item.text()
        assert 'Cell: 1; signals: 2' in window.processing_summary.text()
        window.pipeline.setCurrentIndex(window.pipeline.findData('consensus'))
        assert window.pipeline.count() == 2
        assert window.pipeline.currentData() == 'consensus'
        window.channel_roles[str(ims_file)] = dict(cell=0, signals=[])
        assert window.channel_summary(item.data(QtCore.Qt.ItemDataRole.UserRole))[1]
    finally:
        window.close()


def test_overview_tracks_edits_and_panel_is_visible(qapp, tmp_path, ims_file):
    window, _ = window_with_scan(qapp, tmp_path, ims_file)
    try:
        window.show()
        qapp.processEvents()
        assert not window.annotation_dock.isVisible()
        assert window.tabs.tabText(3) == 'Results'
        assert 'Pre-existing: 0 cells · 0 signals' in window.results_overview.text()
        window.editor.document.add_signal([12, 22, 31], 1)
        assert 'This run: 0 cells · 1 signals · 0 measured distances' in window.results_overview.text()
        assert window.overview_excel.isEnabled()
        assert not window.overview_folder.isEnabled()
        window.update_results_overview(dict(points=[dict(distance_um=1), dict(distance_um=3)], cell_volumes=[{}]))
        assert 'median 2.0000; mean 2.0000; range 1.0000–3.0000' in window.results_overview.text()
    finally:
        window.close()


def test_failure_continues_queue_and_cancel_preserves_pending(qapp, tmp_path, ims_file, monkeypatch):
    window, first = window_with_scan(qapp, tmp_path, ims_file)
    second = QtWidgets.QListWidgetItem()
    second.setData(QtCore.Qt.ItemDataRole.UserRole, dict(first.data(QtCore.Qt.ItemDataRole.UserRole), image=str(ims_file.parent / 'second.ims')))
    window.scans.addItem(second)
    first.setSelected(True)
    second.setSelected(True)
    calls = []
    monkeypatch.setattr(window.controller, 'start', lambda *a, **kw: calls.append(a[0]))
    try:
        window.start()
        assert first.data(QtCore.Qt.ItemDataRole.UserRole)['status'] == 'running'
        window.failed('First scan worker failed')
        deadline = time.monotonic() + 2
        while len(calls) < 2 and time.monotonic() < deadline:
            qapp.processEvents()
        assert len(calls) == 2
        assert 'First scan worker failed' in first.toolTip()
        assert first.data(QtCore.Qt.ItemDataRole.UserRole)['status'] == 'failed'
        assert second.data(QtCore.Qt.ItemDataRole.UserRole)['status'] == 'running'
        window.cancelled()
        assert second.data(QtCore.Qt.ItemDataRole.UserRole)['status'] == 'cancelled'
        assert not window.running
        window.running = True
        window.queue = [0]
        window.cancelled()
        assert first.data(QtCore.Qt.ItemDataRole.UserRole)['status'] == 'failed'
    finally:
        window.close()
