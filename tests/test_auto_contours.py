from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from fish_viewer.neural.config import NeuralConfig
from fish_viewer.neural.precision import evidence_preset


def test_bundled_models_remain_discoverable_without_historical_runs():
    from fish_viewer.neural.model_picker import discover_models
    from fish_viewer.project_paths import ROOT

    models = dict(discover_models([ROOT / 'models']))
    assert {path.name for path in models} == {'cells-sensitive'}
    assert discover_models([ROOT / 'models'])[0][0].name == 'cells-sensitive'


def saved_model(path, status='complete'):
    path.mkdir(parents=True)
    config = replace(replace(evidence_preset(NeuralConfig()),use_shape_filter=True,min_sphericity=.65,max_elongation=1.8), probability_threshold=.65, membrane_support=.6)
    (path / 'fish-model.json').write_text(json.dumps(dict(version=1, engine='stardist3d', config=asdict(config), checkpoint='best.weights.h5')))
    (path / 'best.weights.h5').write_bytes(b'test checkpoint placeholder')
    (path.parent / 'report.json').write_text(json.dumps(dict(status=status, mode='train', epochs=[dict(epoch=1)])))
    return path


def test_visible_elongated_cell_passes_without_roundness_but_empty_mask_does_not():
    from fish_viewer.neural.engine import filter_instances
    grid = (np.indices((96,) * 3) - 47.5) * .3
    radius = np.sqrt((grid[0] / 2.2) ** 2 + grid[1] ** 2 + grid[2] ** 2)
    labels = (radius < 3).astype(np.uint32)
    prepared = SimpleNamespace(image=np.exp(-((radius - 3) / .25) ** 2), spacing=(.3,) * 3, origin=(0.,) * 3)
    assert not filter_instances(labels, prepared, replace(evidence_preset(NeuralConfig()),use_shape_filter=True,min_sphericity=.65,max_elongation=1.8)).cells
    config = evidence_preset(NeuralConfig())
    assert len(filter_instances(labels, prepared, config).cells) == 1
    for intensity in (0., .01, 1.):
        prepared.image.fill(intensity)
        result = filter_instances(labels, prepared, config)
        assert not result.cells and result.rejected['membrane_evidence'] == 1


def test_old_models_keep_historical_shape_policy():
    values = asdict(NeuralConfig())
    del values['use_shape_filter']
    restored = NeuralConfig.from_dict(values)
    assert restored.use_shape_filter
    updated = evidence_preset(restored)
    assert not updated.use_shape_filter and not updated.curate_training_shapes
    assert updated.precision_mode and not updated.calibrate_probability


@pytest.mark.gui
def test_model_picker_lists_completed_models_and_rejects_missing_checkpoint(qapp, tmp_path):
    from fish_viewer.neural.model_picker import ModelPicker, discover_models
    balanced = saved_model(tmp_path / 'training-one' / 'run' / 'model')
    saved_model(tmp_path / 'training-two' / 'run' / 'model')
    saved_model(tmp_path / 'cancelled' / 'run' / 'model', 'cancelled')
    saved_model(tmp_path / 'smoke' / 'run' / 'model')
    assert len(discover_models([tmp_path])) == 2
    dialog = ModelPicker('selected.ims', roots=[tmp_path])
    try:
        assert dialog.models.currentData() == str(balanced)
        (balanced / 'best.weights.h5').unlink()
        dialog.accept()
        assert 'missing' in dialog.error.text() and dialog.model_path is None
        dialog.models.setCurrentIndex(1)
        dialog.accept()
        assert dialog.model_path.name == 'model'
    finally:
        dialog.close()


@pytest.mark.gui
def test_small_window_reports_and_selected_scan_application(qapp, ims_file, tmp_path, monkeypatch):
    from PySide6 import QtWidgets
    from fish_viewer.app import MainWindow
    from fish_viewer.models import LibraryEntry
    window = MainWindow(off_screen=True)
    try:
        window.resize(1024, 500)
        window.show()
        qapp.processEvents()
        assert window.height() == 500
        assert window.viewer_scroll.verticalScrollBar().maximum() > 0
        window.show_neural()
        window.neural_dock.setFloating(True)
        window.neural_dock.resize(800, 500)
        panel = window.neural_panel
        panel.tabs.setCurrentIndex(3)
        for _ in range(4):
            qapp.processEvents()
        assert window.neural_scroll.verticalScrollBar().maximum() > 0
        window.neural_scroll.ensureWidgetVisible(panel.results_open_report)
        qapp.processEvents()
        position = panel.results_open_report.mapTo(window.neural_scroll.viewport(), panel.results_open_report.rect().center())
        assert 0 <= position.y() < window.neural_scroll.viewport().height()
        assert not panel.config().use_shape_filter
        assert not panel.controls['min_sphericity'].isEnabled()
        report = tmp_path / 'report.json'
        report.write_text(json.dumps(dict(status='complete')))
        monkeypatch.setattr(QtWidgets.QFileDialog, 'getOpenFileName', lambda *args: (str(report), ''))
        displayed = []
        monkeypatch.setattr(panel, 'display_report', displayed.append)
        panel.results_open_report.click()
        window.neural_report_button.click()
        assert displayed == [dict(status='complete')] * 2
        model = saved_model(tmp_path / 'selected' / 'model')
        entry = LibraryEntry(ims_file)
        started = []
        monkeypatch.setattr(panel, 'start', lambda: started.append((panel.checked_entries(), panel.config())))
        panel.apply_to_scan(entry, model)
        entries, config = started[0]
        assert entries == [entry] and panel.mode.currentIndex() == 1
        assert not config.use_shape_filter and config.precision_mode
        assert config.probability_threshold == .65 and config.membrane_support == .6
    finally:
        window.close()
        qapp.processEvents()
