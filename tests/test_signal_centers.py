import numpy as np
import pytest


def test_windows_signal_dispatch_preserves_arguments_environment_and_exit_code(monkeypatch):
    from types import SimpleNamespace
    from fish_viewer.spots import runtime
    monkeypatch.setattr(runtime.sys, 'platform', 'win32')
    monkeypatch.setattr(runtime, 'interpreter', lambda: 'isolated-python.exe')
    calls = []
    def run(command, *, env):
        calls.append((command, env))
        return SimpleNamespace(returncode=7)
    monkeypatch.setattr(runtime.subprocess, 'run', run)
    with pytest.raises(SystemExit) as exited:
        runtime.dispatch(['--mode', 'controls', '--model', 'model with spaces'])
    assert exited.value.code == 7
    assert calls[0][0] == ['isolated-python.exe', '-m', 'fish_viewer.spots',
                           '--mode', 'controls', '--model', 'model with spaces']
    assert calls[0][1]['PYTHONPATH'].startswith(str(runtime.Path(runtime.__file__).resolve().parents[2]))

from fish_viewer.spots.config import CenterConfig
from fish_viewer.spots.data import gaussian_targets, exclusion_mask, resample_crop
from fish_viewer.spots.detection import local_centers


def test_direct_heatmap_localizes_subvoxel_centers_in_physical_coordinates():
    config = CenterConfig()
    point = np.array([[12.2, 14.3, 11.7]], np.float32)
    heatmap = gaussian_targets(point, (32,) * 3, config.sigma_um / config.spacing_um)
    origin = np.array([10, 20, 30])
    detected = local_centers(heatmap, np.ones(heatmap.shape, bool), origin, config, 7, 2)
    assert len(detected) == 1
    assert np.allclose(detected[0]['xyz'], origin + (point[0] * config.spacing_um)[::-1], atol=1e-5)
    assert detected[0]['cell_id'] == 7 and detected[0]['channel'] == 2


def test_centers_at_cell_boundary_are_not_eroded_and_blank_has_no_predictions():
    config = CenterConfig()
    mask = np.zeros((24,) * 3, bool); mask[6:18, 6:18, 6:18] = True
    heat = gaussian_targets([[6, 12, 12], [20, 12, 12]], mask.shape, config.sigma_um / config.spacing_um)
    points = local_centers(heat, mask, np.zeros(3), config, 1, 1)
    assert len(points) == 1 and points[0]['xyz'][2] == pytest.approx(.9)
    assert not local_centers(np.zeros(mask.shape), mask, np.zeros(3), config, 1, 1)


def test_unsafe_unassigned_a_neighborhood_is_unknown_but_eligible_a_is_not():
    config = CenterConfig()
    points = [dict(xyz=[1.5, 1.5, 1.5], channel=None), dict(xyz=[3, 3, 3], channel=1)]
    valid = exclusion_mask(np.ones((32,) * 3, bool), points, [points[1]], 1, np.zeros(3), config)
    assert not valid[10, 10, 10] and valid[20, 20, 20]
    assert valid[1, 1, 1]


def test_crop_resampling_honors_different_origins_and_anisotropic_calibration():
    grid = np.indices((16, 20, 24), dtype=float)
    source = grid[0] * .3 + grid[1] * .15 + grid[2] * .1
    found = resample_crop(source, (.3, .15, .1), np.array([10, 20, 30]), np.array([10.2, 20.3, 30.6]), (4, 4, 4), .15)
    expected = .2 + .3 + .6 + np.indices((4,) * 3).sum(0) * .15
    assert np.allclose(found, expected)


def test_unknown_voxels_have_zero_heatmap_loss_gradient():
    torch = pytest.importorskip('torch')
    from fish_viewer.spots.network import masked_heatmap_loss
    prediction = torch.zeros((1, 1, 8, 8, 8), requires_grad=True)
    truth = torch.zeros_like(prediction); truth[:, :, 4, 4, 4] = 1
    valid = torch.zeros_like(prediction); valid[:, :, 2:6, 2:6, 2:6] = 1
    loss = masked_heatmap_loss(prediction, truth, valid, 30)
    loss.backward()
    assert not prediction.grad[:, :, :2].any()
    assert prediction.grad[:, :, 4, 4, 4] < 0 and prediction.grad[:, :, 3, 3, 3] > 0






def test_runtime_preserves_virtual_environment_interpreter_symlink(tmp_path, monkeypatch):
    from fish_viewer.spots.runtime import interpreter
    target = tmp_path / 'system-python'; target.write_text('')
    virtual = tmp_path / 'venv-python'
    try:
        virtual.symlink_to(target)
    except OSError:
        pytest.skip('Creating symlinks requires platform privileges')
    monkeypatch.setenv('FISH_SPOTS_PYTHON', str(virtual))
    assert interpreter() == str(virtual)


@pytest.mark.parametrize('engine', ['signal-center-unet3d'])
def test_direct_model_report_and_manifest_load_without_torch(tmp_path, engine):
    import json
    from dataclasses import asdict
    from fish_viewer.signals.model import read_model
    from fish_viewer.signals.overlays import read_signals
    model = tmp_path / 'model'; model.mkdir()
    (model / 'best.pt').write_bytes(b'checkpoint')
    (model / 'signal-model.json').write_text(json.dumps(dict(
        version=1, engine=engine, config=asdict(CenterConfig()), checkpoint='best.pt', threshold=.8)))
    assert read_model(model)['engine'] == engine
    report = tmp_path / 'report.json'
    report.write_text(json.dumps(dict(version=1, engine=engine, mode='apply', status='complete', points=[], counts=[])))
    assert read_signals(report)[1]['engine'] == engine


def test_direct_training_graphs_keep_current_epoch_visible(qapp, tmp_path):
    from PySide6 import QtWidgets
    from fish_viewer.signals.gui import SignalPanel
    host = QtWidgets.QWidget()
    panel = SignalPanel(host)
    panel.display_report(dict(engine='signal-center-unet3d', status='running', mode='train',
        directory=str(tmp_path), epochs=[dict(epoch=1, loss=.2, precision=.91, recall=.5, f05=.78, threshold=.8)],
        best_epoch=1, progress=dict(phase='Training direct 3D signal centers', current=10, total=100, epoch=2, epochs=60)))
    with pytest.raises(ValueError, match='single-scan application'):
        panel.display_report(dict(engine='signal-center-unet3d', split='test', status='complete'))
    assert panel.epoch_label.text() == 'Epoch 2 / 60'
    assert panel.canvas.axes[2].lines[0].get_ydata()[0] == .5
    panel.shutdown(); host.close()




def test_nonfinite_dense_output_is_rejected_instead_of_silent_empty_result():
    torch = pytest.importorskip('torch')
    from fish_viewer.spots.training import predict_volume
    class Broken(torch.nn.Module):
        def forward(self, x):
            return dict(heatmaps=(torch.full((1, 1, *x.shape[-3:]), float('nan')),))
    with pytest.raises(ValueError, match='Non-finite signal-center inference'):
        predict_volume(Broken(), np.zeros((3, 16, 16, 16), np.float32), torch.device('cpu'))
