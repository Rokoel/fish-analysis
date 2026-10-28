"""Validate retained model checksums and exercise all five networks on CPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fish_viewer.analysis.controller import check_models, default_models

BUNDLES = ('cells-sensitive', 'proposals', 'refinement-bias-candidate',
           'signals-center-unet', 'cells-stardist2d')


def signal_control_command(model, output):
    return [sys.executable, '-m', 'fish_viewer.signals', '--mode', 'controls',
            '--model', str(model), '--output', str(output), '--device', 'cpu']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-latest-models', action='store_true')
    parser.add_argument('--with-direct-signals', action='store_true')
    args = parser.parse_args()
    origins = json.loads((ROOT / 'models/origins.json').read_text())
    if set(origins) != set(BUNDLES):
        parser.error('Model provenance must list exactly the five supported component bundles.')
    for label, info in origins.items():
        for name, digest in info['sha256'].items():
            with (ROOT / 'models' / label / name).open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
                    raise ValueError(f'Checkpoint checksum mismatch: {label}/{name}')
    cells, signals = default_models()
    check_models(cells, signals)
    sliced = ROOT / 'models/cells-stardist2d'
    check_models(sliced, signals)
    from fish_viewer.neural.engine import runtime
    tf, StarDist3D, _, info = runtime('cpu')
    shapes = {}
    manifest = json.loads((cells / 'fish-model.json').read_text())
    detector = StarDist3D(None, name=cells.name, basedir=str(cells.parent))
    detector.load_weights(cells / manifest['checkpoint'])
    outputs = detector.predict(np.zeros((16,) * 3, np.float32), axes='ZYX', verbose=0)
    assert all(np.isfinite(value).all() for value in outputs)
    shapes[cells.name] = [list(value.shape) for value in outputs]
    from stardist.models import StarDist2D
    manifest = json.loads((sliced / 'fish-model.json').read_text())
    detector2d = StarDist2D(None, name=sliced.name, basedir=str(sliced.parent))
    detector2d.load_weights(sliced / manifest['checkpoint'])
    outputs = detector2d.keras_model.predict_on_batch(np.zeros((1, 32, 32, 1), np.float32))
    assert all(np.isfinite(value).all() for value in outputs)
    shapes[sliced.name] = [list(value.shape) for value in outputs]
    from fish_viewer.refinement.config import RefinementConfig, read_model
    from fish_viewer.refinement.network import network
    folder = ROOT / 'models/refinement-bias-candidate'
    manifest = read_model(folder)
    config = RefinementConfig(**manifest['config'])
    model = network(tf, config)
    model.load_weights(folder / manifest['checkpoint'])
    outputs = model(np.zeros((1,) + (config.patch_size,) * 3 + (2,), np.float32), training=False).numpy()
    assert np.isfinite(outputs).all()
    shapes[folder.name] = list(outputs.shape)
    from fish_viewer.proposals.config import ProposalConfig, read_model
    from fish_viewer.proposals.network import network
    folder = ROOT / 'models/proposals'
    manifest = read_model(folder)
    config = ProposalConfig(**manifest['config'])
    model = network(tf, config)
    model.load_weights(folder / manifest['checkpoint'])
    outputs = model(np.zeros((1,) + (config.patch_size,) * 3 + (1,), np.float32), training=False).numpy()
    assert np.isfinite(outputs).all()
    shapes[folder.name] = list(outputs.shape)
    with tempfile.TemporaryDirectory(prefix='fish-center-check-') as output:
        process = subprocess.run(signal_control_command(signals, output), cwd=ROOT, text=True,
            capture_output=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        if process.returncode:
            raise RuntimeError(process.stderr[-6000:] + process.stdout[-6000:])
        controls = json.loads(process.stdout.splitlines()[-1])['report']
        assert controls['passed'], 'Signal network failed empty-signal controls.'
    print(json.dumps(dict(status='complete', default_pipeline='consensus', bundles=list(BUNDLES),
                         shapes=shapes, runtime=info, signal_controls=controls)))


if __name__ == '__main__':
    main()
