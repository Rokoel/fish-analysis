"""Build a portable source bundle with the existing inference checkpoints."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fish_viewer.analysis.controller import default_models, default_refinement_model, default_proposal_model, check_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/FISH-Analysis')
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination.exists() or destination.with_suffix('.zip').exists():
        parser.error('Choose a new output path; existing bundles are never overwritten.')
    cells, signals = default_models()
    check_models(cells, signals)
    destination.mkdir(parents=True)
    shutil.copytree(ROOT / 'fish_viewer', destination / 'fish_viewer',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('pyproject.toml', 'run_analysis.py', 'run-analysis.sh', 'run-analysis.bat',
                 'run_annotator.py', 'run-annotator.sh', 'run-annotator.bat',
                 'run_viewer.py', 'run-viewer.sh', 'run-viewer.bat',
                 'setup-analysis.sh', 'setup-analysis.bat', 'setup-windows-runtime.ps1', 'setup-spots.sh', 'setup-spots.bat',
                 'requirements-spots.txt', 'requirements-build.txt', 'requirements-cuda-bundle.txt'):
        shutil.copy2(ROOT / name, destination / name)
    guide=(ROOT / 'docs/fish-analysis.md').read_text(encoding='utf-8')
    # This guide normally lives under docs/; make its links work from the ZIP root.
    def guide_link(match):
        target=match[1]
        if target.startswith('../'):
            target=target[3:]
        elif not urlsplit(target).scheme and not target.startswith(('#', '/')):
            target='docs/'+target
        return ']('+target+')'
    (destination / 'README.md').write_text(re.sub(r'\]\(([^)]+)\)',guide_link,guide),encoding='utf-8')
    for name in ('scripts', 'docs', 'reports', 'tests', 'packaging'):
        shutil.copytree(ROOT / name, destination / name,
                        ignore=lambda folder, names: [n for n in names if n in ('__pycache__', 'combined') or n.endswith('.pyc')])
    provenance = {}
    model_sources = [(name, ROOT / 'models' / name) for name in ('cells-sensitive', 'proposals', 'refinement-bias-candidate', 'signals-center-unet', 'cells-stardist2d')]
    for label, source in model_sources:
        target = destination / 'models' / label
        target.mkdir(parents=True)
        copied = {}
        for path in source.iterdir():
            if path.is_file():
                shutil.copy2(path, target / path.name)
                with (target / path.name).open('rb') as stream:
                    copied[path.name] = hashlib.file_digest(stream, 'sha256').hexdigest()
        provenance[label] = dict(source=str(source), sha256=copied)
    (destination / 'models/origins.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    versions = {name: importlib.metadata.version(name) for name in (
        'PySide6', 'vtk', 'pyvista', 'pyvistaqt', 'numpy', 'scipy', 'scikit-image',
        'h5py', 'tensorflow', 'keras', 'stardist', 'openpyxl')}
    (destination / 'tested-runtime.json').write_text(json.dumps(dict(platform=sys.platform, versions=versions), indent=2), encoding='utf-8')
    archive = shutil.make_archive(str(destination), 'zip', destination.parent, destination.name)
    print(json.dumps(dict(bundle=str(destination), archive=archive, models=provenance), indent=2))


if __name__ == '__main__':
    main()
