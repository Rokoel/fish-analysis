"""Use an optional separate PyTorch environment, leaving TensorFlow intact."""
from pathlib import Path
import importlib.util
import os
import subprocess
import sys

DIRECT_ENGINES = ('signal-center-unet3d',)


def interpreter():
    if getattr(sys, 'frozen', False):
        # The same bundled executable dispatches the PyTorch worker without
        # requiring a separate interpreter or inheriting a developer's override.
        return sys.executable
    configured = os.environ.get('FISH_SPOTS_PYTHON')
    if configured:
        path = Path(configured).expanduser()
        if not path.is_file():
            raise ValueError('FISH_SPOTS_PYTHON does not name a Python executable.')
        return str(path.absolute())
    root = Path(__file__).resolve().parents[2]
    path = root / '.venv-spots' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if path.is_file():
        return str(path.absolute())
    if importlib.util.find_spec('torch'):
        return sys.executable
    raise ValueError('Direct signal models require the optional PyTorch runtime. Run setup-spots.sh (Linux) or setup-spots.bat (Windows).')


def dispatch(arguments):
    python = interpreter()
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(Path(__file__).resolve().parents[2]) + os.pathsep + environment.get('PYTHONPATH', '')
    if environment.get('FISH_CUDA_PATH_READY'):
        paths = [p for p in environment.get('LD_LIBRARY_PATH', '').split(os.pathsep)
                 if not ('site-packages' in p and '/nvidia/' in p.replace('\\', '/'))]
        environment['LD_LIBRARY_PATH'] = os.pathsep.join(paths)
        environment.pop('FISH_CUDA_PATH_READY', None)
    command = [python, '-m', 'fish_viewer.spots', *arguments]
    if sys.platform == 'win32':
        # Windows has no native exec replacement. Its CRT emulation can crash
        # during interpreter replacement; retain the launcher and wait instead.
        raise SystemExit(subprocess.run(command, env=environment).returncode)
    os.execve(python, command, environment)
