"""Frozen launcher: preserve the existing ``executable -m worker`` protocol."""

import multiprocessing
import os
import runpy
import sys
import subprocess
from pathlib import Path

WORKERS = frozenset(
    {
        "fish_viewer.neural",
        "fish_viewer.neural.slice_inference",
        "fish_viewer.proposals",
        "fish_viewer.proposals.consensus",
        "fish_viewer.refinement",
        "fish_viewer.signals",
        "fish_viewer.spots",
        "fish_viewer.analysis.shape",
        "fish_viewer.workflow",
        "fish_viewer.runtime_check",
    }
)


def hide_owned_console():
    # Retain the console subsystem for worker JSON pipes, but hide a console
    # created by Explorer. Never hide an existing terminal shared with a user.
    if sys.platform != "win32":
        return
    import ctypes

    ids = (ctypes.c_ulong * 2)()
    kernel = ctypes.windll.kernel32
    kernel.GetConsoleWindow.restype = ctypes.c_void_p
    ctypes.windll.user32.ShowWindow.argtypes = [ctypes.c_void_p, ctypes.c_int]
    if kernel.GetConsoleProcessList(ids, 2) == 1:
        ctypes.windll.user32.ShowWindow(kernel.GetConsoleWindow(), 0)


def main():
    multiprocessing.freeze_support()
    arguments = sys.argv[1:]
    if arguments[:1] == ['--prepare-runtime']:
        if '--progress' in arguments:
            from .runtime_setup import prepare_with_progress as prepare
        else:
            from .runtime_cache import prepare
        print(prepare(Path(sys._MEIPASS)))
        return 0
    if arguments[:1] == ['--check-runtime']:
        for engine in ('tensorflow', 'torch'):
            subprocess.run([sys.executable, '-m', 'fish_viewer.runtime_check',
                            '--engine', engine, *arguments[1:]], check=True)
        return 0
    if arguments[:1] == ["-m"]:
        if len(arguments) < 2 or arguments[1] not in WORKERS:
            raise SystemExit("Unsupported FISH worker module.")
        module = arguments[1]
        sys.argv = [module, *arguments[2:]]
        runpy.run_module(module, run_name="__main__", alter_sys=True)
        return 0
    if arguments[:1] == ["--verify-models"]:
        sys.argv = ["verify_release.py", *arguments[1:]]
        root = Path(__file__).resolve().parents[1]
        runpy.run_path(str(root / "scripts/verify_release.py"), run_name="__main__")
        return 0
    if arguments[:1] == ["--self-test"]:
        from .standalone_smoke import main as smoke

        return smoke(arguments[1:])
    hide_owned_console()
    os.environ.setdefault("QT_API", "pyside6")
    from .analysis.__main__ import main as application

    return application()
