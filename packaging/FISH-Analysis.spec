# Build with scripts/build_standalone.py; paths are supplied explicitly.
import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules, collect_data_files, copy_metadata

root = Path(SPECPATH).parent
bundle = Path(os.environ['FISH_SOURCE_BUNDLE'])
datas = [(str(bundle / name), name) for name in ('models', 'docs', 'reports')]
datas += [(str(root / 'scripts/verify_release.py'), 'scripts')]
datas += [(os.environ['FISH_RUNTIME_PROFILE'], '.')]
datas += collect_data_files('fish_viewer')
# Lazy imports in both neural libraries need distribution metadata at runtime.
for package in ('stardist', 'csbdeep', 'tensorflow', 'keras', 'numpy', 'scipy',
                'torch', 'torchvision', 'pyvista', 'pyvistaqt', 'imagecodecs'):
    datas += copy_metadata(package, recursive=True)
hidden = collect_submodules('fish_viewer')
hidden += collect_submodules('stardist') + collect_submodules('csbdeep')
hidden += ['matplotlib.backends.backend_svg', 'matplotlib.backends.backend_pdf',
           'matplotlib.backends.backend_agg', 'matplotlib.backends.backend_qtagg']
binaries = []
if os.environ.get('FISH_BUNDLE_CUDA') == '1':
    import importlib.metadata
    # Explicitly preserve pip's CUDA layout, including ptxas and libdevice.
    # No host driver (libcuda.so) is bundled; the system supplies it.
    seen = set()
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata['Name']
        if not name.startswith('nvidia-') or name in seen:
            continue
        seen.add(name)
        datas += copy_metadata(name)
        for relative in distribution.files or []:
            if relative.parts[0] != 'nvidia' or not any(part in ('lib', 'bin', 'nvvm') for part in relative.parts):
                continue
            source = distribution.locate_file(relative)
            if source.is_file():
                target = str(relative.parent)
                (binaries if '.so' in source.name or 'bin' in relative.parts else datas).append((str(source), target))
if os.name == 'nt':
    crt = Path(os.environ['FISH_WINDOWS_CRT'])
    binaries += [(str(p), '.') for p in crt.glob('*.dll')]
else:
    # Qt's desktop platform plugin needs these on hosts without Qt installed.
    import ctypes.util
    from PyInstaller.depend.bindepend import resolve_library_path
    for name in ('xcb-cursor', 'xcb-icccm', 'xcb-image', 'xcb-keysyms',
                 'xcb-render-util', 'xcb-xinerama', 'xkbcommon-x11', 'EGL', 'GL', 'OpenGL'):
        library = ctypes.util.find_library(name)
        path = resolve_library_path(library) if library else None
        if path:
            binaries.append((path, '.'))

a = Analysis([str(root / 'packaging/entry.py')], pathex=[str(root)],
             binaries=binaries, datas=datas, hiddenimports=hidden,
             excludes=['PyQt5', 'PyQt6', 'PySide2', 'tkinter', 'IPython', 'pytest',
                       'jax', 'jaxlib', 'nvidia', 'tensorboard', 'triton'],
             noarchive=False)
if os.name != 'nt':
    # Dependency analysis also places package libraries at the bundle root.
    # Keep one canonical copy: otherwise CUDA adds several GB of duplicates
    # and may load a different cuDNN version from the root than from nvidia/.
    canonical = {}
    for destination, source, kind in a.binaries:
        if kind == 'BINARY' and (destination.startswith('nvidia/') or destination.startswith('torch/lib/')):
            canonical[Path(destination).name] = destination
    a.binaries = [(destination, canonical[destination], 'SYMLINK')
                  if '/' not in destination and destination in canonical else (destination, source, kind)
                  for destination, source, kind in a.binaries]
pyz = PYZ(a.pure)
# torch JIT and TensorFlow need source files; their maintained hooks collect them.
if os.name == 'nt':
    exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name='FISH-Analysis',
              debug=False, strip=False, upx=False, console=True,
              icon=str(root / 'packaging/fish.ico'))
else:
    exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='FISH-Analysis',
              debug=False, strip=False, upx=False, console=True)
    coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='FISH-Analysis')
