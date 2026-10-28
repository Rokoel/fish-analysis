"""Freeze the tested source bundle as a Windows EXE or Linux AppImage."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
APPIMAGE_URL = "https://github.com/AppImage/appimagetool/releases/download/1.9.1/appimagetool-x86_64.AppImage"
APPIMAGE_SHA256 = "ed4ce84f0d9caff66f50bcca6ff6f35aae54ce8135408b3fa33abfc3cb384eb0"
RUNTIME_URL = "https://github.com/AppImage/type2-runtime/releases/download/20251108/runtime-x86_64"
RUNTIME_SHA256 = "2fca8b443c92510f1483a883f60061ad09b46b978b2631c807cd873a47ec260d"


def compress_runtime_libraries(internal, output=None, tag=None):
    from fish_viewer.runtime_cache import cache_id
    candidates = []
    for path in sorted(internal.rglob('*')):
        if not path.is_file() or path.is_symlink():
            continue
        name = path.relative_to(internal).as_posix()
        if (name.startswith('nvidia/') and '.so' in path.name or
            name.startswith('torch/lib/') and '.so' in path.name and path.stat().st_size >= 32 * 1024**2 or
            name in ('libtensorflow_cc.so.2', 'libtensorflow_framework.so.2')):
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            candidates.append(dict(path=name, size=path.stat().st_size, sha256=digest))
    target = internal / 'runtime-libraries.tar.xz'
    with target.open('wb') as compressed:
        process = subprocess.Popen(['xz', '-T2', '-6', '--lzma2=dict=64MiB', '-c'], stdin=subprocess.PIPE, stdout=compressed)
        try:
            with tarfile.open(fileobj=process.stdin, mode='w|') as archive:
                for item in candidates:
                    path = internal / item['path']
                    archive.add(path, arcname=item['path'], recursive=False)
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError('Neural runtime compression failed.')
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    (internal / 'runtime-cache.json').write_text(json.dumps(dict(id=cache_id(candidates), files=candidates)), encoding='utf-8')
    if output is not None:
        publish_runtime_archive(internal, output, tag)
    for item in candidates:
        (internal / item['path']).unlink()
    print(json.dumps({'runtime_archive_bytes': sum(p.stat().st_size for p in output.glob('*.runtime.part*')) if output is not None else target.stat().st_size}), flush=True)


def publish_runtime_archive(internal, output, tag):
    manifest_path = internal / 'runtime-cache.json'
    manifest = json.loads(manifest_path.read_text())
    parts = []
    limit = 1536 * 1024**2
    with (internal / 'runtime-libraries.tar.xz').open('rb') as source:
        index = 1
        while True:
            name = f'FISH-Analysis-{tag}-linux-x64.runtime.part{index:02d}'
            target = output / name
            digest = hashlib.sha256()
            size = 0
            with target.open('wb') as stream:
                while size < limit:
                    block = source.read(min(1024**2, limit - size))
                    if not block:
                        break
                    stream.write(block)
                    digest.update(block)
                    size += len(block)
            if not size:
                target.unlink()
                break
            parts.append(dict(name=name, size=size, sha256=digest.hexdigest(),
                              url=f'https://github.com/Rokoel/fish-analysis/releases/download/{tag}/{name}'))
            index += 1
    manifest['parts'] = parts
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    (internal / 'runtime-libraries.tar.xz').unlink()


def windows_crt():
    vswhere = (
        Path(os.environ["ProgramFiles(x86)"])
        / "Microsoft Visual Studio/Installer/vswhere.exe"
    )
    install = subprocess.check_output(
        [
            str(vswhere),
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ],
        text=True,
    ).strip()
    candidates = sorted(
        (Path(install) / "VC/Redist/MSVC").glob("*/x64/Microsoft.VC143.CRT")
    )
    if not candidates:
        raise RuntimeError(
            "Visual Studio x64 app-local redistributable DLLs are missing."
        )
    folder = candidates[-1]
    for name in ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"):
        if not (folder / name).is_file():
            raise RuntimeError(f"Missing Windows runtime: {name}")
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    parser.add_argument("--appimagetool", type=Path)
    parser.add_argument("--cuda", action="store_true", help="Bundle Linux CUDA libraries for TensorFlow and PyTorch")
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    if not (bundle / "models/origins.json").is_file():
        parser.error("First create and verify the source/model bundle.")
    if sys.platform not in ("win32", "linux"):
        parser.error("Build on Windows x64 or Linux x64.")
    if args.cuda and sys.platform != 'linux':
        parser.error('CUDA packaging is supported on Linux only.')
    if args.cuda:
        if '+cu' not in importlib.metadata.version('torch'):
            parser.error('Install CUDA-enabled PyTorch before using --cuda.')
        for package in ('nvidia-cudnn-cu12', 'nvidia-cublas-cu12', 'nvidia-cuda-nvcc-cu12'):
            importlib.metadata.version(package)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    work = ROOT / "build/standalone"
    frozen = work / "dist"
    work.mkdir(parents=True, exist_ok=True)
    profile = work / 'runtime-profile.json'
    profile.write_text(json.dumps({'cuda': args.cuda}), encoding='utf-8')
    env = dict(
        os.environ,
        FISH_SOURCE_BUNDLE=str(bundle),
        QT_API="pyside6",
        CUDA_VISIBLE_DEVICES="-1",
        FISH_RUNTIME_PROFILE=str(profile),
        FISH_BUNDLE_CUDA='1' if args.cuda else '0',
    )
    if os.name == "nt":
        env["FISH_WINDOWS_CRT"] = str(windows_crt())
        from PIL import Image, ImageDraw

        image = Image.new("RGBA", (256, 256), "#101c2b")
        drawing = ImageDraw.Draw(image)
        drawing.ellipse((40, 35, 216, 221), fill="#143c45", outline="#4ee5c3", width=10)
        drawing.ellipse((90, 90, 120, 120), fill="#f9de66")
        drawing.ellipse((140, 140, 162, 162), fill="#f9de66")
        image.save(
            ROOT / "packaging/fish.ico",
            sizes=[(16, 16), (32, 32), (48, 48), (256, 256)],
        )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--distpath",
            str(frozen),
            "--workpath",
            str(work / "pyinstaller"),
            str(ROOT / "packaging/FISH-Analysis.spec"),
        ],
        env=env,
        cwd=ROOT,
        check=True,
    )
    if os.name == "nt":
        target = output / f"FISH-Analysis-{args.tag}-windows-x64.exe"
        shutil.copy2(frozen / "FISH-Analysis.exe", target)
    else:
        appdir = work / "FISH-Analysis.AppDir"
        if appdir.exists():
            shutil.rmtree(appdir)
        (appdir / "usr").mkdir(parents=True)
        shutil.move(str(frozen / "FISH-Analysis"), appdir / "usr/FISH-Analysis")
        for source, name in [
            ("AppRun", "AppRun"),
            ("fish-analysis.desktop", "fish-analysis.desktop"),
            ("fish.svg", "fish-analysis.svg"),
        ]:
            shutil.copy2(ROOT / "packaging" / source, appdir / name)
        (appdir / "AppRun").chmod(0o755)
        if args.cuda:
            compress_runtime_libraries(appdir / 'usr/FISH-Analysis/_internal', output, args.tag)
        tool = (
            args.appimagetool.resolve()
            if args.appimagetool
            else work / "appimagetool.AppImage"
        )
        if not tool.exists():
            urllib.request.urlretrieve(APPIMAGE_URL, tool)
        with tool.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != APPIMAGE_SHA256:
            raise RuntimeError(
                "AppImage build tool checksum differs from the pinned release."
            )
        tool.chmod(0o755)
        runtime = work / "runtime-x86_64"
        if not runtime.exists():
            urllib.request.urlretrieve(RUNTIME_URL, runtime)
        with runtime.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != RUNTIME_SHA256:
                raise RuntimeError(
                    "AppImage runtime checksum differs from the pinned release."
                )
        target = output / f"FISH-Analysis-{args.tag}-linux-x64.AppImage"
        subprocess.run(
            [
                str(tool),
                "--appimage-extract-and-run",
                "--no-appstream",
                '--comp', 'zstd',
                '--mksquashfs-opt', '-processors', '--mksquashfs-opt', '2',
                '--mksquashfs-opt', '-b', '--mksquashfs-opt', '1048576',
                '--mksquashfs-opt', '-Xcompression-level', '--mksquashfs-opt', '22' if args.cuda else '15',
                "--runtime-file",
                str(runtime),
                str(appdir),
                str(target),
            ],
            env=dict(env, ARCH="x86_64"),
            check=True,
        )
        target.chmod(0o755)
    versions = {
        dist.metadata["Name"]: dist.version
        for dist in importlib.metadata.distributions()
    }
    (output / f"{target.name}.build.json").write_text(
        json.dumps(
            {
                "platform": sys.platform,
                "python": sys.version,
                "cuda": args.cuda,
                "versions": versions,
                "models": json.loads((bundle / "models/origins.json").read_text()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"executable": str(target), "bytes": target.stat().st_size}))


if __name__ == "__main__":
    main()
