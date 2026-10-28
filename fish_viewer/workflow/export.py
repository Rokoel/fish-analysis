"""Export standalone annotated IMS images and their Excel measurements."""
import hashlib
from pathlib import Path
import tempfile
from copy import deepcopy
from uuid import uuid4

import h5py

from ..models import LibraryEntry
from ..pipeline.runner import read_prediction
from ..pipeline.segmentation import checkpoint
from ..reader import ImarisReader, load_dataset, entry_for_path
from .imaris_scene import scene_xml
from .results import validated_result


def _copy_image(source, destination, emit, cancel):
    if not h5py.is_hdf5(source):
        from .embedded_export import convert_legacy_image
        convert_legacy_image(source, destination, emit, cancel)
        with destination.open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    before = source.stat()
    total, copied = before.st_size, 0
    digest = hashlib.sha256()
    with source.open('rb') as src, destination.open('wb') as dst:
        while block := src.read(8 * 1024 * 1024):
            checkpoint(cancel)
            dst.write(block)
            digest.update(block)
            copied += len(block)
            emit(dict(kind='progress', phase='Copying Imaris image (all channels, times and resolutions)',
                      current=copied, total=total))
    if source.stat().st_mtime_ns != before.st_mtime_ns or copied != total:
        raise ValueError('The source scan changed during export. Please retry.')
    with destination.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != digest.hexdigest():
            raise ValueError('The exported image differs from its source.')
    # Replace embedded scene objects with the current annotations in the copy.
    # Copying first preserves all image datasets, acquisition metadata and pyramids.
    embedded = False
    if h5py.is_hdf5(destination):
        with h5py.File(destination, 'r') as image:
            embedded = any(k.startswith('Scene') for k in image) or any(k.startswith('Scene') for k in image.attrs)
    if embedded:
        with h5py.File(destination, 'r+') as image:
            for key in list(image):
                if key.startswith('Scene'):
                    del image[key]
            for key in list(image.attrs):
                if key.startswith('Scene'):
                    del image.attrs[key]
    with destination.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _publish(staging, destinations):
    created = []
    try:
        for destination in destinations:
            with destination.open('xb'):
                pass
            created.append(destination)
        for destination in destinations:
            (staging / destination.name).replace(destination)
    except BaseException:
        for destination in created:
            destination.unlink(missing_ok=True)
        raise


def export_result(report_path, target, emit=lambda event: None, cancel=None, source_scene=None):
    """Publish an annotated IMS and workbook; refuse to overwrite either file."""
    target = Path(target).expanduser().resolve()
    if target.suffix.lower() != '.ims':
        raise ValueError('Choose an export filename ending in .ims.')
    suffixes = ['.ims', '.xlsx']
    destinations = [target.with_suffix(suffix) for suffix in suffixes]
    if any(path.exists() for path in destinations):
        raise FileExistsError('The export would overwrite existing files. Choose a new name or folder.')
    emit(dict(kind='progress', phase='Validating detections for export', current=0, total=1))
    report, labels, cell_metadata, metadata = validated_result(report_path, cancel)
    if not report.get('signal_filter') or report['signal_filter'] != cell_metadata.get('signal_filter'):
        raise ValueError('Remove cells without signals before exporting this workflow result.')
    surfaces, _ = read_prediction(report['cell_predictions'])
    from .volumes import add_cell_volumes
    from .excel import write_excel
    add_cell_volumes(report, surfaces, labels, cell_metadata['spacing_zyx'])
    from .distances import validate_measurements
    validate_measurements(report, surfaces, metadata)
    if len(surfaces) != len({point['cell_id'] for point in report['points']}):
        raise ValueError('The filtered contour set does not match the detected signals.')
    target.parent.mkdir(parents=True, exist_ok=True)
    from .manual import manual_measurements
    emit(dict(kind='progress', phase='Reading manual cell volumes and signal distances', current=0, total=1))
    if source_scene: report['scene'] = str(source_scene)
    report['manual_measurements'] = manual_measurements(report, cancel)
    source_entry=LibraryEntry(Path(report['image']),Path(source_scene)) if source_scene else entry_for_path(report['image'])
    original_scene=load_dataset(source_entry).scene
    from ..scenes import parse_scene
    from ..annotation_state import split_measurements, timestamp
    from .edit_export import validate_roundtrip
    generated = parse_scene(scene_xml(target.name, surfaces, metadata, report))
    run_id, created_at = str(uuid4()), timestamp()
    cell_ids = {}
    ordered_points = []
    for original, surface in zip(surfaces, generated.surfaces):
        cell = int(original.name.rsplit(' ', 1)[-1])
        cell_ids[cell] = surface.uid
        ordered_points.extend(p for p in report['points'] if p['cell_id'] == cell)
    for surface in generated.surfaces:
        surface.source, surface.run_id, surface.created_at = 'program', run_id, created_at
    for measurement, point in zip(generated.measurements, ordered_points):
        measurement.source, measurement.run_id, measurement.created_at = 'program', run_id, created_at
        measurement.cell_uid = cell_ids[point['cell_id']]
        measurement.channel, measurement.score = point['channel'], point['score']
    combined = deepcopy(original_scene)
    split_measurements(combined)
    imported = {p['signal_id']: p for p in report['manual_measurements']['points']}
    imported_cells = {s.uid for s in combined.surfaces}
    for measurement in combined.measurements:
        row = imported.get(measurement.uid, {})
        if measurement.cell_uid is None and row.get('cell_id') in imported_cells:
            measurement.cell_uid = row['cell_id']
        if measurement.channel is None:
            measurement.channel = row.get('channel')
    combined.surfaces.extend(generated.surfaces)
    combined.measurements.extend(generated.measurements)
    from .signal_pairs import DEFAULT_MAX_DISTANCE_UM, update_scene_pairs
    maximum = report.get('signal_pairing', {}).get('maximum_distance_um', DEFAULT_MAX_DISTANCE_UM)
    report['signal_pairing'] = dict(method='closest_first_one_to_one_per_channel_pair', maximum_distance_um=maximum, unit='um')
    report['signal_pairs'] = update_scene_pairs(combined, metadata, maximum, run_id)
    if any(not w.startswith('No saved annotations found.') for w in original_scene.warnings):
        raise ValueError('Some imported annotations could not be read; export cannot safely discard them: '+'; '.join(original_scene.warnings))
    with tempfile.TemporaryDirectory(prefix='.fish-export-', dir=target.parent) as temporary:
        staging = Path(temporary)
        source = Path(report['image'])
        _copy_image(source, staging / target.name, emit, cancel)
        write_excel(staging / target.with_suffix('.xlsx').name, report)
        annotation_report = dict(report, engine='fish-scan-export', mode='export', image=target.name, scene=None,
                                 directory='.', cell_predictions=None, export_format='Standalone Imaris IMS',
                                 source_signal_report=str(Path(report_path).resolve()),
                                 scene_profile='Native Imaris Scene/Content and Scene/Data')
        annotation_report['annotation_run_id'] = run_id
        from .embedded_export import embed_scene
        digest = embed_scene(staging / target.name, combined, metadata, annotation_report)
        emit(dict(kind='progress', phase='Verifying exported scene', current=0, total=1))
        reopened = load_dataset(LibraryEntry(staging / target.name))
        validate_roundtrip(combined, reopened.scene)
        expected_surfaces = len(combined.surfaces)
        expected_points = sum(len(m.points) for m in combined.measurements)
        expected_lines = sum(len(m.edges) for m in combined.measurements)
        if (len(reopened.scene.surfaces) != expected_surfaces or reopened.scene.warnings
                or sum(len(group.points) for group in reopened.scene.measurements) != expected_points
                or sum(len(group.edges) for group in reopened.scene.measurements) != expected_lines):
            raise ValueError('Exported annotations failed the reload check.')
        checkpoint(cancel)
        _publish(staging, destinations)
    emit(dict(kind='progress', phase='Export complete', current=1, total=1))
    return dict(status='complete', image=str(target), scene=None,
                cells=len(surfaces), signals=len(report['points']), measurements=expected_lines,
                signal_pairs=len(report['signal_pairs']), image_sha256=digest, annotation_run_id=run_id, files=[str(path) for path in destinations])


def export_unannotated(source, target, emit=lambda event: None, cancel=None):
    """Make a standalone image with an empty scene and no source annotations."""
    source, target = Path(source).expanduser().resolve(), Path(target).expanduser().resolve()
    if target.suffix.lower() != '.ims':
        raise ValueError('Choose an export filename ending in .ims.')
    destinations = [target]
    if any(path.exists() for path in destinations):
        raise FileExistsError('The export would overwrite existing files. Choose a new name or folder.')
    metadata = ImarisReader(source).metadata
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.fish-export-', dir=target.parent) as temporary:
        staging = Path(temporary)
        _copy_image(source, staging / target.name, emit, cancel)
        from ..models import Scene
        from .embedded_export import embed_scene
        digest = embed_scene(staging / target.name, Scene(), metadata, dict(points=[]))
        reopened = load_dataset(LibraryEntry(staging / target.name))
        if (reopened.scene.surfaces or reopened.scene.measurements or reopened.scene.contours
                or reopened.scene.warnings):
            raise ValueError('The unannotated copy failed the reload check.')
        checkpoint(cancel)
        _publish(staging, destinations)
    return dict(status='complete', image=str(target), scene=None,
                cells=0, signals=0, source=str(source), image_sha256=digest,
                files=[str(path) for path in destinations])
