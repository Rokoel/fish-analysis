"""Compare exports to native FISH files, not only to our permissive reader."""
from copy import deepcopy
import base64
import bz2
import hashlib
import re
import threading
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pytest

from fish_viewer.reader import ImarisReader, entry_for_path, load_dataset
from fish_viewer.scenes import numbers, read_scene
from fish_viewer.workflow.export import export_unannotated
from fish_viewer.workflow.imaris_scene import scene_xml, validate_scene


@pytest.fixture
def native_export(reference):
    source = read_scene(reference.with_suffix('.imx'))
    metadata = ImarisReader(reference).metadata
    surface = deepcopy(source.surfaces[0])
    surface.name = 'Predicted cell 7'
    points = [dict(signal_id=i+1, cell_id=7, channel=0, xyz=point.tolist(), score=.9991)
              for i, point in enumerate(source.measurements[0].points)]
    report = dict(points=points, membrane_channel=2, channels=[dict(name=c.name, color=c.color) for c in metadata.channels])
    return scene_xml('cells & signals.ims', [surface], metadata, report), metadata, surface, points


@pytest.mark.integration
def test_complete_field_structure_matches_native_fish(reference, native_export):
    data, metadata, surface, points = native_export
    original = ET.parse(reference.with_suffix('.imx')).getroot()
    root = ET.fromstring(data)

    def compare(actual, native):
        assert actual.tag == native.tag
        assert set(actual.attrib) == set(native.attrib), actual.tag
        expected = {e.tag: e for e in native}
        # Ignore repeated records/groups, but require all native field types and
        # ordering. This compares directly to the actual Imaris-written example.
        assert list(dict.fromkeys(e.tag for e in actual)) == list(expected), actual.tag
        for child in actual:
            compare(child, expected[child.tag])

    compare(root, original)
    assert data.startswith(b'<bpImarisApplication ')
    assert b'<?xml' not in data and b'<!--' not in data
    assert b'\n' not in data.replace(b'\r\n', b'')
    assert root.find('bpCommandLoadDataSet').get('file') == 'cells & signals.ims'
    assert root.find('.//bpSurfacesCreationParameters').get('mSourceChannelIndex') == '2'
    native_time = original.find('.//timeInfo').attrib
    assert root.find('.//timeInfo').attrib == native_time
    ids = [e.findtext('id') for e in root.iter('bpSurfaceComponent')]
    assert len(ids) == len(set(ids))
    markers = list(root.iter('bpMeasurementPoints'))
    assert len(markers) == len(points)
    assert all(len(m.findall('Point')) == 1 and m.findtext('LineMode') == 'Polygon' for m in markers)
    # The source's scene screenshot includes annotations. Only reuse the IMS
    # thumbnail, which belongs to the unchanged image, not that scene screenshot.
    thumbnail = root.find('Thumbnail/bpImageField')
    with h5py.File(reference) as f:
        assert bz2.decompress(base64.b64decode(thumbnail.text)) == f['Thumbnail/Data'][()].tobytes()


@pytest.mark.integration
def test_native_mesh_blocks_preserve_geometry(native_export):
    data, metadata, surface, _ = native_export
    root = ET.fromstring(data)
    mesh = root.find('.//surface')
    for tag, width in [('vertices', 6), ('triangles', 3)]:
        blocks = mesh.findall(tag)
        assert len(blocks) > 1  # Exercise chunk boundaries on a real large mesh.
        for i, block in enumerate(blocks):
            lines = re.findall(r'"([^"\n]*)"', block.text)
            assert 1 <= len(lines) <= 40
            records = numbers(block.text).reshape(-1, width)
            assert len(records) == 400 if i < len(blocks)-1 else 1 <= len(records) <= 400
            for line in lines:
                assert len(numbers(line)) % width == 0
                assert len(numbers(line)) <= 10 * width
    vertices = numbers(' '.join(e.text for e in mesh.findall('vertices'))).reshape(-1, 6)
    np.testing.assert_allclose(vertices[:, :3], surface.vertices, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(np.linalg.norm(vertices[:, 3:], axis=1), 1, atol=1e-6)
    validate_scene(root, metadata)


@pytest.mark.integration
@pytest.mark.parametrize('damage', ['missing_time', 'missing_tracks', 'duplicate_id', 'bad_index', 'large_block', 'unknown_line_mode'])
def test_strict_validation_rejects_previous_export_defects(native_export, damage):
    data, metadata, _, _ = native_export
    root = ET.fromstring(data)
    if damage == 'missing_time':
        obj = root.find('.//bpSurfaces'); obj.remove(obj.find('timeInfo'))
    elif damage == 'missing_tracks':
        obj = root.find('.//bpSurfacesViewer'); obj.remove(obj.find('bpTracksProperties'))
    elif damage == 'duplicate_id':
        list(root.iter('id'))[1].text = list(root.iter('id'))[0].text
    elif damage == 'bad_index':
        root.find('.//triangles').text = '"999999999 0 1 "'
    elif damage == 'large_block':
        root.find('.//vertices').text *= 2
    else:
        root.find('.//LineMode').text = 'None'
    with pytest.raises(ValueError):
        validate_scene(root, metadata)


@pytest.mark.integration
def test_annotation_free_copy_preserves_source_and_all_image_bytes(reference, tmp_path):
    original = load_dataset(entry_for_path(reference))
    assert original.scene.surfaces and original.scene.measurements
    source_scene = reference.with_suffix('.imx').read_bytes()
    digest = hashlib.sha256(reference.read_bytes()).hexdigest()
    target = tmp_path / 'test unannotated.ims'
    result = export_unannotated(reference, target)
    assert result['image_sha256'] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert hashlib.sha256(reference.read_bytes()).hexdigest() == digest
    assert reference.with_suffix('.imx').read_bytes() == source_scene
    reopened = load_dataset(entry_for_path(target))
    assert not reopened.scene.surfaces and not reopened.scene.measurements and not reopened.scene.contours
    assert not reopened.scene.warnings
    for a, b in zip(original.volumes, reopened.volumes):
        np.testing.assert_array_equal(a, b)
    with h5py.File(target) as image:
        root = ET.fromstring(image['Scene/Data'][0].tobytes())
        assert root.findall('.//Point') == root.findall('.//surface') == root.findall('.//bpManualSurface') == []
        assert {'DataSet', 'DataSetInfo', 'Thumbnail', 'Scene', 'FISHAnalysis'} <= set(image)


def test_unannotated_export_does_not_overwrite_and_cancels_cleanly(ims_file, tmp_path):
    from fish_viewer.pipeline.segmentation import Cancelled
    target = tmp_path / 'clean.ims'
    result = export_unannotated(ims_file, target)
    with pytest.raises(FileExistsError):
        export_unannotated(ims_file, target)
    assert hashlib.sha256(target.read_bytes()).hexdigest() == result['image_sha256']
    cancel = threading.Event()
    with pytest.raises(Cancelled):
        export_unannotated(ims_file, tmp_path / 'cancelled.ims', lambda _: cancel.set(), cancel)
    assert not list(tmp_path.glob('cancelled.*'))
    assert not list(tmp_path.glob('.fish-export-*'))


def test_rejects_embedded_annotations_instead_of_calling_them_unannotated(ims_file, tmp_path):
    with h5py.File(ims_file, 'r+') as image:
        image.create_group('Scene8/Content')
    export_unannotated(ims_file, tmp_path / 'clean.ims')
    with h5py.File(tmp_path / 'clean.ims') as clean, h5py.File(ims_file) as original:
        assert 'Scene8' not in clean and 'Scene8' in original
        np.testing.assert_array_equal(clean['DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data'],original['DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data'])
