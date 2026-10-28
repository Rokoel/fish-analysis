"""Native Scene/Content meshes and Scene/Data XML in a standalone IMS copy."""

from configparser import ConfigParser
import hashlib
import json
import xml.etree.ElementTree as ET

import h5py
import numpy as np

from ..annotation_state import scene_state
from ..reader import ImarisReader, decode
from ..pipeline.segmentation import checkpoint
from .imaris_scene import _image_context


def _text(group, name, value):
    group.attrs[name] = np.frombuffer(str(value).encode('utf-8'), dtype='S1')


def convert_legacy_image(source, destination, emit, cancel):
    """Convert the legacy TIFF container, preserving every channel/time voxel."""
    import tifffile
    reader = ImarisReader(source)
    metadata = reader.metadata
    with tifffile.TiffFile(source) as original, h5py.File(destination, 'w') as image:
        parser = ConfigParser(interpolation=None, strict=False)
        parser.optionxform = str
        parser.read_string(decode(original.pages[0].tags['ImageDescription'].value))
        for key, value in dict(ImarisDataSet='ImarisDataSet', FormatVersion='5.5.0',
                               DataSetDirectoryName='DataSet', DataSetInfoDirectoryName='DataSetInfo',
                               ThumbnailDirectoryName='Thumbnail').items():
            _text(image, key, value)
        info = image.create_group('DataSetInfo')
        for section in parser.sections():
            name = ('Image' if section == 'Image 0.0.0' else
                    'Channel ' + section.rsplit('.', 1)[-1] if section.startswith('Channel 0.0.0.') else section)
            group = info.require_group(name)
            for key, value in parser[section].items():
                _text(group, key, value)
        for time in metadata.times:
            for channel, volume in enumerate(reader.read(time)):
                checkpoint(cancel)
                group = image.create_group(f'DataSet/ResolutionLevel 0/TimePoint {time}/Channel {channel}')
                for axis, size in zip('ZYX', volume.shape):
                    _text(group, f'ImageSize{axis}', size)
                group.create_dataset('Data', data=volume, chunks=True, compression='gzip', compression_opts=1)
                emit(dict(kind='progress', phase='Copying legacy Imaris image voxels', current=time + 1,
                          total=len(metadata.times)))
        image.create_dataset('Thumbnail/Data', data=np.zeros((32, 128), np.uint8))


def embed_scene(path, scene, metadata, report):
    from .edit_export import editable_scene_xml
    root = ET.fromstring(editable_scene_xml(path.name, scene, metadata))
    root.set('Version', 'Imaris 7.4.2')
    times, _ = _image_context(metadata)
    with h5py.File(path, 'r+') as image:
        for name in list(image):
            if name.startswith('Scene') or name == 'FISHAnalysis':
                del image[name]
        _text(image, 'SceneDirectoryName', 'Scene')
        content = image.create_group('Scene/Content')
        content.attrs['NumberOfSurfaces'] = np.uint64(len(scene.surfaces))
        for index, (surface, viewer) in enumerate(zip(scene.surfaces, root.iter('bpSurfacesViewer'))):
            objects = viewer.find('bpSurfaces')
            for child in list(objects):
                if child.tag in ('surface', 'bpSurfacesId'):
                    objects.remove(child)
            for child in viewer.findall('bpSurfacesId'):
                viewer.remove(child)
            ident = 300001 + index
            viewer.insert(list(viewer).index(objects) + 1, ET.Element('bpSurfacesId', Value=str(ident)))
            group = content.create_group(f'Surfaces{index}')
            group.attrs['Id'] = np.uint64(ident)
            for key, tag in [('Name', 'name'), ('Description', 'description'), ('HistoryLog', 'historyLog'), ('Unit', 'unit')]:
                _text(group, key, objects.findtext(tag, ''))
            _text(group, 'Material', ET.tostring(objects.find('bpMaterial'), encoding='unicode'))
            vertices, faces = np.asarray(surface.vertices, float), np.asarray(surface.triangles, np.uint64)
            normals = np.zeros_like(vertices)
            face_normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                                    vertices[faces[:, 2]] - vertices[faces[:, 0]])
            for corner in range(3):
                np.add.at(normals, faces[:, corner], face_normals)
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
            group.create_dataset('Vertices', data=np.column_stack((vertices, normals)))
            group.create_dataset('Triangles', data=faces)
            group.create_dataset('TimeNVerticesNTriangles', data=np.asarray([[surface.time, len(vertices), len(faces)]], np.uint64))
            group.create_dataset('TimeInfos', data=np.asarray([times[surface.time].encode()], dtype='S23'))
        ET.indent(root)
        data = ET.tostring(root, encoding='utf-8', short_empty_elements=False) + b'\n'
        native = image.create_dataset('Scene/Data', (1,), dtype=h5py.vlen_dtype(np.dtype('uint8')))
        native[0] = np.frombuffer(data, np.uint8)
        report['editor_state'] = dict(**report.get('editor_state', {}), **scene_state(scene, data))
        image.create_dataset('FISHAnalysis/Annotations', data=np.frombuffer(json.dumps(report, allow_nan=False).encode('utf-8'), np.uint8),
                             compression='gzip')
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
