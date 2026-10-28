"""Physical-coordinate crops and supervision confined to fully marked cells."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from ..models import LibraryEntry
from ..reader import load_dataset
from ..scenes import read_scene
from ..neural.audit import read_references
from ..pipeline.runner import _json
from .config import SignalConfig


def normalized(volume):
    sample=volume.ravel()[::max(1,volume.size//1_000_000)]
    lo,hi=np.percentile(sample,(1,99.9))
    if hi<=lo:lo,hi=float(volume.min()),float(volume.max())
    if hi<=lo:return np.zeros(volume.shape,np.float32)
    return np.clip((volume.astype(np.float32)-lo)/(hi-lo),0,1)


def xyz_to_grid(points,origin,spacing):
    return (np.asarray(points).reshape(-1,3)-np.asarray(origin))[:,::-1]/np.asarray(spacing)


def at_labels(labels,points,origin,spacing):
    coords=xyz_to_grid(points,origin,spacing)
    return ndi.map_coordinates(labels,coords.T,order=0,mode='constant',cval=0).astype(np.int32)


def native_labels(labels,metadata,dataset):
    # Separable affine mapping retains calibrated voxel centers and discrete IDs.
    spacing=np.asarray(dataset.spacing[::-1])
    scale=spacing/metadata['spacing_zyx']
    offset=(np.asarray(dataset.center_origin)-metadata['origin_xyz'])[::-1]/metadata['spacing_zyx']
    return ndi.affine_transform(labels,np.diag(scale),offset=offset,
        output_shape=dataset.volumes[0].shape,order=0,mode='constant',cval=0,prefilter=False)


def marker_points(scenes):
    unique={}
    for path in scenes:
        for group in read_scene(Path(path)).measurements:
            for point,label,time_index in zip(group.points,group.labels,group.times):
                if label.strip().upper()=='A' and time_index==0:
                    key=tuple(np.round(point,4))
                    unique.setdefault(key,dict(xyz=point.tolist(),sources=[]))['sources'].append(str(path)+'::'+group.name)
    return list(unique.values())


def reference_memberships(points,references,origin,spacing,tolerance=0):
    coords=xyz_to_grid(points,origin,spacing)
    members=[[] for _ in coords]
    distances=np.full((len(coords),len(references)),np.inf)
    for index,(box,mask) in enumerate(references):
        local=coords-np.array([s.start for s in box])
        if not len(coords):continue
        inside=ndi.map_coordinates(mask.astype(np.uint8),local.T,order=0,mode='constant',cval=0)>0
        for p in np.flatnonzero(inside):members[p].append(index)
        if tolerance:
            distance=ndi.distance_transform_edt(~mask,sampling=spacing)
            distances[:,index]=ndi.map_coordinates(distance,local.T,order=1,mode='constant',cval=1e6)
    for i,found in enumerate(members):
        if not found and tolerance and len(references):
            near=int(distances[i].argmin())
            if distances[i,near]<=tolerance:found.append(near)
    return members


def channel_evidence(image,point,origin,spacing):
    center=xyz_to_grid([point],origin,spacing)[0]
    margin=np.ceil(.9/np.asarray(spacing)).astype(int)+1
    low=np.maximum(0,np.floor(center).astype(int)-margin)
    high=np.minimum(image.shape,np.ceil(center).astype(int)+margin+1)
    if np.any(high<=low):return dict(score=0.,contrast=0.)
    box=tuple(slice(int(a),int(b)) for a,b in zip(low,high))
    radius=np.linalg.norm((np.moveaxis(np.indices(tuple(high-low)),0,-1)+low-center)*spacing,axis=-1)
    patch=image[box];core=patch[radius<=.35];shell=patch[(radius>=.55)&(radius<=.9)]
    if not core.size or shell.size<8:return dict(score=0.,contrast=0.)
    baseline=float(np.median(shell));contrast=float(core.max()-baseline)
    noise=max(.04,float(np.median(np.abs(shell-baseline)))*1.4826)
    return dict(score=max(0.,contrast/noise),contrast=contrast)


def infer_channels(images,points,membrane,origin,spacing,config):
    rows=[]
    for point in points:
        scores=[channel_evidence(image,point['xyz'],origin,spacing) for image in images]
        order=sorted((c for c in range(len(images)) if c!=membrane),key=lambda c:scores[c]['score'],reverse=True)
        channel=order[0] if order else None
        contrast_min=config.channel_contrast_min if config.channel_contrast_min is not None else config.candidate_contrast
        good=channel is not None and scores[channel]['score']>=config.channel_score_min and scores[channel]['contrast']>=contrast_min
        if len(order)>1:good=good and scores[channel]['score']>=config.channel_margin*max(.01,scores[order[1]]['score'])
        rows.append(dict(point,channel=channel if good else None,channel_evidence=scores))
    return rows


def prepare_supervision(case,sources,row,config):
    """Load one scan with checked pixels, grid and sparse manual supervision."""
    source=sources[case['image']]
    for alias in case['audit']['aliases']:
        s=sources[alias];stat=Path(alias).stat()
        if stat.st_size!=s['size'] or stat.st_mtime_ns!=s['mtime_ns']:
            raise ValueError('Source changed since cell audit.')
        if s['fingerprint']!=source['fingerprint']:
            raise ValueError('Aliases differ in raw channels; a new split-safe signal audit is required.')
        for scene in s['scene_stats']:
            st=Path(scene['path']).stat()
            if st.st_size!=scene['size'] or st.st_mtime_ns!=scene['mtime_ns']:
                raise ValueError('Annotations changed since cell audit.')
    with np.load(row['predictions'],allow_pickle=False) as archive:
        labels=archive['labels'];metadata=json.loads(str(archive['metadata']))
    if metadata['fingerprint']!=source['fingerprint']:
        raise ValueError('Cell predictions do not match source pixels.')
    for field in ('spacing_zyx','origin_xyz'):
        if not np.allclose(metadata[field],case['audit'][field],rtol=1e-6,atol=1e-8):
            raise ValueError('Reference and prediction grids differ; use the matching calibrated collection audit.')
    scenes=sorted({scene for alias in case['audit']['aliases'] for scene in sources[alias]['scenes']})
    points=marker_points(scenes)
    references,_=read_references(Path(case['directory'])/'references.npz')
    memberships=reference_memberships([p['xyz'] for p in points],references,metadata['origin_xyz'],metadata['spacing_zyx'],config.assignment_tolerance_um)
    dataset=load_dataset(LibraryEntry(Path(case['image']),Path(case['scene']) if case['scene'] else None))
    digest=hashlib.sha256();digest.update(str(dataset.volumes[0].shape).encode())
    for volume in dataset.volumes:digest.update(memoryview(np.ascontiguousarray(volume)))
    if digest.hexdigest()!=source['fingerprint']:
        raise ValueError('Decoded source pixels differ from the audited fingerprint.')
    images=[normalized(v) for v in dataset.volumes];membrane=int(metadata['channel'])
    origin=np.asarray(dataset.center_origin);spacing=np.asarray(dataset.spacing[::-1])
    points=infer_channels(images,points,membrane,origin,spacing,config)
    marked={m for group in memberships for m in group}
    ambiguous={m for point,group in zip(points,memberships) if point['channel'] is None for m in group}
    usable=marked-ambiguous
    roi=np.zeros(labels.shape,np.uint8)
    for ref in usable:
        box,mask=references[ref];roi[box][mask]=1
    for ref in ambiguous:
        box,mask=references[ref];roi[box][mask]=0
    roi*=labels>0
    truth=[]
    for point,group in zip(points,memberships):
        point['manual_cells']=group
        point['cell_id']=int(at_labels(labels,[point['xyz']],metadata['origin_xyz'],metadata['spacing_zyx'])[0])
        point['usable']=point['channel'] is not None and bool(set(group)&usable) and not bool(set(group)&ambiguous) and point['cell_id']>0
        if point['usable']:truth.append(point)
    return dict(source=source,labels=labels,metadata=metadata,roi=roi,dataset=dataset,images=images,
                membrane=membrane,origin=origin,spacing=spacing,points=points,truth=truth,
                marked=marked,ambiguous=ambiguous,usable=usable,
                native=native_labels(labels,metadata,dataset))


def build_supervision_audit(cell_report, collection_audit, output, emit=print):
    """Build sparse marked-cell supervision without candidate finding or CNN patches."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    cells = json.loads(Path(cell_report).read_text())
    collection = json.loads(Path(collection_audit).read_text())
    if cells['status'] != 'complete' or collection['status'] != 'complete':
        raise ValueError('Completed cell predictions and collection audit required.')
    sources = {source['image']: source for source in collection['sources']}
    predictions = {row['image']: row for row in cells['files']}
    if len(predictions) != len(cells['files']):
        raise ValueError('Duplicate cell prediction rows.')
    report = dict(version=1, engine='signal-supervision', status='running',
        collection_audit=str(Path(collection_audit).resolve()),
        cell_report=str(Path(cell_report).resolve()), directory=str(output), cases=[],
        cell_checkpoint_sha256=cells['checkpoint_sha256'],
        boundary_checkpoint_sha256=cells.get('boundary_sha256'))
    for index, case in enumerate(collection['cases']):
        prepared = prepare_supervision(case, sources, predictions[case['image']], SignalConfig())
        dataset = prepared['dataset']
        report['cases'].append(dict(image=case['image'], scene=case.get('scene'),
            aliases=case['audit']['aliases'], split=case['audit']['split'], group=case['audit']['group'],
            fingerprint=case['audit']['fingerprint'], cell_predictions=predictions[case['image']]['predictions'],
            truth=prepared['truth'], markers=prepared['points'],
            membrane_channel=prepared['membrane'],
            channels=[dict(name=channel.name, color=list(channel.color)) for channel in dataset.metadata.channels]))
        emit(dict(kind='progress', phase='Preparing marked-cell signal supervision',
                  current=index+1, total=len(collection['cases']), file=case['image']))
        _json(output/'audit.json', report)
    report['status'] = 'complete'
    _json(output/'audit.json', report)
    return output/'audit.json'
