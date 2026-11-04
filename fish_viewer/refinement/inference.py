"""Refine existing instances from raw membrane pixels without creating new cells."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
from scipy import ndimage as ndi

from ..models import LibraryEntry
from ..reader import load_dataset
from ..neural.engine import runtime
from ..pipeline.runner import _json
from ..pipeline.segmentation import Cancelled, Prepared, checkpoint, mask_properties
from ..run_paths import new_run_directory
from ..signals.engine import image_fingerprint
from .config import RefinementConfig, read_model
from .data import coordinates, crop_geometry, prepare_membrane, seed_center, seed_image
from .network import network
from .preprocessing import preprocess_membrane


def resample_labels(labels, old_spacing, old_origin, shape, spacing, origin):
    scale=np.asarray(spacing)/old_spacing
    offset=(np.asarray(origin)-old_origin)[::-1]/old_spacing
    return ndi.affine_transform(labels,np.diag(scale),offset,output_shape=tuple(shape),
                                order=0,mode='constant',cval=0,prefilter=False).astype(np.uint32)


def surface_evidence(mask, image, spacing):
    """Sample outward surface normals, including concavities; no centroid rays."""
    spacing=np.asarray(spacing)
    inside=ndi.distance_transform_edt(np.pad(mask,1),sampling=spacing)[1:-1,1:-1,1:-1]
    outside=ndi.distance_transform_edt(~np.pad(mask,1),sampling=spacing)[1:-1,1:-1,1:-1]
    signed=ndi.gaussian_filter(outside-inside,sigma=.6)
    gradient=np.asarray(np.gradient(signed,*spacing))
    boundary=np.argwhere(mask & ~ndi.binary_erosion(mask))
    if not len(boundary):
        return dict(surface_support=0.,surface_contrast=0.)
    points=boundary[np.linspace(0,len(boundary)-1,min(512,len(boundary)),dtype=int)].T
    normals=gradient[(slice(None),*points)]
    norms=np.linalg.norm(normals,axis=0)
    normals=normals/np.maximum(norms,1e-6)
    profiles=np.array([ndi.map_coordinates(image,points+normals*d/spacing[:,None],
                                          order=1,mode='nearest',prefilter=False)
                       for d in (-.9,-.3,0,.3,.9)])
    rim=profiles[1:4].max(axis=0)
    context=(profiles[0]+profiles[-1])/2
    return dict(surface_support=float(np.mean((rim>=.08)&(rim-context>=.025)&(norms>1e-5))),
                surface_contrast=float(np.mean(rim-context)))


def seeded_component(mask, seed):
    seed=tuple(np.clip(np.rint(seed).astype(int),0,np.array(mask.shape)-1))
    components,_=ndi.label(mask)
    label=int(components[seed])
    return components==label if label else np.zeros_like(mask)


def refine(labels, prepared, model, config, emit=lambda e:None, cancel=None, evidence_threshold=.5):
    result=labels.copy(); diagnostics=[]
    boxes=ndi.find_objects(labels)
    ids=[i for i,box in enumerate(boxes,1) if box is not None]
    spacing=np.asarray(prepared.spacing)
    for index, cell in enumerate(ids):
        checkpoint(cancel)
        emit(dict(kind='progress',phase='Refining cell boundaries',current=index,total=len(ids),cell=cell))
        box=boxes[cell-1]; original=labels[box]==cell
        center,width=crop_geometry(box,spacing,config.margin_um)
        seed=seed_center(original,spacing)+np.array([s.start for s in box])
        step=width/config.patch_size
        coords=coordinates(center,width,spacing,config.patch_size)
        sample=ndi.map_coordinates(prepared.image,coords,order=1,mode='constant',cval=0,prefilter=False)
        seed_patch=(seed-center)*spacing/step+(config.patch_size-1)/2
        inputs=np.stack([preprocess_membrane(sample,step,config),seed_image(seed_patch,step,config)],axis=-1)[None]
        logits=np.asarray(model(inputs,training=False))[0,...,0]
        if not np.isfinite(logits).all():
            raise ValueError('Non-finite boundary predictions.')
        probability=1/(1+np.exp(-np.clip(logits,-60,60)))
        low=np.maximum(0,np.floor(center-width/2/spacing).astype(int))
        high=np.minimum(labels.shape,np.ceil(center+width/2/spacing).astype(int)+1)
        region=tuple(slice(int(a),int(b)) for a,b in zip(low,high))
        grid=np.indices(tuple(high-low),dtype=np.float32)+low[:,None,None,None]
        at=(grid-center[:,None,None,None])*spacing[:,None,None,None]/step+(config.patch_size-1)/2
        scores=ndi.map_coordinates(probability,at,order=1,mode='constant',cval=0,prefilter=False)
        proposed=seeded_component(scores>=config.threshold,seed-low)
        current=labels[region]==cell
        # Other existing instances keep their voxels; refinement cannot steal them.
        collision=int(np.sum(proposed & (result[region]>0) & (result[region]!=cell)))
        proposed &= (result[region]==0)|(result[region]==cell)
        proposed=seeded_component(proposed,seed-low)
        volume_ratio=float(proposed.sum()/max(1,current.sum()))
        overlap=float(np.sum(proposed & current)/max(1,current.sum()))
        evidence=surface_evidence(proposed,prepared.image[region],spacing) if proposed.any() else dict(surface_support=0.,surface_contrast=0.)
        reason=None
        if not proposed.any():reason='No connected cell at the detector seed'
        elif overlap<config.min_overlap:reason='Refinement disagrees with the detected instance'
        elif not config.min_volume_ratio<=volume_ratio<=config.max_volume_ratio:reason='Large unsupported volume change'
        elif evidence['surface_support']<evidence_threshold:reason='Insufficient membrane evidence on refined surface'
        elif any(np.any(proposed.take(i,axis=a)) for a in range(3) for i in (0,-1)):
            reason='Refinement touches its search-region boundary'
        accepted=reason is None
        if accepted:
            result[region][current]=0
            result[region][proposed]=cell
        diagnostics.append(dict(cell_id=cell,accepted=accepted,fallback_reason=reason,
                                overlap=overlap,volume_ratio=volume_ratio,collision_voxels=collision,**evidence))
    emit(dict(kind='progress',phase='Cell boundaries ready',current=len(ids),total=len(ids)))
    return result,diagnostics


def redraw_proposals(labels, prepared, model, config, emit=lambda e: None, cancel=None,
                     evidence_threshold=.65):
    """Use detector instances only as boxes/seeds; rejected crops leave no mask."""
    from ..proposals.inference import contour_proposals
    spacing = np.asarray(prepared.spacing)
    origin = np.asarray(prepared.origin)
    proposals = []
    for cell, box in enumerate(ndi.find_objects(labels), 1):
        if box is None:
            continue
        low = np.array([s.start for s in box])
        high = np.array([s.stop - 1 for s in box])
        seed = seed_center(labels[box] == cell, spacing) + low
        proposals.append(dict(proposal_id=cell, center_xyz=((seed * spacing)[::-1] + origin).tolist(),
                             box_low_xyz=((low * spacing)[::-1] + origin).tolist(),
                             box_high_xyz=((high * spacing)[::-1] + origin).tolist()))
    return contour_proposals(prepared, proposals, model, config, emit, cancel, evidence_threshold)


def apply(cells_report, model_directory, output, device='auto',emit=lambda e:None,cancel=None,audit=None,redraw=False):
    directory=new_run_directory(output,'boundary-apply')
    report=dict(version=1,engine='seeded-unet3d',mode='apply',status='running',directory=str(directory),
                parent_cells_report=str(Path(cells_report).resolve()),model_directory=str(Path(model_directory).resolve()),
                files=[],failures=[])
    started=time.monotonic()
    def notify(event):
        if event['kind']=='progress':
            report['progress']={k:v for k,v in event.items() if k!='kind'}
            _json(directory/'report.json',report)
        emit(event)
    emit(dict(kind='run_started',directory=str(directory)))
    _json(directory/'report.json',report)
    try:
        manifest=read_model(model_directory);config=RefinementConfig(**manifest['config'])
        checkpoint_sha256=hashlib.sha256((Path(model_directory)/manifest['checkpoint']).read_bytes()).hexdigest()
        report['checkpoint_sha256']=checkpoint_sha256
        tf,_,_,info=runtime(device,config.seed);report['runtime']=info
        model=network(tf,config);model.load_weights(Path(model_directory)/manifest['checkpoint'])
        source=json.loads(Path(cells_report).read_text())
        if source['status']!='complete':raise ValueError('Completed cell inference is required.')
        cache={}
        if audit:
            collection=json.loads(Path(audit).read_text())
            if collection['status']!='complete':raise ValueError('Incomplete finer-resolution cache.')
            cache={c['image']:c for c in collection['cases']}
        for index,row in enumerate(source['files']):
            checkpoint(cancel)
            notify(dict(kind='progress',phase='Loading membrane image for boundary refinement',
                        current=index,total=len(source['files']),file=row['image']))
            with np.load(row['predictions'],allow_pickle=False) as archive:
                old=archive['labels'];metadata=json.loads(str(archive['metadata']))
            if row['image'] in cache:
                c=cache[row['image']];a=c['audit']
                if a['fingerprint']!=metadata['fingerprint'] or a['channel']!=metadata['channel']:
                    raise ValueError('Cached membrane differs from the detected cells.')
                prepared=Prepared(np.load(Path(c['directory'])/'image.npy',mmap_mode='r'),
                                  tuple(a['spacing_zyx']),tuple(a['origin_xyz']),a['channel'],
                                  metadata['channel_scores'],a['fingerprint'],{}, {})
            else:
                dataset=load_dataset(LibraryEntry(Path(row['image'])))
                fingerprint=image_fingerprint(dataset)
                if fingerprint!=metadata['fingerprint']:raise ValueError('Source pixels changed since cell detection.')
                prepared=prepare_membrane(dataset,metadata['channel'],config,fingerprint)
            labels=resample_labels(old,np.asarray(metadata['spacing_zyx']),np.asarray(metadata['origin_xyz']),
                                   prepared.image.shape,prepared.spacing,prepared.origin)
            if redraw:
                labels,diagnostics=redraw_proposals(labels,prepared,model,config,notify,cancel,
                    max(.65,manifest.get('surface_evidence_threshold',.5)))
            else:
                labels,diagnostics=refine(labels,prepared,model,config,notify,cancel,
                    manifest.get('surface_evidence_threshold',.5))
            folder=directory/f'{index+1:04d}-{Path(row["image"]).stem}';folder.mkdir()
            metadata.update(spacing_zyx=prepared.spacing,origin_xyz=prepared.origin,cells=[],
                            refinement=dict(model=str(Path(model_directory).resolve()),config=asdict(config),
                                            checkpoint_sha256=checkpoint_sha256,cells=diagnostics,
                                            policy=('Detector boxes and seeds only; unsupported crop contours are rejected; '
                                                    'no detector-mask fallback. No roundness filter.' if redraw else
                                                    'Refine existing instances only; invalid refinements retain detector contours. No roundness filter.')))
            metadata.pop('metrics',None)
            for cell,box in enumerate(ndi.find_objects(labels),1):
                if box is None:continue
                mask=labels[box]==cell; props=mask_properties(mask,np.asarray(prepared.spacing))
                center=(np.argwhere(mask).mean(0)+[s.start for s in box])*prepared.spacing
                props.update(label=cell,center_xyz=(center[::-1]+prepared.origin).tolist())
                metadata['cells'].append(props)
            path=folder/'contours.npz'
            np.savez_compressed(path,labels=labels,metadata=np.array(json.dumps(metadata,allow_nan=False)))
            refined_row=dict(row,predictions=str(path),refinement=diagnostics)
            if redraw:
                refined_row['cells']=len(metadata['cells'])
            if 'metrics' in refined_row:
                refined_row['detector_metrics']=refined_row.pop('metrics')
            report['files'].append(refined_row)
            _json(directory/'report.json',report)
        report['summary']=dict(proposals=sum(len(r['refinement']) for r in report['files']),
                               cells=sum(int(d['accepted']) if redraw else 1 for r in report['files'] for d in r['refinement']),
                               refined=sum(d['accepted'] for r in report['files'] for d in r['refinement']))
        report['status']='complete'
    except Cancelled:
        report['status']='cancelled'
    except Exception:
        report.update(status='failed',error=traceback.format_exc())
    report['elapsed_s']=time.monotonic()-started;_json(directory/'report.json',report)
    emit(dict(kind='finished',report=report))
    return report
