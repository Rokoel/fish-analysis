"""Predict points/boxes, then segment independent small volumes; no StarDist required."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
from scipy import ndimage as ndi
from skimage.transform import resize

from ..models import LibraryEntry
from ..reader import load_dataset
from ..neural.engine import runtime
from ..pipeline.config import PipelineConfig
from ..pipeline.evaluation import evaluate
from ..pipeline.runner import _json
from ..pipeline.segmentation import Cancelled,Prepared,checkpoint,mask_properties,prepare
from ..neural.audit import read_references
from ..run_paths import new_run_directory
from ..refinement.config import RefinementConfig,read_model as read_boundary_model
from ..refinement.data import coordinates,prepare_membrane,seed_image
from ..refinement.inference import seeded_component,surface_evidence
from ..refinement.network import network as boundary_network
from .config import ProposalConfig,read_model
from .network import network


def detector_grid(prepared,config):
    physical=np.array(prepared.image.shape)*prepared.spacing
    shape=np.maximum(2,np.rint(physical/config.voxel_um).astype(int))
    spacing=physical/shape
    origin=np.array(prepared.origin)-np.array(prepared.spacing)[::-1]/2+spacing[::-1]/2
    image=resize(prepared.image,tuple(shape),order=1,preserve_range=True,anti_aliasing=True).astype(np.float32)
    return image,spacing,origin


def predict_proposals(prepared,model,config,emit=lambda e:None,cancel=None):
    image,spacing,origin=detector_grid(prepared,config)
    size=config.patch_size;out=config.patch_size//config.stride
    shape=np.array(image.shape);padded_shape=np.maximum(size,((shape+7)//8)*8)
    padded=np.pad(image,[(0,int(n-s)) for n,s in zip(padded_shape,shape)])
    # Half-tile overlap supplies context at internal seams; strongest center wins.
    starts=[sorted(set([*range(0,int(n-size)+1,size//2),int(n-size)])) for n in padded_shape]
    tile_starts=[(z,y,x) for z in starts[0] for y in starts[1] for x in starts[2]]
    heat=np.zeros(tuple(padded_shape//config.stride),np.float32)
    boxes=np.zeros(heat.shape+(6,),np.float32)
    for index,start in enumerate(tile_starts):
        checkpoint(cancel)
        emit(dict(kind='progress',phase='Finding cell centers and rough boxes',current=index,total=len(tile_starts)))
        crop=tuple(slice(s,s+size) for s in start)
        raw=np.asarray(model(padded[crop][None,...,None],training=False))[0]
        if raw.shape!=(out,out,out,7) or not np.isfinite(raw).all():raise ValueError('Invalid point-and-box predictions.')
        scores=1/(1+np.exp(-np.clip(raw[...,0],-60,60)))
        # Outputs without sufficient context at internal tile edges are ignored.
        good=np.ones(scores.shape,bool);border=4
        for a,s in enumerate(start):
            if s>0:
                sl=[slice(None)]*3;sl[a]=slice(0,border);good[tuple(sl)]=False
            if s+size<padded_shape[a]:
                sl=[slice(None)]*3;sl[a]=slice(-border,None);good[tuple(sl)]=False
        scores*=good
        region=tuple(slice(s//config.stride,s//config.stride+out) for s in start)
        stronger=scores>heat[region]
        heat[region][stronger]=scores[stronger];boxes[region][stronger]=raw[...,1:][stronger]
    footprint=tuple(max(3,2*int(np.ceil(config.separation_um/(s*config.stride)))+1) for s in spacing)
    peak=(heat>=config.threshold)&(heat==ndi.maximum_filter(heat,size=footprint))
    points=np.argwhere(peak)
    points=points[np.argsort(-heat[tuple(points.T)])] if len(points) else points
    proposals=[]
    for point in points:
        center=point*config.stride
        if np.any(center>=shape):continue
        distances=boxes[tuple(point)]*config.box_scale_um
        widths=distances[:3]+distances[3:]
        if np.any(distances<.3) or np.any(widths<2.) or np.any(widths>18.):continue
        xyz=center[::-1]*spacing[::-1]+origin
        if any(np.linalg.norm(xyz-np.array(p['center_xyz']))<config.separation_um for p in proposals):continue
        low=center-distances[:3]/spacing;high=center+distances[3:]/spacing
        proposals.append(dict(proposal_id=len(proposals)+1,score=float(heat[tuple(point)]),
            center_xyz=xyz.tolist(),box_low_xyz=(low[::-1]*spacing[::-1]+origin).tolist(),
            box_high_xyz=(high[::-1]*spacing[::-1]+origin).tolist()))
    emit(dict(kind='progress',phase='Cell centers and boxes ready',current=len(tile_starts),total=len(tile_starts)))
    return proposals


def contour_proposals(prepared,proposals,model,config,emit=lambda e:None,cancel=None,evidence_threshold=.5):
    labels=np.zeros(prepared.image.shape,np.uint32)
    spacing=np.array(prepared.spacing);origin=np.array(prepared.origin)
    visible=ndi.gaussian_filter(prepared.image,sigma=.3/spacing)
    diagnostics=[]
    for index,proposal in enumerate(proposals):
        checkpoint(cancel)
        emit(dict(kind='progress',phase='Drawing contours within padded cell boxes',current=index,total=len(proposals)))
        low=(np.array(proposal['box_low_xyz'])-origin)[::-1]/spacing
        high=(np.array(proposal['box_high_xyz'])-origin)[::-1]/spacing
        center=(low+high)/2;seed=(np.array(proposal['center_xyz'])-origin)[::-1]/spacing
        width=float(np.max((high-low)*spacing)+2*config.margin_um);step=width/config.patch_size
        coords=coordinates(center,width,spacing,config.patch_size)
        sample=ndi.map_coordinates(prepared.image,coords,order=1,mode='constant',cval=0,prefilter=False)
        seed_patch=(seed-center)*spacing/step+(config.patch_size-1)/2
        inputs=np.stack([sample,seed_image(seed_patch,step,config)],axis=-1)[None]
        logits=np.asarray(model(inputs,training=False))[0,...,0]
        if not np.isfinite(logits).all():raise ValueError('Non-finite crop contour predictions.')
        probability=1/(1+np.exp(-np.clip(logits,-60,60)))
        lower=np.maximum(0,np.floor(center-width/2/spacing).astype(int))
        upper=np.minimum(labels.shape,np.ceil(center+width/2/spacing).astype(int)+1)
        if (upper<=lower).any():continue
        region=tuple(slice(int(a),int(b)) for a,b in zip(lower,upper))
        grid=np.indices(tuple(upper-lower),dtype=np.float32)+lower[:,None,None,None]
        at=(grid-center[:,None,None,None])*spacing[:,None,None,None]/step+(config.patch_size-1)/2
        scores=ndi.map_coordinates(probability,at,order=1,mode='constant',cval=0,prefilter=False)
        mask=seeded_component(scores>=config.threshold,seed-lower)
        evidence=surface_evidence(mask,visible[region],spacing) if mask.any() else dict(surface_support=0.,surface_contrast=0.)
        reason=None;volume=float(mask.sum()*np.prod(spacing))
        if not mask.any():reason='No connected cell at predicted center'
        elif volume<5.:reason='Contour is too small to be a complete cell'
        elif evidence['surface_support']<evidence_threshold:reason='Insufficient visible membrane evidence'
        elif evidence['surface_contrast']<.04:reason='Weak membrane contrast against local context'
        elif any(np.any(mask.take(i,axis=a)) for a in range(3) for i in (0,-1)):
            reason='Incomplete contour touches crop or scan boundary'
        elif np.any(labels[region][mask]) and np.mean(labels[region][mask]>0)>.4:
            reason='Duplicate or conflicting cell proposal'
        cell=None
        if reason is None:
            # Existing accepted cells retain ownership. The resulting component
            # is checked again, so no overwritten or detached fragments survive.
            mask &= labels[region]==0
            mask=seeded_component(mask,seed-lower)
            if not mask.any():reason='Predicted center belongs to an accepted cell'
            elif mask.sum()*np.prod(spacing)<5.:reason='Contour is too small after resolving overlap'
            else:
                evidence=surface_evidence(mask,visible[region],spacing)
                if evidence['surface_support']<evidence_threshold or evidence['surface_contrast']<.04:
                    reason='Contour lost visible membrane support after resolving overlap'
                else:
                    cell=int(labels.max())+1;labels[region][mask]=cell
        diagnostics.append(dict(proposal,cell_id=cell,accepted=reason is None,rejection_reason=reason,
                                volume_um3=volume,**evidence))
    emit(dict(kind='progress',phase='Crop contours ready',current=len(proposals),total=len(proposals)))
    return labels,diagnostics


def prepare_case(case,collection,boundary_config,emit,cancel,channel_config=None,cell_channel=None):
    if collection:
        a=case['audit']
        if cell_channel is not None and cell_channel != a['channel']:
            raise ValueError('Cannot change channels in an existing cached audit.')
        return Prepared(np.load(Path(case['directory'])/'image.npy',mmap_mode='r'),
            tuple(a['spacing_zyx']),tuple(a['origin_xyz']),a['channel'],a['channel_scores'],a['fingerprint'],{},{})
    dataset=load_dataset(LibraryEntry(Path(case['image'])))
    # Preserve the channel-selection geometry used by the training audit.
    geometry=PipelineConfig(**(channel_config or dict(min_diameter_um=2.5,max_diameter_um=32.)))
    if cell_channel is not None:
        from dataclasses import replace
        geometry=replace(geometry,cell_channel=cell_channel)
    def progress(event):
        # Previews are NumPy image arrays for the full viewer; this worker uses
        # JSON progress messages and the focused app already has a raw preview.
        emit({k:v for k,v in event.items() if k!='previews'})
    selected=prepare(dataset,geometry,progress,cancel,background_correct=False,max_voxels=None)
    prepared=prepare_membrane(dataset,selected.channel,boundary_config,selected.fingerprint)
    prepared.channel_scores=selected.channel_scores
    return prepared


def detect(model_directory,output,device='auto',emit=lambda e:None,cancel=None,image=None,audit=None,split=None,threshold=None,cell_channel=None):
    directory=new_run_directory(output,'cell-centers')
    report=dict(version=1,engine='cell-center-box3d',mode='detect',status='running',directory=str(directory),
                model_directory=str(Path(model_directory).resolve()),files=[])
    started=time.monotonic()
    last_save=[0.]
    def notify(event):
        if event['kind']=='progress':
            report['progress']={k:v for k,v in event.items() if k!='kind'}
            if time.monotonic()-last_save[0]>=.5:
                _json(directory/'report.json',report);last_save[0]=time.monotonic()
        emit(event)
    _json(directory/'report.json',report)
    try:
        from dataclasses import replace
        manifest=read_model(model_directory);config=ProposalConfig(**manifest['config'])
        if threshold is not None:config=replace(config,threshold=threshold).validate()
        report['config']=asdict(config);report['checkpoint_sha256']=hashlib.sha256((Path(model_directory)/manifest['checkpoint']).read_bytes()).hexdigest()
        tf,_,_,info=runtime(device,config.seed);report['runtime']=info
        detector=network(tf,config);detector.load_weights(Path(model_directory)/manifest['checkpoint'])
        collection=json.loads(Path(audit).read_text()) if audit else None
        if collection and collection['status']!='complete':raise ValueError('Completed calibrated cache required.')
        cases=[c for c in collection['cases'] if c['audit']['split']==split] if collection else [dict(image=str(Path(image).resolve()))]
        for number,case in enumerate(cases):
            checkpoint(cancel);notify(dict(kind='progress',phase='Preparing membrane image',current=number,total=len(cases),file=case['image']))
            prepared=prepare_case(case,collection,RefinementConfig(),notify,cancel,manifest.get('channel_selection'),cell_channel)
            proposals=predict_proposals(prepared,detector,config,notify,cancel)
            report['files'].append(dict(image=case['image'],split=split or 'apply',channel=prepared.channel,
                proposals=proposals,spacing_zyx=prepared.spacing,origin_xyz=prepared.origin,fingerprint=prepared.fingerprint))
            _json(directory/'report.json',report)
        report['summary']=dict(scans=len(cases),proposals=sum(len(r['proposals']) for r in report['files']));report['status']='complete'
    except Cancelled:report['status']='cancelled'
    except Exception:report.update(status='failed',error=traceback.format_exc())
    report['elapsed_s']=time.monotonic()-started;_json(directory/'report.json',report)
    emit(dict(kind='finished',report=report));return report




def apply(model_directory,boundary_directory,output,device='auto',emit=lambda e:None,cancel=None,
          image=None,audit=None,split=None,threshold=None,cell_channel=None):
    directory=new_run_directory(output,'center-contours')
    report=dict(version=1,engine='cell-center-box3d',mode='apply',status='running',directory=str(directory),
                model_directory=str(Path(model_directory).resolve()),boundary_model=str(Path(boundary_directory).resolve()),files=[],failures=[])
    started=time.monotonic()
    last_save=[0.]
    def notify(event):
        if event['kind']=='progress':
            report['progress']={k:v for k,v in event.items() if k!='kind'}
            if time.monotonic()-last_save[0]>=.5:
                _json(directory/'report.json',report);last_save[0]=time.monotonic()
        emit(event)
    _json(directory/'report.json',report);emit(dict(kind='run_started',directory=str(directory)))
    try:
        manifest=read_model(model_directory);config=ProposalConfig(**manifest['config'])
        if threshold is not None:
            from dataclasses import replace
            config=replace(config,threshold=threshold).validate()
        boundary_manifest=read_boundary_model(boundary_directory);boundary_config=RefinementConfig(**boundary_manifest['config'])
        from dataclasses import replace
        boundary_config=replace(boundary_config,margin_um=boundary_manifest.get('inference_crop_margin_um',config.box_margin_um)).validate()
        report['config']=asdict(config);report['boundary_config']=asdict(boundary_config)
        report['crop_acceptance']=dict(surface_smoothing_um=.3,
            minimum_surface_support=manifest.get('crop_surface_evidence_threshold',.65),
            minimum_mean_surface_contrast=.04,minimum_volume_um3=5.,fallback=False)
        report['checkpoint_sha256']=hashlib.sha256((Path(model_directory)/manifest['checkpoint']).read_bytes()).hexdigest()
        report['boundary_sha256']=hashlib.sha256((Path(boundary_directory)/boundary_manifest['checkpoint']).read_bytes()).hexdigest()
        tf,_,_,info=runtime(device,config.seed);report['runtime']=info
        detector=network(tf,config);detector.load_weights(Path(model_directory)/manifest['checkpoint'])
        boundary=boundary_network(tf,boundary_config);boundary.load_weights(Path(boundary_directory)/boundary_manifest['checkpoint'])
        collection=json.loads(Path(audit).read_text()) if audit else None
        if collection and collection['status']!='complete':raise ValueError('Completed calibrated cache required.')
        cases=[c for c in collection['cases'] if c['audit']['split']==split] if collection else [dict(image=str(Path(image).resolve()))]
        for number,case in enumerate(cases):
            checkpoint(cancel);notify(dict(kind='progress',phase='Preparing membrane image',current=number,total=len(cases),file=case['image']))
            prepared=prepare_case(case,collection,boundary_config,notify,cancel,manifest.get('channel_selection'),cell_channel)
            proposals=predict_proposals(prepared,detector,config,notify,cancel)
            labels,diagnostics=contour_proposals(prepared,proposals,boundary,boundary_config,notify,cancel,
                manifest.get('crop_surface_evidence_threshold',.65))
            cells=[]
            for cell,box in enumerate(ndi.find_objects(labels),1):
                if box is None:continue
                mask=labels[box]==cell;props=mask_properties(mask,np.asarray(prepared.spacing))
                center=(np.argwhere(mask).mean(0)+[s.start for s in box])*prepared.spacing
                d=next(d for d in diagnostics if d['cell_id']==cell)
                props.update(label=cell,center_xyz=(center[::-1]+prepared.origin).tolist(),detection_score=d['score'])
                cells.append(props)
            accepted=[dict(d,fallback_reason=None) for d in diagnostics if d['accepted']]
            metadata=dict(version=1,image=case['image'],time=0,channel=prepared.channel,channel_scores=prepared.channel_scores,
                fingerprint=prepared.fingerprint,spacing_zyx=prepared.spacing,origin_xyz=prepared.origin,
                config=dict(use_shape_filter=False),cells=cells,detector=dict(engine='cell-center-box3d',
                model=str(Path(model_directory).resolve()),config=asdict(config),proposals=diagnostics,
                crop_acceptance=report['crop_acceptance'],
                checkpoint_sha256=report['checkpoint_sha256']),refinement=dict(model=str(Path(boundary_directory).resolve()),
                config=asdict(boundary_config),cells=accepted,checkpoint_sha256=report['boundary_sha256'],
                policy='Independent padded-box target contours; failed crops rejected; no geometric fallback or roundness filter.'))
            folder=directory/f'{number+1:04d}-{Path(case["image"]).stem}';folder.mkdir()
            path=folder/'contours.npz';np.savez_compressed(path,labels=labels,metadata=np.array(json.dumps(metadata,allow_nan=False)))
            row=dict(image=case['image'],predictions=str(path),split=split or 'apply',channel=prepared.channel,
                     proposals=diagnostics,cells=len(cells),refinement=accepted)
            if collection:
                refs,names=read_references(Path(case['directory'])/'references.npz')
                row['metrics']=evaluate(labels,refs,names,prepared.spacing,cancel)
            report['files'].append(row);_json(directory/'report.json',report)
        report['summary']=dict(scans=len(cases),proposals=sum(len(r['proposals']) for r in report['files']),
                               cells=sum(r['cells'] for r in report['files']))
        report['status']='complete'
    except Cancelled:report['status']='cancelled'
    except Exception:report.update(status='failed',error=traceback.format_exc())
    report['elapsed_s']=time.monotonic()-started;_json(directory/'report.json',report)
    emit(dict(kind='finished',report=report));return report
