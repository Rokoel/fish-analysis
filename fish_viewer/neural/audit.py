"""Collection audit, calibrated caches, and acquisition-safe dataset splits."""
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np

from ..models import LibraryEntry
from ..reader import load_dataset, scan_library
from ..scenes import read_scene
from ..pipeline.evaluation import rasterize_reference
from ..pipeline.runner import _json
from ..pipeline.segmentation import Prepared, checkpoint
from .data import NeuralCase, prepare_neural, sparse_labels


def acquisition_key(path):
    match=re.match(r'^(\d{6})[_ -]',Path(path).name)
    if match:
        try:return 'date:'+datetime.strptime(match[1],'%d%m%y').date().isoformat()
        except ValueError:pass
    # Undated neighboring fields are conservatively kept together.
    return 'folder:'+str(Path(path).parent.resolve())


def unique_references(refs,names):
    result=[];labels=[];seen=set()
    for (box,mask),name in zip(refs,names):
        key=hashlib.sha256(str([(s.start,s.stop) for s in box]).encode()+mask.tobytes()).hexdigest()
        if key not in seen:
            seen.add(key);result.append((box,mask));labels.append(name)
    return result,labels


def save_references(path,refs,names):
    arrays={'boxes':np.array([[(s.start,s.stop) for s in box] for box,_ in refs],dtype=np.int32).reshape(-1,3,2),
        'names':np.asarray(names,dtype=str)}
    arrays.update({f'mask{i}':mask for i,(_,mask) in enumerate(refs)})
    np.savez_compressed(path,**arrays)


def read_references(path):
    with np.load(path,allow_pickle=False) as a:
        return [(tuple(slice(int(x),int(y)) for x,y in box),a[f'mask{i}']) for i,box in enumerate(a['boxes'])],a['names'].tolist()


def split_audit_cases(cases,seed):
    parent=list(range(len(cases)));seen={}
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,c in enumerate(cases):
        for key in c.audit['acquisition_keys']+['pixels:'+c.prepared.fingerprint]:
            if key in seen:parent[root(i)]=root(seen[key])
            else:seen[key]=i
    buckets=defaultdict(list)
    for i,c in enumerate(cases):buckets[root(i)].append(c)
    groups=sorted(buckets.values(),key=lambda cs:min(str(c.entry.image) for c in cs))
    eligible=[cs for cs in groups if any(c.audit['training_objects'] for c in cs)]
    if len(eligible)<3:raise ValueError('At least three independent groups with usable annotations are required.')
    order=np.random.default_rng(seed).permutation(len(eligible))
    ntest=max(1,round(.2*len(eligible)));nval=max(1,round(.2*len(eligible)))
    names=['test']*ntest+['validation']*nval+['training']*(len(eligible)-ntest-nval)
    for index,split in zip(order,names):
        for case in eligible[index]:case.audit['split']=split
    for number,group in enumerate(groups):
        for c in group:
            c.group=f'acquisition-component-{number:04d}'
            c.audit['group']=c.group
            c.audit.setdefault('split','unscored')


def audit_collection(root,config,directory,emit=print,cancel=None):
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    if (directory/'audit.json').exists():raise ValueError('Use a new audit output folder to preserve the existing audit.')
    entries,warnings=scan_library(root);by_image=defaultdict(list)
    for e in entries:by_image[e.image].append(e)
    report=dict(version=1,engine='stardist3d-audit',root=str(Path(root).resolve()),directory=str(directory),
        config=asdict(config),status='running',sources=[],cases=[],warnings=warnings,errors=[],missing_images=[],
        split_policy='Seeded 60/20/20 acquisition components; same date across folders, undated folder, and duplicates stay together.')
    cases=[];canonical={};started=time.monotonic()
    def persist():
        report['cases']=[dict(directory=str(c.directory),image=str(c.entry.image),scene=str(c.entry.scene) if c.entry.scene else None,
            audit=c.audit) for c in cases]
        report['elapsed_s']=time.monotonic()-started
        _json(directory/'audit.json',report)
    persist()
    for number,(path,alternatives) in enumerate(sorted(by_image.items()),1):
        checkpoint(cancel)
        if not path.exists():
            report['missing_images'].append(str(path));persist();continue
        emit(f'Audit {number}/{len(by_image)}: {path.parent.name}/{path.name}')
        try:
            alternatives.sort(key=lambda e:(not(e.scene and e.scene.stem==e.image.stem),str(e.scene)))
            entry=alternatives[0];dataset=load_dataset(entry,time=0,level=0)
            source=dict(image=str(path),scenes=[str(e.scene) for e in alternatives if e.scene],
                size=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns,format=dataset.metadata.format,
                shape=list(dataset.metadata.shape),spacing_xyz=list(dataset.spacing),times=dataset.metadata.times,
                source_warnings=dataset.scene.warnings.copy())
            source['scene_stats']=[dict(path=str(e.scene),size=e.scene.stat().st_size,mtime_ns=e.scene.stat().st_mtime_ns)
                for e in alternatives if e.scene]
            if dataset.metadata.times!=[0]:raise ValueError('Multiple time points require an explicit time-series audit.')
            for alt in alternatives[1:]:
                if alt.scene:
                    scene=read_scene(alt.scene);dataset.scene.surfaces.extend(scene.surfaces)
                    source['source_warnings'].extend(scene.warnings)
            source['manual_meshes']=len(dataset.scene.surfaces)
            prepared=prepare_neural(dataset,config,cancel=cancel)
            source['fingerprint']=prepared.fingerprint
            digest=hashlib.sha256(str((prepared.image.shape,prepared.spacing,prepared.origin)).encode())
            digest.update(memoryview(np.ascontiguousarray(prepared.image)))
            network_hash=digest.hexdigest()
            source['network_fingerprint']=network_hash
            refs,names,skipped=rasterize_reference(dataset.scene.surfaces,prepared,0,cancel)
            source['rasterized_references']=len(refs);source['reference_warnings']=skipped
            del dataset
            if network_hash in canonical:
                case=canonical[network_hash]
                refs,names=unique_references(case.references+refs,case.names+names)
                case.audit['aliases'].append(str(path));case.audit['acquisition_keys'].append(acquisition_key(path))
                source['duplicate_of']=str(case.entry.image)
            else:
                refs,names=unique_references(refs,names)
                folder=directory/f'{number:04d}-{path.stem}';folder.mkdir()
                np.save(folder/'image.npy',prepared.image)
                prepared.image=np.load(folder/'image.npy',mmap_mode='r')
                case=NeuralCase(entry,prepared,refs,names,'',folder,None,[],
                    dict(aliases=[str(path)],acquisition_keys=[acquisition_key(path)]))
                cases.append(case);canonical[network_hash]=case
            labels,cuts,audit=sparse_labels(case.prepared.image.shape,refs,names,config.overlap_policy)
            if case.labels is not None and case.audit.get('background_voxels',0):
                labels[case.labels==0]=0
                audit['background_voxels']=int((labels==0).sum())
            background=path.with_suffix('.background.npz')
            if background.exists():
                with np.load(background,allow_pickle=False) as a:
                    bg=a['background']
                    if (str(a['fingerprint'])!=prepared.fingerprint or bg.shape!=labels.shape or bg.dtype!=bool
                            or not np.allclose(a['spacing_zyx'],prepared.spacing) or not np.allclose(a['origin_xyz'],prepared.origin)):
                        raise ValueError('Reviewed background does not match this image and calibrated grid.')
                if any(np.any(bg[box]&mask) for box,mask in refs):raise ValueError('Reviewed background overlaps a reference cell.')
                labels[bg]=0;audit['background_voxels']=int(bg.sum());audit['background_file']=str(background)
            case.references,case.names,case.reference_cut=refs,names,cuts
            case.audit.update(audit)
            case.audit.update(image=str(case.entry.image),channel=case.prepared.channel,channel_scores=case.prepared.channel_scores,
                shape_zyx=list(labels.shape),spacing_zyx=list(case.prepared.spacing),origin_xyz=list(case.prepared.origin),
                fingerprint=case.prepared.fingerprint,warnings=prepared.warnings+skipped,
                reference_cut=cuts,network_fingerprint=network_hash)
            case.labels=None
            np.save(case.directory/'labels.npy',labels);del labels
            case.labels=np.load(case.directory/'labels.npy',mmap_mode='r')
            save_references(case.directory/'references.npz',refs,names)
            source['status']='duplicate' if 'duplicate_of' in source else 'cached'
            report['sources'].append(source)
        except Exception as exc:
            report['errors'].append(dict(image=str(path),error=str(exc)))
            emit(f'ERROR: {path}: {exc}')
        persist()
    split_audit_cases(cases,config.seed)
    for c in cases:_json(c.directory/'labels.json',c.audit)
    report['summary']=dict(existing_images=len(report['sources']),missing_images=len(report['missing_images']),
        unique_inputs=len(cases),duplicate_inputs=sum('duplicate_of' in s for s in report['sources']),
        references=sum(len(c.references) for c in cases),complete_cells=sum(c.audit['complete_references'] for c in cases),
        training_candidates=sum(c.audit['training_objects'] for c in cases),
        cut_cells=sum(c.audit['cut_references'] for c in cases),
        ambiguous_channel_images=sum(bool(c.prepared.warnings) for c in cases),
        by_split={split:dict(images=sum(c.audit['split']==split for c in cases),
            groups=len({c.group for c in cases if c.audit['split']==split}),
            usable_cells=sum(c.audit['training_objects'] for c in cases if c.audit['split']==split),
            references=sum(len(c.references) for c in cases if c.audit['split']==split))
            for split in ('training','validation','test','unscored')})
    report['status']='complete' if not report['errors'] else 'completed_with_errors'
    persist();emit(json.dumps(report['summary'],indent=2));return report


def load_audit(path,config,directory,emit,cancel=None,training=True):
    path=Path(path).resolve();report=json.loads(path.read_text())
    if report.get('engine')!='stardist3d-audit' or report['status']!='complete':raise ValueError('Select a complete, error-free annotation audit.')
    if report['config'].get('preprocessing', 'none') != config.preprocessing:
        raise ValueError('Audit and training preprocessing differ; build a matching preprocessing cache.')
    for key in ('voxel_um','min_diameter_um','max_diameter_um','native_sampling','overlap_policy','seed'):
        if report['config'].get(key)!=getattr(config,key):raise ValueError(f'Audit and training differ in {key}; regenerate the audit.')
    if training and report.get('parent_audit'):
        for key in ('precision_mode','precision_max_diameter_um','curate_training_shapes','min_sphericity','max_elongation'):
            if report['config'].get(key)!=getattr(config,key):raise ValueError(f'Curated audit and training differ in {key}; derive a new audit.')
        if report['config'].get('use_shape_filter',True)!=config.use_shape_filter:
            raise ValueError('Audit shape selection differs; derive a new audit to train without roundness selection.')
    for source in report['sources']:
        stat=Path(source['image']).stat()
        if stat.st_size!=source['size'] or stat.st_mtime_ns!=source['mtime_ns']:
            raise ValueError(f'Source changed since the audit: {source["image"]}')
        for scene in source.get('scene_stats',[]):
            stat=Path(scene['path']).stat()
            if stat.st_size!=scene['size'] or stat.st_mtime_ns!=scene['mtime_ns']:
                raise ValueError(f'Annotations changed since the audit: {scene["path"]}')
    cases=[]
    for index,row in enumerate(report['cases'],1):
        emit(dict(kind='progress',phase='Loading audited images',current=index-1,total=len(report['cases']),file=row['image']))
        checkpoint(cancel);a=row['audit'];cache=Path(row['directory'])
        required = ['image.npy', 'labels.npy', 'references.npz']
        if config.preprocessing != 'none':
            required.append('evidence.npy')
        missing = [name for name in required if not (cache / name).is_file()]
        if missing:
            raise FileNotFoundError(f'Training cache is missing {", ".join(missing)} in {cache}. '
                                    'Rebuild the arrays before training; archived audit metadata alone is insufficient. '
                                    'See docs/training-data.md.')
        image=np.load(cache/'image.npy',mmap_mode='r');labels=np.load(cache/'labels.npy',mmap_mode='r')
        refs,names=read_references(cache/'references.npz')
        prepared=Prepared(image,tuple(a['spacing_zyx']),tuple(a['origin_xyz']),a['channel'],a['channel_scores'],a['fingerprint'],
            {'Raw selected channel':image[len(image)//2], 'Background corrected':image[len(image)//2]}, {},a['warnings'])
        if config.preprocessing != 'none':
            prepared.evidence_image = np.load(cache/'evidence.npy', mmap_mode='r')
        output=directory/cache.name;output.mkdir()
        entry=LibraryEntry(Path(row['image']),Path(row['scene']) if row['scene'] else None)
        case=NeuralCase(entry,prepared,refs,names,a['group'],output,labels,a['reference_cut'],a)
        cases.append(case)
    emit(dict(kind='warning',message=f'Using audited collection: {len(cases)} unique inputs; source statistics verified.'))
    return cases
