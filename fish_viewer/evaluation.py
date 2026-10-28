"""Manual-reference reconstruction and calibrated evaluation for supported pipelines."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from fish_viewer.analysis.controller import default_proposal_model, default_refinement_model, default_models, sensitive_stardist_model
from fish_viewer.models import LibraryEntry
from fish_viewer.reader import load_dataset
from fish_viewer.scenes import read_scene
from fish_viewer.neural.audit import read_references, save_references, unique_references
from fish_viewer.pipeline.evaluation import rasterize_reference,evaluate
from fish_viewer.pipeline.segmentation import Prepared
from fish_viewer.signals.config import SignalConfig
from fish_viewer.signals.data import normalized,infer_channels,marker_points,reference_memberships
from fish_viewer.signals.engine import image_fingerprint
from fish_viewer.workflow.volumes import surface_volume


def save(path,value):
 path=Path(path);temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def reference_key(ref):
 box,mask=ref
 return hashlib.sha256(str([(s.start,s.stop) for s in box]).encode()+mask.tobytes()).hexdigest()

def frozen_models():
 models={}
 for kind,folder,file in [('center_box_detector',default_proposal_model(),'proposal-model.json'),
                          ('stardist_detector',sensitive_stardist_model(),'fish-model.json'),
                          ('contours',default_refinement_model(),'refinement-model.json'),
                          ('signals',default_models()[1],'signal-model.json'),
                          ('stardist2d',default_models()[0].parent/'cells-stardist2d','fish-model.json')]:
  folder=Path(folder);manifest=json.loads((folder/file).read_text())
  models[kind]=dict(directory=str(folder.resolve()),manifest=manifest,manifest_sha256=sha(folder/file),checkpoint_sha256=sha(folder/manifest['checkpoint']))
 return models

def collect_truth(case,metadata,collection,folder):
 cache=folder/f"manual-channel-{metadata['channel']}.json"
 refs,names=read_references(Path(case['directory'])/'references.npz')
 a=case['audit']
 if not (np.allclose(metadata['spacing_zyx'],a['spacing_zyx']) and np.allclose(metadata['origin_xyz'],a['origin_xyz'])):
  raise ValueError('Prediction grid differs from original calibrated reference cache.')
 if metadata['fingerprint']!=a['fingerprint']:raise ValueError('Prediction fingerprint differs from held-out audit.')
 if cache.is_file():return refs,json.loads(cache.read_text())
 sources={s['image']:s for s in collection['sources']}
 scenes=sorted({p for alias in a['aliases'] for p in sources[alias]['scenes']})
 prepared=Prepared(np.empty(tuple(a['shape_zyx']),np.uint8),tuple(a['spacing_zyx']),tuple(a['origin_xyz']),a['channel'],[],'',{}, {})
 mesh_by_key={};ab={}
 for scene_path in scenes:
  scene=read_scene(Path(scene_path))
  for surface in scene.surfaces:
   if surface.time!=0:continue
   candidates,_,_=rasterize_reference([surface],prepared)
   if candidates:mesh_by_key.setdefault(reference_key(candidates[0]),surface)
  for group in scene.measurements:
   for left,right in group.edges:
    if group.times[left]!=0 or group.times[right]!=0:continue
    if {group.labels[left].strip().upper(),group.labels[right].strip().upper()}!={'A','B'}:continue
    i,j=(left,right) if group.labels[left].strip().upper()=='A' else (right,left)
    key=tuple(np.round(group.points[i],4))
    ab.setdefault(key,[]).append(float(np.linalg.norm(group.points[i]-group.points[j])))
 cells=[]
 for index,(ref,name) in enumerate(zip(refs,names)):
  key=reference_key(ref);surface=mesh_by_key.get(key)
  if surface is None:raise ValueError('No original manual mesh found for reference '+name)
  cells.append(dict(reference_id=index,name=name,reference_key=key,volume_um3=surface_volume(surface),cut=a['reference_cut'][index]))
 dataset=load_dataset(LibraryEntry(Path(case['image']),Path(case['scene']) if case['scene'] else None))
 if image_fingerprint(dataset)!=a['fingerprint']:raise ValueError('Original source pixels changed.')
 points=marker_points(scenes)
 memberships=reference_memberships([p['xyz'] for p in points],refs,a['origin_xyz'],a['spacing_zyx'])
 points=infer_channels([normalized(v) for v in dataset.volumes],points,metadata['channel'],dataset.center_origin,dataset.spacing[::-1],SignalConfig())
 counts=Counter()
 for p,membership in zip(points,memberships):
  values=ab.get(tuple(np.round(p['xyz'],4)),[])
  p.update(reference_ids=membership,manual_distance_um=float(np.mean(values)) if values and np.ptp(values)<=.05 else None)
  if p['channel'] is None:counts['ambiguous_channel']+=1
  if len(membership)!=1:counts['ambiguous_or_missing_cell']+=1
  if not values:counts['no_manual_AB']+=1
  elif np.ptp(values)>.05:counts['conflicting_manual_AB']+=1
 truth=dict(cells=cells,points=points,source_scenes=scenes,manual_A_points=len(points),exclusions=dict(counts),
            distance_reference='Actual manual A-B segments only; no fabricated reference distance for A-only markers.')
 save(cache,truth);return refs,truth

def evaluate_run(case,state,collection,folder):
 run=Path(state['report_path']).parent;report=json.loads((run/state['signal_report']).read_text())
 with np.load(report['unfiltered_cell_predictions'],allow_pickle=False) as f:labels=f['labels'];metadata=json.loads(str(f['metadata']))
 refs,truth=collect_truth(case,metadata,collection,folder)
 names=[str(i) for i in range(len(refs))]
 metrics=evaluate(labels,refs,names,np.array(metadata['spacing_zyx']))
 before=[r for r in metrics['per_cell'] if r['iou']>=.5]
 with np.load(report['cell_predictions'],allow_pickle=False) as f:final_labels=f['labels']
 final=evaluate(final_labels,refs,names,np.array(metadata['spacing_zyx']))
 matched=[r for r in final['per_cell'] if r['iou']>=.5]
 volumes={r['cell_id']:r['volume_um3'] for r in report['cell_volumes']}
 volume_pairs=[];distance_pairs=[]
 for row in matched:
  ref=int(row['name']);cell=truth['cells'][ref];volume=volumes[row['predicted_label']]
  if cell['volume_um3'] and volume:
   volume_pairs.append(dict(reference_id=ref,cell=cell['name'],predicted_cell_id=row['predicted_label'],iou=row['iou'],
      manual_um3=cell['volume_um3'],predicted_um3=volume,cut=cell['cut']))
  manual=[p for p in truth['points'] if p['reference_ids']==[ref] and p['channel'] is not None and p['manual_distance_um'] is not None]
  predicted=[p for p in report['points'] if p['cell_id']==row['predicted_label']]
  if not manual or not predicted:continue
  costs=np.array([[np.linalg.norm(np.asarray(p['xyz'])-q['xyz']) if p['channel']==q['channel'] else 1e6 for q in predicted] for p in manual])
  valid=costs<=.6;weights=valid*(min(costs.shape)+1+np.maximum(0,1-costs/.6))
  aa,bb=linear_sum_assignment(-weights)
  for i,j in zip(aa,bb):
   if not valid[i,j]:continue
   p,q=manual[i],predicted[j]
   distance_pairs.append(dict(reference_id=ref,cell=cell['name'],predicted_cell_id=row['predicted_label'],channel=p['channel'],
       manual_xyz=p['xyz'],predicted_xyz=q['xyz'],center_error_um=float(costs[i,j]),manual_um=p['manual_distance_um'],predicted_um=q['distance_um']))
 count=report['signal_filter']['original_cell_count'];kept=report['signal_filter']['kept_cell_count']
 return dict(image=case['image'],scan=Path(case['image']).name,group=case['audit']['group'],split=case['audit']['split'],
   references=len(refs),complete_references=sum(not c['cut'] for c in truth['cells']),cells_before_filter=count,final_cells=kept,
   signals=len(report['points']),matched_before_filter=len(before),matched_final=len(matched),
   unscored_before_filter=count-len(before),unscored_final=kept-len(matched),
   matches_before_filter=[dict(reference_id=int(r['name']),cell_id=r['predicted_label'],iou=r['iou']) for r in before],
   matches_final=[dict(reference_id=int(r['name']),cell_id=r['predicted_label'],iou=r['iou']) for r in matched],
   volume_pairs=volume_pairs,distance_pairs=distance_pairs,manual_A_points=truth['manual_A_points'],manual_exclusions=truth['exclusions'],pipeline_report=state['report_path'])

def paired_stats(rows,expected,predicted,percentage=False):
 if not rows:return dict(n=0)
 truth=np.array([p[expected] for p in rows]);found=np.array([p[predicted] for p in rows]);error=found-truth;absolute=np.abs(error)
 result=dict(n=len(rows),bias=float(error.mean()),mae=float(absolute.mean()),rmse=float(np.sqrt(np.mean(error**2))),median_absolute_error=float(np.median(absolute)),p95_absolute_error=float(np.percentile(absolute,95)))
 if percentage:
  ape=absolute/truth*100;result.update(mape_percent=float(ape.mean()),median_ape_percent=float(np.median(ape)),p95_ape_percent=float(np.percentile(ape,95)))
 groups=sorted({p['group'] for p in rows})
 if len(groups)>1:
  sums=np.array([[sum(abs(p[predicted]-p[expected]) for p in rows if p['group']==g),sum(p['group']==g for p in rows)] for g in groups])
  draws=np.random.default_rng(42).integers(len(groups),size=(2000,len(groups)));sample=sums[draws].sum(1)
  result['mae_95ci_acquisition_bootstrap']=np.percentile(sample[:,0]/sample[:,1],[2.5,97.5]).tolist()
  if percentage:
   sums[:,0]=[sum(abs(p[predicted]-p[expected])/p[expected]*100 for p in rows if p['group']==g) for g in groups]
   sample=sums[draws].sum(1);result['mape_95ci_acquisition_bootstrap']=np.percentile(sample[:,0]/sample[:,1],[2.5,97.5]).tolist()
 return result


def channel_for(case):
    audit = case["audit"]
    candidates = audit["channel_selection"]["stain_candidates"]
    if candidates:
        return audit[
            "channel"
        ], "Declared Lam/LBR wavelength; ring evidence ranks membrane candidates"
    hints = [r["channel"] for r in audit["channel_scores"] if r.get("metadata_hint")]
    if len(hints) != 1:
        raise ValueError("Ambiguous undeclared cell channel: " + case["image"])
    return hints[
        0
    ], "Single metadata membrane hint; visually reviewed channel projections"


def summarize_complete(rows):
    result = {}
    for split in ("test", "validation", "all_heldout"):
        subset = [r for r in rows if split == "all_heldout" or r["split"] == split]
        volumes = [
            dict(p, group=r["group"])
            for r in subset
            for p in r["volume_pairs"]
            if not p["cut"]
        ]
        distances = [
            dict(p, group=r["group"])
            for r in subset
            for p in r["distance_pairs"]
            if not r["cut_references"][p["reference_id"]]
        ]
        totals = {
            key: sum(r[key] for r in subset)
            for key in (
                "references",
                "complete_references",
                "cells_before_filter",
                "final_cells",
                "signals",
                "matched_before_filter",
                "matched_final",
            )
        }
        totals.update(
            scans=len(subset),
            matched_complete_before=sum(
                sum(
                    not r["cut_references"][m["reference_id"]]
                    for m in r["matches_before_filter"]
                )
                for r in subset
            ),
            matched_complete_final=sum(
                sum(
                    not r["cut_references"][m["reference_id"]]
                    for m in r["matches_final"]
                )
                for r in subset
            ),
            volume=paired_stats(volumes, "manual_um3", "predicted_um3", True),
            distance=paired_stats(distances, "manual_um", "predicted_um"),
        )
        if volumes:
            signed = np.array(
                [(p["predicted_um3"] / p["manual_um3"] - 1) * 100 for p in volumes]
            )
            totals["volume"].update(
                mean_signed_percent=float(signed.mean()),
                median_signed_percent=float(np.median(signed)),
            )
        positive = [p for p in distances if p["manual_um"] > 1e-12]
        totals["distance_relative"] = paired_stats(
            positive, "manual_um", "predicted_um", True
        )
        if distances:
            totals["distance"]["fraction_within_0_2_um"] = float(
                np.mean(
                    [abs(p["predicted_um"] - p["manual_um"]) <= 0.2 for p in distances]
                )
            )
        result[split] = totals
    return result


def rebuild_references(case, collection, folder):
    """Recreate the archived reference grid/order without changing its split."""
    audit = case['audit']
    sources = {source['image']: source for source in collection['sources']}
    scenes = sorted({path for alias in audit['aliases'] for path in sources[alias]['scenes']})
    surfaces = [surface for path in scenes for surface in read_scene(Path(path)).surfaces
                if surface.time == 0 and surface_volume(surface) is not None]
    prepared = Prepared(np.empty(tuple(audit['shape_zyx']), np.uint8),
                        tuple(audit['spacing_zyx']), tuple(audit['origin_xyz']),
                        audit['channel'], [], audit['fingerprint'], {}, {})
    refs, names, _ = rasterize_reference(surfaces, prepared)
    refs, names = unique_references(refs, names)
    if len(refs) != audit['reference_count']:
        raise ValueError(f"Reference reconstruction changed: {case['image']}")
    folder.mkdir()
    save_references(folder / 'references.npz', refs, names)
    return dict(case, directory=str(folder))


def ref_key(reference):
    box, mask = reference
    return hashlib.sha256(str([(s.start,s.stop) for s in box]).encode()+mask.tobytes()).hexdigest()

def manual_meshes(scenes, prepared):
    meshes = {}
    ab = {}
    for path in scenes:
        scene = read_scene(Path(path))
        for surface in scene.surfaces:
            references, _, _ = rasterize_reference([surface], prepared)
            if references:
                meshes.setdefault(ref_key(references[0]), surface)
        for group in scene.measurements:
            for left, right in group.edges:
                if group.times[left] != 0 or group.times[right] != 0:
                    continue
                labels = [group.labels[left].strip().upper(),group.labels[right].strip().upper()]
                if set(labels) != {'A','B'}:
                    continue
                a,b = (left,right) if labels[0]=='A' else (right,left)
                key = tuple(np.round(group.points[a],4))
                ab.setdefault(key,[]).append(float(np.linalg.norm(group.points[a]-group.points[b])))
    # Conflicting annotations on duplicate scans are reported as ambiguous.
    return meshes, {k: float(np.mean(v)) if np.ptp(v)<=.05 else None for k,v in ab.items()}


def legacy_paired_stats(rows, expected, predicted):
    rows = [r for r in rows if r.get(expected) is not None and r.get(predicted) is not None]
    if not rows:
        return dict(n=0)
    truth = np.array([r[expected] for r in rows])
    found = np.array([r[predicted] for r in rows])
    error = found-truth
    result = dict(n=len(rows), bias=float(error.mean()), mae=float(np.abs(error).mean()),
                  mse=float(np.mean(error**2)), rmse=float(np.sqrt(np.mean(error**2))),
                  median_absolute_error=float(np.median(np.abs(error))),
                  p95_absolute_error=float(np.percentile(np.abs(error),95)))
    if np.all(truth > 0):
        result['mean_absolute_percentage_error'] = float(np.mean(np.abs(error)/truth)*100)
        result['median_absolute_percentage_error'] = float(np.median(np.abs(error)/truth)*100)
    groups = sorted({r['group'] for r in rows})
    if len(groups) > 1:
        sums = np.array([[sum(abs(r[predicted]-r[expected]) for r in rows if r['group']==g),
                          sum(r['group']==g for r in rows)] for g in groups])
        indices = np.random.default_rng(42).integers(len(groups), size=(2000,len(groups)))
        samples = sums[indices].sum(axis=1)
        result['mae_95ci_acquisition_bootstrap'] = np.percentile(samples[:,0]/samples[:,1], [2.5,97.5]).tolist()
    return result


def surface_samples(surface):
    triangles=surface.vertices[surface.triangles]
    area=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)
    valid=area>0;triangles=triangles[valid];area=area[valid]
    rng=np.random.default_rng(42)
    selected=rng.choice(len(triangles),size=256,p=area/area.sum())
    uv=rng.random((256,2));uv[uv.sum(axis=1)>1]=1-uv[uv.sum(axis=1)>1]
    t=triangles[selected]
    return t[:,0]+uv[:,0,None]*(t[:,1]-t[:,0])+uv[:,1,None]*(t[:,2]-t[:,0])

def boundary_error(manual,predicted,manual_locator,predicted_locator,manual_samples):
    distances=np.array([manual_locator.closest(p)[1] for p in surface_samples(predicted)]
                       +[predicted_locator.closest(p)[1] for p in manual_samples])
    return float(distances.mean()),float(np.percentile(distances,95))
