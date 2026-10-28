"""Physical contour agreement on complete manual cells; unmatched fields remain unscorable."""
import argparse
import csv
import json
import hashlib
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from openpyxl import Workbook
from fish_viewer.evaluation import manual_meshes,ref_key,legacy_paired_stats as paired_stats
from fish_viewer.evaluation import surface_samples,boundary_error
from fish_viewer.neural.audit import read_references
from fish_viewer.pipeline.runner import read_prediction,_json
from fish_viewer.pipeline.segmentation import Prepared
from fish_viewer.workflow.distances import SurfaceLocator
from fish_viewer.workflow.volumes import surface_volume


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--split',choices=['validation','test'],required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    audit=json.loads(args.audit.read_text());report=json.loads(args.report.read_text())
    if report['status']!='complete':raise ValueError('Completed contour inference required.')
    cases={c['image']:c for c in audit['cases'] if c['audit']['split']==args.split}
    rows=[r for r in report['files'] if r['image'] in cases]
    if len(rows)!=len(cases) or set(cases)!=set(r['image'] for r in rows):raise ValueError('Exactly one prediction for every scan in the chosen split required.')
    sources={s['image']:s for s in audit['sources']};cells=[];scores=[];total=0
    for number,row in enumerate(rows,1):
        if number%50==0 or number==len(cases):print(f'[{number}/{len(cases)}] Evaluate {args.split} contours: {Path(row["image"]).name}',flush=True)
        case=cases[row['image']];refs,names=read_references(Path(case['directory'])/'references.npz')
        surfaces,info=read_prediction(row['predictions']);predicted={int(s.name.rsplit(' ',1)[-1]):s for s in surfaces}
        prepared=Prepared(np.empty(case['audit']['shape_zyx'],np.uint8),tuple(info['spacing_zyx']),tuple(info['origin_xyz']),info['channel'],[],info['fingerprint'],{},{})
        scenes=sorted({s for alias in case['audit']['aliases'] for s in sources[alias]['scenes']})
        manual,_=manual_meshes(scenes,prepared)
        for ref,match in enumerate(row['metrics']['per_cell']):
            if case['audit']['reference_cut'][ref]:continue
            total+=1;scores.append(match['iou'])
            cell=match['predicted_label']
            if cell is None or match['iou']<.5:continue
            truth=manual.get(ref_key(refs[ref]))
            if truth is None or surface_volume(truth) is None:raise ValueError('Complete closed manual reference missing.')
            mean,hd95=boundary_error(truth,predicted[cell],SurfaceLocator(truth),SurfaceLocator(predicted[cell]),surface_samples(truth))
            cells.append(dict(image=str(Path(case['image']).relative_to(ROOT)),group=case['audit']['group'],
                manual_reference_id=ref+1,manual_cell=names[ref],predicted_cell=cell,iou=match['iou'],dice=match['dice'],
                manual_volume_um3=surface_volume(truth),predicted_volume_um3=surface_volume(predicted[cell]),
                surface_mean_error_um=mean,surface_hd95_um=hd95))
    summary=dict(scans=len(cases),complete_manual_cells=total,matched_complete_cells=len(cells),
        annotated_cell_recall_iou_05=len(cells)/total,predicted_cells=sum(r.get('cells',r['metrics'].get('predicted_count',0)) for r in rows),
        mean_manual_cell_iou=float(np.mean(scores)),volume=paired_stats(cells,'manual_volume_um3','predicted_volume_um3'),
        mean_symmetric_surface_error_um=float(np.mean([r['surface_mean_error_um'] for r in cells])) if cells else None,
        mean_surface_hd95_um=float(np.mean([r['surface_hd95_um'] for r in cells])) if cells else None)
    checkpoint_sha256=report.get('checkpoint_sha256')
    if checkpoint_sha256 is None and report.get('model_directory'):
        folder=Path(report['model_directory']);manifest=json.loads((folder/'fish-model.json').read_text())
        with (folder/manifest['checkpoint']).open('rb') as stream:checkpoint_sha256=hashlib.file_digest(stream,'sha256').hexdigest()
    result=dict(status='complete',split=args.split,summary=summary,cells=cells,
        checkpoint_sha256=checkpoint_sha256,boundary_sha256=report.get('boundary_sha256'),source_report=str(args.report.resolve()),
        policy='Complete closed manual-cell matches at whole-cell IoU>=.5, before signal retention. All outliers included. '
        'Volumes use closed physical triangle meshes. Symmetric surface distance uses 256 area-uniform points per mesh. '
        'Sparse labels cannot establish field precision. No threshold selection in this evaluation.')
    args.output.mkdir(parents=True,exist_ok=True);_json(args.output/'report.json',result)
    fields=list(cells[0]) if cells else ['No observations']
    with (args.output/'cells.csv').open('w',newline='',encoding='utf-8') as stream:
        w=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(cells)
    book=Workbook();sheet=book.active;sheet.title='Cells';sheet.append(fields)
    for r in cells:sheet.append([r[k] for k in fields])
    sheet=book.create_sheet('Summary');sheet.append(['Metric','Value'])
    for k,v in summary.items():sheet.append([k,json.dumps(v)])
    sheet.append(['Policy',result['policy']]);book.save(args.output/'contour-evaluation.xlsx');book.close()
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
