"""Evaluate the two supported pipelines on an explicit held-out audit."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
from PySide6 import QtCore
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fish_viewer.evaluation import evaluate_run,frozen_models,paired_stats,save,sha,rebuild_references,channel_for
from fish_viewer.analysis.controller import AnalysisController
MODES=('consensus','stardist2d')
_application = None

def summarize(results):
 output={}
 for split in ('test','validation','all_heldout'):
  subset=[r for r in results if split=='all_heldout' or r['split']==split];modes={}
  for mode in MODES:
   rows=[r for r in subset if r['mode']==mode]
   totals={key:sum(r[key] for r in rows) for key in ('references','complete_references','cells_before_filter','final_cells','signals','matched_before_filter','matched_final','unscored_before_filter','unscored_final')}
   totals['scans']=len(rows);totals['annotated_recall_before_filter']=totals['matched_before_filter']/totals['references'] if totals['references'] else None
   totals['annotated_recall_after_filter']=totals['matched_final']/totals['references'] if totals['references'] else None
   for field,a,b,pct in [('volume_pairs','manual_um3','predicted_um3',True),('distance_pairs','manual_um','predicted_um',False)]:
    pairs=[dict(p,group=r['group']) for r in rows for p in r[field]];totals[field]=paired_stats(pairs,a,b,pct)
   modes[mode]=totals
  shared={}
  by_image={r['image'] for r in subset}
  for field,key,a,b,pct in [('volume_pairs',lambda p:p['reference_id'],'manual_um3','predicted_um3',True),
                           ('distance_pairs',lambda p:(p['reference_id'],p['channel'],tuple(np.round(p['manual_xyz'],4))),'manual_um','predicted_um',False)]:
   pairs={m:[] for m in MODES}
   for image in by_image:
    rows={r['mode']:r for r in subset if r['image']==image}
    if set(rows)!=set(MODES):continue
    index={m:{key(p):p for p in rows[m][field]} for m in MODES};common=set.intersection(*(set(v) for v in index.values()))
    for m in MODES:pairs[m].extend(dict(index[m][k],group=rows[m]['group']) for k in common)
   shared[field]={m:paired_stats(v,a,b,pct) for m,v in pairs.items()}
  output[split]=dict(modes=modes,common_matches=shared)
 return output


def aggregate(root):
 selection=json.loads((root/'selection.json').read_text());cases=selection['cases'];identity=json.loads((root/'identity.json').read_text())
 for kind,model in identity.get('models',{}).items():
  directory=Path(model['directory'])
  manifest_name={'center_box_detector':'proposal-model.json','stardist_detector':'fish-model.json','stardist2d':'fish-model.json','contours':'refinement-model.json','signals':'signal-model.json'}[kind]
  if sha(directory/model['manifest']['checkpoint'])!=model['checkpoint_sha256'] or sha(directory/manifest_name)!=model['manifest_sha256']:
   raise ValueError('Frozen model changed during the benchmark: '+kind)
 results=[];expected=set()
 for i,case in enumerate(cases):
  for mode in MODES:
   expected.add((case['image'],mode));folder=root/(f'{i+1:03d}-'+Path(case['image']).stem)/mode
   row=json.loads((folder/'metrics.json').read_text())
   if row['image']!=case['image'] or row['mode']!=mode or row['split']!=case['audit']['split']:raise ValueError('Incorrect scan/mode/split in '+str(folder))
   state=json.loads((folder/'pipeline-result.json').read_text())
   if state['status']!='complete':raise ValueError('Incomplete pipeline '+str(folder))
   cuts=case['audit']['reference_cut']
   for phase in ('before_filter','final'):
    matches=row['matches_'+phase]
    row['matched_complete_'+phase]=sum(not cuts[p['reference_id']] for p in matches)
    row['matched_cut_'+phase]=sum(cuts[p['reference_id']] for p in matches)
   for p in row['distance_pairs']:p['cut']=cuts[p['reference_id']]
   results.append(row)
 if len(results)!=len(expected) or {(r['image'],r['mode']) for r in results}!=expected:raise ValueError('Coverage does not match all held-out scans.')
 summary=summarize(results)
 for split,data in summary.items():
  subset=[r for r in results if split=='all_heldout' or r['split']==split]
  for mode,totals in data['modes'].items():
   rows=[r for r in subset if r['mode']==mode]
   totals.update({key:sum(r[key] for r in rows) for key in ('matched_complete_before_filter','matched_complete_final','matched_cut_before_filter','matched_cut_final')})
   totals['complete_cell_recall_before_filter']=totals['matched_complete_before_filter']/totals['complete_references'] if totals['complete_references'] else None
   totals['complete_cell_recall_after_filter']=totals['matched_complete_final']/totals['complete_references'] if totals['complete_references'] else None
   for field,a,b,pct in [('volume_pairs','manual_um3','predicted_um3',True),('distance_pairs','manual_um','predicted_um',False)]:
    pairs=[dict(p,group=r['group']) for r in rows for p in r[field] if not p['cut']]
    totals[field+'_complete_cells']=paired_stats(pairs,a,b,pct)
  common={}
  for field,key,a,b,pct in [('volume_pairs',lambda p:p['reference_id'],'manual_um3','predicted_um3',True),
                           ('distance_pairs',lambda p:(p['reference_id'],p['channel'],tuple(np.round(p['manual_xyz'],4))),'manual_um','predicted_um',False)]:
   pairs={m:[] for m in MODES}
   for image in {r['image'] for r in subset}:
    rows={r['mode']:r for r in subset if r['image']==image}
    index={m:{key(p):p for p in rows[m][field] if not p['cut']} for m in MODES}
    shared=set.intersection(*(set(v) for v in index.values()))
    for m in MODES:pairs[m].extend(dict(index[m][k],group=rows[m]['group']) for k in shared)
   common[field]={m:paired_stats(v,a,b,pct) for m,v in pairs.items()}
  data['common_matches_complete_cells']=common
 report=dict(status='complete',expected_scans=len(cases),expected_mode_runs=len(expected),completed_mode_runs=len(results),identity=identity,
   selection=selection,results=results,summary=summary,failures=[],
   policy='Frozen checkpoints, acquisition split, complete-cell primary errors, original manual AB distances only; image export omitted.',
   limitations='Sparse annotations leave unmatched cells unscored. Reused test data are exploratory, not a prospective test.')
 save(root/'report.json',report);save(root/'progress.json',dict(status='complete',completed=len(results),total=len(expected),failures=[]))
 return report


def run_case(case, folder, mode, device, roundness, channel):
    global _application
    _application=QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
    loop=QtCore.QEventLoop()
    controller=AnalysisController(_application)
    result={}
    original=controller._launch
    def launch(index, arguments):
        if controller.stages[index][0]=='export':
            controller.state.update(status='complete',benchmark_only=True,export_omitted=True)
            controller._persist()
            controller._stop()
            result.update(controller.state,report_path=str(controller.directory/'analysis-result.json'))
            loop.quit()
        else: original(index,arguments)
    def failed(message):
        result['error']=message
        loop.quit()
    controller._launch=launch
    controller.failed.connect(failed)
    controller.start(case['image'],folder,pipeline=mode,device=device,roundness=roundness,
                     cell_channel=channel,source_scene=case.get('scene'))
    loop.exec()
    if result.get('error'):raise RuntimeError(result['error'])
    return result


def main(argv=None, pipelines=MODES):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--channel-audit',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['auto','cpu','gpu'],default='auto')
    p.add_argument('--roundness',type=float,default=.82)
    p.add_argument('--split',choices=['test','validation','all_heldout'],default='all_heldout')
    args=p.parse_args(argv)
    collection=json.loads(args.audit.read_text())
    if collection['status']!='complete':raise ValueError('Complete external audit required.')
    cases=sorted([c for c in collection['cases'] if c['audit']['split'] in (('test','validation') if args.split=='all_heldout' else (args.split,))],key=lambda c:c['image'])
    if not cases:raise ValueError('No held-out scans in the selected split.')
    training={c['audit']['group'] for c in collection['cases'] if c['audit']['split']=='training'}
    if training & {c['audit']['group'] for c in cases}:raise ValueError('Training/held-out acquisition overlap.')
    channel_cases={c['image']:c for c in json.loads((args.channel_audit or args.audit).read_text())['cases']}
    channels={c['image']:channel_for(channel_cases[c['image']])[0] for c in cases}
    args.output.mkdir(parents=True,exist_ok=False)
    identity=dict(models=frozen_models(),audit_sha256=sha(args.audit),channel_audit_sha256=sha(args.channel_audit or args.audit),channels=channels,device=args.device,roundness=args.roundness,pipelines=list(pipelines))
    save(args.output/'identity.json',identity)
    save(args.output/'selection.json',dict(cases=cases))
    rows=[]
    for i,case in enumerate(cases):
        for mode in pipelines:
            folder=args.output/(f'{i+1:03d}-'+Path(case['image']).stem)/mode
            folder.mkdir(parents=True)
            with tempfile.TemporaryDirectory(prefix='fish-evaluation-') as temporary:
                scratch=Path(temporary)
                rebuilt=rebuild_references(case,collection,scratch/'references')
                state=run_case(case,scratch/'runs',mode,args.device,args.roundness,channels[case['image']])
                row=evaluate_run(rebuilt,state,collection,scratch/'references')
                row.update(mode=mode,cell_channel=channels[case['image']],cut_references=case['audit']['reference_cut'])
                row.pop('pipeline_report',None)
                save(folder/'metrics.json',row)
                save(folder/'pipeline-result.json',dict(status='complete',benchmark_only=True,export_omitted=True))
                rows.append(row)
            print(f'{i+1}/{len(cases)} {mode}: {row["final_cells"]} cells',flush=True)
    if tuple(pipelines)==MODES: report=aggregate(args.output)
    else:
        from fish_viewer.evaluation import summarize_complete
        report=dict(status='complete',results=rows,summary=summarize_complete(rows),identity=identity)
        save(args.output/'report.json',report)
    print(json.dumps(report['summary'],indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
