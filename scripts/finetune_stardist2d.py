"""Fine-tune official single-channel StarDist2D; benchmark all held-out volumes."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from fish_viewer.worker_runtime import restart_with_cuda as restart_cuda
from fish_viewer.neural.slices import SlicePatches, SliceView, link_sections, predict_sections
from fish_viewer.pipeline.runner import _json,save_prediction

URL='https://github.com/stardist/stardist-models/releases/download/v0.1/python_2D_versatile_fluo.zip'
ARCHIVE_SHA='4ad678d0758eed6e55625f1b5ae30771e59adb79f1239e09b9772eac8846c3dd'




def augment(x,y):
    k=np.random.randint(4);x=np.rot90(x,k);y=np.rot90(y,k)
    for axis in range(2):
        if np.random.random()<.5:x=np.flip(x,axis);y=np.flip(y,axis)
    x=np.clip(x*np.random.uniform(.8,1.2)+np.random.normal(0,.01,x.shape),0,1)
    return np.ascontiguousarray(x,dtype='float32'),np.ascontiguousarray(y)


def inference(model,case,threshold):
    image=np.load(Path(case['directory'])/'image.npy',mmap_mode='r')
    return link_sections(predict_sections(model,image,threshold))


def benchmark(model,cases,directory,threshold,config):
    from fish_viewer.neural.engine import filter_instances
    from fish_viewer.pipeline.evaluation import evaluate,aggregate
    from fish_viewer.neural.audit import read_references
    from fish_viewer.pipeline.segmentation import Prepared
    from fish_viewer.models import LibraryEntry
    directory.mkdir(parents=True,exist_ok=True)
    rows=[]
    for i,case in enumerate(cases,1):
        a=case['audit']; path=Path(case['directory'])
        image=np.load(path/'image.npy',mmap_mode='r');raw=np.load(path/'evidence.npy',mmap_mode='r')
        prepared=Prepared(image,tuple(a['spacing_zyx']),tuple(a['origin_xyz']),a['channel'],a['channel_scores'],a['fingerprint'],{}, {},evidence_image=raw)
        prediction=filter_instances(inference(model,case,threshold),prepared,config)
        refs,names=read_references(path/'references.npz')
        complete=[n for n,cut in enumerate(a['reference_cut']) if not cut]
        metrics=evaluate(prediction.labels,[refs[n] for n in complete],[names[n] for n in complete],prepared.spacing)
        # Preserve original manual IDs for the downstream surface evaluator.
        all_metrics=evaluate(prediction.labels,refs,names,prepared.spacing)
        output=directory/f'{i:04d}-contours.npz'
        entry=LibraryEntry(Path(case['image']),Path(case['scene']) if case['scene'] else None)
        save_prediction(output,prediction,prepared,entry,config,all_metrics)
        rows.append(dict(image=case['image'],scene=case['scene'],predictions=str(output.resolve()),cells=len(prediction.cells),metrics=all_metrics,complete_metrics=metrics))
        _json(directory/'report.json',dict(status='running',files=rows))
        if i%50==0 or i==len(cases):print(f'{directory.name}: {i}/{len(cases)} scans',flush=True)
    summary=aggregate([r['complete_metrics'] for r in rows])
    summary['matched_iou_50']=sum(sum(p['detected'] for p in r['complete_metrics']['per_cell']) for r in rows)
    report=dict(status='complete',files=rows,summary=summary,config=asdict(config))
    _json(directory/'report.json',report);return summary


def main():
    restart_cuda()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--epochs',type=int,default=40)
    parser.add_argument('--steps',type=int,default=100);parser.add_argument('--device',default='gpu',choices=['gpu','cpu'])
    parser.add_argument('--evaluate-only',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=args.evaluate_only)
    from fish_viewer.neural.engine import runtime
    tf,_,_,runtime_info=runtime(args.device)
    from stardist.models import StarDist2D, Config2D, StarDistData2D
    audit=json.loads(args.audit.read_text()); cases=audit['cases']
    splits={s:[c for c in cases if c['audit']['split']==s] for s in ('training','validation','test')}
    groups=[{c['audit']['group'] for c in splits[s]} for s in splits]
    if any(groups[i]&groups[j] for i in range(3) for j in range(i)):raise ValueError('Acquisition leakage')
    archive=ROOT/'artifacts/stardist2d-official/model.zip'
    if not archive.exists():
        from urllib.request import urlretrieve
        archive.parent.mkdir(parents=True,exist_ok=True);urlretrieve(URL,archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=ARCHIVE_SHA:raise ValueError('Official archive checksum mismatch')
    official_dir=args.output/'official';official_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:z.extractall(official_dir)
    original=StarDist2D(None,name='official',basedir=str(args.output))
    threshold=float(original.thresholds.prob)
    model_dir=args.output/'model'
    report=dict(status='training',official=dict(name='2D_versatile_fluo',url=URL,archive_sha256=ARCHIVE_SHA),
        audit=str(args.audit.resolve()),runtime=runtime_info,split={s:[c['image'] for c in cs] for s,cs in splits.items()},epochs=[],
        policy='Acquisition-safe XY sections of complete 3D manual cells; other real pixels unknown=-1; synthetic blank/noise=0. No held-out optimization.',
        input=dict(channel='Corrected cell channel from stain-aware audit',preprocessing='membrane-connected-v1',xy_scale=2,z_sections_per_cell=7),
        inference=dict(probability=threshold,nms=.3,linking_iou=.2))
    if not args.evaluate_only:
        cfg=dict(vars(original.config));cfg.update(train_patch_size=(256,256),train_batch_size=4,train_learning_rate=3e-5,
            train_background_reg=0,train_foreground_only=1,train_sample_cache=False,train_tensorboard=False,
            train_checkpoint='best.weights.h5',train_checkpoint_last='last.weights.h5',train_reduce_lr=None)
        model=StarDist2D(Config2D(**cfg),name='model',basedir=str(args.output))
        model.keras_model.set_weights(original.keras_model.get_weights())
        initial_sha=hashlib.sha256((official_dir/'weights_best.h5').read_bytes()).hexdigest()
        report['initialization_checkpoint_sha256']=initial_sha
        model.prepare_for_training();model.keras_model.jit_compile=False
        train=SlicePatches(splits['training']);val=SlicePatches(splits['validation'],sections=3,synthetic_fraction=.25)
        report['patches']=dict(training_cells=train.cells,training_slices=train.real,synthetic_training=train.synthetic,
            validation_cells=val.cells,validation_slices=val.real,synthetic_validation=val.synthetic)
        print(json.dumps(report['patches']),flush=True)
        kwargs=dict(n_rays=model.config.n_rays,grid=model.config.grid,patch_size=(256,256),batch_size=4,
            use_gpu=False,foreground_prob=1,sample_ind_cache=False,
            keras_kwargs=dict(workers=1,use_multiprocessing=False,max_queue_size=2))
        td=StarDistData2D(SliceView(train,0),SliceView(train,1),length=args.epochs*args.steps,augmenter=augment,**kwargs)
        class Ordered(StarDistData2D):
            def batch(self,i):return np.arange(i*self.batch_size,min((i+1)*self.batch_size,self.data_size))
        vd=Ordered(SliceView(val,0),SliceView(val,1),length=int(np.ceil(len(val)/4)),**kwargs)
        started=time.monotonic()
        class Progress(tf.keras.callbacks.Callback):
            def on_epoch_end(self,epoch,logs=None):
                losses={k:float(v) for k,v in logs.items()}
                if not all(np.isfinite(v) for v in losses.values()):raise ValueError('Non-finite training loss')
                report['epochs'].append(dict(epoch=epoch+1,losses=losses,elapsed_s=time.monotonic()-started))
                _json(args.output/'training.json',report)
                print(f'Epoch {epoch+1}: loss={losses["loss"]:.4f} val={losses["val_loss"]:.4f}',flush=True)
        callbacks=[Progress(),tf.keras.callbacks.ModelCheckpoint(str(model_dir/'best.weights.h5'),monitor='val_loss',save_best_only=True,save_weights_only=True),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=10),tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',patience=4,factor=.5,min_lr=1e-6)]
        model.keras_model.fit(iter(td),validation_data=iter(vd),validation_steps=len(vd),epochs=args.epochs,steps_per_epoch=args.steps,callbacks=callbacks,verbose=0)
        report['status']='complete';report['best_epoch']=min(report['epochs'],key=lambda e:e['losses']['val_loss'])['epoch']
        _json(args.output/'training.json',report)
        _json(model_dir/'fish-model.json',dict(version=1,engine='stardist2d-slice',checkpoint='best.weights.h5',experiment=report,
            warning='Experimental 2D-to-3D linking; not compatible with the StarDist3D app preset'))
        _json(model_dir/'thresholds.json',dict(prob=threshold,nms=.3))
    else:report=json.loads((args.output/'training.json').read_text())
    report['inference'].pop('minimum_adjacent_sections',None)
    _json(args.output/'training.json',report)
    model=StarDist2D(None,name='model',basedir=str(args.output));model.load_weights('best.weights.h5')
    from fish_viewer.neural.config import NeuralConfig
    from fish_viewer.neural.precision import evidence_preset
    config=evidence_preset(NeuralConfig.from_dict(json.loads((ROOT/'models/cells-stardist2d/fish-model.json').read_text())['acceptance_config']))
    manifest=json.loads((model_dir/'fish-model.json').read_text())
    from fish_viewer.neural.preprocessing import RECIPE
    manifest['preprocessing_recipe']=RECIPE
    manifest['acceptance_config']=asdict(config)
    manifest['experiment']['inference'].pop('minimum_adjacent_sections',None)
    _json(model_dir/'fish-model.json',manifest)
    summary={}
    for name,network in [('pretrained',original),('finetuned',model)]:
        summary[name]=benchmark(network,splits['test'],args.output/name,threshold,config)
        path=args.output/name/'report.json';r=json.loads(path.read_text())
        checkpoint=official_dir/'weights_best.h5' if name=='pretrained' else model_dir/'best.weights.h5'
        r['checkpoint_sha256']=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        _json(path,r)
    _json(args.output/'comparison.json',dict(training=report,heldout=summary))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
