"""Train every complete target per epoch; select checkpoint on validation only."""
from dataclasses import asdict
import csv
import hashlib
from pathlib import Path
import time
import traceback

import numpy as np
from scipy import ndimage as ndi

from ..neural.engine import runtime
from ..pipeline.runner import _json
from ..pipeline.segmentation import Cancelled, checkpoint
from ..run_paths import new_run_directory
from .data import cache_collection, cache_patches, seed_image
from .network import network, loss


def charts(directory, report):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    rows = report['epochs']
    boundaries = any('boundary_error_um' in r.get('validation',{}) for r in rows)
    figure = Figure(figsize=(11,8 if boundaries else 4.5), layout='constrained'); FigureCanvasAgg(figure)
    axes = figure.subplots(2,2).ravel() if boundaries else figure.subplots(1,2)
    for ax, key, title in [(axes[0],'loss','Boundary-weighted training loss'),
                            (axes[1],'dice','Soft target-cell Dice')]:
        for split in ('training','validation'):
            ax.plot([r['epoch'] for r in rows], [r[split][key] for r in rows], label=split)
        ax.set(xlabel='Epoch', title=title); ax.grid(alpha=.2); ax.legend()
    if boundaries:
        axes[2].plot([r['epoch'] for r in rows],[r['validation'].get('boundary_error_um') for r in rows])
        axes[2].set(xlabel='Epoch',ylabel='µm',title='Validation symmetric boundary error')
        if report.get('initial_validation'):
            axes[2].axhline(report['initial_validation']['boundary_error_um'],color='gray',linestyle='--',label='Initial checkpoint')
            axes[2].legend()
        for key in ('volume_mape','volume_bias_percent'):
            axes[3].plot([r['epoch'] for r in rows],[r['validation'].get(key) for r in rows],label=key)
        axes[3].axhline(0,color='gray',linestyle='--');axes[3].legend()
        axes[3].set(xlabel='Epoch',ylabel='%',title='Validation volume error and signed bias')
        axes[2].grid(alpha=.2);axes[3].grid(alpha=.2)
    figure.savefig(directory/'training-progress.png', dpi=160)
    with (directory/'epochs.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['epoch','training_loss','validation_loss',
                                                'training_dice','validation_dice','elapsed_s'])
        writer.writeheader()
        for r in rows:
            writer.writerow(dict(epoch=r['epoch'],elapsed_s=r['elapsed_s'],
                **{f'{split}_{key}':r[split][key] for split in ('training','validation') for key in ('loss','dice')}))


def batch(arrays, records, indices, config, rng, augment=False):
    images=[]; targets=[]
    for index in indices:
        if index<0:
            size=(config.patch_size,)*3
            amplitude=rng.uniform(.001,config.negative_noise_max)
            image=np.maximum(0,rng.normal(0,amplitude,size)).astype(np.float32)
            if config.negative_noise_max>.02:
                texture=ndi.gaussian_filter(rng.normal(0,1,size),sigma=rng.uniform(.6,2.))
                image=np.clip(image+texture*rng.uniform(0,.1)/max(texture.std(),1e-6)+rng.uniform(0,.15),0,1)
            center=np.full(3,(config.patch_size-1)/2)
            seed=seed_image(center,.2,config)
            truth=np.zeros(size,np.float32); weight=np.ones(size,np.float32); valid=weight.copy()
        else:
            record=records[index]
            image=np.asarray(arrays['images'][index],np.float32)
            truth=np.asarray(arrays['masks'][index],np.float32)
            weight=np.asarray(arrays['weights'][index],np.float32)
            valid=np.asarray(arrays['valid'][index],np.float32)
            center=np.array(record['seed_patch'])
            if augment:
                trial=center+rng.normal(0,config.seed_jitter_um/record['patch_spacing_um'],3)
                point=np.clip(np.rint(trial).astype(int),0,config.patch_size-1)
                if truth[tuple(point)]>0:
                    center=trial
            seed=seed_image(center,record['patch_spacing_um'],config)
        x=np.stack([image,seed],axis=-1)
        y=np.stack([truth,weight,valid],axis=-1)
        if augment:
            if index>=0 and (config.crop_scale_jitter or config.crop_shift_um):
                scale=rng.uniform(1-config.crop_scale_jitter,1+config.crop_scale_jitter)
                offset=(config.patch_size-1)/2*(1-scale)+rng.uniform(-config.crop_shift_um,config.crop_shift_um,3)/record['patch_spacing_um']
                x=np.stack([ndi.affine_transform(x[...,c],np.eye(3)*scale,offset,order=1,mode='constant',cval=0,prefilter=False) for c in range(2)],axis=-1)
                y=np.stack([ndi.affine_transform(y[...,c],np.eye(3)*scale,offset,order=1 if c==1 else 0,mode='constant',cval=0,prefilter=False) for c in range(3)],axis=-1)
            turns=int(rng.integers(4))
            x=np.rot90(x,turns,axes=(1,2)); y=np.rot90(y,turns,axes=(1,2))
            for axis in range(3):
                if rng.random()<.5:
                    x=np.flip(x,axis); y=np.flip(y,axis)
            x=np.array(x,copy=True)
            x[...,0]=np.clip(x[...,0]*rng.uniform(.75,1.25)+rng.uniform(-.03,.03)
                             +rng.normal(0,.008,x.shape[:3]),0,1)
        if config.surface_loss_weight:
            # Derive distances after spatial augmentation, in physical units.
            mask = y[...,0] > .5
            step_um = record['patch_spacing_um'] if index >= 0 else .2
            if mask.any():
                distance = ndi.distance_transform_edt(~mask, sampling=step_um)-ndi.distance_transform_edt(mask, sampling=step_um)
                distance = np.clip(distance/config.surface_distance_clip_um,-1,1)
            else:
                distance = np.ones(mask.shape,np.float32)
            y = np.concatenate([y,distance[...,None]],axis=-1)
        images.append(x); targets.append(y)
    return np.ascontiguousarray(images,np.float32), np.ascontiguousarray(targets,np.float32)


def validation_boundaries(probability, target, selected, records, config):
    """Physical surface agreement and signed volume error, complete val targets."""
    from .inference import seeded_component
    values=[]
    for p,y,index in zip(probability,target,selected):
        if not records[index].get('validation_checkpoint_score',True):continue
        truth=(y[...,0]>.5)&(y[...,2]>.5)
        found=seeded_component((p[...,0]>=config.threshold)&(y[...,2]>.5),records[index]['seed_patch'])
        a=found & ~ndi.binary_erosion(found);b=truth & ~ndi.binary_erosion(truth)
        spacing=records[index]['patch_spacing_um']
        if a.any() and b.any():
            da=ndi.distance_transform_edt(~a,sampling=spacing)
            db=ndi.distance_transform_edt(~b,sampling=spacing)
            boundary=float(np.mean(np.r_[db[a],da[b]]))
        else:boundary=spacing*config.patch_size
        bias=100*(found.sum()-truth.sum())/max(1,truth.sum())
        values.append((boundary,float(abs(bias)),float(bias)))
    return values


def train(audit, cache, patches, output, config, device='auto', emit=lambda e:None, cancel=None, initial_model=None):
    config.validate()
    directory=new_run_directory(output,'boundary-train')
    model_dir=directory/'model';model_dir.mkdir()
    report=dict(version=1,engine='seeded-unet3d',mode='train',status='running',directory=str(directory),
                config=asdict(config),audit=str(Path(audit).resolve()),epochs=[],model_directory=str(model_dir))
    start=time.monotonic()
    def notify(event):
        if event['kind']=='progress':
            report['progress']={k:v for k,v in event.items() if k!='kind'}
            report['elapsed_s']=time.monotonic()-start
            _json(directory/'report.json',report)
        emit(event)
    emit(dict(kind='run_started',directory=str(directory)))
    _json(directory/'report.json',report)
    try:
        collection=cache_collection(audit,cache,config,notify,cancel)
        index=cache_patches(collection,patches,config,notify,cancel)
        records=index['records']; report['collection']=collection['summary']
        train_ids=[r['index'] for r in records if r['split']=='training']
        val_ids=[r['index'] for r in records if r['split']=='validation']
        if not train_ids or not val_ids:
            raise ValueError('Both independent training and validation targets are required.')
        negative_count=int(np.ceil(len(train_ids)*config.negative_fraction))
        report['samples']=dict(training_cells=len(train_ids),validation_cells=len(val_ids),
                               validation_boundary_scored_cells=sum(records[i].get('validation_checkpoint_score',True) for i in val_ids),
                               synthetic_negative_crops=negative_count,
                               policy='Every training cell once per epoch; no roundness selection. '
                                      'Synthetic negatives supplement targets; no reviewed real negatives available.')
        report['split']={s:[c['image'] for c in collection['cases'] if c['audit']['split']==s]
                         for s in ('training','validation','test')}
        arrays={n:np.load(Path(patches)/(n+'.npy'),mmap_mode='r') for n in ('images','masks','weights','valid')}
        notify(dict(kind='progress',phase='Initializing 3D U-Net',current=0,total=1))
        tf,_,_,info=runtime(device,config.seed);report['runtime']=info
        # Mixed precision reduces memory/compute; logits and loss remain float32.
        if info['device']=='GPU':
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
        model=network(tf,config)
        if initial_model:
            from .config import read_model
            initial=read_model(initial_model)
            for key in ('filters','depth'):
                if initial['config'][key]!=getattr(config,key):raise ValueError('Initial boundary architecture differs: '+key)
            report['initial_model']=str(Path(initial_model).resolve())
            report['initial_checkpoint_sha256']=hashlib.sha256((Path(initial_model)/initial['checkpoint']).read_bytes()).hexdigest()
        optimizer=tf.keras.optimizers.Adam(config.learning_rate,clipnorm=5.)
        if info['device']=='GPU':
            optimizer=tf.keras.mixed_precision.LossScaleOptimizer(optimizer,initial_scale=128.)
        @tf.function(jit_compile=False,reduce_retracing=True)
        def train_step(x,y):
            with tf.GradientTape() as tape:
                value,dice=loss(tf,y,model(x,training=True),config)
                scaled=optimizer.scale_loss(value) if hasattr(optimizer,'scale_loss') else value
            gradients=tape.gradient(scaled,model.trainable_variables)
            optimizer.apply_gradients(zip(gradients,model.trainable_variables))
            return value,dice
        @tf.function(jit_compile=False,reduce_retracing=True)
        def validation_step(x,y):
            logits=model(x,training=False)
            value,dice=loss(tf,y,logits,config)
            if config.surface_loss_weight:
                return value,dice,tf.math.sigmoid(logits)
            return value,dice
        rng=np.random.default_rng(config.seed);best=float('inf');stale=0
        manifest=dict(version=1,engine='seeded-unet3d',config=asdict(config),checkpoint='best.weights.h5',
                      input='Normalized membrane + physical Gaussian seed; target-conditioned full 3D mask.',
                      output='Target-cell logits; no star-convex, sphericity or elongation constraint.',
                      supervision=index['policy'],samples=report['samples'],runtime=info,
                      audit_sha256=hashlib.sha256(Path(audit).read_bytes()).hexdigest())
        if initial_model:
            manifest.update(initial_model=report['initial_model'],initial_checkpoint_sha256=report['initial_checkpoint_sha256'])
        manifest['checkpoint_selection']='Validation physical boundary error with non-increasing volume MAPE; epoch zero retained if no improvement' if config.surface_loss_weight else 'Validation loss'
        _json(model_dir/'refinement-model.json',manifest)
        best_surface=float('inf')
        if initial_model and config.surface_loss_weight:
            measurements=[];initial_losses=[]
            for start_index in range(0,len(val_ids),config.batch_size):
                checkpoint(cancel)
                ids=val_ids[start_index:start_index+config.batch_size]
                x,y=batch(arrays,records,ids,config,rng)
                v,d,p=validation_step(x,y)
                measurements.extend(validation_boundaries(p.numpy(),y,ids,records,config))
                initial_losses.extend([float(v.numpy())]*len(ids))
                if start_index%100==0:
                    notify(dict(kind='progress',phase='Scoring initial contour boundary',current=start_index,total=len(val_ids)))
            initial_mean=np.mean(measurements,axis=0)
            report['initial_validation']=dict(boundary_error_um=float(initial_mean[0]),volume_mape=float(initial_mean[1]),volume_bias_percent=float(initial_mean[2]),loss=float(np.mean(initial_losses)))
            best_surface=float(initial_mean[0]);best=report['initial_validation']['loss']
            report['best_epoch']=0
            model.save_weights(model_dir/'best.weights.h5')
            manifest.update(best_epoch=0,validation_loss=best,validation_boundary_error_um=best_surface)
            _json(model_dir/'refinement-model.json',manifest)
        for epoch in range(1,config.epochs+1):
            checkpoint(cancel)
            row=dict(epoch=epoch)
            for split, ids, step in [('training',rng.permutation(train_ids+[-1]*negative_count),train_step),
                                     ('validation',np.asarray(val_ids),validation_step)]:
                values=[];counts=[];surface_values=[];total=int(np.ceil(len(ids)/config.batch_size))
                for b,start_index in enumerate(range(0,len(ids),config.batch_size)):
                    checkpoint(cancel)
                    selected=ids[start_index:start_index+config.batch_size]
                    x,y=batch(arrays,records,selected,config,rng,augment=split=='training')
                    outputs=step(x,y)
                    result=[float(v.numpy()) for v in outputs[:2]]
                    if split=='validation' and config.surface_loss_weight:
                        surface_values.extend(validation_boundaries(outputs[2].numpy(),y,selected,records,config))
                    if not np.isfinite(result).all():
                        raise ValueError('Non-finite boundary training metrics.')
                    values.append(result);counts.append(len(selected))
                    if b%10==0 or b+1==total:
                        notify(dict(kind='progress',phase=f'{split.title()} boundary network',epoch=epoch,
                                    epochs=config.epochs,current=b+1,total=total,loss=result[0],dice=result[1]))
                means=np.average(values,axis=0,weights=counts)
                row[split]=dict(loss=float(means[0]),dice=float(means[1]))
                if surface_values:
                    metrics=np.mean(surface_values,axis=0)
                    row[split].update(boundary_error_um=float(metrics[0]),volume_mape=float(metrics[1]),volume_bias_percent=float(metrics[2]))
            row['elapsed_s']=time.monotonic()-start
            report['epochs'].append(row)
            if config.surface_loss_weight:
                improved=row['validation']['boundary_error_um']<best_surface-1e-5 and row['validation']['volume_mape']<=report.get('initial_validation',{}).get('volume_mape',float('inf'))
            else:improved=row['validation']['loss']<best-1e-4
            if improved:
                best=row['validation']['loss'];stale=0
                if config.surface_loss_weight:best_surface=row['validation']['boundary_error_um']
                temporary=model_dir/'checkpoint.tmp.weights.h5'
                model.save_weights(temporary);temporary.replace(model_dir/'best.weights.h5')
                report['best_epoch']=epoch
                manifest.update(best_epoch=epoch,validation_loss=best)
                if config.surface_loss_weight:manifest['validation_boundary_error_um']=best_surface
                _json(model_dir/'refinement-model.json',manifest)
            else:
                stale+=1
            charts(directory,report);_json(directory/'report.json',report)
            emit(dict(kind='epoch',**row,best_epoch=report['best_epoch']))
            if stale>=config.patience:
                report['stop_reason']='Validation boundary agreement stopped improving.' if config.surface_loss_weight else 'Validation loss stopped improving.';break
        report['status']='complete'
        report['checkpoint_sha256']=hashlib.sha256((model_dir/'best.weights.h5').read_bytes()).hexdigest()
        notify(dict(kind='progress',phase='Boundary training complete',current=len(report['epochs']),
                    total=len(report['epochs']),epoch=len(report['epochs']),epochs=config.epochs))
    except Cancelled:
        report['status']='cancelled'
    except Exception:
        report.update(status='failed',error=traceback.format_exc())
    report['elapsed_s']=time.monotonic()-start;_json(directory/'report.json',report)
    emit(dict(kind='finished',report=report))
    return report
