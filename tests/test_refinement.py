from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import ndimage as ndi

from fish_viewer.refinement.config import RefinementConfig, read_model
from fish_viewer.refinement.data import sample_target
from fish_viewer.refinement.inference import resample_labels, refine, surface_evidence
from fish_viewer.refinement.preprocessing import augment_membrane, preprocess_membrane


def test_finer_labels_preserve_physical_centers_and_instance_ids():
    coarse=np.zeros((8,)*3,np.uint32);coarse[2:6,2:6,2:6]=7
    fine=resample_labels(coarse,np.array([.3]*3),np.array([.15]*3),
                         (16,)*3,(.15,)*3,(.075,)*3)
    assert set(np.unique(fine))=={0,7}
    assert np.array_equal(np.argwhere(fine==7).mean(0)*.15+.075,[1.2]*3)
    assert np.count_nonzero(fine==7)==8*np.count_nonzero(coarse==7)


def test_target_crop_labels_neighbor_as_other_instance_and_ignores_padding():
    config=replace(RefinementConfig(),patch_size=32,margin_um=.8)
    image=np.zeros((24,)*3,np.float32)
    box=(slice(1,9),slice(1,9),slice(1,9));mask=np.ones((8,)*3,bool)
    image[box]=.2;image[12:20,1:9,1:9]=1.  # unannotated neighbor, not global negative
    sample,truth,weight,valid,seed,step=sample_target(image,box,mask,(.2,)*3,config)
    assert not truth[sample>.8].any()  # not a member of this chosen target
    assert truth.any() and (~valid).any()
    assert (weight[~valid]==0).all()
    assert truth[tuple(np.rint(seed).astype(int))]
    assert step>0


def test_surface_normals_follow_visible_concave_boundary_and_reject_blank():
    grid=np.indices((48,)*3);radius=np.linalg.norm((grid-23.5)*.2,axis=0)
    mask=radius<3.
    mask[16:32,16:32,30:]=False  # a visible indentation
    boundary=mask & ~ndi.binary_erosion(mask)
    image=np.exp(-(ndi.distance_transform_edt(~boundary,sampling=.2)/.2)**2).astype(np.float32)
    visible=surface_evidence(mask,image,(.2,)*3)
    blank=surface_evidence(mask,np.zeros_like(image),(.2,)*3)
    assert visible['surface_support']>.7
    assert blank['surface_support']==0


def test_refinement_can_indent_a_cell_without_adding_instances():
    config=replace(RefinementConfig(),patch_size=32,margin_um=1.)
    shape=(40,)*3;grid=np.indices(shape);radius=np.linalg.norm(grid-19.5,axis=0)
    labels=(radius<8).astype(np.uint32)
    image=np.exp(-((radius-8)/1.2)**2).astype(np.float32)
    prepared=SimpleNamespace(image=image,spacing=(.2,)*3)
    class IndentedModel:
        def __call__(self,x,training=False):
            g=np.indices((32,)*3);r=np.linalg.norm(g-15.5,axis=0)
            target=r<8.7
            target[13:19,13:19,21:]=False
            return np.where(target,12.,-12.)[None,...,None]
    result,diagnostics=refine(labels,prepared,IndentedModel(),config,evidence_threshold=0.)
    assert set(np.unique(result))=={0,1}
    assert diagnostics[0]['accepted']
    assert np.count_nonzero(result)<np.count_nonzero(labels)
    class EmptyModel:
        def __call__(self,x,training=False):return np.full((1,32,32,32,1),-20.)
    fallback,diagnostics=refine(labels,prepared,EmptyModel(),config,evidence_threshold=0.)
    assert np.array_equal(fallback,labels)
    assert not diagnostics[0]['accepted'] and 'seed' in diagnostics[0]['fallback_reason']


def test_refinement_manifest_rejects_invalid_checkpoint_and_config(tmp_path):
    manifest=dict(version=1,engine='seeded-unet3d',config=asdict(RefinementConfig()),checkpoint='best.weights.h5')
    (tmp_path/'refinement-model.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='checkpoint'):read_model(tmp_path)
    (tmp_path/'best.weights.h5').write_bytes(b'placeholder')
    assert read_model(tmp_path)['engine']=='seeded-unet3d'
    with pytest.raises(ValueError,match='finite'):replace(RefinementConfig(),voxel_um=float('nan')).validate()


def test_finer_cache_application_preserves_saved_geometry_without_tensorflow(tmp_path,monkeypatch):
    import fish_viewer.refinement.inference as worker
    shape=(24,)*3;labels=np.zeros(shape,np.uint32);labels[8:16,8:16,8:16]=1
    image=np.zeros(shape,np.float32);np.save(tmp_path/'image.npy',image)
    config=asdict(RefinementConfig());model=tmp_path/'model';model.mkdir()
    (model/'refinement-model.json').write_text(json.dumps(dict(version=1,engine='seeded-unet3d',
        checkpoint='best.weights.h5',config=config)))
    (model/'best.weights.h5').write_bytes(b'placeholder')
    metadata=dict(version=1,image='scan.ims',time=0,channel=0,channel_scores=[],fingerprint='same',
                  spacing_zyx=[.15]*3,origin_xyz=[.075]*3,config=dict(rays=64,use_shape_filter=False))
    prediction=tmp_path/'cells.npz';np.savez_compressed(prediction,labels=labels,metadata=np.array(json.dumps(metadata)))
    source=tmp_path/'cells.json';source.write_text(json.dumps(dict(status='complete',files=[dict(image='scan.ims',predictions=str(prediction))])))
    audit=tmp_path/'audit.json';audit.write_text(json.dumps(dict(status='complete',cases=[dict(image='scan.ims',directory=str(tmp_path),
        audit=dict(fingerprint='same',channel=0,spacing_zyx=[.15]*3,origin_xyz=[.075]*3))])))
    monkeypatch.setattr(worker,'runtime',lambda *a:(None,None,None,{}))
    monkeypatch.setattr(worker,'network',lambda *a:SimpleNamespace(load_weights=lambda p:None))
    monkeypatch.setattr(worker,'refine',lambda labels,*a:(labels,[dict(cell_id=1,accepted=False)]))
    report=worker.apply(source,model,tmp_path/'output',audit=audit)
    assert report['status']=='complete',report.get('error')
    with np.load(report['files'][0]['predictions'],allow_pickle=False) as archive:
        saved=json.loads(str(archive['metadata']))
        assert np.array_equal(archive['labels'],labels)
        assert saved['spacing_zyx']==[.15]*3 and saved['refinement']['cells'][0]['cell_id']==1


def test_excel_remains_two_lists_with_refinement_metadata(tmp_path):
    from openpyxl import load_workbook
    from fish_viewer.workflow.excel import write_excel
    report=dict(points=[],counts=[],cell_volumes=[dict(cell_id=1,volume_um3=10.)],
                boundary_refinement=dict(model='example',cells=[dict(cell_id=1,accepted=False,fallback_reason='Weak membrane')]))
    path=write_excel(tmp_path/'report.xlsx',report)
    book=load_workbook(path,read_only=True,data_only=True)
    assert book.sheetnames == ['Measurements']
    assert list(book['Measurements'].values)[1] == ('Program',1,10.,None,None,None,None)
    book.close()


@pytest.mark.gui
def test_boundary_training_live_epoch_and_graph_progress(qapp,tmp_path):
    from fish_viewer.refinement.gui import TrainingDialog
    dialog=TrainingDialog(output=str(tmp_path))
    try:
        dialog.handle_event(dict(kind='progress',phase='Training boundary network',epoch=12,epochs=60,
                                 current=42,total=859,loss=.15,dice=.91))
        assert '12 / 60' in dialog.epoch.text()
        assert dialog.progress.value()==42 and dialog.progress.maximum()==859
        dialog.handle_event(dict(kind='epoch',epoch=12,training=dict(loss=.15,dice=.91),
                                 validation=dict(loss=.19,dice=.89)))
        assert len(dialog.rows)==1 and len(dialog.axes[0].lines)==2
    finally:dialog.close();qapp.processEvents()


@pytest.mark.gui
def test_analysis_refinement_progress_has_six_stages(qapp,tmp_path):
    from PySide6 import QtCore
    from fish_viewer.analysis.app import AnalysisWindow
    settings=QtCore.QSettings(str(tmp_path/'settings.ini'),QtCore.QSettings.Format.IniFormat)
    window=AnalysisWindow(off_screen=True,settings=settings)
    try:
        window.scan_number=1;window.total_scans=1;window.active_image='example.ims'
        window.processing_progress(dict(stage=2,stages=6,stage_name='Refining cell boundaries',
                                         phase='Refining cell boundaries',current=5,total=10))
        assert '2 / 6' in window.phase.text()
        assert window.overall.value()==1250
        assert Path(window.boundary_model.text()).name=='contours-96'
    finally:window.close();qapp.processEvents()


def test_redrawn_proposals_never_fall_back_to_detector_masks():
    from fish_viewer.refinement.inference import redraw_proposals
    labels=np.zeros((32,)*3,np.uint32);labels[10:22,10:22,10:22]=7
    prepared=SimpleNamespace(image=np.zeros(labels.shape,np.float32),spacing=(.3,)*3,origin=(1.,2.,3.))
    config=replace(RefinementConfig(),patch_size=32)
    class EmptyModel:
        def __call__(self,x,training=False):return np.full((1,32,32,32,1),-20.)
    result,diagnostics=redraw_proposals(labels,prepared,EmptyModel(),config)
    assert not result.any()
    assert len(diagnostics)==1 and not diagnostics[0]['accepted']
    assert diagnostics[0]['proposal_id']==7
    assert np.allclose(diagnostics[0]['box_low_xyz'],[4.,5.,6.])
    assert np.array_equal(labels[10:22,10:22,10:22],np.full((12,)*3,7))


def test_redrawn_positive_network_output_on_blank_image_is_rejected():
    from fish_viewer.refinement.inference import redraw_proposals
    labels=np.zeros((40,)*3,np.uint32);labels[15:25,15:25,15:25]=1
    prepared=SimpleNamespace(image=np.zeros(labels.shape,np.float32),spacing=(.3,)*3,origin=(0.,)*3)
    config=replace(RefinementConfig(),patch_size=32)
    class HallucinatedCell:
        def __call__(self,x,training=False):
            r=np.linalg.norm(np.indices((32,)*3)-15.5,axis=0)
            return np.where(r<9,20.,-20.)[None,...,None]
    result,diagnostics=redraw_proposals(labels,prepared,HallucinatedCell(),config)
    assert not result.any()
    assert diagnostics[0]['rejection_reason']=='Insufficient visible membrane evidence'

from fish_viewer.refinement.training import batch,validation_boundaries
def test_surface_targets_follow_augmented_geometry_and_physical_spacing():
    size=32
    image=np.zeros((1,size,size,size),np.float32)
    mask=np.zeros_like(image);mask[:,8:24,8:24,8:24]=1
    arrays=dict(images=image,masks=mask,weights=np.ones_like(image),valid=np.ones_like(image))
    records=[dict(seed_patch=[15.5]*3,patch_spacing_um=.2)]
    config=replace(RefinementConfig(),patch_size=size,surface_loss_weight=.1,crop_scale_jitter=.1,crop_shift_um=.2)
    _,y=batch(arrays,records,[0],config,np.random.default_rng(2),augment=True)
    truth=y[0,...,0]>.5
    expected=(ndi.distance_transform_edt(~truth,sampling=.2)-ndi.distance_transform_edt(truth,sampling=.2))/2
    assert y.shape[-1]==4
    assert np.allclose(y[0,...,3],np.clip(expected,-1,1))

@pytest.mark.neural
def test_surface_loss_and_asymmetric_loss_ignore_unknown_and_penalize_missing_voxels():
    import tensorflow as tf
    from fish_viewer.refinement.network import loss
    truth=np.zeros((1,8,8,8,4),np.float32)
    truth[...,1]=1;truth[:,2:6,2:6,2:6,2]=1
    truth[:,2:6,2:6,2:6,0]=1;truth[...,3]=1;truth[...,3][truth[...,0]>0]=-1
    logits=tf.Variable(tf.zeros((1,8,8,8,1)))
    config=replace(RefinementConfig(),surface_loss_weight=.1,tversky_beta=.6)
    with tf.GradientTape() as tape:value,_=loss(tf,tf.constant(truth),logits,config)
    gradient=tape.gradient(value,logits).numpy()
    assert not gradient[:,:2].any()
    assert gradient[0,3,3,3,0]<0
    # Shift logits inside marked target: covering it should lower the objective.
    grown=np.zeros(logits.shape,np.float32);grown[:,2:6,2:6,2:6]=2
    assert float(loss(tf,truth,grown,config)[0])<float(value)

def test_validation_boundary_scores_empty_prediction_as_failure():
    truth=np.zeros((1,16,16,16,4),np.float32);truth[:,4:12,4:12,4:12,0]=1;truth[...,2]=1
    record=[dict(seed_patch=[8,8,8],patch_spacing_um=.2)]
    config=RefinementConfig()
    good=validation_boundaries(truth[...,0:1],truth,[0],record,config)[0]
    bad=validation_boundaries(np.zeros((1,16,16,16,1)),truth,[0],record,config)[0]
    assert good==(0.,0.,0.)
    assert bad[0]>1 and bad[1]==100

def test_invalid_new_loss_config_is_rejected():
    with pytest.raises(ValueError):replace(RefinementConfig(),tversky_beta=.9).validate()
    with pytest.raises(ValueError):replace(RefinementConfig(),surface_loss_weight=-.1).validate()


@pytest.mark.neural
@pytest.mark.parametrize('metric,surface_weight', [('loss',0.), ('boundary',0.), ('boundary',.1)])
def test_finetuning_scores_loaded_checkpoint_and_preserves_epoch_zero(tmp_path, monkeypatch, metric, surface_weight):
    import tensorflow as tf
    import fish_viewer.refinement.training as worker
    from fish_viewer.refinement.network import loss
    config = replace(RefinementConfig(), patch_size=32, filters=4, depth=2,
                     epochs=1, batch_size=1, checkpoint_metric=metric,
                     negative_fraction=0, microscopy_augmentation=False, surface_loss_weight=surface_weight)
    def tiny_network(tf, config):
        inputs = tf.keras.Input((config.patch_size,)*3+(2,))
        logits = tf.keras.layers.Conv3D(1, 1, kernel_initializer='zeros',
                                       bias_initializer='zeros')(inputs)
        return tf.keras.Model(inputs, logits)
    parent = tmp_path/'parent'; parent.mkdir()
    model = tiny_network(tf, config)
    # A confidently correct checkpoint cannot improve in boundary error. A zero
    # learning rate below also makes the loss selection retain exactly epoch 0.
    model.layers[-1].bias.assign([4.])
    model.save_weights(parent/'best.weights.h5')
    (parent/'refinement-model.json').write_text(json.dumps(dict(
        version=1, engine='seeded-unet3d', checkpoint='best.weights.h5', config=asdict(config))))
    shape = (2,32,32,32)
    patches = tmp_path/'patches'; patches.mkdir()
    arrays = dict(images=np.zeros(shape,np.float32), masks=np.ones(shape,np.uint8),
                  weights=np.ones(shape,np.float32), valid=np.ones(shape,np.uint8))
    for name, array in arrays.items(): np.save(patches/(name+'.npy'), array)
    records = [dict(index=i, split=split, seed_patch=[15.5]*3, patch_spacing_um=.2)
               for i,split in enumerate(('training','validation'))]
    audit = tmp_path/'audit.json'; audit.write_text('{}')
    monkeypatch.setattr(worker, 'cache_collection', lambda *a: dict(summary={},cases=[]))
    monkeypatch.setattr(worker, 'cache_patches', lambda *a: dict(records=records,policy='test'))
    monkeypatch.setattr(worker, 'runtime', lambda *a: (tf,None,None,dict(device='CPU')))
    monkeypatch.setattr(worker, 'network', tiny_network)
    monkeypatch.setattr(worker, 'charts', lambda *a: None)
    # Keep the initial checkpoint optimal for this regression, without changing
    # the validated configuration or the production optimizer implementation.
    adam = tf.keras.optimizers.Adam
    monkeypatch.setattr(tf.keras.optimizers, 'Adam', lambda *a,**kw: adam(0.,**kw))
    x,y = worker.batch(arrays,records,[1],config,np.random.default_rng(1))
    expected = float(loss(tf,y,model(x,training=False),config)[0])
    report = worker.train(audit,tmp_path/'cache',patches,tmp_path/'runs',config,
                          device='cpu',initial_model=parent)
    assert report['status']=='complete',report.get('error')
    assert report['initial_validation']['loss']==pytest.approx(expected)
    assert report['best_epoch']==0
    restored = tiny_network(tf,config)
    restored.load_weights(Path(report['model_directory'])/'best.weights.h5')
    np.testing.assert_allclose(restored(x,training=False),model(x,training=False))


def test_microscopy_augmentation_preserves_labels_seeds_and_validation_inputs():
    config=replace(RefinementConfig(),patch_size=32,microscopy_augmentation=True)
    image=np.full((1,32,32,32),.3,np.float32)
    mask=np.zeros_like(image);mask[:,8:24,8:24,8:24]=1
    arrays=dict(images=image,masks=mask,weights=np.ones_like(image),valid=np.ones_like(image))
    records=[dict(seed_patch=[15.5]*3,patch_spacing_um=.2)]
    raw=replace(config,microscopy_augmentation=False)
    x,y=batch(arrays,records,[0],config,np.random.default_rng(7),augment=True)
    control,truth=batch(arrays,records,[0],raw,np.random.default_rng(7),augment=True)
    np.testing.assert_array_equal(y,truth)
    np.testing.assert_array_equal(x[...,1],control[...,1])
    assert not np.array_equal(x[...,0],control[...,0])
    assert x.dtype==np.float32 and x.flags.c_contiguous
    assert np.isfinite(x).all() and 0 <= x.min() <= x.max() <= 1
    first=batch(arrays,records,[0],config,np.random.default_rng(7),augment=True)
    np.testing.assert_array_equal(x,first[0])
    for seed in (1,20):
        val,_=batch(arrays,records,[0],config,np.random.default_rng(seed))
        np.testing.assert_array_equal(val[...,0],image)


def test_photon_noise_variance_scales_with_intensity():
    config=replace(RefinementConfig(),illumination_strength=0,membrane_contrast_range=0,
                   psf_lateral_um=0,psf_axial_um=0,photon_count_min=100,photon_count_max=100)
    for intensity in (.2,.6):
        noisy=augment_membrane(np.full((40,)*3,intensity,np.float32),.2,config,np.random.default_rng(4))
        assert noisy.mean()==pytest.approx(intensity,abs=.002)
        assert noisy.var()==pytest.approx(intensity/100,rel=.05)


def test_psf_augmentation_blurs_more_along_physical_z():
    config=replace(RefinementConfig(),illumination_strength=0,membrane_contrast_range=0,
                   photon_count_min=1e6,photon_count_max=1e6)
    impulse=np.zeros((33,)*3,np.float32);impulse[16,16,16]=1
    blurred=augment_membrane(impulse,.1,config,np.random.default_rng(10))
    grid=np.indices(impulse.shape)-16
    variance=np.sum(blurred*grid**2,axis=(1,2,3))/blurred.sum()
    assert variance[0]>4*max(variance[1:])


@pytest.mark.parametrize('mode',['background','clahe'])
def test_preprocessing_removes_diffuse_background_and_keeps_blank_crops_blank(mode):
    config=replace(RefinementConfig(),preprocessing=mode)
    for value in (0.,.3):
        assert not preprocess_membrane(np.full((32,)*3,value,np.float32),.2,config).any()
    grid=np.indices((48,)*3);radius=np.linalg.norm((grid-23.5)*.2,axis=0)
    shell=np.exp(-((radius-2.)/.15)**2)
    image=(.15+.15*grid[2]/47+.5*shell).astype(np.float32)
    corrected=preprocess_membrane(image,.2,config)
    foreground=np.abs(radius-2.)<.15;background=radius>3.5
    before=image[background].mean()/image[foreground].mean()
    after=corrected[background].mean()/corrected[foreground].mean()
    assert after < before/2
    assert corrected.dtype==np.float32 and np.isfinite(corrected).all()
    assert 0 <= corrected.min() <= corrected.max() <= 1


@pytest.mark.parametrize('mode',['none','background','clahe'])
def test_validation_and_both_inference_branches_use_identical_model_inputs(mode):
    from fish_viewer.proposals.inference import contour_proposals
    config=replace(RefinementConfig(),patch_size=32,preprocessing=mode)
    grid=np.indices((40,)*3);radius=np.linalg.norm(grid-19.5,axis=0)
    image=(.1+.5*np.exp(-((radius-6)/1.2)**2)).astype(np.float32)
    labels=(radius<6).astype(np.uint32)
    box=ndi.find_objects(labels)[0];mask=labels[box]>0
    sample,truth,weight,valid,seed,step=sample_target(image,box,mask,(.2,)*3,config)
    arrays={name:value[None] for name,value in dict(images=sample,masks=truth,weights=weight,valid=valid).items()}
    record=[dict(seed_patch=seed,patch_spacing_um=step)]
    expected,_=batch(arrays,record,[0],config,np.random.default_rng(0))
    prepared=SimpleNamespace(image=image,spacing=(.2,)*3,origin=(0.,)*3)
    class Capture:
        def __call__(self,x,training=False):
            np.testing.assert_allclose(x,expected,atol=1e-6)
            return np.full((1,32,32,32,1),-20.)
    refine(labels,prepared,Capture(),config)
    # Proposal boxes span a physical width, whereas manual boxes are half-open
    # voxel ranges. Express the same crop geometry for this comparison.
    low=np.array([s.start for s in box]);high=np.array([s.stop for s in box])
    center=(low+high-1)/2
    from fish_viewer.refinement.data import seed_center
    seed_xyz=((seed_center(mask,(.2,)*3)+low)*.2)[::-1]
    proposal=dict(proposal_id=1,center_xyz=seed_xyz,
                  box_low_xyz=((center-(high-low)/2)*.2)[::-1],
                  box_high_xyz=((center+(high-low)/2)*.2)[::-1])
    contour_proposals(prepared,[proposal],Capture(),config)


def test_96_crop_improves_sampling_without_changing_physical_field_of_view():
    image=np.ones((40,)*3,np.float32)
    box=(slice(10,30),)*3;mask=np.ones((20,)*3,bool)
    small=sample_target(image,box,mask,(.2,)*3,replace(RefinementConfig(),patch_size=64))
    large=sample_target(image,box,mask,(.2,)*3,replace(RefinementConfig(),patch_size=96))
    assert large[0].shape==(96,)*3
    assert large[-1]*96==pytest.approx(small[-1]*64)
    np.testing.assert_allclose((large[-2]-47.5)*large[-1],(small[-2]-31.5)*small[-1])


def test_legacy_manifest_keeps_raw_64_cubed_inference(tmp_path):
    (tmp_path/'best.weights.h5').touch()
    (tmp_path/'refinement-model.json').write_text(json.dumps(dict(version=1,engine='seeded-unet3d',
        config=dict(patch_size=64),checkpoint='best.weights.h5')))
    manifest=read_model(tmp_path)
    config=RefinementConfig(**manifest['config'])
    assert config.patch_size==64 and config.preprocessing=='none'


def test_patch_cache_rejects_64_to_96_geometry_reuse(tmp_path):
    import hashlib
    from fish_viewer.refinement.data import cache_patches
    config=replace(RefinementConfig(),patch_size=64)
    keys=('patch_size','margin_um','boundary_band_um','boundary_weight')
    identity=dict(config={k:getattr(config,k) for k in keys},
                  cases_sha256=hashlib.sha256(b'[]').hexdigest())
    (tmp_path/'index.json').write_text(json.dumps(dict(identity=identity,records=[])))
    assert cache_patches(dict(cases=[]),tmp_path,config)['records']==[]
    with pytest.raises(ValueError,match='another collection/config'):
        cache_patches(dict(cases=[]),tmp_path,replace(RefinementConfig(),patch_size=96))


@pytest.mark.parametrize('parent_size,overrides,size,threshold,preprocessing',[
    (64,[],64,.61,'background'),
    (64,['--patch-size','96','--threshold','.45','--preprocessing','clahe'],96,.45,'clahe'),
    (96,[],96,.61,'background')])
def test_training_cli_inherits_inference_settings_for_validation(tmp_path,monkeypatch,parent_size,overrides,size,threshold,preprocessing):
    import sys
    import fish_viewer.refinement.__main__ as cli
    import fish_viewer.refinement.training as worker
    config=replace(RefinementConfig(),patch_size=parent_size,filters=4,depth=2,threshold=.61,preprocessing='background')
    (tmp_path/'refinement-model.json').write_text(json.dumps(dict(version=1,engine='seeded-unet3d',
        config=asdict(config),checkpoint='best.weights.h5')))
    (tmp_path/'best.weights.h5').write_bytes(b'placeholder')
    monkeypatch.setattr(cli,'restart_with_cuda',lambda *a:None)
    monkeypatch.setattr(cli.signal,'signal',lambda *a:None)
    captured=[]
    def train(*args):captured.append(args[4]);return dict(status='complete')
    monkeypatch.setattr(worker,'train',train)
    monkeypatch.setattr(sys,'argv',['refinement','--mode','train','--audit',str(tmp_path/'audit'),
        '--cache',str(tmp_path/'cache'),'--patches',str(tmp_path/'patches'),
        '--output',str(tmp_path/'runs'),'--initial-model',str(tmp_path),*overrides])
    assert cli.main()==0
    result=captured[0]
    assert result.patch_size==size and result.batch_size==1
    assert result.filters==4 and result.depth==2
    assert result.threshold==threshold and result.preprocessing==preprocessing


@pytest.mark.parametrize('settings',[
    dict(preprocessing='unknown'),dict(checkpoint_metric='test'),dict(illumination_strength=1),
    dict(membrane_contrast_range=1),dict(photon_count_min=1000,photon_count_max=100),
    dict(psf_lateral_um=.6,psf_axial_um=.2),dict(clahe_clip_limit=0),
    dict(clahe_fraction=1.1),dict(background_sigma_um=float('nan'))])
def test_invalid_microscopy_config_is_rejected(settings):
    with pytest.raises(ValueError):replace(RefinementConfig(),**settings).validate()


@pytest.mark.neural
def test_real_unet_transfers_64_weights_and_learns_at_96(tmp_path):
    import tensorflow as tf
    from fish_viewer.refinement.network import network,loss
    tf.keras.mixed_precision.set_global_policy('float32')
    config=replace(RefinementConfig(),patch_size=96,filters=4,depth=2)
    original=network(tf,replace(config,patch_size=64))
    original.save_weights(tmp_path/'best.weights.h5')
    model=network(tf,config);model.load_weights(tmp_path/'best.weights.h5')
    for a,b in zip(original.get_weights(),model.get_weights()):np.testing.assert_array_equal(a,b)
    grid=np.indices((96,)*3);radius=np.linalg.norm(grid-47.5,axis=0)
    x=np.stack([np.exp(-((radius-20)/2)**2),np.exp(-radius**2/50)],axis=-1)[None].astype(np.float32)
    y=np.stack([radius<20,np.ones_like(radius),np.ones_like(radius)],axis=-1)[None].astype(np.float32)
    before=model.layers[-1].get_weights()[0].copy()
    with tf.GradientTape() as tape:
        logits=model(x,training=True)
        value,_=loss(tf,y,logits,config)
    assert logits.shape==(1,96,96,96,1) and np.isfinite(float(value))
    gradients=tape.gradient(value,model.trainable_variables)
    assert all(np.isfinite(g.numpy()).all() for g in gradients)
    tf.keras.optimizers.Adam(config.learning_rate).apply_gradients(zip(gradients,model.trainable_variables))
    assert not np.array_equal(before,model.layers[-1].get_weights()[0])
