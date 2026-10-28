from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
import json

from fish_viewer.neural.config import NeuralConfig
from fish_viewer.neural.precision import evidence_preset,membrane_evidence,evidence_passes
from fish_viewer.neural.data import add_synthetic_background


@pytest.mark.parametrize('support',[.5,.65])
def test_visible_closed_membrane_required_in_all_directions(support):
    config=replace(evidence_preset(NeuralConfig()),membrane_support=support)
    grid=np.indices((48,)*3);radius=np.linalg.norm((grid-23.5)*.3,axis=0)
    mask=radius<3;shell=np.exp(-((radius-3)/.25)**2)
    images=[shell,np.zeros_like(shell),np.ones_like(shell),shell*(grid[0]>24),np.roll(shell,7,axis=1)]
    accepted=[]
    for image in images:
        prepared=SimpleNamespace(image=image,spacing=(.3,)*3)
        evidence=membrane_evidence(mask,(slice(0,48),)*3,prepared,config)
        accepted.append(evidence_passes(evidence,config))
    assert accepted==[True,False,False,False,False]


def test_validation_selection_uses_negative_controls_and_bounded_recall():
    from fish_viewer.neural.selection import choose_candidate
    def row(index,recall,count,passed=True):
        return dict(index=index,controls=dict(passed=passed),eligible_reference_count=100,
            eligible_detection_rate=recall,predicted_count=count,config=dict(probability_threshold=.55,membrane_support=.5))
    # A failed negative control cannot be traded for reference recall.
    rows=[row(0,.95,5,False),row(1,.59,10),row(2,.65,50),row(3,.8,100)]
    assert choose_candidate(rows)['index']==2
    assert choose_candidate(rows[:2])['index']==1
    with pytest.raises(ValueError):choose_candidate(rows[:1])


def test_shape_limits_accept_visible_elongated_cell_without_accepting_empty_image():
    from fish_viewer.neural.engine import filter_instances
    grid=(np.indices((64,)*3)-31.5)*.3
    radius=np.sqrt((grid[0]/1.7)**2+grid[1]**2+grid[2]**2)
    labels=(radius<3).astype(np.uint32)
    prepared=SimpleNamespace(image=np.exp(-((radius-3)/.25)**2),spacing=(.3,)*3,origin=(0.,)*3)
    assert len(filter_instances(labels,prepared,replace(evidence_preset(NeuralConfig()),use_shape_filter=True,min_sphericity=.72,max_elongation=1.5,probability_threshold=.75,select_balanced_acceptance=False)).cells)==0
    assert len(filter_instances(labels,prepared,replace(evidence_preset(NeuralConfig()),use_shape_filter=True,min_sphericity=.65,max_elongation=1.8)).cells)==1
    prepared.image=np.zeros_like(prepared.image)
    assert len(filter_instances(labels,prepared,replace(evidence_preset(NeuralConfig()),use_shape_filter=True,min_sphericity=.65,max_elongation=1.8)).cells)==0


def test_acceptance_selection_evaluates_only_supplied_validation_cases(tmp_path,monkeypatch):
    from fish_viewer.neural import engine,controls
    from fish_viewer.neural.selection import select_acceptance
    from fish_viewer.pipeline.evaluation import evaluate
    mask=np.linalg.norm((np.indices((32,)*3)-15.5)*.3,axis=0)<2
    case=SimpleNamespace(entry=SimpleNamespace(image=tmp_path/'validation.ims'),references=[((slice(0,32),)*3,mask)],
        reference_cut=[False],names=['manual'],prepared=SimpleNamespace(spacing=(.3,)*3))
    visited=[]
    def predict(model,selected,config,*args):
        visited.append(selected.entry.image)
        labels=mask.astype(np.uint32)
        metrics=evaluate(labels,case.references,case.names,(.3,)*3)
        metrics.update(complete_cells=dict(metrics),cut_cells=evaluate(labels,[],[],(.3,)*3),proposal_limit_reached=False)
        return SimpleNamespace(labels=labels,rejected={}),metrics
    monkeypatch.setattr(engine,'predict',predict)
    monkeypatch.setattr(controls,'evaluate_controls',lambda *args:dict(passed=True,controls=[]))
    selected,report=select_acceptance(None,[case],replace(evidence_preset(NeuralConfig()),use_shape_filter=True,min_sphericity=.65,max_elongation=1.8),lambda e:None)
    assert visited==[case.entry.image]*4
    assert report['target_met'] and selected.probability_threshold==.65


def test_synthetic_controls_do_not_relabel_unknown_source_regions():
    image=np.ones((32,)*3,np.float32);labels=np.full(image.shape,-1,np.int16);labels[8:16,8:16,8:16]=1
    config=NeuralConfig(patch_z=32,patch_xy=32,synthetic_background_fraction=1)
    x,y,count=add_synthetic_background([image],[labels],config)
    assert count==1 and len(x)==len(y)==2
    assert np.array_equal(y[0],labels) and (y[1]==0).all()
    assert x[1].max()<.02 and np.isfinite(x[1]).all()


def test_visible_evidence_forbids_reference_only_calibration():
    config=replace(evidence_preset(NeuralConfig()),use_shape_filter=True,min_sphericity=.72,max_elongation=1.5,probability_threshold=.75,select_balanced_acceptance=False).validate()
    assert config.probability_threshold==.75 and config.exclude_border
    with pytest.raises(ValueError,match='Dice-only'):
        replace(config,calibrate_probability=True).validate()


def test_review_export_only_includes_confirmed_contours(tmp_path):
    from fish_viewer.neural.review import CandidateReview
    labels=np.zeros((12,)*3,np.uint32);labels[2:5,2:5,2:5]=1;labels[7:10,7:10,7:10]=2
    metadata=dict(image='example.ims',cells=[dict(label=1),dict(label=2)],metrics=dict(predicted_count=2))
    path=tmp_path/'contours.npz'
    np.savez_compressed(path,labels=labels,metadata=np.array(json.dumps(metadata)))
    review=CandidateReview(path);review.decide(1,'accepted');review.decide(2,'rejected')
    reloaded=CandidateReview(path)
    with np.load(reloaded.export_confirmed(),allow_pickle=False) as archive:
        assert set(np.unique(archive['labels']))=={0,1}
        saved=json.loads(str(archive['metadata']))
        assert 'metrics' not in saved and len(saved['cells'])==1
    assert np.array_equal(review.labels,labels)
    np.savez_compressed(path,labels=np.zeros_like(labels),metadata=np.array(json.dumps(metadata)))
    with pytest.raises(ValueError,match='different predictions'):CandidateReview(path)


@pytest.mark.gui
def test_precision_controls_and_phase_progress(qapp):
    from fish_viewer.app import MainWindow
    window=MainWindow(off_screen=True)
    try:
        window.show_neural();panel=window.neural_panel
        panel.use_evidence_preset()
        assert panel.config().precision_mode and not panel.config().calibrate_probability
        panel.handle_event(dict(kind='progress',phase='Training',current=27,total=200,epoch=12,epochs=120))
        assert panel.epoch_label.text()=='Epoch 12 / 120'
        assert panel.operation_progress.value()==27 and panel.operation_progress.maximum()==200
        panel.handle_event(dict(kind='progress',phase='Validation loss',current=8,total=279,epoch=12,epochs=120))
        assert panel.phase_label.text()=='Validation loss'
        panel.handle_event(dict(kind='progress',phase='Final prediction and evaluation',current=23,total=359,split='test',file='/data/example.ims'))
        panel.handle_event(dict(kind='tile',current=3,total=12,file='/data/example.ims'))
        assert 'test' in panel.phase_label.text() and '3 / 12' in panel.operation_label.text()
    finally:window.close();qapp.processEvents()


@pytest.mark.gui
def test_choosing_model_restores_its_acceptance_policy(qapp,tmp_path):
    from dataclasses import asdict
    from fish_viewer.app import MainWindow
    window=MainWindow(off_screen=True)
    try:
        window.show_neural();panel=window.neural_panel
        panel.use_evidence_preset()
        config=replace(evidence_preset(NeuralConfig()),use_shape_filter=True,min_sphericity=.65,max_elongation=1.8)
        (tmp_path/'fish-model.json').write_text(json.dumps(dict(engine='stardist3d',config=asdict(config))))
        panel.model_path.setText(str(tmp_path));panel.restore_chosen_model()
        assert panel.config().probability_threshold==.55
        assert panel.config().max_elongation==1.8
        panel.use_evidence_preset()
        assert not panel.config().use_shape_filter and not panel.config().calibrate_probability
    finally:window.close();qapp.processEvents()


@pytest.mark.gui
def test_review_dialog_slices_and_export(qapp,tmp_path):
    from fish_viewer.neural.review import ReviewDialog
    grid=np.indices((24,)*3);radius=np.linalg.norm(grid-12,axis=0)
    image=np.exp(-((radius-5)/.8)**2).astype(np.float32);labels=(radius<5).astype(np.uint32)
    image_path=tmp_path/'image.npy';np.save(image_path,image)
    path=tmp_path/'contours.npz'
    metadata=dict(image='example.ims',cells=[dict(label=1)],spacing_zyx=[.3,.3,.3])
    np.savez_compressed(path,labels=labels,metadata=np.array(json.dumps(metadata)))
    exported=[];dialog=ReviewDialog(path,image_path,on_export=exported.append)
    try:
        assert len(dialog.axes)==3 and dialog.sliders[0].maximum()>dialog.sliders[0].minimum()
        dialog.sliders[0].setValue(10);dialog.decide('accepted');dialog.export()
        assert exported[0].exists()
        assert dialog.data.review['decisions']['1']['decision']=='accepted'
    finally:dialog.close();qapp.processEvents()
