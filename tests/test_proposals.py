from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import ndimage as ndi

from fish_viewer.proposals.config import ProposalConfig
from fish_viewer.proposals.data import sparse_target,batch
from fish_viewer.proposals.inference import contour_proposals,predict_proposals
from fish_viewer.refinement.config import RefinementConfig


def test_center_targets_ignore_unmarked_regions_and_include_irregular_cells():
    grid=np.indices((24,)*3)
    mask=np.linalg.norm(grid-11.5,axis=0)<8
    mask[9:15,9:15,17:]=False
    refs=[((slice(0,24),)*3,mask)]
    target=sparse_target(refs,[False],np.ones(3)*.3,grid,ProposalConfig())
    assert np.array_equal(target[...,1]>0,mask)
    assert target[...,0].max()==1
    assert (target[~mask,1]==0).all() and (target[~mask,8]==0).all()
    assert (target[...,8]>0).any()
    cut=sparse_target(refs,[True],np.ones(3)*.3,grid,ProposalConfig())
    assert not cut.any()


def test_box_augmentation_reflects_low_and_high_faces():
    class FlipZ:
        def __init__(self):self.n=0
        def random(self):self.n+=1;return 0 if self.n==1 else 1
        def uniform(self,a,b):return 1 if a==.8 else 0
        def normal(self,a,b,size):return np.zeros(size)
    x=np.zeros((1,8,8,8,1),np.float16);y=np.zeros((1,4,4,4,9),np.float16)
    y[0,0,0,0,2]=2;y[0,0,0,0,5]=5
    _,target=batch(x,y,[0],FlipZ(),augment=True)
    assert target[0,3,0,0,2]==5 and target[0,3,0,0,5]==2


def test_empty_crop_is_rejected_without_box_or_geometric_fallback():
    prepared=SimpleNamespace(image=np.zeros((32,)*3,np.float32),spacing=(.2,)*3,origin=(.1,)*3)
    proposal=dict(proposal_id=1,score=.9,center_xyz=[3.3]*3,box_low_xyz=[2.]*3,box_high_xyz=[4.6]*3)
    class Empty:
        def __call__(self,x,training=False):return np.full((1,32,32,32,1),-20.)
    labels,diagnostics=contour_proposals(prepared,[proposal],Empty(),replace(RefinementConfig(),patch_size=32))
    assert not labels.any() and not diagnostics[0]['accepted']
    assert 'predicted center' in diagnostics[0]['rejection_reason']


def test_point_boxes_have_calibrated_xyz_and_padded_crop_context():
    prepared=SimpleNamespace(image=np.zeros((64,)*3,np.float32),spacing=(.3,)*3,origin=(.15,)*3)
    class Point:
        def __call__(self,x,training=False):
            output=np.full((1,32,32,32,7),-20.,np.float32)
            output[...,1:]=.3
            output[0,16,16,16,0]=10
            return output
    proposals=predict_proposals(prepared,Point(),ProposalConfig())
    assert len(proposals)==1
    assert np.allclose(proposals[0]['center_xyz'],[9.75]*3)
    assert np.allclose(proposals[0]['box_low_xyz'],[6.75]*3)
    assert np.allclose(proposals[0]['box_high_xyz'],[12.75]*3)


def test_uncached_imaris_application_emits_serializable_progress(ims_file,tmp_path,monkeypatch):
    from dataclasses import asdict
    import json
    import fish_viewer.proposals.inference as worker
    detector=tmp_path/'detector';detector.mkdir();boundary=tmp_path/'boundary';boundary.mkdir()
    (detector/'proposal-model.json').write_text(json.dumps(dict(version=1,engine='cell-center-box3d',
        config=asdict(ProposalConfig()),checkpoint='best.weights.h5')))
    (boundary/'refinement-model.json').write_text(json.dumps(dict(version=1,engine='seeded-unet3d',
        config=asdict(RefinementConfig()),checkpoint='best.weights.h5')))
    for folder in (detector,boundary):(folder/'best.weights.h5').write_bytes(b'placeholder')
    class Empty:
        def load_weights(self,p):pass
        def __call__(self,x,training=False):return np.full((1,32,32,32,7),-20.,np.float32)
    monkeypatch.setattr(worker,'runtime',lambda *a:(None,None,None,{}))
    monkeypatch.setattr(worker,'network',lambda *a:Empty())
    monkeypatch.setattr(worker,'boundary_network',lambda *a:Empty())
    events=[]
    def emit(event):events.append(json.dumps(event,allow_nan=False))
    report=worker.apply(detector,boundary,tmp_path/'results',image=ims_file,emit=emit)
    assert report['status']=='complete',report.get('error')
    assert report['summary']['cells']==0 and report['files'][0]['channel']==0
    assert any('Preprocessing ready' in event for event in events)




@pytest.mark.neural
def test_unknown_voxels_have_zero_center_and_box_gradient():
    tf=pytest.importorskip('tensorflow')
    from fish_viewer.proposals.network import loss
    target=np.zeros((1,4,4,4,9),np.float32)
    target[0,1:3,1:3,1:3,1]=1
    target[0,2,2,2,0]=1;target[0,2,2,2,8]=1;target[0,2,2,2,2:8]=.4
    prediction=tf.Variable(tf.zeros((1,4,4,4,7)))
    with tf.GradientTape() as tape:value=loss(tf,target,prediction)[0]
    gradient=tape.gradient(value,prediction).numpy()
    assert not gradient[0,0].any()
    assert gradient[0,2,2,2].any() and np.isfinite(gradient).all()


@pytest.mark.gui
def test_proposal_training_graphs_and_batch_progress(qapp,tmp_path):
    from fish_viewer.refinement.gui import TrainingDialog
    dialog=TrainingDialog(output=str(tmp_path),kind='proposals')
    try:
        dialog.handle_event(dict(kind='progress',phase='Training cell centers and boxes',epoch=3,epochs=35,
                                current=10,total=374,loss=.8,box_mae_um=.4))
        assert '3 / 35' in dialog.epoch.text() and '0.400 µm' in dialog.epoch.text()
        dialog.handle_event(dict(kind='epoch',epoch=3,training=dict(loss=.8,box_mae_um=.4),
                                validation=dict(loss=.9,box_mae_um=.5)))
        assert len(dialog.axes[1].lines)==2
    finally:dialog.close();qapp.processEvents()
