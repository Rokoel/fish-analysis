import json
import numpy as np
import pytest
from fish_viewer.analysis.shape import filter_labels,apply
from fish_viewer.pipeline.segmentation import mask_properties

def test_roundness_uses_physical_surface_and_does_not_change_accepted_cell():
    z,y,x=np.indices((48,48,96)); labels=np.zeros(z.shape,np.uint32)
    labels[(z-24)**2+(y-24)**2+(x-24)**2 < 9**2]=1
    labels[((z-24)/4)**2+((y-24)/4)**2+((x-68)/16)**2 < 1]=2
    a=mask_properties(labels==1,np.ones(3))['sphericity']
    b=mask_properties(labels==2,np.ones(3))['sphericity']
    assert a>b
    result, rows=filter_labels(labels,[1,1,1],(a+b)/2)
    assert np.array_equal(result==1, labels==1)
    assert not (result==2).any()
    assert rows[0]['accepted'] and not rows[1]['accepted']
    off,_=filter_labels(labels,[1,1,1],0)
    assert np.array_equal(off,labels)

def test_roundness_accounts_for_anisotropic_spacing():
    z,y,x=np.indices((40,)*3)
    labels=(((z-20)/4)**2+((y-20)/12)**2+((x-20)/12)**2<1).astype(np.uint32)
    physical=mask_properties(labels>0,np.array([3,1,1]))['sphericity']
    voxel=mask_properties(labels>0,np.ones(3))['sphericity']
    assert physical>voxel
    threshold=(physical+voxel)/2
    accepted,_=filter_labels(labels,[3,1,1],threshold)
    rejected,_=filter_labels(labels,[1,1,1],threshold)
    assert np.array_equal(accepted,labels) and not rejected.any()

def test_roundness_report_retains_calibration_and_detector_provenance(tmp_path):
    labels=np.zeros((12,)*3,np.uint32); labels[3:9,3:9,3:9]=7
    metadata=dict(cells=[dict(label=7)],spacing_zyx=[.3,.2,.1],origin_xyz=[10,20,30],
                  detector=dict(boxes_kept=1),fingerprint='same')
    path=tmp_path/'cells.npz';np.savez_compressed(path,labels=labels,metadata=np.array(json.dumps(metadata)))
    report=tmp_path/'report.json';report.write_text(json.dumps(dict(files=[dict(image='scan.ims',predictions=str(path))])))
    result=apply(report,tmp_path/'runs',0)
    assert result['status']=='complete'
    with np.load(result['files'][0]['predictions']) as f:
        after=json.loads(str(f['metadata']))
        assert np.array_equal(labels,f['labels'])
    assert after['origin_xyz']==metadata['origin_xyz']
    assert after['detector']==metadata['detector']
