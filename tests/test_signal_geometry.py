from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fish_viewer.signals.config import SignalConfig
from fish_viewer.signals.data import (at_labels,native_labels,marker_points,infer_channels)
from fish_viewer.signals.evaluation import evaluate_points,summarize,select_threshold


def test_a_is_signal_and_duplicate_scenes_do_not_duplicate_truth(tmp_path):
    scene=tmp_path/'a.imx'
    scene.write_text('<root><bpMeasurementPoints><Point Name="A" Position="1 2 3"/>'
        '<Point Name="B" Position="4 5 6"/><Point Name="C" Position="7 8 9"/></bpMeasurementPoints></root>')
    points=marker_points([scene,scene])
    assert len(points)==1 and points[0]['xyz']==[1.,2.,3.]

def test_native_label_mapping_preserves_calibrated_voxel_centers():
    labels=np.zeros((4,4,4),np.uint32);labels[1,1,1]=3
    metadata=dict(spacing_zyx=[.3,.3,.3],origin_xyz=[.15,.15,.15])
    dataset=SimpleNamespace(spacing=(.15,.15,.3),center_origin=(.075,.075,.15),volumes=[np.zeros((4,8,8))])
    native=native_labels(labels,metadata,dataset)
    assert native[1,2:4,2:4].min()==3
    assert at_labels(labels,[[.45,.45,.45]],metadata['origin_xyz'],metadata['spacing_zyx']).tolist()==[3]

def test_channel_inference_marks_ambiguous_channels_unknown():
    grid=np.indices((24,)*3);spot=np.exp(-sum((grid[a]-12)**2 for a in range(3))/2).astype(np.float32)
    points=[dict(xyz=[1.8]*3)]
    rows=infer_channels([np.zeros_like(spot),spot,np.zeros_like(spot)],points,0,np.zeros(3),(.15,)*3,SignalConfig())
    assert rows[0]['channel']==1
    rows=infer_channels([np.zeros_like(spot),spot,spot.copy()],points,0,np.zeros(3),(.15,)*3,SignalConfig())
    assert rows[0]['channel'] is None

def test_unknown_cells_wrong_channels_and_other_cells_are_not_positive():
    truth=[dict(xyz=[1,1,1],channel=1,cell_id=3)]
    rows=[dict(xyz=[1,1,1],channel=1,cell_id=3),dict(xyz=[1,1,1],channel=2,cell_id=3),
        dict(xyz=[1,1,1],channel=1,cell_id=4),dict(xyz=[9,9,9],channel=1,cell_id=8)]
    metric = summarize([evaluate_points(rows[:3],truth,.6)])
    assert metric['tp']==1 and metric['fp']==2


def test_center_matching_is_one_to_one_and_reports_false_detections():
    refs=[dict(xyz=[0,0,0],channel=1,cell_id=1),dict(xyz=[3,0,0],channel=1,cell_id=1)]
    predictions=[dict(xyz=[.1,0,0],channel=1,cell_id=1),dict(xyz=[.2,0,0],channel=1,cell_id=1)]
    metric=summarize([evaluate_points(predictions,refs,.6)])
    assert metric['tp']==1 and metric['fp']==1 and metric['fn']==1
    assert metric['precision']==metric['recall']==.5

def test_threshold_selection_uses_precision_constraint():
    refs=[dict(xyz=[i*3,0,0],channel=1,cell_id=1) for i in range(5)]
    rows=refs+[dict(xyz=[20,0,0],channel=1,cell_id=1)]
    selected=select_threshold([dict(candidates=rows,truth=refs)],[np.array([.9]*5+[.2])],SignalConfig())
    assert selected['target_met'] and selected['selected']['metrics']['precision']==1
    assert selected['selected']['metrics']['recall']==1

def test_normalization_preserves_a_rare_bright_punctum():
    from fish_viewer.signals.data import normalized
    volume=np.ones((64,64,64),np.uint16);volume[32,32,32]=500
    result=normalized(volume)
    assert result[32,32,32]==1 and np.count_nonzero(result)==1

def test_signal_suppression_matches_physical_greedy_rule():
    from fish_viewer.signals.evaluation import suppress
    rng=np.random.default_rng(42)
    rows=[dict(xyz=xyz.tolist(),channel=int(rng.integers(2)),cell_id=int(rng.integers(3)))
          for xyz in rng.uniform(-1,1,(350,3))]
    rows.extend([dict(xyz=[0,0,0],channel=0,cell_id=1),
                 dict(xyz=[.25,0,0],channel=0,cell_id=1)])
    scores=np.r_[rng.choice([.1,.5,.9],350),1.,1.]
    for separation in (0,.25,.6):
        expected=[]
        for index in np.argsort(-scores,kind='stable'):
            if scores[index]<.5:continue
            row=rows[index]
            if any(old['channel']==row['channel'] and old['cell_id']==row['cell_id'] and
                   np.linalg.norm(np.asarray(old['xyz'])-row['xyz'])<separation for old in expected):continue
            expected.append(dict(row,score=float(scores[index])))
        assert suppress(rows,scores,.5,separation)==expected

def test_exact_signal_cutoff_matches_brute_force_one_to_one_scores():
    from dataclasses import replace
    from fish_viewer.signals.evaluation import evaluate_cases
    rng=np.random.default_rng(27)
    cases=[];probabilities=[]
    for _ in range(3):
        refs=[dict(xyz=xyz.tolist(),channel=int(rng.integers(2)),cell_id=1) for xyz in rng.normal(0,.4,(8,3))]
        rows=[dict(xyz=xyz.tolist(),channel=int(rng.integers(2)),cell_id=1) for xyz in rng.normal(0,.4,(35,3))]
        cases.append(dict(candidates=rows,truth=refs));probabilities.append(rng.choice([.151,.463,.733,.912,.973],35))
    for target in (.3,.7,.9):
        config=replace(SignalConfig(),target_precision=target,candidate_separation_um=.1,detection_separation_um=.2)
        table=[dict(threshold=float(t),metrics=evaluate_cases(cases,probabilities,float(t),config)) for t in np.unique(np.concatenate(probabilities))]
        good=[r for r in table if r['metrics']['predictions']>=5 and (r['metrics']['precision'] or 0)>=target]
        expected=max(good,key=lambda r:(r['metrics']['recall'],r['metrics']['precision'],r['threshold'])) if good else max(table,key=lambda r:(r['metrics']['f05'],r['metrics']['precision'] or 0,r['threshold']))
        actual=select_threshold(cases,probabilities,config)
        assert actual['target_met']==bool(good)
        assert actual['selected']==expected

@pytest.mark.gui
@pytest.mark.parametrize('path_style',['native','windows'])
def test_viewer_loads_current_cell_archives_with_signal_overlays(qapp,ims_file,tmp_path,path_style):
    from fish_viewer.app import MainWindow
    from fish_viewer.models import LibraryEntry
    from fish_viewer.reader import load_dataset
    from fish_viewer.signals.engine import image_fingerprint
    dataset=load_dataset(LibraryEntry(ims_file));labels=np.ones(dataset.volumes[0].shape,np.uint32)
    source_image=str(ims_file) if path_style=='native' else 'C:\\FISH\\'+ims_file.name
    metadata=dict(version=1,image=source_image,channel=0,channel_scores=[],fingerprint=image_fingerprint(dataset),
                  spacing_zyx=list(dataset.spacing[::-1]),origin_xyz=list(dataset.center_origin),time=0,
                  detector=dict(engine='cell-center-box3d'))
    cells=tmp_path/'cells.npz';np.savez_compressed(cells,labels=labels,metadata=np.array(json.dumps(metadata)))
    report=dict(version=1,engine='signal-cnn3d',mode='apply',status='complete',image=str(ims_file),
                fingerprint=metadata['fingerprint'],cell_predictions=str(cells),
                channels=[dict(name='green',color=[0,1,0])],
                points=[dict(signal_id=1,cell_id=1,channel=0,xyz=list(dataset.center_origin),score=.95)])
    path=tmp_path/'signals.json';path.write_text(json.dumps(report))
    window=MainWindow(off_screen=True)
    try:
        window.run_job=lambda function,result,error=None,progress=None:result(function(lambda *_:None))
        window.open_signal_prediction(path)
        assert any(s.source=='prediction' for s in window.dataset.scene.surfaces)
        assert any(m.source=='signal_prediction' for m in window.dataset.scene.measurements)
        assert not any('Cannot load pipeline contours' in w for w in window.dataset.scene.warnings)
    finally:window.close();qapp.processEvents()

@pytest.mark.gui
def test_signal_report_counts_and_viewer_overlays(qapp,ims_file,tmp_path):
    from fish_viewer.app import MainWindow
    from fish_viewer.reader import load_dataset
    from fish_viewer.models import LibraryEntry
    from fish_viewer.signals.overlays import read_signals
    window=MainWindow(off_screen=True)
    report=dict(version=1,engine='signal-cnn3d',mode='apply',status='complete',directory=str(tmp_path),image=str(ims_file),
        fingerprint='expected',cell_predictions='cells.npz',channels=[dict(name='green',color=[0,1,0])],
        counts=[dict(cell_id=2,channel=0,channel_name='green',count=1)],
        points=[dict(signal_id=1,cell_id=2,channel=0,xyz=[11,21,31],score=.95)])
    path=tmp_path/'report.json';path.write_text(json.dumps(report))
    try:
        with pytest.raises(ValueError,match='different image'):read_signals(path,'wrong')
        measurements,_=read_signals(path,'expected')
        assert measurements[0].source=='signal_prediction' and not measurements[0].edges
        dataset=load_dataset(LibraryEntry(ims_file));dataset.scene.measurements.extend(measurements)
        window.entry=LibraryEntry(ims_file);window.set_dataset(dataset)
        window.slice_slider.setValue(0)
        assert window.slice_view._overlays
        window.show_predicted_signals.setChecked(False)
        assert not measurements[0].visible
        window.show_predicted_signals.setChecked(True)
        assert measurements[0].visible
        window.show_signals();panel=window.signal_panel
        panel.display_report(report)
        assert panel.points.rowCount()==1 and panel.counts.rowCount()==1
        assert panel.counts.item(0,1).text()=='1' and panel.points.item(0,6).text()=='0.950'
        panel.update_progress(dict(phase='Training signal classifier',current=40,total=80,epoch=7,epochs=100,loss=.13))
        assert panel.epoch_label.text()=='Epoch 7 / 100' and panel.progress.value()==40
    finally:window.close();qapp.processEvents()
