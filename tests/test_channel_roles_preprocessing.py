import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6 import QtCore

from fish_viewer.channel_roles import signal_indices
from fish_viewer.neural.preprocessing import membrane_connected
from fish_viewer.neural.config import NeuralConfig
from fish_viewer.pipeline.config import PipelineConfig
from fish_viewer.pipeline.segmentation import prepare
from fish_viewer.analysis.controller import AnalysisController,default_models,default_proposal_model,default_refinement_model


def test_role_selection_excludes_cell_and_respects_explicit_signal_subset():
    assert signal_indices(4,2,None)==[0,1,3]
    assert signal_indices(4,2,[0,2,3])==[0,3]
    for selection in ([],[2],[4],[1,1]):
        with pytest.raises(ValueError):signal_indices(4,2,selection)
    with pytest.raises(ValueError):NeuralConfig(cell_channel=-1).validate()


def test_forced_channel_overrides_ring_ranking(ims_file,monkeypatch):
    from fish_viewer.reader import load_dataset
    from fish_viewer.models import LibraryEntry
    import fish_viewer.pipeline.segmentation as segmentation
    dataset=load_dataset(LibraryEntry(ims_file))
    monkeypatch.setattr(segmentation,'ring_score',lambda *a,**k:dict(score=1.))
    result=prepare(dataset,PipelineConfig(cell_channel=1),background_correct=False)
    assert result.channel==1 and not result.warnings
    with pytest.raises(ValueError):prepare(dataset,PipelineConfig(cell_channel=99))


def test_preprocessing_is_finite_deterministic_and_keeps_blank_blank():
    rng=np.random.default_rng(77)
    image=rng.uniform(0,.05,(24,32,32)).astype(np.float32)
    image[:,12:16,6:26]+=.5
    result=membrane_connected(image,[.5,.3,.3])
    assert result.shape==image.shape and result.dtype==np.float32
    assert np.isfinite(result).all() and result.min()>=0 and result.max()<=1
    assert np.array_equal(result,membrane_connected(image,[.5,.3,.3]))
    assert not membrane_connected(np.full(image.shape,.2,np.float32),[.5,.3,.3]).any()
    assert not np.shares_memory(result,image)


def test_preprocessing_never_becomes_border_acceptance_evidence(monkeypatch):
    from fish_viewer.neural.precision import membrane_evidence
    raw=np.zeros((40,)*3,np.float32);boosted=np.ones(raw.shape,np.float32)
    z,y,x=np.indices(raw.shape);mask=(z-20)**2+(y-20)**2+(x-20)**2<10**2
    prepared=SimpleNamespace(image=boosted,evidence_image=raw,spacing=(.3,)*3)
    evidence=membrane_evidence(mask,(slice(0,40),)*3,prepared,NeuralConfig())
    assert evidence['rim_intensity']==0 and evidence['membrane_support']==0


def test_controller_forced_roles_reach_consensus_workers(tmp_path,monkeypatch):
    controller=AnalysisController()
    monkeypatch.setattr(controller,'_launch',lambda *args:None)
    image=tmp_path/'scan.ims';image.touch()
    controller.start(image,tmp_path/'runs',cell_channel=2,signal_channels=[0,1])
    request=json.loads((controller.directory/'cells-request.json').read_text())
    assert request['config']['cell_channel']==2
    assert controller.cell_channel_arguments==['--cell-channel','2']
    assert controller.signal_channel_arguments==['--signal-channels','0','1']
    assert controller.state['channel_roles']==dict(cell=2,signals=[0,1])


def test_stardist_request_uses_forced_cell_channel(tmp_path,monkeypatch):
    controller=AnalysisController();monkeypatch.setattr(controller,'_launch',lambda *a:None)
    image=tmp_path/'scan.ims';image.touch()
    controller.start(image,tmp_path/'runs',default_models()[0],default_models()[1],cell_channel=1,signal_channels=[0])
    request=json.loads((controller.directory/'cells-request.json').read_text())
    assert request['config']['cell_channel']==1


@pytest.mark.gui
def test_channel_roles_gui_persists_by_scan_and_separates_display(qapp,ims_file,tmp_path):
    from fish_viewer.analysis.app import AnalysisWindow
    from fish_viewer.reader import load_dataset
    from fish_viewer.models import LibraryEntry
    settings=QtCore.QSettings(str(tmp_path/'settings.ini'),QtCore.QSettings.Format.IniFormat)
    window=AnalysisWindow(settings=settings)
    try:
        window._source_image=str(ims_file.resolve())
        window.set_dataset(load_dataset(LibraryEntry(ims_file)))
        original=[c.color for c in window.dataset.metadata.channels]
        window.show_channels()
        window.cell_channel_choice.setCurrentIndex(window.cell_channel_choice.findData(1))
        assert not window.signal_channel_checks[1].isChecked()
        assert not window.signal_channel_checks[1].isEnabled()
        window.signal_channel_checks[0].setChecked(True)
        stored=json.loads(settings.value('channel_roles'))[str(ims_file.resolve())]
        assert stored['cell']==1 and stored['signals']==[0]
        assert original==[c.color for c in window.dataset.metadata.channels]
        window.channel_dialog.close();window.channel_dialog=None
    finally:window.close()


def test_stain_hints_follow_lam_or_lbr_instead_of_display_color():
    from fish_viewer.channel_hints import membrane_stain_channels
    channels=[SimpleNamespace(excitation_nm=488),SimpleNamespace(excitation_nm=543),SimpleNamespace(excitation_nm=633)]
    assert membrane_stain_channels('Lam633m_K9488.ims',channels)==[2]
    assert membrane_stain_channels('Lam488_HDAc633.ims',channels)==[0]
    assert membrane_stain_channels('Lam546_LBR647_FISH488.ims',channels)==[1,2]
    assert membrane_stain_channels('unknown_scan.ims',channels)==[]


def test_synthetic_negatives_use_the_same_preprocessing(monkeypatch):
    from dataclasses import replace
    from fish_viewer.neural.data import WithSyntheticBackground
    import fish_viewer.neural.preprocessing as preprocessing
    calls=[]
    monkeypatch.setattr(preprocessing,'membrane_connected',lambda image,spacing:(calls.append(spacing) or np.full(image.shape,.123,np.float32)))
    config=replace(NeuralConfig(),preprocessing='membrane-connected-v1',synthetic_background_fraction=1)
    images=[np.ones((32,)*3,np.float32)];labels=[np.ones((32,)*3,np.int16)]
    pool=WithSyntheticBackground(images,labels,config)
    image,label=pool.pair(1)
    assert calls==[(config.voxel_um,)*3]
    assert (image==.123).all() and not label.any()
    assert np.array_equal(pool.pair(0)[0],images[0])


def test_saved_preprocessing_recipe_is_required_and_checked():
    from fish_viewer.neural.preprocessing import RECIPE,validate_manifest
    validate_manifest(dict(config=dict(preprocessing='none')))
    manifest=dict(config=dict(preprocessing='membrane-connected-v1'),preprocessing_recipe=RECIPE.copy())
    validate_manifest(manifest)
    manifest['preprocessing_recipe']['closing_um']=9
    with pytest.raises(ValueError):validate_manifest(manifest)
    with pytest.raises(ValueError):validate_manifest(dict(config=dict(preprocessing='membrane-connected-v1')))




def test_stardist2d_preset_and_worker_preserve_channel_roles(qapp,tmp_path,monkeypatch):
    from fish_viewer.analysis.app import AnalysisWindow
    from fish_viewer.analysis.controller import AnalysisController
    settings=QtCore.QSettings(str(tmp_path/'slice.ini'),QtCore.QSettings.Format.IniFormat)
    window=AnalysisWindow(settings=settings)
    try:
        window.pipeline.setCurrentIndex(window.pipeline.findData('consensus'))
        window.pipeline.setCurrentIndex(window.pipeline.findData('stardist2d'))
        assert Path(window.cell_model.text()).name=='cells-stardist2d'
        assert Path(window.signal_model.text()).name=='signals-center-unet'
        assert not window.boundary_model.text()
        scan=tmp_path/'scan.ims';scan.touch()
        controller=AnalysisController();launched=[]
        monkeypatch.setattr(controller,'_launch',lambda index,args:launched.append((index,args)))
        controller.start(scan,tmp_path/'runs',window.cell_model.text(),window.signal_model.text(),
            cell_channel=1,signal_channels=[0],roundness=.82)
        args=launched[0][1]
        assert args[:2]==['-m','fish_viewer.neural.slice_inference']
        assert args[args.index('--cell-channel')+1]=='1'
        assert controller.state['cell_detector_engine']=='stardist2d-slice'
        assert controller.signal_channel_arguments==['--signal-channels','0']
        assert ('shape','Checking 3D cell roundness') in controller.stages
    finally:window.close()
