import json
from pathlib import Path
import pytest
from PySide6 import QtCore
from fish_viewer.analysis.controller import AnalysisController,default_models,default_proposal_model,default_refinement_model,migrate_bundled_defaults


def test_bundled_defaults_match_requested_pipeline():
    assert default_models()[0].name=='cells-sensitive'
    assert default_models()[1].name=='signals-center-unet'
    assert Path(default_proposal_model()).name=='proposals'
    assert Path(default_refinement_model()).name=='contours-96'
    assert Path(default_refinement_model('64')).name=='contours-64'
    assert set(json.loads((default_models()[0].parent/'origins.json').read_text())) == {'cells-sensitive','proposals','contours-96','contours-64','signals-center-unet'}


@pytest.mark.parametrize('selected',['conservative','hybrid','crop','signal-first','preprocessed','consensus','stardist2d'])
def test_retired_choices_migrate_to_consensus(tmp_path,selected):
    settings=QtCore.QSettings(str(tmp_path/'settings.ini'),QtCore.QSettings.Format.IniFormat)
    settings.setValue('pipeline',selected)
    migrate_bundled_defaults(settings)
    assert settings.value('pipeline')=='consensus'
    assert Path(settings.value('cell_model')).name=='cells-sensitive'
    assert settings.value('signal_model')==str(default_models()[1])
    assert settings.value('boundary_model')==default_refinement_model()
    migrate_bundled_defaults(settings)
    assert settings.value('pipeline')=='consensus'


@pytest.mark.parametrize('choice,expected',[('64','64'),('96','96'),('obsolete','96')])
def test_contour_preference_survives_settings_migration(tmp_path,choice,expected):
    settings=QtCore.QSettings(str(tmp_path/'contours.ini'),QtCore.QSettings.Format.IniFormat)
    settings.setValue('contour_model',choice)
    migrate_bundled_defaults(settings)
    migrate_bundled_defaults(settings)
    assert settings.value('contour_model')==expected
    assert settings.value('boundary_model')==default_refinement_model(expected)


def test_consensus_preserves_saved_detector_acceptance_and_refinement(tmp_path,monkeypatch):
    controller=AnalysisController();launches=[]
    monkeypatch.setattr(controller,'_launch',lambda stage,args:launches.append((stage,args)))
    image=tmp_path/'scan.ims';image.touch()
    controller.start(image,tmp_path/'runs',device='cpu')
    request=json.loads((controller.directory/'cells-request.json').read_text())
    from dataclasses import asdict
    from fish_viewer.neural.config import NeuralConfig
    saved=asdict(NeuralConfig.from_dict(json.loads((default_models()[0]/'fish-model.json').read_text())['config']))
    for key in ('probability_threshold','nms_threshold','membrane_support','use_shape_filter'):
        assert request['config'][key]==saved[key]
    assert [name for name,_ in controller.stages]==['cells','refine','consensus_cells','consensus','shape','signals','filter','measure','export']
    assert controller.state['refinement_model']==default_refinement_model()
    assert controller.state['minimum_roundness']==.82
    assert launches[0][0]==0


@pytest.mark.parametrize('pipeline',['hybrid','stardist2d'])
def test_retired_pipeline_cannot_start(tmp_path,pipeline):
    with pytest.raises(ValueError,match='Only both-detector'):
        AnalysisController().start(tmp_path/'scan.ims',tmp_path/'runs',pipeline=pipeline)
