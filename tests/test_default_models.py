import json
from pathlib import Path
import pytest
from PySide6 import QtCore
from fish_viewer.analysis.controller import AnalysisController,default_models,default_proposal_model,default_refinement_model,migrate_bundled_defaults


def test_bundled_defaults_match_requested_pipeline():
    assert default_models()[0].name=='cells-sensitive'
    assert default_models()[1].name=='signals-center-unet'
    assert Path(default_proposal_model()).name=='proposals'
    assert Path(default_refinement_model()).name=='refinement-bias-candidate'
    assert set(json.loads((default_models()[0].parent/'origins.json').read_text())) == {'cells-sensitive','proposals','refinement-bias-candidate','cells-stardist2d','signals-center-unet'}


@pytest.mark.parametrize('selected,expected',[('conservative','consensus'),('hybrid','consensus'),('crop','consensus'),('signal-first','consensus'),('preprocessed','consensus'),('consensus','consensus'),('stardist2d','stardist2d')])
def test_retired_choices_migrate_and_supported_choices_survive(tmp_path,selected,expected):
    settings=QtCore.QSettings(str(tmp_path/'settings.ini'),QtCore.QSettings.Format.IniFormat)
    settings.setValue('pipeline',selected)
    migrate_bundled_defaults(settings)
    assert settings.value('pipeline')==expected
    assert Path(settings.value('cell_model')).name==('cells-stardist2d' if expected=='stardist2d' else 'cells-sensitive')
    assert settings.value('signal_model')==str(default_models()[1])
    assert settings.value('boundary_model')==(default_refinement_model() if expected=='consensus' else '')
    migrate_bundled_defaults(settings)
    assert settings.value('pipeline')==expected


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


def test_retired_pipeline_cannot_start(tmp_path):
    with pytest.raises(ValueError,match='Only both-detector'):
        AnalysisController().start(tmp_path/'scan.ims',tmp_path/'runs',pipeline='hybrid')
