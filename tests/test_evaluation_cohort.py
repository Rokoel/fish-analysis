import json

import pytest

def dataset(tmp_path):
    image=str(tmp_path/'scan.ims');case=dict(image=image,audit=dict(split='test',reference_cut=[False,True],channel=0))
    (tmp_path/'selection.json').write_text(json.dumps(dict(cases=[case])))
    (tmp_path/'identity.json').write_text('{}')
    for mode,volume,distance in [('consensus',12.,.9)]:
        folder=tmp_path/'001-scan'/mode;folder.mkdir(parents=True)
        row=dict(image=image,scan='scan.ims',mode=mode,split='test',group='g1',references=2,complete_references=1,
            cells_before_filter=2,final_cells=2,signals=2,matched_before_filter=2,matched_final=2,
            unscored_before_filter=0,unscored_final=0,
            matches_before_filter=[dict(reference_id=0),dict(reference_id=1)],matches_final=[dict(reference_id=0),dict(reference_id=1)],
            volume_pairs=[dict(reference_id=0,manual_um3=10.,predicted_um3=volume,cut=False),dict(reference_id=1,manual_um3=20.,predicted_um3=5.,cut=True)],
            distance_pairs=[dict(reference_id=0,channel=0,manual_xyz=[0,0,0],manual_um=1.,predicted_um=distance),
                            dict(reference_id=1,channel=0,manual_xyz=[1,0,0],manual_um=2.,predicted_um=3.)])
        (folder/'metrics.json').write_text(json.dumps(row));(folder/'pipeline-result.json').write_text('{"status":"complete"}')
    return tmp_path


def test_consensus_channel_choice_uses_stain_and_unique_metadata_hint():
    from fish_viewer.evaluation import channel_for

    case = dict(image='scan.ims', audit=dict(channel=2,
        channel_selection=dict(stain_candidates=[1, 2]),
        channel_scores=[dict(channel=0, metadata_hint=False), dict(channel=1, metadata_hint=True)]))
    assert channel_for(case)[0] == 2
    case['audit']['channel_selection']['stain_candidates'] = []
    assert channel_for(case)[0] == 1
    case['audit']['channel_scores'].append(dict(channel=2, metadata_hint=True))
    with pytest.raises(ValueError, match='Ambiguous'):
        channel_for(case)


def test_corrected_consensus_summary_excludes_cut_cells_and_zero_denominators(tmp_path):
    from fish_viewer.evaluation import summarize_complete as summarize

    root = dataset(tmp_path)
    row = json.loads((root/'001-scan/consensus/metrics.json').read_text())
    row['cut_references'] = [False, True]
    row['distance_pairs'][0]['manual_um'] = 0
    result = summarize([row])['test']
    assert result['scans'] == 1
    assert result['matched_complete_final'] == 1
    assert result['volume']['n'] == 1
    assert result['volume']['mape_percent'] == pytest.approx(20)
    assert result['distance']['n'] == 1
    assert result['distance']['mae'] == pytest.approx(.9)
    assert result['distance_relative']['n'] == 0
