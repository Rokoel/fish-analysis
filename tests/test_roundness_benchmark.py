"""Check the equivalence used to sweep cell-only gates over frozen inference."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from benchmark_consensus_roundness import gated_result, validate_complete
from fish_viewer.spots.config import CenterConfig
from fish_viewer.spots.data import crop_inputs
from fish_viewer.signals.evaluation import suppress


def test_gate_preserves_surviving_measurements_and_consistent_cell_totals(tmp_path):
    labels = np.zeros((8, 8, 8), np.uint16)
    labels[1:3, 1:3, 1:3] = 1
    labels[4:7, 4:7, 4:7] = 2
    metadata = dict(cells=[dict(label=1), dict(label=2)],
                    roundness_filter=dict(scores=[dict(cell_id=1, sphericity=.78, accepted=True),
                                                   dict(cell_id=2, sphericity=.86, accepted=True)]))
    points = [dict(cell_id=1, channel=0, xyz=[1, 1, 1], distance_um=.3),
              dict(cell_id=2, channel=0, xyz=[5, 5, 5], distance_um=.8)]
    report = dict(points=points, counts=[dict(cell_id=1), dict(cell_id=2)],
                  cell_volumes=[dict(cell_id=1, volume_um3=8), dict(cell_id=2, volume_um3=27)],
                  signal_filter=dict(original_cell_count=2, kept_cell_count=2, removed_cell_count=0))
    before = deepcopy(report)
    result = gated_result(report, metadata, labels, labels, .85, tmp_path / 'gate')
    assert report == before
    assert result['points'] == points[1:]
    assert result['cell_volumes'] == report['cell_volumes'][1:]
    assert result['signal_filter'] == dict(original_cell_count=1, kept_cell_count=1, removed_cell_count=0)
    with np.load(result['unfiltered_cell_predictions'], allow_pickle=False) as archive:
        filtered, meta = archive['labels'], json.loads(str(archive['metadata']))
    assert set(np.unique(filtered)) == {0, 2}
    np.testing.assert_array_equal(filtered == 2, labels == 2)
    assert meta['cells'] == [dict(label=2)]
    assert [row['accepted'] for row in meta['roundness_filter']['scores']] == [False, True]
    empty = gated_result(report, metadata, labels, labels, .9, tmp_path / 'empty')
    assert not empty['points'] and not empty['cell_volumes']
    assert empty['signal_filter']['original_cell_count'] == empty['signal_filter']['kept_cell_count'] == 0


def test_removing_other_cells_does_not_change_cell_signal_inputs_or_suppression():
    labels = np.zeros((12, 12, 12), np.uint16)
    labels[3:7, 3:7, 3:7] = 1
    labels[7:9, 3:7, 3:7] = 2
    reduced = labels.copy()
    reduced[reduced == 2] = 0
    images = [np.arange(labels.size, dtype=float).reshape(labels.shape) / labels.size]
    metadata = dict(spacing_zyx=[.15] * 3, origin_xyz=[0] * 3)
    box = (slice(3, 7),) * 3
    before = crop_inputs(images, 0, labels, metadata, np.zeros(3), np.full(3, .15), 1, box, CenterConfig())
    after = crop_inputs(images, 0, reduced, metadata, np.zeros(3), np.full(3, .15), 1, box, CenterConfig())
    np.testing.assert_array_equal(before[0], after[0])
    assert before[1] == after[1]
    np.testing.assert_array_equal(before[2], after[2])
    np.testing.assert_array_equal(before[3], after[3])
    rows = [dict(cell_id=1, channel=0, xyz=[1., 1., 1.]),
            dict(cell_id=1, channel=0, xyz=[1.01, 1., 1.]),
            dict(cell_id=2, channel=0, xyz=[1., 1., 1.])]
    all_cells = suppress(rows, [.8, .7, .99], .5, .35)
    one_cell = suppress(rows[:2], [.8, .7], .5, .35)
    assert [row for row in all_cells if row['cell_id'] == 1] == one_cell


def test_roundness_report_rejects_missing_duplicate_or_wrong_channel_results():
    identity = dict(channels={'scan': 1}, thresholds=[.5, .75, .8, .82, .85])
    rows = [dict(image='scan', minimum_roundness=threshold, cell_channel=1)
            for threshold in identity['thresholds']]
    validate_complete(rows, identity)
    for invalid in (rows[:-1], rows + [rows[0]], rows[:-1] + [rows[0]]):
        with pytest.raises(ValueError, match='exactly once'):
            validate_complete(invalid, identity)
    wrong = deepcopy(rows)
    wrong[2]['cell_channel'] = 0
    with pytest.raises(ValueError, match='different membrane channel'):
        validate_complete(wrong, identity)


def test_future_roundness_publication_handles_no_manual_matches(tmp_path):
    from benchmark_consensus_roundness import publish, THRESHOLDS
    identity=dict(channels={'scan':1},thresholds=list(THRESHOLDS),scans=1)
    rows=[dict(image='scan',minimum_roundness=minimum,cell_channel=1,split='test',group='g',
        references=0,complete_references=0,cells_before_filter=0,final_cells=0,signals=0,
        matched_before_filter=0,matched_final=0,cut_references=[],volume_pairs=[],distance_pairs=[],
        matches_before_filter=[],matches_final=[]) for minimum in THRESHOLDS]
    report=publish(tmp_path,rows,identity,None)
    assert report['status']=='complete'
    assert all(summary['volume']['n']==summary['distance']['n']==0 for summary in report['summary'].values())
    assert 'All 1 supplied test scans' in report['policy']
    assert 'n/a' in (tmp_path/'README.md').read_text()
