import importlib.util
from pathlib import Path
import sys

import numpy as np
from pycocotools import mask as mu
import pytest

TOOLS = Path(__file__).resolve().parents[1]/'tools'
sys.path.insert(0, str(TOOLS))
import chase_postprocess_ablation as ablation
import submission_native_recipe as native
from submission_driver import validate_predictions


def rectangle(top, left, height=20, width=20):
    return native._encode_roi(np.ones((height, width), bool), top, left)


def union(values):
    if values:
        return mu.merge(values)
    return mu.encode(np.zeros((2048, 2048), dtype=np.uint8, order='F'))


def component(value, mean=.95):
    return dict(rle=value, area=int(mu.area(value)),
                stats=dict(foreground=dict(mean=mean)))


def final_fixture():
    first, second, third = rectangle(10, 10), rectangle(50, 50), rectangle(90, 90)
    admitted, lower = rectangle(130, 130), rectangle(170, 170)
    # Exact .4/.25 boundary objects are retained; third has .3/.0 and is removed.
    support = union([rectangle(10, 10, 8), rectangle(90, 90, 6), admitted, lower])
    views = dict(a=union([rectangle(50, 50, 5), admitted, lower]),
                 b=union([rectangle(50, 50, 5), admitted, lower]),
                 c=union([admitted, lower]), d=union([]))
    rich = dict(view_unions=views, tta=dict(unions=dict(foreground={'0.2': support}),
                components={'0.35': [component(admitted), component(lower, .85)]}))
    return [first, second, third], rich, admitted, lower


def test_parameterized_control_matches_frozen_and_strict_boundaries():
    before, rich, added, _ = final_fixture()
    expected, decisions = native._final_admission_pruning(before, rich)
    actual = ablation.final_variant(before, rich, native)
    assert ablation.canonical(actual) == ablation.canonical(expected)
    assert decisions == dict(final_removed=1, final_added=1)
    assert ablation.canonical(actual) == ablation.canonical(before[:2]+[added])


def test_each_ablation_changes_only_its_declared_rule():
    before, rich, added, lower = final_fixture()
    no_veto = ablation.final_variant(before, rich, native, veto=False)
    relaxed = ablation.final_variant(before, rich, native, coverage=.2, stable_coverage=.1)
    looser = ablation.final_variant(before, rich, native, admission_mean=.8)
    assert ablation.canonical(no_veto) == ablation.canonical(before+[added])
    assert ablation.canonical(relaxed) == ablation.canonical(before+[added])
    assert ablation.canonical(looser) == ablation.canonical(before[:2]+[added, lower])
    for values in (no_veto, relaxed, looser):
        validate_predictions(values)


def test_minimal_native_uses_union_including_small_objects_and_no_tta():
    large, small = rectangle(20, 20), rectangle(100, 100, 3, 3)
    rich = dict(unions=dict(foreground={'0.5': union([large, small])}),
                components={'0.5': [component(large)]},
                tta=dict(unions=dict(foreground={'0.5': union([])})))
    all_values = ablation.native_minimal(rich, native, 1)
    filtered = ablation.native_minimal(rich, native, 100)
    assert ablation.canonical(all_values) == ablation.canonical([large, small])
    assert ablation.canonical(filtered) == ablation.canonical([large])
    validate_predictions(all_values)


def test_minimal_yolo_uses_confidence_first_pixel_ownership():
    low, high = rectangle(10, 10), rectangle(10, 20)
    frame = dict(geometric=[dict(view='base', instances=[
        dict(confidence=.2, rle=low['counts']), dict(confidence=.8, rle=high['counts'])])])
    legal, raw = ablation.yolo_minimal(frame, native, .2)
    assert len(raw) == len(legal) == 2
    assert int(mu.area(legal[0])) == 400
    assert int(mu.area(legal[1])) == 200
    assert ablation.canonical([legal[0]]) == ablation.canonical([high])
    validate_predictions(legal)
    with pytest.raises(ValueError, match='겹칩니다'):
        validate_predictions(raw)
    selected, _ = ablation.yolo_minimal(frame, native, .5)
    assert ablation.canonical(selected) == ablation.canonical([high])


def test_gt_transition_counts_each_truth_once_despite_multiple_hits():
    truth = rectangle(10, 10)
    # Raw duplicate proposals illustrate why removed-candidate counts cannot be TP losses.
    assert ablation.transition([[truth]], [truth, truth], [truth])['gt_lost'] == 0
    lost = ablation.transition([[truth], [truth]], [truth, truth], [])
    assert lost['gt_lost'] == 2  # one per annotator group, not four candidate pairs
    coverage = ablation.candidate_coverage([[truth]], [truth, truth])
    assert coverage == dict(raw_candidate_count=2, gt_with_any_raw_candidate_iou_gt05=1)


def test_metric_uses_strict_iou_threshold_and_global_aggregation():
    truth = rectangle(0, 0, 20, 20)
    half = rectangle(0, 0, 10, 20)
    counts = ablation.counts_for([[truth]], [half])
    assert counts == dict(tp_pairs=0, fp=1, fn=1, matched_iou_sum=0.)
    perfect = ablation.counts_for([[truth]], [truth])
    assert ablation.aggregate([counts, perfect])['pq'] == .5


def test_date_groups_remain_together_and_crossfit_is_diagnostic():
    names = [f'202401{day:02d}{hour:02d}0000Ch.jpeg'
             for day in range(1, 11) for hour in (10, 12)]
    folds = ablation.fold_plan(names)
    assert set(folds.values()) == set(range(5))
    for day in range(1, 11):
        assert len({fold for name, fold in folds.items() if name.startswith(f'202401{day:02d}')}) == 1
    methods = {'baseline': {name: dict(tp_pairs=1, fp=0, fn=0, matched_iou_sum=.7)
                            for name in folds},
               'poor': {name: dict(tp_pairs=0, fp=1, fn=1, matched_iou_sum=0.)
                        for name in folds}}
    result = ablation.crossfit(methods, folds)
    assert result['pooled_held_out']['pq'] == pytest.approx(.7)
    assert all(row['selected_using_other_dates'] == 'baseline' for row in result['folds'])
    assert 'NOT five newly trained fold models' in result['scope']
