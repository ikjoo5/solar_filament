from pathlib import Path
import sys

import numpy as np
from pycocotools import mask as mu
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import chase_independent_decode as decoder


def binary(top=0, left=0, height=10, width=10):
    result = np.zeros((2048, 2048), np.uint8)
    result[top:top+height, left:left+width] = 1
    return result


def encoded(mask):
    return mu.encode(np.asfortranarray(mask))


def test_native_diagonal_connectivity_exact_area_and_empty_union():
    # Two area50 rectangles join diagonally: eight-connectivity yields area100.
    mask = binary(10, 10, 5, 10) | binary(15, 20, 5, 10) | binary(100, 100, 9, 11)
    result = decoder.native_masks(encoded(mask))
    assert len(result) == 1
    assert int(mu.area(result[0])) == 100
    assert np.array_equal(mu.decode(result[0]), mask & (np.indices(mask.shape)[0] < 30))
    assert decoder.native_masks(encoded(np.zeros_like(mask))) == []
    with pytest.raises(ValueError): decoder.native_masks(encoded(mask), min_area=0)


def test_decoder_uses_supplied_union_without_input_mode_or_tta():
    supplied = encoded(binary(23, 41, 10, 10))
    assert decoder.native_masks(supplied)[0]['counts'] == supplied['counts']
    # No preprocessing parameter exists; the same union is decoded identically.
    with pytest.raises(TypeError): decoder.native_masks(supplied, input_mode='raw')


def test_yolo_confidence_priority_inclusive_threshold_and_nonempty_remainder():
    low, high = encoded(binary(10, 10, 20, 20)), encoded(binary(10, 20, 20, 20))
    one = encoded(binary(500, 500, 1, 1))
    result = decoder.yolo_masks(dict(instances=[
        dict(confidence=.2, rle=low), dict(confidence=.9, rle=high),
        dict(confidence=.2, rle=one), dict(confidence=.19999, rle=encoded(binary(700, 700)))
    ]))
    assert [int(mu.area(x)) for x in result] == [400, 200, 1]
    assert result[0]['counts'] == high['counts']
    decoder.validate_masks(result)


def test_yolo_ties_keep_original_order_and_fully_covered_predictions_disappear():
    first, second = encoded(binary(10, 10, 20, 20)), encoded(binary(10, 20, 20, 20))
    result = decoder.yolo_masks(dict(instances=[dict(confidence=.5, rle=x) for x in [first, second, first]]))
    assert len(result) == 2
    assert result[0]['counts'] == first['counts']
    assert [int(mu.area(x)) for x in result] == [400, 200]


def test_malformed_empty_and_overlapping_rles_rejected():
    with pytest.raises(ValueError): decoder.native_masks(dict(size=[10, 10], counts=b'00'))
    with pytest.raises(ValueError): decoder.native_masks(dict(size=[2048, 2048], counts=b'0'))
    with pytest.raises(ValueError): decoder.native_masks(dict(size=[2048, 2048], counts=b'P'))
    with pytest.raises(ValueError): decoder.yolo_masks(dict(instances=[dict(confidence=float('nan'), rle=b'0')]))
    with pytest.raises(ValueError, match='Empty'): decoder.validate_masks([encoded(np.zeros((2048, 2048), np.uint8))])
    mask = encoded(binary())
    with pytest.raises(ValueError, match='Overlapping'): decoder.validate_masks([mask, mask])


def test_score_strict_half_iou_group_reuse_and_global_sum():
    full = encoded(binary(10, 10, 20, 20)); half = encoded(binary(10, 10, 10, 20))
    groups = {'a-first': dict(image='first', masks=[full]),
              'b-first': dict(image='first', masks=[full]),
              'a-second': dict(image='second', masks=[full])}
    actual = decoder.score_groups(groups, dict(first=[half], second=[full]))
    assert (actual['tp_pairs'], actual['fp'], actual['fn']) == (1, 2, 2)
    assert actual['pq'] == pytest.approx(1/3)
    assert actual['sq'] == 1
    with pytest.raises(ValueError): decoder.score_groups(groups, dict(second=[full]))
    empty = decoder.score_groups(groups, dict(first=[], second=[]))
    assert empty['pq'] == 0 and empty['fn'] == 3
