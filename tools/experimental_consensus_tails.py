"""Portable fixed R1 tail restoration. Requires only numpy/opencv/pycocotools.

Inference-only: no annotations, paths, folds, training, or test-set lookups.
Originals: list of canonical COCO compressed RLE dicts, size=[2048,2048],
counts bytes or str; all original masks must be mutually disjoint.
Geometric_views: list of exactly 4 dicts with distinct `view` strings,
`shape`=[2048,2048], `coordinate_space`="original", `instances` list.
Every instance has `confidence` in [0,1], `rle` as compressed COCO string.
Native: dict with native["unions"]["foreground"]["0.35"] and
native["tta"]["unions"]["foreground"]["0.35"] as compressed RLE dicts.
Native unions may be empty. All masks must share original image coordinates.

A static submission CSV alone does not supply geometric/native model outputs.
Do not substitute predictions from another image or model snapshot.
Function returns (list of RLE masks in original order, audit Counter).
Existing IDs/count and original pixels are preserved; output stays disjoint.
"""
from collections import Counter
import math
import cv2
import numpy as np
from pycocotools import mask as mu
SIZE=2048

RULES={'R1_two_views_native_union': {'minimum_views': 2, 'native_support': 'base_OR_tta_at_035'}}
COMMON={'raw_family': 'geometric_only', 'raw_minimum_confidence': 0.15, 'association_minimum_iou': 0.3, 'association_minimum_iou_margin_over_other_anchor': 0.1, 'raw_area_ratio_range': [0.5, 2.0], 'view_peer_selection': 'maximum anchor IoU, then confidence, then lexical RLE', 'one_peer_per_view': True, 'exact_peer_RLE_duplicates_count_as_one_vote': True, 'native_threshold': 0.35, 'preserve_every_original_pixel': True, 'additions': 'Unoccupied foreground supported by the required number of chosen geometric views and native support', 'connectivity': 'Keep only new union components that intersect an original mask pixel, 8-connectivity', 'tail_gate': 'At least 8 added pixels must lie outside the original mask elliptical radius-4 dilation', 'maximum_added_area_fraction': 0.25, 'over_area_policy': 'Rollback the entire anchor; never rank/select pixels by area budget', 'collision_policy': 'Rollback every changed anchor involved in any output-output pixel collision', 'count_preserved': True, 'masks_disjoint': True, 'no_new_objects': True, 'views_are_correlated_not_independent_models': True, 'novel_mechanism': 'Restore consensus missing tails beyond the prior C2 four-pixel band, without replacing an instance or changing global ranking', 'GT_used_for_prediction': False}

def checked_rle(value):
    """Bound compressed runs before calling the native COCO decoder."""
    if not isinstance(value, dict) or value.get('size') != [2048, 2048]:
        raise ValueError('Expected2048-square RLE')
    counts = value.get('counts')
    if not isinstance(counts, str) or not counts:
        raise ValueError('Expected nonempty compressed RLE string')
    previous, total, area, index, position = [], 0, 0, 0, 0
    while position < len(counts):
        number = shift = 0
        while True:
            if position >= len(counts):
                raise ValueError('Truncated compressed RLE')
            digit = ord(counts[position])-48
            position += 1
            if not 0 <= digit <= 63:
                raise ValueError('Invalid compressed RLE character')
            number |= (digit & 31) << shift
            shift += 5
            if shift > 65:
                raise ValueError('Compressed RLE integer overflow')
            if not digit & 32:
                if digit & 16:
                    number |= -1 << shift
                break
        if index > 2:
            number += previous[-2]
        if number < 0 or total+number > 2048**2:
            raise ValueError('Compressed RLE run outside image')
        total += number
        area += number if index % 2 else 0
        previous.append(number)
        previous = previous[-2:]
        index += 1
    if total != 2048**2 or area <= 0:
        raise ValueError('Invalid/empty instance RLE')
    result = dict(size=[2048, 2048], counts=counts.encode('ascii'))
    if int(mu.area(result)) != area:
        raise ValueError('Compressed RLE area mismatch')
    return result

def union_rle(value):
    if value == dict(size=[2048,2048], counts="PPPP4"):
        return dict(size=[2048,2048], counts=b"PPPP4")
    return checked_rle(value)

def _encode_roi(mask, top, left):
    counts, previous = [], 0
    for column in range(mask.shape[1]):
        transitions = np.flatnonzero(np.diff(np.r_[0, mask[:, column].astype(np.int8), 0]))
        for start, end in transitions.reshape(-1, 2):
            first = (left + column) * SIZE + top + int(start)
            last = (left + column) * SIZE + top + int(end)
            counts.extend([first - previous, last - first])
            previous = last
    counts.append(SIZE * SIZE - previous)
    return mu.frPyObjects(dict(size=[SIZE, SIZE], counts=counts), SIZE, SIZE)

def area(mask):
    return int(mu.area(mask))

def _restore(originals, raw, native):
    """No GT, file name, fold, or outcome is an input to the decision rule."""
    outputs = {method: list(originals) for method in RULES}
    counts = Counter(originals=len(originals))
    if not originals:
        return outputs, counts
    assert area(mu.merge(originals)) == sum(map(area, originals))
    peers = []
    views = raw['geometric']
    assert len(views) == 4 and len({view['view'] for view in views}) == 4
    for view in views:
        assert view['shape'] == [2048, 2048] and view['coordinate_space'] == 'original'
        for row in view['instances']:
            confidence = float(row['confidence'])
            assert math.isfinite(confidence) and 0 <= confidence <= 1
            if confidence >= COMMON['raw_minimum_confidence']:
                mask = checked_rle(dict(size=[2048, 2048], counts=row['rle']))
                peers.append(dict(view=view['view'], confidence=confidence, rle=mask, area=area(mask)))
    counts['raw_eligible_peers'] += len(peers)
    if not peers:
        return outputs, counts
    matrix = np.asarray(mu.iou(originals, [p['rle'] for p in peers], [0]*len(peers)))
    assignment = matrix.argmax(axis=0)
    occupied = mu.decode(mu.merge(originals)).astype(bool)
    native_base = mu.decode(union_rle(native['unions']['foreground']['0.35'])).astype(bool)
    native_tta = mu.decode(union_rle(native['tta']['unions']['foreground']['0.35'])).astype(bool)
    supports = dict(base_OR_tta_at_035=native_base | native_tta, base_AND_tta_at_035=native_base & native_tta)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    for index, original in enumerate(originals):
        original_area = area(original)
        by_view = {}
        for j, peer in enumerate(peers):
            if assignment[j] != index or matrix[index, j] < .30 or not .5 <= peer['area']/original_area <= 2.:
                continue
            second = np.delete(matrix[:, j], index).max(initial=0.)
            if matrix[index, j]-second < .10:
                continue
            key = (-matrix[index, j], -peer['confidence'], peer['rle']['counts'])
            if peer['view'] not in by_view or key < by_view[peer['view']][0]:
                by_view[peer['view']] = (key, peer)
        # Equal masks appearing in two views are still only one foreground vote.
        unique = {item[1]['rle']['counts']: item[1] for item in by_view.values()}
        selected = list(unique.values())
        if len(selected) < 2:
            counts['too_few_distinct_peers'] += 1
            continue
        counts['eligible_anchor_with_two_peers'] += 1
        boxes = [mu.toBbox(original)] + [mu.toBbox(peer['rle']) for peer in selected]
        left = max(0, int(min(box[0] for box in boxes))-5)
        top = max(0, int(min(box[1] for box in boxes))-5)
        right = min(2048, int(max(box[0]+box[2] for box in boxes))+5)
        bottom = min(2048, int(max(box[1]+box[3] for box in boxes))+5)
        region = np.s_[top:bottom, left:right]
        old = mu.decode(original)[region].astype(bool)
        votes = np.zeros(old.shape, dtype=np.uint8)
        for peer in selected:
            votes += mu.decode(peer['rle'])[region].astype(np.uint8)
        band = cv2.dilate(old.astype(np.uint8), kernel).astype(bool)
        free = ~occupied[region]
        for method, rule in RULES.items():
            if len(selected) < rule['minimum_views']:
                counts[method+'_insufficient_views'] += 1
                continue
            additions = (votes >= rule['minimum_views']) & supports[rule['native_support']][region] & free
            candidate = old | additions
            _, labels = cv2.connectedComponents(candidate.astype(np.uint8), connectivity=8)
            touched = np.unique(labels[old])
            candidate = np.isin(labels, touched[touched != 0])
            assert np.all(candidate[old])
            added = candidate & ~old
            if int((added & ~band).sum()) < 8:
                counts[method+'_no_consensus_tail'] += 1
                continue
            if added.sum() > .25*original_area:
                counts[method+'_area_rollback'] += 1
                continue
            outputs[method][index] = _encode_roi(candidate, top, left)
            counts[method+'_proposed_changes'] += 1
    for method, values in outputs.items():
        overlaps = np.asarray(mu.iou(values, values, [0]*len(values))) > 0
        np.fill_diagonal(overlaps, False)
        collisions = overlaps.any(axis=1)
        counts[method+'_collision_rollbacks'] += int(collisions.sum())
        outputs[method] = [old if collision else new for old, new, collision in zip(originals, values, collisions)]
        assert len(outputs[method]) == len(originals)
        assert area(mu.merge(outputs[method])) == sum(map(area, outputs[method]))
        for old, new in zip(originals, outputs[method]):
            assert area(mu.merge([old, new], intersect=True)) == area(old)
            assert area(new) <= 1.25*area(old)
            if old['counts'] != new['counts']:
                counts[method+'_actual_changed'] += 1
                counts[method+'_added_pixels'] += area(new)-area(old)
    return outputs, counts

def restore_consensus_tails(originals, geometric_views, native):
    """Apply frozen R1; metadata and GT are neither accepted nor consulted."""
    normalized = [checked_rle(dict(size=list(mask['size']), counts=mask['counts'].decode('ascii')
        if isinstance(mask['counts'], bytes) else mask['counts'])) for mask in originals]
    outputs, counters = _restore(normalized, {'geometric': geometric_views}, native)
    return outputs['R1_two_views_native_union'], counters
