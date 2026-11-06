"""Frozen native-model stages of the development-PQ 0.472593 submission recipe.

This module receives predictions for one image. It neither reads annotations
nor performs dataset-specific selection. Inputs and outputs use native COCO RLE.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from pycocotools import mask as mu
from scipy import ndimage

SIZE = 2048


def _rle(value):
    """Normalize JSON counts to bytes without changing the encoded mask."""
    counts = value['counts']
    return dict(size=list(value['size']), counts=counts.encode('ascii') if isinstance(counts, str) else counts)


def _area(rle):
    return int(mu.area(rle))


def _intersection(first, second):
    return _area(mu.merge([first, second], intersect=True))


def _iou(first, second):
    if not first or not second:
        return np.zeros((len(first), len(second)))
    return np.asarray(mu.iou(first, second, [0] * len(second)))


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


def _trim(candidate, covered):
    if covered is None or not _intersection(candidate, covered):
        return candidate
    x, y, w, h = map(int, mu.toBbox(candidate))
    local = mu.decode(candidate)[y:y+h, x:x+w].astype(bool)
    local &= ~mu.decode(covered)[y:y+h, x:x+w].astype(bool)
    return _encode_roi(local, y, x) if local.any() else None


def _weak_proposals(image, geometric_records, preprocessing_records, tiled_records):
    result = []
    for family, records in (('geometric', geometric_records),
                            ('preprocessing', preprocessing_records),
                            ('native_tiles', tiled_records)):
        for record in records:
            if record['image'] != image:
                raise ValueError('A raw prediction belongs to a different image')
            if record['shape'] != [SIZE, SIZE] or record['coordinate_space'] != 'original':
                raise ValueError('Expected native original-coordinate YOLO predictions')
            for proposal in record['instances']:
                if proposal['confidence'] < .6:
                    continue
                if family == 'native_tiles' and proposal['edge_clipped']:
                    continue
                result.append(dict(size=[SIZE, SIZE], counts=proposal['rle'].encode('ascii')
                    if isinstance(proposal['rle'], str) else proposal['rle']))
    return result


def _add_native_reference(originals, rich, weak):
    components = rich['components']['0.5']
    native = [_rle(c['rle']) for c in components]
    original_iou, weak_iou = _iou(native, originals), _iou(native, weak)
    covered = mu.merge(originals) if originals else None
    output, added = list(originals), 0
    for i, component in enumerate(components):
        area = component['area']
        if area < 300 or (originals and original_iou[i].max() > .1):
            continue
        if covered is not None and _intersection(native[i], covered) / area > .25:
            continue
        if not weak or not np.any(weak_iou[i] >= .3):
            continue
        candidate = _trim(native[i], covered)
        if candidate is not None and _area(candidate) >= 300:
            output.append(candidate)
            added += 1
    return output, added


def _bounding_region(boxes):
    left, top = min(b[0] for b in boxes), min(b[1] for b in boxes)
    right = max(b[0] + b[2] for b in boxes)
    bottom = max(b[1] + b[3] for b in boxes)
    return np.s_[top:bottom, left:right]


def _place(local, box, region):
    result = np.zeros((region[0].stop-region[0].start,
                       region[1].stop-region[1].start), bool)
    x, y, width, height = box
    top, left = y-region[0].start, x-region[1].start
    result[top:top+height, left:left+width] = local
    return result


def _resolve_rois(candidates, originals, regions):
    owner = np.full((SIZE, SIZE), -1, np.int32)
    for index, (mask, region) in enumerate(zip(originals, regions)):
        if np.any(mask & (owner[region] >= 0)):
            raise ValueError('Reference masks overlap')
        owner[region][mask] = index
    winner = np.full((SIZE, SIZE), -1, np.int32)
    best = np.full((SIZE, SIZE), np.inf, np.float64)
    for index, (candidate, original, region) in enumerate(zip(candidates, originals, regions)):
        allowed = candidate & ((owner[region] < 0) | (owner[region] == index))
        distance = ndimage.distance_transform_edt(~original)
        better = allowed & (distance < best[region])
        winner[region][better] = index
        best[region][better] = distance[better]
    return [candidate & (winner[region] == index)
            for index, (candidate, region) in enumerate(zip(candidates, regions))]


def _refine_boundaries(originals, rich):
    proposals = {'base': rich['components']['0.65'],
                 'tta': rich['tta']['components']['0.65']}
    peer_rles = {key: [_rle(c['rle']) for c in value] for key, value in proposals.items()}
    original_boxes = [tuple(map(int, mu.toBbox(rle))) for rle in originals]
    peer_boxes = {key: [tuple(map(int, mu.toBbox(rle))) for rle in value]
                  for key, value in peer_rles.items()}
    matrices = {key: _iou(originals, value) for key, value in peer_rles.items()}
    assignments = {key: matrix.argmax(axis=0) if matrix.shape[0] and matrix.shape[1]
                   else np.empty(0, int) for key, matrix in matrices.items()}
    candidates, local_originals, regions, decisions = [], [], [], Counter()
    for i, original in enumerate(originals):
        pair = []
        for model in ('base', 'tta'):
            valid = [j for j, component in enumerate(proposals[model])
                if assignments[model][j] == i and matrices[model][i, j] >= .3
                and component['stats']['foreground']['mean'] >= .8]
            pair.append(max(valid, key=lambda j: proposals[model][j]['stats']['foreground']['mean'])
                        if valid else None)
        boxes = [original_boxes[i]] + [peer_boxes[model][j]
            for model, j in zip(('base', 'tta'), pair) if j is not None]
        region = _bounding_region(boxes)
        x, y, w, h = original_boxes[i]
        local = _place(mu.decode(original)[y:y+h, x:x+w].astype(bool), original_boxes[i], region)
        if None in pair:
            candidate = local.copy()
            decisions['boundary_fallback_missing_peer'] += 1
        else:
            peers = []
            for model, j in zip(('base', 'tta'), pair):
                x, y, w, h = peer_boxes[model][j]
                mask = mu.decode(peer_rles[model][j])[y:y+h, x:x+w].astype(bool)
                peers.append(_place(mask, peer_boxes[model][j], region))
            candidate = (local.astype(np.uint8) + peers[0] + peers[1]) >= 2
            ratio = int(candidate.sum()) / int(local.sum())
            overlap = int((candidate & local).sum()) / int((candidate | local).sum())
            if not .5 <= ratio <= 1.5 or overlap < .85:
                candidate = local.copy()
                decisions['boundary_fallback_change_guard'] += 1
            else:
                decisions['boundary_changed' if np.any(candidate != local) else 'boundary_unchanged'] += 1
        candidates.append(candidate)
        local_originals.append(local)
        regions.append(region)
    resolved = _resolve_rois(candidates, local_originals, regions)
    output = []
    for original, mask, local, region in zip(originals, resolved, local_originals, regions):
        if not mask.any():
            raise ValueError('Boundary refinement unexpectedly removed an instance')
        if np.array_equal(mask, local):
            output.append(original)
            continue
        yy, xx = np.nonzero(mask)
        top, left = int(yy.min()), int(xx.min())
        output.append(_encode_roi(mask[top:int(yy.max())+1, left:int(xx.max())+1],
            region[0].start+top, region[1].start+left))
    return output, dict(decisions)


def _final_admission_pruning(originals, rich):
    covered = mu.merge(originals) if originals else None
    views = [_rle(value) for value in rich['view_unions'].values()]
    count = sum((mu.decode(value) for value in views), np.zeros((SIZE, SIZE), np.uint8))
    stable = mu.encode(np.asfortranarray((count >= 2).astype(np.uint8)))
    support = _rle(rich['tta']['unions']['foreground']['0.2'])
    components = rich['tta']['components']['0.35']
    native = [_rle(c['rle']) for c in components]
    matrix = _iou(native, originals)
    admitted = []
    for i, component in enumerate(components):
        area = component['area']
        if area < 300 or component['stats']['foreground']['mean'] < .9:
            continue
        if originals and matrix[i].max() > .1:
            continue
        if covered is not None and _intersection(native[i], covered)/area > .25:
            continue
        if sum(_intersection(native[i], view)/area >= .5 for view in views) < 3:
            continue
        candidate = _trim(native[i], covered)
        if candidate is not None and _area(candidate) >= 300:
            admitted.append(candidate)
    kept, removed = [], 0
    for original in originals:
        area = _area(original)
        if _intersection(original, support)/area < .4 and _intersection(original, stable)/area < .25:
            removed += 1
        else:
            kept.append(original)
    return kept + admitted, dict(final_removed=removed, final_added=len(admitted))


def build_native_submission(image, yolo_rles, geometric_records, preprocessing_records,
                            tiled_records, rich_record, return_stages=False):
    """Return final native RLEs for one image, with the highest-PQ frozen settings."""
    if Path(rich_record['file_name']).stem != image or rich_record['size'] != [SIZE, SIZE]:
        raise ValueError('Rich predictions belong to a different image or coordinate size')
    originals = [_rle(value) for value in yolo_rles]
    if any(value['size'] != [SIZE, SIZE] or _area(value) <= 0 for value in originals):
        raise ValueError('Invalid YOLO reference RLE')
    weak = _weak_proposals(image, geometric_records, preprocessing_records, tiled_records)
    native, added = _add_native_reference(originals, rich_record, weak)
    refined, decisions = _refine_boundaries(native, rich_record)
    final, final_decisions = _final_admission_pruning(refined, rich_record)
    if return_stages:
        return dict(native_reference=native, rich_refined=refined, final=final,
                    decisions=dict(native_reference_added=added, **decisions, **final_decisions))
    return final
