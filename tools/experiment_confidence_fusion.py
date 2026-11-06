#!/usr/bin/env python3
"""Bounded confidence-score fusion from existing, original-coordinate RLEs.

Confidence maps are heuristics, not calibrated pixel probabilities. Twenty-six
settings are fixed before scoring, plus one existing binary-vote reference.
No image/GT-specific threshold selection or model inference is performed.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from pycocotools import mask as mu
from scipy import ndimage

from experiment_mask_postprocess import read_rows
from filament_eval import Filament, _check_compressed_counts, evaluate


ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path('/workspace/artifacts/validation_results_20261027T120958Z_444da7f4')
GT = Path('/workspace/artifacts/solar_filament_gt_v1/gt.csv')
VAL = Path('/workspace/attachments/bcb64220-b208-4cd7-9c3f-3f2be0f04167/val_split.csv')
VIEWS = ('base', 'scale', 'hflip', 'vflip')
SIZE = 2048
REFERENCE_PQ = 0.4535254054790343


def plan():
    result = []
    for threshold in (.10, .15, .20, .25, .30, .35):
        for area in (100, 200, 300):
            result.append(dict(kind='mean4', threshold=threshold, min_area=area,
                               name=f'mean4_t{threshold:.2f}_a{area}'))
    for threshold in (.15, .20, .25, .30):
        for area in (200, 300):
            result.append(dict(kind='top3mean', threshold=threshold, min_area=area,
                               name=f'top3mean_t{threshold:.2f}_a{area}'))
    assert len(result) == 26
    result.append(dict(kind='reference', threshold=.20, min_area=200,
                       name='reference_binary3of4_c0.20_a200'))
    return result


def encode_local(mask, y0, x0, shape=(SIZE, SIZE)):
    """Encode a local component into a full-image RLE without a full canvas."""
    # np.nonzero(mask.T) enumerates pixels in global column-major order.
    xx, yy = np.nonzero(mask.T)
    if not len(xx):
        raise ValueError('empty component')
    flat = (xx.astype(np.int64)+x0)*shape[0]+yy+y0
    breaks = np.flatnonzero(np.diff(flat) > 1)
    starts = flat[np.r_[0, breaks+1]]
    ends = flat[np.r_[breaks, len(flat)-1]]
    counts = [int(starts[0])]
    for index, (start, end) in enumerate(zip(starts, ends)):
        if index:
            counts.append(int(start-ends[index-1]-1))
        counts.append(int(end-start+1))
    tail = shape[0]*shape[1]-int(ends[-1])-1
    if tail:
        counts.append(tail)
    return mu.frPyObjects({'size': list(shape), 'counts': counts}, *shape)


def encode_components(foreground, minimum_area):
    labels, count = ndimage.label(foreground, structure=np.ones((3, 3), np.uint8))
    areas = np.bincount(labels.ravel())
    slices = ndimage.find_objects(labels)
    result = []
    for label in sorted((i for i in range(1, count+1) if areas[i] >= minimum_area),
                        key=lambda i: -int(areas[i])):
        ys, xs = slices[label-1]
        rle = encode_local(labels[ys, xs] == label, ys.start, xs.start)
        result.append((rle['counts'].decode('ascii'), int(areas[label])))
    return result


def encoder_checks():
    rng = np.random.default_rng(42)
    for shape, origin, local_shape in [((20, 23), (0, 0), (8, 9)),
                                       ((20, 23), (5, 11), (8, 9)),
                                       ((20, 23), (12, 14), (8, 9))]:
        for _ in range(5):
            local = (rng.random(local_shape) > .6).astype(np.uint8)
            local[0, 0] = 1
            full = np.zeros(shape, np.uint8)
            y, x = origin
            full[y:y+local_shape[0], x:x+local_shape[1]] = local
            actual = mu.decode(encode_local(local, y, x, shape))
            np.testing.assert_array_equal(actual, full)
    maps = np.array([.1, .2, .4, .9], np.float32)[:, None, None]
    np.testing.assert_allclose((maps.sum(0)-maps.min(0))/3, np.sort(maps, axis=0)[1:].mean(0))


def load_cache(path, expected_images):
    cache = defaultdict(dict)
    masks = capped = 0
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            image, view = row['image'], row['view']
            if image not in expected_images or view not in VIEWS or view in cache[image]:
                raise ValueError('unknown/duplicate cache image-view')
            if row['shape'] != [SIZE, SIZE] or row['coordinate_space'] != 'original':
                raise ValueError('cache masks are not in original 2048 coordinates')
            for instance in row['instances']:
                confidence = instance['confidence']
                if not np.isfinite(confidence) or not .05 <= confidence <= 1:
                    raise ValueError('invalid actual confidence')
                if _check_compressed_counts(instance['rle'], SIZE**2) <= 0:
                    raise ValueError('empty cached mask')
            cache[image][view] = row
            masks += len(row['instances'])
            capped += row.get('max_det_reached', False)
    if set(cache) != expected_images or any(set(rows) != set(VIEWS) for rows in cache.values()):
        raise ValueError('missing image/view in raw cache')
    return cache, masks, capped


def confidence_maps(rows):
    maps = np.zeros((4, SIZE, SIZE), np.float32)
    for index, view in enumerate(VIEWS):
        for instance in rows[view]['instances']:
            rle = {'size': [SIZE, SIZE], 'counts': instance['rle'].encode('ascii')}
            x, y, w, h = [int(v) for v in mu.toBbox(rle)]
            mask = mu.decode(rle)[y:y+h, x:x+w].astype(bool)
            roi = maps[index, y:y+h, x:x+w]
            np.maximum(roi, instance['confidence'], out=roi, where=mask)
    return maps


def main():
    started = time.monotonic()
    encoder_checks()
    with VAL.open(newline='') as handle:
        selection = list(csv.DictReader(handle))
    selected = {r['image_id'] for r in selection}
    images = {Path(r['file_name']).stem for r in selection}
    truth = read_rows(GT, gt=True, selected=selected)
    assert len(images) == 91 and len(selected) == 101 and len(truth) == 708
    cache, mask_count, capped = load_cache(RESULTS/'raw_views.jsonl', images)
    settings = plan()
    predictions = {s['name']: [] for s in settings}
    for index, image in enumerate(sorted(images), 1):
        maps = confidence_maps(cache[image])
        means = {'mean4': maps.mean(axis=0),
                 'top3mean': (maps.sum(axis=0)-maps.min(axis=0))/3}
        grouped = defaultdict(list)
        for setting in settings:
            grouped[setting['kind'], setting['threshold']].append(setting)
        for (kind, threshold), members in grouped.items():
            foreground = ((maps >= .20).sum(axis=0) >= 3) if kind == 'reference' else means[kind] >= threshold
            components = encode_components(foreground, min(s['min_area'] for s in members))
            for setting in members:
                kept = [(counts, area) for counts, area in components if area >= setting['min_area']]
                predictions[setting['name']].extend(
                    Filament(f'{image}_{j}', image, image, counts, area)
                    for j, (counts, area) in enumerate(kept, 1))
        if index == 1 or index % 10 == 0:
            print(f'confidence fusion {index}/{len(images)}', flush=True)
    reports = []
    for setting in settings:
        result = evaluate(truth, predictions[setting['name']], SIZE, SIZE)
        reports.append({**setting, **{k: result[k] for k in ['pq','sq','rq','tp_pairs','fp','fn']},
                        'prediction_rows': len(predictions[setting['name']]),
                        'extra_pairs_beyond_unique_gt': result['tp_pairs']-(708-result['fn']),
                        'overlap_images': 0, 'disjoint_by_construction': True})
    reference = next(r for r in reports if r['kind'] == 'reference')
    if abs(reference['pq']-REFERENCE_PQ) > 1e-12:
        raise ValueError(f'Binary reference did not reproduce: {reference["pq"]}')
    best = max(reports, key=lambda r: r['pq'])
    saved = None
    # Ignore float64 summation noise relative to the reference reduction order.
    if best['kind'] != 'reference' and best['pq'] > REFERENCE_PQ+1e-10:
        out = Path('/workspace/artifacts/confidence_fusion_candidates_v1')
        out.mkdir(exist_ok=False)
        saved = str(out/f'{best["name"]}.csv')
        with Path(saved).open('x', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['filament_id','segmentation_rle'])
            writer.writerows((r.filament_id,r.counts) for r in predictions[best['name']])
    output = dict(scope='Fixed-rule same-development-set confidence fusion, not independent CV or leaderboard score',
                  raw_cache_sha256=hashlib.sha256((RESULTS/'raw_views.jsonl').read_bytes()).hexdigest(),
                  physical_images=91, annotation_groups=101, gt_instances=708,
                  raw_records=364, raw_masks=mask_count, capped_records=capped,
                  new_settings=26, reference_settings=1, dtype='float32',
                  rule='Each view uses per-pixel maximum actual instance confidence; equal mean4 or mean of top3 views. These are heuristic scores, not calibrated probabilities.',
                  coverage_validated=True, reference_reproduced=True,
                  local_encoder_reference_checks=15,
                  experiments=reports, best=best, saved_candidate_csv=saved,
                  best_new_setting=max((r for r in reports if r['kind'] != 'reference'), key=lambda r: r['pq']),
                  reference_comparison_tolerance=1e-10,
                  improved_over_reference=best['pq'] > REFERENCE_PQ+1e-10,
                  target_exceeded=best['pq']>.5, elapsed_seconds=time.monotonic()-started)
    (ROOT/'docs/evidence/confidence_fusion_experiments.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'best':best,'reference':reference,'saved_candidate_csv':saved,
                      'elapsed_seconds':output['elapsed_seconds']},indent=2))


if __name__ == '__main__':
    main()
