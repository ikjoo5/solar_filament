#!/usr/bin/env python3
"""Thirteen fixed tests of area filtering before versus after closing.

Small components are retained until after closing in the main hypotheses.
All rules act on raw 2/3 consensus foreground, independent of GT geometry.
"""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from pycocotools import mask as mu
from scipy import ndimage

from experiment_confidence_fusion import encode_components, load_cache
from experiment_consensus_morphology import transform
from experiment_mask_postprocess import read_rows
from filament_eval import Filament, evaluate


ROOT = Path(__file__).resolve().parents[1]
CACHE = Path('/workspace/artifacts/validation_results_20261027T120958Z_444da7f4/raw_views.jsonl')
GT = Path('/workspace/artifacts/solar_filament_gt_v1/gt.csv')
VAL = Path('/workspace/attachments/bcb64220-b208-4cd7-9c3f-3f2be0f04167/val_split.csv')
VIEWS = ('base', 'scale', 'vflip')
REFERENCE_PQ = .4566206036185416


def plan():
    settings = [dict(before=before, close=close, after=after,
                     name=f'before{before}_close{close}_after{after}')
                for before in (0, 100) for close in (3, 5) for after in (100, 200, 300)]
    settings.append(dict(before=300, close=5, after=300, name='reference_before300_close5_after300'))
    assert len(settings) == 13
    return settings


def remove_small(foreground, minimum):
    if minimum <= 1:
        return foreground
    labels, _ = ndimage.label(foreground, structure=np.ones((3,3), np.uint8))
    areas = np.bincount(labels.ravel())
    keep = areas >= minimum
    keep[0] = False
    return keep[labels]


def consensus(rows):
    votes = np.zeros((2048,2048), np.uint8)
    for view in VIEWS:
        rles = [{'size':[2048,2048], 'counts':instance['rle'].encode('ascii')}
                for instance in rows[view]['instances'] if instance['confidence'] >= .2]
        if rles:
            votes += mu.decode(mu.merge(rles))
    return votes >= 2


def fixture_checks():
    image = np.zeros((20,30), np.uint8)
    image[3:10,3:10] = 1
    image[15,20] = 1
    kept = remove_small(image, 10)
    assert kept.sum() == 49 and not kept[15,20]
    assert np.array_equal(remove_small(image,0),image)
    # Closing can connect a tiny fragment to a nearby larger object, which is
    # the predeclared motivation for postponing its area filter.
    image = np.zeros((30,40),np.uint8)
    image[10:20,8:18] = 1
    image[13:17,19:21] = 1
    early = transform(remove_small(image,10),'close5')
    late = transform(image,'close5')
    assert int(late.sum()) >= int(early.sum())


def main():
    started = time.monotonic()
    fixture_checks()
    with VAL.open(newline='') as handle:
        selected = list(csv.DictReader(handle))
    groups = {r['image_id'] for r in selected}
    images = {Path(r['file_name']).stem for r in selected}
    truth = read_rows(GT, gt=True, selected=groups)
    assert len(truth) == 708 and len(groups) == 101 and len(images) == 91
    cache, mask_count, capped = load_cache(CACHE, images)
    settings = plan()
    predictions = {s['name']:[] for s in settings}
    foreground_pixels = 0
    for index, image in enumerate(sorted(images),1):
        foreground = consensus(cache[image])
        foreground_pixels += int(foreground.sum())
        grouped = defaultdict(list)
        for setting in settings:
            grouped[setting['before'],setting['close']].append(setting)
        before_maps = {before:remove_small(foreground,before) for before in (0,100,300)}
        for (before,close), members in grouped.items():
            changed = transform(before_maps[before],f'close{close}')
            encoded = encode_components(changed,min(s['after'] for s in members))
            for setting in members:
                components = [(counts,area) for counts,area in encoded if area >= setting['after']]
                predictions[setting['name']].extend(
                    Filament(f'{image}_{j}',image,image,counts,area)
                    for j,(counts,area) in enumerate(components,1))
        if index == 1 or index % 20 == 0:
            print(f'morphology order {index}/91',flush=True)
    reports = []
    for setting in settings:
        score = evaluate(truth,predictions[setting['name']],2048,2048)
        reports.append({**setting, **{k:score[k] for k in ['pq','sq','rq','tp_pairs','fp','fn']},
                        'prediction_rows':len(predictions[setting['name']]),
                        'overlap_images':0,'disjoint_by_construction':True,
                        'extra_pairs_beyond_unique_gt':score['tp_pairs']-(708-score['fn'])})
    reference = next(r for r in reports if r['before']==300)
    assert abs(reference['pq']-REFERENCE_PQ) < 1e-12, reference
    best = max(reports,key=lambda r:r['pq'])
    improved = best['pq'] > REFERENCE_PQ+1e-10
    saved = None
    if improved:
        folder = Path('/workspace/artifacts/morph_order_candidates_v1')
        folder.mkdir(exist_ok=False)
        path = folder/f'{best["name"]}.csv'
        with path.open('x',newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['filament_id','segmentation_rle'])
            writer.writerows((r.filament_id,r.counts) for r in predictions[best['name']])
        saved = str(path)
    output = dict(scope='Thirteen fixed development morphology-order comparisons; not independent CV or leaderboard performance',
                  raw_cache_sha256=hashlib.sha256(CACHE.read_bytes()).hexdigest(),
                  physical_images=91,annotation_groups=101,gt_instances=708,
                  views=VIEWS,confidence_threshold=.2,required_views=2,
                  raw_cache_records=364,raw_cache_masks=mask_count,capped_records=capped,
                  operation_order='Optional 8CC minimum-area filter -> ellipse closing -> 8CC final minimum-area filter',
                  comparisons=13,changed_settings=12,reference_reproduced=True,
                  foreground_union_pixels_before_any_filter=foreground_pixels,
                  experiments=reports,best=best,improved_over_reference=improved,
                  target_exceeded=best['pq']>.5,saved_candidate_csv=saved,
                  fixture_checks='Early area filtering, identity/no-filter, closing-fragment fixture passed',
                  elapsed_seconds=time.monotonic()-started)
    (ROOT/'docs/evidence/morphology_order_experiments.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(best=best,reference=reference,saved_candidate_csv=saved,
                         elapsed_seconds=output['elapsed_seconds']),indent=2))


if __name__ == '__main__':
    main()
