#!/usr/bin/env python3
"""Twelve fixed foreground morphology comparisons on two consensus sources.

The source foreground is unioned per physical image, transformed using a global
rule, and re-instanced with 8-connectivity. No GT-specific pixel/instance choice
or parameter adaptation is performed.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from pycocotools import mask as mu

from experiment_confidence_fusion import encode_components
from experiment_mask_postprocess import read_rows
from filament_eval import Filament, evaluate


ROOT = Path(__file__).resolve().parents[1]
GT = Path('/workspace/artifacts/solar_filament_gt_v1/gt.csv')
VAL = Path('/workspace/attachments/bcb64220-b208-4cd7-9c3f-3f2be0f04167/val_split.csv')
SOURCES = [
    dict(name='omit_hflip_2of3_c0.20_a300', min_area=300,
         path=Path('/workspace/artifacts/view_combo_candidates_v1/omit_hflip_2votes_c0.20_a300.csv'),
         expected_pq=0.4541833848875917),
    dict(name='all4_3of4_c0.20_a200', min_area=200,
         path=Path('/workspace/artifacts/validation_results_20261027T120958Z_444da7f4/vote75_200_conf0.20.csv'),
         expected_pq=0.4535254054790343),
]
OPERATIONS = ('baseline', 'close3', 'close5', 'open3', 'holes25', 'close3_holes25')


def fill_small_holes(foreground, maximum=25):
    """Fill only enclosed 4-connected background components of limited area."""
    padded = np.pad(np.asarray(foreground, dtype=np.uint8), 1, constant_values=0)
    background = 1-padded
    count, labels, stats, _ = cv2.connectedComponentsWithStats(background, connectivity=4)
    external = int(labels[0, 0])
    fill = np.zeros(count, dtype=bool)
    for label in range(1, count):
        if label != external and stats[label, cv2.CC_STAT_AREA] <= maximum:
            fill[label] = True
    padded[np.take(fill, labels)] = 1
    return padded[1:-1, 1:-1]


def transform(foreground, operation):
    if operation not in OPERATIONS:
        raise ValueError(f'unknown morphology operation: {operation}')
    result = np.asarray(foreground, dtype=np.uint8)
    if operation.startswith('close'):
        size = 5 if operation == 'close5' else 3
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel,
                                 borderType=cv2.BORDER_CONSTANT, borderValue=0)
    elif operation == 'open3':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel,
                                 borderType=cv2.BORDER_CONSTANT, borderValue=0)
    if operation in ('holes25', 'close3_holes25'):
        result = fill_small_holes(result, maximum=25)
    return result


def fixture_checks():
    mask = np.zeros((40, 50), np.uint8)
    mask[5:35, 5:45] = 1
    mask[10, 10] = 0
    mask[20:29, 20:29] = 0
    mask[5:10, 15] = 0  # notch attached to outer background is not a hole.
    filled = fill_small_holes(mask)
    assert filled[10, 10] == 1 and not filled[20:29, 20:29].any()
    assert not filled[5:10, 15].any()
    assert int(filled.sum()-mask.sum()) == 1
    for op in OPERATIONS:
        changed = transform(mask, op)
        assert changed.shape == mask.shape and np.all((changed == 0) | (changed == 1))
    assert np.array_equal(transform(mask, 'baseline'), mask)


def main():
    started = time.monotonic()
    fixture_checks()
    with VAL.open(newline='') as handle:
        selection = list(csv.DictReader(handle))
    groups = {r['image_id'] for r in selection}
    image_ids = {Path(r['file_name']).stem for r in selection}
    truth = read_rows(GT, gt=True, selected=groups)
    assert len(truth) == 708 and len(groups) == 101 and len(image_ids) == 91
    reports, artifacts = [], {}
    for source in SOURCES:
        rows = read_rows(source['path'])
        if {r.image_id for r in rows} - image_ids:
            raise ValueError('source predictions contain an unknown image')
        members = defaultdict(list)
        for row in rows:
            members[row.image_id].append(row)
        predictions = {op: [] for op in OPERATIONS}
        pixel_changes = {op: 0 for op in OPERATIONS}
        for index, image in enumerate(sorted(image_ids), 1):
            current = members.get(image, [])
            if not current:
                continue
            # RLE union first, then exactly one dense decode per source/image.
            foreground = mu.decode(mu.merge([r.rle(2048, 2048) for r in current]))
            for operation in OPERATIONS:
                changed = transform(foreground, operation)
                pixel_changes[operation] += int(np.count_nonzero(changed != foreground))
                components = encode_components(changed, source['min_area'])
                predictions[operation].extend(
                    Filament(f'{image}_{i}', image, image, counts, area)
                    for i, (counts, area) in enumerate(components, 1))
            if index == 1 or index % 20 == 0:
                print(f'{source["name"]}: {index}/91', flush=True)
        for operation in OPERATIONS:
            predicted = predictions[operation]
            score = evaluate(truth, predicted, 2048, 2048)
            report = dict(name=source['name']+'_'+operation, source=source['name'],
                          operation=operation, minimum_area=source['min_area'],
                          source_sha256=hashlib.sha256(source['path'].read_bytes()).hexdigest(),
                          **{k: score[k] for k in ['pq', 'sq', 'rq', 'tp_pairs', 'fp', 'fn']},
                          prediction_rows=len(predicted), overlap_images=0,
                          disjoint_by_construction=True,
                          foreground_pixels_changed=pixel_changes[operation],
                          extra_pairs_beyond_unique_gt=score['tp_pairs']-(708-score['fn']))
            if operation == 'baseline' and abs(score['pq']-source['expected_pq']) > 1e-12:
                raise ValueError('source baseline did not reproduce')
            reports.append(report)
            artifacts[report['name']] = predicted
            print(report['name'], f'PQ={score["pq"]:.8f}', 'FP', score['fp'], 'FN', score['fn'], flush=True)
    assert len(reports) == 12
    best = max(reports, key=lambda r: r['pq'])
    saved = None
    improved = best['pq'] > SOURCES[0]['expected_pq']+1e-10
    if improved:
        folder = Path('/workspace/artifacts/consensus_morph_candidates_v1')
        folder.mkdir(exist_ok=False)
        path = folder/(best['name']+'.csv')
        with path.open('x', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['filament_id', 'segmentation_rle'])
            writer.writerows((r.filament_id, r.counts) for r in artifacts[best['name']])
        saved = str(path)
    output = dict(scope='Twelve fixed same-development morphology comparisons; not independent CV or leaderboard score',
                  physical_images=91, annotation_groups=101, gt_instances=708,
                  rules='Whole-image source union -> fixed morphology -> 8-connected components -> source minimum area. No GT-dependent rule selection.',
                  hole_policy='4-connected background, padded exterior protected, fill enclosed holes of area <=25 pixels only',
                  border_policy='OpenCV constant-zero border',
                  experiments=reports, best=best, improved_over_reference=improved,
                  target_exceeded=best['pq']>.5, saved_candidate_csv=saved,
                  fixture_checks='small-hole, large-hole, external-notch and binary/shape invariants passed',
                  elapsed_seconds=time.monotonic()-started)
    (ROOT/'docs/evidence/consensus_morphology_experiments.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(dict(best=best, saved_candidate_csv=saved, elapsed_seconds=output['elapsed_seconds']), indent=2))


if __name__ == '__main__':
    main()
