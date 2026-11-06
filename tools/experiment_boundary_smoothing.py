"""Twelve fixed, GT-free instance boundary comparisons on the current dev CSV.

No new model inference or pixel access. Keep one output per input instance,
limit shape change, and resolve expansion overlap by nearest original mask.
"""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from pycocotools import mask as mu

from experiment_mask_postprocess import group_rows, overlap_stats, read_rows
from experiment_view_combinations import ROOT, GT, VAL, SIZE, encode_roi
from filament_eval import Filament, evaluate
from refine_masks_with_pixels import resolve_rois


BASELINE = Path('/workspace/artifacts/gated_rescue_candidates_v1/c0.3_old0.3_novel0.1_area300_prep1.csv')
EXPECTED = .45911798265305065


def settings():
    result = [dict(name='baseline', operation='identity')]
    for sigma in (.7, 1.2, 2.):
        for threshold in (.4, .5, .6):
            result.append(dict(name=f'gaussian_s{sigma:g}_t{threshold:g}', operation='gaussian',
                               sigma=sigma, threshold=threshold))
    result += [dict(name='dilate1_cross', operation='dilate'),
               dict(name='erode1_cross', operation='erode')]
    assert len(result) == 12
    return result


def change(roi, setting):
    if setting['operation'] == 'gaussian':
        candidate = cv2.GaussianBlur(roi.astype(np.float32), (0, 0), setting['sigma'],
                                     borderType=cv2.BORDER_CONSTANT) >= setting['threshold']
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        operation = cv2.dilate if setting['operation'] == 'dilate' else cv2.erode
        candidate = operation(roi.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT,
                              borderValue=0).astype(bool)
    ratio = int(candidate.sum()) / int(roi.sum())
    union = int((candidate | roi).sum())
    iou = int((candidate & roi).sum()) / union if union else 0.
    if not .5 <= ratio <= 1.5 or iou < .5:
        return roi.copy(), 'fallback_excessive_change'
    return candidate, 'changed' if np.any(candidate != roi) else 'unchanged'




def main():
    cv2.setNumThreads(1)
    started = time.monotonic()
    with VAL.open(newline='') as handle:
        selected = list(csv.DictReader(handle))
    groups = {r['image_id'] for r in selected}
    images = {Path(r['file_name']).stem for r in selected}
    truth = read_rows(GT, gt=True, selected=groups)
    original_rows = read_rows(BASELINE)
    assert len(truth) == 708 and len(groups) == 101 and len(images) == 91
    assert len(original_rows) == 626 and {r.image_id for r in original_rows} <= images
    score = evaluate(truth, original_rows, SIZE, SIZE)
    assert abs(score['pq'] - EXPECTED) < 1e-12
    assert overlap_stats(original_rows)['overlap_images'] == 0
    grid = settings()
    predictions = {s['name']: [] for s in grid}
    statuses = {s['name']: Counter() for s in grid}
    grouped = group_rows(original_rows)
    for number, image in enumerate(sorted(images), 1):
        members = grouped.get(image, [])
        originals, regions = [], []
        for row in members:
            x, y, w, h = map(int, mu.toBbox(row.rle(SIZE, SIZE)))
            x1, y1 = max(0, x - 9), max(0, y - 9)
            x2, y2 = min(SIZE, x + w + 9), min(SIZE, y + h + 9)
            region = np.s_[y1:y2, x1:x2]
            regions.append(region)
            originals.append(mu.decode(row.rle(SIZE, SIZE))[region].astype(bool))
        for setting in grid:
            if setting['operation'] == 'identity':
                predictions[setting['name']].extend(members)
                continue
            candidates = []
            for original in originals:
                local, status = change(original, setting)
                statuses[setting['name']][status] += 1
                candidates.append(local)
            resolved = resolve_rois(candidates, originals, regions, (SIZE, SIZE))
            for row, mask, region in zip(members, resolved, regions):
                ys, xs = np.nonzero(mask)
                assert len(ys) > 0, 'Must preserve all instance rows'
                top, left = int(ys.min()), int(xs.min())
                rle = encode_roi(mask[top:int(ys.max())+1, left:int(xs.max())+1],
                                 region[0].start+top, region[1].start+left)
                predictions[setting['name']].append(Filament(row.filament_id, image, image,
                    rle['counts'].decode('ascii'), int(mu.area(rle))))
        if number % 10 == 0 or number == len(images):
            print('Boundary grid', number, '/', len(images), flush=True)
    reports = []
    for setting in grid:
        rows = predictions[setting['name']]
        assert len(rows) == len(original_rows)
        result = evaluate(truth, rows, SIZE, SIZE)
        overlap = overlap_stats(rows)
        assert overlap['overlap_images'] == 0
        reports.append(dict(**setting, **{k: result[k] for k in ('pq', 'sq', 'rq', 'tp_pairs', 'fp', 'fn')},
                            statuses=dict(statuses[setting['name']]), prediction_rows=len(rows),
                            extra_matching_pairs=result['tp_pairs']-(len(truth)-result['fn']), **overlap))
    best = max(reports, key=lambda r: r['pq'])
    saved = None
    if best['pq'] > score['pq']:
        directory = Path('/workspace/artifacts/boundary_smoothing_candidates_v2')
        directory.mkdir(exist_ok=False)
        path = directory / (best['name'] + '.csv')
        with path.open('x', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['filament_id', 'segmentation_rle'])
            writer.writerows((r.filament_id, r.counts) for r in predictions[best['name']])
        saved = str(path)
    report = dict(scope='12 fixed same-development geometry comparisons; not independent CV or LB',
                  rules='Fixed global parameters; one output per input; .5-1.5 area ratio and >=.5 original IoU; nearest original ownership for new pixels; GT only used for final scoring',
                  physical_images=len(images), annotation_groups=len(groups), gt_instances=len(truth),
                  input_sha256=hashlib.sha256(BASELINE.read_bytes()).hexdigest(), baseline_reproduced=True,
                  experiments=reports, best=best, saved_candidate=saved,
                  target_exceeded=best['pq'] > .5, elapsed_seconds=time.monotonic()-started)
    path = ROOT / 'docs/evidence/boundary_smoothing_experiments.json'
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(best=best, saved=saved, elapsed_seconds=report['elapsed_seconds']), indent=2))


if __name__ == '__main__':
    main()
