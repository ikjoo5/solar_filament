"""Bounded native U-Net geometry tuning using saved masks only.

All global settings are declared before GT is read. Scores are development
tuning results, not independent validation or leaderboard measurements.
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

from experiment_mask_postprocess import group_rows, overlap_stats, read_rows
from experiment_view_combinations import GT, ROOT, SIZE, VAL, encode_roi
from filament_eval import Filament, evaluate
from refine_masks_with_pixels import resolve_rois

INPUT = Path('/workspace/artifacts/a100_results_20261028_bedf3e21')
OUTPUT = Path('/workspace/artifacts/native_postprocess_candidates_20261028')


def plan():
    result = [dict(name=f't{p:g}_area{a}', rule='area', probability=p, area=a)
              for p in (.35, .5, .65)
              for a in (300, 400, 500, 600, 800, 1000, 1500, 2000, 3000)]
    result += [dict(name=f'round_e{e:g}_c{c:g}', rule='shape', probability=.5,
                    area=300, elongation=e, compactness=c)
               for e in (1.5, 2., 3.) for c in (.2, .4, .6)]
    result += [dict(name=f'blur_s{s:g}_t{t:g}', rule='blur', probability=.5,
                    area=300, sigma=s, threshold=t)
               for s in (.7, 1.2, 2.) for t in (.4, .5, .6)]
    assert len(result) == 45
    return result


def row_shape(row):
    rle = row.rle(SIZE, SIZE)
    x, y, w, h = map(int, mu.toBbox(rle))
    roi = mu.decode(rle)[y:y+h, x:x+w].astype(np.uint8)
    yy, xx = np.nonzero(roi)
    xy = np.stack([xx, yy], axis=-1).astype(float)
    xy -= xy.mean(axis=0)
    eig = np.linalg.eigvalsh(xy.T @ xy / len(xy))
    elongation = float(np.sqrt(max(eig[-1], .25) / max(eig[0], .25)))
    contours, _ = cv2.findContours(np.pad(roi, 1), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    compactness = float(4 * np.pi * row.area / max(perimeter**2, 1e-12))
    return elongation, compactness


def smooth(rows, setting):
    output = []
    for image, members in group_rows(rows).items():
        originals, regions, changed = [], [], []
        for row in members:
            rle = row.rle(SIZE, SIZE)
            x, y, w, h = map(int, mu.toBbox(rle))
            region = np.s_[max(0,y-9):min(SIZE,y+h+9), max(0,x-9):min(SIZE,x+w+9)]
            original = mu.decode(rle)[region].astype(bool)
            candidate = cv2.GaussianBlur(original.astype(np.float32), (0, 0),
                                         setting['sigma'], borderType=cv2.BORDER_CONSTANT) >= setting['threshold']
            ratio = candidate.sum() / original.sum()
            iou = (candidate & original).sum() / (candidate | original).sum()
            if not (.5 <= ratio <= 1.5 and iou >= .5):
                candidate = original.copy()
            originals.append(original); regions.append(region); changed.append(candidate)
        resolved = resolve_rois(changed, originals, regions, (SIZE, SIZE))
        for row, mask, region in zip(members, resolved, regions):
            assert mask.any()
            rle = encode_roi(mask, region[0].start, region[1].start)
            output.append(Filament(row.filament_id, image, image, rle['counts'].decode('ascii'), int(mu.area(rle))))
    return output


def main():
    cv2.setNumThreads(1)
    started = time.monotonic()
    grid = plan()
    selected = list(csv.DictReader(VAL.open()))
    groups = {r['image_id'] for r in selected}
    images = {Path(r['file_name']).stem for r in selected}
    inputs = {p: read_rows(INPUT/f'best_t{p:g}_a100.csv') for p in (.35,.5,.65)}
    assert all({r.image_id for r in rows} <= images for rows in inputs.values())
    for rows in inputs.values():
        assert overlap_stats(rows)['overlap_images'] == 0
    shape = {r.filament_id: row_shape(r) for r in inputs[.5] if r.area>=300}
    predictions = {}
    for setting in grid:
        rows = [r for r in inputs[setting['probability']] if r.area>=setting['area']]
        if setting['rule'] == 'shape':
            rows = [r for r in rows if not (shape[r.filament_id][0] < setting['elongation']
                                           and shape[r.filament_id][1] > setting['compactness'])]
        elif setting['rule'] == 'blur':
            rows = smooth(rows, setting)
        predictions[setting['name']] = rows
    # Only now bring annotations into scope for scoring fixed predictions.
    truth = read_rows(GT, gt=True, selected=groups)
    assert len(images)==91 and len(groups)==101 and len(truth)==708
    reports = []
    for setting in grid:
        rows = predictions[setting['name']]
        score = evaluate(truth, rows, SIZE, SIZE)
        overlap = overlap_stats(rows)
        assert overlap['overlap_images']==0
        assert score['tp_pairs'] == len(truth)-score['fn']
        reports.append(dict(**setting, **{k:score[k] for k in ('pq','sq','rq','tp_pairs','fp','fn')},
                            prediction_rows=len(rows), **overlap))
    best = max(reports,key=lambda r:r['pq'])
    OUTPUT.mkdir(exist_ok=True)
    path=OUTPUT/(best['name']+'.csv')
    with path.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['filament_id','segmentation_rle'])
        w.writerows((r.filament_id,r.counts) for r in predictions[best['name']])
    rescored=evaluate(truth,read_rows(path),SIZE,SIZE)
    assert abs(rescored['pq']-best['pq'])<1e-12
    baseline=next(r for r in reports if r['name']=='t0.5_area300')
    assert abs(baseline['pq']-.4299305777121099)<1e-12
    report=dict(scope='45 same-development fixed postprocess settings; no independent CV or LB score',
                source={str(INPUT/f'best_t{p:g}_a100.csv'):hashlib.sha256((INPUT/f'best_t{p:g}_a100.csv').read_bytes()).hexdigest() for p in inputs},
                physical_images=91,annotation_groups=101,gt_instances=708,baseline=baseline,
                settings=reports,best=best,saved_candidate=str(path),
                saved_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),score_recomputed_after_save=True,
                target_exceeded=best['pq']>.5,elapsed_seconds=time.monotonic()-started)
    (ROOT/'docs/evidence/native_postprocess_experiments.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(baseline=baseline,best=best,saved_candidate=str(path),elapsed_seconds=report['elapsed_seconds']),indent=2))


if __name__=='__main__':
    main()
