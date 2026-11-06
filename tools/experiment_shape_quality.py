"""Sixteen fixed GT-free shape filters on two cached development predictions.

Feature computation and keep/delete rules never inspect annotations. GT enters
only after all masks, features and fixed-setting predictions have been built.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from pycocotools import mask as mu
from scipy import ndimage

from experiment_gated_fn_rescue import BASELINE, NEW, NEW_VIEWS, OLD, OLD_VIEWS, load
from experiment_mask_postprocess import group_rows, overlap_stats, read_rows
from experiment_view_combinations import GT, ROOT, SIZE, VAL, encode_roi
from filament_eval import Filament, evaluate


CURRENT = Path('/workspace/artifacts/boundary_smoothing_candidates_v2/gaussian_s1.2_t0.6.csv')
CURRENT_EXPECTED = .4607591314345197
LOW_EXPECTED = .4500768531334606


def settings():
    """Declare all comparisons before loading GT or inspecting metric results."""
    grid = [dict(name='baseline', rule='identity')]
    grid += [dict(name=f'elong_lt{e:g}_compact_gt{c:g}', rule='round_compact',
                  elongation=e, compactness=c)
             for e in (1.5, 2., 3.) for c in (.2, .4, .6)]
    grid += [dict(name=f'interior_gt{d:g}_elong_lt2', rule='thick_round',
                  median_interior=d, elongation=2.) for d in (4., 6., 8.)]
    grid += [dict(name=f'diagonal_lt{d:g}_area_lt500', rule='small_box',
                  diagonal=d, area=500) for d in (20., 30., 40.)]
    assert len(grid) == 16 and len({s['name'] for s in grid}) == 16
    return grid


def shape_features(row):
    """Use only native ROI mask pixels, including holes in perimeter sums."""
    rle = row.rle(SIZE, SIZE)
    x, y, w, h = map(int, mu.toBbox(rle))
    mask = mu.decode(rle)[y:y+h, x:x+w].astype(bool)
    assert int(mask.sum()) == row.area and row.area > 0
    yy, xx = np.nonzero(mask)
    coords = np.column_stack((xx, yy)).astype(np.float64)
    centered = coords - coords.mean(axis=0)
    covariance = centered.T @ centered / len(coords)
    eigenvalues = np.linalg.eigvalsh(covariance)
    elongation = float(np.sqrt(max(eigenvalues[-1], .25) / max(eigenvalues[0], .25)))
    padded = np.pad(mask.astype(np.uint8), 1)
    contours, _ = cv2.findContours(padded, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(float(cv2.arcLength(contour, True)) for contour in contours)
    compactness = float(4 * np.pi * row.area / max(perimeter**2, 1e-12))
    interior = ndimage.distance_transform_edt(padded)[1:-1, 1:-1]
    result = dict(elongation=elongation, compactness=compactness,
                  median_interior=float(np.median(interior[mask])),
                  diagonal=float(np.hypot(w, h)), area=row.area)
    assert all(np.isfinite(value) for value in result.values())
    return result


def keep(feature, setting):
    if setting['rule'] == 'identity':
        return True
    if setting['rule'] == 'round_compact':
        return not (feature['elongation'] < setting['elongation']
                    and feature['compactness'] > setting['compactness'])
    if setting['rule'] == 'thick_round':
        return not (feature['median_interior'] > setting['median_interior']
                    and feature['elongation'] < setting['elongation'])
    if setting['rule'] == 'small_box':
        return not (feature['diagonal'] < setting['diagonal'] and feature['area'] < setting['area'])
    raise ValueError('Unknown shape filter')


def rebuild_low_confidence(images):
    """Reproduce exactly c.2/oldIoU.3/novelty.1/area300/prep1 rescue."""
    base_by_image = group_rows(read_rows(BASELINE))
    old, old_meta = load(OLD, images, OLD_VIEWS)
    new, new_meta = load(NEW, images, NEW_VIEWS)
    result = []
    added = 0
    for number, image in enumerate(sorted(images), 1):
        base = base_by_image.get(image, [])
        base_rles = [r.rle(SIZE, SIZE) for r in base]
        covered = (mu.decode(mu.merge(base_rles)).astype(bool) if base_rles
                   else np.zeros((SIZE, SIZE), bool))
        candidates = [dict(**proposal, view=view)
                      for view, rows in new[image].items() if view != 'limb'
                      for proposal in rows if proposal['confidence'] >= .2]
        candidates.sort(key=lambda p: -p['confidence'])
        cr = [p['native_rle'] for p in candidates]
        old_rles = [p['native_rle'] for rows in old[image].values()
                    for p in rows if p['confidence'] >= .1]
        old_iou = (np.asarray(mu.iou(cr, old_rles, [0]*len(old_rles))).max(axis=1)
                   if cr and old_rles else np.zeros(len(cr)))
        # prep1 means each candidate's own view already supplies one support.
        for index, row in enumerate(base, 1):
            result.append(Filament(f'{image}_{index}', image, image, row.counts, row.area))
        index = len(base)
        for i, proposal in enumerate(candidates):
            if proposal['area'] < 300 or old_iou[i] < .3:
                continue
            x, y, w, h = map(int, mu.toBbox(proposal['native_rle']))
            mask = mu.decode(proposal['native_rle'])[y:y+h, x:x+w].astype(bool)
            occupied = covered[y:y+h, x:x+w]
            if int((mask & occupied).sum()) / proposal['area'] >= .1:
                continue
            remaining = mask & ~occupied
            area = int(remaining.sum())
            if area < 300:
                continue
            occupied |= remaining
            rle = encode_roi(remaining, y, x)
            assert int(mu.area(rle)) == area
            index += 1
            added += 1
            result.append(Filament(f'{image}_{index}', image, image,
                                   rle['counts'].decode('ascii'), area))
        if number % 30 == 0:
            print('Rebuilt low-confidence rescue', number, '/', len(images), flush=True)
    assert added == 84
    return result, dict(old=old_meta, new=new_meta, added_physical_masks=added)


def fixture():
    square = np.ones((20, 20), bool)
    stripe = np.ones((4, 100), bool)
    features = []
    for index, mask in enumerate((square, stripe)):
        rle = encode_roi(mask, 10, 10)
        row = Filament(f'x_{index}', 'x', 'x', rle['counts'].decode('ascii'), int(mu.area(rle)))
        features.append(shape_features(row))
    assert features[0]['elongation'] < 1.01 and features[1]['elongation'] > 20
    rule = dict(rule='round_compact', elongation=2., compactness=.4)
    assert not keep(features[0], rule) and keep(features[1], rule)
    assert features[0]['median_interior'] > features[1]['median_interior']


def save_csv(path, rows):
    with path.open('x', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filament_id', 'segmentation_rle'])
        writer.writerows((r.filament_id, r.counts) for r in rows)


def main():
    started = time.monotonic()
    cv2.setNumThreads(1)
    grid = settings()
    fixture()
    with VAL.open(newline='') as f:
        selection = list(csv.DictReader(f))
    groups = {r['image_id'] for r in selection}
    images = {Path(r['file_name']).stem for r in selection}
    current = read_rows(CURRENT)
    lower, cache_meta = rebuild_low_confidence(images)
    sources = dict(current_best=current, low_confidence_rescue=lower)
    overlaps = {source: overlap_stats(rows) for source, rows in sources.items()}
    assert all(s['overlap_images'] == 0 for s in overlaps.values())
    features = {source: [shape_features(row) for row in rows] for source, rows in sources.items()}
    predictions = {
        (source, setting['name']): [r for r, feature in zip(rows, features[source]) if keep(feature, setting)]
        for source, rows in sources.items() for setting in grid}
    # GT is first loaded only after fixed-rule mask lists are fully constructed.
    truth = read_rows(GT, gt=True, selected=groups)
    assert len(truth) == 708 and len(groups) == 101 and len(images) == 91
    reports = []
    for source, rows in sources.items():
        for setting in grid:
            selected = predictions[source, setting['name']]
            score = evaluate(truth, selected, SIZE, SIZE)
            reports.append(dict(source=source, **setting,
                **{k: score[k] for k in ('pq', 'sq', 'rq', 'tp_pairs', 'fp', 'fn')},
                prediction_rows=len(selected), rejected_rows=len(rows)-len(selected),
                overlap_images=0, extra_matching_pairs=score['tp_pairs']-(708-score['fn'])))
    baselines = {r['source']: r for r in reports if r['name'] == 'baseline'}
    assert abs(baselines['current_best']['pq']-CURRENT_EXPECTED) < 1e-12
    assert abs(baselines['low_confidence_rescue']['pq']-LOW_EXPECTED) < 1e-12
    assert (baselines['low_confidence_rescue']['tp_pairs'], baselines['low_confidence_rescue']['fp'],
            baselines['low_confidence_rescue']['fn']) == (491, 266, 217)
    best = max(reports, key=lambda r: r['pq'])
    best_per_source = {source: max((r for r in reports if r['source'] == source), key=lambda r: r['pq'])
                       for source in sources}
    directory = Path('/workspace/artifacts/shape_quality_candidates_v1')
    directory.mkdir(exist_ok=False)
    saved = {}
    for source, record in best_per_source.items():
        path = directory/f'{source}_{record["name"]}.csv'
        chosen = predictions[source, record['name']]
        save_csv(path, chosen)
        saved[source] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             overlap=overlap_stats(chosen))
    report = dict(
        scope='16 fixed same-development GT-free shape rules on each of two inputs; not independent CV or LB',
        feature_rules='Mask-pixel covariance elongation sqrt(maxEigen/max(minEigen,.25)); compactness4piA/sum(all contour perimeters)^2; zero-padded EDT median inside mask; native bbox diagonal',
        selection_rules='Fixed global keep/delete only; no per-image/GT object choices; GT loaded after predictions built',
        physical_images=len(images), annotation_groups=len(groups), gt_instances=len(truth),
        settings_per_input=len(grid), total_comparisons=len(reports), current_input_sha256=hashlib.sha256(CURRENT.read_bytes()).hexdigest(),
        cache_source=cache_meta, baselines_reproduced=baselines, input_overlap=overlaps,
        experiments=reports, best=best, best_per_source=best_per_source, saved_candidates=saved,
        improved_over_current=best['pq'] > CURRENT_EXPECTED, target_exceeded=best['pq'] > .5,
        elapsed_seconds=time.monotonic()-started)
    destination = ROOT/'docs/evidence/shape_quality_experiments.json'
    destination.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(best=best, best_per_source=best_per_source,
                         improved_over_current=report['improved_over_current'],
                         saved_candidates=saved, elapsed_seconds=report['elapsed_seconds']), indent=2))


if __name__ == '__main__':
    main()
