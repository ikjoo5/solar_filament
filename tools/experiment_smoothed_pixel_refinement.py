"""Thirteen fixed GC comparisons: identity plus band1/2/3 x gray blur0/1/2/3.

This is the final small smoothing-before-GrabCut ablation. Ground truth is
read only for complete-setting evaluation, never for image/object decisions.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
from pycocotools import mask as mu

from experiment_mask_postprocess import group_rows, overlap_stats, read_rows
from experiment_view_combinations import ROOT, GT, VAL, SIZE, encode_roi
from filament_eval import Filament, evaluate
from refine_masks_with_pixels import TilePixels, refine_mask, resolve_rois


PIXELS = Path('/workspace/artifacts/pixel_tiles_20261027T203313Z_fd420182')
BASELINE = Path('/workspace/artifacts/boundary_smoothing_candidates_v2/gaussian_s1.2_t0.6.csv')
OLD_SUMMARY = Path('/workspace/artifacts/pixel_refinement_full91_smooth_v1/summary.json')
OUTPUT = Path('/workspace/artifacts/smoothed_pixel_refinement_candidates_v1')
STORE = None


def settings():
    result = [dict(name='baseline', method=None)]
    for band in (1, 2, 3):
        for sigma in (0, 1, 2, 3):
            result.append(dict(name=f'grabcut_b{band}_gray_sigma{sigma}', method='grabcut',
                               band=band, gray_sigma=sigma, core_mode='erode3'))
    assert len(result) == 13 and len({row['name'] for row in result}) == 13
    return result


def initialize_worker():
    global STORE
    cv2.setNumThreads(1)
    STORE = TilePixels(PIXELS)


def one_image(task):
    image, members = task
    gray, coverage = STORE.load(image)
    originals = [mu.decode(row.rle(SIZE, SIZE)).astype(bool) for row in members]
    union = np.zeros((SIZE, SIZE), bool)
    for original in originals:
        union |= original
    output = {'baseline': members}
    statuses = {}
    for setting in settings()[1:]:
        rois, regions, changed = [], [], []
        counts = Counter()
        for index, (row, original) in enumerate(zip(members, originals)):
            x, y, w, h = map(int, mu.toBbox(row.rle(SIZE, SIZE)))
            pad = setting['band'] + 3 + math.ceil(3 * setting['gray_sigma'])
            region = np.s_[max(0, y-pad):min(SIZE, y+h+pad), max(0, x-pad):min(SIZE, x+w+pad)]
            native = original[region]
            local_gray = gray[region]
            if not coverage[region].all():
                candidate, status = native.copy(), 'fallback_missing_pixels'
            else:
                if setting['gray_sigma']:
                    local_gray = cv2.GaussianBlur(local_gray, (0, 0), setting['gray_sigma'],
                                                  borderType=cv2.BORDER_REFLECT_101)
                candidate, status = refine_mask(local_gray, native, union[region] & ~native,
                    method='grabcut', band=setting['band'], core_mode='erode3',
                    coverage=coverage[region], seed=index)
            rois.append(native); regions.append(region); changed.append(candidate)
            counts[status] += 1
        resolved = resolve_rois(changed, rois, regions, (SIZE, SIZE))
        rows = []
        for row, candidate, region in zip(members, resolved, regions):
            ys, xs = np.nonzero(candidate)
            assert len(ys) > 0, 'One instance must remain per input'
            top, left = int(ys.min()), int(xs.min())
            rle = encode_roi(candidate[top:int(ys.max())+1, left:int(xs.max())+1],
                             region[0].start+top, region[1].start+left)
            rows.append(Filament(row.filament_id, image, image, rle['counts'].decode('ascii'), int(mu.area(rle))))
        output[setting['name']] = rows
        statuses[setting['name']] = counts
    return image, output, statuses, dict(STORE.missing)


def main():
    started = time.monotonic()
    cv2.setNumThreads(1)
    grid = settings()  # Frozen before any GT score is computed.
    with VAL.open(newline='') as handle:
        selected = list(csv.DictReader(handle))
    groups = {row['image_id'] for row in selected}
    images = {Path(row['file_name']).stem for row in selected}
    original_rows = read_rows(BASELINE)
    assert len(original_rows) == 626 and overlap_stats(original_rows)['overlap_images'] == 0
    grouped = group_rows(original_rows)
    predictions = {row['name']: [] for row in grid}
    statuses = {row['name']: Counter() for row in grid}
    missing = {}
    tasks = [(image, grouped.get(image, [])) for image in sorted(images)]
    with ProcessPoolExecutor(max_workers=4, initializer=initialize_worker) as pool:
        for number, (image, rows, counts, absent) in enumerate(pool.map(one_image, tasks), 1):
            for setting in grid:
                name = setting['name']
                predictions[name].extend(rows[name])
                statuses[name].update(counts.get(name, {}))
            missing.update(absent)
            if number % 10 == 0 or number == len(images):
                print('Pre-GC smoothing', number, '/', len(images), flush=True)
    truth = read_rows(GT, gt=True, selected=groups)
    assert len(truth) == 708 and len(groups) == 101 and len(images) == 91
    old = json.loads(OLD_SUMMARY.read_text())
    raw_reference = next(row for row in old['experiments'] if row['name'] == 'grabcut_b3_erode3')
    OUTPUT.mkdir(parents=True, exist_ok=False)
    reports = []
    for setting in grid:
        name = setting['name']
        rows = predictions[name]
        assert len(rows) == 626
        score = evaluate(truth, rows, SIZE, SIZE)
        overlap = overlap_stats(rows)
        assert overlap['overlap_images'] == 0
        assert score['tp_pairs']-(len(truth)-score['fn']) == 0
        if setting['method'] is None:
            assert abs(score['pq']-.4607591314345197) < 1e-12
        elif setting['band'] == 3 and setting['gray_sigma'] == 0:
            assert abs(score['pq']-raw_reference['pq']) < 1e-12, (score, raw_reference)
        path = OUTPUT/(name+'.csv')
        with path.open('x', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['filament_id', 'segmentation_rle'])
            writer.writerows((row.filament_id, row.counts) for row in rows)
        reports.append(dict(**setting, **{k: score[k] for k in ('pq', 'sq', 'rq', 'tp_pairs', 'fp', 'fn')},
                            statuses=dict(statuses[name]), prediction_rows=len(rows),
                            extra_matching_pairs=0, **overlap, csv=str(path)))
    best = max(reports, key=lambda row: row['pq'])
    actual = max(reports[1:], key=lambda row: row['pq'])
    report = dict(scope='13 fixed same-development full91 comparisons; no per-image/object GT decisions; not independent CV or Kaggle LB',
        rules='Fixed band1/2/3 x grayGaussian sigma0/1/2/3; coreerode3; GC3iterations; same darkcontrast/area/IoU safeguards; nearest original ownership; whole ROIcoverage including3sigma blur context required',
        physical_images=len(images), annotation_groups=len(groups), gt_instances=len(truth),
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (BASELINE, PIXELS/'manifest.json')},
        missing_tile_files=missing, worker_count=4, old_raw_band3_reproduced=True,
        raw_band3_reference_pq=raw_reference['pq'], experiments=reports,
        baseline=reports[0], best=best, best_actual_refinement=actual,
        improved_over_baseline=best['pq'] > reports[0]['pq'], target_exceeded=best['pq'] > .5,
        elapsed_seconds=time.monotonic()-started)
    (ROOT/'docs/evidence/smoothed_pixel_refinement_experiments.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ('baseline', 'best', 'best_actual_refinement',
          'old_raw_band3_reproduced', 'improved_over_baseline', 'elapsed_seconds')}, indent=2))


if __name__ == '__main__':
    main()
