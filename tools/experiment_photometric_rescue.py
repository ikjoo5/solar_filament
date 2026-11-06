"""Twenty-four fixed photometric gates for novel preprocessing proposals.

No baseline instances are removed or modified. Pixel features use no GT;
ground truth is consulted only after the full grid's predictions are built.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from pycocotools import mask as mu
from scipy import ndimage

from experiment_gated_fn_rescue import (
    BASELINE, EXPECTED, GT, NEW, NEW_VIEWS, OLD, OLD_VIEWS, ROOT, SIZE, VAL,
    encode_roi, fixture as ownership_fixture, load,
)
from experiment_mask_postprocess import group_rows, overlap_stats, read_rows
from filament_eval import Filament, evaluate
from refine_masks_with_pixels import TilePixels


PIXELS = Path('/workspace/artifacts/pixel_tiles_20261027T203313Z_fd420182')
CURRENT = Path('/workspace/artifacts/gated_rescue_candidates_v1/c0.3_old0.3_novel0.1_area300_prep1.csv')
CURRENT_PQ = .45911798265305065
OTHER_REFERENCE_PQ_ROUNDED = .460759


def plan(contrasts=(0., .5, 1.)):
    return [dict(name=f'photo_c{confidence:g}_z{contrast:g}_s{support}',
                 confidence=confidence, contrast=contrast, support=support)
            for confidence in (.15, .20, .25, .30)
            for contrast in contrasts
            for support in (1, 2)]


def contrast_feature(gray, original, covered, existing):
    """Robust darker-core contrast with an explicit missing-pixel rejection.

Only selected FG/BG sample coordinates need coverage; unused rectangular ROI
corners have no effect. Core is relative distance>=60% maximum, minimum1px.
BG is 3..10px from the proposal and excludes established baseline masks.
"""
    original = np.asarray(original, bool)
    if not (gray.dtype == np.uint8 and gray.ndim == 2 and
            gray.shape == original.shape == covered.shape == existing.shape):
        raise ValueError('Inconsistent photometric ROI arrays')
    if not original.any():
        raise ValueError('Empty proposal')
    inside = ndimage.distance_transform_edt(original)
    core = original & (inside >= max(1., .6 * float(inside.max())))
    outside = ndimage.distance_transform_edt(~original)
    ring = (outside >= 3.) & (outside <= 10.) & ~original & ~existing
    if core.sum() < 6 or ring.sum() < 12:
        return None, 'insufficient_samples'
    if not np.all(covered[core | ring]):
        return None, 'missing_pixels'
    fg = gray[core].astype(np.float64)
    bg = gray[ring].astype(np.float64)
    fm, bm = float(np.median(fg)), float(np.median(bg))
    fs = max(3., 1.4826 * float(np.median(np.abs(fg - fm))))
    bs = max(3., 1.4826 * float(np.median(np.abs(bg - bm))))
    sigma = float(np.sqrt(.5 * (fs*fs + bs*bs)))
    difference = bm - fm
    feature = dict(difference=difference, normalized_contrast=difference/sigma,
                   foreground_median=fm, background_median=bm,
                   foreground_samples=int(core.sum()), background_samples=int(ring.sum()))
    return feature, 'ok' if difference >= 1. else 'low_dark_contrast'


def fixture():
    original = np.zeros((44, 50), bool)
    original[15:29, 20:30] = True
    pixels = np.full(original.shape, 150, np.uint8)
    pixels[original] = 80
    coverage = np.ones_like(original)
    existing = np.zeros_like(original)
    feature, status = contrast_feature(pixels, original, coverage, existing)
    assert status == 'ok' and feature['difference'] == 70 and feature['normalized_contrast'] > 20
    feature, status = contrast_feature(np.full_like(pixels, 150), original, coverage, existing)
    assert status == 'low_dark_contrast' and feature['normalized_contrast'] == 0
    coverage[21, 24] = False
    feature, status = contrast_feature(pixels, original, coverage, existing)
    assert feature is None and status == 'missing_pixels'
    coverage[:] = True
    coverage[0, 0] = False
    feature, status = contrast_feature(pixels, original, coverage, existing)
    assert status == 'ok'  # irrelevant corners are never read as samples
    ownership_fixture()
    assert len(plan()) == 24 and len({s['name'] for s in plan()}) == 24
    assert len(plan((2., 3., 4.))) == 24


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--high-contrast', action='store_true',
                        help='Run the fixed follow-up grid z2/3/4, preserving first-grid evidence')
    args = parser.parse_args(argv)
    started = time.monotonic()
    cv2.setNumThreads(1)
    fixture()
    contrasts = (2., 3., 4.) if args.high_contrast else (0., .5, 1.)
    settings = plan(contrasts)  # fixed before any score computation
    output = {s['name']: [] for s in settings}
    additions = Counter()
    rejection = {s['name']: Counter() for s in settings}
    reference_rows = []
    with VAL.open(newline='') as f:
        selection = list(csv.DictReader(f))
    ids = {r['image_id'] for r in selection}
    images = {Path(r['file_name']).stem for r in selection}
    baseline = read_rows(BASELINE)
    base_by_image = group_rows(baseline)
    old, old_meta = load(OLD, images, OLD_VIEWS)
    new, new_meta = load(NEW, images, NEW_VIEWS)
    store = TilePixels(PIXELS)
    if store.shape != (SIZE, SIZE) or not images <= set(store.manifest['images']):
        raise ValueError('Missing native validation image metadata')
    previous = json.loads((OLD.parent/'summary.json').read_text())
    latest = json.loads((NEW.parent/'summary.json').read_text())
    assert previous['source_sha256'] == latest['source_sha256'] == store.manifest['source_sha256']
    with VAL.open('rb') as f:
        assert hashlib.file_digest(f, 'sha256').hexdigest() == store.manifest['source_sha256']['val_split.csv']
    feature_counts = Counter()
    used_contrast = []
    control = dict(name='rescue_control_no_pixels', confidence=.3, contrast=None, support=1)
    all_settings = [control] + settings
    for number, image in enumerate(sorted(images), 1):
        gray, coverage = store.load(image)
        base = base_by_image.get(image, [])
        base_rles = [r.rle(SIZE, SIZE) for r in base]
        if base_rles:
            union = mu.merge(base_rles)
            assert int(mu.area(union)) == sum(r.area for r in base)
            base_covered = mu.decode(union).astype(bool)
        else:
            base_covered = np.zeros((SIZE, SIZE), bool)
        candidates = []
        for view, rows in new[image].items():
            if view == 'limb':
                continue
            candidates.extend(dict(**p, view=view) for p in rows if p['confidence'] >= .15)
        candidates.sort(key=lambda p: -p['confidence'])
        cr = [p['native_rle'] for p in candidates]
        old_rles = [p['native_rle'] for rows in old[image].values() for p in rows if p['confidence'] >= .1]
        matrix = np.asarray(mu.iou(cr, cr, [0]*len(cr))) if cr else np.zeros((0, 0))
        corroboration = (np.asarray(mu.iou(cr, old_rles, [0]*len(old_rles))).max(axis=1)
                         if cr and old_rles else np.zeros(len(cr)))
        supports = {}
        for confidence in (.15, .20, .25, .30):
            supports[confidence] = [len({p['view'] for j, p in enumerate(candidates)
                if p['confidence'] >= confidence and matrix[i, j] >= .3}) for i in range(len(candidates))]
        decoded = {}
        features = {}
        for i, proposal in enumerate(candidates):
            if proposal['area'] < 300 or corroboration[i] < .3:
                continue
            x, y, w, h = map(int, mu.toBbox(proposal['native_rle']))
            native = mu.decode(proposal['native_rle']).astype(bool)
            # Retain a small ROI, rather than keeping its full native canvas alive.
            mask = native[y:y+h, x:x+w].copy()
            decoded[i] = x, y, w, h, mask
            # Non-novel proposals cannot be admitted by any setting.
            if np.count_nonzero(mask & base_covered[y:y+h, x:x+w]) / proposal['area'] >= .1:
                continue
            x1, y1, x2, y2 = max(0, x-12), max(0, y-12), min(SIZE, x+w+12), min(SIZE, y+h+12)
            region = np.s_[y1:y2, x1:x2]
            feature, status = contrast_feature(gray[region], native[region], coverage[region], base_covered[region])
            features[i] = feature, status
            feature_counts[status] += 1
            if status == 'ok':
                used_contrast.append(feature['normalized_contrast'])
        for setting in all_settings:
            covered = base_covered.copy()
            rows = reference_rows if setting['contrast'] is None else output[setting['name']]
            for index, row in enumerate(base, 1):
                rows.append(Filament(f'{image}_{index}', image, image, row.counts, row.area))
            index = len(base)
            for i, proposal in enumerate(candidates):
                if (proposal['confidence'] < setting['confidence'] or proposal['area'] < 300
                        or corroboration[i] < .3 or supports[setting['confidence']][i] < setting['support']):
                    continue
                x, y, w, h, mask = decoded[i]
                occupied = covered[y:y+h, x:x+w]
                if np.count_nonzero(mask & occupied)/proposal['area'] >= .1:
                    continue
                remaining = mask & ~occupied
                area = int(remaining.sum())
                if area < 300:
                    continue
                if setting['contrast'] is not None:
                    feature, status = features[i]
                    if status != 'ok':
                        rejection[setting['name']][status] += 1
                        continue
                    if feature['normalized_contrast'] < setting['contrast']:
                        rejection[setting['name']]['contrast_threshold'] += 1
                        continue
                occupied |= remaining
                rle = encode_roi(remaining, y, x)
                assert int(mu.area(rle)) == area
                index += 1
                rows.append(Filament(f'{image}_{index}', image, image, rle['counts'].decode('ascii'), area))
                if setting['contrast'] is not None:
                    additions[setting['name']] += 1
        if number == 1 or number % 20 == 0 or number == len(images):
            print('Photometric rescue', number, '/', len(images), flush=True)
    # No GT was read while proposing, measuring, or selecting candidate masks.
    truth = read_rows(GT, gt=True, selected=ids)
    assert len(truth) == 708 and len(images) == 91 and len(ids) == 101
    base_score = evaluate(truth, baseline, SIZE, SIZE)
    assert abs(base_score['pq'] - EXPECTED) < 1e-12
    reference = evaluate(truth, reference_rows, SIZE, SIZE)
    saved_reference = read_rows(CURRENT)
    assert len(reference_rows) == 626 and abs(reference['pq'] - CURRENT_PQ) < 1e-12
    assert [(r.filament_id, r.counts) for r in reference_rows] == [(r.filament_id, r.counts) for r in saved_reference]
    reports = []
    for setting in settings:
        rows = output[setting['name']]
        score = evaluate(truth, rows, SIZE, SIZE)
        extra = score['tp_pairs'] - (len(truth)-score['fn'])
        reports.append(dict(**setting, **{k: score[k] for k in ['pq', 'sq', 'rq', 'tp_pairs', 'fp', 'fn']},
            added_physical_instances=additions[setting['name']], prediction_rows=len(rows),
            rejected_by_pixels=dict(rejection[setting['name']]), extra_matching_pairs=extra))
    best = max(reports, key=lambda r: r['pq'])
    folder = Path('/workspace/artifacts/photometric_rescue_candidates_v2' if args.high_contrast
                  else '/workspace/artifacts/photometric_rescue_candidates_v1')
    folder.mkdir(exist_ok=False)
    path = folder/(best['name']+'.csv')
    saved = output[best['name']]
    audit = overlap_stats(saved)
    assert not any(audit.values()) and best['extra_matching_pairs'] == 0
    with path.open('x', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filament_id', 'segmentation_rle'])
        writer.writerows((r.filament_id, r.counts) for r in saved)
    report = dict(
        scope='24 fixed same-development photometric admission comparisons, plus two exact controls; no independent CV/LB claim',
        physical_images=91, annotation_groups=101, gt_instances=708, settings_declared=24,
        contrast_thresholds=list(contrasts),
        grid_note=('Follow-up global thresholds chosen to span the GT-free candidate contrast distribution; initial low-z grid evidence retained'
                   if args.high_contrast else 'Initial low-z grid'),
        source_cache={'old': old_meta, 'new': new_meta},
        pixel_manifest_sha256=hashlib.sha256((PIXELS/'manifest.json').read_bytes()).hexdigest(),
        pixel_source_sha256=store.manifest['source_sha256'], pixel_manifest_source_hashes_match=True,
        rules='Preserve every baseline mask; confidence-sorted novel preprocessing proposal; old conf>=.10 IoU>=.30 corroboration; novelty<.10; remaining area>=300; distinct preprocessing support; robust relative-core versus3..10px ring dark contrast; missing samples reject candidate',
        feature_status_counts=dict(feature_counts),
        eligible_feature_contrast_quantiles=np.quantile(used_contrast, [0, .25, .5, .75, 1]).tolist() if used_contrast else [],
        fixture_checks=['dark stripe contrast', 'constant rejection', 'missing FG rejection', 'irrelevant corners ignored', 'ROI ownership RLE roundtrip', '24 unique fixed settings'],
        missing_tile_files=dict(store.missing), baseline_reproduced=True,
        baseline={k: base_score[k] for k in ['pq', 'sq', 'rq', 'tp_pairs', 'fp', 'fn']},
        rescue_control_reproduced=True, rescue_control_byte_identical_counts=True,
        rescue_control={k: reference[k] for k in ['pq', 'sq', 'rq', 'tp_pairs', 'fp', 'fn']},
        experiments=reports, best=best, saved_candidate=str(path), saved_candidate_audit=audit,
        improved_over_rescue_reference=best['pq'] > CURRENT_PQ,
        compared_other_reference_pq_rounded=OTHER_REFERENCE_PQ_ROUNDED,
        improved_over_other_reference_rounded=best['pq'] > OTHER_REFERENCE_PQ_ROUNDED,
        target_exceeded=best['pq'] > .5, elapsed_seconds=time.monotonic()-started,
    )
    report_name = ('photometric_rescue_highcontrast_experiments.json' if args.high_contrast
                   else 'photometric_rescue_experiments.json')
    (ROOT/'docs/evidence'/report_name).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(best=best, saved=str(path), seconds=report['elapsed_seconds']), indent=2))


if __name__ == '__main__':
    main()
