"""Twenty-four fixed instance-preserving tile fusion rules plus one control.

The grid and proposal assignment inspect predictions only. GT is loaded after
all prediction sets are complete. Each tile proposal belongs to at most one
baseline instance, and each instance uses at most one proposal per input scale.
"""
from collections import Counter, defaultdict
import argparse
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
from filament_eval import Filament, _check_compressed_counts, evaluate
from refine_masks_with_pixels import resolve_rois

TILED = Path('/workspace/artifacts/tiled_results_20261027T205426Z_eba01cfb/raw_tiled_views.jsonl')
BASELINE = Path('/workspace/artifacts/boundary_smoothing_candidates_v2/gaussian_s1.2_t0.6.csv')
EXPECTED = .4607591314345197


def plan(guard_followup=False):
    if guard_followup:
        settings = [dict(name=f'c{c:g}_iou0.3_majority_guard0.5', confidence=c,
                         match_iou=.3, operation='majority', original_iou_guard=.5)
                    for c in (.45, .55, .65, .75)]
        settings += [dict(name=f'c{c:g}_iou0.3_majority_guard{guard:g}', confidence=c,
                          match_iou=.3, operation='majority', original_iou_guard=guard)
                     for c in (.35, .55, .75) for guard in (.7, .85, .95)]
        settings += [dict(name='c0.35_iou0.3_majority_guard0.5_control', confidence=.35,
                          match_iou=.3, operation='majority', original_iou_guard=.5)]
        assert len(settings) == 14 and len({s['name'] for s in settings}) == 14
        return settings
    return [dict(name=f'c{c:g}_iou{overlap:g}_{operation}', confidence=c,
                 match_iou=overlap, operation=operation)
            for c in (.15, .25, .35) for overlap in (.3, .5)
            for operation in ('union', 'majority', 'intersection', 'best_tile')]


def fused_mask(original, first, second, operation, confidences, minimum_iou=.5):
    if operation == 'best_tile':
        candidate = first.copy() if confidences[0] >= confidences[1] else second.copy()
    else:
        count = original.astype(np.uint8) + first + second
        needed = {'union': 1, 'majority': 2, 'intersection': 3}[operation]
        candidate = count >= needed
    area = int(candidate.sum())
    ratio = area / int(original.sum())
    union = int((candidate | original).sum())
    overlap = int((candidate & original).sum()) / union if union else 0.
    if not .5 <= ratio <= 1.5 or overlap < minimum_iou:
        return original.copy(), 'fallback_excessive_change'
    return candidate, 'changed' if np.any(candidate != original) else 'unchanged'


def fixture():
    original = np.zeros((30, 40), bool); original[10:20, 12:25] = True
    original[13:16, 16:19] = False
    first = original.copy(); first[9, 12:25] = True
    second = original.copy(); second[20, 12:25] = True
    for operation in ('union', 'majority', 'intersection', 'best_tile'):
        mask, status = fused_mask(original, first, second, operation, (.7, .6))
        assert mask.any() and mask.dtype == bool
        assert not mask[14, 17], 'Hole must not be filled by rectangular ROI placement'
    changed, status = fused_mask(original, np.ones_like(original), np.ones_like(original), 'union', (.7, .6))
    assert status == 'fallback_excessive_change' and np.array_equal(changed, original)
    native = mu.decode(encode_roi(original, 2000, 1990))
    assert np.array_equal(native[2000:2030, 1990:2030], original)
    assert len(plan()) == 24 and len({s['name'] for s in plan()}) == 24


def load_tiles(images):
    grouped = defaultdict(list); seen = set(); total = eligible = 0
    for line in TILED.open():
        row = json.loads(line)
        image, view = row['image'], row['view']
        assert image in images and (image, view) not in seen
        seen.add((image, view))
        assert row['shape'] == [SIZE, SIZE] and row['coordinate_space'] == 'original'
        assert row['input_size'] in (1536, 2048)
        for proposal in row['instances']:
            total += 1
            confidence = proposal['confidence']
            assert np.isfinite(confidence) and .05 <= confidence <= 1.
            area = _check_compressed_counts(proposal['rle'], SIZE*SIZE)
            assert area > 0
            if proposal['edge_clipped'] or confidence < .15:
                continue
            rle = dict(size=[SIZE, SIZE], counts=proposal['rle'].encode('ascii'))
            grouped[image].append(dict(confidence=confidence, size=row['input_size'],
                view=view, rle=rle, area=area, bbox=list(map(int, mu.toBbox(rle)))))
            eligible += 1
    assert len(seen) == len(images)*18
    return grouped, dict(records=len(seen), instances=total, eligible=eligible,
                         sha256=hashlib.sha256(TILED.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--guard-followup', action='store_true')
    args = parser.parse_args()
    started = time.monotonic(); cv2.setNumThreads(1); fixture()
    settings = plan(args.guard_followup)
    with VAL.open(newline='') as handle:
        selection = list(csv.DictReader(handle))
    groups = {r['image_id'] for r in selection}
    images = {Path(r['file_name']).stem for r in selection}
    baseline = read_rows(BASELINE)
    assert len(images) == 91 and len(groups) == 101 and len(baseline) == 626
    assert overlap_stats(baseline)['overlap_images'] == 0
    by_image = group_rows(baseline)
    tiled, source = load_tiles(images)
    predictions = {s['name']: [] for s in settings}
    statuses = {s['name']: Counter() for s in settings}
    unique_builds = 0
    for number, image in enumerate(sorted(images), 1):
        members = by_image.get(image, [])
        proposals = tiled[image]
        br = [r.rle(SIZE, SIZE) for r in members]
        tr = [p['rle'] for p in proposals]
        matrix = (np.asarray(mu.iou(br, tr, [0]*len(tr))) if br and tr
                  else np.zeros((len(br), len(tr))))
        assignment = matrix.argmax(axis=0) if len(br) and len(tr) else np.empty(0, int)
        original_bboxes = [list(map(int, mu.toBbox(r))) for r in br]
        decoded = {}

        def tile_roi(index):
            if index not in decoded:
                x, y, w, h = proposals[index]['bbox']
                decoded[index] = mu.decode(tr[index])[y:y+h, x:x+w].astype(bool)
            return decoded[index]

        built = {}
        for setting in settings:
            matches = []
            for i, row in enumerate(members):
                selected = []
                for scale in (1536, 2048):
                    valid = [j for j, p in enumerate(proposals)
                             if assignment[j] == i and p['size'] == scale
                             and p['confidence'] >= setting['confidence']
                             and matrix[i, j] >= setting['match_iou']]
                    selected.append(max(valid, key=lambda j: proposals[j]['confidence']) if valid else None)
                matches.append(tuple(selected))
            signature = setting['operation'], setting.get('original_iou_guard', .5), tuple(matches)
            if signature not in built:
                candidates, originals, regions = [], [], []
                counters = Counter()
                for i, (row, pair) in enumerate(zip(members, matches)):
                    x, y, w, h = original_bboxes[i]
                    boxes = [(x, y, w, h)] + [proposals[j]['bbox'] for j in pair if j is not None]
                    x1 = min(b[0] for b in boxes); y1 = min(b[1] for b in boxes)
                    x2 = max(b[0]+b[2] for b in boxes); y2 = max(b[1]+b[3] for b in boxes)
                    region = np.s_[y1:y2, x1:x2]
                    original = mu.decode(br[i])[region].astype(bool)
                    if None in pair:
                        candidate = original.copy(); status = 'fallback_missing_scale'
                    else:
                        tile_masks = []
                        for j in pair:
                            bx, by, bw, bh = proposals[j]['bbox']
                            mask = np.zeros(original.shape, bool)
                            mask[by-y1:by-y1+bh, bx-x1:bx-x1+bw] = tile_roi(j)
                            tile_masks.append(mask)
                        candidate, status = fused_mask(original, *tile_masks, setting['operation'],
                                                       [proposals[j]['confidence'] for j in pair],
                                                       setting.get('original_iou_guard', .5))
                    candidates.append(candidate); originals.append(original); regions.append(region)
                    counters[status] += 1
                resolved = resolve_rois(candidates, originals, regions, (SIZE, SIZE))
                rows = []
                for row, mask, region in zip(members, resolved, regions):
                    ys, xs = np.nonzero(mask); assert len(ys) > 0
                    top, left = int(ys.min()), int(xs.min())
                    rle = encode_roi(mask[top:int(ys.max())+1, left:int(xs.max())+1],
                                     region[0].start+top, region[1].start+left)
                    rows.append(Filament(row.filament_id, image, image,
                                        rle['counts'].decode('ascii'), int(mu.area(rle))))
                built[signature] = rows, counters
                unique_builds += 1
            rows, counters = built[signature]
            predictions[setting['name']].extend(rows)
            statuses[setting['name']].update(counters)
        if number % 15 == 0:
            print('Object fusion', number, '/', len(images), flush=True)
    # Only now inspect GT; no rule/instance/image selection uses it.
    truth = read_rows(GT, gt=True, selected=groups)
    assert len(truth) == 708
    reference = evaluate(truth, baseline, SIZE, SIZE)
    assert abs(reference['pq'] - EXPECTED) < 1e-12
    reports = [dict(name='baseline', operation='identity', **{k: reference[k] for k in
                   ('pq', 'sq', 'rq', 'tp_pairs', 'fp', 'fn')}, prediction_rows=len(baseline))]
    for setting in settings:
        rows = predictions[setting['name']]
        assert len(rows) == len(baseline)
        score = evaluate(truth, rows, SIZE, SIZE)
        overlap = overlap_stats(rows)
        assert not overlap['overlap_images']
        extra = score['tp_pairs'] - (len(truth)-score['fn'])
        assert extra == 0
        reports.append(dict(**setting, **{k: score[k] for k in
                            ('pq', 'sq', 'rq', 'tp_pairs', 'fp', 'fn')},
                            prediction_rows=len(rows), statuses=dict(statuses[setting['name']]),
                            extra_matching_pairs=extra, **overlap))
    best = max(reports, key=lambda r: r['pq'])
    best_actual = max(reports[1:], key=lambda r: r['pq'])
    eligible = [r for r in reports if r['fn'] <= reference['fn'] and r['fp'] <= reference['fp']]
    best_fn_fp_retention = max(eligible, key=lambda r: r['pq'])
    if args.guard_followup:
        control = next(r for r in reports if r['name'].endswith('_control'))
        assert abs(control['pq']-.46263674283303874) < 1e-12
    saved = None
    if best['pq'] > reference['pq']:
        folder = Path('/workspace/artifacts/tiled_object_fusion_candidates_v2' if args.guard_followup
                      else '/workspace/artifacts/tiled_object_fusion_candidates_v1')
        folder.mkdir(exist_ok=False)
        path = folder/(best['name']+'.csv')
        with path.open('x', newline='') as handle:
            writer = csv.writer(handle); writer.writerow(['filament_id', 'segmentation_rle'])
            writer.writerows((r.filament_id, r.counts) for r in predictions[best['name']])
        reloaded = read_rows(path)
        verified = evaluate(truth, reloaded, SIZE, SIZE)
        assert abs(verified['pq']-best['pq']) < 1e-12
        saved = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     score_recomputed_after_save=True)
    retention_saved = saved if best_fn_fp_retention['name'] == best['name'] else None
    if best_fn_fp_retention['pq'] > reference['pq'] and retention_saved is None:
        path = folder/(best_fn_fp_retention['name']+'.csv')
        with path.open('x', newline='') as handle:
            writer = csv.writer(handle); writer.writerow(['filament_id', 'segmentation_rle'])
            writer.writerows((r.filament_id, r.counts) for r in predictions[best_fn_fp_retention['name']])
        verified = evaluate(truth, read_rows(path), SIZE, SIZE)
        assert abs(verified['pq']-best_fn_fp_retention['pq']) < 1e-12
        retention_saved = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               score_recomputed_after_save=True)
    report = dict(scope=('13 adaptive confidence/original-IoU guard follow-up comparisons plus prior fusion and identity controls; same-development, not independent CV/LB'
                         if args.guard_followup else '24 fixed same-development object fusion comparisons plus identity; not independent CV/LB'),
        physical_images=len(images), annotation_groups=len(groups), gt_instances=len(truth),
        source_cache=source, baseline_sha256=hashlib.sha256(BASELINE.read_bytes()).hexdigest(),
        baseline_reproduced=True, selection_rules='Tile assigned only to highest-IoU baseline; highest confidence per input scale; require both scales; area0.5..1.5 and per-setting original IoU guard (legacy0.5); nearest-original ownership; GT loaded after all predictions built',
        experiments=reports, unique_per_image_builds=unique_builds, best=best,
        best_actual=best_actual, best_fn_fp_retention=best_fn_fp_retention,
        saved_candidate=saved, saved_fn_fp_retention_candidate=retention_saved,
        improved=best['pq']>reference['pq'],
        target_exceeded=best['pq']>.5, elapsed_seconds=time.monotonic()-started)
    if args.guard_followup:
        report['prior_fusion_control'] = control
    report_name = ('tiled_object_fusion_guard_experiments.json' if args.guard_followup
                   else 'tiled_object_fusion_experiments.json')
    (ROOT/'docs/evidence'/report_name).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(best=best, best_actual=best_actual, saved_candidate=saved,
                          best_fn_fp_retention=best_fn_fp_retention,
                          elapsed_seconds=report['elapsed_seconds']), indent=2))


if __name__ == '__main__':
    main()
