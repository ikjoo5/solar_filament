"""Compact GT-free admissions of novel native-tile mask proposals.

Retain the 626-mask reference unchanged. Match support must span distinct
input scales, not merely overlapping crop locations. GT is scored only after
every candidate prediction is complete. The initial 32-setting grid has a
documented 16-setting higher-confidence follow-up after its boundary optimum.
"""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from pycocotools import mask as mu

from experiment_gated_fn_rescue import OLD, NEW, OLD_VIEWS, NEW_VIEWS, load
from experiment_mask_postprocess import read_rows, group_rows, overlap_stats
from experiment_view_combinations import ROOT, GT, VAL, SIZE, encode_roi
from filament_eval import Filament, _check_compressed_counts, evaluate


TILES = Path('/workspace/artifacts/tiled_results_20261027T205426Z_eba01cfb/raw_tiled_views.jsonl')
BASELINE = Path('/workspace/artifacts/boundary_smoothing_candidates_v2/gaussian_s1.2_t0.6.csv')
EXPECTED = .4607591314345197


def plan(high_confidence=False):
    if high_confidence:
        return [dict(name=f'tile_c{c:g}_s{s}_old{o:g}_novel0.1',
                     confidence=c, support=s, old_iou=o, novelty=.1, minimum_area=300)
                for c in (.55, .65, .75, .85) for s in (1, 2) for o in (0., .3)]
    return [dict(name=f'tile_c{c:g}_s{s}_old{o:g}_novel{n:g}',
                 confidence=c, support=s, old_iou=o, novelty=n, minimum_area=300)
            for c in (.15, .25, .35, .45) for s in (1, 2)
            for o in (0., .3) for n in (.1, .25)]


def load_tiles(path, images):
    result = defaultdict(list)
    seen = set()
    masks = edge = 0
    views = {f'tile{s}_y{y}_x{x}' for s in (1536, 2048)
             for y in (0, 512, 1024) for x in (0, 512, 1024)}
    per_image = defaultdict(set)
    for line in path.open():
        row = json.loads(line)
        image, view = row['image'], row['view']
        assert image in images and view in views and (image, view) not in seen
        assert row['shape'] == [SIZE, SIZE] and row['coordinate_space'] == 'original'
        assert row['input_size'] in (1536, 2048)
        seen.add((image, view)); per_image[image].add(view)
        for p in row['instances']:
            confidence = p['confidence']
            assert np.isfinite(confidence) and .05 <= confidence <= 1
            area = _check_compressed_counts(p['rle'], SIZE*SIZE)
            assert area > 0
            masks += 1; edge += bool(p['edge_clipped'])
            if not p['edge_clipped'] and confidence >= .15 and area >= 300:
                result[image].append(dict(**p, scale=row['input_size'], view=view, area=area,
                    native_rle=dict(size=[SIZE, SIZE], counts=p['rle'].encode('ascii'))))
    assert set(per_image) == images and all(v == views for v in per_image.values())
    return result, dict(records=len(seen), masks=masks, edge_clipped_masks=edge,
        eligible_nonedge_masks=sum(map(len, result.values())),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def fixture():
    candidates = [dict(scale=1536), dict(scale=1536), dict(scale=2048)]
    matrix = np.array([[1., .8, .5], [.8, 1., .4], [.5, .4, 1.]])
    support = [len({p['scale'] for j, p in enumerate(candidates) if matrix[i, j] >= .3})
               for i in range(3)]
    assert support == [2, 2, 2]
    candidates.pop(); matrix = matrix[:2, :2]
    assert [len({p['scale'] for j, p in enumerate(candidates) if matrix[i, j] >= .3})
            for i in range(2)] == [1, 1]
    original = np.zeros((32, 48), bool); original[5:16, 7:19] = True
    previous = original.copy(); previous[:, 14:] = False
    remaining = original[5:16, 7:19] & ~previous[5:16, 7:19]
    np.testing.assert_array_equal(mu.decode(encode_roi(remaining, 5, 7, 32, 48)), original & ~previous)
    assert len(plan()) == 32 and len({s['name'] for s in plan()}) == 32
    assert len(plan(True)) == 16 and len({s['name'] for s in plan(True)}) == 16


def main():
    started = time.monotonic(); cv2.setNumThreads(1); fixture()
    high_confidence = '--high-confidence' in sys.argv
    settings = plan(high_confidence)  # Fixed before the current run opens GT.
    with VAL.open(newline='') as f:
        selection = list(csv.DictReader(f))
    ids = {r['image_id'] for r in selection}
    images = {Path(r['file_name']).stem for r in selection}
    assert len(images) == 91 and len(ids) == 101
    baseline = read_rows(BASELINE)
    assert len(baseline) == 626 and overlap_stats(baseline)['overlap_images'] == 0
    grouped = group_rows(baseline)
    tiles, tile_meta = load_tiles(TILES, images)
    old, old_meta = load(OLD, images, OLD_VIEWS)
    new, new_meta = load(NEW, images, NEW_VIEWS)
    output = {s['name']: [] for s in settings}
    additions = {s['name']: 0 for s in settings}
    for number, image in enumerate(sorted(images), 1):
        base = grouped.get(image, [])
        rles = [r.rle(SIZE, SIZE) for r in base]
        if rles:
            union = mu.merge(rles)
            assert int(mu.area(union)) == sum(r.area for r in base)
            initial = mu.decode(union).astype(bool)
        else:
            initial = np.zeros((SIZE, SIZE), bool)
        candidates = sorted(tiles[image], key=lambda p: -p['confidence'])
        cr = [p['native_rle'] for p in candidates]
        matrix = np.asarray(mu.iou(cr, cr, [0]*len(cr))) if cr else np.zeros((0, 0))
        prior = [p['native_rle'] for cache in (old[image], new[image])
                 for rows in cache.values() for p in rows if p['confidence'] >= .1]
        old_iou = (np.asarray(mu.iou(cr, prior, [0]*len(prior))).max(axis=1)
                   if cr and prior else np.zeros(len(cr)))
        support = {c: [len({p['scale'] for j, p in enumerate(candidates)
                            if p['confidence'] >= c and matrix[i, j] >= .3})
                       for i in range(len(candidates))] for c in sorted({s['confidence'] for s in settings})}
        decoded = {}
        for setting in settings:
            covered = initial.copy()
            rows = output[setting['name']]
            # Reuse exact original rows/IDs; the source masks are immutable.
            rows.extend(base)
            index = len(base)
            for i, p in enumerate(candidates):
                if (p['confidence'] < setting['confidence'] or
                    old_iou[i] < setting['old_iou'] or
                    support[setting['confidence']][i] < setting['support']):
                    continue
                if i not in decoded:
                    x, y, w, h = map(int, mu.toBbox(p['native_rle']))
                    decoded[i] = x, y, w, h, mu.decode(p['native_rle'])[y:y+h, x:x+w].astype(bool)
                x, y, w, h, mask = decoded[i]
                occupied = covered[y:y+h, x:x+w]
                if np.count_nonzero(mask & occupied)/p['area'] >= setting['novelty']:
                    continue
                remaining = mask & ~occupied
                area = int(remaining.sum())
                if area < setting['minimum_area']:
                    continue
                occupied |= remaining
                rle = encode_roi(remaining, y, x)
                assert int(mu.area(rle)) == area
                index += 1; additions[setting['name']] += 1
                rows.append(Filament(f'{image}_{index}', image, image, rle['counts'].decode('ascii'), area))
        if number % 15 == 0 or number == len(images):
            print('Tiled novel rescue', number, '/', len(images), flush=True)
    # GT is used only now, after all fixed-rule predictions have been formed.
    truth = read_rows(GT, gt=True, selected=ids)
    assert len(truth) == 708
    reference = evaluate(truth, baseline, SIZE, SIZE)
    assert abs(reference['pq']-EXPECTED) < 1e-12
    reports = []
    for setting in settings:
        rows = output[setting['name']]
        score = evaluate(truth, rows, SIZE, SIZE)
        assert score['tp_pairs']-(len(truth)-score['fn']) == 0
        reports.append(dict(**setting, **{k: score[k] for k in ('pq', 'sq', 'rq', 'tp_pairs', 'fp', 'fn')},
            added_physical_instances=additions[setting['name']], prediction_rows=len(rows),
            extra_matching_pairs=0, overlap_images=0))
    best = max(reports, key=lambda r: r['pq'])
    # A second saved candidate exposes the FN/FP tradeoff without GT object selection.
    balanced = [r for r in reports if r['fp'] <= reference['fp']+25]
    recall = min(balanced, key=lambda r: (r['fn'], -r['pq'])) if balanced else best
    out = Path('/workspace/artifacts/tiled_rescue_candidates_v2' if high_confidence else
               '/workspace/artifacts/tiled_rescue_candidates_v1'); out.mkdir(exist_ok=False)
    saved = {}
    for label, result in [('best_pq', best), ('fn_fp_budget', recall)]:
        rows = output[result['name']]
        assert overlap_stats(rows)['overlap_images'] == 0
        path = out/(label+'_'+result['name']+'.csv')
        with path.open('x', newline='') as f:
            writer = csv.writer(f); writer.writerow(['filament_id', 'segmentation_rle'])
            writer.writerows((r.filament_id, r.counts) for r in rows)
        reloaded = read_rows(path)
        actual = evaluate(truth, reloaded, SIZE, SIZE)
        assert len(reloaded) == len(rows) and abs(actual['pq']-result['pq']) < 1e-12
        saved[label] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            score_recomputed=True, **overlap_stats(reloaded))
    report = dict(scope='Fixed same-development native-tile admission settings; not independent CV or LB',
        physical_images=len(images), annotation_groups=len(ids), gt_instances=len(truth), settings=len(settings),
        rules='Immutable baseline; nonedge proposals descending actual confidence; paired support from distinct input scales at IoU>=.3; optional old/preprocessing corroboration; novelty and remaining-area gate; no GT-dependent object/image rules',
        source_cache=dict(tiles=tile_meta, old=old_meta, preprocessing=new_meta),
        input_sha256=hashlib.sha256(BASELINE.read_bytes()).hexdigest(),
        baseline_reproduced=True, baseline={k: reference[k] for k in ('pq', 'sq', 'rq', 'tp_pairs', 'fp', 'fn')},
        experiments=reports, best=best, fn_fp_budget_candidate=recall, saved_candidates=saved,
        improved_over_reference=best['pq']>reference['pq'], target_exceeded=best['pq']>.5,
        elapsed_seconds=time.monotonic()-started)
    evidence = ROOT/'docs/evidence/tiled_rescue_experiments.json'
    if high_confidence:
        initial = json.loads(evidence.read_text())
        assert initial['settings'] == 32
        initial['initial_32_best'] = initial['best']
        initial['followup_high_confidence'] = dict(
            rationale='Initial best occurred at confidence upper boundary. Sixteen higher-confidence rules were then frozen before follow-up scoring.',
            **report)
        initial['experiments'].extend(reports)
        initial['settings'] = len(initial['experiments'])
        initial['best'] = max(initial['experiments'], key=lambda r: r['pq'])
        if best['pq'] > initial['initial_32_best']['pq']:
            initial['initial_saved_candidates'] = initial['saved_candidates']
            initial['saved_candidates'] = saved
        global_balanced = [r for r in initial['experiments'] if r['fp'] <= reference['fp']+25]
        initial['fn_fp_budget_candidate'] = min(global_balanced, key=lambda r: (r['fn'], -r['pq']))
        initial['improved_over_reference'] = initial['best']['pq'] > reference['pq']
        initial['target_exceeded'] = initial['best']['pq'] > .5
        initial['elapsed_seconds'] += report['elapsed_seconds']
        evidence.write_text(json.dumps(initial, indent=2)+'\n')
    else:
        evidence.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(best=best, fn_fp_budget=recall, saved=saved, seconds=report['elapsed_seconds']), indent=2))


if __name__ == '__main__':
    main()
