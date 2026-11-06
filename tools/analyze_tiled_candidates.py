"""Audit native tile proposals and diagnose GT availability without predictions.

Candidate availability is GT-aware diagnosis only. This program never chooses
an oracle prediction, reports oracle PQ, or claims achievable FP-controlled recall.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from pycocotools import mask as mu

from analyze_preprocessing_candidates import cache, overlaps, available, OLD_VIEWS, NEW_VIEWS
from analyze_validation_results import read_rows
from filament_eval import _check_compressed_counts


ROOT = Path(__file__).resolve().parents[1]
TILED = Path('/workspace/artifacts/tiled_results_20261027T205426Z_eba01cfb')
OLD = Path('/workspace/artifacts/validation_results_20261027T120958Z_444da7f4')
NEW = Path('/workspace/artifacts/validation_results_20261027T125130Z_2312e759')
PIXELS = Path('/workspace/artifacts/pixel_tiles_20261027T203313Z_fd420182/manifest.json')
GT = Path('/workspace/artifacts/solar_filament_gt_v1/gt.csv')
VAL = Path('/workspace/attachments/bcb64220-b208-4cd7-9c3f-3f2be0f04167/val_split.csv')
BEST = Path('/workspace/artifacts/boundary_smoothing_candidates_v2/gaussian_s1.2_t0.6.csv')
CONFIDENCES = (.05, .1, .2, .3, .4)


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load_tiled(images):
    summary = json.loads((TILED/'summary.json').read_text())
    assert digest(TILED/'raw_tiled_views.jsonl') == summary['raw_tiled_views_sha256']
    assert digest(TILED/'worker_source.py') == summary['worker_source_sha256']
    pixels = json.loads(PIXELS.read_text())
    previous = json.loads((NEW/'summary.json').read_text())
    assert summary['source_sha256'] == pixels['source_sha256'] == previous['source_sha256']
    actual = {
        'MAGFiLO_1.0_Annotations_kaggle2026_train.json': Path('/workspace/datasets/solar_filament/MAGFiLO_1.0_Annotations_kaggle2026_train.json'),
        'best.pt': Path('/workspace/attachments/f27dbac6-f4d7-4f71-bdea-2df7dc252775/best.pt'),
        'train_split.csv': Path('/workspace/attachments/1cca1607-f94f-4334-9586-c1fb5ab292d7/train_split.csv'),
        'val_split.csv': VAL,
    }
    assert {name: digest(path) for name, path in actual.items()} == summary['source_sha256']
    assert set(summary['image_sha256']) == set(summary['limb_sha256']) == images
    assert all(summary['image_sha256'][stem] == pixels['images'][stem]['raw_sha256']
               and summary['limb_sha256'][stem] == pixels['images'][stem]['limb_sha256']
               for stem in images)
    expected = {f'tile{s}_y{y}_x{x}': (s, x, y) for s in (1536, 2048)
                for y in (0, 512, 1024) for x in (0, 512, 1024)}
    grouped = defaultdict(dict)
    counts = Counter()
    for line in (TILED/'raw_tiled_views.jsonl').open():
        row = json.loads(line)
        stem, view = row['image'], row['view']
        assert stem in images and view in expected and view not in grouped[stem]
        size, left, top = expected[view]
        assert row['input_size'] == size and row['crop_bbox'] == [left, top, 1024, 1024]
        assert row['shape'] == [2048, 2048] and row['coordinate_space'] == 'original'
        assert row['raw_min_confidence'] == .05 and row['max_det'] == 100
        assert row['raw_box_count'] == len(row['instances']) + row['empty_mask_count']
        assert bool(row['max_det_reached']) == (row['raw_box_count'] >= 100)
        counts['records'] += 1
        counts['instances'] += len(row['instances'])
        counts['max_det_records'] += bool(row['max_det_reached'])
        counts['empty_masks'] += row['empty_mask_count']
        for proposal in row['instances']:
            confidence = proposal['confidence']
            assert np.isfinite(confidence) and .05 <= confidence <= 1
            assert _check_compressed_counts(proposal['rle'], 2048**2) > 0
            native = dict(size=[2048, 2048], counts=proposal['rle'].encode('ascii'))
            x, y, width, height = map(int, mu.toBbox(native))
            assert left <= x < x+width <= left+1024 and top <= y < y+height <= top+1024
            sides = []
            if top > 0 and y < top+4: sides.append('top')
            if left > 0 and x < left+4: sides.append('left')
            if top+1024 < 2048 and y+height > top+1020: sides.append('bottom')
            if left+1024 < 2048 and x+width > left+1020: sides.append('right')
            assert proposal['edge_sides'] == sides and bool(proposal['edge_clipped']) == bool(sides)
            counts['edge_clipped_instances'] += bool(sides)
            counts[f'instances_{size}'] += 1
        grouped[stem][view] = row['instances']
    assert set(grouped) == images and all(set(views) == set(expected) for views in grouped.values())
    for name, value in summary['counts'].items():
        assert counts[name] == value, (name, counts[name], value)
    assert summary['physical_images'] == len(images) == 91
    assert summary['annotation_groups'] == 101 and summary['views_per_image'] == 18
    return grouped, summary, dict(counts)


def main():
    started = time.monotonic()
    with VAL.open(newline='') as handle:
        selection = list(csv.DictReader(handle))
    groups = {row['image_id'] for row in selection}
    images = {Path(row['file_name']).stem for row in selection}
    truth = read_rows(GT, gt=True, allowed_groups=groups)
    prediction = read_rows(BEST)
    assert len(groups) == 101 and sum(map(len, truth.values())) == 708
    tiled, summary, counts = load_tiled(images)
    old, old_meta = cache(OLD/'raw_views.jsonl', images, OLD_VIEWS)
    new, new_meta = cache(NEW/'raw_preprocess_views.jsonl', images, NEW_VIEWS)
    # Baseline/new-limb duplicate masks never generate additional availability.
    labels = ('old4', 'preprocess8', 'old4_plus_preprocess8', 'tiles_all',
              'tiles_nonclipped', 'tiles1536_nonclipped', 'tiles2048_nonclipped',
              'combined_all', 'combined_nonclipped')
    totals = {str(c): {label: Counter() for label in labels} for c in CONFIDENCES}
    per_gt = []
    transitions = Counter()
    for group, rows in truth.items():
        stem = group.split('-', 1)[1]
        gt = [row['rle'] for row in rows]
        current = [row['rle'] for row in prediction.get(stem, [])]
        ci = np.asarray(mu.iou(gt, current, [0]*len(current))) if current else np.zeros((len(gt), 0))
        matched = (ci > .5).any(axis=1)
        missed = ~matched
        transitions.update(gt=len(rows), baseline_matched=int(matched.sum()), baseline_fn=int(missed.sum()))
        oi, oc = overlaps(gt, [p for view in OLD_VIEWS for p in old[stem][view]])
        ni, nc = overlaps(gt, [p for view in NEW_VIEWS for p in new[stem][view]])
        all_tiles = [(view, p) for view, proposals in tiled[stem].items() for p in proposals]
        ti, tc = overlaps(gt, [p for view, p in all_tiles])
        nonclipped = np.asarray([not p['edge_clipped'] for view, p in all_tiles], bool)
        sizes = np.asarray([int(view.split('_', 1)[0][4:]) for view, p in all_tiles], int)
        for confidence in CONFIDENCES:
            of = available(oi, oc, confidence)
            nf = available(ni, nc, confidence)
            old_all = of | nf
            flags = dict(old4=of, preprocess8=nf, old4_plus_preprocess8=old_all,
                         tiles_all=available(ti, tc, confidence),
                         tiles_nonclipped=available(ti[:, nonclipped], tc[nonclipped], confidence))
            for size in (1536, 2048):
                eligible = nonclipped & (sizes == size)
                flags[f'tiles{size}_nonclipped'] = available(ti[:, eligible], tc[eligible], confidence)
            flags['combined_all'] = old_all | flags['tiles_all']
            flags['combined_nonclipped'] = old_all | flags['tiles_nonclipped']
            for label, flag in flags.items():
                totals[str(confidence)][label].update(
                    gt_available=int(flag.sum()), baseline_fn_available=int((flag & missed).sum()),
                    beyond_old12=int((flag & ~old_all).sum()),
                    baseline_fn_beyond_old12=int((flag & missed & ~old_all).sum()))
            if confidence == .05:
                for index, row in enumerate(rows):
                    per_gt.append(dict(gt_id=row['id'], image=stem, baseline_matched=bool(matched[index]),
                        old12_available=bool(old_all[index]), tile_available=bool(flags['tiles_all'][index]),
                        nonclipped_tile_available=bool(flags['tiles_nonclipped'][index]),
                        baseline_best_iou=float(ci[index].max()) if current else 0.,
                        tile_best_iou=float(ti[index].max()) if len(all_tiles) else 0.,
                        nonclipped_tile_best_iou=float(ti[index, nonclipped].max()) if nonclipped.any() else 0.))
    assert transitions == dict(gt=708, baseline_matched=479, baseline_fn=229)
    assert totals['0.05']['old4']['gt_available'] == 639
    assert totals['0.05']['preprocess8']['gt_available'] == 629
    assert totals['0.05']['old4_plus_preprocess8']['gt_available'] == 650
    matched_delta = np.asarray([row['nonclipped_tile_best_iou']-row['baseline_best_iou']
                                for row in per_gt if row['baseline_matched']])
    metadata = {row['filament_id']: row for row in json.loads((GT.parent/'manifest.json').read_text())['annotations']}
    novel = [row for row in per_gt if row['nonclipped_tile_available'] and not row['old12_available']]
    novel_categories = Counter(metadata[row['gt_id']]['category_name'] for row in novel)
    novel_areas = [metadata[row['gt_id']]['rasterized_area'] for row in novel]
    result = dict(scope='Independent raw tile integrity audit plus GT-aware candidate-availability diagnosis; not achieved recall/PQ or deployable oracle selection',
        availability_rule='Any raw mask with confidence at least threshold and strict IoU>.5; annotator GT variants remain correlated',
        source_sha256=summary['source_sha256'], raw_tiled_sha256=summary['raw_tiled_views_sha256'],
        uploaded_worker_sha256=summary['worker_source_sha256'], counts=counts,
        integrity=dict(source_four_hashes_match_actual_local_inputs=True,
            raw_and_worker_hashes_match=True, all91_raw_image_and_limb_hashes_match_pixel_export=True,
            native_rle_valid=True, all_masks_within_declared_crop=True,
            exactly18_views_per_image=True, edge_flags_recomputed_and_exact=True,
            no_max_det_records=counts['max_det_records'] == 0),
        previous_caches=dict(old4=old_meta, preprocess8=new_meta), baseline_prediction=str(BEST),
        baseline_counts=dict(transitions), availability_by_confidence=totals,
        already_matched_gt_tile_iou_diagnostic=dict(
            scope='GT-aware best-candidate shape opportunity, not a deployable replacement rule or achieved SQ/PQ',
            baseline_matched_gt=len(matched_delta),
            gt_with_a_nonclipped_tile_candidate_better_by={str(t):int((matched_delta>t).sum()) for t in (0., .02, .05, .1)},
            gt_with_best_tile_iou_lower_than_baseline=int((matched_delta<0).sum())),
        new_only_nonclipped_gt_at_confidence_005=dict(count=len(novel), category_counts=dict(novel_categories),
            area_quantiles={str(q):float(np.quantile(novel_areas,q)) for q in (0., .25, .5, .75, 1.)}),
        per_gt_diagnostics=per_gt, elapsed_seconds=time.monotonic()-started)
    target = ROOT/'docs/evidence/tiled_candidate_audit.json'
    target.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(counts=counts, baseline_counts=dict(transitions),
        availability_by_confidence=totals, elapsed_seconds=result['elapsed_seconds']), indent=2))


if __name__ == '__main__':
    main()
