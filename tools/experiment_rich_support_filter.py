"""Fixed TTA/view-confirmation pruning of the current 630-mask incumbent.

All candidate rules inspect prediction-only support and preserve input geometry.
Annotations are loaded only after all 49 prediction sets have been constructed.
"""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from pycocotools import mask as mu

from experiment_mask_postprocess import group_rows, overlap_stats, read_rows
from experiment_native_cross_model import matched_ids
from experiment_view_combinations import GT, ROOT, SIZE, VAL
from filament_eval import evaluate

SOURCE = Path('/workspace/artifacts/best84_rich_predictions_tta/predictions.jsonl')
BASELINE = Path('/workspace/artifacts/native_high_confidence_candidates_v1/native_t0.5_a300_weak0.6_families1.csv')
OUTPUT = Path('/workspace/artifacts/rich_support_filter_v1')


def plan():
    result = [dict(name='incumbent', rule='identity')]
    for threshold in (.2, .35, .5):
        for coverage in (.1, .25, .4, .55):
            for stable in (.05, .25):
                for maximum_area in (600, 10**10):
                    result.append(dict(name=f't{threshold:g}_cov{coverage:g}_stable{stable:g}_a{maximum_area}',
                        rule='prune_low_support', threshold=threshold,
                        minimum_tta_coverage=coverage, minimum_two_view_coverage=stable,
                        maximum_area=maximum_area))
    assert len(result) == 49
    return result


def byte_rle(value):
    return dict(size=value['size'], counts=value['counts'].encode('ascii'))


def main():
    started = time.monotonic()
    grid = plan()
    OUTPUT.mkdir(exist_ok=True)
    plan_path = OUTPUT/'predeclared_plan.json'
    if plan_path.exists():
        assert json.loads(plan_path.read_text()) == grid
    else:
        plan_path.write_text(json.dumps(grid, indent=2)+'\n')
    with VAL.open(newline='') as handle:
        selection = list(csv.DictReader(handle))
    groups = {r['image_id'] for r in selection}
    images = {Path(r['file_name']).stem for r in selection}
    base = read_rows(BASELINE)
    assert len(base) == 630 and len(images) == 91 and len(groups) == 101
    assert not any(overlap_stats(base).values())
    originals = group_rows(base)
    features, seen = {}, set()
    for line in SOURCE.open():
        record = json.loads(line)
        image = Path(record['file_name']).stem
        assert image in images and image not in seen
        seen.add(image)
        assert record['size'] == [SIZE, SIZE]
        views = [mu.decode(byte_rle(rle)) for rle in record['view_unions'].values()]
        assert len(views) == 4
        count = sum(views, np.zeros((SIZE, SIZE), np.uint8))
        stable_rle = mu.encode(np.asfortranarray((count >= 2).astype(np.uint8)))
        support = {p: byte_rle(record['tta']['unions']['foreground'][f'{p:g}'])
                   for p in (.2, .35, .5)}
        for row in originals.get(image, []):
            original = row.rle(SIZE, SIZE)
            coverage = {p: int(mu.area(mu.merge([original, rle], intersect=True)))/row.area
                        for p, rle in support.items()}
            stable = int(mu.area(mu.merge([original, stable_rle], intersect=True)))/row.area
            features[row.filament_id] = dict(tta_coverage=coverage, two_view_coverage=stable)
    assert seen == images and len(features) == len(base)
    outputs = {}
    for setting in grid:
        if setting['rule'] == 'identity':
            outputs[setting['name']] = base
            continue
        outputs[setting['name']] = [r for r in base if not (
            r.area <= setting['maximum_area']
            and features[r.filament_id]['tta_coverage'][setting['threshold']] < setting['minimum_tta_coverage']
            and features[r.filament_id]['two_view_coverage'] < setting['minimum_two_view_coverage'])]
    # All masks and decisions now exist; only this scoring stage reads GT.
    truth = read_rows(GT, gt=True, selected=groups)
    assert len(truth) == 708
    ref_matches = matched_ids(truth, base)
    reports = []
    for setting in grid:
        rows = outputs[setting['name']]
        metric = evaluate(truth, rows, SIZE, SIZE)
        matches = matched_ids(truth, rows)
        assert matches <= ref_matches
        assert metric['tp_pairs'] == len(matches)
        reports.append(dict(**setting, **{k:metric[k] for k in ('pq','sq','rq','tp_pairs','fp','fn')},
            physical_masks=len(rows), removed_physical_masks=len(base)-len(rows),
            preserved_gt=len(matches), recovered_gt=0, lost_gt=len(ref_matches-matches),
            lost_gt_ids=sorted(ref_matches-matches)))
    assert abs(reports[0]['pq']-.46556248016002527)<1e-12
    best = max(reports, key=lambda r:r['pq'])
    best_no_loss = max((r for r in reports if r['lost_gt']==0), key=lambda r:r['pq'])
    saved = []
    for result in {r['name']:r for r in (best, best_no_loss)}.values():
        path = OUTPUT/(result['name']+'.csv')
        with path.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['filament_id','segmentation_rle'])
            writer.writerows((r.filament_id,r.counts) for r in outputs[result['name']])
        reloaded = read_rows(path)
        assert not any(overlap_stats(reloaded).values())
        score = evaluate(truth, reloaded, SIZE, SIZE)
        assert abs(score['pq']-result['pq'])<1e-12
        saved.append(dict(name=result['name'], path=str(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            score_recomputed_after_save=True, adopted_without_match_loss=result['lost_gt']==0))
    document = dict(scope='48 globally fixed support-pruning rules plus identity; same development set, not independent CV/LB',
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        baseline_sha256=hashlib.sha256(BASELINE.read_bytes()).hexdigest(),
        physical_images=91, annotation_groups=101, gt_instances=708,
        feature_definitions='Original mask area fraction covered by full TTAforeground union; two_view fraction covered by >=2of4 foreground threshold.5 view unions. Prune only if both are low and mask area within bound.',
        experiments=reports, best=best, best_preserving_all_matches=best_no_loss,
        saved_candidates=saved, all_outputs_disjoint=True,
        elapsed_seconds=time.monotonic()-started)
    (ROOT/'docs/evidence/rich_support_filter.json').write_text(json.dumps(document, indent=2)+'\n')
    print(json.dumps(dict(best=best, best_no_loss=best_no_loss, saved_candidates=saved,
                         elapsed_seconds=document['elapsed_seconds']), indent=2))


if __name__ == '__main__':
    main()
