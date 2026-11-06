"""Four fixed downstream boundary ensembles of two trained native models.

All candidate CSVs are generated before reading GT. No instance is added or
removed. This is repeatedly reused development data, not an independent test.
"""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from pycocotools import mask as mu
from scipy import ndimage

from experiment_mask_postprocess import group_rows, overlap_stats, read_rows
from experiment_native_cross_model import matched_ids
from experiment_paper_densecrf import paired_delta, site_metrics
from experiment_view_combinations import SIZE, encode_roi
from filament_eval import Filament, evaluate

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/workspace/artifacts/rich_boundary_composition_v1/tta_precision.csv')
OLD = Path('/workspace/artifacts/best84_rich_predictions_tta/predictions.jsonl')
NEW = Path('/workspace/artifacts/context_only_uploaded_20261029/multiscale_context/raw_paper_views.jsonl')
OUT = Path('/workspace/artifacts/round10_joint_boundary')


def plan():
    return [dict(name=f'tta{threshold:g}_{fusion}', threshold=threshold, fusion=fusion,
        peer_original_iou=.65, peer_pair_iou=.65, area_ratio=[.67, 1.5],
        minimum_original_iou=.85, minimum_area=300,
        preserve_connected_component_count=True, reject_other_instance_intrusion=True)
        for threshold in (.5, .65) for fusion in ('majority', 'signed_distance')]


def signed_distance(mask):
    mask = np.asarray(mask, bool)
    return ndimage.distance_transform_edt(mask) - ndimage.distance_transform_edt(~mask)


def refine_roi(original, first, second, other, setting):
    masks = [np.asarray(value, bool) for value in (original, first, second)]
    original, first, second = masks
    other = np.asarray(other, bool)
    if any(value.shape != original.shape for value in (*masks, other)) or not original.any():
        raise ValueError('Matched nonempty masks and disjoint occupancy of the same shape required')
    if np.any(original & other):
        raise ValueError('Original and another original instance overlap')
    if setting['fusion'] == 'majority':
        candidate = np.stack(masks).sum(axis=0) >= 2
    elif setting['fusion'] == 'signed_distance':
        candidate = sum(signed_distance(value) for value in masks) > 0
    else:
        raise ValueError('Unknown fixed boundary fusion')
    area, old_area = int(candidate.sum()), int(original.sum())
    union = int((candidate | original).sum())
    if (area < setting['minimum_area'] or not setting['area_ratio'][0] <= area/old_area <= setting['area_ratio'][1]
            or int((candidate & original).sum())/union < setting['minimum_original_iou']):
        return original.copy(), 'fallback_geometry'
    if np.any(candidate & other):
        return original.copy(), 'fallback_other_original_intrusion'
    if ndimage.label(candidate, np.ones((3, 3)))[1] != ndimage.label(original, np.ones((3, 3)))[1]:
        return original.copy(), 'fallback_component_count'
    return candidate, 'changed' if np.any(candidate != original) else 'unchanged'


def mutual_assignment(original_rles, peers, setting):
    if not peers or not original_rles:
        return [None] * len(original_rles)
    proposals = [row['rle'] for row in peers]
    iou = np.asarray(mu.iou(original_rles, proposals, [0]*len(proposals)))
    chosen = iou.argmax(axis=1)
    assigned = iou.argmax(axis=0)
    result = []
    for i, j in enumerate(chosen):
        ratio = int(mu.area(proposals[j])) / int(mu.area(original_rles[i]))
        result.append(int(j) if assigned[j] == i and iou[i, j] >= setting['peer_original_iou']
            and setting['area_ratio'][0] <= ratio <= setting['area_ratio'][1] else None)
    return result


def revert_collisions(rows, originals):
    """Expansion into shared old background must not introduce overlapping masks."""
    if len(rows) < 2:
        return rows, 0
    rles = [r.rle(SIZE, SIZE) for r in rows]
    overlaps = np.asarray(mu.iou(rles, rles, [0]*len(rles))) > 0
    np.fill_diagonal(overlaps, False)
    conflicts = overlaps.any(axis=0)
    return [old if conflict else new for new, old, conflict in zip(rows, originals, conflicts)], int(conflicts.sum())


def image_variants(image, originals, old, new, settings):
    if not originals:
        return {s['name']: ([], Counter()) for s in settings}
    br = [r.rle(SIZE, SIZE) for r in originals]
    occupied = mu.decode(mu.merge(br)).astype(bool)
    outputs = {}
    for setting in settings:
        threshold = f"{setting['threshold']:g}"
        peers = [record['tta']['components'][threshold] for record in (old, new)]
        assignment = [mutual_assignment(br, values, setting) for values in peers]
        rows, counters = [], Counter()
        for i, original in enumerate(originals):
            choices = [a[i] for a in assignment]
            if None in choices:
                rows.append(original); counters['fallback_missing_mutual_peer'] += 1
                continue
            peer_rles = [members[j]['rle'] for members, j in zip(peers, choices)]
            if mu.iou(peer_rles[:1], peer_rles[1:], [0])[0, 0] < setting['peer_pair_iou']:
                rows.append(original); counters['fallback_peer_pair_disagreement'] += 1
                continue
            boxes = [mu.toBbox(r).astype(int) for r in [br[i], *peer_rles]]
            left = max(0, min(b[0] for b in boxes)-2)
            top = max(0, min(b[1] for b in boxes)-2)
            right = min(SIZE, max(b[0]+b[2] for b in boxes)+2)
            bottom = min(SIZE, max(b[1]+b[3] for b in boxes)+2)
            region = np.s_[top:bottom, left:right]
            local = mu.decode(br[i])[region].astype(bool)
            first, second = [mu.decode(r)[region].astype(bool) for r in peer_rles]
            candidate, status = refine_roi(local, first, second, occupied[region] & ~local, setting)
            counters[status] += 1
            if status != 'changed':
                rows.append(original)
                continue
            rle = encode_roi(candidate, top, left)
            rows.append(Filament(original.filament_id, image, image,
                                 rle['counts'].decode(), int(mu.area(rle))))
        rows, collisions = revert_collisions(rows, originals)
        counters['collision_rollbacks'] += collisions
        counters['actual_changed_masks'] = sum(a.counts != b.counts for a, b in zip(rows, originals))
        outputs[setting['name']] = rows, counters
    return outputs


def save(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['filament_id', 'segmentation_rle'])
        writer.writerows((r.filament_id, r.counts) for r in rows)


def main():
    started = time.monotonic(); cv2.setNumThreads(1)
    settings = plan()
    OUT.mkdir(exist_ok=False)
    (OUT/'predeclared_plan.json').write_text(json.dumps(settings, indent=2)+'\n')
    original = read_rows(BASE); grouped = group_rows(original)
    assert len(original) == 625 and not any(overlap_stats(original).values())
    old = {Path(r['file_name']).stem:r for r in map(json.loads, OLD.open())}
    new = {Path(r['file_name']).stem:r for r in map(json.loads, NEW.open())}
    assert len(old) == len(new) == 91 and old.keys() == new.keys()
    assert all(old[key]['image_sha256'] == new[key]['image_sha256'] for key in old)
    assert not set(grouped)-set(old)
    outputs = {s['name']: [] for s in settings}
    statuses = {s['name']: Counter() for s in settings}
    for i, image in enumerate(sorted(old), 1):
        result = image_variants(image, grouped.get(image, []), old[image], new[image], settings)
        for name, (rows, counters) in result.items():
            outputs[name].extend(rows); statuses[name].update(counters)
        if i % 15 == 0 or i == 91:
            print('Boundary ensemble', i, '/91', flush=True)
    shutil.copyfile(BASE, OUT/'frozen_control.csv')
    for setting in settings:
        rows = outputs[setting['name']]
        assert len(rows) == 625 and {r.filament_id for r in rows} == {r.filament_id for r in original}
        assert not any(overlap_stats(rows).values())
        save(OUT/(setting['name']+'.csv'), rows)
    # All four complete91-image predictions now exist; labels are first opened here.
    from verify_native_training_results import GT, VAL
    groups = {r['image_id'] for r in csv.DictReader(VAL.open())}
    truth = read_rows(GT, gt=True, selected=groups)
    assert len(truth) == 708 and len(groups) == 101
    reference = evaluate(truth, read_rows(OUT/'frozen_control.csv'), SIZE, SIZE)
    assert abs(reference['pq'] - .47259304445299505) < 1e-12
    before = matched_ids(truth, original)
    reference_sites = site_metrics(reference)
    reports = []
    for setting in settings:
        path = OUT/(setting['name']+'.csv')
        rows = read_rows(path)
        score = evaluate(truth, rows, SIZE, SIZE)
        assert score['tp_pairs'] == len(truth)-score['fn']
        after = matched_ids(truth, rows)
        sites = site_metrics(score)
        reports.append(dict(**setting, **{k: score[k] for k in ('pq', 'sq', 'rq', 'tp_pairs', 'fp', 'fn')},
            delta_pq=score['pq']-reference['pq'], physical_masks=len(rows),
            recovered_gt=len(after-before), lost_gt=len(before-after),
            recovered_gt_ids=sorted(after-before), lost_gt_ids=sorted(before-after),
            statuses=dict(statuses[setting['name']]), **overlap_stats(rows),
            by_site=sites, improved_sites=sum(sites[s]['pq'] > reference_sites[s]['pq'] for s in sites),
            **paired_delta(reference, score), csv=str(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), scored_after_save=True))
    report = dict(scope='Four predeclared downstream two-native-model boundary ensembles on reused91-image development cohort',
        validation=dict(physical_images=91, annotation_groups=101, gt_instances=708),
        prediction_instance_count_fixed=625, baseline=dict(name='frozen_control',
            **{k:reference[k] for k in ('pq', 'sq', 'rq', 'tp_pairs', 'fp', 'fn')}),
        baseline_csv_byte_identical=BASE.read_bytes() == (OUT/'frozen_control.csv').read_bytes(),
        baseline_by_site=reference_sites, source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (BASE, OLD, NEW)},
        experiments=reports, best=max(reports, key=lambda r:r['pq']),
        adopted=False, new_network_inference=False, training_performed=False, external_data_used=False,
        limitation='All91images have been used repeatedly for development. Bootstrap is conditional and not corrected for prior model/method selection; this is not five-fold CV or publicLB.',
        elapsed_seconds=time.monotonic()-started)
    (ROOT/'docs/evidence/round10_joint_native_boundary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(baseline=report['baseline'], experiments=[{k:r[k] for k in
        ('name', 'pq', 'tp_pairs', 'fp', 'fn', 'recovered_gt', 'lost_gt', 'improved_sites', 'statuses')}
        for r in reports], adopted=False, elapsed_seconds=report['elapsed_seconds']), indent=2))


if __name__ == '__main__':
    main()
