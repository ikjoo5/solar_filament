"""Four predeclared two-model instance regroupings, preserving every FG pixel.

This uses repeated-development predictions, not independent CV or new inference.
Unlike prior one-model watershed grids, two trained models must corroborate each
split/merge. All CSVs are generated before annotation masks are opened.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from pycocotools import mask as mu
from scipy import ndimage

from experiment_mask_postprocess import group_rows, overlap_stats, read_rows
from experiment_native_cross_model import matched_ids
from experiment_paper_densecrf import paired_delta, site_metrics
from experiment_view_combinations import encode_roi
from filament_eval import evaluate

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/workspace/artifacts/rich_boundary_composition_v1/tta_precision.csv')
OLD = Path('/workspace/artifacts/best84_rich_predictions_tta/predictions.jsonl')
NEW = Path('/workspace/artifacts/context_only_uploaded_20261029/multiscale_context/raw_paper_views.jsonl')
OUT = Path('/workspace/artifacts/round10_joint_topology')


def plan():
    return [dict(name=f'{operation}_tta{threshold:g}', operation=operation,
                 threshold=threshold, cross_model_iou=.5 if operation == 'split' else .65,
                 minimum_area=300, containment=.8, result_native_iou=.5,
                 maximum_split_children=3, merge_objects=2)
            for operation, thresholds in [('split', (.5, .65)), ('merge', (.35, .5))]
            for threshold in thresholds]


def intersection(a, b):
    return int(mu.area(mu.merge([a, b], intersect=True)))


def mutual_pairs(old, new, minimum):
    if not old or not new:
        return []
    matrix = np.asarray(mu.iou(old, new, [0]*len(new)))
    return [(i, int(j)) for i, j in enumerate(matrix.argmax(axis=1))
            if matrix[i, j] >= minimum and matrix[:, j].argmax() == i]


def partition_by_seeds(foreground, seeds, minimum_area=300):
    """Deterministic Euclidean Voronoi partition, never adding/removing FG."""
    foreground = np.asarray(foreground, bool)
    seeds = [np.asarray(seed, bool) for seed in seeds]
    if len(seeds) < 2 or any(s.shape != foreground.shape or not s.any() for s in seeds):
        return None
    if any(np.any(s & ~foreground) for s in seeds) or np.any(np.sum(seeds, axis=0) > 1):
        return None
    distances = np.stack([ndimage.distance_transform_edt(~seed) for seed in seeds])
    assignment = distances.argmin(axis=0)
    pieces = [foreground & (assignment == i) for i in range(len(seeds))]
    if any(int(piece.sum()) < minimum_area for piece in pieces):
        return None
    assert np.array_equal(np.sum(pieces, axis=0), foreground.astype(int))
    return pieces


def split_masks(originals, old, new, setting):
    pairs = mutual_pairs(old, new, setting['cross_model_iou'])
    result, events, statuses = [], [], Counter()
    for number, original in enumerate(originals):
        cores, witnesses = [], []
        for a, b in pairs:
            if (intersection(original, old[a]) / int(mu.area(old[a])) < setting['containment']
                    or intersection(original, new[b]) / int(mu.area(new[b])) < setting['containment']):
                continue
            core = mu.merge([original, old[a], new[b]], intersect=True)
            if int(mu.area(core)) >= setting['minimum_area']:
                cores.append(core)
                witnesses.append((old[a], new[b]))
        if not 2 <= len(cores) <= setting['maximum_split_children']:
            result.append(original)
            statuses['unchanged_core_count'] += 1
            continue
        x, y, w, h = map(int, mu.toBbox(original))
        foreground = mu.decode(original)[y:y+h, x:x+w].astype(bool)
        seeds = [mu.decode(core)[y:y+h, x:x+w].astype(bool) for core in cores]
        pieces = partition_by_seeds(foreground, seeds, setting['minimum_area'])
        if pieces is None:
            result.append(original)
            statuses['unchanged_partition_guard'] += 1
            continue
        encoded = [encode_roi(piece, y, x) for piece in pieces]
        if any(min(mu.iou([child], list(peers), [0, 0])[0]) < setting['result_native_iou']
               for child, peers in zip(encoded, witnesses)):
            result.append(original)
            statuses['unchanged_native_boundary_guard'] += 1
            continue
        result.extend(encoded)
        statuses['split_original_objects'] += 1
        events.append(dict(original_indices=[number], output_objects=len(encoded),
                           original_area=int(mu.area(original)),
                           output_areas=[int(mu.area(r)) for r in encoded]))
    return result, events, statuses


def merge_masks(originals, old, new, setting):
    pairs = mutual_pairs(old, new, setting['cross_model_iou'])
    proposals, statuses = [], Counter()
    for a, b in pairs:
        selected = [i for i, original in enumerate(originals)
                    if intersection(original, old[a]) / int(mu.area(original)) >= setting['containment']
                    and intersection(original, new[b]) / int(mu.area(original)) >= setting['containment']]
        if len(selected) != setting['merge_objects']:
            statuses['unchanged_covered_object_count'] += 1
            continue
        merged = mu.merge([originals[i] for i in selected])
        if min(mu.iou([merged], [old[a], new[b]], [0, 0])[0]) < setting['result_native_iou']:
            statuses['unchanged_native_boundary_guard'] += 1
            continue
        proposals.append((tuple(selected), merged))
    # If two corroborated proposals claim one original, preserve that ambiguity.
    claims = Counter(i for indices, _ in proposals for i in indices)
    accepted = [(indices, merged) for indices, merged in proposals
                if all(claims[i] == 1 for i in indices)]
    used = {i for indices, _ in accepted for i in indices}
    result = [r for i, r in enumerate(originals) if i not in used]
    result.extend(merged for _, merged in accepted)
    events = [dict(original_indices=list(indices), output_objects=1,
                   original_area=sum(int(mu.area(originals[i])) for i in indices),
                   output_areas=[int(mu.area(merged))]) for indices, merged in accepted]
    statuses['merged_object_groups'] += len(accepted)
    statuses['unchanged_conflicting_merge_proposals'] += len(proposals)-len(accepted)
    return result, events, statuses


def save(path, predictions):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['filament_id', 'segmentation_rle'])
        for image, values in sorted(predictions.items()):
            for number, rle in enumerate(values, 1):
                counts = rle['counts']
                writer.writerow([f'{image}_{number}', counts.decode() if isinstance(counts, bytes) else counts])


def matching_audit(truth, rows):
    pred = group_rows(rows)
    groups = defaultdict(list)
    for row in truth:
        groups[row.group_id].append(row)
    extra_gt = extra_pred = 0
    for values in groups.values():
        candidates = pred.get(values[0].image_id, [])
        if not candidates:
            continue
        hits = np.asarray(mu.iou([r.rle(2048, 2048) for r in values],
            [r.rle(2048, 2048) for r in candidates], [0]*len(candidates))) > .5
        extra_gt += int(np.maximum(hits.sum(axis=1)-1, 0).sum())
        extra_pred += int(np.maximum(hits.sum(axis=0)-1, 0).sum())
    return dict(extra_matches_per_gt=extra_gt, extra_matches_per_prediction=extra_pred)


def main():
    started = time.monotonic()
    settings = plan()
    OUT.mkdir(exist_ok=False)
    (OUT/'predeclared_plan.json').write_text(json.dumps(settings, indent=2)+'\n')
    old = {Path(r['file_name']).stem: r for r in map(json.loads, OLD.open())}
    new = {Path(r['file_name']).stem: r for r in map(json.loads, NEW.open())}
    assert len(old) == len(new) == 91 and old.keys() == new.keys()
    assert all(old[k]['image_sha256'] == new[k]['image_sha256'] for k in old)
    rows = read_rows(BASE)
    assert len(rows) == 625 and not any(overlap_stats(rows).values())
    grouped = group_rows(rows)
    outputs = {setting['name']: {} for setting in settings}
    outputs['frozen_control'] = {}
    statuses = {setting['name']: Counter() for setting in settings}
    events = {setting['name']: [] for setting in settings}
    for number, image in enumerate(sorted(old), 1):
        originals = [r.rle(2048, 2048) for r in grouped.get(image, [])]
        outputs['frozen_control'][image] = originals
        union = mu.merge(originals) if originals else None
        for setting in settings:
            threshold = f"{setting['threshold']:g}"
            peers = [[c['rle'] for c in record['tta']['components'][threshold]
                      if c['area'] >= setting['minimum_area']] for record in (old[image], new[image])]
            function = split_masks if setting['operation'] == 'split' else merge_masks
            result, changed, counts = function(originals, *peers, setting)
            if union is not None:
                new_union = mu.merge(result)
                assert int(mu.area(new_union)) == int(mu.area(union)) == sum(int(mu.area(r)) for r in result)
                assert float(mu.iou([union], [new_union], [0])[0, 0]) == 1.
            else:
                assert not result
            outputs[setting['name']][image] = result
            statuses[setting['name']].update(counts)
            events[setting['name']].extend(dict(image=image, **change) for change in changed)
        if number % 20 == 0:
            print(f'Topology generated {number}/91', flush=True)
    for name, predictions in outputs.items():
        save(OUT/(name+'.csv'), predictions)
    # Label masks are first opened after all four variants and control are saved.
    from verify_native_training_results import GT, VAL
    groups = {r['image_id'] for r in csv.DictReader(VAL.open())}
    truth = read_rows(GT, gt=True, selected=groups)
    assert len(truth) == 708 and len(groups) == 101
    baseline = evaluate(truth, rows, 2048, 2048)
    assert abs(baseline['pq']-.47259304445299505) < 1e-12
    before = matched_ids(truth, rows)
    reports = []
    for name in outputs:
        path = OUT/(name+'.csv')
        reloaded = read_rows(path)
        score = evaluate(truth, reloaded, 2048, 2048)
        audit = overlap_stats(reloaded)
        extra = matching_audit(truth, reloaded)
        assert not any(audit.values()) and not any(extra.values())
        matched = matched_ids(truth, reloaded)
        sites = site_metrics(score)
        reports.append(dict(name=name,
            **{k:score[k] for k in ('pq', 'sq', 'rq', 'tp_pairs', 'fp', 'fn')},
            physical_masks=len(reloaded), delta_pq=score['pq']-baseline['pq'],
            recovered_gt=len(matched-before), lost_gt=len(before-matched),
            same_foreground_union_all91=True, **audit, **extra,
            statuses=dict(statuses.get(name, {})), changed_events=events.get(name, []),
            **paired_delta(baseline, score), by_site=sites,
            sites_improved=sum(sites[s]['pq'] > site_metrics(baseline)[s]['pq']+1e-12 for s in sites),
            csv=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    report = dict(scope='Four predeclared two-native-model instance regroupings on reused91-image development; no external/CV/LB result',
        ground_truth_opened_after_all_csvs_saved=True, training_performed=False,
        new_neural_inference_performed=False, foreground_preserved_all_images_all_settings=True,
        validation=dict(physical_images=91, annotation_groups=101, gt_instances=708),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (BASE, OLD, NEW)},
        settings=settings, experiments=reports, best=max(reports, key=lambda r:r['pq']),
        adopted=False, elapsed_seconds=time.monotonic()-started)
    (ROOT/'docs/evidence/round10_joint_native_topology.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps([{k:r[k] for k in ('name', 'pq', 'sq', 'tp_pairs', 'fp', 'fn', 'recovered_gt', 'lost_gt', 'physical_masks', 'statuses')} for r in reports], indent=2))


if __name__ == '__main__':
    main()
