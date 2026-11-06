#!/usr/bin/env python3
"""Twenty fixed development comparisons using existing binary predictions.

No image-specific or GT-specific selection is performed. Area thresholds are
motivated by training GT geometry; all results remain same-set tuning results.
Merge groups require every pair's mask IoU to exceed the cutoff, avoiding
unbounded transitive single-link unions.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import time

import numpy as np
from pycocotools import mask as mu

from experiment_mask_postprocess import read_rows, report, group_rows
from filament_eval import Filament


ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path('/workspace/artifacts/validation_results_20261027T112905Z_947c4b1f')
GT = Path('/workspace/artifacts/solar_filament_gt_v1/gt.csv')
VAL = Path('/workspace/attachments/bcb64220-b208-4cd7-9c3f-3f2be0f04167/val_split.csv')


def merge_cliques(rows: list[Filament], threshold: float, minimum_area: int):
    output, stats = [], {'merged_groups': 0, 'source_masks_in_merged_groups': 0,
                         'maximum_merge_size': 1, 'cross_cluster_edges_not_merged': 0}
    for image, members in group_rows(rows).items():
        rles = [r.rle(2048, 2048) for r in members]
        iou = np.asarray(mu.iou(rles, rles, [0] * len(rles)))
        groups = []
        assignment = {}
        for index in range(len(members)):
            selected = next((j for j, indices in enumerate(groups)
                             if all(iou[index, other] > threshold for other in indices)), None)
            if selected is None:
                selected = len(groups)
                groups.append([])
            groups[selected].append(index)
            assignment[index] = selected
        stats['cross_cluster_edges_not_merged'] += sum(
            iou[i, j] > threshold and assignment[i] != assignment[j]
            for i in range(len(members)) for j in range(i))
        for indices in groups:
            if len(indices) > 1:
                stats['merged_groups'] += 1
                stats['source_masks_in_merged_groups'] += len(indices)
                stats['maximum_merge_size'] = max(stats['maximum_merge_size'], len(indices))
            rle = mu.merge([rles[i] for i in indices], intersect=False)
            area = int(mu.area(rle))
            if area >= minimum_area:
                output.append(Filament(f'{image}_{sum(r.image_id == image for r in output)+1}',
                                       image, image, rle['counts'].decode('ascii'), area))
    stats = {k: int(v) for k, v in stats.items()}
    return output, stats


def main():
    started = time.monotonic()
    with VAL.open(newline='') as handle:
        groups = {r['image_id'] for r in csv.DictReader(handle)}
    truth = read_rows(GT, gt=True, selected=groups)
    assert len(truth) == 708 and len(groups) == 101
    reports, artifacts = [], {}
    for mode in ['union_cc', 'preserve']:
        for confidence in ['0.15', '0.25']:
            prefix = f'{mode}_conf{confidence}'
            source = read_rows(RESULTS/f'{prefix}.csv')
            settings = [(prefix, 'Unmodified reference', source, {})]
            for area in [100, 200, 300]:
                settings.append((f'{prefix}_area{area}', f'Keep source instances with area >= {area}',
                                 [row for row in source if row.area >= area], {}))
            if mode == 'preserve':
                for cutoff, area in [(.3, 100), (.5, 200)]:
                    merged, merge_stats = merge_cliques(source, cutoff, area)
                    settings.append((f'{prefix}_cliqueIoU{cutoff}_area{area}',
                        f'Input-order complete-link mask IoU > {cutoff} cliques; union each clique; area >= {area}; no transitive-only merge',
                        merged, merge_stats))
            for name, rule, predicted, merge_stats in settings:
                result = report(truth, predicted, name, rule)
                result['merge_statistics'] = merge_stats
                reports.append(result)
                artifacts[name] = predicted
                print(name, f'PQ={result["pq"]:.8f}', f'FP={result["fp"]}',
                      f'FN={result["fn"]}', f'overlap_images={result["overlap_images"]}', flush=True)
    assert len(reports) == 20
    refs = [r for r in reports if r['rule'] == 'Unmodified reference']
    best_any_ref = max(r['pq'] for r in refs)
    best_clean_ref = max(r['pq'] for r in refs if r['overlap_images'] == 0)
    candidates = []
    for key, eligible, target in [('best_any', reports, best_any_ref),
                                  ('best_nonoverlap', [r for r in reports if r['overlap_images'] == 0], best_clean_ref)]:
        best = max(eligible, key=lambda r: r['pq'])
        if best['pq'] > target:
            candidates.append((key, best))
    saved = []
    if candidates:
        out = Path('/workspace/artifacts/object_filter_candidates_v1')
        out.mkdir(parents=True, exist_ok=False)
        for kind, result in candidates:
            path = out/f'{kind}_{result["name"]}.csv'
            with path.open('x', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['filament_id', 'segmentation_rle'])
                writer.writerows((r.filament_id, r.counts) for r in artifacts[result['name']])
            saved.append({'kind': kind, 'name': result['name'], 'pq': result['pq'], 'path': str(path),
                          'scope': 'Validation predictions only, not a test submission'})
    output = {'scope': 'Fixed-rule development tuning on 91 images, not independent CV or leaderboard performance',
              'physical_images': 91, 'annotation_groups': 101, 'gt_instances': 708,
              'comparisons': 20, 'changed_settings': 16, 'unmodified_references': 4,
              'selection_policy': 'Area-only instance filtering and complete-link overlap unions; no GT-dependent per-mask choice or deliberate duplicate addition',
              'merge_order': 'Original CSV input order, not asserted to be confidence order',
              'experiments': reports, 'saved_candidates': saved,
              'best_any': max(reports, key=lambda r: r['pq'])['name'],
              'best_nonoverlap': max((r for r in reports if r['overlap_images'] == 0), key=lambda r: r['pq'])['name'],
              'elapsed_seconds': time.monotonic()-started}
    path = ROOT/'docs/evidence/object_filter_experiments.json'
    path.write_text(json.dumps(output, indent=2)+'\n')
    print('REPORT', path)


if __name__ == '__main__':
    main()
