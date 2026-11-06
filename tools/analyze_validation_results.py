#!/usr/bin/env python3
"""Diagnose existing validation RLEs without images or new inference.

Strata recall counts unique GT with any IoU>.5 hit; the supplied notebook's
TP counts all qualifying pairs. These quantities differ for overlapping masks.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mu


def read_rows(path, gt=False, allowed_groups=None):
    result = defaultdict(list)
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            group = row['filament_id'].split('_', 1)[0]
            if allowed_groups is not None and group not in allowed_groups:
                continue
            rle = {'size': [2048, 2048], 'counts': row['segmentation_rle'].encode('ascii')}
            result[group].append({'id': row['filament_id'], 'rle': rle, 'area': int(mu.area(rle))})
    return result


def bins(values):
    a = np.asarray(values, dtype=float)
    return {'exact_zero': int((a == 0).sum()), '(0,.1)': int(((a > 0) & (a < .1)).sum()),
            '[.1,.3)': int(((a >= .1) & (a < .3)).sum()),
            '[.3,.4)': int(((a >= .3) & (a < .4)).sum()),
            '[.4,.5]': int(((a >= .4) & (a <= .5)).sum()),
            '>.5': int((a > .5).sum())}


def quantiles(values):
    if not len(values):
        return None
    return {str(q): float(np.quantile(values, q)) for q in [0, .1, .25, .5, .75, .9, 1]}


def recall_strata(records, key):
    grouped = defaultdict(list)
    for r in records:
        grouped[str(r[key])].append(r)
    return {k: {'gt': len(rs), 'unique_matched_gt': sum(r['matched'] for r in rs),
                'unique_gt_recall': sum(r['matched'] for r in rs)/len(rs),
                'best_iou_mean': float(np.mean([r['best_iou'] for r in rs]))}
            for k, rs in sorted(grouped.items())}


def analyze(pred_path, truth, ann_meta, area_edges):
    pred = read_rows(pred_path)
    records, fp_ious, pairs, group_results = [], [], [], []
    tp = fp = fn = 0
    total_iou = 0.
    pixel_intersection = pixel_gt = pixel_pred = 0
    duplicate_gt_pairs = duplicate_pred_pairs = 0
    multi_gt_rows = multi_pred_columns = 0
    split_rows = merge_cols = 0
    for group, gt in truth.items():
        image_id = group.split('-', 1)[1]
        ps = pred.get(image_id, [])
        gr, pr = [r['rle'] for r in gt], [r['rle'] for r in ps]
        ga, pa = np.asarray([r['area'] for r in gt]), np.asarray([r['area'] for r in ps])
        iou = np.asarray(mu.iou(gr, pr, [0]*len(pr))) if pr else np.zeros((len(gr), 0))
        hit = iou > .5
        n_tp, n_fp, n_fn = int(hit.sum()), int((~hit.any(axis=0)).sum()), int((~hit.any(axis=1)).sum())
        tp += n_tp; fp += n_fp; fn += n_fn; total_iou += float(iou[hit].sum())
        duplicate_gt_pairs += n_tp - int(hit.any(axis=1).sum())
        duplicate_pred_pairs += n_tp - int(hit.any(axis=0).sum())
        multi_gt_rows += int((hit.sum(axis=1) > 1).sum())
        multi_pred_columns += int((hit.sum(axis=0) > 1).sum())
        if pr:
            inter = iou*(ga[:, None]+pa[None, :])/(1+iou)
            graph = (inter/ga[:, None] >= .05) & (inter/pa[None, :] >= .05)
            split_rows += int((graph.sum(axis=1) > 1).sum())
            merge_cols += int((graph.sum(axis=0) > 1).sum())
            union_pred = mu.merge(pr)
            union_gt = mu.merge(gr)
            pixel_intersection += int(mu.area(mu.merge([union_gt, union_pred], intersect=True)))
            pixel_pred += int(mu.area(union_pred))
            for g, p in zip(*np.where(hit)):
                pairs.append({'precision': float(inter[g, p]/pa[p]), 'recall': float(inter[g, p]/ga[g]),
                              'pred_to_gt_area_ratio': float(pa[p]/ga[g])})
            best_gt = iou.max(axis=1)
            best_pred = iou.max(axis=0)
            fp_ious.extend(best_pred[~hit.any(axis=0)].tolist())
        else:
            best_gt = np.zeros(len(gt))
        pixel_gt += int(mu.area(mu.merge(gr)))
        for j, row in enumerate(gt):
            coverage = int(mu.area(mu.merge([row['rle'], union_pred], intersect=True)))/row['area'] if pr else 0.
            meta = ann_meta[row['id']]
            records.append({'id': row['id'], 'group': group, 'image': image_id,
                            'site': image_id[-2:], 'category': meta['category_name'],
                            'area': row['area'], 'area_quartile': int(np.searchsorted(area_edges, row['area'], side='right'))+1,
                            'matched': bool(hit[j].any()), 'best_iou': float(best_gt[j]),
                            'foreground_union_coverage': coverage})
        den = n_tp+.5*n_fp+.5*n_fn
        group_results.append({'group': group, 'tp': n_tp, 'fp': n_fp, 'fn': n_fn,
                              'iou_sum': float(iou[hit].sum()),
                              'pq': float(iou[hit].sum())/den if den else 0.})
    missed = [r for r in records if not r['matched']]
    site_pq = {}
    for site in sorted({r['site'] for r in records}):
        gs = [r for r in group_results if r['group'].endswith(site)]
        st, sfp, sfn = (sum(r[k] for r in gs) for k in ['tp', 'fp', 'fn'])
        denom = st+.5*sfp+.5*sfn
        site_pq[site] = {'groups': len(gs), 'tp_pairs': st, 'fp': sfp, 'fn': sfn,
                         'pq': sum(r['iou_sum'] for r in gs)/denom if denom else 0.}
    return {'prediction_csv': Path(pred_path).name, 'gt_instances': len(records),
            'tp_pairs': tp, 'fp': fp, 'fn': fn, 'unique_matched_gt': sum(r['matched'] for r in records),
            'pq': total_iou/(tp+.5*fp+.5*fn), 'gt_best_iou_bins': bins([r['best_iou'] for r in records]),
            'unmatched_gt_best_iou_bins': bins([r['best_iou'] for r in missed]),
            'fp_best_iou_bins': bins(fp_ious),
            'extra_tp_pairs_beyond_unique_matched_gt': duplicate_gt_pairs,
            'extra_tp_pairs_beyond_unique_matched_pred_per_group': duplicate_pred_pairs,
            'gt_with_multiple_qualifying_predictions': multi_gt_rows,
            'predictions_matching_multiple_gt_per_group': multi_pred_columns,
            'overlap_graph': {'definition': 'intersection/GT_area>=.05 AND intersection/pred_area>=.05; diagnostic, not causal split/merge labels',
                              'gt_with_two_or_more_pred_neighbors': split_rows,
                              'pred_with_two_or_more_gt_neighbors': merge_cols},
            'matched_pair_precision': quantiles([r['precision'] for r in pairs]),
            'matched_pair_recall': quantiles([r['recall'] for r in pairs]),
            'matched_pair_pred_to_gt_area_ratio': quantiles([r['pred_to_gt_area_ratio'] for r in pairs]),
            'unmatched_gt_with_foreground_union_coverage_ge_80pct': sum(r['foreground_union_coverage'] >= .8 for r in missed),
            'foreground_union_micro_precision': pixel_intersection/pixel_pred if pixel_pred else 0.,
            'foreground_union_micro_recall': pixel_intersection/pixel_gt if pixel_gt else 0.,
            'recall_by_area_quartile': recall_strata(records, 'area_quartile'),
            'recall_by_category': recall_strata(records, 'category'),
            'recall_by_site': recall_strata(records, 'site'), 'global_pq_by_site': site_pq,
            'gt_records': records, 'group_results': group_results}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', required=True)
    p.add_argument('--gt', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--val-split', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    with Path(args.val_split).open(newline='', encoding='utf-8-sig') as f:
        val_groups = {r['image_id'] for r in csv.DictReader(f)}
    truth = read_rows(args.gt, gt=True, allowed_groups=val_groups)
    assert set(truth) == val_groups
    manifest = json.loads(Path(args.manifest).read_text())
    ann_meta = {a['filament_id']: a for a in manifest['annotations']}
    areas = [r['area'] for rows in truth.values() for r in rows]
    edges = np.quantile(areas, [.25, .5, .75])
    report = {'scope': 'development validation diagnostics; no independent leaderboard prediction',
              'aggregation': 'all-pairs strict IoU>.5 global PQ; unique-GT recall reported separately',
              'area_quartile_edges': edges.tolist(),
              'limitations': ['Annotation variants share physical images and are correlated.',
                             'CSV has binary masks but no original confidence values; no confidence reconstruction.',
                             'Category/area strata report GT recall, not category PQ with arbitrarily assigned FP.'],
              'runs': [analyze(Path(args.results)/name, truth, ann_meta, edges)
                       for name in ['union_cc_conf0.25.csv', 'preserve_conf0.15.csv']]}
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    for run in report['runs']:
        print(json.dumps({k:v for k,v in run.items() if k not in {'gt_records','group_results'}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
