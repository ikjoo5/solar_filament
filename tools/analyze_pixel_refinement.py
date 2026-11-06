"""Explain the fixed full-validation pixel-refinement results and crossings.

GT chooses diagnostic examples only. It never selects per-image/object outputs.
The highest actual refinement is compared even if retaining baseline is better.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from pycocotools import mask as mu

from experiment_mask_postprocess import group_rows, read_rows
from experiment_view_combinations import ROOT, GT, VAL, SIZE
from filament_eval import evaluate
from refine_masks_with_pixels import TilePixels


RESULTS = [Path('/workspace/artifacts/pixel_refinement_full91_rescue_v1'),
           Path('/workspace/artifacts/pixel_refinement_full91_smooth_v1')]
BASELINE = Path('/workspace/artifacts/boundary_smoothing_candidates_v2/gaussian_s1.2_t0.6.csv')
PIXELS = Path('/workspace/artifacts/pixel_tiles_20261027T203313Z_fd420182')
MANIFEST = Path('/workspace/artifacts/solar_filament_gt_v1/manifest.json')
OVERLAYS = Path('/workspace/artifacts/pixel_refinement_diagnostic_overlays_v1')


def compact(score):
    return {k: score[k] for k in ('pq', 'sq', 'rq', 'tp_pairs', 'fp', 'fn', 'matched_iou_sum')}


def area_bin(area):
    if area < 500:
        return '<500px'
    if area < 1500:
        return '500-1499px'
    if area < 5000:
        return '1500-4999px'
    return '>=5000px'


def overlay_case(case, gt, base, refined, store, path):
    gray, coverage = store.load(case['image_id'])
    masks = [mu.decode(row.rle(SIZE, SIZE)).astype(bool) if row else np.zeros((SIZE, SIZE), bool)
             for row in (gt, base, refined)]
    union = np.logical_or.reduce(masks)
    ys, xs = np.nonzero(union)
    y1, y2 = max(0, int(ys.min())-18), min(SIZE, int(ys.max())+19)
    x1, x2 = max(0, int(xs.min())-18), min(SIZE, int(xs.max())+19)
    region = np.s_[y1:y2, x1:x2]
    if not coverage[region].all():
        return None
    masks = [mask[region] for mask in masks]
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ['Native limb pixels', f"Baseline IoU {case['baseline_best_iou']:.4f}",
              f"Pixel refinement IoU {case['refined_best_iou']:.4f}"]
    for index, axis in enumerate(axes):
        axis.imshow(gray[region], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        axis.contour(masks[0].astype(np.uint8), levels=[.5], colors=['#30ef60'], linewidths=1.1)
        if index:
            axis.contour(masks[index].astype(np.uint8), levels=[.5],
                         colors=['#229bff' if index == 1 else '#ff4a43'], linewidths=1.1)
        axis.set_title(titles[index], fontsize=10)
        axis.set_xticks([]); axis.set_yticks([])
    figure.suptitle(f"{case['transition']} | {case['filament_id']} | {case['category']}\n"
                   'Diagnostic example selected using GT; global inference rule unchanged', fontsize=11)
    figure.legend(handles=[Line2D([0], [0], color=c, label=t) for c, t in
                          [('#30ef60', 'GT'), ('#229bff', 'Baseline'), ('#ff4a43', 'Refined')]],
                  loc='lower center', ncol=3, fontsize=9)
    figure.subplots_adjust(left=.025, right=.975, bottom=.15, top=.75, wspace=.12)
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return dict(path=str(path), x=x1, y=y1, width=x2-x1, height=y2-y1,
                full_original_pixel_coverage=True, **case)


def main():
    summaries = [(folder, json.loads((folder/'summary.json').read_text())) for folder in RESULTS]
    all_settings = [dict(source=str(folder), **row) for folder, summary in summaries
                    for row in summary['experiments']]
    overall = max(all_settings, key=lambda row: row['pq'])
    actual = max([row for row in all_settings if row['method'] is not None], key=lambda row: row['pq'])
    refined_path = Path(actual['source'])/(actual['name']+'.csv')
    with VAL.open(newline='') as handle:
        selection = list(csv.DictReader(handle))
    groups = {row['image_id'] for row in selection}
    truth = read_rows(GT, gt=True, selected=groups)
    baseline = read_rows(BASELINE)
    refined = read_rows(refined_path)
    before = evaluate(truth, baseline, SIZE, SIZE)
    after = evaluate(truth, refined, SIZE, SIZE)
    source_before = evaluate(truth, read_rows(Path(actual['source'])/'baseline.csv'), SIZE, SIZE)
    assert abs(before['pq']-.4607591314345197) < 1e-12
    assert abs(after['pq']-actual['pq']) < 1e-12
    assert len(truth) == 708 and len(groups) == 101 and len(baseline) == len(refined) == 626
    meta = {a['filament_id']: a for a in json.loads(MANIFEST.read_text())['annotations']}
    by_group = defaultdict(list)
    for row in truth:
        by_group[row.group_id].append(row)
    base_images, ref_images = group_rows(baseline), group_rows(refined)
    cases = []
    for group, truths in sorted(by_group.items()):
        image = truths[0].image_id
        predictions = [base_images.get(image, []), ref_images.get(image, [])]
        matrices = [np.asarray(mu.iou([row.rle(SIZE, SIZE) for row in truths],
                    [row.rle(SIZE, SIZE) for row in rows], [0]*len(rows)))
                    if rows else np.zeros((len(truths), 0)) for rows in predictions]
        for index, row in enumerate(truths):
            maxima, best_ids, sums = [], [], []
            for matrix, rows in zip(matrices, predictions):
                values = matrix[index]
                maxima.append(float(values.max()) if len(values) else 0.)
                best_ids.append(rows[int(values.argmax())].filament_id if len(values) else None)
                sums.append(float(values[values > .5].sum()))
            old, new = maxima
            transition = ('retained' if new > .5 else 'lost') if old > .5 else (
                         'recovered' if new > .5 else 'still_missed')
            cases.append(dict(filament_id=row.filament_id, image_id=image, group=group,
                category=meta[row.filament_id]['category_name'], area=row.area, area_bin=area_bin(row.area),
                baseline_best_iou=old, refined_best_iou=new, iou_delta=new-old,
                baseline_best_prediction=best_ids[0], refined_best_prediction=best_ids[1],
                baseline_qualifying_iou_sum=sums[0], refined_qualifying_iou_sum=sums[1],
                qualifying_iou_sum_delta=sums[1]-sums[0], transition=transition,
                near_matching_threshold=(.45 <= old <= .55 or .45 <= new <= .55)))
    counts = Counter(case['transition'] for case in cases)
    assert counts['recovered']-counts['lost'] == before['fn']-after['fn']
    assert abs(sum(c['qualifying_iou_sum_delta'] for c in cases)
               -(after['matched_iou_sum']-before['matched_iou_sum'])) < 1e-9
    breakdowns = {}
    for field in ('category', 'area_bin'):
        breakdowns[field] = []
        for value in sorted({c[field] for c in cases}):
            subset = [case for case in cases if case[field] == value]
            breakdowns[field].append(dict(value=value, gt_instances=len(subset),
                transitions=dict(Counter(c['transition'] for c in subset)),
                baseline_matched=sum(c['baseline_best_iou'] > .5 for c in subset),
                refined_matched=sum(c['refined_best_iou'] > .5 for c in subset),
                matched_iou_sum_delta=sum(c['qualifying_iou_sum_delta'] for c in subset)))
    OVERLAYS.mkdir(parents=True, exist_ok=False)
    store = TilePixels(PIXELS)
    maps = [{r.filament_id: r for r in rows} for rows in (truth, baseline, refined)]
    chosen, used = [], set()
    for transition in ('recovered', 'lost'):
        pool = sorted((c for c in cases if c['transition'] == transition),
                      key=lambda c: (min(abs(c['baseline_best_iou']-.5),
                                         abs(c['refined_best_iou']-.5)), c['filament_id']))
        added = 0
        for case in pool:
            if case['image_id'] in used:
                continue
            path = OVERLAYS/f'{transition}_{added+1}_{case["filament_id"]}.png'
            value = overlay_case(case, maps[0][case['filament_id']],
                maps[1].get(case['baseline_best_prediction']), maps[2].get(case['refined_best_prediction']),
                store, path)
            if value:
                chosen.append(value); used.add(case['image_id']); added += 1
            if added == 2:
                break
    numerator_changes = {transition: sum(c['qualifying_iou_sum_delta'] for c in cases
                                        if c['transition'] == transition) for transition in counts}
    report = dict(scope='Full user-supplied native pixels; fixed global rule comparisons on the same91-image development set; not independent CV or Kaggle LB',
        diagnostic_only='GT selects explanations/overlays and matching transitions only, never per-image/object inference output',
        physical_images=91, annotation_groups=len(groups), gt_instances=len(truth),
        settings_evaluated=len(all_settings), best_overall_fixed_setting=overall,
        best_actual_pixel_refinement=actual, overall_best_remains_baseline=overall['method'] is None,
        baseline_path=str(BASELINE), actual_refinement_path=str(refined_path),
        source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in (BASELINE, refined_path, PIXELS/'manifest.json')},
        baseline_score=compact(before), actual_refinement_score=compact(after),
        same_source_baseline_score=compact(source_before),
        same_source_delta={key: after[key]-source_before[key] for key in compact(source_before)},
        delta={key: after[key]-before[key] for key in compact(before)},
        transitions=dict(counts), matched_iou_sum_delta_by_transition=numerator_changes,
        near_half_definition='Either before or after maximum GT IoU is in [.45,.55]; strict match requires IoU>.5',
        near_half_transitions=dict(Counter(c['transition'] for c in cases if c['near_matching_threshold'])),
        breakdowns=breakdowns, overlays=chosen, per_gt=cases)
    (ROOT/'docs/evidence/pixel_refinement_diagnostics.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('best_overall_fixed_setting', 'best_actual_pixel_refinement',
                      'baseline_score', 'actual_refinement_score', 'delta', 'transitions', 'matched_iou_sum_delta_by_transition')}, indent=2))
    print('overlays', len(chosen), str(OVERLAYS))


if __name__ == '__main__':
    main()
