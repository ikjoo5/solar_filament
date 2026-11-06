"""Read-only PQ error-budget and raw-candidate diagnosis on reused validation.

No GT-derived prediction file is produced. Arithmetic counterfactuals are not
achieved scores, independent holdout estimates, or leaderboard forecasts.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

from analyze_round8_generalization import RAW
from experiment_mask_postprocess import read_rows
from filament_eval import evaluate

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'docs/evidence'
BASE = Path('/workspace/artifacts/rich_boundary_composition_v1/tta_precision.csv')
OLD = Path('/workspace/artifacts/best84_rich_predictions_tta/predictions.jsonl')
NEW = Path('/workspace/artifacts/context_only_uploaded_20261029/multiscale_context/raw_paper_views.jsonl')
GT = Path('/workspace/artifacts/solar_filament_gt_v1/gt.csv')
VAL = Path('/workspace/attachments/bcb64220-b208-4cd7-9c3f-3f2be0f04167/val_split.csv')


def hypothetical(tp, fp, fn, iou_sum):
    denominator = tp + .5 * fp + .5 * fn
    return dict(tp=tp, fp=fp, fn=fn, matched_iou_sum=iou_sum,
                pq=iou_sum / denominator, sq=iou_sum / tp, rq=tp / denominator)


def error_budget(score):
    tp, fp, fn = (score[key] for key in ('tp_pairs', 'fp', 'fn'))
    sq, iou = score['sq'], score['matched_iou_sum']
    return dict(
        interpretation='Counterfactual arithmetic only; no GT oracle is deployed. Each row changes only its named quantities.',
        current_SQ_limit_note='PQ cannot exceed SQ because RQ<=1. The current SQ0.6779 is a ceiling only while that average matched-IoU is held fixed, not a ceiling for future models, changed predictions or the competition.',
        remove_all_FP_only=hypothetical(tp, 0, fn, iou),
        perfect_boundaries_only_same_matching_counts=hypothetical(tp, fp, fn, float(tp)),
        recover_all_FN_at_current_SQ_keep_FP=hypothetical(tp+fn, fp, 0, iou+fn*sq),
        recover_all_FN_at_perfect_IoU_keep_FP=hypothetical(tp+fn, fp, 0, iou+fn),
        perfect_detection_counts_at_current_SQ=hypothetical(tp+fn, 0, 0, (tp+fn)*sq),
        remove_FP_and_recover_FN_at_perfect_IoU=hypothetical(tp+fn, 0, 0, iou+fn),
        target_070_example_budgets=[hypothetical(650, 34, 58, 650*.75),
                                    hypothetical(620, 89, 88, 620*.8)],
        add_one_prediction_break_even=dict(
            assumptions='Existing disjoint masks and matches remain unchanged; one new prediction per annotation group either uniquely recovers one FN or is one FP.',
            required_expected_matched_IoU_per_added_prediction=score['pq']/2,
            required_precision_at_current_SQ=score['pq']/(2*sq)))


def candidate_availability(truth, frozen):
    sources = {name: defaultdict(dict) for name in ('yolo_all_views', 'best84_all_thresholds', 'context45_all_thresholds')}
    def add(family, image, counts):
        value = counts.encode() if isinstance(counts, str) else counts
        sources[family][image][value] = dict(size=[2048, 2048], counts=value)
    for path in RAW:
        for line in path.open():
            record = json.loads(line)
            for instance in record['instances']:
                add('yolo_all_views', record['image'], instance['rle'])
    for family, path in [('best84_all_thresholds', OLD), ('context45_all_thresholds', NEW)]:
        for line in path.open():
            record = json.loads(line)
            image = Path(record['file_name']).stem
            for branch in (record, record['tta']):
                for components in branch['components'].values():
                    for component in components:
                        add(family, image, component['rle']['counts'])
    groups = defaultdict(list)
    for row in truth:
        groups[row.group_id].append(row)
    misses = {r['id'] for r in frozen['gt_records'] if not r['matched']}
    summary = {name: Counter() for name in (*sources, 'all_prior', 'all_with_context')}
    new_only = []
    total_cover = Counter()
    for group, rows in sorted(groups.items()):
        image = rows[0].image_id
        gt = [r.rle(2048, 2048) for r in rows]
        best = {}
        for family, by_image in sources.items():
            predictions = list(by_image[image].values())
            best[family] = (np.asarray(mu.iou(gt, predictions, [0]*len(predictions))).max(axis=1)
                            if predictions else np.zeros(len(rows)))
        best['all_prior'] = np.maximum(best['yolo_all_views'], best['best84_all_thresholds'])
        best['all_with_context'] = np.maximum(best['all_prior'], best['context45_all_thresholds'])
        for index, row in enumerate(rows):
            for family, values in best.items():
                total_cover[family] += int(values[index] > .5)
                if row.filament_id in misses:
                    value = values[index]
                    summary[family]['IoU_gt_0.5' if value > .5 else 'IoU_0.3_to_0.5' if value >= .3 else 'IoU_lt_0.3'] += 1
            if row.filament_id in misses and best['all_prior'][index] <= .5 < best['all_with_context'][index]:
                new_only.append(row.filament_id)
    return dict(
        interpretation='GT-aware candidate-availability diagnosis, never an attainable recall selector or measured ensemble PQ. Candidates can conflict with each other and each physical image has one shared output for all annotators.',
        baseline_FN=len(misses),
        unique_cached_masks_by_family={family: sum(len(v) for v in images.values()) for family, images in sources.items()},
        FN_availability={key: dict(value) for key, value in summary.items()},
        all_GT_with_some_candidate_IoU_gt_half=dict(total_cover),
        new_context_only_FN_opportunities=len(new_only), new_context_only_FN_ids=new_only,
        thresholds=dict(best84=[.2,.35,.5,.65,.8], context45=[.35,.5,.65]))


def main():
    prior = json.loads((EVIDENCE/'round8_generalization_audit.json').read_text())
    frozen = prior['runs']['frozen_highest']
    groups = {row['image_id'] for row in csv.DictReader(VAL.open())}
    truth = read_rows(GT, gt=True, selected=groups)
    score = evaluate(truth, read_rows(BASE), 2048, 2048)
    assert len(truth) == 708 and len(groups) == 101
    assert abs(score['pq']-frozen['pq']) < 1e-12
    availability = candidate_availability(truth, frozen)
    assert availability['FN_availability']['all_prior']['IoU_gt_0.5'] == 178
    context = json.loads((EVIDENCE/'context_uploaded_audit.json').read_text())
    rescue = json.loads((EVIDENCE/'context_consensus_rescue.json').read_text())
    rescue_row = next(r for r in rescue['experiments'] if r['name'] == 'new_t50')
    extra_tp, extra_fp = rescue_row['tp_pairs']-score['tp_pairs'], rescue_row['fp']-score['fp']
    after_iou = rescue_row['pq']*(rescue_row['tp_pairs']+.5*rescue_row['fp']+.5*rescue_row['fn'])
    native_gain = dict(
        measured_rule='Previous fixed new_t50 old+new native consensus admission',
        recovered_GT=extra_tp, extra_FP=extra_fp,
        precision_of_additions_across_annotation_groups=extra_tp/(extra_tp+extra_fp),
        expected_matched_IoU_per_addition=(after_iou-score['matched_iou_sum'])/(extra_tp+extra_fp),
        matched_IoU_mean_of_recovered_GT=(after_iou-score['matched_iou_sum'])/extra_tp,
        pq_before=score['pq'], pq_after=rescue_row['pq'],
        original_masks_preserved=True,
        note='A low FN count alone does not imply improved PQ; agreement between related native models did not establish sufficient addition precision.')
    fn_records = [r for r in frozen['gt_records'] if not r['matched']]
    training = prior['training_annotations']
    report = dict(
        scope='Existing-data read-only diagnosis; no external data, fitting, new predictions or production changes.',
        target_pq=.7, target_scope='User goal; local development and public leaderboard remain separate metrics.',
        measured_baseline={k:score[k] for k in ('pq','sq','rq','tp_pairs','fp','fn','matched_iou_sum')},
        dataset=dict(model_train_physical_images=616, synthetic_train_physical_images=462,
            validation_physical_images=91, validation_annotation_groups=101, validation_GT=708),
        error_budget=error_budget(score),
        frozen_FN=dict(total=220, best_IoU_bins=frozen['unmatched_gt_best_iou_bins'],
            by_train_area_quartile=dict(Counter(r['area_quartile'] for r in fn_records)),
            train_area_quartile_edges=prior['train_only_area_quartile_edges'],
            by_observatory=dict(Counter(r['site'] for r in fn_records)),
            by_annotation_category=dict(Counter(r['category'] for r in fn_records)),
            union_foreground_coverage_ge_80pct=frozen['unmatched_gt_with_foreground_union_coverage_ge_80pct'],
            overlap_graph=frozen['overlap_graph'],
            interpretation='48 missed objects have >=80% union-foreground coverage, so some failures are boundaries/instance grouping; this count does not prove all48 can be fixed by splitting.'),
        raw_candidate_availability=availability,
        observed_native_consensus_tradeoff=native_gain,
        context_model=dict(best_epoch=context['history']['best_epoch'], completed_epochs=context['history']['epochs_completed'],
            independently_rescored_best=context['native_best'],
            last_epoch_history_only=context['history']['last_validation'],
            last_epoch_predictions_available=False),
        training_annotation_variability=dict(
            physical_images=training['train_physical_images'],
            physical_images_with_multiple_annotation_sets=sum(int(v) for k,v in training['variants_per_physical_image'].items() if int(k)>1),
            variants_per_physical_image=training['variants_per_physical_image'],
            median_pair_foreground_union_IoU=training['foreground_union_iou_quantiles']['0.5'],
            pairwise_object_observations=training['paired_objects_total'],
            pairwise_unmatched_object_observations=training['unmatched_objects_total'],
            note='Pairwise counts repeat objects across annotator pairs and are not unique object counts. Alternative annotations are not declared wrong; a single image can supply contradictory binary targets.',
            current_training_implementation='native_mixed_data.py sample_record chooses one annotation variant uniformly for a selected physical image; mixed_segmentation_loss uses binary foreground BCE+Dice.'),
        recommended_experiment_priorities=[
            dict(priority=1, direction='Joint boundary and instance grouping refinement with fixed cross-model geometric support',
                rationale='Boundary-only perfect-IoU counterfactual is still below0.7 at current matching counts;48FN already have80% union coverage. Compare a small fixed family before enlarging threshold searches.'),
            dict(priority=2, direction='Improve proposal quality/ranking before broader low-confidence rescue',
                rationale='178 of220FN already have a cached matching candidate, but actual consensus rescue added22FP for6TP. A raw availability oracle is not a usable selector. Any learned selector must train on training-only out-of-fold predictions, not these91 validation labels.'),
            dict(priority=3, direction='Short train-only pilot for annotation-aware foreground supervision and boundary precision',
                rationale='287/616 training images have alternate annotation sets, with median pairwise foreground IoU0.470. Retain separate annotations or explicit uncertainty; avoid a naive union that may increase FP. Compare against fixedbest84 without retraining a second baseline.'),
            dict(priority=4, direction='Inference-only comparison of saved epoch60 before additional long training',
                rationale='History records FN228 vs best45FN234 but last masks are not available here; effect in the full pipeline must be measured, not inferred from history.'),
        ],
        generalization=prior['temporal_independence'],
        limitations=['All91 images were used repeatedly for development; conditional bootstrap is not independent confirmation.',
                    'No .7 local or leaderboard result has been achieved; arithmetic examples are not performance promises.',
                    'Public LB remains user-reported0.39; a local score increase need not transfer.',
                    'No weights are fitted to validation or external annotations.'],
        verification=dict(baseline_rescored_from_saved_csv=True,
            baseline_tp_plus_fn_equals_GT=score['tp_pairs']+score['fn']==len(truth),
            FN_family_counts_sum_to_total=all(sum(values.values())==len(fn_records)
                for values in availability['FN_availability'].values()),
            no_GT_oracle_prediction_exported=True),
        sources_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (BASE,OLD,NEW,GT,VAL,*RAW,EVIDENCE/'round8_generalization_audit.json',
             EVIDENCE/'context_uploaded_audit.json',EVIDENCE/'context_consensus_rescue.json')})
    output=EVIDENCE/'round10_error_budget.json'
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('measured_baseline','error_budget','raw_candidate_availability','observed_native_consensus_tradeoff')},indent=2))


if __name__ == '__main__':
    main()
