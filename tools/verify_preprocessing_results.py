"""Independently verify uploaded preprocessing results and the saved rescue CSV.

Only compressed RLE operations are needed; this performs no model inference.
The validation set is the same development set used for parameter selection.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

from check_validation_results import load
from filament_eval import evaluate


ROOT = Path(__file__).resolve().parents[1]
OLD = Path('/workspace/artifacts/validation_results_20261027T120958Z_444da7f4')
NEW = Path('/workspace/artifacts/validation_results_20261027T125130Z_2312e759')
VAL = Path('/workspace/attachments/bcb64220-b208-4cd7-9c3f-3f2be0f04167/val_split.csv')
SOURCES = {
    'MAGFiLO_1.0_Annotations_kaggle2026_train.json': Path('/workspace/datasets/solar_filament/MAGFiLO_1.0_Annotations_kaggle2026_train.json'),
    'best.pt': Path('/workspace/attachments/f27dbac6-f4d7-4f71-bdea-2df7dc252775/best.pt'),
    'train_split.csv': Path('/workspace/attachments/1cca1607-f94f-4334-9586-c1fb5ab292d7/train_split.csv'),
    'val_split.csv': VAL,
}


def sha256(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def raw_records(path, images):
    records = {}
    masks = 0
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            key = row['image'], row['view']
            assert key not in records
            assert row['image'] in images
            assert row['shape'] == [2048, 2048]
            assert row['coordinate_space'] == 'original'
            assert not row['max_det_reached']
            records[key] = row
            masks += len(row['instances'])
    return records, {'records': len(records), 'masks': masks,
                     'sha256': sha256(path), 'max_det_capped_records': 0}


def overlap_check(prediction):
    """Count intersecting pairs directly from compressed RLEs."""
    groups = {}
    for row in prediction:
        groups.setdefault(row.image_id, []).append(row)
    overlap_images = overlap_pairs = duplicates = 0
    for rows in groups.values():
        encodings = [row.rle(2048, 2048) for row in rows]
        iou = np.asarray(mu.iou(encodings, encodings, [0] * len(rows)))
        overlaps = np.triu(iou > 0, 1)
        overlap_images += int(overlaps.any())
        overlap_pairs += int(overlaps.sum())
        duplicates += int(np.triu(iou == 1, 1).sum())
    return dict(overlap_images=overlap_images, overlap_mask_pairs=overlap_pairs,
                exact_duplicate_mask_pairs=duplicates)


def main():
    before = json.loads((OLD / 'summary.json').read_text())
    after = json.loads((NEW / 'summary.json').read_text())
    actual_hashes = {name: sha256(path) for name, path in SOURCES.items()}
    assert actual_hashes == before['source_sha256'] == after['source_sha256']
    assert before['versions'] == after['versions']
    with VAL.open(newline='', encoding='utf-8-sig') as f:
        selection = list(csv.DictReader(f))
    groups = {row['image_id'] for row in selection}
    images = {Path(row['file_name']).stem for row in selection}
    truth = load(Path('/workspace/artifacts/solar_filament_gt_v1/gt.csv'), True, groups)
    assert len(groups) == 101 and len(images) == 91 and len(truth) == 708
    comparisons = []
    assert len(after['comparisons']) == 68
    for given in after['comparisons']:
        prediction = load(NEW / given['prediction_csv'])
        actual = evaluate(truth, prediction, 2048, 2048)
        for reported, computed in [('PQ', 'pq'), ('SQ', 'sq'), ('RQ', 'rq'),
                                   ('TP', 'tp_pairs'), ('FP', 'fp'), ('FN', 'fn')]:
            assert abs(given[reported] - actual[computed]) < 1e-12, (given['prediction_csv'], reported)
        comparisons.append(dict(**given, prediction_rows=len(prediction),
                                unique_matched_gt=len(truth)-actual['fn'],
                                extra_matching_pairs=actual['tp_pairs']-(len(truth)-actual['fn'])))
    old, old_info = raw_records(OLD / 'raw_views.jsonl', images)
    new, new_info = raw_records(NEW / 'raw_preprocess_views.jsonl', images)
    assert len(old) == 91 * 4 and len(new) == 91 * 8
    for stem in images:
        assert old[stem, 'base']['instances'] == new[stem, 'limb']['instances']
    rescue = json.loads((ROOT / 'docs/evidence/gated_fn_rescue_experiments.json').read_text())
    candidate_path = Path(rescue['saved_candidate'])
    candidate = load(candidate_path)
    score = evaluate(truth, candidate, 2048, 2048)
    for field in ['pq', 'sq', 'rq', 'tp_pairs', 'fp', 'fn']:
        assert abs(rescue['best'][field] - score[field]) < 1e-12
    overlap = overlap_check(candidate)
    assert not any(overlap.values()) and len(candidate) == 626
    extra = score['tp_pairs'] - (len(truth) - score['fn'])
    assert extra == 0
    report = dict(
        scope='Same 91-image development set; no independent validation or leaderboard claim',
        physical_images=91, annotation_groups=101, gt_instances=708,
        source_summary_sha256=sha256(NEW / 'summary.json'),
        source_sha256=actual_hashes, all_input_source_hashes_match=True,
        software_versions_match_previous_upload=True, versions=after['versions'],
        all_68_reported_scores_recomputed=True, comparisons=comparisons,
        best_new_preprocessing_setting=max(comparisons, key=lambda row: row['PQ']),
        raw_cache=dict(old=old_info, new=new_info),
        baseline_raw_base_equals_new_limb_images=91,
        saved_rescue=dict(path=str(candidate_path), sha256=sha256(candidate_path),
                          rows=len(candidate), all_rle_run_lengths_validated=True,
                          **{field: score[field] for field in ['pq', 'sq', 'rq', 'tp_pairs', 'fp', 'fn']},
                          extra_matching_pairs=extra, **overlap),
        target_exceeded=score['pq'] > .5,
    )
    output = ROOT / 'docs/evidence/preprocessing_results_verification.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    progress_path = ROOT / 'docs/evidence/target050_progress.json'
    progress = json.loads(progress_path.read_text())
    progress.update(
        achieved=score['pq'] > .5,
        best_full_validation_nonoverlap=score['pq'], best_full_validation_any=score['pq'],
        unit_tests_passed=61, current_tp=score['tp_pairs'], current_fp=score['fp'],
        current_fn=score['fn'], current_best_csv=str(candidate_path),
        current_recipe='base/scale/vflip conf.20 ->2of3 foreground ->close5 ellipse ->8CC area300; preserve these masks and add novel new-preprocessing candidates conf>=.30 with old-view IoU>=.30 corroboration, pre-existing overlap fraction<.10, remaining area>=300',
        preprocessing_mixed_experiments=dict(settings=85, new_settings=84, reference_settings=1,
                                            full_cohort_executed=True, improved_over_reference=False),
        gated_candidate_rescue=dict(settings=32, full_cohort_executed=True,
                                    baseline_pq=rescue['baseline']['pq'], best_pq=score['pq'],
                                    added_physical_masks=27, delta_tp=12, delta_fp=16, delta_fn=-12),
        pixel_refinement=dict(status='implementation and reference smoke pending',
                              full_competition_validation_executed=False,
                              input_needed='Native limb-preprocessed validation pixels near candidate masks; full train images not required'),
    )
    progress['preprocessing_plan'].update(full_cohort_executed=True, reported_scores_recomputed=68)
    progress['raw_candidate_availability']['reference_pq'] = rescue['baseline']['pq']
    progress['raw_candidate_availability']['scope'] = 'GT-aware diagnosis relative to pre-rescue reference with FN241, not achieved recall or current FN229 availability'
    progress['execution_blocker'] = 'Current cloud has CPU only and no direct Colab/Drive access. Cached-mask experiments run locally; full 91-image pixel refinement awaits lossless candidate-neighborhood pixel export from existing Drive images. Further whole-cohort neural inference/training runs beside Drive data on Colab GPU.'
    progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'recomputed':68, 'best_preprocessing':report['best_new_preprocessing_setting'],
                      'rescued':report['saved_rescue']}, indent=2))


if __name__ == '__main__':
    main()
