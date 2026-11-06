"""Read-only verification of the first uploaded 100-epoch A100 result bundle."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

from experiment_mask_postprocess import group_rows, overlap_stats, read_rows
from filament_eval import evaluate


ROOT = Path(__file__).resolve().parents[1]
RESULT = Path('/workspace/artifacts/a100_results_20261028_bedf3e21')
VAL = Path('/workspace/attachments/bcb64220-b208-4cd7-9c3f-3f2be0f04167/val_split.csv')
OLD = Path('/workspace/artifacts/tiled_object_fusion_candidates_v2/c0.35_iou0.3_majority_guard0.85.csv')
GT = Path('/workspace/artifacts/solar_filament_gt_v1/gt.csv')
FIELDS = ('pq', 'sq', 'rq', 'tp_pairs', 'fp', 'fn', 'matched_iou_sum')


def hits_by_gt(truth, predictions):
    grouped = group_rows(predictions)
    answer = {}
    for row in truth:
        members = grouped.get(row.image_id, [])
        scores = (np.asarray(mu.iou([row.rle(2048, 2048)],
            [r.rle(2048, 2048) for r in members], [0]*len(members)))[0]
            if members else np.empty(0))
        answer[row.filament_id] = float(scores.max(initial=0.))
    return answer


def main():
    with VAL.open(newline='') as f:
        split = list(csv.DictReader(f))
    groups = {r['image_id'] for r in split}
    images = {Path(r['file_name']).stem for r in split}
    truth = read_rows(GT, gt=True, selected=groups)
    assert len(groups) == 101 and len(images) == 91 and len(truth) == 708
    best = json.loads((RESULT/'best_epoch.json').read_text())
    cfg = json.loads((RESULT/'config.json').read_text())
    original_source = RESULT/'run_source.py'
    local_source = ROOT/'tools/solar_a100_train_standalone.py'
    grid = []
    for setting in best['grid']:
        file = RESULT/f"best_t{setting['threshold']:g}_a{setting['min_area']}.csv"
        rows = read_rows(file)
        assert {r.image_id for r in rows} <= images
        score = evaluate(truth, rows, 2048, 2048)
        for field in ('pq', 'sq', 'rq', 'tp_pairs', 'fp', 'fn'):
            assert abs(score[field]-setting[field]) < 1e-12, (file, field)
        audit = overlap_stats(rows)
        assert not any(audit.values()), (file, audit)
        grid.append(dict(file=file.name, threshold=setting['threshold'],
            min_area=setting['min_area'], **{k:score[k] for k in FIELDS},
            prediction_rows=len(rows), extra_matching_pairs=score['tp_pairs']-(708-score['fn']),
            sha256=hashlib.sha256(file.read_bytes()).hexdigest(), **audit))
    native = read_rows(RESULT/'best_validation.csv')
    old = read_rows(OLD)
    native_score = evaluate(truth, native, 2048, 2048)
    old_score = evaluate(truth, old, 2048, 2048)
    before, after = hits_by_gt(truth, old), hits_by_gt(truth, native)
    status = Counter()
    changes = []
    for fid in before:
        b, a = before[fid] > .5, after[fid] > .5
        key = 'retained' if b and a else 'native_only' if a else 'old_only' if b else 'both_miss'
        status[key] += 1
        if b != a:
            changes.append(dict(filament_id=fid, status=key, old_iou=before[fid], native_iou=after[fid]))
    physical = group_rows(native)
    legacy = group_rows(old)
    agreement = Counter()
    for image, members in physical.items():
        br = [r.rle(2048,2048) for r in legacy.get(image, [])]
        nr = [r.rle(2048,2048) for r in members]
        matrix = np.asarray(mu.iou(nr, br, [0]*len(br))) if br else np.zeros((len(nr),0))
        best_iou = matrix.max(axis=1) if br else np.zeros(len(nr))
        for value in best_iou:
            agreement['new_mask_best_iou_gt_0.5' if value>.5 else
                'new_mask_best_iou_0.1_to_0.5' if value>.1 else 'new_mask_best_iou_le_0.1'] += 1
    report = dict(scope='Same fixed development cohort; independent recomputation, not leaderboard or new holdout',
        archive_sha256='637f0cf617c6806be0fcd26c598e30a9c3b753dd074b16940b3def13ae390f71',
        validation=dict(physical_images=91, annotation_groups=101, gt_instances=708),
        uploaded_source_sha256=hashlib.sha256(original_source.read_bytes()).hexdigest(),
        uploaded_source_equals_prepared_worker=original_source.read_bytes()==local_source.read_bytes(),
        uploaded_source_equals_worker_after_leading_whitespace=original_source.read_bytes().lstrip()==local_source.read_bytes().lstrip(),
        source_difference_note='Uploaded source has one additional initial newline; executable source is unchanged.',
        source_hashes_match_pinned=cfg['source_sha256']==cfg['expected_source_sha256'],
        best_epoch=best['epoch'], epochs_completed=json.loads((RESULT/'summary.json').read_text())['epochs_completed'],
        all_reported_grid_scores_reproduced=True, grid=grid,
        native_best={k:native_score[k] for k in FIELDS},
        incumbent={k:old_score[k] for k in FIELDS},
        best_validation_equals_selected_grid=(RESULT/'best_validation.csv').read_bytes()==(RESULT/'best_t0.5_a300.csv').read_bytes(),
        cross_model_gt_diagnostic=dict(status), changed_gt_diagnostic=changes,
        diagnostic_recall_union_only=int(status['retained']+status['native_only']+status['old_only']),
        physical_mask_agreement=dict(agreement),
        checkpoint_files_in_zip=False,
        drive_checkpoint_directory=cfg['resume_dir'],
        diagnostic_note='GT used only after prediction generation for error analysis. Recall union is an oracle diagnostic, not an achieved fusion score.')
    destination=ROOT/'docs/evidence/a100_uploaded_results_verification.json'
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('grid','changed_gt_diagnostic')},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
