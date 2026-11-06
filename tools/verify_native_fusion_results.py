"""Independently reload all nominated native/YOLO candidates and verify tradeoffs."""
import csv
import hashlib
import json
from pathlib import Path

from experiment_mask_postprocess import overlap_stats, read_rows
from filament_eval import evaluate
from verify_native_training_results import GT, OLD, ROOT, VAL, hits_by_gt


def candidate_paths(value):
    if isinstance(value, str) and value.endswith('.csv') and Path(value).is_absolute():
        yield value
    elif isinstance(value, list):
        for member in value:
            yield from candidate_paths(member)
    elif isinstance(value, dict):
        for member in value.values():
            yield from candidate_paths(member)


def main():
    with VAL.open(newline='') as f:
        ids = {row['image_id'] for row in csv.DictReader(f)}
    truth = read_rows(GT, gt=True, selected=ids)
    before = hits_by_gt(truth, read_rows(OLD))
    old_matches = {fid for fid, iou in before.items() if iou > .5}
    paths = {str(OLD)}
    sources = []
    for report in sorted((ROOT/'docs/evidence').glob('native_*experiments.json')):
        document = json.loads(report.read_text())
        paths.update(candidate_paths(document))
        sources.append(dict(path=str(report), sha256=hashlib.sha256(report.read_bytes()).hexdigest()))
    results = []
    for filename in sorted(paths):
        path = Path(filename)
        rows = read_rows(path)
        metric = evaluate(truth, rows, 2048, 2048)
        audit = overlap_stats(rows)
        assert not any(audit.values()), (path, audit)
        extra = metric['tp_pairs']-(len(truth)-metric['fn'])
        assert extra == 0
        after = hits_by_gt(truth, rows)
        matches = {fid for fid, iou in after.items() if iou > .5}
        results.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            **{k:metric[k] for k in ('pq','sq','rq','tp_pairs','fp','fn','matched_iou_sum')},
            physical_masks=len(rows), **audit, extra_matching_pairs=extra,
            preserved_gt=len(matches & old_matches), recovered_gt=len(matches-old_matches),
            lost_gt=len(old_matches-matches), recovered_gt_ids=sorted(matches-old_matches),
            lost_gt_ids=sorted(old_matches-matches)))
    best = max(results, key=lambda r:r['pq'])
    preserved = max((r for r in results if not r['lost_gt']), key=lambda r:r['pq'])
    report = dict(scope='Independent saved-CSV recheck on the fixed91-image development cohort; not Kaggle LB',
        physical_images=91, annotation_groups=101, gt_instances=len(truth),
        experiment_reports=sources, saved_candidates=results,
        highest_development_pq=best, highest_pq_preserving_all_incumbent_matches=preserved,
        target_exceeded=best['pq'] > .5)
    (ROOT/'docs/evidence/native_fusion_saved_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(candidates_verified=len(results),best=best,fn_preserving=preserved),indent=2))


if __name__ == '__main__':
    main()
