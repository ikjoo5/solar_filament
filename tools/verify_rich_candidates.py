"""Independently re-score saved rich-export experiments and GT match transitions."""
import csv
import hashlib
import json
from pathlib import Path

from experiment_mask_postprocess import overlap_stats, read_rows
from filament_eval import evaluate
from verify_native_fusion_results import candidate_paths
from verify_native_training_results import GT, ROOT, VAL, hits_by_gt
from verify_rich_predictions import INCUMBENT


def main():
    with VAL.open(newline='') as f:
        groups = {r['image_id'] for r in csv.DictReader(f)}
    truth = read_rows(GT, gt=True, selected=groups)
    assert len(truth) == 708 and len(groups) == 101
    before = hits_by_gt(truth, read_rows(INCUMBENT))
    old_matches = {fid for fid, value in before.items() if value > .5}
    assert len(old_matches) == 483
    paths = {str(INCUMBENT)}
    reports = []
    for path in sorted((ROOT/'docs/evidence').glob('rich_*.json')):
        if path.name in ('rich_candidates_verification.json', 'rich_predictions_verification.json'):
            continue
        data = json.loads(path.read_text())
        paths.update(candidate_paths(data))
        reports.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    results = []
    for filename in sorted(paths):
        path = Path(filename)
        if path.name in ('gt.csv','train_split.csv','val_split.csv'):
            continue
        rows = read_rows(path)
        score = evaluate(truth, rows, 2048, 2048)
        audit = overlap_stats(rows)
        assert not any(audit.values()), (path, audit)
        assert score['tp_pairs'] == len(truth)-score['fn'], path
        after = hits_by_gt(truth, rows)
        matches = {fid for fid, value in after.items() if value > .5}
        results.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            **{k:score[k] for k in ('pq','sq','rq','tp_pairs','fp','fn','matched_iou_sum')},
            physical_masks=len(rows), **audit, extra_matching_pairs=0,
            retained_gt=len(old_matches & matches), recovered_gt=len(matches-old_matches),
            lost_gt=len(old_matches-matches), recovered_gt_ids=sorted(matches-old_matches),
            lost_gt_ids=sorted(old_matches-matches)))
    best = max(results,key=lambda r:r['pq'])
    preserving = max((r for r in results if r['lost_gt']==0),key=lambda r:r['pq'])
    recall = min(results,key=lambda r:(r['fn'],-r['pq']))
    report = dict(scope='Independent saved-CSV verification on fixed91-image/101-group development set; not Kaggle LB',
        physical_images=91,annotation_groups=101,gt_instances=708,
        previous_reference=str(INCUMBENT),previous_matched_gt=483,
        source_reports=reports,saved_candidates=results,highest_pq=best,
        highest_pq_preserving_previous_matches=preserving,lowest_fn=recall,target_exceeded=best['pq']>.5)
    (ROOT/'docs/evidence/rich_candidates_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(candidates=len(results),highest_pq=best,all_previous_matches_preserved=preserving,
        lowest_fn=recall),indent=2))


if __name__ == '__main__':
    main()
