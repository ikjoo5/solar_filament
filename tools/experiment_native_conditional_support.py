"""Sixteen fixed conditional native-support pruning rules, starting incumbent."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import time

from experiment_mask_postprocess import overlap_stats, read_rows
from experiment_native_support_pruning import INCUMBENT, NATIVE, matched_ids, supported
from experiment_view_combinations import GT, ROOT, SIZE, VAL
from filament_eval import evaluate

OUTPUT=Path('/workspace/artifacts/native_conditional_support_20261028')


def plan():
    result=[]
    for iou in (.2,.3):
        result += [dict(name=f'iou{iou:g}_area{area}',rule='area',maximum_iou=iou,maximum_area=area)
                   for area in (400,600,1000,2000)]
        result += [dict(name=f'iou{iou:g}_coverage{coverage:g}',rule='coverage',maximum_iou=iou,maximum_coverage=coverage)
                   for coverage in (.1,.2,.4,.6)]
    assert len(result)==16
    return result


def main():
    started=time.monotonic()
    settings=plan()
    selected=list(csv.DictReader(VAL.open()))
    groups={r['image_id'] for r in selected}
    base=read_rows(INCUMBENT)
    assert len(base)==626 and overlap_stats(base)['overlap_images']==0
    native_path=NATIVE/'best_t0.35_a100.csv'
    native=read_rows(native_path)
    features=supported(base,native)
    predictions={}
    for setting in settings:
        rows=[]
        for row in base:
            f=features[row.filament_id]
            condition=(row.area<setting['maximum_area'] if setting['rule']=='area'
                       else f['coverage']<setting['maximum_coverage'])
            if not (f['max_iou']<setting['maximum_iou'] and condition):
                rows.append(row)
        predictions[setting['name']]=rows
    truth=read_rows(GT,gt=True,selected=groups)
    baseline_matches=matched_ids(truth,base)
    baseline=evaluate(truth,base,SIZE,SIZE)
    assert abs(baseline['pq']-.46493857594671645)<1e-12
    reports=[]
    for setting in settings:
        rows=predictions[setting['name']]
        result=evaluate(truth,rows,SIZE,SIZE)
        matches=matched_ids(truth,rows)
        assert not matches-baseline_matches
        assert result['tp_pairs']==len(matches)
        reports.append(dict(**setting,**{k:result[k] for k in ('pq','sq','rq','tp_pairs','fp','fn')},
                            prediction_rows=len(rows),removed_physical_masks=len(base)-len(rows),
                            recovered_gt=0,lost_gt=len(baseline_matches-matches),
                            lost_gt_ids=sorted(baseline_matches-matches),preserved_gt=len(matches)))
    best=max(reports,key=lambda r:r['pq'])
    preserving=[r for r in reports if r['lost_gt']==0]
    best_preserving=max(preserving,key=lambda r:r['pq']) if preserving else None
    best_fn=min(reports,key=lambda r:(r['fn'],-r['pq']))
    OUTPUT.mkdir(exist_ok=True)
    saved=[]
    for item in (best,best_preserving,best_fn):
        if item is None or item['name'] in [r['name'] for r in saved]:
            continue
        path=OUTPUT/(item['name']+'.csv')
        with path.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['filament_id','segmentation_rle'])
            w.writerows((r.filament_id,r.counts) for r in predictions[item['name']])
        reloaded=read_rows(path)
        assert overlap_stats(reloaded)['overlap_images']==0
        rescored=evaluate(truth,reloaded,SIZE,SIZE)
        assert abs(rescored['pq']-item['pq'])<1e-12
        saved.append(dict(name=item['name'],path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    report=dict(scope='16 conditional support rules; same-development tuning only',
                source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (INCUMBENT,native_path)},
                baseline={k:baseline[k] for k in ('pq','sq','rq','tp_pairs','fp','fn')},
                rule='Reject incumbent iff maxIoU below threshold AND area/union-coverage below threshold; unchanged original geometry otherwise',
                settings=reports,best=best,best_all_matches_preserving=best_preserving,best_fn=best_fn,
                saved_candidates=saved,saved_scores_recomputed=True,
                elapsed_seconds=time.monotonic()-started)
    (ROOT/'docs/evidence/native_conditional_support_experiments.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(best=best,best_all_matches_preserving=best_preserving,best_fn=best_fn,saved_candidates=saved),indent=2))


if __name__=='__main__':
    main()
