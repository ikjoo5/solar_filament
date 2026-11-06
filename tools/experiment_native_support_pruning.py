"""Prune incumbent masks by fixed independent native-model support rules.

No masks are added or changed; annotations are only loaded for final scoring.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from pycocotools import mask as mu

from experiment_mask_postprocess import group_rows, overlap_stats, read_rows
from experiment_view_combinations import GT, ROOT, SIZE, VAL
from filament_eval import evaluate

INCUMBENT = Path('/workspace/artifacts/tiled_object_fusion_candidates_v2/c0.35_iou0.3_majority_guard0.85.csv')
NATIVE = Path('/workspace/artifacts/a100_results_20261028_bedf3e21')
OUTPUT = Path('/workspace/artifacts/native_support_pruning_20261028')


def plan():
    settings = [dict(name='incumbent',rule='identity')]
    for p in (.35,.5,.65):
        settings += [dict(name=f't{p:g}_coverage{c:g}', rule='coverage', probability=p, minimum=c)
                     for c in (.05,.1,.2,.35,.5)]
        settings += [dict(name=f't{p:g}_iou{c:g}', rule='max_iou', probability=p, minimum=c)
                     for c in (.05,.1,.2,.3)]
    assert len(settings)==28
    return settings


def supported(base, native):
    native_by_image=group_rows(native)
    result={}
    for image, originals in group_rows(base).items():
        support=native_by_image.get(image,[])
        old_rles=[r.rle(SIZE,SIZE) for r in originals]
        new_rles=[r.rle(SIZE,SIZE) for r in support]
        if not support:
            result.update({r.filament_id:dict(coverage=0.,max_iou=0.) for r in originals})
            continue
        # New components are disjoint, so intersections add to union coverage.
        ious=np.asarray(mu.iou(old_rles,new_rles,[0]*len(new_rles)))
        old_area=np.asarray([r.area for r in originals],float)[:,None]
        new_area=np.asarray([r.area for r in support],float)[None,:]
        intersections=ious/(1+ious)*(old_area+new_area)
        coverage=np.clip(intersections.sum(axis=1)/old_area[:,0],0.,1.)
        for i,row in enumerate(originals):
            result[row.filament_id]=dict(coverage=float(coverage[i]),max_iou=float(ious[i].max()))
    return result


def matched_ids(truth,predictions):
    predicted=group_rows(predictions)
    grouped={}
    for row in truth:
        grouped.setdefault(row.group_id,[]).append(row)
    matches=set()
    for rows in grouped.values():
        preds=predicted.get(rows[0].image_id,[])
        if not preds:
            continue
        iou=np.asarray(mu.iou([r.rle(SIZE,SIZE) for r in rows],
                             [r.rle(SIZE,SIZE) for r in preds],[0]*len(preds)))
        matches.update(r.filament_id for r,hit in zip(rows,(iou>.5).any(axis=1)) if hit)
    return matches


def main():
    started=time.monotonic()
    grid=plan()
    selected=list(csv.DictReader(VAL.open()))
    groups={r['image_id'] for r in selected}
    images={Path(r['file_name']).stem for r in selected}
    base=read_rows(INCUMBENT)
    assert len(base)==626 and {r.image_id for r in base}<=images
    assert overlap_stats(base)['overlap_images']==0
    source_paths={p:NATIVE/f'best_t{p:g}_a100.csv' for p in (.35,.5,.65)}
    native={p:read_rows(path) for p,path in source_paths.items()}
    assert all(overlap_stats(rows)['overlap_images']==0 for rows in native.values())
    features={p:supported(base,rows) for p,rows in native.items()}
    predictions={}
    for setting in grid:
        if setting['rule']=='identity':
            rows=base
        else:
            features_at_p=features[setting['probability']]
            rows=[r for r in base if features_at_p[r.filament_id][setting['rule']]>=setting['minimum']]
        predictions[setting['name']]=rows
    # Predictions are complete before any ground truth is loaded.
    truth=read_rows(GT,gt=True,selected=groups)
    assert len(truth)==708 and len(groups)==101 and len(images)==91
    baseline_matches=matched_ids(truth,base)
    reports=[]
    for setting in grid:
        rows=predictions[setting['name']]
        score=evaluate(truth,rows,SIZE,SIZE)
        matches=matched_ids(truth,rows)
        assert not matches-baseline_matches
        assert score['tp_pairs']==len(matches)
        reports.append(dict(**setting,**{k:score[k] for k in ('pq','sq','rq','tp_pairs','fp','fn')},
                            prediction_rows=len(rows),removed_physical_masks=len(base)-len(rows),
                            recovered_gt=len(matches-baseline_matches),lost_gt=len(baseline_matches-matches),
                            lost_gt_ids=sorted(baseline_matches-matches),preserved_gt=len(matches&baseline_matches)))
    best=max(reports,key=lambda r:r['pq'])
    best_nontrivial=max((r for r in reports if r['rule']!='identity'),key=lambda r:r['pq'])
    assert abs(reports[0]['pq']-.46493857594671645)<1e-12
    OUTPUT.mkdir(exist_ok=True)
    saved=[]
    for item in (best,best_nontrivial):
        path=OUTPUT/(item['name']+'.csv')
        if str(path) in [r['path'] for r in saved]:
            continue
        with path.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['filament_id','segmentation_rle'])
            w.writerows((r.filament_id,r.counts) for r in predictions[item['name']])
        reloaded=read_rows(path)
        assert overlap_stats(reloaded)['overlap_images']==0
        score=evaluate(truth,reloaded,SIZE,SIZE)
        assert abs(score['pq']-item['pq'])<1e-12
        saved.append(dict(name=item['name'],path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    evidence=dict(scope='28 fixed independent-model support rules on same development set; no new masks or geometry changes',
                  source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [INCUMBENT,*source_paths.values()]},
                  physical_images=91,annotation_groups=101,gt_instances=708,
                  support_source='union of saved native components area>=100 at probability thresholds.35/.5/.65',
                  experiments=reports,best=best,best_nontrivial=best_nontrivial,saved_candidates=saved,
                  saved_scores_recomputed=True,overlap_images_all_outputs=0,extra_matching_pairs_all_outputs=0,
                  elapsed_seconds=time.monotonic()-started)
    (ROOT/'docs/evidence/native_support_pruning_experiments.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(dict(best=best,best_nontrivial=best_nontrivial,saved_candidates=saved,elapsed_seconds=evidence['elapsed_seconds']),indent=2))


if __name__=='__main__':
    main()
