"""Bounded adaptive follow-up: fixed old-mask confirmation plus fixed new rescues.

Two global controls were selected on this same development set in the preceding
48-rule weak-support experiment. This follow-up is explicitly tuning, not CV.
No per-instance or per-image selection uses GT.
"""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from pycocotools import mask as mu

from experiment_mask_postprocess import read_rows, group_rows, overlap_stats
from experiment_native_cross_model import BASELINE, SOURCE, matched_ids, trim
from experiment_view_combinations import ROOT, GT, VAL, SIZE
from filament_eval import Filament, evaluate

WEAK=Path('/workspace/artifacts/native_weak_support_candidates_v1')
CONTROLS={
    'precision':WEAK/'native_t0.35_a300_c0.3_f2_iou0.3_replaceFalse.csv',
    'recall':WEAK/'native_t0.35_a300_c0.1_f1_iou0.3_replaceTrue.csv',
}
OUT=Path('/workspace/artifacts/native_composition_candidates_v1')


def plan():
    return [dict(name=f'{label}_confirm{t}_{i}',addition_control=label,
                 confirmation_threshold=t,min_confirmation_iou=i)
            for label in CONTROLS for t,i in [(None,0.)]+[(t,i) for t in (.35,.5,.65) for i in (.1,.2,.3)]]


def main():
    started=time.monotonic();settings=plan();assert len(settings)==20
    OUT.mkdir(exist_ok=False);(OUT/'predeclared_plan.json').write_text(json.dumps(settings,indent=2)+'\n')
    with VAL.open(newline='') as f:selection=list(csv.DictReader(f))
    ids={r['image_id'] for r in selection};images={Path(r['file_name']).stem for r in selection}
    baseline=read_rows(BASELINE);grouped=group_rows(baseline)
    native={t:group_rows(read_rows(SOURCE/f'best_t{t:g}_a100.csv')) for t in (.35,.5,.65)}
    controls={k:group_rows(read_rows(p)) for k,p in CONTROLS.items()}
    outputs={s['name']:[] for s in settings};counts={s['name']:defaultdict(int) for s in settings}
    for image in sorted(images):
        old=grouped.get(image,[]);br=[r.rle(SIZE,SIZE) for r in old]
        base_union=mu.merge(br) if br else None
        additions={}
        for label,control in controls.items():
            # The original weak-support implementation emits original/replacement
            # rows first and true additions after exactly len(old) rows.
            tails=control.get(image,[])[len(old):]
            occupied=base_union;extras=[]
            for row in tails:
                clipped=trim(row,occupied)
                if clipped is None or clipped.area<300:continue
                extras.append(clipped)
                occupied=mu.merge([occupied,clipped.rle(SIZE,SIZE)]) if occupied is not None else clipped.rle(SIZE,SIZE)
            additions[label]=extras
        supports={}
        for t in native:
            nr=[r.rle(SIZE,SIZE) for r in native[t].get(image,[])]
            matrix=np.asarray(mu.iou(br,nr,[0]*len(nr))) if br and nr else np.zeros((len(br),len(nr)))
            supports[t]=matrix.max(axis=1) if nr else np.zeros(len(br))
        for s in settings:
            t=s['confirmation_threshold'];label=s['addition_control']
            keep=old if t is None else [r for r,score in zip(old,supports[t]) if score>=s['min_confirmation_iou']]
            result=keep+additions[label]
            counts[s['name']]['old_removed']+=len(old)-len(keep)
            counts[s['name']]['new_added']+=len(additions[label])
            outputs[s['name']].extend(Filament(f'{image}_{j}',image,image,r.counts,r.area) for j,r in enumerate(result,1))
    truth=read_rows(GT,gt=True,selected=ids);assert len(truth)==708
    reference=evaluate(truth,baseline,SIZE,SIZE);refmatches=matched_ids(truth,baseline)
    reports=[dict(name='original_reference',**{k:reference[k] for k in ('pq','sq','rq','tp_pairs','fp','fn')},prediction_rows=len(baseline))]
    for s in settings:
        rows=outputs[s['name']];score=evaluate(truth,rows,SIZE,SIZE)
        assert score['tp_pairs']==len(truth)-score['fn']
        assert overlap_stats(rows)['overlap_images']==0
        reports.append(dict(**s,**{k:score[k] for k in ('pq','sq','rq','tp_pairs','fp','fn')},prediction_rows=len(rows),**dict(counts[s['name']])))
    best=max(reports,key=lambda r:r['pq']);retention=max([r for r in reports if r['fn']<=reference['fn']],key=lambda r:r['pq'])
    saved=[]
    for item in {r['name']:r for r in [best,retention] if r['name']!='original_reference'}.values():
        rows=outputs[item['name']];p=OUT/(item['name']+'.csv')
        with p.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['filament_id','segmentation_rle']);w.writerows((r.filament_id,r.counts) for r in rows)
        reload=read_rows(p);check=evaluate(truth,reload,SIZE,SIZE);assert abs(check['pq']-item['pq'])<1e-12
        now=matched_ids(truth,reload)
        saved.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),recovered_gt_ids=sorted(now-refmatches),lost_gt_ids=sorted(refmatches-now)))
    evidence=dict(scope='20 adaptive global-control compositions on same development set; no independent CV/LB',
        baseline_sha256=hashlib.sha256(BASELINE.read_bytes()).hexdigest(),
        control_sha256={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in CONTROLS.items()},
        recall_control_derivation='Discard firstNbase potential replacements per image, keep tail additions, re-trim against complete originalbase before pruning, retain area>=300',
        physical_images=len(images),annotation_groups=len(ids),gt_instances=len(truth),experiments=reports,best=best,best_fn_retention=retention,
        saved_candidates=saved,target_exceeded=best['pq']>.5,elapsed_seconds=time.monotonic()-started)
    (ROOT/'docs/evidence/native_composition_experiments.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(dict(best=best,best_fn_retention=retention,saved_candidates=saved,elapsed_seconds=evidence['elapsed_seconds']),indent=2))


if __name__=='__main__':main()
