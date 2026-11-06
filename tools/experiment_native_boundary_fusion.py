"""Bounded, GT-free cross-model boundary and foreground-component fusion."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from pycocotools import mask as mu

from experiment_mask_postprocess import read_rows, group_rows, overlap_stats
from experiment_native_cross_model import BASELINE, SOURCE, matched_ids
from experiment_view_combinations import ROOT, GT, VAL, SIZE, encode_roi, component_rles
from filament_eval import Filament, evaluate

OUT=Path('/workspace/artifacts/native_boundary_fusion_candidates_v1')


def plan():
    settings=[dict(name=f'pair_t{t:g}_match{m:g}_{o}_guard{g:g}',mode='pair',threshold=t,
                   match_iou=m,operation=o,original_iou_guard=g)
              for t in (.35,.5,.65) for m in (.2,.4) for o in ('union','intersection') for g in (.6,.8,.9)]
    settings += [dict(name=f'unioncc_t{t:g}_a{a}',mode='unioncc',threshold=t,min_area=a)
                 for t in (.35,.5,.65) for a in (300,600)]
    settings += [dict(name=f'confirm_t{t:g}_iou{i:g}_small{a}',mode='confirm',threshold=t,
                     match_iou=i,max_filter_area=a)
                 for t in (.35,.65) for i in (.05,.2,.4) for a in (1000,1e10)]
    assert len(settings)==54
    return settings


def main():
    started=time.monotonic();cv2.setNumThreads(1)
    settings=plan();OUT.mkdir(exist_ok=True)
    assert not list(OUT.glob('*.csv')), 'Do not overwrite completed candidates'
    (OUT/'predeclared_plan.json').write_text(json.dumps(settings,indent=2)+'\n')
    with VAL.open(newline='') as f: selection=list(csv.DictReader(f))
    ids={r['image_id'] for r in selection};images={Path(r['file_name']).stem for r in selection}
    baseline=read_rows(BASELINE);grouped=group_rows(baseline)
    sources={}
    for t in (.35,.5,.65):
        rows=read_rows(SOURCE/f'best_t{t:g}_a100.csv')
        assert overlap_stats(rows)['overlap_images']==0
        sources[t]=group_rows(rows)
    outputs={s['name']:[] for s in settings};counters={s['name']:Counter() for s in settings}
    for number,image in enumerate(sorted(images),1):
        old=grouped.get(image,[]);br=[r.rle(SIZE,SIZE) for r in old]
        base_union=mu.merge(br) if br else None
        base_canvas=mu.decode(base_union).astype(bool) if base_union else np.zeros((SIZE,SIZE),bool)
        original_boxes=[list(map(int,mu.toBbox(r))) for r in br]
        decoded_old=[mu.decode(r).astype(bool) for r in br]
        for threshold in (.35,.5,.65):
            new=sources[threshold].get(image,[]);nr=[r.rle(SIZE,SIZE) for r in new]
            matrix=np.asarray(mu.iou(br,nr,[0]*len(nr))) if br and nr else np.zeros((len(br),len(nr)))
            assignment=matrix.argmax(axis=0) if br and nr else np.empty(0,int)
            new_boxes=[list(map(int,mu.toBbox(r))) for r in nr]
            decoded_new=[mu.decode(r).astype(bool) for r in nr]
            for setting in [s for s in settings if s['threshold']==threshold]:
                name=setting['name']
                if setting['mode']=='unioncc':
                    canvas=base_canvas.copy()
                    if nr:canvas |= mu.decode(mu.merge(nr)).astype(bool)
                    for index,(rle,area) in enumerate(component_rles(canvas,setting['min_area']),1):
                        outputs[name].append(Filament(f'{image}_{index}',image,image,rle['counts'].decode('ascii'),area))
                    continue
                if setting['mode']=='confirm':
                    for k,row in enumerate(old):
                        support=float(matrix[k].max()) if len(nr) else 0.
                        if row.area>=setting['max_filter_area'] or support>=setting['match_iou']:
                            outputs[name].append(row)
                        else:counters[name]['dropped']+=1
                    continue
                candidates=[];originals=[];regions=[]
                for k,row in enumerate(old):
                    valid=[j for j in range(len(new)) if assignment[j]==k and matrix[k,j]>=setting['match_iou']]
                    chosen=max(valid,key=lambda j:matrix[k,j]) if valid else None
                    boxes=[original_boxes[k]]+([new_boxes[chosen]] if chosen is not None else [])
                    x1=min(b[0] for b in boxes);y1=min(b[1] for b in boxes)
                    x2=max(b[0]+b[2] for b in boxes);y2=max(b[1]+b[3] for b in boxes)
                    region=np.s_[y1:y2,x1:x2];original=decoded_old[k][region]
                    candidate=original.copy();status='unmatched'
                    if chosen is not None:
                        other=decoded_new[chosen][region]
                        candidate=original|other if setting['operation']=='union' else original&other
                        area=int(candidate.sum());ratio=area/row.area
                        iou=int((original&candidate).sum())/int((original|candidate).sum())
                        if not .5<=ratio<=1.5 or iou<setting['original_iou_guard']:
                            candidate=original.copy();status='guard_fallback'
                        else:status='changed' if np.any(candidate!=original) else 'unchanged'
                    # A native mask is assigned to only one original and native masks
                    # are disjoint, so expansion-vs-expansion conflicts cannot occur.
                    # Protect every other original exactly as nearest-original
                    # ownership would do; this avoids dense per-setting distance maps.
                    candidate &= ~base_canvas[region] | original
                    candidates.append(candidate);originals.append(original);regions.append(region);counters[name][status]+=1
                resolved=candidates
                for row,mask,region in zip(old,resolved,regions):
                    assert mask.any()
                    ys,xs=np.nonzero(mask);top,left=int(ys.min()),int(xs.min())
                    rle=encode_roi(mask[top:int(ys.max())+1,left:int(xs.max())+1],region[0].start+top,region[1].start+left)
                    outputs[name].append(Filament(row.filament_id,image,image,rle['counts'].decode('ascii'),int(mu.area(rle))))
        if number%20==0:print('Boundary fusion',number,'/',len(images),flush=True)
    # GT first opened only after all predictions are generated.
    truth=read_rows(GT,gt=True,selected=ids);assert len(truth)==708
    reference=evaluate(truth,baseline,SIZE,SIZE);assert abs(reference['pq']-.46493857594671645)<1e-12
    reports=[dict(name='reference',**{k:reference[k] for k in ('pq','sq','rq','tp_pairs','fp','fn')},prediction_rows=len(baseline))]
    for setting in settings:
        rows=outputs[setting['name']];score=evaluate(truth,rows,SIZE,SIZE)
        overlap=overlap_stats(rows);assert overlap['overlap_images']==0
        assert score['tp_pairs']==len(truth)-score['fn']
        reports.append(dict(**setting,**{k:score[k] for k in ('pq','sq','rq','tp_pairs','fp','fn')},prediction_rows=len(rows),statuses=dict(counters[setting['name']]),**overlap))
    best=max(reports,key=lambda r:r['pq']);reference_matched=matched_ids(truth,baseline)
    retention=max([r for r in reports if r['fn']<=reference['fn']],key=lambda r:r['pq'])
    saved=[]
    for item in {r['name']:r for r in [best,retention] if r['name']!='reference'}.values():
        rows=outputs[item['name']];p=OUT/(item['name']+'.csv')
        with p.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['filament_id','segmentation_rle']);w.writerows((r.filament_id,r.counts) for r in rows)
        reload=read_rows(p);check=evaluate(truth,reload,SIZE,SIZE);assert abs(check['pq']-item['pq'])<1e-12
        matches=matched_ids(truth,reload)
        saved.append(dict(name=item['name'],path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),recovered_gt_ids=sorted(matches-reference_matched),lost_gt_ids=sorted(reference_matched-matches)))
    evidence=dict(scope='54 predeclared globally fixed same-development rules; not independent CV/LB',
        baseline_sha256=hashlib.sha256(BASELINE.read_bytes()).hexdigest(),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.glob('best_t*_a100.csv')},
        physical_images=len(images),annotation_groups=len(ids),gt_instances=len(truth),reference=reports[0],experiments=reports,
        best=best,best_fn_retention=retention,saved_candidates=saved,target_exceeded=best['pq']>.5,elapsed_seconds=time.monotonic()-started)
    (ROOT/'docs/evidence/native_boundary_fusion_experiments.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(dict(best=best,best_fn_retention=retention,saved_candidates=saved,elapsed_seconds=evidence['elapsed_seconds']),indent=2))


if __name__=='__main__':main()
