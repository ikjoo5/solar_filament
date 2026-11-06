"""Final 32-rule upper-confidence follow-up of weak-YOLO U-Net rescues.

The preceding weak-support grid's best actual setting was at its highest tested
weak confidence (.3). This bounded follow-up tests .45/.6/.75/.9 globally, with
no replacement and no per-image or instance tuning from GT.
"""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from pycocotools import mask as mu

from experiment_mask_postprocess import read_rows, group_rows, overlap_stats
from experiment_native_cross_model import BASELINE, SOURCE, matched_ids, trim
from experiment_native_weak_support import load_sources
from experiment_view_combinations import ROOT, GT, VAL, SIZE
from filament_eval import Filament, evaluate

OUT=Path('/workspace/artifacts/native_high_confidence_candidates_v1')


def plan():
    return [dict(name=f'native_t{t:g}_a{a}_weak{c:g}_families{f}',threshold=t,min_area=a,
                 minimum_weak_confidence=c,minimum_families=f,minimum_support_iou=.3,
                 max_original_iou=.1,max_union_overlap_fraction=.25)
            for t in (.35,.5) for a in (300,800) for c in (.45,.6,.75,.9) for f in (1,2)]


def main():
    started=time.monotonic();settings=plan();assert len(settings)==32
    OUT.mkdir(exist_ok=False);(OUT/'predeclared_plan.json').write_text(json.dumps(settings,indent=2)+'\n')
    with VAL.open(newline='') as f:selection=list(csv.DictReader(f))
    ids={r['image_id'] for r in selection};images={Path(r['file_name']).stem for r in selection}
    baseline=read_rows(BASELINE);base_by_image=group_rows(baseline)
    weak,source_meta=load_sources(images)
    native={t:group_rows(read_rows(SOURCE/f'best_t{t:g}_a100.csv')) for t in (.35,.5)}
    outputs={s['name']:[] for s in settings};counts={s['name']:Counter() for s in settings}
    for image in sorted(images):
        old=base_by_image.get(image,[]);br=[r.rle(SIZE,SIZE) for r in old]
        union=mu.merge(br) if br else None
        proposals=weak[image];wr=[p['rle'] for p in proposals]
        for t in native:
            rows=native[t].get(image,[]);nr=[r.rle(SIZE,SIZE) for r in rows]
            wm=np.asarray(mu.iou(nr,wr,[0]*len(wr))) if nr and wr else np.zeros((len(nr),len(wr)))
            bm=np.asarray(mu.iou(nr,br,[0]*len(br))) if nr and br else np.zeros((len(nr),len(br)))
            metadata=[]
            for k,row in enumerate(rows):
                maxbase=float(bm[k].max()) if br else 0.
                fraction=int(mu.area(mu.merge([nr[k],union],intersect=True)))/row.area if union else 0.
                if maxbase>.1 or fraction>.25 or row.area<300:continue
                family_conf={}
                for j,p in enumerate(proposals):
                    if wm[k,j]>=.3:family_conf[p['family']]=max(p['confidence'],family_conf.get(p['family'],0.))
                clipped=trim(row,union)
                if clipped is not None:metadata.append((row,clipped,family_conf))
            for s in [s for s in settings if s['threshold']==t]:
                result=list(old)
                for row,clipped,confs in metadata:
                    families=sum(c>=s['minimum_weak_confidence'] for c in confs.values())
                    if row.area<s['min_area'] or clipped.area<s['min_area'] or families<s['minimum_families']:continue
                    result.append(clipped);counts[s['name']]['physical_additions']+=1
                outputs[s['name']].extend(Filament(f'{image}_{j}',image,image,r.counts,r.area) for j,r in enumerate(result,1))
    truth=read_rows(GT,gt=True,selected=ids);assert len(truth)==708
    reference=evaluate(truth,baseline,SIZE,SIZE);refmatches=matched_ids(truth,baseline)
    reports=[dict(name='reference',**{k:reference[k] for k in ('pq','sq','rq','tp_pairs','fp','fn')},prediction_rows=len(baseline))]
    for s in settings:
        rows=outputs[s['name']];score=evaluate(truth,rows,SIZE,SIZE)
        assert score['tp_pairs']==len(truth)-score['fn']
        assert overlap_stats(rows)['overlap_images']==0
        reports.append(dict(**s,**{k:score[k] for k in ('pq','sq','rq','tp_pairs','fp','fn')},prediction_rows=len(rows),**dict(counts[s['name']])))
    best=max(reports,key=lambda r:r['pq']);actual=max(reports[1:],key=lambda r:r['pq']);saved=None
    if best['name']!='reference':
        rows=outputs[best['name']];p=OUT/(best['name']+'.csv')
        with p.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['filament_id','segmentation_rle']);w.writerows((r.filament_id,r.counts) for r in rows)
        reload=read_rows(p);score=evaluate(truth,reload,SIZE,SIZE);assert abs(score['pq']-best['pq'])<1e-12
        matches=matched_ids(truth,reload)
        saved=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),recovered_gt_ids=sorted(matches-refmatches),lost_gt_ids=sorted(refmatches-matches))
    evidence=dict(scope='32 bounded upper-confidence follow-up global rules; same-development tuning, not CV/LB',
        baseline_sha256=hashlib.sha256(BASELINE.read_bytes()).hexdigest(),weak_sources=source_meta,
        native_source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.glob('best_t*_a100.csv')},
        physical_images=len(images),annotation_groups=len(ids),gt_instances=len(truth),experiments=reports,
        best=best,best_actual=actual,saved_candidate=saved,target_exceeded=best['pq']>.5,elapsed_seconds=time.monotonic()-started)
    (ROOT/'docs/evidence/native_high_confidence_experiments.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(dict(best=best,best_actual=actual,saved_candidate=saved,elapsed_seconds=evidence['elapsed_seconds']),indent=2))


if __name__=='__main__':main()
