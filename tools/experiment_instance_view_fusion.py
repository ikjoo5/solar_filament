"""Sixteen fixed, GT-free object correspondence experiments from four-view RLEs."""
from collections import Counter
import argparse
import csv
import json
from pathlib import Path
import time

import cv2
import numpy as np
from pycocotools import mask as mu

from experiment_view_combinations import load_cache, encode_roi, RESULTS, GT, VAL, ROOT, SIZE
from experiment_mask_postprocess import read_rows
from filament_eval import Filament, evaluate


def correspond(proposals, iou, minimum_confidence, cutoff):
    remaining = {i for i,p in enumerate(proposals) if p['confidence'] >= minimum_confidence}
    order = sorted(remaining, key=lambda i: -proposals[i]['confidence'])
    groups = []
    for seed in order:
        if seed not in remaining:
            continue
        members = [seed]
        remaining.remove(seed)
        for view in ['base','scale','hflip','vflip']:
            if view == proposals[seed]['view']:
                continue
            eligible = [i for i in remaining if proposals[i]['view'] == view and iou[seed,i] > cutoff]
            if eligible:
                chosen = max(eligible,key=lambda i:(iou[seed,i],proposals[i]['confidence']))
                members.append(chosen)
                remaining.remove(chosen)
        groups.append(members)
    return groups


def group_roi(proposals, members, decoded, rule):
    bounds = []
    for i in members:
        if i not in decoded:
            rle=proposals[i]['rle'];x,y,w,h=map(int,mu.toBbox(rle))
            decoded[i]=(x,y,w,h,mu.decode(rle)[y:y+h,x:x+w].astype(bool))
        bounds.append(decoded[i])
    x=min(b[0] for b in bounds);y=min(b[1] for b in bounds)
    right=max(b[0]+b[2] for b in bounds);bottom=max(b[1]+b[3] for b in bounds)
    count=np.zeros((bottom-y,right-x),np.uint8)
    for px,py,w,h,mask in bounds:
        count[py-y:py-y+h,px-x:px-x+w]+=mask
    required=1 if rule=='union' else len(members)//2+1
    return x,y,count>=required


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--low-confidence',action='store_true',help='Second fixed plan: conf .05/.10, support 3/4')
    args=parser.parse_args()
    cv2.setNumThreads(1);started=time.monotonic()
    with VAL.open(newline='') as f:selection=list(csv.DictReader(f))
    ids={r['image_id'] for r in selection};images={Path(r['file_name']).stem for r in selection}
    truth=read_rows(GT,gt=True,selected=ids)
    cache,_=load_cache(RESULTS/'raw_views.jsonl',images)
    confidences,supports=([.05,.10],[3,4]) if args.low_confidence else ([.15,.20],[2,3])
    settings=[dict(name=f'object_c{c}_iou{i}_support{s}_{r}',confidence=c,iou=i,support=s,rule=r)
              for c in confidences for i in [.3,.5] for s in supports for r in ['union','majority']]
    assert len(settings)==16
    predictions={s['name']:[] for s in settings}
    for number,image in enumerate(sorted(images),1):
        proposals=[]
        for view,instances in cache[image].items():
            for p in instances:
                if p['confidence']>=min(confidences):
                    proposals.append(dict(view=view,confidence=p['confidence'],
                        rle={'size':[SIZE,SIZE],'counts':p['rle'].encode('ascii')}))
        rles=[p['rle'] for p in proposals]
        iou=np.asarray(mu.iou(rles,rles,[0]*len(rles))) if rles else np.empty((0,0))
        group_cache={};roi_cache={};decoded={}
        for setting in settings:
            key=(setting['confidence'],setting['iou'])
            if key not in group_cache:
                groups=correspond(proposals,iou,*key)
                groups=[g for g in groups if len(g)>=2]
                groups.sort(key=lambda g:-float(np.mean([proposals[i]['confidence'] for i in g])))
                group_cache[key]=groups
            covered=np.zeros((SIZE,SIZE),bool)
            output=predictions[setting['name']];index=0
            for members in group_cache[key]:
                if len(members)<setting['support']:continue
                roi_key=(tuple(members),setting['rule'])
                if roi_key not in roi_cache:
                    roi_cache[roi_key]=group_roi(proposals,members,decoded,setting['rule'])
                x,y,mask=roi_cache[roi_key];h,w=mask.shape
                mask=mask & ~covered[y:y+h,x:x+w]
                area=int(mask.sum())
                if area<200:continue
                covered[y:y+h,x:x+w] |= mask
                rle=encode_roi(mask,y,x);index+=1
                assert int(mu.area(rle))==area
                output.append(Filament(f'{image}_{index}',image,image,rle['counts'].decode('ascii'),area))
        if number%15==0:print(number,'/',len(images),'images',flush=True)
    reports=[]
    for setting in settings:
        score=evaluate(truth,predictions[setting['name']],SIZE,SIZE)
        reports.append(dict(**setting,**{k:score[k] for k in ['pq','sq','rq','tp_pairs','fp','fn']}))
    best=max(reports,key=lambda r:r['pq']);saved=None
    baseline=.4566206036185416 if args.low_confidence else .4541833849
    if best['pq']>baseline:
        folder=Path('/workspace/artifacts/instance_view_lowconf_candidates_v1' if args.low_confidence else '/workspace/artifacts/instance_view_candidates_v1');folder.mkdir(exist_ok=False)
        path=folder/(best['name']+'.csv')
        with path.open('x',newline='') as f:
            w=csv.writer(f);w.writerow(['filament_id','segmentation_rle'])
            w.writerows((r.filament_id,r.counts) for r in predictions[best['name']])
        saved=str(path)
    report=dict(scope='Fixed development tuning; no GT-dependent object or pixel choices',
        matching='Greedy confidence seed, at most one mask per view, seed mask IoU cutoff',
        pixel_policy='Mean-confidence group ordering; remove pixels already assigned to earlier groups',
        minimum_area=200,comparisons=reports,best=best,saved_candidate=saved,
        baseline_best=baseline,low_confidence_plan=args.low_confidence,elapsed_seconds=time.monotonic()-started)
    name='instance_view_lowconf_experiments.json' if args.low_confidence else 'instance_view_fusion_experiments.json'
    (ROOT/'docs/evidence'/name).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['best'],indent=2));print('seconds',report['elapsed_seconds'])


if __name__=='__main__':main()
