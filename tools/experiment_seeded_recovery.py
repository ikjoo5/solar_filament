"""Bounded high-confidence seed / low-confidence region recovery experiments."""
import csv
import json
from pathlib import Path
import time

import cv2
import numpy as np
from pycocotools import mask as mu

from experiment_view_combinations import ROOT, RESULTS, GT, VAL, SIZE, load_cache, component_rles
from experiment_mask_postprocess import read_rows
from filament_eval import Filament, evaluate


def recover(seed, allowed, radius=None):
    allowed=allowed.copy()
    if radius is not None:
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,radius*2+1))
        allowed &= cv2.dilate(seed.astype(np.uint8),kernel)>0
    n,labels=cv2.connectedComponents(allowed.astype(np.uint8),connectivity=8)
    selected=np.unique(labels[seed & allowed])
    selected=selected[selected!=0]
    lookup=np.zeros(n,bool);lookup[selected]=True
    return lookup[labels]


def fixture():
    allowed=np.zeros((20,25),bool);allowed[2:8,2:8]=True;allowed[12:18,15:22]=True
    seed=np.zeros_like(allowed);seed[4,4]=True
    recovered=recover(seed,allowed)
    assert int(recovered.sum())==36 and not recovered[13,16]
    assert not recover(np.zeros_like(seed),allowed).any()
    assert int(recover(seed,allowed,1).sum())<36


def main():
    started=time.monotonic();cv2.setNumThreads(1);fixture()
    with VAL.open(newline='') as f:selection=list(csv.DictReader(f))
    groups={r['image_id'] for r in selection};images={Path(r['file_name']).stem for r in selection}
    truth=read_rows(GT,gt=True,selected=groups);cache,_=load_cache(RESULTS/'raw_views.jsonl',images)
    settings=[dict(name=f'seed{hi}_grow{lo}_v{v}_r{r}_a{a}',high=hi,low=lo,low_votes=v,radius=r,area=a)
              for hi in [.20,.25] for lo in [.10,.15] for v in [2,3] for r in [None,5] for a in [200,300]]
    assert len(settings)==32
    predictions={s['name']:[] for s in settings}
    for number,image in enumerate(sorted(images),1):
        counts={c:np.zeros((SIZE,SIZE),np.uint8) for c in [.10,.15,.20,.25]}
        for instances in cache[image].values():
            maps={c:np.zeros((SIZE,SIZE),bool) for c in counts}
            for instance in instances:
                if instance['confidence']<.10:continue
                rle={'size':[SIZE,SIZE],'counts':instance['rle'].encode('ascii')}
                x,y,w,h=map(int,mu.toBbox(rle));mask=mu.decode(rle)[y:y+h,x:x+w].astype(bool)
                for c in maps:
                    if instance['confidence']>=c:maps[c][y:y+h,x:x+w]|=mask
            for c in counts:counts[c]+=maps[c]
        converted={}
        for spec in settings:
            key=(spec['high'],spec['low'],spec['low_votes'],spec['radius'])
            if key not in converted:
                foreground=recover(counts[spec['high']]>=3,counts[spec['low']]>=spec['low_votes'],spec['radius'])
                converted[key]=component_rles(foreground,200)
            rows=predictions[spec['name']]
            for index,(rle,area) in enumerate(converted[key],1):
                if area>=spec['area']:
                    rows.append(Filament(f'{image}_{index}',image,image,rle['counts'].decode('ascii'),area))
        if number%20==0:print(number,'/',len(images),'images',flush=True)
    results=[]
    for spec in settings:
        score=evaluate(truth,predictions[spec['name']],SIZE,SIZE)
        results.append(dict(**spec,**{k:score[k] for k in ['pq','sq','rq','tp_pairs','fp','fn']}))
    best=max(results,key=lambda r:r['pq']);saved=None
    if best['pq']>.4566206036185416:
        out=Path('/workspace/artifacts/seeded_recovery_candidates_v1');out.mkdir(exist_ok=False)
        path=out/(best['name']+'.csv')
        with path.open('x',newline='') as f:
            w=csv.writer(f);w.writerow(['filament_id','segmentation_rle'])
            w.writerows((p.filament_id,p.counts) for p in predictions[best['name']])
        saved=str(path)
    report=dict(scope='Fixed 32-setting development tuning; no GT-dependent pixel/object choices',
        method='3/4 high-confidence view consensus seeds; keep low-consensus components containing seed pixels; optional radius-limited growth',
        all_outputs_disjoint=True,experiments=results,best=best,saved_candidate=saved,elapsed_seconds=time.monotonic()-started)
    (ROOT/'docs/evidence/seeded_recovery_experiments.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(best,indent=2));print('seconds',report['elapsed_seconds'])


if __name__=='__main__':main()
