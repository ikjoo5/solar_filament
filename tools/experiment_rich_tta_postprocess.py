"""Ninety-six fixed standalone native U-Net probability/TTA postprocess rules.

All rules and predictions are built before reading GT. This is repeated tuning
on the fixed development set, not independent CV or leaderboard performance.
"""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from pycocotools import mask as mu

from experiment_mask_postprocess import read_rows, overlap_stats
from experiment_view_combinations import GT, VAL, SIZE, ROOT, component_rles
from filament_eval import Filament, _check_compressed_counts, evaluate

SOURCE = Path('/workspace/artifacts/best84_rich_predictions_tta/predictions.jsonl')
OUT = Path('/workspace/artifacts/rich_tta_postprocess_v1')


def plan():
    settings = [dict(name=f'{variant}_t{t:g}_a{area}', rule='area', variant=variant,
                     threshold=t, min_area=area)
                for variant in ('base', 'tta') for t in (.2, .35, .5, .65, .8)
                for area in (100, 200, 300, 400, 600, 800)]
    settings += [dict(name=f'tta_t{t:g}_fgmean{c:g}', rule='foreground', variant='tta',
                      threshold=t, min_area=300, minimum=c)
                 for t in (.35,.5) for c in (.8,.9,.95)]
    settings += [dict(name=f'tta_t{t:g}_spinemean{c:g}', rule='spine', variant='tta',
                      threshold=t, min_area=300, minimum=c)
                 for t in (.35,.5) for c in (.3,.5,.7,.9)]
    settings += [dict(name=f'tta_t{t:g}_views{count}', rule='views', variant='tta',
                      threshold=t, min_area=300, minimum_views=count, minimum_coverage=.5)
                 for t in (.35,.5) for count in (2,3,4)]
    settings += [dict(name=f'tta_t{t:g}_baseiou{c:g}', rule='stability', variant='tta',
                      threshold=t, min_area=300, minimum=c)
                 for t in (.35,.5) for c in (.3,.5,.7)]
    settings += [dict(name=f'tta_t{t:g}_{operation}', rule='morphology', variant='tta',
                      threshold=t, min_area=300, operation=operation)
                 for t in (.35,.5) for operation in ('close3','close5','open3','blur0.7','blur1.2')]
    assert len(settings)==96 and len({s['name'] for s in settings})==96
    return settings


def convert(image, components):
    return [Filament(f'{image}_{i}', image, image, c['rle']['counts'], int(c['area']))
            for i,c in enumerate(components, 1)]


def morphology(image, union, setting):
    mask = mu.decode(union)
    operation = setting['operation']
    if operation.startswith('blur'):
        foreground=cv2.GaussianBlur(mask.astype(np.float32),(0,0),float(operation[4:]))>=.5
    else:
        kernel = np.ones((int(operation[-1]),)*2,np.uint8)
        code = cv2.MORPH_CLOSE if operation.startswith('close') else cv2.MORPH_OPEN
        foreground=cv2.morphologyEx(mask,code,kernel)>0
    return [Filament(f'{image}_{i}',image,image,rle['counts'].decode('ascii'),area)
            for i,(rle,area) in enumerate(component_rles(foreground,setting['min_area']),1)]


def main():
    started=time.monotonic();cv2.setNumThreads(1)
    settings=plan();OUT.mkdir(exist_ok=False)
    (OUT/'predeclared_plan.json').write_text(json.dumps(settings,indent=2)+'\n')
    with VAL.open(newline='') as f:selection=list(csv.DictReader(f))
    ids={r['image_id'] for r in selection};images={Path(r['file_name']).stem for r in selection}
    assert len(ids)==101 and len(images)==91
    records=[json.loads(line) for line in SOURCE.open()]
    assert len(records)==91 and {Path(r['file_name']).stem for r in records}==images
    outputs={s['name']:[] for s in settings};counts=Counter()
    for number,record in enumerate(records,1):
        image=Path(record['file_name']).stem
        base=record['components']['0.5'];br=[c['rle'] for c in base]
        views=record['view_unions']
        for variant in ('base','tta'):
            data=record if variant=='base' else record['tta']
            for key,components in data['components'].items():
                for component in components:
                    assert _check_compressed_counts(component['rle']['counts'],SIZE*SIZE)==component['area']
                    for head in ('foreground','spine'):
                        stats=component['stats'][head]
                        assert all(np.isfinite(stats[k]) and 0<=stats[k]<=1 for k in ('mean','max','p90'))
                        assert stats['mean']<=stats['max'] and stats['p90']<=stats['max']
                matching=[s for s in settings if s['variant']==variant and f"{s['threshold']:g}"==key]
                if not matching:continue
                cr=[c['rle'] for c in components]
                biou=(np.asarray(mu.iou(cr,br,[0]*len(br))) if cr and br
                       else np.zeros((len(cr),len(br))))
                stability=biou.max(axis=1) if br else np.zeros(len(cr))
                cover=np.array([[int(mu.area(mu.merge([c['rle'],view],intersect=True)))/c['area']
                                 for view in views.values()] for c in components])
                for setting in matching:
                    if setting['rule']=='morphology':
                        rows=morphology(image,data['unions']['foreground'][key],setting)
                    else:
                        kept=[]
                        for i,c in enumerate(components):
                            if c['area']<setting['min_area']:continue
                            rule=setting['rule']
                            if rule in ('foreground','spine') and c['stats'][rule]['mean']<setting['minimum']:continue
                            if rule=='stability' and stability[i]<setting['minimum']:continue
                            if rule=='views' and (cover[i]>=setting['minimum_coverage']).sum()<setting['minimum_views']:continue
                            kept.append(c)
                        rows=convert(image,kept)
                    outputs[setting['name']].extend(rows)
        if number%20==0:print('built',number,'/91',flush=True)
    # All prediction sets exist before annotations are opened.
    truth=read_rows(GT,gt=True,selected=ids);assert len(truth)==708
    reports=[]
    for setting in settings:
        rows=outputs[setting['name']];score=evaluate(truth,rows,SIZE,SIZE)
        overlap=overlap_stats(rows)
        assert not any(overlap.values())
        assert score['tp_pairs']==708-score['fn']
        reports.append(dict(**setting,**{k:score[k] for k in ('pq','sq','rq','tp_pairs','fp','fn')},
                            prediction_rows=len(rows),**overlap))
    best=max(reports,key=lambda r:r['pq'])
    fn=min(reports,key=lambda r:(r['fn'],-r['pq']))
    saved=[]
    for setting in {r['name']:r for r in (best,fn)}.values():
        path=OUT/(setting['name']+'.csv')
        with path.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['filament_id','segmentation_rle'])
            w.writerows((r.filament_id,r.counts) for r in outputs[setting['name']])
        check=evaluate(truth,read_rows(path),SIZE,SIZE)
        assert abs(check['pq']-setting['pq'])<1e-12
        saved.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          name=setting['name'],score_recomputed_after_save=True))
    baseline=next(r for r in reports if r['name']=='base_t0.5_a300')
    assert abs(baseline['pq']-.4299305777121099)<1e-12
    report=dict(scope='96 fixed standalone same-development rules; not independent CV or LB',
                source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                physical_images=91,annotation_groups=101,gt_instances=708,settings=reports,
                baseline=baseline,best=best,lowest_fn=fn,saved_candidates=saved,
                target_exceeded=best['pq']>.5,elapsed_seconds=time.monotonic()-started)
    (ROOT/'docs/evidence/rich_tta_postprocess.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('baseline','best','lowest_fn','saved_candidates','elapsed_seconds')},indent=2))


if __name__=='__main__':main()
