"""Eighty-four fixed mixed-input ensembles and one reproduced reference.

All candidates are GT-free pixel consensus, ellipse5 closing, then 8CC/area.
The preprocessing identity input is deliberately not counted as a second vote
alongside the identical old base input. This is development tuning only.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from pycocotools import mask as mu

from experiment_view_combinations import component_rles, roi_fixture
from experiment_mask_postprocess import read_rows
from filament_eval import Filament, _check_compressed_counts, evaluate


ROOT = Path(__file__).resolve().parents[1]
OLD = Path('/workspace/artifacts/validation_results_20261027T120958Z_444da7f4')
NEW = Path('/workspace/artifacts/validation_results_20261027T125130Z_2312e759')
GT = Path('/workspace/artifacts/solar_filament_gt_v1/gt.csv')
VAL = Path('/workspace/attachments/bcb64220-b208-4cd7-9c3f-3f2be0f04167/val_split.csv')
OLD_VIEWS = ('base','scale','hflip','vflip')
NEW_VIEWS = ('limb','clahe16','clahe32','percentile','unsharp','gaussian','bilateral','blackhat')
BASE = ('base','scale','vflip')
THRESHOLDS = (.15,.20,.25)
REFERENCE_PQ = .4571518474011121
SIZE = 2048


def load(path, images, views):
    cache=defaultdict(dict);records=masks=capped=0
    with path.open() as handle:
        for line in handle:
            row=json.loads(line);image,view=row['image'],row['view']
            assert image in images and view in views
            assert view not in cache[image]
            assert row['shape']==[SIZE,SIZE] and row['coordinate_space']=='original'
            for instance in row['instances']:
                confidence=instance['confidence']
                assert np.isfinite(confidence) and .05<=confidence<=1
                assert _check_compressed_counts(instance['rle'],SIZE*SIZE)>0
            cache[image][view]=row['instances'];records+=1;masks+=len(row['instances'])
            capped+=bool(row.get('max_det_reached',False))
    assert set(cache)==images
    assert all(set(v)==set(views) for v in cache.values())
    assert records==len(images)*len(views)
    return cache,dict(records=records,masks=masks,capped_records=capped,
                      sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def plan():
    settings=[]
    for extra in NEW_VIEWS[1:]:
        for votes in (2,3):
            for confidence in THRESHOLDS:
                for area in (200,300):
                    settings.append(dict(name=f'add_{extra}_v{votes}_c{confidence:.2f}_a{area}',
                        views=list(BASE)+[extra],required=votes,confidence=confidence,minimum_area=area))
    settings.append(dict(name='reference_base_scale_vflip_v2_c0.20_a300_close5',
                        views=list(BASE),required=2,confidence=.20,minimum_area=300))
    assert len(settings)==85 and len({s['name'] for s in settings})==85
    assert all('limb' not in s['views'] for s in settings)
    return settings


def union_map(instances, confidence):
    rles=[{'size':[SIZE,SIZE],'counts':r['rle'].encode('ascii')}
          for r in instances if r['confidence']>=confidence]
    return mu.decode(mu.merge(rles)).astype(bool) if rles else np.zeros((SIZE,SIZE),bool)


def main():
    started=time.monotonic();cv2.setNumThreads(1);roi_fixture()
    with VAL.open(newline='') as f:selection=list(csv.DictReader(f))
    groups={r['image_id'] for r in selection};images={Path(r['file_name']).stem for r in selection}
    truth=read_rows(GT,gt=True,selected=groups)
    assert len(truth)==708 and len(groups)==101 and len(images)==91
    before=json.loads((OLD/'summary.json').read_text())
    after=json.loads((NEW/'summary.json').read_text())
    assert before['source_sha256']==after['source_sha256']
    old,old_stats=load(OLD/'raw_views.jsonl',images,OLD_VIEWS)
    new,new_stats=load(NEW/'raw_preprocess_views.jsonl',images,NEW_VIEWS)
    settings=plan();predictions={s['name']:[] for s in settings}
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    for number,image in enumerate(sorted(images),1):
        sources={view:old[image][view] for view in BASE}
        sources.update({view:new[image][view] for view in NEW_VIEWS[1:]})
        maps={(view,c):union_map(instances,c) for view,instances in sources.items() for c in THRESHOLDS}
        base_votes={c:sum((maps[v,c].astype(np.uint8) for v in BASE),np.zeros((SIZE,SIZE),np.uint8))
                    for c in THRESHOLDS}
        components={}
        for spec in settings:
            key=(tuple(spec['views']),spec['required'],spec['confidence'])
            if key not in components:
                c=spec['confidence'];count=base_votes[c]
                if len(spec['views'])==4:count=count+maps[spec['views'][-1],c]
                foreground=(count>=spec['required']).astype(np.uint8)
                changed=cv2.morphologyEx(foreground,cv2.MORPH_CLOSE,kernel)
                components[key]=component_rles(changed,200)
            output=predictions[spec['name']]
            for index,(rle,area) in enumerate(components[key],1):
                if area>=spec['minimum_area']:
                    output.append(Filament(f'{image}_{index}',image,image,rle['counts'].decode('ascii'),area))
        if number==1 or number%15==0 or number==91:
            print(f'mixed preprocessing {number}/91 elapsed={time.monotonic()-started:.1f}s',flush=True)
    reports=[]
    for spec in settings:
        score=evaluate(truth,predictions[spec['name']],SIZE,SIZE)
        reports.append(dict(**spec,**{k:score[k] for k in ['pq','sq','rq','tp_pairs','fp','fn']},
            overlap_images=0,disjoint_by_construction=True,
            extra_pairs_beyond_unique_gt=score['tp_pairs']-(708-score['fn']),
            prediction_rows=len(predictions[spec['name']])))
    reference=reports[-1];assert abs(reference['pq']-REFERENCE_PQ)<1e-12,reference
    best=max(reports,key=lambda r:r['pq']);improved=best['pq']>REFERENCE_PQ+1e-10;saved=None
    if improved:
        folder=Path('/workspace/artifacts/mixed_prep_candidates_v1');folder.mkdir(exist_ok=False)
        path=folder/(best['name']+'.csv')
        with path.open('x',newline='') as handle:
            writer=csv.writer(handle);writer.writerow(['filament_id','segmentation_rle'])
            writer.writerows((r.filament_id,r.counts) for r in predictions[best['name']])
        saved=str(path)
    result=dict(scope='Eighty-four fixed mixed-input ensembles plus reference; same development set, not independent CV or LB',
        physical_images=91,annotation_groups=101,gt_instances=708,old_cache=old_stats,new_cache=new_stats,
        input_source_sha256=before['source_sha256'],comparisons=85,new_settings=84,
        rule='Original base+scale+vflip plus exactly one new preprocessing view; binary voting -> ellipse5 closing -> 8CC/min-area',
        deduplication='New limb identity omitted; old base counted only once',
        reference_reproduced=True,experiments=reports,best=best,improved_over_reference=improved,
        saved_candidate_csv=saved,target_exceeded=best['pq']>.5,
        fixture_checks='Imported native-coordinate ROI encoder fixtures passed',
        elapsed_seconds=time.monotonic()-started)
    (ROOT/'docs/evidence/mixed_preprocessing_experiments.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(best=best,saved_candidate_csv=saved,seconds=result['elapsed_seconds']),indent=2))


if __name__=='__main__':main()
