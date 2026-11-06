"""Forty-eight predeclared view/threshold comparisons from cached predictions.

No model inference or GT-dependent image/object choices are made. All candidates
use connected components and hence have disjoint masks within each image.
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

from filament_eval import Filament, _check_compressed_counts, evaluate
from experiment_mask_postprocess import read_rows


ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path('/workspace/artifacts/validation_results_20261027T120958Z_444da7f4')
GT = Path('/workspace/artifacts/solar_filament_gt_v1/gt.csv')
VAL = Path('/workspace/attachments/bcb64220-b208-4cd7-9c3f-3f2be0f04167/val_split.csv')
VIEWS = ('base', 'scale', 'hflip', 'vflip')
CONFIDENCES = (.15, .20, .25, .30, .35)
SIZE = 2048


def plan():
    settings = []
    for view in VIEWS:
        for threshold in (.15,.20,.25):
            for area in (200,300):
                settings.append(dict(name=f'{view}_c{threshold:.2f}_a{area}', views=[view],
                                     threshold=threshold, required=1, minimum_area=area))
    for threshold in (.20,.25,.30,.35):
        for area in (200,300):
            settings.append(dict(name=f'all4_3votes_c{threshold:.2f}_a{area}', views=list(VIEWS),
                                 threshold=threshold, required=3, minimum_area=area))
    for omitted in VIEWS:
        for threshold in (.20,.25):
            for area in (200,300):
                settings.append(dict(name=f'omit_{omitted}_2votes_c{threshold:.2f}_a{area}',
                                     views=[v for v in VIEWS if v!=omitted], threshold=threshold,
                                     required=2, minimum_area=area))
    assert len(settings)==48
    return settings


def encode_roi(mask, top, left, height=SIZE, width=SIZE):
    """Embed a small binary ROI in native column-major RLE without a dense canvas."""
    counts, previous = [], 0
    for col in range(mask.shape[1]):
        transitions = np.flatnonzero(np.diff(np.r_[0,mask[:,col].astype(np.int8),0]))
        for start, end in transitions.reshape(-1,2):
            first = (left+col)*height+top+int(start)
            last = (left+col)*height+top+int(end)
            counts.extend([first-previous,last-first])
            previous = last
    counts.append(height*width-previous)
    rle = mu.frPyObjects({'size':[height,width],'counts':counts},height,width)
    return rle


def component_rles(mask, minimum=200):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8),connectivity=8)
    result = []
    for label in range(1,n):
        x,y,w,h,area = map(int,stats[label])
        if area >= minimum:
            rle = encode_roi(labels[y:y+h,x:x+w]==label,y,x)
            assert int(mu.area(rle))==area
            result.append((rle,area))
    return result


def load_cache(path, expected_images):
    grouped = defaultdict(dict)
    total = 0
    for line in path.read_text().splitlines():
        row = json.loads(line)
        image,view = row['image'],row['view']
        assert image in expected_images and view in VIEWS
        assert row['shape']==[SIZE,SIZE] and row['coordinate_space']=='original'
        assert view not in grouped[image], (image,view,'duplicate')
        for instance in row['instances']:
            assert np.isfinite(instance['confidence']) and .05<=instance['confidence']<=1
            assert _check_compressed_counts(instance['rle'],SIZE*SIZE)>0
            total+=1
        grouped[image][view]=row['instances']
    assert set(grouped)==expected_images and all(set(v)==set(VIEWS) for v in grouped.values())
    return grouped,total


def roi_fixture():
    rng=np.random.default_rng(2026)
    for _ in range(15):
        mask=rng.random((11,17))>.7
        rle=encode_roi(mask,7,9,40,50)
        canvas=np.zeros((40,50),np.uint8);canvas[7:18,9:26]=mask
        np.testing.assert_array_equal(mu.decode(rle),canvas)
        assert _check_compressed_counts(rle['counts'].decode('ascii'),40*50)==int(canvas.sum())


def main():
    started=time.monotonic();cv2.setNumThreads(1);roi_fixture()
    with VAL.open(newline='') as f: selection=list(csv.DictReader(f))
    groups={r['image_id'] for r in selection}
    images={Path(r['file_name']).stem for r in selection}
    truth=read_rows(GT,gt=True,selected=groups)
    assert len(truth)==708 and len(images)==91
    cache,total=load_cache(RESULTS/'raw_views.jsonl',images)
    settings=plan();predictions={s['name']:[] for s in settings}
    # The binary masks for each threshold/view are built once per physical image.
    for number,image in enumerate(sorted(images),1):
        maps={(v,t):np.zeros((SIZE,SIZE),bool) for v in VIEWS for t in CONFIDENCES}
        for view in VIEWS:
            for instance in cache[image][view]:
                confidence=instance['confidence']
                if confidence<min(CONFIDENCES):continue
                rle={'size':[SIZE,SIZE],'counts':instance['rle'].encode('ascii')}
                x,y,w,h=map(int,mu.toBbox(rle))
                mask=mu.decode(rle)[y:y+h,x:x+w].astype(bool)
                for threshold in CONFIDENCES:
                    if confidence>=threshold:maps[view,threshold][y:y+h,x:x+w] |= mask
        # Area200/300 share connected components; encode a component once.
        components={}
        for setting in settings:
            key=(tuple(setting['views']),setting['threshold'],setting['required'])
            if key not in components:
                count=np.zeros((SIZE,SIZE),np.uint8)
                for view in setting['views']:count+=maps[view,setting['threshold']]
                components[key]=component_rles(count>=setting['required'])
            rows=predictions[setting['name']]
            for index,(rle,area) in enumerate(components[key],1):
                if area>=setting['minimum_area']:
                    rows.append(Filament(f'{image}_{index}',image,image,rle['counts'].decode('ascii'),area))
        if number==1 or number%10==0 or number==len(images):
            print(f'Built {number}/{len(images)} images, elapsed {time.monotonic()-started:.1f}s',flush=True)
    reports=[]
    for setting in settings:
        score=evaluate(truth,predictions[setting['name']],SIZE,SIZE)
        reports.append(dict(**setting,**{k:score[k] for k in ['pq','sq','rq','tp_pairs','fp','fn']},
                            unique_matched_gt=708-score['fn'],
                            extra_pairs_beyond_unique_gt=score['tp_pairs']-(708-score['fn']),
                            prediction_rows=len(predictions[setting['name']]),overlap_images=0))
    baseline=next(r for r in reports if r['name']=='all4_3votes_c0.20_a200')
    assert abs(baseline['pq']-.4535254054790343)<1e-12, baseline
    best=max(reports,key=lambda r:r['pq']);saved=None
    if best['pq']>baseline['pq']:
        out=Path('/workspace/artifacts/view_combo_candidates_v1');out.mkdir(exist_ok=False)
        path=out/(best['name']+'.csv')
        with path.open('x',newline='') as f:
            writer=csv.writer(f);writer.writerow(['filament_id','segmentation_rle'])
            writer.writerows((r.filament_id,r.counts) for r in predictions[best['name']])
        saved=str(path)
    output=dict(scope='Forty-eight predeclared same-development-set comparisons, not independent CV or leaderboard gains',
                images=len(images),annotation_groups=len(groups),gt_instances=len(truth),
                cache_records=len(images)*4,cache_instances=total,
                raw_cache_sha256=hashlib.sha256((RESULTS/'raw_views.jsonl').read_bytes()).hexdigest(),
                rules='Global per-setting view/threshold/area only; no GT-dependent object/image selection; no overlapping predictions',
                roi_encoding_fixture_cases=15,baseline_reproduced=baseline,experiments=reports,
                best=best,best_candidate_csv=saved,elapsed_seconds=time.monotonic()-started)
    path=ROOT/'docs/evidence/view_combination_experiments.json'
    path.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'best':best,'saved':saved,'elapsed_seconds':output['elapsed_seconds']},indent=2),flush=True)


if __name__=='__main__':main()
