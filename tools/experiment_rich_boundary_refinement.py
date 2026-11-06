"""Twenty-seven fixed instance-preserving base/TTA boundary refinements.

Proposals are assigned uniquely from prediction IoU. GT is opened only after
all prediction sets exist. Scores remain same-development tuning results.
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
from experiment_native_cross_model import matched_ids
from experiment_view_combinations import GT, VAL, ROOT, SIZE, encode_roi
from filament_eval import Filament, evaluate
from refine_masks_with_pixels import resolve_rois

SOURCE=Path('/workspace/artifacts/best84_rich_predictions_tta/predictions.jsonl')
ORIGINAL=Path('/workspace/artifacts/native_high_confidence_candidates_v1/native_t0.5_a300_weak0.6_families1.csv')
OUT=Path('/workspace/artifacts/rich_boundary_refinement_v1')
EXPECTED_SETTINGS=27
REPORT_NAME='rich_boundary_refinement.json'


def plan():
    return [dict(name=f't{t:g}_fgmean{confidence:g}_guard{guard:g}', threshold=t,
                 minimum_foreground_mean=confidence, minimum_match_iou=.3,
                 minimum_original_iou=guard, area_ratio=[.5,1.5], fusion='2of3majority')
            for t in (.35,.5,.65) for confidence in (.8,.9,.95) for guard in (.7,.85,.95)]


def mask_bbox(rle):
    return tuple(map(int,mu.toBbox(rle)))


def bounding_region(boxes):
    left=min(b[0] for b in boxes);top=min(b[1] for b in boxes)
    right=max(b[0]+b[2] for b in boxes);bottom=max(b[1]+b[3] for b in boxes)
    return np.s_[top:bottom,left:right]


def place(local, bbox, region):
    result=np.zeros((region[0].stop-region[0].start,region[1].stop-region[1].start),bool)
    x,y,w,h=bbox
    top=y-region[0].start;left=x-region[1].start
    result[top:top+h,left:left+w]=local
    return result


def guarded_majority(original, first, second, setting):
    candidate=(original.astype(np.uint8)+first+second)>=2
    area=int(candidate.sum());original_area=int(original.sum())
    ratio=area/original_area
    intersection=int((candidate&original).sum());union=int((candidate|original).sum())
    if not setting['area_ratio'][0]<=ratio<=setting['area_ratio'][1] or intersection/union<setting['minimum_original_iou']:
        return original.copy(),'fallback_excessive_change'
    return candidate,'changed' if np.any(candidate!=original) else 'unchanged'


def main():
    started=time.monotonic();settings=plan();assert len(settings)==EXPECTED_SETTINGS
    OUT.mkdir(exist_ok=False);(OUT/'predeclared_plan.json').write_text(json.dumps(settings,indent=2)+'\n')
    with VAL.open(newline='') as f:selection=list(csv.DictReader(f))
    groups={r['image_id'] for r in selection};images={Path(r['file_name']).stem for r in selection}
    assert len(images)==91 and len(groups)==101
    original=read_rows(ORIGINAL);assert len(original)==630 and not overlap_stats(original)['overlap_images']
    grouped=group_rows(original)
    records={Path(r['file_name']).stem:r for r in map(json.loads,SOURCE.open())}
    assert set(records)==images
    predictions={s['name']:[] for s in settings};status={s['name']:Counter() for s in settings}
    unique_builds=0
    for number,image in enumerate(sorted(images),1):
        members=grouped.get(image,[]);rles=[r.rle(SIZE,SIZE) for r in members]
        bboxes=[mask_bbox(rle) for rle in rles]
        local_originals=[mu.decode(rle)[y:y+h,x:x+w].astype(bool)
                         for rle,(x,y,w,h) in zip(rles,bboxes)]
        proposals={};matrices={};assignments={};boxes={};decoded={}
        for t in sorted({setting['threshold'] for setting in settings}):
            for model in ('base','tta'):
                data=records[image] if model=='base' else records[image]['tta']
                key=model,t;components=data['components'][f'{t:g}'];proposals[key]=components
                nr=[c['rle'] for c in components]
                matrices[key]=(np.asarray(mu.iou(rles,nr,[0]*len(nr))) if rles and nr
                               else np.zeros((len(rles),len(nr))))
                assignments[key]=matrices[key].argmax(axis=0) if rles and nr else np.empty(0,int)
                boxes[key]=[mask_bbox(rle) for rle in nr]
        def proposal_roi(model,t,index):
            cache_key=model,t,index
            if cache_key not in decoded:
                x,y,w,h=boxes[model,t][index]
                decoded[cache_key]=mu.decode(proposals[model,t][index]['rle'])[y:y+h,x:x+w].astype(bool)
            return decoded[cache_key]
        built={}
        for setting in settings:
            t=setting['threshold'];matches=[]
            for i,row in enumerate(members):
                pair=[]
                for model in ('base','tta'):
                    key=model,t
                    valid=[j for j,c in enumerate(proposals[key])
                           if assignments[key][j]==i and matrices[key][i,j]>=setting['minimum_match_iou']
                           and c['stats']['foreground']['mean']>=setting['minimum_foreground_mean']]
                    pair.append(max(valid,key=lambda j:proposals[key][j]['stats']['foreground']['mean']) if valid else None)
                matches.append(tuple(pair))
            signature=t,setting['minimum_original_iou'],tuple(matches)
            if signature not in built:
                candidates=[];originals=[];regions=[];counters=Counter()
                for i,(row,pair) in enumerate(zip(members,matches)):
                    candidate_boxes=[bboxes[i]]+[boxes[model,t][j] for model,j in zip(('base','tta'),pair) if j is not None]
                    region=bounding_region(candidate_boxes)
                    local=place(local_originals[i],bboxes[i],region)
                    if None in pair:
                        candidate=local.copy();decision='fallback_missing_peer'
                    else:
                        peers=[place(proposal_roi(model,t,j),boxes[model,t][j],region)
                               for model,j in zip(('base','tta'),pair)]
                        candidate,decision=guarded_majority(local,*peers,setting)
                    candidates.append(candidate);originals.append(local);regions.append(region);counters[decision]+=1
                resolved=resolve_rois(candidates,originals,regions,(SIZE,SIZE))
                rows=[]
                for row,mask,local,region in zip(members,resolved,originals,regions):
                    assert mask.any()
                    if np.array_equal(mask,local):rows.append(row);continue
                    ys,xs=np.nonzero(mask);top,left=int(ys.min()),int(xs.min())
                    encoded=encode_roi(mask[top:int(ys.max())+1,left:int(xs.max())+1],region[0].start+top,region[1].start+left)
                    rows.append(Filament(row.filament_id,image,image,encoded['counts'].decode('ascii'),int(mu.area(encoded))))
                built[signature]=rows,counters;unique_builds+=1
            rows,counters=built[signature];predictions[setting['name']].extend(rows);status[setting['name']].update(counters)
        if number%15==0:print('boundary',number,'/91',flush=True)
    # GT cannot affect assignment, confidence filtering or geometry.
    truth=read_rows(GT,gt=True,selected=groups);assert len(truth)==708
    reference=evaluate(truth,original,SIZE,SIZE);assert abs(reference['pq']-.46556248016002527)<1e-12
    before=matched_ids(truth,original)
    reports=[dict(name='original',rule='identity',**{k:reference[k] for k in ('pq','sq','rq','tp_pairs','fp','fn')},
                  prediction_rows=len(original),lost_gt=0,recovered_gt=0)]
    for setting in settings:
        rows=predictions[setting['name']];assert len(rows)==630
        score=evaluate(truth,rows,SIZE,SIZE);overlap=overlap_stats(rows)
        assert not any(overlap.values()) and score['tp_pairs']==708-score['fn']
        after=matched_ids(truth,rows)
        reports.append(dict(**setting,**{k:score[k] for k in ('pq','sq','rq','tp_pairs','fp','fn')},
                            prediction_rows=len(rows),statuses=dict(status[setting['name']]),**overlap,
                            lost_gt=len(before-after),recovered_gt=len(after-before),
                            lost_gt_ids=sorted(before-after),recovered_gt_ids=sorted(after-before)))
    best=max(reports,key=lambda r:r['pq']);best_actual=max(reports[1:],key=lambda r:r['pq'])
    preserving=max([r for r in reports if r['lost_gt']==0],key=lambda r:r['pq'])
    saved=[]
    for setting in {r['name']:r for r in (best,preserving,best_actual) if r['name']!='original'}.values():
        path=OUT/(setting['name']+'.csv')
        with path.open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['filament_id','segmentation_rle'])
            writer.writerows((r.filament_id,r.counts) for r in predictions[setting['name']])
        check=evaluate(truth,read_rows(path),SIZE,SIZE);assert abs(check['pq']-setting['pq'])<1e-12
        saved.append(dict(name=setting['name'],path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),score_recomputed_after_save=True,
                          adopted=setting['pq']>reference['pq']))
    report=dict(scope=f'{EXPECTED_SETTINGS} fixed same-development matched-object majority rules; no independent CV or LB',
                source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),original_sha256=hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),
                physical_images=91,annotation_groups=101,gt_instances=708,original_masks=630,
                assignment='Each component uniquely assigned to highest original IoU>=.3; highest foreground mean per model; require both base/TTA; guarded2of3majority then nearest-original ownership',
                experiments=reports,best=best,best_actual=best_actual,best_preserving_all_original_matches=preserving,
                saved_candidates=saved,unique_image_builds=unique_builds,target_exceeded=best['pq']>.5,elapsed_seconds=time.monotonic()-started)
    (ROOT/'docs/evidence'/REPORT_NAME).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('best','best_actual','best_preserving_all_original_matches','saved_candidates','elapsed_seconds')},indent=2))


if __name__=='__main__':main()
