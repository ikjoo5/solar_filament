"""Final 18 fixed geometry polishes on newly composed TTA-edge masks.

Keep one mask per input, bound changes against the input mask, and resolve
expansion overlap by nearest original ownership. GT is read after all outputs.
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

from experiment_mask_postprocess import group_rows, overlap_stats, read_rows
from experiment_native_cross_model import matched_ids
from experiment_view_combinations import ROOT, GT, VAL, SIZE, encode_roi
from filament_eval import Filament, evaluate
from refine_masks_with_pixels import resolve_rois

SOURCES={name:Path('/workspace/artifacts/rich_boundary_composition_v1')/f'tta_{name}.csv'
         for name in ('preserve','precision')}
ORIGINAL=Path('/workspace/artifacts/native_high_confidence_candidates_v1/native_t0.5_a300_weak0.6_families1.csv')
OUT=Path('/workspace/artifacts/rich_final_smoothing_v1')


def plan():
    return [dict(name=f'{source}_s{sigma:g}_t{threshold:g}',source=source,sigma=sigma,
                 threshold=threshold,minimum_input_iou=.85,area_ratio=[.5,1.5])
            for source in SOURCES for sigma in (.7,1.2,2.) for threshold in (.4,.5,.6)]


def guard(candidate, original):
    ratio=int(candidate.sum())/int(original.sum())
    intersection=int((candidate & original).sum());union=int((candidate | original).sum())
    return .5<=ratio<=1.5 and union>0 and intersection/union>=.85


def main():
    started=time.monotonic();cv2.setNumThreads(1)
    settings=plan();assert len(settings)==18
    OUT.mkdir(exist_ok=False)
    (OUT/'predeclared_plan.json').write_text(json.dumps(settings,indent=2)+'\n')
    selection=list(csv.DictReader(VAL.open()));ids={r['image_id'] for r in selection}
    images={Path(r['file_name']).stem for r in selection}
    inputs={name:read_rows(path) for name,path in SOURCES.items()}
    for rows in inputs.values():
        assert {r.image_id for r in rows}<=images and not any(overlap_stats(rows).values())
    output={s['name']:[] for s in settings};statuses={s['name']:Counter() for s in settings}
    for source,rows in inputs.items():
        for image,members in sorted(group_rows(rows).items()):
            regions=[];originals=[]
            for row in members:
                x,y,w,h=map(int,mu.toBbox(row.rle(SIZE,SIZE)))
                region=np.s_[max(0,y-9):min(SIZE,y+h+9),max(0,x-9):min(SIZE,x+w+9)]
                regions.append(region);originals.append(mu.decode(row.rle(SIZE,SIZE))[region].astype(bool))
            for setting in [s for s in settings if s['source']==source]:
                candidates=[]
                for original in originals:
                    candidate=cv2.GaussianBlur(original.astype(np.float32),(0,0),setting['sigma'],
                        borderType=cv2.BORDER_CONSTANT)>=setting['threshold']
                    if not guard(candidate,original):
                        candidate=original.copy();status='preownership_guard_fallback'
                    else:
                        status='changed' if np.any(candidate!=original) else 'unchanged'
                    statuses[setting['name']][status]+=1;candidates.append(candidate)
                resolved=resolve_rois(candidates,originals,regions,(SIZE,SIZE))
                for row,mask,original,region in zip(members,resolved,originals,regions):
                    if not guard(mask,original):
                        mask=original.copy();statuses[setting['name']]['postownership_guard_fallback']+=1
                    assert mask.any() and guard(mask,original)
                    yy,xx=np.nonzero(mask);top,left=int(yy.min()),int(xx.min())
                    encoded=encode_roi(mask[top:int(yy.max())+1,left:int(xx.max())+1],
                        region[0].start+top,region[1].start+left)
                    output[setting['name']].append(Filament(row.filament_id,image,image,
                        encoded['counts'].decode('ascii'),int(mu.area(encoded))))
    # Complete predictions exist before loading any ground truth.
    truth=read_rows(GT,gt=True,selected=ids);assert len(truth)==708 and len(ids)==101 and len(images)==91
    original=read_rows(ORIGINAL);original_matches=matched_ids(truth,original)
    assert len(original)==630 and len(original_matches)==483
    input_matches={name:matched_ids(truth,rows) for name,rows in inputs.items()}
    input_scores={name:evaluate(truth,rows,SIZE,SIZE) for name,rows in inputs.items()}
    reports=[]
    for setting in settings:
        rows=output[setting['name']];assert len(rows)==len(inputs[setting['source']])
        score=evaluate(truth,rows,SIZE,SIZE);audit=overlap_stats(rows);matches=matched_ids(truth,rows)
        assert not any(audit.values()) and score['tp_pairs']==708-score['fn']
        im=input_matches[setting['source']]
        reports.append(dict(**setting,**{k:score[k] for k in ('pq','sq','rq','tp_pairs','fp','fn')},
            physical_masks=len(rows),statuses=dict(statuses[setting['name']]),**audit,
            delta_input_pq=score['pq']-input_scores[setting['source']]['pq'],
            preserved_original_gt=len(matches & original_matches),recovered_original_gt=len(matches-original_matches),
            lost_original_gt=len(original_matches-matches),lost_original_gt_ids=sorted(original_matches-matches),
            recovered_original_gt_ids=sorted(matches-original_matches),
            recovered_input_gt=len(matches-im),lost_input_gt=len(im-matches),
            recovered_input_gt_ids=sorted(matches-im),lost_input_gt_ids=sorted(im-matches)))
    best=max(reports,key=lambda r:r['pq'])
    preserving=[r for r in reports if r['lost_original_gt']==0 and r['lost_input_gt']==0]
    best_preserving=max(preserving,key=lambda r:r['pq']) if preserving else None
    targets=[r for r in (best,best_preserving) if r is not None and r['delta_input_pq']>1e-12]
    saved=[]
    for result in {r['name']:r for r in targets}.values():
        path=OUT/(result['name']+'.csv')
        with path.open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['filament_id','segmentation_rle'])
            writer.writerows((r.filament_id,r.counts) for r in output[result['name']])
        score=evaluate(truth,read_rows(path),SIZE,SIZE);assert abs(score['pq']-result['pq'])<1e-12
        saved.append(dict(name=result['name'],path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    report=dict(scope='18 final global Gaussian polishes on newly composed TTA masks; fixeddevelopment, notCV/LB',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in list(SOURCES.values())+[ORIGINAL]},
        original=dict(path=str(ORIGINAL),**{k:v for k,v in evaluate(truth,original,SIZE,SIZE).items() if k in ('pq','sq','rq','tp_pairs','fp','fn')}),
        inputs={name:dict(path=str(SOURCES[name]),**{k:input_scores[name][k] for k in ('pq','sq','rq','tp_pairs','fp','fn')}) for name in SOURCES},
        physical_images=91,annotation_groups=101,gt_instances=708,experiments=reports,
        best=best,best_preserving=best_preserving,saved_candidates=saved,
        rules='bbox+9pixelpadding;Gaussianblur; pre/postownership area.5..1.5 and maskIoU>=.85 fallback; nearest-originalownership; instancecountpreserved',
        target_exceeded=best['pq']>.5,elapsed_seconds=time.monotonic()-started)
    (ROOT/'docs/evidence/rich_final_smoothing.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(inputs=report['inputs'],best=best,best_preserving=best_preserving,
        saved_candidates=saved,elapsed_seconds=report['elapsed_seconds']),indent=2))


if __name__=='__main__':main()
