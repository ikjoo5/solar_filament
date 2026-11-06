"""Bounded prediction-only spine support and marker-watershed decoding.

The five foreground/three spine threshold masks are cached predictions, not raw
probability maps. Rules are fixed globally before any GT is read. All split
pieces partition the original foreground and never duplicate any pixel.
"""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from pycocotools import mask as mu
from scipy import ndimage

from experiment_mask_postprocess import overlap_stats, read_rows
from experiment_native_cross_model import matched_ids
from experiment_view_combinations import ROOT, GT, VAL, SIZE, encode_roi
from filament_eval import Filament, evaluate

SOURCE = Path('/workspace/artifacts/best84_rich_predictions_tta/predictions.jsonl')
OUT = Path('/workspace/artifacts/rich_spine_candidates_v1')


def plan():
    settings = []
    for domain in ('base', 'tta'):
        for foreground in (.35, .5):
            settings.append(dict(name=f'{domain}_fg{foreground:g}_control', domain=domain,
                foreground=foreground, rule='identity', minimum_area=300))
            for spine in (.3, .5, .7):
                settings.extend(dict(name=f'{domain}_fg{foreground:g}_sp{spine:g}_seed{minimum}',
                    domain=domain, foreground=foreground, spine=spine,
                    rule='watershed', minimum_area=300, minimum_seed=minimum,
                    maximum_seeds=8) for minimum in (10, 50))
                settings.extend(dict(name=f'{domain}_fg{foreground:g}_sp{spine:g}_cover{coverage:g}',
                    domain=domain, foreground=foreground, spine=spine,
                    rule='support', minimum_area=300, minimum_coverage=coverage)
                    for coverage in (.1, .5))
    assert len(settings) == 52
    return settings


def marker_split(foreground, spine, minimum_seed, minimum_area=300, maximum_seeds=8):
    labels, count = ndimage.label(foreground & spine, structure=np.ones((3,3), np.uint8))
    areas = np.bincount(labels.ravel())
    seeds = [i for i in range(1, count+1) if areas[i] >= minimum_seed]
    if len(seeds) < 2:
        return [foreground], 'single_or_no_seed'
    if len(seeds) > maximum_seeds:
        return [foreground], 'too_many_seeds'
    markers = np.zeros(foreground.shape, np.int32)
    for index, label in enumerate(seeds, 1):
        markers[labels == label] = index
    distance = ndimage.distance_transform_edt(np.pad(foreground,1))[1:-1,1:-1]
    gradient = np.full(foreground.shape,255,np.uint8)
    gradient[foreground] = np.clip(254 * (1-distance[foreground]/distance.max()),0,254).astype(np.uint8)
    assignment = ndimage.watershed_ift(gradient, markers, structure=np.ones((3,3),np.uint8))
    pieces = [foreground & (assignment == index) for index in range(1,len(seeds)+1)]
    if any(int(p.sum()) < minimum_area for p in pieces):
        return [foreground], 'small_piece_fallback'
    assert np.array_equal(np.sum(pieces,axis=0),foreground.astype(np.int64))
    return pieces, 'split'


def fixture():
    mask = np.zeros((36,60),bool);mask[5:31,5:55] = True
    seeds = np.zeros_like(mask);seeds[10:26,10:16] = True;seeds[10:26,44:50] = True
    pieces, status = marker_split(mask,seeds,10,300)
    assert status == 'split' and len(pieces) == 2
    assert np.array_equal(sum(pieces),mask.astype(int))
    unchanged, status = marker_split(mask,mask,10,300)
    assert status == 'single_or_no_seed' and np.array_equal(unchanged[0],mask)


def main():
    start = time.monotonic();fixture()
    settings = plan();OUT.mkdir(exist_ok=False)
    (OUT/'predeclared_plan.json').write_text(json.dumps(settings,indent=2)+'\n')
    selection = list(csv.DictReader(VAL.open()))
    groups = {row['image_id'] for row in selection}
    images = {Path(row['file_name']).stem for row in selection}
    records = [json.loads(line) for line in SOURCE.open()]
    assert len(records) == 91 and {Path(r['file_name']).stem for r in records} == images
    outputs = {s['name']:[] for s in settings};statuses = {s['name']:Counter() for s in settings}
    diagnostics = []; global_union = []
    for record in records:
        image = Path(record['file_name']).stem
        for domain in ('base','tta'):
            data = record if domain == 'base' else record['tta']
            spine = {t:mu.decode(data['unions']['spine'][f'{t:g}']).astype(bool) for t in (.3,.5,.7)}
            fg_union = mu.decode(data['unions']['foreground']['0.5']).astype(bool)
            for t, sp in spine.items():
                union = (fg_union | sp).sum()
                global_union.append(dict(image=image,domain=domain,spine=t,
                    foreground_area=int(fg_union.sum()),spine_area=int(sp.sum()),
                    inside_foreground=int((fg_union & sp).sum()),
                    foreground_coverage=float((fg_union & sp).sum()/max(fg_union.sum(),1)),
                    spine_outside_foreground=int((sp & ~fg_union).sum()),
                    union_iou=float((fg_union & sp).sum()/max(union,1))))
            for threshold in (.35,.5):
                components = [r for r in data['components'][f'{threshold:g}'] if r['area']>=300]
                matching = [s for s in settings if s['domain']==domain and s['foreground']==threshold]
                for component in components:
                    x,y,w,h = map(int,component['bbox'])
                    foreground = mu.decode(component['rle'])[y:y+h,x:x+w].astype(bool)
                    original = component['rle']['counts']
                    for t in spine:
                        sp = spine[t][y:y+h,x:x+w]
                        labels,count = ndimage.label(foreground & sp,structure=np.ones((3,3),np.uint8))
                        areas = np.bincount(labels.ravel())
                        diagnostics.append(dict(domain=domain,foreground=threshold,spine=t,
                            coverage=float((foreground & sp).sum()/foreground.sum()),
                            valid_seeds10=int(sum(areas[1:]>=10)),valid_seeds50=int(sum(areas[1:]>=50))))
                    for setting in matching:
                        kind = setting['rule'];pieces=[foreground];status='kept'
                        if kind == 'support':
                            coverage = (foreground & spine[setting['spine']][y:y+h,x:x+w]).sum()/foreground.sum()
                            if coverage < setting['minimum_coverage']:
                                statuses[setting['name']]['rejected'] += 1
                                continue
                        elif kind == 'watershed':
                            pieces,status = marker_split(foreground,spine[setting['spine']][y:y+h,x:x+w],
                                setting['minimum_seed'],setting['minimum_area'],setting['maximum_seeds'])
                        statuses[setting['name']][status] += 1
                        for piece in pieces:
                            encoded = encode_roi(piece,y,x) if len(pieces)>1 else component['rle']
                            counts = encoded['counts']
                            if isinstance(counts,bytes):counts=counts.decode('ascii')
                            rows=outputs[setting['name']]
                            rows.append(Filament(f'{image}_{len(rows)+1}',image,image,counts,int(piece.sum())))
    # Prediction rules and all outputs are complete before GT becomes available.
    truth = read_rows(GT,gt=True,selected=groups)
    assert len(truth)==708 and len(groups)==101 and len(images)==91
    reports=[]
    for setting in settings:
        rows=outputs[setting['name']];score=evaluate(truth,rows,SIZE,SIZE);overlap=overlap_stats(rows)
        assert not any(overlap.values()) and score['tp_pairs']==708-score['fn']
        reports.append(dict(**setting,**{k:score[k] for k in ('pq','sq','rq','tp_pairs','fp','fn')},
                            prediction_rows=len(rows),statuses=dict(statuses[setting['name']]),**overlap))
    best=max(reports,key=lambda r:r['pq']);recall=min(reports,key=lambda r:(r['fn'],-r['pq']))
    saved=[]
    for result in {r['name']:r for r in (best,recall)}.values():
        path=OUT/(result['name']+'.csv')
        with path.open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['filament_id','segmentation_rle'])
            writer.writerows((r.filament_id,r.counts) for r in outputs[result['name']])
        score=evaluate(truth,read_rows(path),SIZE,SIZE);assert abs(score['pq']-result['pq'])<1e-12
        saved.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),name=result['name']))
    summary=[]
    for domain in ('base','tta'):
        for fg in (.35,.5):
            for sp in (.3,.5,.7):
                d=[r for r in diagnostics if (r['domain'],r['foreground'],r['spine'])==(domain,fg,sp)]
                summary.append(dict(domain=domain,foreground=fg,spine=sp,components=len(d),
                    coverage_quantiles=np.quantile([r['coverage'] for r in d],[0,.1,.5,.9,1]).tolist(),
                    valid_seeds10_histogram=dict(Counter(r['valid_seeds10'] for r in d)),
                    valid_seeds50_histogram=dict(Counter(r['valid_seeds50'] for r in d))))
    union_summary=[]
    for domain in ('base','tta'):
        for spine in (.3,.5,.7):
            selected=[r for r in global_union if r['domain']==domain and r['spine']==spine]
            fg=sum(r['foreground_area'] for r in selected)
            sp=sum(r['spine_area'] for r in selected)
            intersection=sum(r['inside_foreground'] for r in selected)
            union_summary.append(dict(domain=domain,spine=spine,foreground_threshold=.5,
                foreground_area=fg,spine_area=sp,intersection=intersection,
                foreground_coverage=intersection/fg,
                spine_outside_foreground_fraction=(sp-intersection)/sp,
                union_iou=intersection/(fg+sp-intersection)))
    evidence=dict(scope='52 fixed prediction-only spine decoder rules on same development cohort; not CV/LB',
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),physical_images=91,annotation_groups=101,
        gt_instances=708,experiments=reports,best=best,best_recall=recall,saved_candidates=saved,
        component_spine_diagnostics=summary,union_spine_diagnostics=global_union,
        union_spine_summary=union_summary,
        decoder='8connected spine seeds>=10/50pixels; max8seeds; watershed_ift on inverse foreground EDT; outside cost255; fallback unless all disjoint pieces>=300',
        limitations='No dense probability maps are available; threshold masks and component stats only. Spine auxiliary head may be broad rather than a thin centerline; diagnostics quantify actual behavior.',
        target_exceeded=best['pq']>.5,elapsed_seconds=time.monotonic()-start)
    (ROOT/'docs/evidence/rich_spine_decoding.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(dict(best=best,best_recall=recall,saved_candidates=saved,diagnostics=summary,elapsed_seconds=evidence['elapsed_seconds']),indent=2))


if __name__=='__main__':main()
