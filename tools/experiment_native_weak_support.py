"""Bounded U-Net rescues corroborated by previously cached YOLO proposals.

All masks, proposal choices, replacements and support gates are derived from
predictions only. The 48-rule plan is persisted before loading GT. Development
set scores do not establish independent generalization or leaderboard gains.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from pycocotools import mask as mu

from experiment_mask_postprocess import read_rows, group_rows, overlap_stats
from experiment_native_cross_model import matched_ids
from experiment_view_combinations import GT, VAL, ROOT, SIZE, encode_roi
from filament_eval import Filament, _check_compressed_counts, evaluate

SOURCE=Path('/workspace/artifacts/a100_results_20261028_bedf3e21')
BASELINE=Path('/workspace/artifacts/tiled_object_fusion_candidates_v2/c0.35_iou0.3_majority_guard0.85.csv')
OLD=Path('/workspace/artifacts/validation_results_20261027T120958Z_444da7f4/raw_views.jsonl')
NEW=Path('/workspace/artifacts/validation_results_20261027T125130Z_2312e759/raw_preprocess_views.jsonl')
TILES=Path('/workspace/artifacts/tiled_results_20261027T205426Z_eba01cfb/raw_tiled_views.jsonl')
OUT=Path('/workspace/artifacts/native_weak_support_candidates_v1')


def plan():
    return [dict(name=f'native_t{t:g}_a{a}_c{c:g}_f{f}_iou{i:g}_replace{r}',
                 native_threshold=t,min_area=a,min_weak_confidence=c,
                 minimum_families=f,minimum_support_iou=i,allow_replacement=r,
                 min_replacement_agreement_gain=.1,min_replacement_iou=.2,
                 replacement_area_ratio=[.5,2.],max_addition_base_iou=.1,
                 max_addition_union_fraction=.25)
            for t in (.35,.5) for a in (300,800) for c in (.1,.3)
            for f,i in ((1,.3),(1,.5),(2,.3)) for r in (False,True)]


def load_sources(images):
    grouped=defaultdict(list);metadata={}
    for family,path in [('geometric',OLD),('preprocessing',NEW),('native_tiles',TILES)]:
        seen=set();total=eligible=0
        for line in path.open():
            record=json.loads(line);image=record['image'];assert image in images
            assert record['shape']==[SIZE,SIZE] and record['coordinate_space']=='original'
            assert (image,record['view']) not in seen;seen.add((image,record['view']))
            for p in record['instances']:
                total+=1
                confidence=p['confidence'];assert np.isfinite(confidence) and .05<=confidence<=1.
                area=_check_compressed_counts(p['rle'],SIZE*SIZE);assert area>0
                if confidence<.1 or (family=='native_tiles' and p['edge_clipped']):continue
                grouped[image].append(dict(rle={'size':[SIZE,SIZE],'counts':p['rle'].encode('ascii')},
                                           confidence=confidence,family=family))
                eligible+=1
        metadata[family]=dict(path=str(path),records=len(seen),total_instances=total,
                              eligible_instances=eligible,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return grouped,metadata


def encode_remaining(row,covered,remove=None):
    rle=row.rle(SIZE,SIZE);x,y,w,h=map(int,mu.toBbox(rle));roi=mu.decode(rle)[y:y+h,x:x+w].astype(bool)
    existing=covered[y:y+h,x:x+w].copy()
    if remove is not None:existing &= ~remove[y:y+h,x:x+w]
    roi &= ~existing
    if not roi.any():return None
    encoded=encode_roi(roi,y,x)
    return Filament(row.filament_id,row.image_id,row.image_id,encoded['counts'].decode('ascii'),int(mu.area(encoded)))


def main():
    started=time.monotonic();settings=plan();assert len(settings)==48
    export_controls='--export-controls' in sys.argv
    OUT.mkdir(exist_ok=export_controls)
    plan_path=OUT/'predeclared_plan.json'
    if plan_path.exists():assert json.loads(plan_path.read_text())==settings
    else:plan_path.write_text(json.dumps(settings,indent=2)+'\n')
    with VAL.open(newline='') as f:selection=list(csv.DictReader(f))
    ids={r['image_id'] for r in selection};images={Path(r['file_name']).stem for r in selection}
    assert len(ids)==101 and len(images)==91
    baseline=read_rows(BASELINE);base_by_image=group_rows(baseline)
    assert len(baseline)==626 and not overlap_stats(baseline)['overlap_images']
    native={t:group_rows(read_rows(SOURCE/f'best_t{t:g}_a100.csv')) for t in (.35,.5)}
    weak,weak_meta=load_sources(images)
    outputs={s['name']:[] for s in settings};statuses={s['name']:Counter() for s in settings}
    for number,image in enumerate(sorted(images),1):
        base=base_by_image.get(image,[]);br=[r.rle(SIZE,SIZE) for r in base]
        union=mu.merge(br) if br else None
        covered=mu.decode(union).astype(bool) if union else np.zeros((SIZE,SIZE),bool)
        proposals=weak[image];wr=[p['rle'] for p in proposals]
        bm=np.asarray(mu.iou(br,wr,[0]*len(wr))) if br and wr else np.zeros((len(br),len(wr)))
        for t in (.35,.5):
            candidates=native[t].get(image,[]);cr=[r.rle(SIZE,SIZE) for r in candidates]
            cm=np.asarray(mu.iou(cr,wr,[0]*len(wr))) if cr and wr else np.zeros((len(cr),len(wr)))
            cb=np.asarray(mu.iou(cr,br,[0]*len(br))) if cr and br else np.zeros((len(cr),len(br)))
            novelty=[int(mu.area(mu.merge([rle,union],intersect=True)))/r.area if union else 0.
                     for rle,r in zip(cr,candidates)]
            maxbase=cb.max(axis=1) if br else np.zeros(len(cr))
            assign=cb.argmax(axis=1) if br else np.zeros(len(cr),dtype=int)
            for s in [p for p in settings if p['native_threshold']==t]:
                eligible=[j for j,p in enumerate(proposals) if p['confidence']>=s['min_weak_confidence']]
                support=[len({proposals[j]['family'] for j in eligible if cm[i,j]>=s['minimum_support_iou']}) for i in range(len(cr))]
                cquality=cm[:,eligible].max(axis=1) if eligible else np.zeros(len(cr))
                bquality=bm[:,eligible].max(axis=1) if eligible and br else np.zeros(len(br))
                admitted=[i for i,row in enumerate(candidates) if row.area>=s['min_area'] and support[i]>=s['minimum_families']]
                replacements={}
                if s['allow_replacement']:
                    for i in admitted:
                        if not br or maxbase[i]<s['min_replacement_iou']:continue
                        j=int(assign[i]);ratio=candidates[i].area/base[j].area
                        if not s['replacement_area_ratio'][0]<=ratio<=s['replacement_area_ratio'][1]:continue
                        gain=cquality[i]-bquality[j]
                        if gain<s['min_replacement_agreement_gain']:continue
                        if j not in replacements or gain>replacements[j][1]:replacements[j]=(i,float(gain))
                result=list(base);occupied=covered.copy();replaced_native=set()
                for j,(i,gain) in sorted(replacements.items()):
                    original=mu.decode(br[j]).astype(bool)
                    candidate=encode_remaining(candidates[i],occupied,remove=original)
                    if candidate is None or candidate.area<s['min_area']:continue
                    occupied &= ~original;occupied |=mu.decode(candidate.rle(SIZE,SIZE)).astype(bool)
                    result[j]=candidate;replaced_native.add(i);statuses[s['name']]['replaced_instances']+=1
                for i in admitted:
                    if i in replaced_native or maxbase[i]>s['max_addition_base_iou'] or novelty[i]>s['max_addition_union_fraction']:continue
                    candidate=encode_remaining(candidates[i],occupied)
                    if candidate is None or candidate.area<s['min_area']:continue
                    occupied |=mu.decode(candidate.rle(SIZE,SIZE)).astype(bool)
                    result.append(candidate);statuses[s['name']]['added_instances']+=1
                rows=[Filament(f'{image}_{j}',image,image,r.counts,r.area) for j,r in enumerate(result,1)]
                outputs[s['name']].extend(rows)
        if number%20==0:print('Support',number,'/',len(images),flush=True)
    # Prediction construction is complete; GT cannot influence admission.
    truth=read_rows(GT,gt=True,selected=ids);assert len(truth)==708
    reference=evaluate(truth,baseline,SIZE,SIZE);assert abs(reference['pq']-.46493857594671645)<1e-12
    reports=[dict(name='reference',**{k:reference[k] for k in ('pq','sq','rq','tp_pairs','fp','fn')},prediction_rows=len(baseline))]
    for s in settings:
        rows=outputs[s['name']];score=evaluate(truth,rows,SIZE,SIZE);assert score['tp_pairs']==len(truth)-score['fn']
        reports.append(dict(**s,**{k:score[k] for k in ('pq','sq','rq','tp_pairs','fp','fn')},
                            prediction_rows=len(rows),statuses=dict(statuses[s['name']])))
    best=max(reports,key=lambda r:r['pq']);best_actual=max(reports[1:],key=lambda r:r['pq']);saved=None
    best_recall=min(reports[1:],key=lambda r:(r['fn'],-r['pq']))
    controls={}
    targets=[best] if best['name']!='reference' else []
    if export_controls:targets.extend([best_actual,best_recall])
    for setting in {r['name']:r for r in targets}.values():
        rows=outputs[setting['name']];overlaps=overlap_stats(rows);assert overlaps['overlap_images']==0
        path=OUT/(setting['name']+'.csv')
        with path.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['filament_id','segmentation_rle']);w.writerows((r.filament_id,r.counts) for r in rows)
        reload=read_rows(path);check=evaluate(truth,reload,SIZE,SIZE);assert abs(check['pq']-setting['pq'])<1e-12
        prev=matched_ids(truth,baseline);now=matched_ids(truth,reload)
        record=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),overlap_audit=overlaps,
                    adopted=setting['pq']>reference['pq'],score_recomputed_after_save=True,
                    score={k:check[k] for k in ('pq','sq','rq','tp_pairs','fp','fn')},
                    recovered_gt_ids=sorted(now-prev),lost_gt_ids=sorted(prev-now))
        if setting['name']==best['name']:saved=record
        controls[setting['name']]=record
    evidence=dict(scope='48 fixed same-development weakYOLO corroboration rules; no independent CV/LB',
                  source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [BASELINE,SOURCE/'best_t0.35_a100.csv',SOURCE/'best_t0.5_a100.csv']},
                  weak_sources=weak_meta,physical_images=91,annotation_groups=101,gt_instances=708,
                  experiments=reports,best=best,best_actual=best_actual,saved_candidate=saved,
                  best_recall=best_recall,saved_development_controls=controls,
                  elapsed_seconds=time.monotonic()-started)
    (ROOT/'docs/evidence/native_weak_support_experiments.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(dict(best=best,best_actual=best_actual,saved_candidate=saved,
                         saved_development_controls=controls,elapsed_seconds=evidence['elapsed_seconds']),indent=2))


if __name__=='__main__':main()
