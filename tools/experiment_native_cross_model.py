"""Predeclared GT-free native U-Net additions to the existing disjoint YOLO reference.

No input rows contain confidence scores. High-threshold U-Net mask agreement is
used as geometric stability, not a calibrated probability. This is development
set tuning, not independent CV or leaderboard evidence.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from pycocotools import mask as mu

from experiment_mask_postprocess import read_rows, group_rows, overlap_stats
from experiment_view_combinations import GT, VAL, SIZE, ROOT, encode_roi
from filament_eval import Filament, evaluate

SOURCE = Path('/workspace/artifacts/a100_results_20261028_bedf3e21')
BASELINE = Path('/workspace/artifacts/tiled_object_fusion_candidates_v2/c0.35_iou0.3_majority_guard0.85.csv')
OUT = Path('/workspace/artifacts/native_cross_model_candidates_v1')


def plan():
    return [dict(name=f'native_t{t:g}_a{a}_nov{n:g}_stable{s:g}', threshold=t,
                 min_area=a, max_union_overlap_fraction=n, max_baseline_iou=.1,
                 minimum_high_threshold_iou=s)
            for t in (.35, .5) for a in (300, 600, 1000, 1500)
            for n in (.05, .2) for s in (0., .5, .8)]


def trim(row, covered):
    """Remove any claimed pixels; never modify reference masks."""
    original = row.rle(SIZE, SIZE)
    if covered is None or int(mu.area(mu.merge([original, covered], intersect=True))) == 0:
        return row
    x, y, w, h = map(int, mu.toBbox(original))
    candidate = mu.decode(original)[y:y+h, x:x+w].astype(bool)
    candidate &= ~mu.decode(covered)[y:y+h, x:x+w].astype(bool)
    if not candidate.any():
        return None
    rle = encode_roi(candidate, y, x)
    return Filament(row.filament_id, row.image_id, row.image_id,
                    rle['counts'].decode('ascii'), int(mu.area(rle)))


def matched_ids(truth, rows):
    by_image = group_rows(rows); groups = defaultdict(list)
    for r in truth: groups[r.group_id].append(r)
    matched = set()
    for gt in groups.values():
        pred = by_image.get(gt[0].image_id, [])
        if not pred: continue
        matrix = np.asarray(mu.iou([g.rle(SIZE, SIZE) for g in gt],
                                   [p.rle(SIZE, SIZE) for p in pred], [0]*len(pred)))
        matched.update(g.filament_id for g, hit in zip(gt, (matrix > .5).any(axis=1)) if hit)
    return matched


def main():
    started = time.monotonic()
    settings = plan()
    OUT.mkdir(exist_ok=False)
    (OUT/'predeclared_plan.json').write_text(json.dumps(settings, indent=2)+'\n')
    with VAL.open(newline='') as f: selection=list(csv.DictReader(f))
    ids={r['image_id'] for r in selection}; images={Path(r['file_name']).stem for r in selection}
    assert len(ids)==101 and len(images)==91
    baseline=read_rows(BASELINE); grouped=group_rows(baseline)
    assert len(baseline)==626 and overlap_stats(baseline)['overlap_images']==0
    source={t:group_rows(read_rows(SOURCE/f'best_t{t:g}_a100.csv')) for t in (.35,.5,.65)}
    for grouped_source in source.values(): assert not set(grouped_source)-images
    outputs={s['name']:[] for s in settings}; counters={s['name']:Counter() for s in settings}
    for image in sorted(images):
        ref=grouped.get(image,[]); br=[r.rle(SIZE,SIZE) for r in ref]
        union=mu.merge(br) if br else None
        high=source[.65].get(image,[]); hr=[r.rle(SIZE,SIZE) for r in high]
        for threshold in (.35,.5):
            candidates=source[threshold].get(image,[]); cr=[r.rle(SIZE,SIZE) for r in candidates]
            biou=np.asarray(mu.iou(cr,br,[0]*len(br))) if br and cr else np.zeros((len(cr),len(br)))
            hiou=np.asarray(mu.iou(cr,hr,[0]*len(hr))) if hr and cr else np.zeros((len(cr),len(hr)))
            metadata=[]
            for k,row in enumerate(candidates):
                frac=(int(mu.area(mu.merge([cr[k],union],intersect=True)))/row.area if union else 0.)
                metadata.append((float(biou[k].max()) if br else 0.,frac,float(hiou[k].max()) if hr else 0.))
            for setting in [s for s in settings if s['threshold']==threshold]:
                result=list(ref); covered=union
                for row,(overlap,frac,stability) in zip(candidates,metadata):
                    if row.area<setting['min_area'] or overlap>setting['max_baseline_iou'] or frac>setting['max_union_overlap_fraction'] or stability<setting['minimum_high_threshold_iou']:
                        continue
                    admitted=trim(row,covered)
                    if admitted is None or admitted.area<setting['min_area']: continue
                    admitted=Filament(f'{image}_{len(result)+1}',image,image,admitted.counts,admitted.area)
                    result.append(admitted)
                    covered=mu.merge([covered,admitted.rle(SIZE,SIZE)]) if covered is not None else admitted.rle(SIZE,SIZE)
                    counters[setting['name']]['added_physical_instances']+=1
                assert len({r.filament_id for r in result})==len(result)
                outputs[setting['name']].extend(result)
    # Rules and all predictions are complete before opening GT.
    truth=read_rows(GT,gt=True,selected=ids); assert len(truth)==708
    reference=evaluate(truth,baseline,SIZE,SIZE); assert abs(reference['pq']-.46493857594671645)<1e-12
    reports=[dict(name='reference',**{k:reference[k] for k in ('pq','sq','rq','tp_pairs','fp','fn')},prediction_rows=len(baseline))]
    for setting in settings:
        rows=outputs[setting['name']]; score=evaluate(truth,rows,SIZE,SIZE)
        assert score['tp_pairs']==len(truth)-score['fn']
        reports.append(dict(**setting,**{k:score[k] for k in ('pq','sq','rq','tp_pairs','fp','fn')},
                            prediction_rows=len(rows),**dict(counters[setting['name']])))
    best=max(reports,key=lambda r:r['pq']); best_actual=max(reports[1:],key=lambda r:r['pq'])
    reference_matched=matched_ids(truth,baseline); saved=None
    if best['name']!='reference':
        rows=outputs[best['name']]; overlaps=overlap_stats(rows); assert overlaps['overlap_images']==0
        p=OUT/(best['name']+'.csv')
        with p.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['filament_id','segmentation_rle']);w.writerows((r.filament_id,r.counts) for r in rows)
        reload=read_rows(p); check=evaluate(truth,reload,SIZE,SIZE); assert abs(check['pq']-best['pq'])<1e-12
        matches=matched_ids(truth,reload)
        saved=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),overlap_audit=overlaps,
                   recovered_gt_ids=sorted(matches-reference_matched),lost_gt_ids=sorted(reference_matched-matches))
    evidence=dict(scope='48 predeclared cross-model addition rules; same development set, not independent CV/LB',
                  baseline_sha256=hashlib.sha256(BASELINE.read_bytes()).hexdigest(),
                  sources_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.glob('best_t*_a100.csv')},
                  physical_images=len(images),annotation_groups=len(ids),gt_instances=len(truth),
                  reference=reports[0],experiments=reports,best=best,best_actual=best_actual,saved_candidate=saved,
                  target_exceeded=best['pq']>.5,elapsed_seconds=time.monotonic()-started)
    path=ROOT/'docs/evidence/native_cross_model_experiments.json';path.write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(dict(best=best,best_actual=best_actual,saved_candidate=saved,elapsed_seconds=evidence['elapsed_seconds']),indent=2))


if __name__=='__main__':main()
