"""Thirty-six high-FG-core / low-FG-support watershed decoding rules.

This follow-up replaces the saturated spine seeds with independent high
foreground threshold cores. It uses cached predictions only, before GT scoring.
"""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from pycocotools import mask as mu

from experiment_mask_postprocess import overlap_stats, read_rows
from experiment_rich_spine import marker_split, fixture
from experiment_view_combinations import ROOT, GT, VAL, SIZE, encode_roi
from filament_eval import Filament, evaluate

SOURCE=Path('/workspace/artifacts/best84_rich_predictions_tta/predictions.jsonl')
OUT=Path('/workspace/artifacts/rich_hysteresis_watershed_v1')


def plan():
    return [dict(name=f'{domain}_low{low:g}_high{high:g}_seed{seed}',domain=domain,
                 low=low,high=high,minimum_seed=seed,minimum_area=300,maximum_seeds=8)
            for domain in ('base','tta') for low in (.2,.35,.5)
            for high in (.65,.8) for seed in (50,150,300)]


def main():
    started=time.monotonic();fixture()
    settings=plan();assert len(settings)==36
    OUT.mkdir(exist_ok=False)
    (OUT/'predeclared_plan.json').write_text(json.dumps(settings,indent=2)+'\n')
    selection=list(csv.DictReader(VAL.open()));ids={r['image_id'] for r in selection}
    images={Path(r['file_name']).stem for r in selection}
    records=[json.loads(line) for line in SOURCE.open()]
    assert len(records)==91 and {Path(r['file_name']).stem for r in records}==images
    output={s['name']:[] for s in settings};statuses={s['name']:Counter() for s in settings}
    controls={f'{domain}_low{low:g}_control':[] for domain in ('base','tta') for low in (.2,.35,.5)}
    changed=[]
    for record in records:
        image=Path(record['file_name']).stem
        for domain in ('base','tta'):
            data=record if domain=='base' else record['tta']
            seeds={t:mu.decode(data['unions']['foreground'][f'{t:g}']).astype(bool) for t in (.65,.8)}
            for low in (.2,.35,.5):
                selected=[s for s in settings if s['domain']==domain and s['low']==low]
                components=[c for c in data['components'][f'{low:g}'] if c['area']>=300]
                for number,component in enumerate(components,1):
                    x,y,w,h=map(int,component['bbox'])
                    foreground=mu.decode(component['rle'])[y:y+h,x:x+w].astype(bool)
                    control=controls[f'{domain}_low{low:g}_control']
                    control.append(Filament(f'{image}_{len(control)+1}',image,image,
                        component['rle']['counts'],int(component['area'])))
                    for setting in selected:
                        pieces,status=marker_split(foreground,seeds[setting['high']][y:y+h,x:x+w],
                            setting['minimum_seed'],setting['minimum_area'],setting['maximum_seeds'])
                        statuses[setting['name']][status]+=1
                        if status=='split':
                            changed.append(dict(setting=setting['name'],image=image,component=number,
                                original_area=int(component['area']),piece_areas=[int(p.sum()) for p in pieces]))
                        for piece in pieces:
                            encoded=encode_roi(piece,y,x) if len(pieces)>1 else component['rle']
                            counts=encoded['counts']
                            if isinstance(counts,bytes):counts=counts.decode('ascii')
                            rows=output[setting['name']]
                            rows.append(Filament(f'{image}_{len(rows)+1}',image,image,counts,int(piece.sum())))
    # No annotation masks have been read during core extraction or decoding.
    truth=read_rows(GT,gt=True,selected=ids)
    assert len(truth)==708 and len(ids)==101 and len(images)==91
    reports=[]
    for setting in settings:
        rows=output[setting['name']];score=evaluate(truth,rows,SIZE,SIZE);audit=overlap_stats(rows)
        assert not any(audit.values()) and score['tp_pairs']==708-score['fn']
        reports.append(dict(**setting,**{k:score[k] for k in ('pq','sq','rq','tp_pairs','fp','fn')},
            prediction_rows=len(rows),statuses=dict(statuses[setting['name']]),**audit))
    baseline=[]
    for name,rows in controls.items():
        score=evaluate(truth,rows,SIZE,SIZE)
        baseline.append(dict(name=name,**{k:score[k] for k in ('pq','sq','rq','tp_pairs','fp','fn')},prediction_rows=len(rows)))
    all_reports=reports+baseline
    best=max(all_reports,key=lambda r:r['pq']);best_actual=max(reports,key=lambda r:r['pq'])
    recall=min(all_reports,key=lambda r:(r['fn'],-r['pq']))
    saved=[]
    for row in {r['name']:r for r in (best,best_actual,recall)}.values():
        path=OUT/(row['name']+'.csv');rows=output.get(row['name'],controls.get(row['name']))
        with path.open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['filament_id','segmentation_rle']);writer.writerows((r.filament_id,r.counts) for r in rows)
        score=evaluate(truth,read_rows(path),SIZE,SIZE);assert abs(score['pq']-row['pq'])<1e-12
        saved.append(dict(name=row['name'],path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    evidence=dict(scope='36 globally fixed foreground-hierarchy watershed rules plus6 identity controls; fixeddev, notCV/LB',
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),physical_images=91,annotation_groups=101,
        gt_instances=708,experiments=reports,controls=baseline,best=best,best_actual=best_actual,
        best_recall=recall,saved_candidates=saved,changed_components=changed,
        decoder='HighFG.65/.8 cores inside lowFG.2/.35/.5 connectedcomponent; seeds>=50/150/300; split2..8seeds using watershed_ift inverseEDT; allparts>=300 elseoriginal; exactdisjointpartition',
        target_exceeded=best['pq']>.5,elapsed_seconds=time.monotonic()-started)
    (ROOT/'docs/evidence/rich_hysteresis_watershed.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(dict(best=best,best_actual=best_actual,best_recall=recall,saved_candidates=saved,
        total_changed_components_across_rules=len(changed),elapsed_seconds=evidence['elapsed_seconds']),indent=2))


if __name__=='__main__':main()
