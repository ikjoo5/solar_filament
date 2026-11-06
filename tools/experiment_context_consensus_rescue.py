"""Six fixed two-native-model rescue rules; originals never change.

All six predictions are built before GT is opened. This is reused-development
exploration, not an independent evaluation or a claim about public LB.
"""
from collections import Counter, defaultdict
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from pycocotools import mask as mu

from experiment_mask_postprocess import group_rows, overlap_stats, read_rows
from filament_eval import Filament, evaluate
from submission_native_recipe import _rle, _trim

ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/workspace/artifacts/rich_boundary_composition_v1/tta_precision.csv')
OLD=Path('/workspace/artifacts/best84_rich_predictions_tta/predictions.jsonl')
NEW=Path('/workspace/artifacts/context_only_uploaded_20261029/multiscale_context/raw_paper_views.jsonl')
OUT=Path('/workspace/artifacts/context_consensus_rescue_v1')
SIZE=2048


def plan():
    specifications=[('new_boundary_loose',.35,'new',.5,.8,2),
        ('new_boundary_strict',.35,'new',.7,.9,3),
        ('old_boundary_loose',.35,'old',.5,.8,2),
        ('new_t50',.5,'new',.5,.8,2),
        ('union_boundary',.35,'union',.5,.8,2),
        ('intersection_boundary',.35,'intersection',.5,.8,2)]
    return [dict(name=name,threshold=t,geometry=geometry,minimum_cross_model_iou=iou,
        minimum_fg_mean=fg,minimum_views=views,per_view_coverage=.5,minimum_area=300,
        maximum_baseline_iou=.1,maximum_union_overlap_fraction=.25,
        mutual_best_pair=True) for name,t,geometry,iou,fg,views in specifications]


def _intersection(a,b):
    return int(mu.area(mu.merge([a,b],intersect=True)))


def make_rescue(image,originals,old,new,setting):
    """Read model output and frozen masks only; no truth-dependent decisions."""
    threshold=f"{setting['threshold']:g}"
    components={key:record['tta']['components'][threshold] for key,record in [('old',old),('new',new)]}
    rles={key:[_rle(c['rle']) for c in values] for key,values in components.items()}
    oldr,newr=rles['old'],rles['new']
    output=list(originals);diagnostics=Counter()
    if not oldr or not newr:return output,diagnostics
    pairs=np.asarray(mu.iou(newr,oldr,[0]*len(oldr)))
    old_best=pairs.argmax(axis=0)
    original_rles=[r.rle(SIZE,SIZE) for r in originals]
    covered=mu.merge(original_rles) if original_rles else None
    views={key:[_rle(rle) for rle in record['view_unions'].values()]
           for key,record in [('old',old),('new',new)]}
    for i,new_component in enumerate(components['new']):
        j=int(pairs[i].argmax());old_component=components['old'][j]
        if pairs[i,j]<setting['minimum_cross_model_iou'] or int(old_best[j])!=i:continue
        if min(new_component['area'],old_component['area'])<setting['minimum_area']:continue
        if min(new_component['stats']['foreground']['mean'],old_component['stats']['foreground']['mean'])<setting['minimum_fg_mean']:continue
        if any(sum(_intersection(candidate,view)/int(mu.area(candidate))>=setting['per_view_coverage']
                   for view in views[key])<setting['minimum_views']
               for key,candidate in [('old',oldr[j]),('new',newr[i])]):continue
        diagnostics['corroborated_pairs']+=1
        geometry=setting['geometry']
        if geometry=='new':candidate=newr[i]
        elif geometry=='old':candidate=oldr[j]
        else:candidate=mu.merge([newr[i],oldr[j]],intersect=geometry=='intersection')
        area=int(mu.area(candidate))
        if area<setting['minimum_area']:continue
        # Union/intersection guards are checked on the actual candidate shape.
        if original_rles and float(np.asarray(mu.iou([candidate],original_rles,[0]*len(original_rles))).max())>setting['maximum_baseline_iou']:continue
        if covered is not None and _intersection(candidate,covered)/area>setting['maximum_union_overlap_fraction']:continue
        candidate=_trim(candidate,covered)
        if candidate is None or int(mu.area(candidate))<setting['minimum_area']:continue
        if covered is not None:assert _intersection(candidate,covered)==0
        output.append(Filament(f'{image}_{len(output)+1}',image,image,candidate['counts'].decode('ascii'),int(mu.area(candidate))))
        covered=mu.merge([covered,candidate]) if covered is not None else candidate
        diagnostics['physical_additions']+=1
    assert Counter(r.counts for r in originals)<=Counter(r.counts for r in output)
    return output,diagnostics


def matches(truth,pred):
    candidates=group_rows(pred);groups=defaultdict(list);answer=set()
    for row in truth:groups[row.group_id].append(row)
    for members in groups.values():
        rows=candidates.get(members[0].image_id,[])
        if not rows:continue
        iou=np.asarray(mu.iou([r.rle(SIZE,SIZE) for r in members],[r.rle(SIZE,SIZE) for r in rows],[0]*len(rows)))
        answer.update(r.filament_id for r,hit in zip(members,(iou>.5).any(axis=1)) if hit)
    return answer


def uncertainty(first,second):
    grouped=defaultdict(lambda:np.zeros((2,4),np.float64))
    for k,score in enumerate((first,second)):
        for row in score['per_group']:
            grouped[row['image_id']][k]+=np.asarray([row['matched_iou_sum'],row['tp_pairs'],row['fp'],row['fn']])
    values=np.stack(list(grouped.values()));rng=np.random.default_rng(20261029)
    sampled=values[rng.integers(0,len(values),(10000,len(values)))].sum(axis=1)
    pq=sampled[:,:,0]/(sampled[:,:,1]+.5*sampled[:,:,2]+.5*sampled[:,:,3])
    return dict(delta_pq_percentile95=np.quantile(pq[:,1]-pq[:,0],[.025,.975]).tolist(),
        unit='Physical image with all raters; conditional on reused development data and selected models, no selection-bias correction')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUT);args=parser.parse_args()
    started=time.monotonic();output=args.output;output.mkdir(parents=True,exist_ok=False)
    settings=plan();assert len(settings)==6
    (output/'predeclared_plan.json').write_text(json.dumps(settings,indent=2)+'\n')
    baseline=read_rows(BASE);assert len(baseline)==625 and not any(overlap_stats(baseline).values())
    originals=group_rows(baseline)
    def load(path):
        result={}
        for row in map(json.loads,path.open()):
            image=Path(row['file_name']).stem
            assert image not in result and row['size']==[SIZE,SIZE]
            result[image]=row
        return result
    old,new=load(OLD),load(NEW);assert set(old)==set(new) and len(new)==91
    predictions={s['name']:[] for s in settings};counters={s['name']:Counter() for s in settings}
    for number,image in enumerate(sorted(new),1):
        for setting in settings:
            rows,counts=make_rescue(image,originals.get(image,[]),old[image],new[image],setting)
            predictions[setting['name']].extend(rows);counters[setting['name']].update(counts)
        if number%25==0:print('Two-model rescue predictions',number,'/91',flush=True)
    # All predictions and global rules are complete before opening GT/split.
    GT=Path('/workspace/artifacts/solar_filament_gt_v1/gt.csv')
    VAL=Path('/workspace/attachments/bcb64220-b208-4cd7-9c3f-3f2be0f04167/val_split.csv')
    selection=list(csv.DictReader(VAL.open()));ids={r['image_id'] for r in selection}
    assert {Path(r['file_name']).stem for r in selection}==set(new)
    truth=read_rows(GT,gt=True,selected=ids);assert len(truth)==708 and len(ids)==101
    before=evaluate(truth,baseline,SIZE,SIZE);assert abs(before['pq']-.47259304445299505)<1e-12
    before_matches=matches(truth,baseline);fields=('pq','sq','rq','tp_pairs','fp','fn')
    reports=[]
    for setting in settings:
        rows=predictions[setting['name']];score=evaluate(truth,rows,SIZE,SIZE);audit=overlap_stats(rows)
        assert not any(audit.values()) and score['tp_pairs']==708-score['fn']
        assert Counter((r.image_id,r.counts) for r in baseline)<=Counter((r.image_id,r.counts) for r in rows)
        after=matches(truth,rows);assert not before_matches-after
        path=output/(setting['name']+'.csv')
        with path.open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['filament_id','segmentation_rle']);writer.writerows((r.filament_id,r.counts) for r in rows)
        reloaded=read_rows(path);again=evaluate(truth,reloaded,SIZE,SIZE);assert abs(again['pq']-score['pq'])<1e-12
        reports.append(dict(**setting,**{k:score[k] for k in fields},**dict(counters[setting['name']]),
            delta_pq=score['pq']-before['pq'],physical_masks=len(rows),recovered_gt=len(after-before_matches),lost_gt=0,
            recovered_gt_ids=sorted(after-before_matches),original_625_masks_preserved=True,**audit,
            **uncertainty(before,score),path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            score_recomputed_after_save=True))
    report=dict(scope='Six predeclared global old+new native consensus rescues on reused91-image development set',
        validation=dict(physical_images=91,annotator_groups=101,gt_instances=708),
        baseline={k:before[k] for k in fields},baseline_masks=625,experiments=reports,
        best=max(reports,key=lambda r:r['pq']),highest_recall=min(reports,key=lambda r:(r['fn'],-r['pq'])),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (BASE,OLD,NEW)},
        training_performed=False,external_data_used=False,adopted=False,
        note='Candidate selection on reused development only; no independent generalization or LB gain claimed.',
        elapsed_seconds=time.monotonic()-started)
    (ROOT/'docs/evidence/context_consensus_rescue.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(baseline=report['baseline'],best=report['best'],highest_recall=report['highest_recall'],
        elapsed_seconds=report['elapsed_seconds']),indent=2))


if __name__=='__main__':main()
