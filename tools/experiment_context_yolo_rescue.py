"""Four fixed YOLO-support gates on the 27 predeclared consensus additions.

The old/new U-Nets share their backbone. Additional YOLO agreement is tested
without changing the 625 frozen masks, choosing individual GT hits, or tuning
any further gate. All four outputs exist before GT is read.
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

from experiment_context_consensus_rescue import matches, uncertainty
from experiment_mask_postprocess import group_rows, overlap_stats, read_rows
from filament_eval import Filament, evaluate

ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/workspace/artifacts/rich_boundary_composition_v1/tta_precision.csv')
CANDIDATES=Path('/workspace/artifacts/context_consensus_rescue_v1/new_t50.csv')
SOURCES={
    'geometric':Path('/workspace/artifacts/validation_results_20261027T120958Z_444da7f4/raw_views.jsonl'),
    'preprocessing':Path('/workspace/artifacts/validation_results_20261027T125130Z_2312e759/raw_preprocess_views.jsonl'),
    'native_tiles':Path('/workspace/artifacts/tiled_results_20261027T205426Z_eba01cfb/raw_tiled_views.jsonl'),
}
SIZE=2048
OUT=Path('/workspace/artifacts/context_yolo_rescue_v1')


def plan():
    return [dict(name=name,minimum_confidence=confidence,minimum_families=families,minimum_iou=.3,
                 exclude_edge_clipped_native_tiles=True)
        for name,confidence,families in [('any_c035',.35,1),('any_c050',.5,1),
            ('two_families_c025',.25,2),('two_families_c035',.35,2)]]


def load_raw():
    result=defaultdict(lambda:defaultdict(list));records=Counter()
    for family,path in SOURCES.items():
        with path.open() as handle:
            for row in map(json.loads,handle):
                assert row['shape']==[SIZE,SIZE] and row['coordinate_space']=='original'
                records[family]+=1
                for instance in row['instances']:
                    if instance['confidence']<.25:continue
                    if family=='native_tiles' and instance['edge_clipped']:continue
                    rle=dict(size=[SIZE,SIZE],counts=instance['rle'].encode('ascii'))
                    result[row['image']][family].append(dict(rle=rle,confidence=instance['confidence']))
    return result,dict(records)


def family_support(candidate,by_family,minimum_iou=.3):
    """Maximum supporting confidence per family, never number of raw views."""
    result={}
    for family,proposals in by_family.items():
        if not proposals:continue
        matrix=np.asarray(mu.iou([candidate],[p['rle'] for p in proposals],[0]*len(proposals)))[0]
        result[family]=max((float(p['confidence']) for p,iou in zip(proposals,matrix) if iou>=minimum_iou),default=0.)
    return result


def admitted(support,setting):
    return sum(confidence>=setting['minimum_confidence'] for confidence in support.values())>=setting['minimum_families']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUT);args=parser.parse_args()
    started=time.monotonic();output=args.output;output.mkdir(exist_ok=False,parents=True)
    settings=plan();assert len(settings)==4
    (output/'predeclared_plan.json').write_text(json.dumps(settings,indent=2)+'\n')
    baseline=read_rows(BASE);candidates=read_rows(CANDIDATES)
    assert len(baseline)==625 and len(candidates)==652
    assert not any(overlap_stats(baseline).values()) and not any(overlap_stats(candidates).values())
    frozen=Counter((r.image_id,r.counts) for r in baseline)
    assert frozen<=Counter((r.image_id,r.counts) for r in candidates)
    novel=[];remaining=frozen.copy()
    for row in candidates:
        key=(row.image_id,row.counts)
        if remaining[key]:remaining[key]-=1
        else:novel.append(row)
    assert len(novel)==27
    raw,record_counts=load_raw();metadata=[]
    for row in novel:
        support=family_support(row.rle(SIZE,SIZE),raw[row.image_id])
        metadata.append(dict(image=row.image_id,rle=row.counts,area=row.area,support=support))
    predictions={}
    addition_stats={}
    for setting in settings:
        selected=[row for row in metadata if admitted(row['support'],setting)]
        original=group_rows(baseline);adds=defaultdict(list)
        for row in selected:adds[row['image']].append(row)
        rows=[]
        for image in sorted(set(original)|set(adds)):
            members=original.get(image,[])
            rows.extend(members)
            rows.extend(Filament(f'{image}_{len(members)+index}',image,image,row['rle'],row['area'])
                        for index,row in enumerate(adds[image],1))
        predictions[setting['name']]=rows
        addition_stats[setting['name']]=dict(physical_additions=len(selected),affected_images=len({row['image'] for row in selected}))
    # Predictions from all four gates are now fixed. Truth is opened only here.
    GT=Path('/workspace/artifacts/solar_filament_gt_v1/gt.csv')
    VAL=Path('/workspace/attachments/bcb64220-b208-4cd7-9c3f-3f2be0f04167/val_split.csv')
    selection=list(csv.DictReader(VAL.open()));ids={r['image_id'] for r in selection}
    assert len(ids)==101 and len({r['file_name'] for r in selection})==91
    truth=read_rows(GT,gt=True,selected=ids);assert len(truth)==708
    before=evaluate(truth,baseline,SIZE,SIZE);assert abs(before['pq']-.47259304445299505)<1e-12
    before_matches=matches(truth,baseline);fields=('pq','sq','rq','tp_pairs','fp','fn');reports=[]
    for setting in settings:
        rows=predictions[setting['name']];score=evaluate(truth,rows,SIZE,SIZE);audit=overlap_stats(rows)
        assert frozen<=Counter((r.image_id,r.counts) for r in rows)
        assert not any(audit.values()) and score['tp_pairs']==708-score['fn']
        after=matches(truth,rows);assert not before_matches-after
        path=output/(setting['name']+'.csv')
        with path.open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['filament_id','segmentation_rle']);writer.writerows((r.filament_id,r.counts) for r in rows)
        check=evaluate(truth,read_rows(path),SIZE,SIZE);assert abs(check['pq']-score['pq'])<1e-12
        reports.append(dict(**setting,**{k:score[k] for k in fields},**addition_stats[setting['name']],
            delta_pq=score['pq']-before['pq'],physical_masks=len(rows),recovered_gt=len(after-before_matches),lost_gt=0,
            recovered_gt_ids=sorted(after-before_matches),original_625_masks_preserved=True,**audit,
            **uncertainty(before,score),path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            score_recomputed_after_save=True))
    report=dict(scope='Four fixed YOLO-family support gates on27 non-GT-selected old/new native consensus additions; reused development',
        validation=dict(physical_images=91,annotator_groups=101,gt_instances=708),
        baseline={k:before[k] for k in fields},candidate_physical_additions=27,raw_record_counts=record_counts,
        experiments=reports,best=max(reports,key=lambda r:r['pq']),
        highest_recall=min(reports,key=lambda r:(r['fn'],-r['pq'])),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (BASE,CANDIDATES,*SOURCES.values())},
        candidate_support_metadata=metadata,training_performed=False,external_data_used=False,adopted=False,
        note='No further gates were tuned. Independent external confirmation required for adoption; no LB gain claimed.',
        elapsed_seconds=time.monotonic()-started)
    (ROOT/'docs/evidence/context_yolo_rescue.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(baseline=report['baseline'],experiments=reports,elapsed_seconds=report['elapsed_seconds']),indent=2))


if __name__=='__main__':main()
