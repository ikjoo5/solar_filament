"""Fixed-model error/generalization audit; creates no GT-dependent predictions."""
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mu
from scipy.optimize import linear_sum_assignment

from analyze_validation_results import analyze, read_rows

ROOT=Path(__file__).resolve().parents[1]
ANN=Path('/workspace/datasets/solar_filament/MAGFiLO_1.0_Annotations_kaggle2026_train.json')
GT=Path('/workspace/artifacts/solar_filament_gt_v1/gt.csv')
MANIFEST=GT.with_name('manifest.json')
TRAIN=Path('/workspace/attachments/1cca1607-f94f-4334-9586-c1fb5ab292d7/train_split.csv')
VAL=Path('/workspace/attachments/bcb64220-b208-4cd7-9c3f-3f2be0f04167/val_split.csv')
PRED={
 'original_yolo':Path('/workspace/artifacts/validation_results_20261027T112905Z_947c4b1f/union_cc_conf0.25.csv'),
 'old_consensus':Path('/workspace/artifacts/morph_order_candidates_v1/before0_close5_after300.csv'),
 'native_unet':Path('/workspace/artifacts/a100_results_20261028_bedf3e21/best_validation.csv'),
 'frozen_highest':Path('/workspace/artifacts/rich_boundary_composition_v1/tta_precision.csv'),
}
RAW=[
 Path('/workspace/artifacts/validation_results_20261027T120958Z_444da7f4/raw_views.jsonl'),
 Path('/workspace/artifacts/validation_results_20261027T125130Z_2312e759/raw_preprocess_views.jsonl'),
 Path('/workspace/artifacts/tiled_results_20261027T205426Z_eba01cfb/raw_tiled_views.jsonl'),
]


def physical_metrics(run):
    grouped=defaultdict(lambda:np.zeros(4,np.float64))
    for row in run['group_results']:
        image=row['group'].split('-',1)[1]
        grouped[image]+=np.asarray([row['iou_sum'],row['tp'],row['fp'],row['fn']])
    return grouped


def pq(values):
    return values[...,0]/(values[...,1]+.5*values[...,2]+.5*values[...,3])


def bootstrap(runs,reference='original_yolo',draws=10000):
    matrices={name:physical_metrics(run) for name,run in runs.items()}
    images=sorted(matrices[reference]);rng=np.random.default_rng(20261028)
    arrays={name:np.stack([values[image] for image in images]) for name,values in matrices.items()}
    indices=rng.integers(0,len(images),(draws,len(images)))
    point=pq(arrays[reference].sum(axis=0));sample=pq(arrays[reference][indices].sum(axis=1))
    results={}
    for name,values in arrays.items():
        scores=pq(values[indices].sum(axis=1));delta=scores-sample
        results[name]=dict(pq=float(pq(values.sum(axis=0))),delta_vs_reference=float(pq(values.sum(axis=0))-point),
            paired_delta_percentile95=[float(x) for x in np.quantile(delta,[.025,.975])],
            fraction_bootstrap_delta_positive=float((delta>0).mean()))
    return dict(resampling_unit='Physical image; all annotator groups remain together',draws=draws,
        reference=reference,results=results,
        limitation='Conditional stability of already selected models on reused development data; excludes hyperparameter-selection bias and is not an independent holdout estimate.')


def annotation_disagreement(train_truth,images):
    groups=defaultdict(list)
    for image_id,rows in train_truth.items():groups[images[image_id]['file_name']].append((image_id,rows))
    pair_records=[]
    for filename,variants in sorted(groups.items()):
        for i in range(len(variants)):
            for j in range(i+1,len(variants)):
                aid,a=variants[i];bid,b=variants[j]
                matrix=np.asarray(mu.iou([r['rle'] for r in a],[r['rle'] for r in b],[0]*len(b)))
                aa,bb=linear_sum_assignment(1-matrix)
                matches=int((matrix[aa,bb]>.5).sum())
                union_a,union_b=mu.merge([r['rle'] for r in a]),mu.merge([r['rle'] for r in b])
                area_a,area_b=int(mu.area(union_a)),int(mu.area(union_b))
                intersection=int(mu.area(mu.merge([union_a,union_b],intersect=True)))
                pair_records.append(dict(file_name=filename,group_a=aid,group_b=bid,objects_a=len(a),objects_b=len(b),
                    one_to_one_iou_gt_half_matches=matches,unmatched_a=len(a)-matches,unmatched_b=len(b)-matches,
                    no_counterpart_iou_ge_point1_a=int((matrix.max(axis=1)<.1).sum()),
                    no_counterpart_iou_ge_point1_b=int((matrix.max(axis=0)<.1).sum()),
                    foreground_union_iou=intersection/(area_a+area_b-intersection),
                    fraction_a_foreground_not_in_b=1-intersection/area_a,
                    fraction_b_foreground_not_in_a=1-intersection/area_b))
    return dict(train_physical_images=len(groups),variants_per_physical_image=dict(Counter(map(len,groups.values()))),
        image_pairs=len(pair_records),paired_objects_total=sum(r['objects_a']+r['objects_b'] for r in pair_records),
        unmatched_objects_total=sum(r['unmatched_a']+r['unmatched_b'] for r in pair_records),
        no_counterpart_iou_ge_point1_total=sum(r['no_counterpart_iou_ge_point1_a']+r['no_counterpart_iou_ge_point1_b'] for r in pair_records),
        foreground_union_iou_quantiles={str(q):float(np.quantile([r['foreground_union_iou'] for r in pair_records],q)) for q in (0,.1,.25,.5,.75,.9,1)},
        matching_note='Hungarian matching is only an annotation-correspondence diagnostic here; official evaluation still uses all pairs IoU>.5.',
        interpretation='Different annotations are not declared wrong. Alternate complete annotation sets can give conflicting foreground/background supervision.',
        pairs=pair_records)


def main():
    data=json.loads(ANN.read_text());images={r['id']:r for r in data['images']}
    train=list(csv.DictReader(TRAIN.open()));val=list(csv.DictReader(VAL.open()))
    train_ids={r['image_id'] for r in train};val_ids={r['image_id'] for r in val}
    train_files={r['file_name'] for r in train};val_files={r['file_name'] for r in val}
    assert not train_files & val_files and len(train_files)==616 and len(val_files)==91
    truth=read_rows(GT,gt=True,allowed_groups=val_ids)
    train_truth=read_rows(GT,gt=True,allowed_groups=train_ids)
    manifest=json.loads(MANIFEST.read_text());meta={r['filament_id']:r for r in manifest['annotations']}
    edges=np.quantile([r['area'] for rows in train_truth.values() for r in rows],[.25,.5,.75])
    runs={name:analyze(path,truth,meta,edges) for name,path in PRED.items()}
    assert abs(runs['frozen_highest']['pq']-.47259304445299505)<1e-12
    candidates=defaultdict(list)
    for path in RAW:
        for line in path.open():
            record=json.loads(line)
            candidates[record['image']].extend(dict(size=[2048,2048],counts=p['rle'].encode()) for p in record['instances'])
    for line in Path('/workspace/artifacts/best84_rich_predictions_tta/predictions.jsonl').open():
        record=json.loads(line);image=Path(record['file_name']).stem
        for source in (record,record['tta']):
            for components in source['components'].values():
                candidates[image].extend(dict(size=[2048,2048],counts=c['rle']['counts'].encode()) for c in components)
    missed={r['id']:r for r in runs['frozen_highest']['gt_records'] if not r['matched']}
    raw_available=Counter()
    for group,rows in truth.items():
        image=group.split('-',1)[1];proposals=candidates[image]
        matrix=np.asarray(mu.iou([r['rle'] for r in rows],proposals,[0]*len(proposals)))
        for row,best in zip(rows,matrix.max(axis=1)):
            if row['id'] in missed:
                raw_available['candidate_iou_gt_half' if best>.5 else 'best_candidate_point3_to_half' if best>=.3 else 'best_candidate_below_point3']+=1
    by_file={r['file_name']:r for r in data['images']}
    day=lambda f:np.datetime64(by_file[f]['date_captured'][:10],'D')
    train_days=np.asarray([day(f) for f in sorted(train_files)])
    val_days=np.asarray([day(f) for f in sorted(val_files)])
    distances=np.abs((val_days[:,None]-train_days[None,:]).astype('timedelta64[D]').astype(int)).min(axis=1)
    temporal=dict(validation_physical_images=91,nearest_training_day_distances={str(n):int((distances<=n).sum()) for n in (0,1,3,7,14,27)},
        note='Distinct files/annotator groups remain disjoint. Nearby dates may share evolving solar structures; this is a dependence diagnostic, not proof of duplicate-pixel leakage.')
    history=dict(previous_user_reported_public_lb=.37,current_user_reported_public_lb=.39,absolute_public_gain=.02,
        original_development_pq=runs['original_yolo']['pq'],current_development_pq=runs['frozen_highest']['pq'],
        original_development_minus_public=runs['original_yolo']['pq']-.37,
        current_development_minus_public=runs['frozen_highest']['pq']-.39,
        leader_public_lb_user_reported=.7,leader_verified_independently=False,
        note='A larger development/public gap alone does not establish overfitting; distribution shift, sampling and repeated selection can all contribute.')
    report=dict(scope='Round8 fixed-pipeline diagnosis and training-only annotation audit; no newprediction gates or training',
        sources=dict(annotations_sha256=hashlib.sha256(ANN.read_bytes()).hexdigest(),prediction_sha256={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in PRED.items()}),
        leaderboard=history,train_only_area_quartile_edges=edges.tolist(),runs=runs,
        paired_physical_bootstrap=bootstrap(runs),current_FN_raw_candidate_availability=dict(raw_available),
        candidate_availability_note='GT-aware diagnostic across every cached candidate, not an achieved recall selector.',
        temporal_independence=temporal,training_annotations=annotation_disagreement(train_truth,images))
    destination=ROOT/'docs/evidence/round8_generalization_audit.json'
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(leaderboard=history,temporal=temporal,bootstrap=report['paired_physical_bootstrap'],
        current_FN_candidate_availability=dict(raw_available),current_error_summary={k:v for k,v in runs['frozen_highest'].items() if k not in ('gt_records','group_results')},
        training_annotations={k:v for k,v in report['training_annotations'].items() if k!='pairs'}),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
