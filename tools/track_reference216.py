"""Track a fixed216 external-prediction references alongside real PQ experiments.

This is prediction agreement, never ground-truth recall. No model or GT is read.
The existing public_reference_audit CSV can also be summarized without rerunning
inference or introducing another Colab cell.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from pycocotools import mask as mu
from filament_eval import Filament, _check_compressed_counts, parse_filament_id

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = REPO/'configs/reference216.json'
DEFAULT_REFERENCE = REPO.parent/'attachments/d03f8d87-a646-4f04-89dd-4ad6ab534cec/submission.csv'
REFERENCE_SHA = '0ab3932122a563ba96f4d243934d5c00680dd49b9d1165912a4f180907d2c603'
TARGETS_SHA = 'aada8589533eabe42f4260e891ed640d79bd0204e246536c5798dc3e7730fe3f'
ORIGINS = ('own_model', 'hybrid', 'public_reference')
SIZE = 2048


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_targets(path):
    value = json.loads(Path(path).read_text())
    items = value['targets']
    if value.get('count') != 216 or len(items) != 216:
        raise ValueError('The fixed reference panel must contain216 targets')
    if len({row['reference_id'] for row in items}) != 216:
        raise ValueError('Duplicate reference target')
    normalized = sorted((dict(image=row['image'],reference_id=row['reference_id']) for row in items),
                        key=lambda row:row['reference_id'])
    digest = hashlib.sha256(json.dumps(normalized,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if digest != TARGETS_SHA or value.get('target_set_sha256') != TARGETS_SHA:
        raise ValueError('Fixed216 target identity changed')
    if value.get('reference_csv_sha256') != REFERENCE_SHA:
        raise ValueError('Wrong reference source identity')
    for row in normalized:
        if parse_filament_id(row['reference_id'])[0] != row['image']:
            raise ValueError('Reference target image mismatch')
    return normalized


def read_masks(path):
    """Validate COCO runs before calling its C API, without full-image allocation."""
    csv.field_size_limit(min(sys.maxsize,2**31-1))
    result=[]; seen=set()
    with Path(path).open(newline='',encoding='utf-8-sig') as stream:
        reader=csv.DictReader(stream)
        if reader.fieldnames is None or len(reader.fieldnames)!=2 or set(reader.fieldnames)!={'filament_id','segmentation_rle'}:
            raise ValueError('Expected filament_id,segmentation_rle CSV columns')
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError('Malformed prediction CSV row')
            identifier=row['filament_id']; stem,group=parse_filament_id(identifier)
            if identifier in seen:raise ValueError('Duplicate prediction ID: '+identifier)
            seen.add(identifier); counts=row['segmentation_rle']
            area=_check_compressed_counts(counts,SIZE*SIZE)
            if area<=0:raise ValueError('Empty instance mask')
            item=Filament(identifier,stem,group,counts,area)
            if int(mu.area(item.rle(SIZE,SIZE)))!=area:raise ValueError('RLE area mismatch')
            result.append(item)
    return result


def agreement(targets, predictions):
    """Per-reference maximum IoU, with explicit false on an absent prediction."""
    by_image=defaultdict(list)
    for row in predictions:by_image[row.image_id].append(row)
    values=[]
    for target in targets:
        candidates=by_image[target.image_id]
        matrix=np.asarray(mu.iou([target.rle(SIZE,SIZE)],
            [row.rle(SIZE,SIZE) for row in candidates],[0]*len(candidates)),float) if candidates else None
        index=int(matrix[0].argmax()) if candidates else None
        score=float(matrix[0,index]) if candidates else 0.
        values.append(dict(image=target.image_id,reference_id=target.filament_id,
            max_iou=score,matched=score>.5,best_prediction_id=candidates[index].filament_id if index is not None else ''))
    return values


def summarize_scores(values):
    available=[float(v) for v in values if v is not None]
    total=len(values); count=sum(v>.5 for v in available)
    return dict(target_count=total,available_count=len(available),unavailable_count=total-len(available),
        matched_count=count if available else None,
        coverage=count/total if len(available)==total and total else None,
        mean_max_iou=float(np.mean(available)) if available else None,
        median_max_iou=float(np.median(available)) if available else None)


def changes(previous,current):
    old={r['reference_id']:r['max_iou']>.5 for r in previous}
    now={r['reference_id']:r['max_iou']>.5 for r in current}
    if set(old)!=set(now):raise ValueError('Previous/current target sets differ')
    gained=sorted(key for key in now if now[key] and not old[key])
    lost=sorted(key for key in now if old[key] and not now[key])
    return dict(gained_count=len(gained),lost_count=len(lost),gained_reference_ids=gained,lost_reference_ids=lost)


def load_audit(path, targets):
    """Missing raw values stay None; a missing target row is an input error."""
    with Path(path).open(newline='',encoding='utf-8-sig') as stream:
        reader=csv.DictReader(stream)
        required={'image','reference_id','baseline_max_iou'}
        if not required.issubset(reader.fieldnames or []):raise ValueError('Not a reference_instances.csv audit')
        by_id={}
        for row in reader:
            identifier=row['reference_id']
            if identifier in by_id:raise ValueError('Duplicate audit reference ID')
            by_id[identifier]=row
    selected=[]
    for target in targets:
        if target['reference_id'] not in by_id:raise ValueError('Missing fixed216 audit target: '+target['reference_id'])
        row=by_id[target['reference_id']]
        if row['image']!=target['image']:raise ValueError('Audit target image mismatch')
        item=dict(image=row['image'],reference_id=row['reference_id'])
        for source,key in (('raw','raw_max_iou'),('baseline','baseline_max_iou'),('r1','r1_max_iou')):
            text=row.get(key,'')
            value=None if text in ('',None) else float(text)
            if value is not None and (not math.isfinite(value) or not 0<=value<=1):
                raise ValueError('Invalid audit IoU')
            if source=='baseline' and value is None:raise ValueError('Missing baseline audit IoU')
            item[source+'_max_iou']=value
        for key in ('similarity_loss_transitions','similarity_gain_transitions','raw_best_source','raw_best_view'):
            item[key]=row.get(key,'')
        selected.append(item)
    return selected


def run(output_root, *, origin, prediction_csv=None, audit_csv=None,
        reference_csv=DEFAULT_REFERENCE, targets=DEFAULT_TARGETS,
        previous_prediction_csv=None, audit_summary=None,
        complete_test_inference=False, not_measured=False):
    if origin not in ORIGINS:raise ValueError('Explicit origin must be own_model, hybrid or public_reference')
    if prediction_csv is None and audit_csv is None and not not_measured:
        raise ValueError('Supply predictions/audit or explicitly mark --not-measured')
    if not_measured and (prediction_csv is not None or audit_csv is not None):
        raise ValueError('Not-measured mode cannot include prediction or audit data')
    if prediction_csv is not None and not complete_test_inference:
        raise ValueError('CSV tracking requires explicit --complete-test-inference; absent rows can be blank or incomplete inference')
    if previous_prediction_csv is not None and prediction_csv is None:
        raise ValueError('Previous comparison requires current prediction CSV')
    output=Path(output_root).resolve()
    if output.exists():raise ValueError('Use a new tracking output directory')
    fixed=load_targets(targets); snapshot={str(Path(targets).resolve()):sha(targets)}
    table={row['reference_id']:dict(row) for row in fixed}
    report=dict(schema_version=1,metric='reference216_agreement',prediction_origin=origin,
        scope='Agreement with fixed external predictions; not ground-truth recall or PQ',
        target_count=216,target_images=100,target_set_sha256=TARGETS_SHA,match_rule='strict mask IoU >0.5',
        leaderboard_evidence=dict(hybrid_public_lb=.42,baseline_public_lb=.39,displayed_batch_gain=.03),
        copied_reference_can_trivially_score_100_percent=True,
        model_recovery_interpretation_allowed=origin=='own_model',
        target_correctness_labels_available=False)
    report['measurement_status']='not_measured' if not_measured else 'measured'
    if not_measured:
        report['prediction']=summarize_scores([None]*216)
        report['not_measured_reason']='No complete official-test predictions or reference audit supplied; training/validation-only run'
    if prediction_csv is not None:
        if sha(reference_csv)!=REFERENCE_SHA:raise ValueError('Reference CSV hash differs from pinned public file')
        reference=read_masks(reference_csv); by_id={r.filament_id:r for r in reference}
        if any(row['reference_id'] not in by_id or by_id[row['reference_id']].image_id!=row['image'] for row in fixed):
            raise ValueError('Fixed216 targets not in reference source')
        ref_targets=[by_id[row['reference_id']] for row in fixed]
        known_images={row.image_id for row in reference}
        def read_prediction(path):
            rows=read_masks(path)
            if {row.image_id for row in rows}-known_images:raise ValueError('Unknown image in prediction CSV')
            groups=defaultdict(list)
            for row in rows:groups[row.image_id].append(row)
            for image,members in groups.items():
                if sum(r.area for r in members)!=int(mu.area(mu.merge([r.rle(SIZE,SIZE) for r in members]))):
                    raise ValueError('Prediction CSV has overlapping masks: '+image)
            return rows
        prediction=read_prediction(prediction_csv)
        current=agreement(ref_targets,prediction)
        report['prediction']=summarize_scores([row['max_iou'] for row in current])
        report['prediction'].update(rows=len(prediction),images_with_predictions=len({r.image_id for r in prediction}),
            missing_reference_ids=[row['reference_id'] for row in current if not row['matched']],
            complete_official_test_inference_declared=True,
            completeness_evidence='Caller declaration; CSV alone cannot prove images with no rows were processed',
            missing_prediction_image_policy='After explicit complete-test declaration, no row for a target image counts as no detection')
        for row in current:table[row['reference_id']].update({('prediction_'+key):value for key,value in row.items() if key not in ('image','reference_id')})
        if previous_prediction_csv is not None:
            previous=agreement(ref_targets,read_prediction(previous_prediction_csv))
            report['previous_prediction']=summarize_scores([row['max_iou'] for row in previous])
            report['change']=changes(previous,current)
            for row in previous:table[row['reference_id']]['previous_max_iou']=row['max_iou']
        for path in (reference_csv,prediction_csv,previous_prediction_csv):
            if path is not None:snapshot[str(Path(path).resolve())]=sha(path)
    if audit_csv is not None:
        selected=load_audit(audit_csv,fixed)
        report['audit']={source:summarize_scores([row[source+'_max_iou'] for row in selected])
                         for source in ('raw','baseline','r1')}
        report['audit']['raw_present_baseline_missing_count']=sum(
            row['raw_max_iou'] is not None and row['raw_max_iou']>.5 and row['baseline_max_iou']<=.5 for row in selected)
        report['audit']['raw_present_r1_missing_count']=sum(
            row['raw_max_iou'] is not None and row['raw_max_iou']>.5 and row['r1_max_iou'] is not None and row['r1_max_iou']<=.5 for row in selected)
        if report['audit']['raw']['available_count']!=216:
            report['audit']['raw_present_baseline_missing_count']=None
        if report['audit']['raw']['available_count']!=216 or report['audit']['r1']['available_count']!=216:
            report['audit']['raw_present_r1_missing_count']=None
        for row in selected:table[row['reference_id']].update({('audit_'+key):value for key,value in row.items() if key not in ('image','reference_id')})
        snapshot[str(Path(audit_csv).resolve())]=sha(audit_csv)
        report['audit']['summary_source_identity_matches']=False
        report['audit']['audit_csv_authenticated_by_summary']=False
        if audit_summary is not None:
            summary=json.loads(Path(audit_summary).read_text())
            if REFERENCE_SHA not in summary.get('input_sha256',{}).values():
                raise ValueError('Audit summary does not identify the fixed public reference source')
            report['audit']['summary_source_identity_matches']=True
            report['audit']['summary_claims_source_identity_only']=True
            snapshot[str(Path(audit_summary).resolve())]=sha(audit_summary)
    for path,expected in snapshot.items():
        if sha(path)!=expected:raise ValueError('Input changed during tracking')
    report['input_sha256']=snapshot
    report['source_sha256']=sha(__file__)
    output.mkdir(parents=True)
    rows=list(table.values()); fields=list(dict.fromkeys(key for row in rows for key in row))
    with (output/'targets216.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    report['output_csv_sha256']=sha(output/'targets216.csv')
    (output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',required=True,type=Path)
    parser.add_argument('--origin',required=True,choices=ORIGINS)
    parser.add_argument('--prediction-csv',type=Path)
    parser.add_argument('--audit-csv',type=Path)
    parser.add_argument('--reference-csv',type=Path,default=DEFAULT_REFERENCE)
    parser.add_argument('--targets',type=Path,default=DEFAULT_TARGETS)
    parser.add_argument('--previous-prediction-csv',type=Path)
    parser.add_argument('--audit-summary',type=Path)
    parser.add_argument('--complete-test-inference',action='store_true',
                        help='Declare the prediction CSV covers inference over the full official test set, including blank detections')
    parser.add_argument('--not-measured',action='store_true',help='Record unavailable216 metrics for a training/validation-only experiment')
    args=parser.parse_args(); result=run(**vars(args))
    print(json.dumps({key:result[key] for key in ('prediction_origin','target_count','prediction','audit','change') if key in result},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
