"""Evaluation-only external COCO planning and scoring; no model fitting APIs."""
from collections import Counter, defaultdict
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np
from pycocotools import mask as mu
from submission_driver import checked_rle

IMAGE_EXTENSIONS={'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}
GONG=re.compile(r'(\d{14}[a-z]h)',re.I)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def physical_key(value):
    match=GONG.search(str(value))
    return match.group(1).lower() if match else None


def capture_day(row):
    value=str(row.get('date_captured',''))[:10]
    try:return date.fromisoformat(value)
    except ValueError:
        key=physical_key(row.get('file_name','')) or physical_key(row.get('url',''))
        return date(int(key[:4]),int(key[4:6]),int(key[6:8])) if key else None


def fingerprint(path):
    path=Path(path);data=path.read_bytes()
    image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape!=(2048,2048) or image.dtype!=np.uint8:
        raise ValueError(f'Native uint8 2048-square image required: {path}')
    return dict(file_sha256=sha(data),pixel_sha256=sha(image.tobytes()))


def assign_external_splits(records,seed=42,confirm_fraction=.3,embargo_days=3):
    """27-day temporal blocks; date/site views share a split; no label access."""
    if not 0<confirm_fraction<1 or embargo_days<0:
        raise ValueError('Invalid external evaluation split settings')
    epoch=date(2000,1,1)
    for record in records:
        day=date.fromisoformat(record['capture_day']) if record['capture_day'] else None
        block=str((day-epoch).days//27) if day else 'undated:'+record['pixel_sha256']
        value=int(sha(f'{seed}:{block}'.encode())[:16],16)/2**64
        record['time_block_27days']=block
        record['split']='external_confirm' if value<confirm_fraction else 'external_dev'
    confirm_days=[date.fromisoformat(r['capture_day']) for r in records
                  if r['split']=='external_confirm' and r['capture_day']]
    for record in records:
        if record['split']=='external_dev' and record['capture_day'] and confirm_days:
            day=date.fromisoformat(record['capture_day'])
            if min(abs((day-other).days) for other in confirm_days)<=embargo_days:
                record['split']='external_embargo'
    return records


def prepare_external_plan(annotations,images,official_train_images,official_test_images,
                          seed=42,confirm_fraction=.3,embargo_days=3):
    """Read external labels only as an inert JSON object; selection uses metadata."""
    ann_path,root=Path(annotations),Path(images)
    if not root.is_dir():raise FileNotFoundError(root)
    raw=ann_path.read_bytes();data=json.loads(raw)
    if not isinstance(data.get('images'),list) or not isinstance(data.get('annotations'),list):
        raise ValueError('External annotations must use COCO images/annotations lists')
    ids=[str(row['id']) for row in data['images']]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate external annotation image IDs')
    known_ids=set(ids)
    if any(str(row['image_id']) not in known_ids for row in data['annotations']):
        raise ValueError('External annotation references an unknown image ID')
    official_keys=set();official_file_hashes=set();official_pixel_hashes=set();official_days=set()
    official_counts={}
    for role,source in [('competition_train_validation',official_train_images),('competition_test',official_test_images)]:
        source=Path(source)
        paths=sorted(p for p in source.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS) if source.is_dir() else []
        if not paths:raise ValueError(f'Complete official image directory required: {source}')
        official_counts[role]=len(paths)
        for path in paths:
            info=fingerprint(path);official_file_hashes.add(info['file_sha256']);official_pixel_hashes.add(info['pixel_sha256'])
            key=physical_key(path.name)
            if key:
                official_keys.add(key);official_days.add(date(int(key[:4]),int(key[4:6]),int(key[6:8])))
    paths=sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    by_name=defaultdict(list)
    for path in paths:by_name[path.name].append(path)
    cache={};buckets=defaultdict(list);excluded=[];canonical_pixels=defaultdict(set)
    for row in data['images']:
        identifier=str(row['id']);name=Path(str(row['file_name']).replace('\\','/')).name
        key=physical_key(name) or physical_key(row.get('url',''))
        if key in official_keys:
            excluded.append(dict(image_id=identifier,file_name=name,reason='official_physical_identifier'));continue
        relative=Path(str(row['file_name']).replace('\\','/'))
        candidate=root/relative
        if not candidate.resolve().is_relative_to(root.resolve()) or not candidate.is_file():
            values=by_name.get(name,[])
            if len(values)!=1:
                excluded.append(dict(image_id=identifier,file_name=name,reason='missing_or_ambiguous_image'));continue
            candidate=values[0]
        if row.get('width',2048)!=2048 or row.get('height',2048)!=2048:
            excluded.append(dict(image_id=identifier,file_name=name,reason='unsupported_native_dimensions'));continue
        if candidate not in cache:
            cache[candidate]=fingerprint(candidate)
        info=cache[candidate]
        if info['file_sha256'] in official_file_hashes or info['pixel_sha256'] in official_pixel_hashes:
            excluded.append(dict(image_id=identifier,file_name=name,reason='official_file_or_decoded_pixel_duplicate'));continue
        if key:canonical_pixels[key].add(info['pixel_sha256'])
        day=capture_day(row)
        buckets[info['pixel_sha256']].append(dict(image_id=identifier,file_name=name,path=str(candidate.resolve()),
            physical_identifier=key,capture_day=day.isoformat() if day else None,**info))
    conflicts={key for key,values in canonical_pixels.items() if len(values)>1}
    records=[]
    for pixel_hash,members in sorted(buckets.items()):
        if any(row['physical_identifier'] in conflicts for row in members):
            excluded.extend(dict(image_id=r['image_id'],file_name=r['file_name'],reason='conflicting_versions_of_same_physical_identifier') for r in members)
            continue
        first=sorted(members,key=lambda r:(r['file_name'],r['image_id']))[0]
        keys=sorted({r['physical_identifier'] for r in members if r['physical_identifier']})
        days=sorted({r['capture_day'] for r in members if r['capture_day']})
        if len(days)>1:raise ValueError('Identical decoded image has conflicting capture dates')
        nearest=min(abs((date.fromisoformat(days[0])-d).days) for d in official_days) if days and official_days else None
        records.append(dict(evaluation_key=keys[0] if keys else 'external'+pixel_hash[:20],
            source_image_ids=sorted(r['image_id'] for r in members),path=first['path'],file_name=first['file_name'],
            file_sha256=first['file_sha256'],pixel_sha256=pixel_hash,capture_day=days[0] if days else None,
            observatory=keys[0][-2:] if keys else 'unknown',nearest_official_day_distance=nearest,
            time_clean_3days=nearest is not None and nearest>3))
    records=assign_external_splits(records,seed,confirm_fraction,embargo_days)
    if len({r['evaluation_key'] for r in records})!=len(records):raise ValueError('Evaluation key collision')
    summary=dict(source_annotation_records=len(data['images']),eligible_physical_images=len(records),
        eligible_annotation_groups=sum(len(r['source_image_ids']) for r in records),
        excluded_annotation_records=len(excluded),exclusion_reasons=dict(Counter(r['reason'] for r in excluded)),
        split_physical_counts=dict(Counter(r['split'] for r in records)),
        split_time_clean_3day_counts=dict(Counter(r['split'] for r in records if r['time_clean_3days'])),
        temporal_block_count_27days=len({r['time_block_27days'] for r in records}),
        observatories=dict(Counter(r['observatory'] for r in records)),official_image_counts=official_counts)
    return dict(schema_version=1,data_role='external_evaluation_only',allow_training=False,allow_pseudo_label_training=False,
        annotations=str(ann_path.resolve()),annotations_sha256=sha(raw),images=str(root.resolve()),
        split_policy=dict(seed=seed,group='27-day blocks across all observatories',confirm_fraction=confirm_fraction,
            dev_confirm_embargo_days=embargo_days,labels_used_for_assignment=False,
            default_evaluation='external_dev',confirmation_scores_not_computed_during_preparation=True),
        summary=summary,records=records,excluded=excluded)


def annotation_rle(annotation):
    seg=annotation['segmentation']
    if isinstance(seg,list):
        if not seg or any(len(poly)<6 or len(poly)%2 or not np.isfinite(poly).all() for poly in seg):
            raise ValueError('Invalid external polygon')
        result=mu.merge(mu.frPyObjects(seg,2048,2048))
    elif isinstance(seg,dict) and isinstance(seg.get('counts'),list):result=mu.frPyObjects(seg,2048,2048)
    elif isinstance(seg,dict):
        result=dict(seg)
        if isinstance(result.get('counts'),str):result['counts']=result['counts'].encode('ascii')
    else:raise ValueError('Unsupported COCO segmentation')
    if result['size']!=[2048,2048] or int(mu.area(result))<=0:raise ValueError('Empty/invalid external GT mask')
    return result


def score_external(plan,predictions,split='external_dev'):
    """Official all-pairs PQ; report blank GT groups explicitly as a second metric."""
    if plan.get('data_role')!='external_evaluation_only' or plan.get('allow_training') is not False:
        raise ValueError('An evaluation-only plan is required')
    if split not in ('external_dev','external_confirm'):raise ValueError('Choose dev or reserved confirmation explicitly')
    source=Path(plan['annotations'])
    if sha(source.read_bytes())!=plan['annotations_sha256']:raise ValueError('External annotations changed after planning')
    records=[r for r in plan['records'] if r['split']==split]
    if not records:raise ValueError(f'No eligible physical images in {split}')
    if set(predictions)!={r['evaluation_key'] for r in records}:raise ValueError('Predictions must cover every selected physical image, including empty results')
    predictions={key:[checked_rle(rle) for rle in values] for key,values in predictions.items()}
    for values in predictions.values():
        if values and int(mu.area(mu.merge(values)))!=sum(int(mu.area(value)) for value in values):
            raise ValueError('External predictions overlap; duplicate-mask PQ inflation is not accepted')
    data=json.loads(source.read_text());allowed={sid:r for r in records for sid in r['source_image_ids']}
    truth=defaultdict(list)
    for row in data['annotations']:
        if str(row['image_id']) in allowed:truth[str(row['image_id'])].append(annotation_rle(row))
    groups=[]
    for sid,record in sorted(allowed.items()):
        gt=truth[sid];pred=predictions[record['evaluation_key']]
        if gt and pred:
            matrix=np.asarray(mu.iou(gt,pred,[0]*len(pred)));hits=matrix>.5
            tp=int(hits.sum());fp=int((~hits.any(axis=0)).sum());fn=int((~hits.any(axis=1)).sum());iou=float(matrix[hits].sum())
        else:tp,fp,fn,iou=0,len(pred),len(gt),0.
        groups.append(dict(group_id=sid,evaluation_key=record['evaluation_key'],observatory=record['observatory'],
            time_block_27days=record['time_block_27days'],time_clean_3days=record['time_clean_3days'],
            gt_instances=len(gt),tp_pairs=tp,fp=fp,fn=fn,matched_iou_sum=iou))
    def aggregate(values):
        tp,fp,fn=[sum(r[k] for r in values) for k in ('tp_pairs','fp','fn')];iou=sum(r['matched_iou_sum'] for r in values)
        den=tp+.5*fp+.5*fn
        return dict(pq=iou/den if den else 0.,sq=iou/tp if tp else 0.,rq=tp/den if den else 0.,
            tp_pairs=tp,fp=fp,fn=fn,groups=len(values),physical_images=len({r['evaluation_key'] for r in values}))
    positive=[r for r in groups if r['gt_instances']]
    return dict(split=split,training_performed=False,physical_images=len(records),annotation_groups=len(groups),
        official_nonblank_pq=aggregate(positive),all_metadata_groups_including_blank_GT=aggregate(groups),
        time_clean_3day_nonblank=aggregate([r for r in positive if r['time_clean_3days']]),
        by_observatory={site:aggregate([r for r in positive if r['observatory']==site]) for site in sorted({r['observatory'] for r in positive})},
        temporal_block_count_27days=len({r['time_block_27days'] for r in records}),per_group=groups,
        interpretation='Separate external evaluation, never pooled with the repeatedly tuned original91-image development score.')
