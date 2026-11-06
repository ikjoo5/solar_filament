"""GT-aware diagnosis of raw proposal availability, never oracle predictions.

No pixels are decoded and no masks are selected for a new prediction. Counts
are per annotator's GT; multiple annotations of one image are correlated.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from pycocotools import mask as mu

from analyze_validation_results import read_rows
from filament_eval import _check_compressed_counts


ROOT = Path(__file__).resolve().parents[1]
OLD = Path('/workspace/artifacts/validation_results_20261027T120958Z_444da7f4')
NEW = Path('/workspace/artifacts/validation_results_20261027T125130Z_2312e759')
GT = Path('/workspace/artifacts/solar_filament_gt_v1/gt.csv')
MANIFEST = GT.parent/'manifest.json'
VAL = Path('/workspace/attachments/bcb64220-b208-4cd7-9c3f-3f2be0f04167/val_split.csv')
BEST = Path('/workspace/artifacts/morph_order_candidates_v1/before0_close5_after300.csv')
OLD_VIEWS = ('base','scale','hflip','vflip')
NEW_VIEWS = ('limb','clahe16','clahe32','percentile','unsharp','gaussian','bilateral','blackhat')
CONFIDENCES = (.05,.10,.15,.20,.25)


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def cache(path, expected_images, views):
    grouped = defaultdict(dict)
    masks = records = capped = 0
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            image,view = row['image'],row['view']
            assert image in expected_images and view in views
            assert row['shape']==[2048,2048] and row['coordinate_space']=='original'
            assert view not in grouped[image], (image,view,'duplicate record')
            assert row['raw_min_confidence']==.05
            for instance in row['instances']:
                assert np.isfinite(instance['confidence']) and .05<=instance['confidence']<=1
                assert _check_compressed_counts(instance['rle'],2048**2)>0
                masks += 1
            grouped[image][view] = row['instances']
            records += 1
            capped += bool(row.get('max_det_reached'))
    assert set(grouped)==expected_images
    assert all(set(v)==set(views) for v in grouped.values())
    return grouped,dict(records=records,masks=masks,capped_records=capped,sha256=digest(path))


def fingerprint(instances):
    pairs = sorted((p['rle'],p['confidence']) for p in instances)
    return hashlib.sha256(json.dumps(pairs,separators=(',',':')).encode()).hexdigest()


def overlaps(truth, instances):
    if not instances:
        return np.zeros((len(truth),0)),np.empty(0)
    rles = [{'size':[2048,2048],'counts':p['rle'].encode('ascii')} for p in instances]
    return np.asarray(mu.iou(truth,rles,[0]*len(rles))),np.array([p['confidence'] for p in instances])


def available(matrix, confidences, threshold):
    return (matrix[:,confidences>=threshold]>.5).any(axis=1)


def main():
    started=time.monotonic()
    old_summary=json.loads((OLD/'summary.json').read_text())
    new_summary=json.loads((NEW/'summary.json').read_text())
    assert old_summary['source_sha256']==new_summary['source_sha256']
    assert new_summary['images']==old_summary['images']==91
    assert new_summary['annotation_groups']==old_summary['annotation_groups']==101
    conversion=json.loads((GT.parent/'summary.json').read_text())
    ann_name='MAGFiLO_1.0_Annotations_kaggle2026_train.json'
    assert conversion['source']['sha256']==new_summary['source_sha256'][ann_name]
    assert digest(VAL)==new_summary['source_sha256']['val_split.csv']
    with VAL.open(newline='') as handle:
        selected=list(csv.DictReader(handle))
    ids={r['image_id'] for r in selected}
    images={Path(r['file_name']).stem for r in selected}
    truth=read_rows(GT,gt=True,allowed_groups=ids)
    assert set(truth)==ids and sum(map(len,truth.values()))==708
    prediction=read_rows(BEST)
    old,old_metadata=cache(OLD/'raw_views.jsonl',images,OLD_VIEWS)
    new,new_metadata=cache(NEW/'raw_preprocess_views.jsonl',images,NEW_VIEWS)
    baseline_differences=[image for image in sorted(images)
        if fingerprint(old[image]['base'])!=fingerprint(new[image]['limb'])]
    manifest=json.loads(MANIFEST.read_text())
    meta={a['filament_id']:a for a in manifest['annotations']}
    per_view={view:{str(c):dict(gt_available=0,current_fn_available=0,
        beyond_old4_available=0,current_fn_beyond_old4=0) for c in CONFIDENCES}
        for view in NEW_VIEWS}
    old_counts={str(c):0 for c in CONFIDENCES}
    new_pool_counts={str(c):0 for c in CONFIDENCES}
    combined_counts={str(c):0 for c in CONFIDENCES}
    totals=Counter();new_only_categories=Counter();new_only_areas=[]
    new_only_view_support=Counter();new_only_view_contributions=Counter()
    records=[]
    for group,rows in truth.items():
        image=group.split('-',1)[1]
        gr=[r['rle'] for r in rows]
        current=[r['rle'] for r in prediction.get(image,[])]
        ci=np.asarray(mu.iou(gr,current,[0]*len(current))) if current else np.zeros((len(gr),0))
        matched=(ci>.5).any(axis=1);missed=~matched
        old_instances=[p for view in OLD_VIEWS for p in old[image][view]]
        oi,oc=overlaps(gr,old_instances)
        old_flags={c:available(oi,oc,c) for c in CONFIDENCES}
        new_flags={c:np.zeros(len(rows),bool) for c in CONFIDENCES}
        view_flags={}
        for view in NEW_VIEWS:
            vi,vc=overlaps(gr,new[image][view])
            view_flags[view]={c:available(vi,vc,c) for c in CONFIDENCES}
            for c in CONFIDENCES:
                flags=view_flags[view][c];new_flags[c]|=flags
                report=per_view[view][str(c)]
                report['gt_available']+=int(flags.sum())
                report['current_fn_available']+=int((flags&missed).sum())
                report['beyond_old4_available']+=int((flags&~old_flags[c]).sum())
                report['current_fn_beyond_old4']+=int((flags&missed&~old_flags[c]).sum())
        for c in CONFIDENCES:
            old_counts[str(c)]+=int(old_flags[c].sum())
            new_pool_counts[str(c)]+=int(new_flags[c].sum())
            combined_counts[str(c)]+=int((old_flags[c]|new_flags[c]).sum())
        of,nf=old_flags[.05],new_flags[.05]
        totals.update(gt=len(rows),current_matched=int(matched.sum()),current_fn=int(missed.sum()),
                      current_fn_old_raw_available=int((missed&of).sum()),
                      current_fn_new_raw_available=int((missed&nf).sum()),
                      current_fn_raw_combined_available=int((missed&(of|nf)).sum()),
                      current_fn_new_only_available=int((missed&nf&~of).sum()),
                      all_gt_new_only_available=int((nf&~of).sum()),
                      current_fn_no_qualifying_any_raw=int((missed&~(of|nf)).sum()))
        for i,row in enumerate(rows):
            if nf[i] and not of[i]:
                new_only_categories[meta[row['id']]['category_name']]+=1
                new_only_areas.append(row['area'])
                supporting=[v for v in NEW_VIEWS if view_flags[v][.05][i]]
                new_only_view_support[len(supporting)]+=1
                new_only_view_contributions.update(supporting)
            if missed[i]:
                records.append(dict(gt_id=row['id'],image=image,category=meta[row['id']]['category_name'],
                    area=row['area'],old_raw_available=bool(of[i]),new_raw_available=bool(nf[i]),
                    preprocessing_views_with_qualifying_mask=[v for v in NEW_VIEWS if view_flags[v][.05][i]]))
    assert totals['current_matched']==467 and totals['current_fn']==241
    assert old_counts['0.05']==639 and totals['current_fn_old_raw_available']==172
    output=dict(scope='GT-aware proposal-availability diagnosis only; no oracle prediction selection, achieved PQ, or deployable recall',
        rule='Unique GT with ANY raw mask IoU>0.5 at the stated confidence; annotator variants are correlated',
        source_sha256=new_summary['source_sha256'],old_cache=old_metadata,new_cache=new_metadata,
        baseline_identity=dict(equal_images=91-len(baseline_differences),different_images=baseline_differences,
            method='Exact sorted (compressed RLE, real confidence) fingerprint; old base and new limb must not be counted as independent views'),
        current_prediction=str(BEST),counts=dict(totals),
        old4_raw_availability_by_confidence=old_counts,new8_raw_availability_by_confidence=new_pool_counts,
        combined_raw_availability_by_confidence=combined_counts,per_preprocessing_view=per_view,
        new_only_gt_category_counts=dict(new_only_categories),
        new_only_gt_preprocessing_view_support_histogram=dict(sorted(new_only_view_support.items())),
        new_only_gt_preprocessing_view_contributions=dict(new_only_view_contributions),
        new_only_gt_area_quantiles={str(q):float(np.quantile(new_only_areas,q)) for q in [0,.25,.5,.75,1]} if new_only_areas else {},
        current_fn_diagnostics=records,
        limitations=['Does not quantify an executable FP-controlled rescue rule.',
                    'Dataset metadata alone does not establish image contrast or why a preprocessing view changes a proposal.',
                    'Availability is not independent CV or leaderboard improvement.'],elapsed_seconds=time.monotonic()-started)
    path=ROOT/'docs/evidence/preprocessing_candidate_diagnostics.json'
    path.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:output[k] for k in ['baseline_identity','counts','old4_raw_availability_by_confidence','new8_raw_availability_by_confidence','combined_raw_availability_by_confidence','new_only_gt_category_counts','new_only_gt_area_quantiles','elapsed_seconds']},indent=2))
    for view in NEW_VIEWS:print(view,json.dumps(per_view[view]))


if __name__=='__main__':main()
