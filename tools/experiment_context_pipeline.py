"""Eight fixed, label-free applications of the new context model to our pipeline.

This is reused-development ablation, not external validation or a new LB score.
Frozen submission code is imported unchanged; GT is opened after all CSVs exist.
"""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import time

import cv2
from pycocotools import mask as mu

from experiment_mask_postprocess import group_rows, overlap_stats, read_rows
from experiment_native_cross_model import matched_ids
from experiment_paper_densecrf import paired_delta, site_metrics
from filament_eval import Filament, evaluate
from submission_native_recipe import (_add_native_reference, _final_admission_pruning,
    _refine_boundaries, _weak_proposals)
from verify_submission_native_recipe import BASE, OLD, PREP, TILES, RICH, EXPECTED, load_raw

ROOT=Path(__file__).resolve().parents[1]
UPLOAD=Path('/workspace/artifacts/context_only_uploaded_20261029/multiscale_context/raw_paper_views.jsonl')
OUT=Path('/workspace/artifacts/context_pipeline_v1')
NAMES=('frozen_best84','context_replace_all','context_boundary_only',
    'context_final_only','context_reference_only','cross_model_tta_boundary',
    'context_admit_only','context_prune_only')


def variants(image,yolo,old,new,weak):
    old_native,_=_add_native_reference(yolo,old,weak)
    new_native,_=_add_native_reference(yolo,new,weak)
    old_refined,_=_refine_boundaries(old_native,old)
    new_refined,_=_refine_boundaries(new_native,new)
    old_final,_=_final_admission_pruning(old_refined,old)
    new_final,_=_final_admission_pruning(new_refined,new)
    boundary,_=_refine_boundaries(old_native,new)
    reference,_=_refine_boundaries(new_native,old)
    # Same frozen boundary rule, with one high-threshold peer per model.
    cross=dict(components={'0.65':old['tta']['components']['0.65']},
               tta=dict(components={'0.65':new['tta']['components']['0.65']}))
    cross_refined,_=_refine_boundaries(old_native,cross)
    final_only,_=_final_admission_pruning(old_refined,new)
    proposed,decision=_final_admission_pruning(old_final,new)
    added=decision['final_added']
    admit=old_final+(proposed[-added:] if added else [])
    prune=proposed[:-added] if added else proposed
    assert len(admit)>=len(old_final) and admit[:len(old_final)]==old_final
    return dict(zip(NAMES,(old_final,new_final,
        _final_admission_pruning(boundary,old)[0],final_only,
        _final_admission_pruning(reference,old)[0],
        _final_admission_pruning(cross_refined,old)[0],admit,prune)))


def save(path,image_masks):
    with path.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['filament_id','segmentation_rle'])
        for image,masks in sorted(image_masks.items()):
            for index,rle in enumerate(masks,1):
                counts=rle['counts'];counts=counts.decode('ascii') if isinstance(counts,bytes) else counts
                w.writerow([f'{image}_{index}',counts])


def main():
    cv2.setNumThreads(1);started=time.monotonic()
    OUT.mkdir(exist_ok=False)
    (OUT/'predeclared_plan.json').write_text(json.dumps(dict(names=NAMES,
        fixed_thresholds_from_frozen_recipe=True,no_GT_selector=True,
        scope='Same fixed91-image development set; not independent confirmation'),indent=2)+'\n')
    old={Path(r['file_name']).stem:r for r in map(json.loads,RICH.open())}
    new={Path(r['file_name']).stem:r for r in map(json.loads,UPLOAD.open())}
    assert len(new)==len(old)==91 and new.keys()==old.keys()
    assert all(new[k]['image_sha256']==old[k]['image_sha256'] for k in old)
    geo,prep,tiles=load_raw(OLD),load_raw(PREP),load_raw(TILES)
    yolo=group_rows(read_rows(BASE));expected=group_rows(read_rows(EXPECTED['final']))
    outputs={name:{} for name in NAMES}
    for index,image in enumerate(sorted(new),1):
        weak=_weak_proposals(image,geo[image],prep[image],tiles[image])
        result=variants(image,[r.rle(2048,2048) for r in yolo.get(image,[])],old[image],new[image],weak)
        # Exact reproduction of already-selected baseline before scoring new methods.
        assert Counter(r['counts'].decode() for r in result['frozen_best84'])==Counter(r.counts for r in expected.get(image,[]))
        for name,masks in result.items():outputs[name][image]=masks
        if index%15==0:print('ContextPipeline',index,'/91',flush=True)
    for name,masks in outputs.items():save(OUT/(name+'.csv'),masks)
    del outputs
    # All eight candidate CSVs have now been produced, without loading labels.
    from verify_native_training_results import GT, VAL
    groups={r['image_id'] for r in csv.DictReader(VAL.open())}
    truth=read_rows(GT,gt=True,selected=groups)
    assert len(truth)==708 and len(groups)==101
    rows=read_rows(OUT/'frozen_best84.csv');reference=evaluate(truth,rows,2048,2048)
    assert abs(reference['pq']-.47259304445299505)<1e-12
    before=matched_ids(truth,rows);reports=[]
    for name in NAMES:
        path=OUT/(name+'.csv');pred=read_rows(path);score=evaluate(truth,pred,2048,2048)
        audit=overlap_stats(pred);assert not any(audit.values())
        assert score['tp_pairs']==len(truth)-score['fn']
        after=matched_ids(truth,pred)
        reports.append(dict(name=name,**{k:score[k] for k in ('pq','sq','rq','tp_pairs','fp','fn')},
            physical_masks=len(pred),recovered_gt=len(after-before),lost_gt=len(before-after),
            delta_pq=score['pq']-reference['pq'],**audit,extra_matching_pairs=0,
            **paired_delta(reference,score),by_site=site_metrics(score),
            csv=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    report=dict(scope='Eight fixed frozen-recipe stage ablations on our reused91-image development cohort',
        validation=dict(physical_images=91,annotation_groups=101,gt_instances=708),
        upload_sha256=hashlib.sha256(UPLOAD.read_bytes()).hexdigest(),
        image_hashes_match_prior_all91=True,baseline_mask_parity=True,experiments=reports,
        best=max(reports,key=lambda r:r['pq']),adopted=False,
        external_evaluated=False,training_performed_here=False,
        elapsed_seconds=time.monotonic()-started)
    path=ROOT/'docs/evidence/context_pipeline_experiments.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(experiments=[{k:r[k] for k in ('name','pq','tp_pairs','fp','fn','recovered_gt','lost_gt','physical_masks')} for r in reports],
        best=report['best']['name'],elapsed_seconds=report['elapsed_seconds']),indent=2))


if __name__=='__main__':main()
