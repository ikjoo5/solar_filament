"""Four predeclared actual DenseCRF refinements of our existing instance masks.

This applies the DenseCRF method, not an entire published solar-filament model.
Continuous CNN probabilities were not exported: unary terms use a fixed signed
distance prior around the frozen binary prediction. Missing raw pixels cause an
unchanged-mask fallback, never zero-filled appearance inference.
"""
from collections import Counter, defaultdict
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import cv2
import numpy as np
from pycocotools import mask as mu
from scipy import ndimage

from experiment_mask_postprocess import group_rows, overlap_stats, read_rows
from experiment_native_cross_model import matched_ids
from experiment_view_combinations import ROOT, GT, VAL, SIZE, encode_roi
from filament_eval import Filament, evaluate
from refine_masks_with_pixels import TilePixels, resolve_rois

SOURCE=Path('/workspace/artifacts/rich_boundary_composition_v1/tta_precision.csv')
PIXELS=Path('/workspace/artifacts/pixel_tiles_20261027T203313Z_fd420182')
OUT=Path('/workspace/artifacts/paper_densecrf_v1')


def plan():
    return [dict(name=f'gaussian{spatial:g}_bilateral_rgb{color:g}',spatial_compat=spatial,
        spatial_sigma=3.,bilateral_compat=3.,bilateral_spatial_sigma=10.,bilateral_color_sigma=color,
        iterations=5,unary_distance_temperature=1.5,hard_band=4,minimum_original_iou=.85,
        area_ratio=[.5,1.5],roi_padding=8)
        for spatial in (1.,3.) for color in (5.,10.)]


def geometry_guard(candidate,original):
    area=int(original.sum());new=int(candidate.sum())
    union=int((candidate|original).sum())
    return bool(area and new and .5<=new/area<=1.5 and int((candidate&original).sum())/union>=.85)


def densecrf_refine(gray,original,other=None,coverage=None,setting=None):
    """One instance ROI. Actual fully connected Gaussian/bilateral mean field."""
    setting=plan()[0] if setting is None else setting
    original=np.asarray(original,bool);gray=np.asarray(gray)
    if original.ndim!=2 or gray.shape!=original.shape or gray.dtype!=np.uint8 or not original.any():
        raise ValueError('Matching uint8 grayscale and nonempty binary mask required')
    if coverage is None or np.asarray(coverage,bool).shape!=original.shape:
        raise ValueError('Explicit pixel coverage is required')
    if not np.asarray(coverage,bool).all():return original.copy(),'fallback_missing_pixels'
    if other is None:other=np.zeros_like(original)
    other=np.asarray(other,bool)
    if other.shape!=original.shape or np.any(other&original):raise ValueError('Original instances must be disjoint')
    import pydensecrf.densecrf as dcrf
    inside=ndimage.distance_transform_edt(original)
    outside=ndimage.distance_transform_edt(~original)
    signed=inside-outside
    probability=1/(1+np.exp(-np.clip(signed/setting['unary_distance_temperature'],-20,20)))
    core=inside>=setting['hard_band']
    background=(outside>=setting['hard_band'])|other
    probability[core]=.999;probability[background]=.001
    unary=-np.log(np.stack([1-probability,probability]).clip(1e-6,1-1e-6))
    unary=np.ascontiguousarray(unary.reshape(2,-1),dtype=np.float32)
    height,width=original.shape
    model=dcrf.DenseCRF2D(width,height,2)
    model.setUnaryEnergy(unary)
    model.addPairwiseGaussian(sxy=setting['spatial_sigma'],compat=setting['spatial_compat'],
        kernel=dcrf.DIAG_KERNEL,normalization=dcrf.NORMALIZE_SYMMETRIC)
    rgb=np.ascontiguousarray(np.repeat(gray[:,:,None],3,axis=2))
    model.addPairwiseBilateral(sxy=setting['bilateral_spatial_sigma'],srgb=setting['bilateral_color_sigma'],
        rgbim=rgb,compat=setting['bilateral_compat'],kernel=dcrf.DIAG_KERNEL,normalization=dcrf.NORMALIZE_SYMMETRIC)
    posterior=np.asarray(model.inference(setting['iterations']))
    if posterior.shape!=(2,height*width) or not np.isfinite(posterior).all():raise RuntimeError('Nonfinite DenseCRF inference')
    candidate=(posterior.argmax(axis=0)==1).reshape(height,width)
    candidate[core]=True;candidate[background]=False
    if not geometry_guard(candidate,original):return original.copy(),'fallback_geometry_guard'
    return candidate,'changed' if np.any(candidate!=original) else 'unchanged'


def site_metrics(score):
    grouped=defaultdict(lambda:np.zeros(4,np.float64))
    for r in score['per_group']:
        grouped[r['image_id'][-2:]]+=np.asarray([r['matched_iou_sum'],r['tp_pairs'],r['fp'],r['fn']])
    return {site:dict(pq=float(a[0]/(a[1]+.5*a[2]+.5*a[3])),tp_pairs=int(a[1]),fp=int(a[2]),fn=int(a[3])) for site,a in sorted(grouped.items())}


def paired_delta(reference,score):
    by_image=defaultdict(lambda:np.zeros((2,4),np.float64))
    for index,source in enumerate((reference,score)):
        for r in source['per_group']:
            by_image[r['image_id']][index]+=np.asarray([r['matched_iou_sum'],r['tp_pairs'],r['fp'],r['fn']])
    a=np.stack(list(by_image.values()));rng=np.random.default_rng(20261028)
    sampled=a[rng.integers(0,len(a),(10000,len(a)))].sum(axis=1)
    q=sampled[:,:,0]/(sampled[:,:,1]+.5*sampled[:,:,2]+.5*sampled[:,:,3])
    return dict(paired_delta_percentile95=np.quantile(q[:,1]-q[:,0],[.025,.975]).tolist(),
        unit='Physical image with all annotator variants; reused-development conditional check, not independent validation')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args();out=args.output
    started=time.monotonic();cv2.setNumThreads(1)
    import pydensecrf.densecrf
    settings=plan();assert len(settings)==4
    out.mkdir(parents=True,exist_ok=False)
    (out/'predeclared_plan.json').write_text(json.dumps(settings,indent=2)+'\n')
    selection=list(csv.DictReader(VAL.open()));ids={r['image_id'] for r in selection}
    images={Path(r['file_name']).stem for r in selection}
    rows=read_rows(SOURCE);assert len(rows)==625 and not any(overlap_stats(rows).values())
    store=TilePixels(PIXELS);assert store.shape==(SIZE,SIZE)
    predictions={s['name']:[] for s in settings};statuses={s['name']:Counter() for s in settings}
    for number,(image,members) in enumerate(sorted(group_rows(rows).items()),1):
        gray,covered=store.load(image)
        rles=[r.rle(SIZE,SIZE) for r in members];all_original=mu.decode(mu.merge(rles)).astype(bool)
        regions=[];originals=[]
        for row in members:
            x,y,w,h=map(int,mu.toBbox(row.rle(SIZE,SIZE)))
            region=np.s_[max(0,y-8):min(SIZE,y+h+8),max(0,x-8):min(SIZE,x+w+8)]
            regions.append(region);originals.append(mu.decode(row.rle(SIZE,SIZE))[region].astype(bool))
        for setting in settings:
            candidates=[]
            for original,region in zip(originals,regions):
                candidate,status=densecrf_refine(gray[region],original,all_original[region]&~original,covered[region],setting)
                candidates.append(candidate);statuses[setting['name']][status]+=1
            resolved=resolve_rois(candidates,originals,regions,(SIZE,SIZE))
            for row,mask,original,region in zip(members,resolved,originals,regions):
                if not geometry_guard(mask,original):
                    mask=original.copy();statuses[setting['name']]['postownership_guard_fallback']+=1
                assert geometry_guard(mask,original)
                encoded=encode_roi(mask,region[0].start,region[1].start)
                predictions[setting['name']].append(Filament(row.filament_id,image,image,encoded['counts'].decode(),int(mu.area(encoded))))
        if number%15==0:print('ActualDenseCRF',number,'/91',flush=True)
    # No annotation was loaded above; all four outputs now exist.
    truth=read_rows(GT,gt=True,selected=ids);assert len(truth)==708 and len(ids)==101 and len(images)==91
    reference=evaluate(truth,rows,SIZE,SIZE);assert abs(reference['pq']-.47259304445299505)<1e-12
    before=matched_ids(truth,rows);reports=[];saved=[]
    for setting in settings:
        pred=predictions[setting['name']];score=evaluate(truth,pred,SIZE,SIZE);audit=overlap_stats(pred)
        assert len(pred)==625 and not any(audit.values()) and score['tp_pairs']==708-score['fn']
        matches=matched_ids(truth,pred)
        path=out/(setting['name']+'.csv')
        with path.open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['filament_id','segmentation_rle']);writer.writerows((r.filament_id,r.counts) for r in pred)
        reload=read_rows(path);check=evaluate(truth,reload,SIZE,SIZE);assert abs(check['pq']-score['pq'])<1e-12
        reports.append(dict(**setting,**{k:score[k] for k in ('pq','sq','rq','tp_pairs','fp','fn')},
            physical_masks=len(pred),statuses=dict(statuses[setting['name']]),**audit,extra_matching_pairs=0,
            recovered_gt=len(matches-before),lost_gt=len(before-matches),recovered_gt_ids=sorted(matches-before),lost_gt_ids=sorted(before-matches),
            delta_pq=score['pq']-reference['pq'],by_site=site_metrics(score),**paired_delta(reference,score)))
        saved.append(dict(name=setting['name'],path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),score_recomputed_after_save=True))
    best=max(reports,key=lambda r:r['pq']);preserving=[r for r in reports if not r['lost_gt']]
    report=dict(scope='Four actual DenseCRF postprocess applications to our fixed91-image pipeline; no new training or external evaluation',
        paper_method_references=['https://doi.org/10.1088/1538-3873/ac6e07','https://doi.org/10.1088/1674-4527/ae983a'],
        not_a_full_paper_model_replication=True,engine='pydensecrf DenseCRF2D fully connected Gaussian/bilateral mean-field',
        versions={name:importlib.metadata.version(name) for name in ('pydensecrf2','numpy','scipy')},
        unary_scope='Fixed signed-distance prior from binary masks; original continuous network probabilities not available',
        source_csv=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        pixel_manifest_sha256=hashlib.sha256((PIXELS/'manifest.json').read_bytes()).hexdigest(),
        validation=dict(physical_images=91,annotation_groups=101,gt_instances=708),
        baseline={k:reference[k] for k in ('pq','sq','rq','tp_pairs','fp','fn')},baseline_by_site=site_metrics(reference),
        experiments=reports,best=best,best_preserving=max(preserving,key=lambda r:r['pq']) if preserving else None,
        saved_candidates=saved,external_data_used=False,training_performed=False,adopted=False,
        adoption_note='Existing91 images have been repeatedly tuned. No candidate is deployed from a small same-development gain; independent external confirmation is still pending.',
        elapsed_seconds=time.monotonic()-started)
    destination=ROOT/'docs/evidence/paper_densecrf_experiments.json';destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(baseline=report['baseline'],best=best,best_preserving=report['best_preserving'],adopted=False,elapsed_seconds=report['elapsed_seconds']),indent=2))


if __name__=='__main__':main()
