"""Independently audit the uploaded context-only result as data, never code."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

from experiment_mask_postprocess import overlap_stats, read_rows
from filament_eval import Filament, _check_compressed_counts, evaluate
from verify_native_training_results import GT, ROOT, VAL, hits_by_gt

DATA=Path('/workspace/artifacts/context_only_uploaded_20261029')
OLD=Path('/workspace/artifacts/a100_results_20261028_bedf3e21/best_validation.csv')
OLD_RICH=Path('/workspace/artifacts/best84_rich_predictions_tta/predictions.jsonl')
FIELDS=('pq','sq','rq','tp_pairs','fp','fn','matched_iou_sum')


def metrics(score):
    return {key:score[key] for key in FIELDS if key in score}


def rows_from(records,mode,threshold,area=300):
    rows=[]
    for record in records:
        source=record if mode=='base' else record['tta']
        image=Path(record['file_name']).stem
        for i,component in enumerate(source['components'][f'{threshold:g}'],1):
            if component['area']>=area:
                rows.append(Filament(f'{image}_{i}',image,image,component['rle']['counts'],component['area']))
    return rows


def physical_arrays(score,images):
    grouped=defaultdict(lambda:np.zeros(4,np.float64))
    for row in score['per_group']:
        grouped[row['image_id']]+=np.array([row['matched_iou_sum'],row['tp_pairs'],row['fp'],row['fn']])
    return np.stack([grouped[image] for image in images])


def pq(values):
    return values[...,0]/(values[...,1]+.5*values[...,2]+.5*values[...,3])


def main():
    comparison=json.loads((DATA/'context_only_comparison.json').read_text())
    cfg=json.loads((DATA/'multiscale_context/config.json').read_text())
    history=json.loads((DATA/'multiscale_context/history.json').read_text())
    best=json.loads((DATA/'multiscale_context/best_epoch.json').read_text())
    progress=json.loads((DATA/'context_only_progress.json').read_text())
    split=list(csv.DictReader(VAL.open()))
    groups={row['image_id'] for row in split};names={row['file_name'] for row in split}
    images=sorted(Path(name).stem for name in names)
    truth=read_rows(GT,gt=True,selected=groups)
    assert len(names)==91 and len(groups)==101 and len(truth)==708
    assert cfg['source_sha256']==cfg['expected_source_sha256']
    local_metadata={
        'annotations':Path('/workspace/datasets/solar_filament/MAGFiLO_1.0_Annotations_kaggle2026_train.json'),
        'train_split':Path('/workspace/attachments/1cca1607-f94f-4334-9586-c1fb5ab292d7/train_split.csv'),
        'val_split':VAL,
        'synthetic_annotations':Path('/workspace/attachments/c2dd3ebc-ec85-4b4d-9502-0e40e85d418f/synthetic_annotations.json'),
        'synthetic_split':Path('/workspace/attachments/9213b745-cf92-429e-a13b-cedeb7c9720d/split_info.json'),
    }
    actual_source_hashes={key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in local_metadata.items()}
    assert actual_source_hashes==cfg['source_sha256']
    assert cfg['amp_dtype']=='torch.float16' and cfg['precision_override']=='float16'
    assert cfg['synthetic_source_pool']['donor_validation_overlap']==0
    assert cfg['execution_methods']==['multiscale_context']
    assert [row['epoch'] for row in history]==list(range(1,61))
    validations=[row for row in history if 'validation' in row]
    assert [row['epoch'] for row in validations]==[1]+list(range(5,61,5))
    selected=max(validations,key=lambda row:row['validation']['pq'])
    assert selected['epoch']==best['epoch']==comparison['best_context_epoch']==45
    assert selected['validation']==best['best']==comparison['multiscale_context']
    assert history[-1]['epoch']==comparison['epochs_completed']==60
    assert comparison['status']=='paused' and cfg['epochs']==100
    assert progress['runs']['multiscale_context']['best']==best['best']
    pred=read_rows(DATA/'multiscale_context/best_validation.csv')
    assert {row.image_id for row in pred}<=set(images)
    score=evaluate(truth,pred,2048,2048)
    for key in ('pq','sq','rq','tp_pairs','fp','fn'):
        assert abs(score[key]-best['best'][key])<1e-12,key
    audit=overlap_stats(pred)
    assert not any(audit.values()) and score['tp_pairs']==708-score['fn']

    old_records={r['file_name']:r for r in map(json.loads,OLD_RICH.read_text().splitlines())}
    records=list(map(json.loads,(DATA/'multiscale_context/raw_paper_views.jsonl').read_text().splitlines()))
    assert len(records)==91 and {r['file_name'] for r in records}==names
    assert len({r['file_name'] for r in records})==91
    declared_checkpoint={r['checkpoint_sha256'] for r in records}
    assert len(declared_checkpoint)==1 and len(next(iter(declared_checkpoint)))==64
    checked_rles=0
    for record in records:
        assert record['size']==[2048,2048]
        assert record['paper_method']=='multiscale_context'
        assert record['image_sha256']==old_records[record['file_name']]['image_sha256']
        assert record['tta']['checkpoint_sha256']==record['checkpoint_sha256']
        assert set(record['view_unions'])=={'base','hflip','vflip','hvflip'}
        for source in (record,record['tta']):
            assert set(source['components'])=={'0.35','0.5','0.65'}
            assert source['file_name']==record['file_name'] and source['size']==[2048,2048]
            union=source['unions']['foreground']['0.2']
            assert union['size']==[2048,2048]
            _check_compressed_counts(union['counts'],2048**2);checked_rles+=1
            for threshold,components in source['components'].items():
                encoded=[]
                for component in components:
                    encoded.append(component['rle']);rle=component['rle']
                    assert rle['size']==[2048,2048]
                    assert _check_compressed_counts(rle['counts'],2048**2)==component['area']==int(mu.area(rle))
                    assert component['area']>=100
                    assert int(mu.area(mu.merge([rle,union],intersect=True)))==component['area']
                    stats=component['stats']['foreground']
                    assert all(np.isfinite(x) and 0<=x<=1 for x in stats.values())
                    assert stats['mean']+1e-6>=float(threshold)
                    assert stats['mean']<=stats['max']+1e-6 and stats['p90']<=stats['max']+1e-6
                    checked_rles+=1
                if encoded:
                    assert int(mu.area(mu.merge(encoded)))==sum(int(mu.area(rle)) for rle in encoded)
        for rle in record['view_unions'].values():
            assert rle['size']==[2048,2048]
            _check_compressed_counts(rle['counts'],2048**2);checked_rles+=1
    base_rows=rows_from(records,'base',.5)
    assert Counter((r.image_id,r.counts) for r in base_rows)==Counter((r.image_id,r.counts) for r in pred)
    grid=[]
    for mode in ('base','tta'):
        for threshold in (.35,.5,.65):
            rows=rows_from(records,mode,threshold)
            s=evaluate(truth,rows,2048,2048)
            grid.append(dict(mode=mode,threshold=threshold,min_area=300,**metrics(s),physical_masks=len(rows)))

    old_rows=read_rows(OLD);old=evaluate(truth,old_rows,2048,2048)
    before,after=hits_by_gt(truth,old_rows),hits_by_gt(truth,pred)
    changes=Counter('retained' if before[k]>.5 and after[k]>.5 else 'new_only' if after[k]>.5 else
                    'old_only' if before[k]>.5 else 'both_miss' for k in before)
    baseline=comparison['baseline']['validation']
    reference_arrays=physical_arrays(old,images);new_arrays=physical_arrays(score,images)
    indices=np.random.default_rng(20261029).integers(0,len(images),(10000,len(images)))
    delta=pq(new_arrays[indices].sum(axis=1))-pq(reference_arrays[indices].sum(axis=1))
    site=[]
    for camera in sorted({image[-2:] for image in images}):
        selected=[i for i,image in enumerate(images) if image.endswith(camera)]
        a,b=reference_arrays[selected].sum(axis=0),new_arrays[selected].sum(axis=0)
        site.append(dict(observatory=camera,physical_images=len(selected),historical_baseline_pq=float(pq(a)),
                         context_pq=float(pq(b)),delta_pq=float(pq(b)-pq(a))))
    report=dict(scope='Independent data-only recomputation on reused fixed development set; no uploaded code executed',
        validation=dict(physical_images=91,annotation_groups=101,gt_instances=708),
        csv_score_reproduced=True,native_best=metrics(score),physical_masks=len(pred),**audit,
        history=dict(epochs_completed=60,status='paused',schedule_total_epochs=100,best_epoch=45,
            validation_evaluations=len(validations),first10_mean_train_loss=float(np.mean([r['train_loss'] for r in history[:10]])),
            last10_mean_train_loss=float(np.mean([r['train_loss'] for r in history[-10:]])),
            last_validation=history[-1]['validation'],train_elapsed_hours=history[-1]['elapsed_seconds']/3600,
            real_only_final_epochs_not_reached=True,
            validation_curve=[dict(epoch=r['epoch'],train_loss=r['train_loss'],**r['validation']) for r in validations],
            interpretation='Late validation fluctuates without sustained collapse. Declining training loss and repeated checkpoint selection do not prove generalization.'),
        training=dict(precision=cfg['amp_dtype'],gpu=cfg['gpu'],real_train_images=616,synthetic_images=462,
            donor_validation_overlap=0,batch=1,accumulate=4,samples_per_epoch=400,optimizer_updates_per_epoch=100),
        same_device_baseline_reported=baseline,
        delta_vs_same_device_reported={k:score[k]-baseline[k] for k in baseline if k in score},
        historical_baseline_recomputed=metrics(old),delta_vs_historical={k:score[k]-old[k] for k in FIELDS},
        changed_GT_vs_historical=dict(changes),grid_area300=grid,observatory_vs_historical=site,
        conditional_bootstrap_vs_historical=dict(draws=10000,resampling_unit='Physical image with all annotators together',
            delta_pq=score['pq']-old['pq'],percentile95=[float(x) for x in np.quantile(delta,[.025,.975])],
            fraction_positive=float((delta>0).mean()),
            limitation='Selected epoch on reused dev; excludes epoch/model selection bias and is not external validation.'),
        integrity=dict(expected_names_match=True,image_sha_matches_prior_export=True,checked_rles=checked_rles,
            single_declared_checkpoint_sha256=next(iter(declared_checkpoint)),
            base_rich_masks_exactly_equal_selected_csv=True,source_metadata_hashes_match_pinned=True,
            actual_local_metadata_hashes=actual_source_hashes,
            archive_has_no_checkpoint=True,actual_images_and_probability_maps_not_available=True),
        limitations=['Same-device baseline masks are absent, so its PQ cannot be independently recomputed or paired-bootstrap compared.',
            'Metadata image/checkpoint hashes are internally consistent but actual JPEG/weight bytes are absent.',
            'Delta includes extra training and adapter architecture together; no continued-training control.',
            'Single-model native PQ must not be compared directly to full ensemble PQ .472593 or public LB .39.'])
    report['source_sha256']={str(p.relative_to(DATA)):hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA.rglob('*') if p.is_file()}
    path=ROOT/'docs/evidence/context_uploaded_audit.json';path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('history','source_sha256')},indent=2))


if __name__=='__main__':main()
