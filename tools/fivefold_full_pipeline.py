"""Fold-local inference and evaluation of the frozen YOLO+native recipe.

Neither training checkpoint comes from best84 or the old YOLO solar model.
Prediction CSVs exist before any annotation masks are decoded for scoring.
"""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path

import cv2
from pycocotools import mask as mu

from grouped_cv_training import atomic_json,digest,metric,configure_cuda
from submission_inference import InferenceProvider
from submission_yolo_recipe import build_yolo_reference
from submission_native_recipe import build_native_submission
from evaluate_fivefold_study import metadata_fold
from evaluate_grouped_oof import aggregate,checked_rle,truth_for

METHODS=('full_selected','without_final_admission','clahe_native_base','limb_only_yolo_preprocessing')


def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def encoded(values):
    return [dict(size=list(v['size']),counts=v['counts'].decode() if isinstance(v['counts'],bytes) else v['counts']) for v in values]


def compose(stem,geo,prep,tiles,rich,variants=True):
    reference=build_yolo_reference(stem,geo,prep,tiles)
    result=build_native_submission(stem,reference,geo,prep,tiles,rich,return_stages=True)
    masks=dict(full_selected=encoded(result['final']))
    if not variants:return masks
    masks['without_final_admission']=encoded(result['rich_refined'])
    if rich.get('preprocess_views',{}).get('clahe'):
        replaced=dict(rich,**rich['preprocess_views']['clahe'])
        masks['clahe_native_base']=encoded(build_native_submission(stem,reference,geo,prep,tiles,replaced))
    ablated=[dict(row,instances=[]) if row['view']!='limb' else row for row in prep]
    reference2=build_yolo_reference(stem,geo,ablated,tiles)
    masks['limb_only_yolo_preprocessing']=encoded(build_native_submission(stem,reference2,geo,ablated,tiles,rich))
    return masks


def write_csv(path,records,method):
    pending=path.with_name(path.name+'.pending')
    with pending.open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['filament_id','segmentation_rle'])
        writer.writerows((f"{Path(r['file_name']).stem}_{index}",value['counts'])
            for r in sorted(records,key=lambda r:r['file_name']) for index,value in enumerate(r['masks'][method],1))
    pending.replace(path)


def selected_native(root,stage,fold):
    progress=json.loads((root/f'stage{stage:03d}_progress.json').read_text())
    return next(row for row in progress['summaries'] if row['fold']==fold)


def local_artifact(root,value,fold):
    path=Path(value)
    if path.is_file():return path
    marker=f'fold{fold}'
    if marker not in path.parts:raise ValueError('Artifact has no matching fold path')
    result=root/Path(*path.parts[path.parts.index(marker):])
    if not result.resolve().is_relative_to(root.resolve()):raise ValueError('Artifact path escapes study root')
    return result


def audit_yolo(root,native,stage,fold,check_weights=True):
    meta=json.loads((root/f'fold{fold}/yolo/stage{stage:03d}.json').read_text())
    cfg=json.loads((root/f'fold{fold}/yolo/config.json').read_text())
    if (cfg['identity']!=fingerprint(cfg['settings']) or cfg['identity']!=meta['identity']
            or cfg['settings']['fold']!=fold or meta['fold']!=fold or meta['stage_epoch']!=stage
            or cfg['settings']['architecture']!='yolo26s-seg.pt'):
        raise ValueError('Wrong YOLO fold/stage/initialization identity')
    for key in ('annotations','fold_plan','train_split','val_split','base_runtime'):
        if cfg['settings']['sources'][key]!=native['training_config']['source_sha256'][key]:
            raise ValueError('YOLO and native training sources differ: '+key)
    path=Path(meta['checkpoint'])
    if check_weights and digest(path)!=meta['checkpoint_sha256']:raise ValueError('YOLO stage weights changed')
    return meta


def run_export(config,_provider=None):
    configure_cuda()
    root=Path(config['study_root']);stage=int(config['stage_epoch']);fold=int(config['fold'])
    native=selected_native(root,stage,fold)
    # The fold's YOLO metadata binds its weights to the very same train/val CSVs.
    yolo_meta=audit_yolo(root,native,stage,fold)
    model_path=Path(yolo_meta['checkpoint'])
    if digest(model_path)!=yolo_meta['checkpoint_sha256']:raise ValueError('YOLO stage checkpoint changed')
    native_records=Path(native['export']['rich_predictions'])
    if digest(native_records)!=native['rich_predictions_sha256']:raise ValueError('Native rich records changed')
    rich={row['file_name']:row for row in map(json.loads,native_records.read_text().splitlines())}
    if set(rich)!=set(native['image_names']):raise ValueError('Native image coverage mismatch')
    folder=root/f'fold{fold}'/f'stage{stage:03d}'/'full_pipeline';folder.mkdir(exist_ok=True)
    source={name:digest(Path(__file__).with_name(name)) for name in
            ('fivefold_full_pipeline.py','submission_inference.py','submission_yolo_recipe.py','submission_native_recipe.py')}
    identity=dict(study_identity=native['study_identity'],fold=fold,stage_epoch=stage,
        yolo_sha256=yolo_meta['checkpoint_sha256'],native_sha256=native['model_sha256'],
        native_records_sha256=native['rich_predictions_sha256'],sources=source,
        baseline='selected_production_recipe_validation0.472593_publicLB0.39')
    signature=fingerprint(identity)
    if (folder/'identity.json').exists() and json.loads((folder/'identity.json').read_text())!=dict(signature=signature,payload=identity):
        folder=folder/'variants'/signature[:12];folder.mkdir(parents=True,exist_ok=True)
    records_dir=folder/'records';records_dir.mkdir(exist_ok=True)
    atomic_json(folder/'identity.json',dict(signature=signature,payload=identity))
    provider=_provider or InferenceProvider(model_path,None,device=config.get('device','cuda'),verify_weights=False)
    raw_cache=root/f'fold{fold}'/f'stage{stage:03d}'/'yolo_records';raw_cache.mkdir(exist_ok=True)
    records=[]
    for index,name in enumerate(sorted(rich),1):
        image_path=Path(config['images'])/name;image_sha=digest(image_path)
        path=records_dir/(Path(name).stem+'.json')
        if path.exists():
            record=json.loads(path.read_text());stored=record.pop('record_sha256')
            if fingerprint(record)!=stored or record['identity']!=signature or record['image_sha256']!=image_sha:
                raise ValueError('Full-pipeline cache changed; preserved: '+name)
            record['record_sha256']=stored
        else:
            if rich[name]['image_sha256']!=image_sha:raise ValueError('Native/YOLO input bytes differ')
            pixels=cv2.imread(str(image_path),cv2.IMREAD_GRAYSCALE)
            if pixels is None or pixels.shape!=(2048,2048):raise ValueError('Original2048image required')
            stem=Path(name).stem
            raw_path=raw_cache/(stem+'.json.gz')
            raw_identity=dict(image_sha256=image_sha,yolo_sha256=yolo_meta['checkpoint_sha256'],provider_sha256=source['submission_inference.py'])
            if raw_path.is_file():
                raw=json.loads(gzip.decompress(raw_path.read_bytes()));checksum=raw.pop('checksum')
                if fingerprint(raw)!=checksum or raw['identity']!=raw_identity:raise ValueError('YOLO raw cache changed')
                geo,prep,tiles=raw['geometric'],raw['preprocessing'],raw['tiled']
            else:
                if config.get('cached_only'):
                    raise FileNotFoundError('Postprocessing-only mode requires the saved YOLO raw cache: '+str(raw_path))
                geo=provider.geometric(stem,pixels);prep=provider.preprocessing(stem,pixels);tiles=provider.tiled(stem,pixels)
                raw=dict(identity=raw_identity,geometric=geo,preprocessing=prep,tiled=tiles)
                raw['checksum']=fingerprint(raw)
                pending=raw_path.with_name(raw_path.name+'.pending')
                pending.write_bytes(gzip.compress(json.dumps(raw).encode(),mtime=0));pending.replace(raw_path)
            record=dict(file_name=name,image_sha256=image_sha,identity=signature,fold=fold,stage_epoch=stage,
                raw_yolo_cache=str(raw_path.relative_to(root)),raw_yolo_cache_sha256=digest(raw_path),
                masks=compose(stem,geo,prep,tiles,rich[name]))
            record['record_sha256']=fingerprint(record);atomic_json(path,record)
        records.append(record);print(f'Full pipeline fold{fold} [{index}/{len(rich)}] {name}',flush=True)
    combined=folder/'full_records.jsonl'
    pending=combined.with_name(combined.name+'.pending');pending.write_text(''.join(json.dumps(r)+'\n' for r in records));pending.replace(combined)
    available=set(METHODS).intersection(*(set(r['masks']) for r in records))
    for method in sorted(available):write_csv(folder/(method+'.csv'),records,method)
    summary=dict(status='complete',fold=fold,stage_epoch=stage,study_identity=native['study_identity'],
        identity=signature,payload=identity,yolo_training_identity=yolo_meta['identity'],image_names=sorted(rich),
        records_sha256=digest(combined),methods=sorted(available),
        csv_sha256={method:digest(folder/(method+'.csv')) for method in sorted(available)},
        native_export_directory=str(native_records.parent),source_sha256=source,ground_truth_used=False,output_directory=str(folder))
    atomic_json(folder/'summary.json',summary)
    return summary


def evaluate(annotations,fold_plan,study_root,stage,report):
    root=Path(study_root);plan=json.loads(Path(fold_plan).read_text());study=json.loads((root/'study_config.json').read_text())
    progress=json.loads((root/f'stage{stage:03d}_progress.json').read_text())
    natives={r['fold']:r for r in progress['summaries']}
    selected={};regimes=[];composition_sources=None
    for fold in range(5):
        if fold not in natives:continue
        canonical=root/f'fold{fold}'/f'stage{stage:03d}'
        selected_path=canonical/'full_export_result.json'
        folder=canonical/'full_pipeline'
        if selected_path.is_file():
            chosen=json.loads(selected_path.read_text())
            suffix=Path(chosen['output_directory']).parts
            marker=f'fold{fold}';relative=Path(*suffix[suffix.index(marker):])
            folder=root/relative
        if not (folder/'summary.json').is_file():continue
        summary=json.loads((folder/'summary.json').read_text())
        native=natives[fold]
        yolo=audit_yolo(root,native,stage,fold,check_weights=False)
        # Metadata audit reads no GT polygons.
        native_rows,_,regime=metadata_fold(local_artifact(root,native['export']['rich_predictions'],fold).parent,
            study,plan,fold,stage,digest(annotations),digest(fold_plan))
        if regimes and regime!=regimes[0]:raise ValueError('Mixed inference settings across folds')
        regimes.append(regime)
        if summary['fold']!=fold or summary['stage_epoch']!=stage or summary['study_identity']!=study['study_identity']:
            raise ValueError('Wrong full-pipeline stage')
        if fingerprint(summary['payload'])!=summary['identity']:raise ValueError('Full identity changed')
        if composition_sources is not None and summary['payload']['sources']!=composition_sources:
            raise ValueError('Mixed postprocessing implementations across folds')
        composition_sources=summary['payload']['sources']
        if summary['payload']['yolo_sha256']!=yolo['checkpoint_sha256'] or summary['payload']['native_sha256']!=native['model_sha256']:
            raise ValueError('Full model hashes differ from the fold weights')
        if summary['payload']['native_records_sha256']!=native['rich_predictions_sha256']:
            continue  # A different current preprocessing export has not been composed yet.
        records=[json.loads(line) for line in (folder/'full_records.jsonl').read_text().splitlines()]
        if digest(folder/'full_records.jsonl')!=summary['records_sha256'] or len(records)!=len(native['image_names']) or {r['file_name'] for r in records}!=set(native['image_names']):
            raise ValueError('Full pipeline image coverage/checksum mismatch')
        for method in summary['methods']:
            if digest(folder/(method+'.csv'))!=summary['csv_sha256'][method]:raise ValueError('Full CSV changed')
        native_hashes={r['file_name']:r['image_sha256'] for r in native_rows}
        for record in records:
            unsigned={k:v for k,v in record.items() if k!='record_sha256'}
            if (record['record_sha256']!=fingerprint(unsigned) or record['identity']!=summary['identity']
                    or record['fold']!=fold or record['stage_epoch']!=stage
                    or record['image_sha256']!=native_hashes[record['file_name']]):
                raise ValueError('Full per-image provenance changed')
            for values in record['masks'].values():
                masks=[checked_rle(v) for v in values]
                if masks and int(mu.area(mu.merge(masks)))!=sum(int(mu.area(v)) for v in masks):raise ValueError('Overlapping full-pipeline masks')
        selected[fold]=records
    # Every method's CSV already exists before any GT is decoded below.
    data=json.loads(Path(annotations).read_text());results={}
    complete=len(selected)==5
    observations={Path(r['file_name']).stem:r for r in plan['observations']}
    available=set(METHODS).intersection(*(set(r['masks']) for records in selected.values() for r in records)) if selected else set()
    for method in sorted(available):
        all_rows=[];per_fold={}
        for fold,records in selected.items():
            truth,mapping=truth_for(data,{r['file_name'] for r in records})
            predictions={Path(r['file_name']).stem:[checked_rle(v) for v in r['masks'][method]] for r in records}
            measured=metric(truth,predictions,mapping);all_rows.extend(measured['per_group']);per_fold[str(fold)]=aggregate(measured['per_group'])
        pooled=aggregate(all_rows) if all_rows else None
        sites={observations[r['image']]['site'] for r in all_rows}
        def period(image):
            year=int(observations[image]['capture_day'][:4])
            return next((f'{lo}-{hi}' for lo,hi in ((2011,2014),(2015,2018),(2019,2022)) if lo<=year<=hi),'other')
        periods={period(r['image']) for r in all_rows}
        results[method]=dict(pooled_oof=pooled if complete else None,pooled_completed_folds=pooled,by_fold=per_fold,
            by_site={site:aggregate([r for r in all_rows if observations[r['image']]['site']==site]) for site in sorted(sites)},
            by_period={p:aggregate([r for r in all_rows if period(r['image'])==p]) for p in sorted(periods)})
    result=dict(status='complete_full_pipeline' if complete else 'partial_full_pipeline',complete_fivefold=complete,
        stage_epoch=stage,completed_folds=sorted(selected),physical_images=sum(len(r) for r in selected.values()),
        pipeline='Frozen YOLO+native production recipe; fold-local newlytrainedweights',
        comparison_scope='Inference preprocessing and postprocessing only; same two trained models in every method',
        historical_reference=dict(validation_pq=.47259304445299505,user_reported_public_lb=.39),
        results=results,selection_performed=False,external_data_used=False,
        deltas_from_selected={name:{key:r['pooled_completed_folds'][key]-results['full_selected']['pooled_completed_folds'][key]
            for key in ('pq','tp_pairs','fp','fn')} for name,r in results.items() if name!='full_selected'},
        interpretation='Internal same-budget OOF comparison, not a guaranteed reproduction of historical score or a new LB score.')
    atomic_json(report,result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--result',required=True)
    parser.add_argument('--cached-only',action='store_true')
    args=parser.parse_args();config=json.loads(Path(args.config).read_text())
    if args.cached_only:config.update(cached_only=True,device='cpu')
    result=run_export(config);atomic_json(args.result,result)
