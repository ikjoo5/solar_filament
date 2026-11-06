"""Fixed-split T4 inference experiment: cached DETR boxes route best84 tiles.

No fitting or test-set inference takes place here. The reviewed YOLO masks and
A100 DETR boxes are fixed prediction inputs; native and ROI masks are newly
computed under the same T4 FP16 runtime. GT is passed only to the scorer.
"""
from __future__ import annotations

import argparse
import fcntl
import gc
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import time
import zipfile
import zlib

import cv2
import numpy as np
from pycocotools import mask as mu
import torch

import detr_best84_tiles as native
import roi_filament_pipeline as roi
import submission_native_recipe as recipe
from submission_driver import checked_rle, validate_predictions
from native_instance_data import EXPECTED, load_cohort
from native_instance_training import (atomic_json, configure_cuda, device_for,
    digest, identity, runtime_versions)
from roi_filament_core import RoiFilamentNet, crop_roi, prepare_image, restore_mask


SEED_SHA = 'd7d1a513d16fd5dcffd2a6aabe65a9a452739c2d7468ed430436992ee28a481c'
NATIVE_SHA = 'ae6b209f72355fd52d2cb07182e69028e97c703a1977ad1e7b38b27cf8cebe69'
ROI_SHA = '4d1a97c0baf3b7d820754de141d68f1f56cffb0f5fb1b06be6f1b93a9a90ad7f'
SOURCE_FILES = ('detr_best84_pipeline.py', 'detr_best84_tiles.py',
    'submission_driver.py', 'submission_inference.py', 'submission_native_recipe.py',
    'submission_yolo_recipe.py', 'roi_filament_core.py', 'roi_filament_pipeline.py',
    'native_instance_data.py', 'native_instance_training.py', 'native_instance_model.py')
NATIVE_ARMS = ('full', 'detr70_c08', 'detr80_c06', 'detr80_c08')
ROI_ARMS = ('roi70_c08', 'roi80_c06', 'roi80_c08')
ROI_SPECS = dict(roi70_c08=('70', .8), roi80_c06=('80', .6), roi80_c08=('80', .8))
DEFAULTS = dict(device='cuda', native_precision='fp16', roi_precision='fp16',
    native_batch=1, roi_batch=1, strict=True, include_selected_only=False,
    roi_encoder='resnet34', roi_size=384, roi_context=1.5, preprocessing='limb',
    seed_sha256=SEED_SHA)


def normalize_config(config):
    cfg = dict(DEFAULTS, **config)
    for key in ('output_root', 'seed_path', 'images', 'annotations', 'train_split',
                'val_split', 'native_checkpoint', 'roi_checkpoint'):
        cfg[key] = str(Path(cfg[key]).resolve())
    out = Path(cfg['output_root'])
    for key in ('images', 'source_root', 'original_base_root'):
        if key in cfg:
            protected = Path(cfg[key]).resolve()
            if out == protected or out in protected.parents or protected in out.parents:
                raise ValueError('New output must be separate from read-only '+key)
    for key in ('native_checkpoint', 'roi_checkpoint', 'annotations', 'train_split', 'val_split'):
        if out in Path(cfg[key]).parents:
            raise ValueError('Input files must be outside the experiment output: '+key)
    if cfg['native_batch'] != 1 or cfg['roi_batch'] != 1:
        raise ValueError('The reviewed T4 experiment uses native and ROI batch one')
    if cfg['include_selected_only'] is not False:
        raise ValueError('This experiment selects only the three predeclared fallback arms')
    if any(cfg[k] not in ('fp16', 'fp32') for k in ('native_precision', 'roi_precision')):
        raise ValueError('T4 uses FP16; CPU fixtures must explicitly select FP32')
    if not isinstance(cfg['roi_size'], int) or cfg['roi_size'] < 32 or cfg['roi_size'] % 32:
        raise ValueError('ROI size must be a positive multiple of32')
    if cfg['strict'] and any(cfg[k] != DEFAULTS[k] for k in
            ('native_precision', 'roi_precision', 'roi_encoder', 'roi_size', 'roi_context', 'preprocessing')):
        raise ValueError('Production inference settings are fixed for this T4 comparison')
    if cfg['strict'] and cfg['seed_sha256'] != SEED_SHA:
        raise ValueError('Use the reviewed fixed91 prediction seed')
    return cfg


def _load_seed(cfg):
    path = Path(cfg['seed_path'])
    if digest(path) != cfg['seed_sha256']:
        raise ValueError('Prediction seed checksum mismatch')
    seed = json.loads(gzip.decompress(path.read_bytes()))
    if seed.get('schema_version') != 1 or seed.get('inference_contains_GT') is not False:
        raise ValueError('Expected the prediction-only fixed-split seed schema')
    frames = seed.get('frames')
    if not isinstance(frames, list) or not frames:
        raise ValueError('Prediction seed has no physical images')
    keyed = {}
    for row in frames:
        if set(row) != {'file_name', 'image_sha256', 'yolo_reference', 'weak', 'boxes'}:
            raise ValueError('Unexpected data in prediction-only seed frame')
        name = row['file_name']
        if Path(name).name != name or name in keyed:
            raise ValueError('Duplicate or unsafe prediction seed filename')
        if set(row['boxes']) != {'70', '80'}:
            raise ValueError('Both reviewed DETR checkpoints must be represented')
        checked = dict(row, yolo_reference=validate_predictions(row['yolo_reference']),
            weak=[checked_rle(x) for x in row['weak']],
            boxes={epoch:roi.checked_boxes(rows) for epoch,rows in row['boxes'].items()})
        if any(x['score'] < .05 for rows in checked['boxes'].values() for x in rows):
            raise ValueError('Seed bbox minimum confidence changed')
        keyed[name] = checked
    if len(keyed) != seed.get('physical_images'):
        raise ValueError('Prediction seed photo count mismatch')
    for name, expected in seed.get('frozen_recipe_sha256', {}).items():
        if name not in ('submission_native_recipe.py', 'submission_yolo_recipe.py'):
            raise ValueError('Unexpected seed recipe file')
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError('Frozen YOLO/native recipe changed: '+name)
    return seed, keyed


def prepare_context(config):
    cfg = normalize_config(config)
    device = device_for(cfg['device'])
    if device.type != 'cuda' and any(cfg[k] != 'fp32' for k in ('native_precision', 'roi_precision')):
        raise ValueError('CPU fixtures require explicit FP32 for both segmenters')
    if cfg['strict'] and device.type != 'cuda':
        raise ValueError('Select the Colab T4 GPU runtime')
    cohort = load_cohort(cfg)
    seed, frames = _load_seed(cfg)
    if seed['metadata'] != cohort['input_sha256']:
        raise ValueError('Prediction seed and annotation/split files differ')
    names = cohort['validation_names']
    if set(names) != set(frames) or len({Path(n).stem for n in names}) != len(names):
        raise ValueError('Prediction seed must cover exactly the entire fixed validation cohort')
    if {Path(n).stem for n in names} != set(cohort['mapping'].values()):
        raise ValueError('Validation photo/annotator mapping is incomplete')
    if cfg['strict'] and (len(names), len(cohort['truth']), sum(map(len,cohort['truth'].values()))) != (91,101,708):
        raise ValueError('Expected fixed91 photos /101 rater groups /708 GT')
    checkpoints = {key:digest(cfg[key+'_checkpoint']) for key in ('native','roi')}
    if cfg['strict'] and checkpoints != {'native':NATIVE_SHA, 'roi':ROI_SHA}:
        raise ValueError('Use the reviewed best84 and frozen ROI30 weights')
    if checkpoints != {'native':seed['source_checkpoints']['native_best84'],
                       'roi':seed['source_checkpoints']['roi30']}:
        raise ValueError('Weights do not match the seed provenance')
    stamps = {}
    for name in names:
        stamps[name] = digest(Path(cfg['images'])/name)
        if stamps[name] != frames[name]['image_sha256']:
            raise ValueError('Image pixels differ from cached detections: '+name)
    protocol = dict(version=1, scope='Fixed-split inference only; no training or test-set inference',
        prediction_seed_sha256=cfg['seed_sha256'], seed_scope=seed['scope'],
        source_checkpoints=seed['source_checkpoints'], metadata=cohort['input_sha256'],
        images=stamps, current_checkpoints=checkpoints,
        native_arms=dict(full='full native4view control', **native.ARM_SPECS),
        roi_arms=ROI_SPECS,
        inference={k:cfg[k] for k in ('native_precision','roi_precision','native_batch','roi_batch',
            'roi_encoder','roi_size','roi_context','preprocessing','include_selected_only')},
        native_protocol=dict(tile=1024,overlap=.25,views=['base','hflip','vflip','hvflip'],
            margin=1.5,missing_pixels='same-view full-native fallback before TTA averaging',
            shared_full_tile_forwards_per_image=36, measured_compute_reduction=False),
        roi_decoding=dict(mask_threshold=.5,min_area=300,minimum_quality=.6,nms_iou=.5),
        sources={name:digest(Path(__file__).with_name(name)) for name in SOURCE_FILES},
        runtime=runtime_versions(),device=str(device),
        gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        cuda_environment={key:os.environ.get(key) for key in
            ('TORCH_CUDNN_V8_API_DISABLED','CUBLAS_WORKSPACE_CONFIG')},
        public_LB_reference=.39, historical_development_reference=[.472593,.471930],
        comparative_control='full native recomputed on the same T4; cached YOLO and A100 DETR shared by all arms',
        validation_used_repeatedly=True, GT_enters_prediction=False)
    # JSON round-trips tuples as lists; compare the same canonical structure on
    # restart rather than rejecting an otherwise identical saved protocol.
    protocol = roi.json_ready(protocol)
    run_id = identity(protocol)
    root = Path(cfg['output_root']);root.mkdir(parents=True,exist_ok=True)
    path = root/'run_identity.json'
    if path.exists():
        old = json.loads(path.read_text())
        if old != dict(identity=run_id,protocol=protocol):
            raise ValueError('Source, pixels, runtime or settings changed; preserve this run and use a new output')
    elif any((root/name).exists() for name in ('records','result.json','latest_results.json')):
        raise ValueError('Existing records have no matching run identity; refusing to mix runs')
    atomic_json(path,dict(identity=run_id,protocol=protocol))
    atomic_json(root/'resolved_config.json',cfg)
    return dict(config=cfg,cohort=cohort,frames=frames,seed=seed,identity=run_id,
                protocol=protocol,device=device,stamps=stamps)


def _load_native(cfg, device):
    import segmentation_models_pytorch as smp
    saved = torch.load(cfg['native_checkpoint'],map_location='cpu',weights_only=True)
    config = saved['config']
    if cfg['strict'] and (saved['epoch'] != 84 or any(config.get(k) != value for k,value in
            [('encoder','tu-convnext_tiny'),('crop',1024),('overlap',.25),('preprocessing','limb_only')])):
        raise ValueError('Native checkpoint architecture or epoch is not the reviewed best84')
    model = smp.Unet(encoder_name=config['encoder'],encoder_weights=None,in_channels=3,classes=2,activation=None)
    model.load_state_dict(saved['model_state_dict']);del saved
    return model.requires_grad_(False).to(device).eval()


def _load_roi(cfg, device):
    saved = torch.load(cfg['roi_checkpoint'],map_location='cpu',weights_only=True)
    if cfg['strict'] and (saved.get('completed_epoch') != 30 or saved.get('config',{}).get('roi_encoder') != 'resnet34'):
        raise ValueError('ROI checkpoint must be the frozen ResNet34 epoch30')
    model = RoiFilamentNet(encoder=cfg['roi_encoder'],encoder_weights=None)
    model.load_state_dict(saved['model']);del saved
    return model.requires_grad_(False).to(device).eval()


def native_from_seed(frame, rich):
    """Exact frozen native stages, with the verified ordered weak-mask pool.

    The seed producer verifies this adapter against build_native_submission
    on all91 original cached rich records. No confidence or GT is invented.
    """
    if rich.get('file_name') != frame['file_name'] or rich.get('size') != [2048,2048]:
        raise ValueError('Native rich record has the wrong image coordinates')
    originals = [checked_rle(x) for x in frame['yolo_reference']]
    weak = [checked_rle(x) for x in frame['weak']]
    initial, added = recipe._add_native_reference(originals,rich,weak)
    refined, decisions = recipe._refine_boundaries(initial,rich)
    final, finishing = recipe._final_admission_pruning(refined,rich)
    return dict(rles=validate_predictions(final), decisions=dict(native_reference_added=added,**decisions,**finishing))


def infer_roi_variants(model, image_uint8, boxes_by_epoch, cfg, device):
    corrected = prepare_image(image_uint8,cfg['preprocessing'])
    candidates = {}
    with torch.inference_mode():
        for epoch,floor in [('70',.8),('80',.6)]:
            rows=[]
            for proposal in roi.checked_boxes(boxes_by_epoch[epoch]):
                if proposal['score'] < floor:continue
                inputs,_,transform=crop_roi(corrected,proposal['box'],context=cfg['roi_context'],size=cfg['roi_size'])
                tensor=torch.from_numpy(np.ascontiguousarray(inputs)).float()[None].to(device)
                with torch.autocast(device_type=device.type,dtype=torch.float16,
                                    enabled=device.type=='cuda' and cfg['roi_precision']=='fp16'):
                    probability=model(tensor).sigmoid()[0,0].float().cpu().numpy()
                if not np.isfinite(probability).all():raise FloatingPointError('Nonfinite ROI probabilities')
                foreground=probability>=.5
                quality=float(probability[foreground].mean()) if foreground.any() else 0.
                rle=restore_mask(probability,transform,threshold=.5)
                if rle is not None and int(mu.area(rle)):
                    rows.append(dict(rle=rle,box=proposal['box'],box_id=identity(proposal['box']),
                        box_score=proposal['score'],roi_quality=quality,selection_score=proposal['score']*quality))
            candidates[epoch]=rows
    final={arm:validate_predictions(roi.mask_predictions(candidates[epoch],confidence,300,.5,.6))
           for arm,(epoch,confidence) in ROI_SPECS.items()}
    return dict(candidates=candidates,predictions=final,
                export_floor={'70':.8,'80':.6},ROI_training_epochs_added=0)


def _read_pixels(context, name):
    path=Path(context['config']['images'])/name;raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=context['stamps'][name]:
        raise ValueError('Image changed during inference: '+name)
    pixels=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_GRAYSCALE)
    if pixels is None or pixels.shape!=(2048,2048) or pixels.dtype!=np.uint8:
        raise ValueError('Original uint8 2048-square image is required: '+name)
    return pixels


def _validate_record(data, context, name, stage):
    if not isinstance(data,dict):raise ValueError('Malformed cache record')
    if (data.get('identity'),data.get('file_name'),data.get('stage'),data.get('image_sha256')) != (
            context['identity'],name,stage,context['stamps'][name]):
        raise RuntimeError('Valid cache belongs to different image or protocol; refusing to mix predictions')
    expected=NATIVE_ARMS if stage=='native' else ROI_ARMS
    if set(data.get('predictions',{}))!=set(expected):raise ValueError('Incomplete prediction arms')
    for masks in data['predictions'].values():validate_predictions(masks)
    if stage=='native' and set(data.get('rich',{}))!=set(NATIVE_ARMS):
        raise ValueError('Incomplete native rich records')
    if stage=='roi' and set(data.get('candidates',{}))!={'70','80'}:
        raise ValueError('Incomplete ROI candidate records')
    return data


def read_stage(context,name,stage):
    path=Path(context['config']['output_root'])/'records'/stage/(Path(name).stem+'.json.gz')
    if not path.is_file():return None
    try:
        return _validate_record(roi.read_record(path),context,name,stage)
    except (ValueError,KeyError,TypeError,EOFError,OSError,zlib.error):
        print('Recomputing damaged '+stage+' record: '+name,flush=True)
        return None


def _run_stage(context, stage, wanted):
    cfg=context['config'];device=context['device'];model=None
    try:
        for index,name in enumerate(wanted,1):
            if read_stage(context,name,stage) is not None:
                print(f'{stage} cache {index}/{len(wanted)} {name}',flush=True);continue
            if model is None:
                print('Loading '+stage+' segmenter; other GPU model is unloaded',flush=True)
                if digest(cfg[stage+'_checkpoint'])!=context['protocol']['current_checkpoints'][stage]:
                    raise ValueError('Checkpoint changed before '+stage+' loading')
                model=(_load_native if stage=='native' else _load_roi)(cfg,device)
                if digest(cfg[stage+'_checkpoint'])!=context['protocol']['current_checkpoints'][stage]:
                    raise ValueError('Checkpoint changed during '+stage+' loading')
            image=_read_pixels(context,name);frame=context['frames'][name];started=time.monotonic()
            record=dict(identity=context['identity'],stage=stage,file_name=name,image_sha256=context['stamps'][name])
            if stage=='native':
                inference_cfg=dict(native_precision=cfg['native_precision'],native_batch=cfg['native_batch'],
                    include_selected_only=False,file_name=name,image_sha256=context['stamps'][name],identity=context['identity'],
                    progress=lambda view,n,total:print(f'  native {view} tiles {n}/{total}',flush=True))
                result=native.infer_native_variants(model,image,frame['boxes'],inference_cfg,device)
                if set(result['rich'])!=set(NATIVE_ARMS):raise ValueError('Wrong native arm set')
                final={arm:native_from_seed(frame,rich) for arm,rich in result['rich'].items()}
                record.update(rich=result['rich'],diagnostics=result['diagnostics'],
                    predictions={arm:value['rles'] for arm,value in final.items()},
                    postprocess_decisions={arm:value['decisions'] for arm,value in final.items()})
            else:
                inference_cfg={key:cfg[key] for key in ('preprocessing','roi_context','roi_size','roi_precision')}
                record.update(infer_roi_variants(model,image,frame['boxes'],inference_cfg,device))
            record['seconds']=time.monotonic()-started
            _validate_record(record,context,name,stage)
            path=Path(cfg['output_root'])/'records'/stage/(Path(name).stem+'.json.gz')
            roi.write_record(path,record)
            print(f'{stage} completed {index}/{len(wanted)} {name} ({record["seconds"]:.1f}s)',flush=True)
    finally:
        del model;gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()


def _matched_GT(truth,mapping,predictions):
    result=set()
    for sid,masks in truth.items():
        pred=predictions[mapping[sid]]
        hits=np.asarray(mu.iou(masks,pred,[0]*len(pred)))>.5 if masks and pred else np.zeros((len(masks),len(pred)),bool)
        result.update((sid,index) for index in np.flatnonzero(hits.any(axis=1)))
    return result


def paired_photo_bootstrap(reference, candidate, photo_names, samples=5000):
    """Descriptive paired PQ interval; alternative raters stay in one photo.

    This is not an independent confirmation and does not correct the repeated
    checkpoint/arm selection on this development cohort.
    """
    stems=[Path(name).stem for name in photo_names]
    positions={stem:index for index,stem in enumerate(stems)}
    if len(positions)!=len(stems) or not stems:raise ValueError('Unique physical photos required')
    columns=('tp_pairs','fp','fn','matched_iou_sum')
    arrays=[]
    for report in (reference,candidate):
        array=np.zeros((len(stems),len(columns)),np.float64)
        for row in report['per_group']:
            array[positions[row['image']]] += [row[key] for key in columns]
        arrays.append(array)
    weights=np.random.default_rng(20261102).multinomial(len(stems),np.full(len(stems),1/len(stems)),size=samples)
    scores=[]
    for array in arrays:
        sums=weights@array;den=sums[:,0]+.5*(sums[:,1]+sums[:,2])
        scores.append(np.divide(sums[:,3],den,out=np.zeros(samples),where=den>0))
    differences=scores[1]-scores[0]
    return dict(delta_pq=float(candidate['score']['pq']-reference['score']['pq']),
        ci95=np.quantile(differences,[.025,.975]).tolist(),samples=samples,
        physical_photo_clusters=len(stems),seed=20261102,
        scope='Descriptive paired physical-photo bootstrap on repeatedly used development images; checkpoint/arm selection not corrected')


def _score(context, records):
    cohort=context['cohort'];names=cohort['validation_names'];root=Path(context['config']['output_root'])
    predictions={arm:{Path(name).stem:[checked_rle(r) for r in records[stage][name]['predictions'][arm]] for name in names}
        for stage,arms in [('native',NATIVE_ARMS),('roi',ROI_ARMS)] for arm in arms}
    baseline=roi.score_groups(cohort['truth'],cohort['mapping'],predictions['full'])
    old_matches=_matched_GT(cohort['truth'],cohort['mapping'],predictions['full'])
    metrics={}
    destination=root/'predictions';destination.mkdir(parents=True,exist_ok=True)
    for arm,prediction in predictions.items():
        score=roi.score_groups(cohort['truth'],cohort['mapping'],prediction)
        matches=_matched_GT(cohort['truth'],cohort['mapping'],prediction)
        csv_path=destination/(arm+'.csv');roi.write_predictions(csv_path,prediction)
        metrics[arm]=dict(**score,csv=str(csv_path),csv_sha256=digest(csv_path),
            recovered_GT=len(matches-old_matches),lost_GT=len(old_matches-matches),
            delta_vs_full={key:score['score'][key]-baseline['score'][key] for key in ('pq','sq','rq','fp','fn','tp_pairs')},
            comparison_scope='Same cached YOLO + new native full/fallback routing' if arm in NATIVE_ARMS else
                'DETR + frozen ROI only; different total pipeline, not an isolated decoder comparison')
        if arm in native.ARM_SPECS:
            metrics[arm]['paired_photo_bootstrap_delta_pq']=paired_photo_bootstrap(baseline,score,names)
    best=max(NATIVE_ARMS,key=lambda arm:metrics[arm]['score']['pq'])
    gain=metrics[best]['score']['pq']-baseline['score']['pq']
    candidate=None
    if best!='full' and gain>0:
        spec=native.ARM_SPECS[best]
        candidate=dict(arm=best,development_pq=metrics[best]['score']['pq'],paired_development_gain=gain,
            determinant='Highest observed PQ among three predeclared routing arms; no independent generalization claim',
            detector_epoch=int(spec['epoch']),detector_confidence=spec['confidence'],
            detector_checkpoint_sha256=context['seed']['source_checkpoints']['detr'][spec['epoch']]['available_checkpoint_sha256'],
            native_checkpoint_sha256=context['protocol']['current_checkpoints']['native'],
            native_protocol=context['protocol']['native_protocol'],
            detector_source='Validation boxes reused from A100 BF16 body + FP32 bbox head; fresh test boxes required',
            seed_reusable_on_test=False,needs_fresh_test_inference=True,kaggle_submitted=False,
            public_LB_reference=.39,public_LB_for_candidate=None,
            paired_photo_bootstrap_delta_pq=metrics[best]['paired_photo_bootstrap_delta_pq'],
            frozen_postprocess_sha256=context['seed']['frozen_recipe_sha256'])
    return dict(metrics=metrics,best_development_native_arm=best,submission_candidate=candidate,
        candidate_status='development_gain_requires_leaderboard_check' if candidate else 'no_native_routing_gain_observed')


def export_results(context):
    """Bounded results only. Replace owned ZIPs only after all new parts verify."""
    root=Path(context['config']['output_root']);destination=root/'exports';destination.mkdir(exist_ok=True)
    common=[p for p in root.iterdir() if p.is_file() and p.suffix=='.json' and p.name!='latest_results.json']
    files=[]
    for folder in ('records','predictions','code'):
        for path in (root/folder).rglob('*') if (root/folder).exists() else []:
            if path.is_file() and path.suffix in ('.gz','.csv','.json','.py','.md') and '.pending' not in path.name:
                files.append((path,str(path.relative_to(root))))
    seed_path=Path(context['config']['seed_path']);files.append((seed_path,'inputs/prediction_seed.json.gz'))
    common=[(path,str(path.relative_to(root))) for path in common]
    entries={name:dict(bytes=path.stat().st_size,sha256=digest(path)) for path,name in common+files}
    if len(entries)!=len(common)+len(files):raise ValueError('Duplicate export paths')
    export_id=identity(entries)
    overhead=sum(p.stat().st_size+2*len(n)+512 for p,n in common)+2*len(json.dumps(entries))+65536
    budget=26*1024*1024
    if overhead>budget//2:raise ValueError('Metadata exceeds the upload budget')
    parts=[];part=[];size=overhead
    for path,name in sorted(files):
        amount=path.stat().st_size+2*len(name)+512
        if amount+overhead>budget:raise ValueError('One result exceeds upload budget: '+name)
        if part and size+amount>budget:parts.append(part);part=[];size=overhead
        part.append((path,name));size+=amount
    parts.append(part);pending_files=[];archives=[]
    try:
        for index,part in enumerate(parts,1):
            path=destination/f'detr_best84_{export_id[:12]}_part{index:02d}.zip';pending=path.with_suffix('.pending')
            pending_files.append((pending,path))
            manifest=dict(owner='detr_best84_pipeline',run_root=str(root),identity=context['identity'],
                export_id=export_id,part=index,parts_total=len(parts),entries=entries)
            with zipfile.ZipFile(pending,'w',zipfile.ZIP_DEFLATED) as z:
                for source,name in sorted(common+part):z.write(source,name)
                z.writestr('export_manifest.json',json.dumps(manifest,indent=2))
            if pending.stat().st_size>28*1024*1024:raise ValueError('ZIP exceeds28MiB')
            with zipfile.ZipFile(pending) as z:
                for name in z.namelist():
                    if name=='export_manifest.json':continue
                    raw=z.read(name)
                    if len(raw)!=entries[name]['bytes'] or hashlib.sha256(raw).hexdigest()!=entries[name]['sha256']:
                        raise ValueError('Result changed during export: '+name)
        for pending,path in pending_files:pending.replace(path);archives.append(str(path))
        atomic_json(destination/'current_exports.json',dict(owner='detr_best84_pipeline',run_root=str(root),archives=archives))
    finally:
        for pending,_ in pending_files:pending.unlink(missing_ok=True)
    for old in destination.glob('detr_best84_*_part*.zip'):
        if str(old) in archives or old.is_symlink():continue
        try:
            with zipfile.ZipFile(old) as z:m=json.loads(z.read('export_manifest.json'))
            owned=(m.get('owner')=='detr_best84_pipeline' and m.get('run_root')==str(root)
                   and identity(m['entries'])==m.get('export_id')
                   and old.name==f'detr_best84_{m["export_id"][:12]}_part{m["part"]:02d}.zip')
        except (ValueError,KeyError,TypeError,OSError,zipfile.BadZipFile):continue
        if owned:old.unlink()
    return archives


def run(config,max_images=None):
    cfg=normalize_config(config);root=Path(cfg['output_root']);root.mkdir(parents=True,exist_ok=True)
    lockpath=Path('/tmp')/('detr_best84_'+identity(str(root))[:16]+'.lock')
    with lockpath.open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another cell is running this experiment')
        configure_cuda();context=prepare_context(cfg)
        names=context['cohort']['validation_names']
        if max_images is not None and (type(max_images) is not int or not 1<=max_images<=len(names)):
            raise ValueError('max_images must be an absolute photo count within the validation cohort')
        wanted=names if max_images is None else names[:max_images]
        print(f'Inference only: {len(wanted)}/{len(names)} photos, no training. Public LB reference0.39.',flush=True)
        _run_stage(context,'native',wanted)
        _run_stage(context,'roi',wanted)
        if digest(cfg['seed_path'])!=cfg['seed_sha256']:
            raise ValueError('Prediction seed changed during inference')
        for stage in ('native','roi'):
            if digest(cfg[stage+'_checkpoint'])!=context['protocol']['current_checkpoints'][stage]:
                raise ValueError('Checkpoint changed during experiment: '+stage)
        records={stage:{name:data for name in names if (data:=read_stage(context,name,stage)) is not None}
                 for stage in ('native','roi')}
        complete=all(len(rows)==len(names) for rows in records.values())
        report=dict(status='complete' if complete else 'paused',identity=context['identity'],
            validation_physical_images=len(names),completed_images={stage:len(rows) for stage,rows in records.items()},
            full_cohort_score_available=complete,training_epochs_added=0,
            public_LB_reference=.39,public_LB_for_new_arms=None,kaggle_submitted=False,
            scope='Repeated fixed development split. All cached detections are validation-only, not test predictions.',
            baseline='Same-run T4 full native + shared frozen YOLO predictions',
            detector='Reviewed A10070/80 boxes shared unchanged across comparisons',
            candidate_threshold_search=False,notes=['Do not compare partial-image PQ to the full91 control.',
                'No automatic Kaggle submission. A successful development candidate needs fresh test inference and LB confirmation.',
                'Full/native-fallback branches share36 native forwards per image; no runtime reduction is claimed.'])
        if complete:report.update(_score(context,records))
        else:report.update(metrics={},submission_candidate=None,candidate_status='incomplete_no_selection')
        atomic_json(root/'result.json',report)
        archives=export_results(context)
        latest=dict(report,archives=archives);atomic_json(root/'latest_results.json',latest)
        print(json.dumps({k:latest[k] for k in ('status','completed_images','full_cohort_score_available','candidate_status','archives')},ensure_ascii=False,indent=2),flush=True)
        return latest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--max-images',type=int)
    args=parser.parse_args();run(json.loads(Path(args.config).read_text()),args.max_images)


if __name__=='__main__':main()
