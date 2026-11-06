"""Fixed616/91 YOLO synthetic-data ablation, completed-epoch resume and PQ.

Only two declared continuation arms are trained. Native best84 stays frozen.
The production schedule really ends at30; no hidden100-epoch schedule exists.
"""
from collections import Counter
from copy import deepcopy
import csv
import gc
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import zipfile

import cv2
import numpy as np
from pycocotools import mask as mu
import torch

ULTRALYTICS_VERSION = '8.4.182'
ARMS = ('control_v2', 'targeted_v3')
ORIGINAL_SHA = '81c2a27d95821b5ede283169c0da8c08d360ecc4c2a60a1b9c6baf96d76bcccc'
NATIVE_SHA = 'ae6b209f72355fd52d2cb07182e69028e97c703a1977ad1e7b38b27cf8cebe69'
INPUT_SHA = dict(annotations='5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1',
    train_split='aa0733424815731c5c108ada91e45525eb17ca3a42b377bf19e3089d86aede7e',
    val_split='54c6a9aac5a3b70b3ecc6721109a8b0fe8ec020e90cb56e3848fccc92eadc61f')
SOURCE_FILES = ('targeted_synthetic_training.py','fixed_split_mask_quality.py',
    'evaluate_grouped_oof.py','grouped_cv_training.py','submission_inference.py',
    'submission_driver.py','submission_yolo_recipe.py','submission_native_recipe.py')

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_name(path.name+'.pending')
    pending.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    pending.replace(path)

def atomic_torch(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_name(path.name+'.pending')
    torch.save(value,pending);pending.replace(path)

def runtime_versions():
    versions={name:importlib.metadata.version(name) for name in ('torch','torchvision','ultralytics','numpy','pycocotools','scipy')}
    for name in ('segmentation-models-pytorch','timm'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    versions.update(opencv=cv2.__version__,torch_cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version())
    return versions

def source_hashes():
    return {name:digest(Path(__file__).with_name(name)) for name in SOURCE_FILES}

def capture_rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda'] and torch.cuda.is_available():torch.cuda.set_rng_state_all(state['cuda'])

def cpu_copy(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_copy(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_copy(v) for v in value)
    return deepcopy(value)

def reseed_transforms(obj,seed,seen=None):
    """Reset optional Albumentations RNG too; workers are deliberately zero."""
    seen=set() if seen is None else seen
    if obj is None or id(obj) in seen:return
    seen.add(id(obj))
    setter=getattr(obj,'set_random_seed',None)
    if callable(setter):setter(seed)
    children=getattr(obj,'transforms',None)
    if isinstance(children,(tuple,list)):
        for child in children:reseed_transforms(child,seed,seen)
    elif children is not None:reseed_transforms(children,seed,seen)
    child=getattr(obj,'transform',None)
    if child is not None:reseed_transforms(child,seed,seen)

def seed_epoch(trainer,seed):
    value=int(seed)+10007*int(trainer.epoch)
    random.seed(value);np.random.seed(value%(2**32));torch.manual_seed(value)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(value)
    loader=trainer.train_loader
    if loader.num_workers!=0:raise ValueError('Resume protocol requires workers=0')
    if loader.generator is not None:loader.generator.manual_seed(value)
    sampler=getattr(loader,'sampler',None)
    if getattr(sampler,'generator',None) is not None:sampler.generator.manual_seed(value)
    reseed_transforms(loader.dataset,value)
    loader.reset()
    trainer._synthetic_completed_batches=0
    trainer._synthetic_step_attempts=0
    trainer._synthetic_steps_completed=0

def compact_checkpoint(payload):
    result=dict(payload)
    for key in ('optimizer','scaler','live_state','ema_full_state','scheduler_state','rng_state'):
        result[key]=None
    return result

def training_options(cfg,dataset_yaml,arm):
    # batch1/nbs1 means every batch has a completed optimizer update. This avoids
    # the library's cross-invocation accumulation offset and lost partial grads.
    return dict(data=str(dataset_yaml),epochs=int(cfg.get('epochs',30)),imgsz=int(cfg.get('imgsz',2048)),
        batch=1,nbs=1,workers=0,device=cfg.get('device',0),cache=False,
        project=str(Path(cfg['cache_root'])/'targeted_training'),name=arm,exist_ok=True,
        optimizer='AdamW',lr0=float(cfg.get('lr',1e-4)),lrf=.1,cos_lr=True,warmup_epochs=0.,
        amp=bool(cfg.get('amp',True)),seed=int(cfg.get('seed',42)),deterministic=True,
        channels_last=False,patience=1000,fliplr=.5,flipud=.5,degrees=0.,translate=0.,scale=.1,
        shear=0.,perspective=0.,hsv_h=0.,hsv_s=0.,hsv_v=.1,mosaic=0.,close_mosaic=0,
        mixup=0.,copy_paste=0.,mask_ratio=4,overlap_mask=True,max_det=100,
        save=True,save_period=-1,val=False,plots=False,pretrained=True)

def train_arm(cfg,arm,dataset_yaml,dataset_identity,run_until=None):
    """Fine-tune one fixed-split arm, reusing only its own completed checkpoint."""
    if arm not in ARMS:raise ValueError('Unknown synthetic comparison arm')
    if int(cfg.get('batch',1))!=1 or int(cfg.get('nbs',1))!=1:
        raise ValueError('This resumable comparison uses batch=1 and nbs=1, without accumulation')
    from grouped_cv_training import configure_cuda
    configure_cuda()
    torch.backends.cudnn.enabled=bool(cfg.get('cudnn_enabled',True))
    epochs=int(cfg.get('epochs',30));cap=epochs if run_until is None else int(run_until)
    if not 1<=cap<=epochs:raise ValueError('run_until must be between1 and the fixed total epochs')
    if digest(cfg['yolo'])!=ORIGINAL_SHA:raise ValueError('Use the reviewed original fixed-split YOLO best.pt')
    from ultralytics import YOLO,__version__
    from ultralytics.models.yolo.segment.train import SegmentationTrainer
    if __version__!=ULTRALYTICS_VERSION:raise ValueError('Use ultralytics=='+ULTRALYTICS_VERSION)
    options=training_options(cfg,dataset_yaml,arm)
    fixed=dict(arm=arm,dataset_identity=dataset_identity,options=options,
        initial_checkpoint_sha256=ORIGINAL_SHA,sources=source_hashes(),runtime=runtime_versions(),
        cudnn_enabled=torch.backends.cudnn.enabled,safe_depthwise_cuda=True,
        validation_selects_checkpoint=False,scope='original fixed616/91; no fold-model training',
        resume='completed epoch; full precision live/EMA/optimizer; deterministic per-epoch data seeds',
        bitwise_GPU_trajectory_guaranteed=False)
    run_id=fingerprint(fixed)
    folder=Path(cfg['output_root'])/'arms'/arm;folder.mkdir(parents=True,exist_ok=True)
    config_path=folder/'training_config.json'
    if config_path.exists() and json.loads(config_path.read_text())['identity']!=run_id:
        raise ValueError('Training code/data/settings/runtime changed; use a new output_root')
    atomic_json(config_path,dict(identity=run_id,settings=fixed))
    last=folder/'last_full.pt';stage=folder/f'epoch{cap:03d}.pt';saved=None
    if last.exists():
        saved=torch.load(last,map_location='cpu',weights_only=False)
        if saved.get('synthetic_identity')!=run_id:raise ValueError('Wrong arm or settings in resume checkpoint')
        atomic_json(folder/'history.json',saved['history'])
        if saved['completed_epoch']>cap and not stage.exists():raise ValueError('Cannot recreate an earlier missing stage')
    if stage.exists():
        done=torch.load(stage,map_location='cpu',weights_only=False)
        if done.get('synthetic_identity')!=run_id or done.get('completed_epoch')!=cap:raise ValueError('Wrong saved stage')
        return dict(arm=arm,epochs_completed=cap,fixed_final_epoch=epochs,checkpoint=str(stage),
                    checkpoint_sha256=digest(stage),identity=run_id,status='complete' if cap==epochs else 'paused')
    if saved is not None and saved['completed_epoch']==cap:
        atomic_torch(stage,compact_checkpoint(saved))
        return train_arm(cfg,arm,dataset_yaml,dataset_identity,cap)

    class FixedTrainer(SegmentationTrainer):
        def validate(self):return {},0.
        def final_eval(self):pass
        def _build_train_pipeline(self):
            if getattr(self,'_fixed_pipeline_built',False):
                raise RuntimeError('CUDA memory retry stopped; no automatic batch/data changes. Resume last completed epoch after fixing runtime.')
            self._fixed_pipeline_built=True
            return super()._build_train_pipeline()
        def _handle_nan_recovery(self,epoch):
            if self.loss is not None and not torch.isfinite(self.loss).all():
                raise FloatingPointError('Nonfinite loss; completed checkpoint retained, no silent recovery')
            return False
        def optimizer_step(self):
            before=float(self.scaler.get_scale())
            super().optimizer_step()
            self._synthetic_step_attempts+=1
            self._synthetic_steps_completed+=int(float(self.scaler.get_scale())>=before)

    model=YOLO(str(last if saved is not None else cfg['yolo']),task='segment')
    if saved is None and len(model.names)!=1:raise ValueError('Expected the original single-class solar detector')
    def restore(trainer):
        if saved is not None:
            trainer.model.load_state_dict(saved['live_state'])
            trainer.ema.ema.load_state_dict(saved['ema_full_state'])
            trainer.ema.updates=saved['updates']
            trainer.optimizer.load_state_dict(saved['optimizer'])
            trainer.scaler.load_state_dict(saved['scaler'])
            trainer.scheduler.load_state_dict(saved['scheduler_state'])
            restore_rng(saved['rng_state'])
        if trainer.accumulate!=1 or trainer.batch_size!=1:
            raise ValueError('Fixed comparison requires batch1 without accumulation')
    def save(trainer):
        if getattr(trainer,'_synthetic_completed_batches',0)!=len(trainer.train_loader):
            raise RuntimeError('Incomplete epoch is not a resumable completed checkpoint')
        payload=torch.load(trainer.last,map_location='cpu',weights_only=False)
        completed=int(trainer.epoch)+1
        history=list(saved.get('history',[])) if saved is not None else []
        history_path=folder/'history.json'
        if history_path.exists():history=json.loads(history_path.read_text())
        losses=getattr(trainer,'tloss',{})
        losses={k:float(v.detach().cpu()) for k,v in losses.items()} if isinstance(losses,dict) else {}
        history=[row for row in history if row['epoch']<completed]
        history.append(dict(epoch=completed,train_losses=losses,learning_rates=[g['lr'] for g in trainer.optimizer.param_groups],
            optimizer_updates=trainer._synthetic_steps_completed,optimizer_step_attempts=trainer._synthetic_step_attempts,
            amp_skipped_updates=trainer._synthetic_step_attempts-trainer._synthetic_steps_completed,
            training_records=len(trainer.train_loader.dataset)))
        payload.update(synthetic_identity=run_id,completed_epoch=completed,fixed_final_epoch=epochs,
            live_state=cpu_copy(trainer.model.state_dict()),ema_full_state=cpu_copy(trainer.ema.ema.state_dict()),
            optimizer=cpu_copy(trainer.optimizer.state_dict()),scaler=cpu_copy(trainer.scaler.state_dict()),
            scheduler_state=trainer.scheduler.state_dict(),rng_state=capture_rng(),history=history)
        atomic_torch(last,payload)
        atomic_json(history_path,history)
        atomic_json(folder/'progress.json',dict(epoch=completed,fixed_final_epoch=epochs,identity=run_id))
        if completed==cap:atomic_torch(stage,compact_checkpoint(payload))
    model.add_callback('on_train_start',restore)
    model.add_callback('on_train_epoch_start',lambda trainer:seed_epoch(trainer,cfg.get('seed',42)))
    model.add_callback('on_train_batch_end',lambda trainer:setattr(trainer,'_synthetic_completed_batches',trainer._synthetic_completed_batches+1))
    model.add_callback('on_model_save',save)
    model.add_callback('on_fit_epoch_end',lambda trainer:setattr(trainer,'stop',trainer.epoch+1>=cap))
    print(f'{arm}: 완료 epoch {saved["completed_epoch"] if saved else 0}에서 {cap}까지; 전체 목표 {epochs}. 배치1, 누적없음.',flush=True)
    model.train(trainer=FixedTrainer,**options,resume=str(last) if saved is not None else False)
    if not stage.exists():raise RuntimeError('Training exited before the requested completed epoch')
    del model;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    result=dict(arm=arm,epochs_completed=cap,fixed_final_epoch=epochs,checkpoint=str(stage),
                checkpoint_sha256=digest(stage),identity=run_id,status='complete' if cap==epochs else 'paused')
    atomic_json(folder/'result.json',result)
    return result

def write_csv(path,predictions):
    path=Path(path);pending=path.with_name(path.name+'.pending')
    with pending.open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['filament_id','segmentation_rle'])
        for stem,masks in predictions.items():
            for index,value in enumerate(masks):
                counts=value['counts'].decode('ascii') if isinstance(value['counts'],bytes) else value['counts']
                writer.writerow([f'{stem}_{index}',counts])
    pending.replace(path)

def raw_items(records):
    unique={}
    for record in records:
        for row in record['instances']:
            rle=dict(size=[2048,2048],counts=row['rle'].encode('ascii'))
            if int(mu.area(rle))==0:continue
            if row['rle'] not in unique or row['confidence']>unique[row['rle']]['confidence']:
                unique[row['rle']]=dict(rle=row['rle'],confidence=float(row['confidence']))
    return list(unique.values())

def matched_counts(truth,mapping,predictions,reference):
    counts=Counter();small=Counter()
    for group,gt in truth.items():
        pred=predictions[mapping[group]];old=reference[mapping[group]]
        hits=np.asarray(mu.iou(gt,pred,[0]*len(pred)))>.5 if gt and pred else np.zeros((len(gt),len(pred)),bool)
        before=np.asarray(mu.iou(gt,old,[0]*len(old)))>.5 if gt and old else np.zeros((len(gt),len(old)),bool)
        now=hits.any(axis=1);previous=before.any(axis=1)
        counts.update(GT=len(gt),predictions=len(pred),matched_GT=int(now.sum()),matched_predictions=int(hits.any(axis=0).sum()),
            recovered_GT=int((now&~previous).sum()),lost_GT=int((previous&~now).sum()))
        is_small=np.asarray(mu.area(gt))<670 if gt else np.zeros(0,bool)
        small.update(GT=int(is_small.sum()),matched_GT=int((is_small&now).sum()))
    return dict(counts,recall=counts['matched_GT']/counts['GT'],
        precision=counts['matched_predictions']/counts['predictions'] if counts['predictions'] else 0.,
        small_GT=dict(small,recall=small['matched_GT']/small['GT'] if small['GT'] else None,
                      definition='GT area<670 native pixels; train-derived fixed quartile, not validation fit'))

def evaluate_checkpoints(cfg,checkpoints):
    """Cache fixed91 inference, then score frozen rules; never fit on validation."""
    from fixed_split_mask_quality import read_splits,decode
    from evaluate_grouped_oof import truth_for,aggregate
    from grouped_cv_training import metric,configure_cuda
    from submission_inference import InferenceProvider
    from submission_driver import finalize_frame,write_envelope,read_envelope,valid_stage
    for name,expected in INPUT_SHA.items():
        if digest(cfg[name])!=expected:raise ValueError('Original fixed split input differs: '+name)
    if digest(cfg['yolo'])!=ORIGINAL_SHA:raise ValueError('Original YOLO checkpoint changed')
    configure_cuda()
    torch.backends.cudnn.enabled=bool(cfg.get('cudnn_enabled',True))
    native=cfg.get('native')
    full=bool(cfg.get('full_ensemble',True))
    if full and (not native or not Path(native).is_file()):
        raise FileNotFoundError('full_ensemble=True requires the original frozen best84 native checkpoint; set full_ensemble=False explicitly for YOLO-only development.')
    if full and digest(native)!=NATIVE_SHA:raise ValueError('Frozen best84 native checkpoint differs')
    data=json.loads(Path(cfg['annotations']).read_text());roles=read_splits(data,cfg['train_split'],cfg['val_split'])
    names=roles['validation'];base_folder=Path(cfg['output_root'])/'evaluation'
    # Bind completed reports as well as individual prediction caches to input pixels.
    image_hashes={name:digest(Path(cfg['images'])/name) for name in names}
    models={'baseline':str(cfg['yolo']),**{arm:str(checkpoints[arm]) for arm in ARMS}}
    completed=set()
    for arm in ARMS:
        checkpoint=torch.load(checkpoints[arm],map_location='cpu',weights_only=False)
        expected=json.loads((Path(cfg['output_root'])/'arms'/arm/'training_config.json').read_text())
        if checkpoint.get('synthetic_identity')!=expected['identity']:raise ValueError('Evaluation checkpoint belongs to another experiment')
        completed.add(int(checkpoint['completed_epoch']))
        del checkpoint
    if len(completed)!=1:raise ValueError('Compare control and targeted at the same completed epoch')
    common=dict(version=1,source=source_hashes(),runtime=runtime_versions(),inputs=INPUT_SHA,
        names=names,validation_image_sha256=image_hashes,
        decode=dict(confidence=.30,min_area=300,nms_strict_iou=.5),
        geometric=['base2048','scale2560','hflip2048','vflip2048'],ensemble=full,
        native_sha256=NATIVE_SHA if full else None,cudnn_enabled=torch.backends.cudnn.enabled,safe_depthwise_cuda=True,
        inference_gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        selection='fixed completed epoch; no threshold sweep')
    model_info={label:dict(path=path,sha256=digest(path)) for label,path in models.items()}
    evaluation_id=fingerprint(dict(protocol=common,models=model_info))
    folder=base_folder/f'epoch{next(iter(completed)):03d}_{evaluation_id[:12]}'
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'report.json').is_file():
        prior=json.loads((folder/'report.json').read_text())
        if prior.get('evaluation_identity')==evaluation_id and all((folder/name).is_file() and digest(folder/name)==expected for name,expected in prior.get('prediction_csv_sha256',{}).items()) and prior.get('prediction_csv_sha256'):
            print('동일 모델·설정의 완료된 검증 결과를 재사용합니다.',flush=True)
            return prior
    def pixels_at(path):
        pixels=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
        if pixels is None or pixels.dtype!=np.uint8 or pixels.shape!=(2048,2048):
            raise ValueError('Original uint8 grayscale2048 image required: '+str(path))
        return pixels
    # Compute native once; no YOLO module lives on GPU in this phase.
    native_records={}
    device='cuda:0' if str(cfg.get('device',0)) in ('0','cuda','cuda:0') else str(cfg['device'])
    if full:
        native_id=fingerprint(dict(common=common,kind='fixed_native84'))
        provider=InferenceProvider(None,native,device=device,native_batch=1,verify_weights=True)
        for number,name in enumerate(names,1):
            stem=Path(name).stem;path=Path(cfg['images'])/name;image_sha=image_hashes[name]
            dest=base_folder/'fixed_native'/native_id[:12]/f'{stem}.json.gz';dest.parent.mkdir(parents=True,exist_ok=True)
            row=read_envelope(dest,native_id,image_sha)
            if not valid_stage('native',row,stem):
                pixels=pixels_at(path)
                row=provider.native(stem,pixels,file_name=name,image_sha256=image_sha,identity=native_id)
                if not valid_stage('native',row,stem):raise ValueError('Native inference returned an invalid record')
                write_envelope(dest,row,native_id,image_sha)
            native_records[stem]=row
            if number%10==0 or number==len(names):print(f'고정 native 검증 캐시 {number}/{len(names)}',flush=True)
        del provider;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    all_predictions={};identities={}
    for label,checkpoint in models.items():
        model_id=fingerprint(dict(common=common,checkpoint_sha256=digest(checkpoint),label=label))
        identities[label]=model_id
        provider=InferenceProvider(checkpoint,None,device=device,native_batch=1,verify_weights=False)
        output={'yolo_conf030':{}}
        if full:output['frozen_ensemble']={}
        for number,name in enumerate(names,1):
            stem=Path(name).stem;path=Path(cfg['images'])/name;image_sha=image_hashes[name];pixels=None;stages={}
            stage_names=['geometric','preprocessing','tiled'] if full else ['geometric']
            for stage_name in stage_names:
                dest=folder/label/'cache'/stage_name/f'{stem}.json.gz';dest.parent.mkdir(parents=True,exist_ok=True)
                stage_id=fingerprint(dict(model=model_id,stage=stage_name))
                row=read_envelope(dest,stage_id,image_sha)
                if not valid_stage(stage_name,row,stem):
                    if pixels is None:pixels=pixels_at(path)
                    row=getattr(provider,stage_name)(stem,pixels)
                    if not valid_stage(stage_name,row,stem):raise ValueError('YOLO inference returned an invalid stage')
                    write_envelope(dest,row,stage_id,image_sha)
                stages[stage_name]=row
            items=raw_items(stages['geometric'])
            output['yolo_conf030'][stem]=decode(items,[r['confidence'] for r in items],.30)
            if full:
                stages['native']=native_records[stem]
                output['frozen_ensemble'][stem]=finalize_frame(stem,stages)
            if number%10==0 or number==len(names):print(f'{label} 검증 추론 {number}/{len(names)}',flush=True)
        for method,pred in output.items():
            for masks in pred.values():
                if masks and int(mu.area(mu.merge(masks)))!=sum(int(mu.area(x)) for x in masks):raise ValueError('Final predictions overlap')
            write_csv(folder/(label+'_'+method+'.csv'),pred)
        all_predictions[label]=output
        del provider;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    # No fitting/threshold choice occurs: GT is used here only to report final CSVs.
    truth,mapping=truth_for(data,set(names));results={}
    if len(truth)!=101 or sum(map(len,truth.values()))!=708:raise ValueError('Expected original fixed91/708GT cohort')
    for label,output in all_predictions.items():
        results[label]={}
        for method,pred in output.items():
            measured=metric(truth,pred,mapping);score=aggregate(measured['per_group'])
            counts=matched_counts(truth,mapping,pred,all_predictions['baseline'][method])
            if score['fn']!=708-counts['matched_GT'] or score['fp']!=counts['predictions']-counts['matched_predictions']:
                raise AssertionError('Unique matching and official metric disagree')
            results[label][method]=dict(score=score,matches=counts,per_group=measured['per_group'])
            print(label,method,{k:score[k] for k in ('pq','fp','fn')},flush=True)
    report=dict(status='complete',scope='reused fixed91 development; no full707 OOF or public LB claim',
        protocol=common,identities=identities,checkpoints=model_info,evaluation_identity=evaluation_id,
        evaluation_root=str(folder),epochs_completed=next(iter(completed)),
        prediction_csv_sha256={path.name:digest(path) for path in folder.glob('*.csv')},
        results=results,ensemble_evaluated=full,ensemble_pending=not full,
        historical_reference=dict(yolo_conf030_pq=.4309597846752626,frozen_ensemble_pq=.47259304445299505,
            note='Historical references; current runtime baseline is recomputed, no forced bitwise GPU assertion'),
        validation_training=False,threshold_search=False,automatic_adoption=False,
        all_final_CSVs_written_before_metric_evaluation=True)
    atomic_json(folder/'report.json',report)
    atomic_json(base_folder/'latest_evaluation.json',dict(report=str(folder/'report.json'),identity=evaluation_id))
    return report

def export_results(cfg):
    """Return independently openable ZIP parts below28MiB (no checkpoints/images)."""
    root=Path(cfg['output_root']);output=root/'exports';output.mkdir(exist_ok=True)
    selected=[];common=[]
    for path in root.rglob('*'):
        if not path.is_file() or output in path.parents:continue
        relative=path.relative_to(root)
        if path.name=='latest_results.json':continue
        preview=path.name in ('synthetic_preview.png','synthetic_native_preview.png') and 'data' in relative.parts
        if (path.suffix in ('.pt','.png','.jpg','.jpeg','.npy','.npz') and not preview) or path.name.endswith('.pending'):continue
        # RLE caches are needed for offline evaluation; no original images/weights.
        if preview or path.suffix in ('.json','.csv','.yaml','.py','.gz','.md','.txt'):
            (selected if path.suffix=='.gz' else common).append((path,relative))
    entries={str(relative):dict(sha256=digest(path),bytes=path.stat().st_size) for path,relative in common+selected}
    export_id=fingerprint(entries);common_bytes=sum(path.stat().st_size+len(str(relative))*2+256 for path,relative in common)
    limit=26*1024*1024
    if common_bytes>limit//2:raise ValueError('Summary metadata exceeds export budget; keep the experiment folder specific to this comparison')
    parts=[];current=[];size=common_bytes+len(json.dumps(entries))*2+65536
    base_size=size
    for path,relative in sorted(selected):
        amount=path.stat().st_size+len(str(relative))*2+512
        if base_size+amount>limit:raise ValueError('One cache file exceeds multipart budget: '+str(relative))
        if current and size+amount>limit:parts.append(current);current=[];size=base_size
        current.append((path,relative));size+=amount
    parts.append(current);paths=[]
    for index,payload in enumerate(parts,1):
        destination=output/f'targeted_synthetic_{export_id[:12]}_part{index:02d}.zip';pending=destination.with_suffix('.pending')
        manifest=dict(export_id=export_id,part=index,parts_total=len(parts),entries=entries,
            common_files=[str(r) for _,r in common],part_files=[str(r) for _,r in payload])
        with zipfile.ZipFile(pending,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for path,relative in sorted(common+payload):archive.write(path,str(relative))
            archive.writestr('export_manifest.json',json.dumps(manifest,indent=2))
        if pending.stat().st_size>28*1024*1024:raise ValueError('Export part exceeded28MiB')
        pending.replace(destination);paths.append(destination)
    atomic_json(root/'latest_results.json',dict(archives=list(map(str,paths)),parts=len(paths),export_id=export_id,
        scope='training settings/history, fixed91 inference/metrics and small synthetic QC preview; no full-resolution source images or model weights'))
    return paths
