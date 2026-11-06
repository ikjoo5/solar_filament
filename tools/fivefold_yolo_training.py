"""Fold-local YOLO26s training for the selected YOLO+native study.

Only generic YOLO weights initialize fresh runs. Validation selection never
chooses a model: every stage exports the declared completed epoch.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np
from pycocotools import mask as mu
import torch

import grouped_cv_training as native

PRETRAINED = 'yolo26s-seg.pt'
ULTRALYTICS_VERSION = '8.4.182'


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def segment_lines(annotation):
    seg=annotation['segmentation']
    if isinstance(seg,list):rle=mu.merge(mu.frPyObjects(seg,2048,2048))
    elif isinstance(seg['counts'],list):rle=mu.frPyObjects(seg,2048,2048)
    else:
        rle=dict(seg)
        if isinstance(rle['counts'],str):rle['counts']=rle['counts'].encode()
    mask=mu.decode(rle).astype(np.uint8)
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask,connectivity=8)
    lines=[]
    for lab in range(1,count):
        if stats[lab,cv2.CC_STAT_AREA]<20:continue
        contours,_=cv2.findContours((labels==lab).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        contour=max(contours,key=cv2.contourArea)
        if len(contour)<3 or cv2.contourArea(contour)<=0:continue
        _,_,w,h=cv2.boundingRect(contour)
        if min(w,h)<2:continue
        xy=np.clip(contour.reshape(-1,2).astype(np.float32)/2048,0,1)
        lines.append('0 '+' '.join(f'{v:.6f}' for v in xy.reshape(-1)))
    return lines


def prepare_dataset(cfg, runtime):
    # This checks all heldout siblings/embargo, but only train labels enter YOLO data.
    data,train_ids,_,_,_,info=native.prepare_inputs(cfg)
    data_key=identity({key:native.digest(cfg[key]) for key in ('annotations','train_split','base_runtime')})[:12]
    root=Path(cfg['cache_root'])/'yolo_datasets'/f"fold{cfg['fold']}_{data_key}"
    image_dir=root/'images/train';labels_dir=root/'labels/train'
    image_dir.mkdir(parents=True,exist_ok=True);labels_dir.mkdir(parents=True,exist_ok=True)
    store=runtime.ImageStore(Path(cfg['images']),Path(cfg['cache_root'])/'images_real','limb_only')
    by_id={str(r['id']):r for r in data['images']};anns=defaultdict(list)
    for ann in data['annotations']:
        if str(ann['image_id']) in train_ids:anns[str(ann['image_id'])].append(ann)
    cache=root/'physical_png';cache.mkdir(exist_ok=True)
    expected_aliases={re.sub('[^A-Za-z0-9_-]','_',sid)+'.png' for sid in train_ids}
    if len(expected_aliases)!=len(train_ids):raise ValueError('YOLO alias collision')
    if {p.name for p in image_dir.glob('*.png')}-expected_aliases:
        raise ValueError('Dataset cache contains images outside this fold training set')
    for index,sid in enumerate(sorted(train_ids),1):
        row=by_id[sid];name=row['file_name'];physical=cache/(Path(name).stem+'.png')
        if not physical.exists():
            if not cv2.imwrite(str(physical),store.load(name)):raise OSError('PNG cache write failed')
        alias=re.sub('[^A-Za-z0-9_-]','_',sid)
        target=image_dir/(alias+'.png')
        if not target.exists():target.symlink_to(physical.resolve())
        lines=[line for ann in anns[sid] for line in segment_lines(ann)]
        (labels_dir/(alias+'.txt')).write_text('\n'.join(lines)+('\n' if lines else ''))
        if index%100==0:print(f'YOLO training labels {index}/{len(train_ids)}',flush=True)
    # The internal validator is disabled. Its placeholder points to training only.
    path=root/'filament.yaml'
    path.write_text(json.dumps(dict(path=str(root.resolve()),train='images/train',val='images/train',names={0:'filament'})))
    native.atomic_json(root/'data_manifest.json',dict(train_annotation_ids=sorted(train_ids),
        train_physical=info['train_physical'],validation_used_for_training=False,
        annotations_sha256=native.digest(cfg['annotations']),train_split_sha256=native.digest(cfg['train_split'])))
    return path


def checkpoint_callback(folder, runtime, run_identity, stage, stage_path):
    """Copy complete optimizer/live/EMA/RNG state before Ultralytics can strip it."""
    def save(trainer):
        payload=torch.load(trainer.last,map_location='cpu',weights_only=False)
        payload['cv_identity']=run_identity
        payload['cv_live_state_dict']={k:v.detach().cpu().clone() for k,v in trainer.model.state_dict().items()}
        payload['optimizer']=deepcopy(trainer.optimizer.state_dict())
        payload['cv_scheduler']=trainer.scheduler.state_dict()
        payload['cv_rng']=runtime.capture_rng()
        payload['cv_epoch']=int(trainer.epoch)+1
        history_path=folder/'history.json'
        history=json.loads(history_path.read_text()) if history_path.exists() else []
        raw_losses=getattr(trainer,'tloss',{})
        losses={k:float(v.detach().cpu()) if hasattr(v,'detach') else float(v)
                for k,v in raw_losses.items()} if isinstance(raw_losses,dict) else {}
        history=[row for row in history if row['epoch']<payload['cv_epoch']]
        history.append(dict(epoch=payload['cv_epoch'],train_losses=losses,
                            learning_rates=[group['lr'] for group in trainer.optimizer.param_groups]))
        payload['cv_history']=history
        runtime.atomic_torch_save(folder/'last_full.pt',payload)
        native.atomic_json(history_path,history)
        native.atomic_json(folder/'progress.json',dict(epoch=payload['cv_epoch'],identity=run_identity))
        if payload['cv_epoch']==stage:
            compact=dict(payload)
            compact.update(optimizer=None,scaler=None,cv_live_state_dict=None,cv_rng=None,cv_scheduler=None)
            runtime.atomic_torch_save(stage_path,compact)
    return save


def run(cfg, _yolo=None, _runtime=None):
    if cfg.get('yolo_pretrained',PRETRAINED)!=PRETRAINED:
        raise ValueError('Only generic yolo26s-seg.pt initialization is allowed')
    for key in ('initial_checkpoint','warm_start'):
        if cfg.get(key):raise ValueError('Old solar weights cannot initialize a fold')
    stage=int(cfg['stage_epoch']);fold=int(cfg['fold'])
    if not 1<=stage<=100:raise ValueError('Stage must be1..100')
    runtime=_runtime or native.load_runtime(cfg['base_runtime'])
    if torch.cuda.is_available():native.configure_cuda()
    folder=Path(cfg['study_root'])/f'fold{fold}'/'yolo';folder.mkdir(parents=True,exist_ok=True)
    stage_path=folder.parent/f'stage{stage:03d}'/'yolo.pt';stage_path.parent.mkdir(exist_ok=True)
    settings=dict(architecture=PRETRAINED,fold=fold,epochs=100,imgsz=2048,batch=1,
        optimizer='AdamW',lr0=.001,lrf=.01,cos_lr=True,degrees=180.,hsv_v=.3,
        fliplr=.5,flipud=.5,mask_ratio=4,overlap_mask=True,seed=42+fold,
        model_selection='Fixed completed stage, never outer validation best',
        preprocessing='limb_only',synthetic_ratio=0,
        sources={k:native.digest(cfg[k]) for k in ('annotations','fold_plan','train_split','val_split','base_runtime')},
        runner_sha256=native.digest(__file__),ultralytics=ULTRALYTICS_VERSION)
    run_id=identity(settings);config_path=folder/'config.json'
    if config_path.exists() and json.loads(config_path.read_text())['identity']!=run_id:
        raise ValueError('Different YOLO training experiment; existing files preserved')
    native.atomic_json(config_path,dict(identity=run_id,settings=settings))
    last=folder/'last_full.pt';saved=None
    if stage_path.exists():
        checkpoint=torch.load(stage_path,map_location='cpu',weights_only=False)
        if checkpoint.get('cv_identity')!=run_id or checkpoint.get('cv_epoch')!=stage:raise ValueError('Wrong YOLO stage weights')
        return dict(fold=fold,stage_epoch=stage,identity=run_id,checkpoint=str(stage_path),checkpoint_sha256=native.digest(stage_path))
    if last.exists():
        saved=torch.load(last,map_location='cpu',weights_only=False)
        if saved.get('cv_identity')!=run_id:raise ValueError('Wrong YOLO resume identity')
        if saved.get('cv_history'):native.atomic_json(folder/'history.json',saved['cv_history'])
        if saved['cv_epoch']>stage:raise ValueError('Cannot reconstruct an earlier missing YOLO stage')
        if saved['cv_epoch']==stage:
            compact=dict(saved);compact.update(optimizer=None,scaler=None,cv_live_state_dict=None,cv_rng=None,cv_scheduler=None)
            runtime.atomic_torch_save(stage_path,compact)
            return dict(fold=fold,stage_epoch=stage,identity=run_id,checkpoint=str(stage_path),checkpoint_sha256=native.digest(stage_path))
    yaml_path=prepare_dataset(cfg,runtime)
    from ultralytics import YOLO,__version__
    from ultralytics.models.yolo.segment.train import SegmentationTrainer
    if __version__!=ULTRALYTICS_VERSION:raise ValueError('Use pinned Ultralytics8.4.182')
    class FoldTrainer(SegmentationTrainer):
        def validate(self):return {},0.
        def final_eval(self):pass  # Keep optimizer checkpoints; no best-on-outer selection.
    model=(_yolo or YOLO)(str(last) if saved is not None else PRETRAINED,task='segment')
    if saved is None and _yolo is None and len(model.names)!=80:
        raise ValueError('Fresh YOLO initialization must be the generic80-class COCO model, not renamed solar weights')
    model.add_callback('on_model_save',checkpoint_callback(folder,runtime,run_id,stage,stage_path))
    def restore(trainer):
        if saved is not None:
            trainer.model.load_state_dict(saved['cv_live_state_dict'])
            trainer.scheduler.load_state_dict(saved['cv_scheduler'])
            runtime.restore_rng(saved['cv_rng'])
    model.add_callback('on_train_start',restore)
    model.add_callback('on_fit_epoch_end',lambda trainer:setattr(trainer,'stop',int(trainer.epoch)+1>=stage))
    options=dict(data=str(yaml_path),epochs=100,imgsz=2048,batch=1,device=cfg.get('yolo_device',0),workers=0,
        cache=False,max_det=100,patience=1000,project=str(Path(cfg['cache_root'])/'yolo_runs'),
        name=f'fold{fold}',exist_ok=True,optimizer='AdamW',lr0=.001,lrf=.01,cos_lr=True,amp=True,
        seed=42+fold,deterministic=True,fliplr=.5,flipud=.5,degrees=180.,translate=0.,scale=0.,shear=0.,
        perspective=0.,hsv_h=0.,hsv_s=0.,hsv_v=.3,mosaic=0.,mixup=0.,copy_paste=0.,
        mask_ratio=4,overlap_mask=True,save=True,save_period=-1,val=False,plots=False,
        pretrained=True,resume=str(last) if saved is not None else False)
    print(f'YOLO fold{fold}: 이번목표{stage}epoch.100은재개용스케줄이며이번에100까지학습하지않습니다.',flush=True)
    model.train(trainer=FoldTrainer,**options)
    if not stage_path.exists():raise RuntimeError('YOLO stopped before the declared common stage')
    return dict(fold=fold,stage_epoch=stage,identity=run_id,checkpoint=str(stage_path),checkpoint_sha256=native.digest(stage_path))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--result',required=True)
    args=parser.parse_args();result=run(json.loads(Path(args.config).read_text()))
    native.atomic_json(args.result,result)
