"""A one-class DETR detector for the fixed616/91 crop-segmentation experiment.

The production initialization is COCO-pretrained facebook/detr-resnet-50. This
is a practical initialization, not an exact reproduction of the paper's
ImageNet-initialized detector. Every epoch visits every training rater group.
Only bbox proposals are exported; no segmentation-area filter is applied.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import gc
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random

os.environ.setdefault('TORCH_CUDNN_V8_API_DISABLED', '1')
import cv2
import numpy as np
from pycocotools import mask as mu
import torch
from torch.utils.data import Dataset, DataLoader

from native_instance_data import load_cohort, annotation_rle, preprocess_image, NativeInstanceDataset

TRANSFORMERS_VERSION = '4.57.1'
DEFAULTS = dict(detr_epochs=30, detr_batch=1, detr_accumulate=4,
    detr_image_size=800, detr_lr=1e-5, detr_backbone_lr=1e-6,
    detr_weight_decay=1e-4, detr_clip_norm=.1, detr_pretrained=True,
    detr_model_name='facebook/detr-resnet-50', detr_revision=None, detr_confidence=.05,
    detr_max_det=100, detr_amp=True, detr_tiny_fixture=False,
    preprocessing='limb', strict=True, seed=42, device='cuda')
FIXED = tuple(key for key in DEFAULTS if key not in ('device','detr_confidence','detr_max_det'))
SOURCE_FILES = ('filament_detr.py', 'native_instance_data.py')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.pending')
    try:
        temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        temporary.replace(path)
    finally:temporary.unlink(missing_ok=True)


def atomic_torch(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.pending')
    try:torch.save(value,temporary);temporary.replace(path)
    finally:temporary.unlink(missing_ok=True)


def cpu_copy(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_copy(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_copy(v) for v in value)
    return value


def seed_everything(seed):
    random.seed(seed);np.random.seed(seed%(2**32));torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda'] and torch.cuda.is_available():torch.cuda.set_rng_state_all(state['cuda'])


def normalize_config(config):
    cfg=dict(DEFAULTS,**config)
    if 'detr_epochs' not in config and not cfg.get('strict',True) and 'epochs' in config:
        cfg['detr_epochs']=config['epochs']
    for key in ('detr_epochs','detr_batch','detr_accumulate','detr_image_size','detr_max_det'):
        if not isinstance(cfg[key],int) or cfg[key]<1:raise ValueError(key+' must be a positive integer')
    if cfg['detr_image_size']%32:raise ValueError('DETR input size must be a multiple of32')
    if not 0<cfg['detr_lr']<.1 or not 0<cfg['detr_backbone_lr']<=cfg['detr_lr']:
        raise ValueError('Invalid DETR learning rates')
    if not 0<=cfg['detr_confidence']<=1 or cfg['detr_clip_norm']<=0:raise ValueError('Invalid DETR score/gradient clip')
    if cfg['detr_tiny_fixture'] and cfg['strict']:raise ValueError('Tiny fixture model cannot be used in strict production')
    return cfg


def device_for(value):
    device=torch.device('cuda:0' if str(value)=='0' else value)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('Select a Colab T4 GPU runtime')
    return device


def runtime_versions():
    values={key:importlib.metadata.version(key) for key in ('torch','torchvision','transformers','numpy','scipy','timm','pycocotools')}
    if values['transformers']!=TRANSFORMERS_VERSION:
        raise ValueError('Use transformers=='+TRANSFORMERS_VERSION)
    return dict(values,opencv=cv2.__version__,torch_cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version())


def source_hashes():
    return {name:digest(Path(__file__).with_name(name)) for name in SOURCE_FILES}


def build_detector(config, pretrained=True, model_config=None):
    """Build a full pretrained detector or an explicitly tiny CPU test model."""
    from transformers import DetrConfig, DetrForObjectDetection, ResNetConfig
    cfg=normalize_config(config)
    if model_config is not None:
        options=DetrConfig.from_dict(model_config)
        # Loading a completed checkpoint must never fetch backbone weights.
        options.use_pretrained_backbone=False
        return DetrForObjectDetection(options)
    if cfg['detr_tiny_fixture']:
        backbone=ResNetConfig(depths=[1,1,1,1],hidden_sizes=[16,32,64,128],embedding_size=16,
                              layer_type='basic',out_features=['stage4'])
        options=DetrConfig(use_timm_backbone=False,use_pretrained_backbone=False,backbone=None,
            backbone_config=backbone.to_dict(),d_model=32,encoder_layers=1,decoder_layers=1,
            encoder_attention_heads=4,decoder_attention_heads=4,encoder_ffn_dim=64,decoder_ffn_dim=64,
            num_queries=10,num_labels=1,dropout=.1,auxiliary_loss=False)
        return DetrForObjectDetection(options)
    if pretrained:
        print('DETR: COCO-pretrained box/transformer weights; new single-filament classifier head.',flush=True)
        return DetrForObjectDetection.from_pretrained(cfg['detr_model_name'],num_labels=1,
            id2label={0:'filament'},label2id={'filament':0},ignore_mismatched_sizes=True,
            use_pretrained_backbone=False,revision=cfg['detr_revision'])
    options=DetrConfig(backbone='resnet50',use_timm_backbone=True,use_pretrained_backbone=False,
                       num_labels=1,id2label={0:'filament'},label2id={'filament':0})
    return DetrForObjectDetection(options)


def box_tensor(annotations):
    boxes=[]
    for annotation in annotations:
        x,y,w,h=map(float,mu.toBbox(annotation_rle(annotation)))
        if w>0 and h>0:boxes.append([(x+w/2)/2048,(y+h/2)/2048,w/2048,h/2048])
    return torch.tensor(boxes,dtype=torch.float32).reshape(-1,4)


def prepare_pixels(pixels, image_size=800, preprocessing='limb'):
    corrected=preprocess_image(pixels,preprocessing)
    return _resized_normalized(corrected,image_size)


def _resized_normalized(corrected,image_size):
    resized=cv2.resize(corrected,(image_size,image_size),interpolation=cv2.INTER_AREA)
    rgb=torch.from_numpy(np.ascontiguousarray(np.repeat(resized[None],3,axis=0))).float()
    mean=torch.tensor([.485,.456,.406])[:,None,None]
    std=torch.tensor([.229,.224,.225])[:,None,None]
    return (rgb-mean)/std


class DetectionDataset(Dataset):
    """One deterministic sample per real annotation group in each epoch."""
    def __init__(self,groups,images,config,source_sha256=None):
        self.groups=list(groups);self.images=Path(images);self.cfg=normalize_config(config);self.epoch=0
        if not self.groups or any(row.get('role','train')!='train' for row in self.groups):
            raise ValueError('Nonempty training groups only')
        self.source_sha256=source_sha256 or {}
        cache_dir=Path(self.cfg.get('cache_root','/content/solar_roi_detr_cache'))/'detr_source_images'
        self.source=NativeInstanceDataset(self.groups,self.images,crop=2048,steps_per_epoch=1,batch_size=1,
            training=False,preprocessing=self.cfg['preprocessing'],cache_dir=cache_dir,
            source_sha256=self.source_sha256)
        self._image=lru_cache(maxsize=4)(self._read_image)
        self._boxes=[box_tensor(group['annotations']) for group in self.groups]
    def __len__(self):return len(self.groups)
    def set_epoch(self,epoch):self.epoch=int(epoch)
    def _read_image(self,name):
        corrected=self.source._load_image(name).astype(np.float32)/255.
        return _resized_normalized(corrected,self.cfg['detr_image_size'])
    def __getitem__(self,index):
        group=self.groups[index];pixels=self._image(group['file_name']).clone();boxes=self._boxes[index].clone()
        rng=np.random.default_rng(np.random.SeedSequence([self.cfg['seed'],self.epoch,int(index)]))
        if rng.random()<.5:
            pixels=pixels.flip(-1)
            if len(boxes):boxes[:,0]=1-boxes[:,0]
        if rng.random()<.5:
            pixels=pixels.flip(-2)
            if len(boxes):boxes[:,1]=1-boxes[:,1]
        return pixels,dict(class_labels=torch.zeros(len(boxes),dtype=torch.int64),boxes=boxes)


def collate(batch):
    images,targets=zip(*batch)
    return torch.stack(images),list(targets)


def input_identity(cfg,cohort):
    names=sorted({row['file_name'] for row in cohort['train_groups']}|set(cohort['validation_names']))
    stamps={}
    for number,name in enumerate(names,1):
        stamps[name]=digest(Path(cfg['images'])/name)
        if number%100==0 or number==len(names):print(f'DETR image hash {number}/{len(names)}',flush=True)
    initialization_path=Path(cfg['output_root'])/'detr/initialization.json'
    initialization=json.loads(initialization_path.read_text()) if initialization_path.exists() else None
    return dict(settings={key:cfg[key] for key in FIXED},metadata=cohort['input_sha256'],images=stamps,
                sources=source_hashes(),runtime=runtime_versions(),
                initialization=initialization,
                initial_weights='COCO DETR box+transformer; new one-class head' if cfg['detr_pretrained'] else 'random fixture/architecture',
                epoch_unit='one complete pass over every training annotation/rater group',
                validation_selects_checkpoint=False,bitwise_GPU_resume_guaranteed=False)


def _compact(state):
    output=dict(state)
    for name in ('optimizer','scheduler','scaler','rng'):output[name]=None
    return output


def parameter_sha256(model):
    checksum=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        tensor=value.detach().cpu().contiguous()
        checksum.update(name.encode());checksum.update(str(tensor.dtype).encode())
        checksum.update(str(tuple(tensor.shape)).encode());checksum.update(tensor.numpy().tobytes())
    return checksum.hexdigest()


def train_detr(config,run_until=None):
    cfg=normalize_config(config);cap=cfg['detr_epochs'] if run_until is None else int(run_until)
    if not 1<=cap<=cfg['detr_epochs']:raise ValueError('DETR run-until must be within fixed final epochs')
    device=device_for(cfg['device']);torch.backends.cudnn.benchmark=False
    cohort=load_cohort(cfg);fixed=input_identity(cfg,cohort);run_id=fingerprint(fixed)
    folder=Path(cfg['output_root'])/'detr';folder.mkdir(parents=True,exist_ok=True)
    config_path=folder/'training_identity.json';last=folder/'last_full.pt';stage=folder/f'epoch{cap:03d}.pt'
    if config_path.exists() and json.loads(config_path.read_text())['identity']!=run_id:
        # Recover the tiny atomic setup window where initialization.json was
        # saved but training_identity.json still contains initialization=None.
        prior=json.loads(config_path.read_text())
        before_initialization=dict(fixed,initialization=None)
        recoverable=(not last.exists() and not stage.exists() and
                     prior.get('identity')==fingerprint(before_initialization))
        if not recoverable:
            raise ValueError('DETR data/code/settings/runtime changed; use a new output root')
    if last.exists() and not config_path.exists():raise ValueError('Unidentified DETR last_full.pt retained')
    atomic_json(config_path,dict(identity=run_id,protocol=fixed))
    atomic_json(folder/'source_images.json',fixed['images'])
    if stage.exists():
        state=torch.load(stage,map_location='cpu',weights_only=False)
        if (state.get('identity'),state.get('completed_epoch'))!=(run_id,cap):raise ValueError('Wrong DETR saved stage')
        return dict(status='complete' if cap==cfg['detr_epochs'] else 'stage_complete',checkpoint=str(stage),
            checkpoint_sha256=digest(stage),epochs_completed=cap,fixed_final_epoch=cfg['detr_epochs'],identity=run_id)
    saved=torch.load(last,map_location='cpu',weights_only=False) if last.exists() else None
    if saved and (saved.get('identity')!=run_id or saved.get('completed_epoch',-1)>cap):raise ValueError('Wrong/later DETR checkpoint')
    seed_everything(cfg['seed'])
    model_cfg=dict(cfg)
    if fixed['initialization'] and fixed['initialization'].get('resolved_revision'):
        model_cfg['detr_revision']=fixed['initialization']['resolved_revision']
    model=build_detector(model_cfg,pretrained=bool(cfg['detr_pretrained'] and saved is None),
                         model_config=saved['model_config'] if saved else None).to(device)
    if saved is None:
        initialization=dict(model_name=cfg['detr_model_name'],requested_revision=cfg['detr_revision'],
            resolved_revision=getattr(model.config,'_commit_hash',None),
            initial_parameter_sha256=parameter_sha256(model),
            new_single_class_head=True,pretrained=cfg['detr_pretrained'],tiny_fixture=cfg['detr_tiny_fixture'])
        if fixed['initialization'] is not None and initialization!=fixed['initialization']:
            raise ValueError('Initial DETR weights or resolved revision changed; previous initialization retained')
        fixed['initialization']=initialization;run_id=fingerprint(fixed)
        atomic_json(folder/'initialization.json',initialization)
        atomic_json(config_path,dict(identity=run_id,protocol=fixed))
    regular=[];backbone=[]
    for name,param in model.named_parameters():
        if param.requires_grad:(backbone if 'backbone' in name else regular).append(param)
    optimizer=torch.optim.AdamW([dict(params=regular,lr=cfg['detr_lr']),dict(params=backbone,lr=cfg['detr_backbone_lr'])],
                                weight_decay=cfg['detr_weight_decay'])
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda e:.1+.9*(1+math.cos(math.pi*e/cfg['detr_epochs']))/2)
    amp=bool(cfg['detr_amp'] and device.type=='cuda');scaler=torch.amp.GradScaler('cuda',enabled=amp)
    history=[];start=0
    if saved:
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        scaler.load_state_dict(saved['scaler']);restore_rng(saved['rng']);history=saved['history'];start=int(saved['completed_epoch'])
        if len(history)!=start:raise ValueError('DETR completed history inconsistent')
    dataset=DetectionDataset(cohort['train_groups'],cfg['images'],cfg,fixed['images'])
    print(f"DETR completed{start} → requested{cap}; final{cfg['detr_epochs']} epochs, {len(dataset)} training rater records/epoch",flush=True)
    for epoch in range(start,cap):
        seed_everything(cfg['seed']+10007*epoch);dataset.set_epoch(epoch)
        loader=DataLoader(dataset,batch_size=cfg['detr_batch'],shuffle=True,num_workers=0,collate_fn=collate,
                          generator=torch.Generator().manual_seed(cfg['seed']+10007*epoch))
        model.train();optimizer.zero_grad(set_to_none=True);totals={};batches=attempts=updates=0
        for index,(pixels,targets) in enumerate(loader):
            pixels=pixels.to(device);targets=[{key:value.to(device) for key,value in target.items()} for target in targets]
            window_start=(index//cfg['detr_accumulate'])*cfg['detr_accumulate']
            window=min(cfg['detr_accumulate'],len(loader)-window_start)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):
                result=model(pixel_values=pixels,labels=targets);loss=result.loss
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite DETR loss; previous completed epoch retained')
            scaler.scale(loss/window).backward();batches+=1
            terms=dict(result.loss_dict,total_loss=loss)
            for key,value in terms.items():totals[key]=totals.get(key,0.)+float(value.detach().cpu())
            if (index+1)%cfg['detr_accumulate']==0 or index+1==len(loader):
                scaler.unscale_(optimizer)
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['detr_clip_norm'])
                if not amp and not torch.isfinite(norm):
                    raise FloatingPointError('Nonfinite DETR gradients; previous completed epoch retained')
                before=scaler.get_scale();scaler.step(optimizer);scaler.update();attempts+=1;updates+=int(scaler.get_scale()>=before)
                optimizer.zero_grad(set_to_none=True)
            if (index+1)%50==0 or index+1==len(loader):
                print(f"DETR epoch{epoch+1}/{cfg['detr_epochs']} batch{index+1}/{len(loader)} loss{totals['total_loss']/batches:.5f}",flush=True)
        if batches!=len(loader):raise RuntimeError('Incomplete DETR epoch cannot be saved')
        scheduler.step()
        history.append(dict(epoch=epoch+1,losses={key:value/batches for key,value in totals.items()},
                            annotation_records=len(dataset),batches=batches,optimizer_attempts=attempts,
                            optimizer_updates=updates,amp_skipped_updates=attempts-updates,next_lr=scheduler.get_last_lr()))
        state=dict(schema_version=1,identity=run_id,completed_epoch=epoch+1,fixed_final_epoch=cfg['detr_epochs'],
                   model_config=model.config.to_dict(),model=cpu_copy(model.state_dict()),optimizer=cpu_copy(optimizer.state_dict()),
                   scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),rng=rng_state(),history=history)
        atomic_torch(last,state);atomic_json(folder/'history.json',history)
        atomic_json(folder/'progress.json',dict(completed_epoch=epoch+1,fixed_final_epoch=cfg['detr_epochs'],identity=run_id))
    if start==cap:state=saved
    atomic_torch(stage,_compact(state))
    del model;gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    result=dict(status='complete' if cap==cfg['detr_epochs'] else 'stage_complete',checkpoint=str(stage),
                checkpoint_sha256=digest(stage),epochs_completed=cap,fixed_final_epoch=cfg['detr_epochs'],identity=run_id)
    atomic_json(folder/'result.json',result);return result


def infer_detr_image(model,image,config):
    cfg=normalize_config(config);device=next(model.parameters()).device
    pixels=prepare_pixels(image,cfg['detr_image_size'],cfg['preprocessing'])[None].to(device)
    amp=bool(cfg['detr_amp'] and device.type=='cuda')
    model.eval()
    with torch.inference_mode(),torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):
        result=model(pixel_values=pixels)
    scores=result.logits.float().softmax(-1)[0,:,:-1].max(-1).values.cpu().numpy()
    boxes=result.pred_boxes.float()[0].cpu().numpy()
    output=[]
    for index in sorted(range(len(scores)),key=lambda i:(-float(scores[i]),i)):
        if scores[index]<cfg['detr_confidence']:continue
        cx,cy,w,h=map(float,boxes[index]);box=np.clip([(cx-w/2)*2048,(cy-h/2)*2048,(cx+w/2)*2048,(cy+h/2)*2048],0,2048)
        if not np.isfinite(box).all() or box[2]<=box[0] or box[3]<=box[1]:continue
        output.append(dict(box=list(map(float,box)),score=float(scores[index])))
        if len(output)>=cfg['detr_max_det']:break
    return output


def export_proposals(config,checkpoint=None,role='validation'):
    cfg=normalize_config(config)
    if role not in ('train','validation'):raise ValueError('Proposal role must be train or validation')
    cohort=load_cohort(cfg);folder=Path(cfg['output_root'])/'detr'
    checkpoint=Path(checkpoint or folder/f'epoch{cfg["detr_epochs"]:03d}.pt')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    fixed=json.loads((folder/'training_identity.json').read_text())
    if saved.get('identity')!=fixed['identity'] or fingerprint(input_identity(cfg,cohort))!=fixed['identity']:
        raise ValueError('DETR export inputs/settings/code differ from training')
    export_id=fingerprint(dict(training=saved['identity'],checkpoint_sha256=digest(checkpoint),
                               confidence=cfg['detr_confidence'],max_det=cfg['detr_max_det']))
    destination=Path(cfg['output_root'])/'proposals/detr'/export_id[:16]/role;destination.mkdir(parents=True,exist_ok=True)
    names=sorted({group['file_name'] for group in cohort['train_groups']}) if role=='train' else cohort['validation_names']
    proposals={};model=None
    for number,name in enumerate(names,1):
        stem=Path(name).stem;path=destination/(stem+'.json.gz');record=None;stamp=fixed['protocol']['images'][name]
        if path.exists():
            try:
                record=json.loads(gzip.decompress(path.read_bytes()))
                if (record['identity'],record['image_sha256'],record['proposal_sha256'])!=(export_id,stamp,fingerprint(record['proposals'])):
                    record=None
            except (ValueError,OSError,KeyError):record=None
        if record is None:
            if model is None:
                model=build_detector(cfg,pretrained=False,model_config=saved['model_config']).to(device_for(cfg['device']))
                model.load_state_dict(saved['model']);model.eval()
            image=cv2.imread(str(Path(cfg['images'])/name),cv2.IMREAD_GRAYSCALE)
            rows=infer_detr_image(model,image,cfg)
            record=dict(identity=export_id,image_sha256=stamp,file_name=name,proposals=rows,proposal_sha256=fingerprint(rows),
                        completed_epoch=saved['completed_epoch'],fixed_final_epoch=saved['fixed_final_epoch'])
            temporary=path.with_name(path.name+'.pending');temporary.write_bytes(gzip.compress(json.dumps(record,separators=(',',':')).encode()));temporary.replace(path)
        proposals[stem]=record['proposals']
        if number%25==0 or number==len(names):print(f'DETR proposals {role} {number}/{len(names)}',flush=True)
    atomic_json(destination/'manifest.json',dict(identity=export_id,role=role,names=names,
        checkpoint_sha256=digest(checkpoint),confidence=cfg['detr_confidence'],max_det=cfg['detr_max_det'],
        mask_area_filter=False,proposal_count=sum(map(len,proposals.values())),
        completed_epoch=saved['completed_epoch'],fixed_final_epoch=saved['fixed_final_epoch']))
    if model is not None:del model;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return proposals


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--run-until',type=int)
    parser.add_argument('--phase',choices=('train','export'),default='train');parser.add_argument('--role',choices=('train','validation'),default='validation')
    args=parser.parse_args();cfg=json.loads(Path(args.config).read_text())
    result=train_detr(cfg,args.run_until) if args.phase=='train' else export_proposals(cfg,role=args.role)
    print(json.dumps(result if args.phase=='train' else dict(role=args.role,images=len(result),proposals=sum(map(len,result.values()))),indent=2),flush=True)


if __name__=='__main__':main()
