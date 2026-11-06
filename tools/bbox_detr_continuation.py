"""Continue the reviewed DETR30 for an explicit additional30-epoch stage.

The source run stays read-only. AdamW moments are retained while the learning-
rate schedule restarts explicitly. Production moves800/FP16 to1280/BF16;
GradScaler is disabled for BF16. This is not an uninterrupted60-epoch schedule.
ROI is never loaded, modified, or trained in this module.
"""
from __future__ import annotations
import argparse
import fcntl
import gc
import gzip
import json
import math
from functools import wraps
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import filament_detr as legacy
from native_instance_data import load_cohort

BASE_COMPACT_SHA = '4ea4a09244b24b00ac61a9618a11327a31f7156bcf279e7da87a1ea81b6f6565'
BASE_IDENTITY = '5fea3859b6675921c443de1174edd4fc36517b594cf18e6727b12028d8df97c0'
EXTENSION_DEFAULTS = dict(added_epochs=30, continuation_lr=1e-5,
    continuation_backbone_lr=1e-6, continuation_lr_floor=.1,
    continuation_image_size=1280, continuation_batch=2,
    continuation_accumulate=2, continuation_precision='bf16')


def load_checkpoint(path):
    """Load our tensor/dictionary checkpoints with only NumPy RNG allowlisted."""
    numpy_core = getattr(np, '_core', None)
    if numpy_core is None:
        numpy_core = np.core
    safe = [numpy_core.multiarray._reconstruct, np.ndarray, np.dtype,
            type(np.dtype(np.uint32))]
    with torch.serialization.safe_globals(safe):
        return torch.load(path, map_location='cpu', weights_only=True)


def normalize_config(config):
    base_root = Path(config['base_root']).resolve()
    output_root = Path(config['output_root']).resolve()
    if output_root == base_root or output_root in base_root.parents or base_root in output_root.parents:
        raise ValueError('Extension output must be separate from the read-only source run')
    base = json.loads((base_root/'experiment_config.json').read_text())
    base = legacy.normalize_config(dict(base, output_root=str(base_root)))
    cfg = dict(EXTENSION_DEFAULTS, **config)
    cfg.update(base_root=str(base_root), output_root=str(output_root))
    if not isinstance(cfg['added_epochs'], int) or cfg['added_epochs'] < 1:
        raise ValueError('added_epochs must be positive integer')
    if not 0 < cfg['continuation_backbone_lr'] <= cfg['continuation_lr'] < .1:
        raise ValueError('Invalid continuation learning rates')
    if not 0 < cfg['continuation_lr_floor'] <= 1:
        raise ValueError('Invalid continuation learning rate floor')
    for key in ('continuation_image_size','continuation_batch','continuation_accumulate'):
        if not isinstance(cfg[key],int) or cfg[key]<1:raise ValueError(key+' must be positive integer')
    if cfg['continuation_image_size']%32:raise ValueError('Continuation image size must be a multiple of32')
    if cfg['continuation_precision'] not in ('bf16','fp16','fp32'):
        raise ValueError('Continuation precision must be bf16/fp16/fp32')
    if base['strict'] and (base['detr_epochs'] != 30 or cfg['added_epochs'] != 30):
        raise ValueError('Reviewed production plan is DETR30 plus30 additional epochs')
    cfg['device'] = cfg.get('device', base['device'])
    cfg['cache_root'] = cfg.get('cache_root', '/content/solar_bbox_extension_cache')
    cache = Path(cfg['cache_root']).resolve()
    if any(cache == p or cache in p.parents or p in cache.parents for p in (base_root,output_root)):
        raise ValueError('Disposable cache must be outside source/output directories')
    cfg['_base_config'] = dict(base, device=cfg['device'], cache_root=cfg['cache_root'])
    cfg['_training_config'] = dict(cfg['_base_config'],detr_image_size=cfg['continuation_image_size'],
        detr_batch=cfg['continuation_batch'],detr_accumulate=cfg['continuation_accumulate'])
    return cfg


def _precision(cfg,device):
    precision=cfg['continuation_precision']
    if precision!='fp32' and device.type!='cuda':
        raise ValueError('Mixed-precision continuation requires CUDA; CPU fixtures must request fp32')
    if precision=='bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('Select an A100 or BF16-capable GPU; BF16 is not silently replaced by FP16')
    return dict(autocast=precision!='fp32',dtype=torch.bfloat16 if precision=='bf16' else torch.float16,
                scaler=precision=='fp16')


def install_fp32_bbox_head(model):
    """Keep normalized box coordinates precise under BF16/FP16 autocast.

    Only the existing MLP forward is wrapped, so parameter/state_dict names and
    checkpoint compatibility are unchanged. HF applies sigmoid to this FP32
    output after the MLP. Casting already-quantized pred_boxes afterwards would
    not recover coordinate precision, especially for thin objects at2048px.
    """
    head=model.bbox_predictor
    if getattr(head,'_solar_fp32_bbox_head',False):return model
    original=head.forward
    @wraps(original)
    def full_precision_forward(*args,**kwargs):
        tensors=[x for x in (*args,*kwargs.values()) if isinstance(x,torch.Tensor)]
        if not tensors:raise TypeError('BBox MLP requires tensor hidden states')
        cast=lambda value:value.float() if isinstance(value,torch.Tensor) and value.is_floating_point() else value
        with torch.autocast(device_type=tensors[0].device.type,enabled=False):
            return original(*(cast(value) for value in args),**{key:cast(value) for key,value in kwargs.items()})
    head.forward=full_precision_forward
    head._solar_fp32_bbox_head=True
    return model


def _tensor_states_equal(first, second):
    return set(first) == set(second) and all(torch.equal(first[k],second[k]) for k in first)


def _source_context(config):
    cfg = normalize_config(config)
    base = cfg['_base_config']; root = Path(cfg['base_root'])
    folder = root/'detr'; epoch = base['detr_epochs']
    compact = folder/f'epoch{epoch:03d}.pt'; full = folder/'last_full.pt'
    source_document = json.loads((folder/'training_identity.json').read_text())
    if legacy.fingerprint(source_document['protocol'])!=source_document['identity']:
        raise ValueError('Source recorded training protocol fingerprint is inconsistent')
    if base['strict']:
        if legacy.digest(compact) != BASE_COMPACT_SHA or source_document['identity'] != BASE_IDENTITY:
            raise ValueError('Source must be the reviewed uploaded DETR30 checkpoint')
    cohort = load_cohort(base)
    source_protocol = legacy.input_identity(base,cohort)
    current_runtime=source_protocol['runtime']
    # Moving a reviewed checkpoint from T4/FP16 into a new A100/BF16 stage is
    # intentional. Validate every source field against its original protocol,
    # and bind this stage's runtime separately so later resume cannot drift.
    source_protocol['runtime']=source_document['protocol']['runtime']
    if (legacy.fingerprint(source_protocol) != source_document['identity']
            or source_protocol != source_document['protocol']):
        raise ValueError('Source DETR data, code, or settings changed')
    compact_state = load_checkpoint(compact)
    full_state = load_checkpoint(full)
    for state in (compact_state,full_state):
        if (state.get('identity'),state.get('completed_epoch'),state.get('fixed_final_epoch')) != (
                source_document['identity'],epoch,epoch):
            raise ValueError('Source checkpoint must be the completed fixed-final-epoch model')
    if (not _tensor_states_equal(compact_state['model'],full_state['model'])
            or compact_state['history'] != full_state['history']
            or compact_state['model_config'] != full_state['model_config']):
        raise ValueError('Source last_full.pt and reviewed compact checkpoint disagree')
    if (not full_state.get('optimizer') or full_state.get('scaler') is None
            or not full_state.get('rng') or len(full_state['history']) != epoch):
        raise ValueError('Source full optimizer/scaler/RNG history is required')
    if len(full_state['optimizer']['param_groups']) != 2:
        raise ValueError('Expected DETR regular/backbone optimizer parameter groups')
    fixed = dict(
        stage='DETR-only continuation; ROI frozen outside this trainer',
        base_epoch=epoch,added_epochs=cfg['added_epochs'],
        schedule=dict(kind='explicit cosine restart retaining AdamW moments',
            learning_rates=[cfg['continuation_lr'],cfg['continuation_backbone_lr']],
            floor_fraction=cfg['continuation_lr_floor'],
            equivalent_to_uninterrupted_60_epoch_schedule=False),
        continuation=dict(image_size=cfg['continuation_image_size'],batch=cfg['continuation_batch'],
            accumulate=cfg['continuation_accumulate'],precision=cfg['continuation_precision'],
            bbox_head_precision='fp32 before final sigmoid',
            effective_batch=cfg['continuation_batch']*cfg['continuation_accumulate'],
            source_image_size=base['detr_image_size'],source_precision='fp16' if base['detr_amp'] else 'fp32',
            scaler_transition='retain source AMP scaler only for fp16; disabled/new empty state for bf16/fp32'),
        source=dict(identity=source_document['identity'],compact_sha256=legacy.digest(compact),
            full_state_sha256=legacy.digest(full),config_sha256=legacy.digest(root/'experiment_config.json'),
            metadata=source_protocol['metadata'],images=source_protocol['images']),
        source_protocol=source_protocol,
        extension_sources=dict(legacy.source_hashes(),bbox_detr_continuation=legacy.digest(__file__)),
        runtime=current_runtime,
        runtime_transition=dict(source=source_document['protocol']['runtime'],continuation=current_runtime,
            allowed_only_at_new_stage=True,continuation_resume_requires_same_runtime=True),
        validation_selects_checkpoint=False,
        epoch_unit='one complete pass over training annotation/rater records',
        bitwise_GPU_resume_guaranteed=False)
    del compact_state
    return cfg,cohort,fixed,full_state


def _report(cfg,state,path):
    added=int(state['added_epochs_completed'])
    return dict(status='complete' if added==cfg['added_epochs'] else 'stage_complete',
        checkpoint=str(path),checkpoint_sha256=legacy.digest(path),
        identity=state['identity'],added_epochs_completed=added,
        cumulative_epochs_completed=state['completed_epoch'],base_epochs=state['base_epoch'],
        fixed_added_epochs=cfg['added_epochs'],fixed_final_epoch=state['fixed_final_epoch'])


def train_extension(config,run_until_added=30):
    """Run a capped portion of the fixed additional stage; never changes base files."""
    cfg=normalize_config(config); cap=int(run_until_added)
    if not 0<=cap<=cfg['added_epochs']:
        raise ValueError('run_until_added must be within the fixed additional schedule')
    root=Path(cfg['output_root']);root.mkdir(parents=True,exist_ok=True)
    lock_path=Path('/tmp')/('bbox_detr_'+legacy.fingerprint(str(root))[:16]+'.lock')
    with lock_path.open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Another cell is running this DETR extension')
        return _train_locked(config,cap)


def _train_locked(config,cap):
    cfg,cohort,fixed,source = _source_context(config)
    root=Path(cfg['output_root']);run_id=legacy.fingerprint(fixed)
    document_path=root/'training_identity.json';last=root/'last_full.pt'
    base=cfg['_base_config'];training=cfg['_training_config']
    base_epoch=fixed['base_epoch'];cumulative=base_epoch+cap
    stage=root/f'epoch{cumulative:03d}.pt'
    if document_path.exists():
        doc=json.loads(document_path.read_text())
        if doc != dict(identity=run_id,protocol=fixed):
            raise ValueError('Extension source/data/code/runtime/schedule changed; retain previous results')
    elif last.exists() or stage.exists():
        raise ValueError('Unidentified extension checkpoint retained')
    legacy.atomic_json(document_path,dict(identity=run_id,protocol=fixed))
    legacy.atomic_json(root/'source_images.json',fixed['source']['images'])
    if stage.exists():
        saved=load_checkpoint(stage)
        _validate_state(saved,run_id,base_epoch,cfg['added_epochs'],cap)
        return _report(cfg,saved,stage)
    saved=load_checkpoint(last) if last.exists() else None
    if saved:
        _validate_state(saved,run_id,base_epoch,cfg['added_epochs'])
        if saved['added_epochs_completed']>cap:
            raise ValueError('Later extension checkpoint exists; cannot move backwards')
    device=legacy.device_for(cfg['device']);torch.backends.cudnn.benchmark=False
    precision=_precision(cfg,device)
    legacy.seed_everything(base['seed'])
    model=install_fp32_bbox_head(legacy.build_detector(base,pretrained=False,model_config=source['model_config']).to(device))
    regular=[];backbone=[]
    for name,param in model.named_parameters():
        if param.requires_grad: (backbone if 'backbone' in name else regular).append(param)
    optimizer=torch.optim.AdamW([dict(params=regular,lr=cfg['continuation_lr']),
        dict(params=backbone,lr=cfg['continuation_backbone_lr'])],weight_decay=base['detr_weight_decay'])
    initial = saved if saved else source
    model.load_state_dict(initial['model']);optimizer.load_state_dict(initial['optimizer'])
    rates=[cfg['continuation_lr'],cfg['continuation_backbone_lr']]
    for group,rate in zip(optimizer.param_groups,rates):
        group['lr']=rate;group['initial_lr']=rate
    schedule=lambda e:cfg['continuation_lr_floor']+(1-cfg['continuation_lr_floor'])*(1+math.cos(math.pi*e/cfg['added_epochs']))/2
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    scaler=torch.amp.GradScaler('cuda',enabled=precision['scaler'])
    if saved or (precision['scaler'] and initial['scaler']):scaler.load_state_dict(initial['scaler'])
    if saved:
        # LambdaLR initialization sets lr; restore optimizer after it, then its schedule state.
        optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        history=list(saved['extension_history']);start=int(saved['added_epochs_completed'])
    else:
        history=[];start=0
    legacy.restore_rng(initial['rng'])
    base_history=source['history'];del source,initial
    dataset=legacy.DetectionDataset(cohort['train_groups'],base['images'],training,fixed['source']['images'])
    print(f'DETR continuation: source{base_epoch}; added{start} ->{cap}/{cfg["added_epochs"]}; cumulative{base_epoch+cap}',flush=True)
    for added in range(start,cap):
        global_epoch=base_epoch+added
        legacy.seed_everything(base['seed']+10007*global_epoch);dataset.set_epoch(global_epoch)
        loader=DataLoader(dataset,batch_size=training['detr_batch'],shuffle=True,num_workers=0,
            collate_fn=legacy.collate,generator=torch.Generator().manual_seed(base['seed']+10007*global_epoch))
        model.train();optimizer.zero_grad(set_to_none=True);totals={};batches=attempts=updates=0
        for index,(pixels,targets) in enumerate(loader):
            pixels=pixels.to(device);targets=[{key:value.to(device) for key,value in t.items()} for t in targets]
            window_start=(index//training['detr_accumulate'])*training['detr_accumulate']
            window=min(training['detr_accumulate'],len(loader)-window_start)
            with torch.autocast(device_type=device.type,dtype=precision['dtype'],enabled=precision['autocast']):
                result=model(pixel_values=pixels,labels=targets);loss=result.loss
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite DETR continuation loss; completed checkpoint retained')
            scaler.scale(loss/window).backward();batches+=1
            for key,value in dict(result.loss_dict,total_loss=loss).items():
                totals[key]=totals.get(key,0.)+float(value.detach().cpu())
            if (index+1)%training['detr_accumulate']==0 or index+1==len(loader):
                scaler.unscale_(optimizer)
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),base['detr_clip_norm'])
                if not precision['scaler'] and not torch.isfinite(norm):raise FloatingPointError('Nonfinite continuation gradients')
                before=scaler.get_scale();scaler.step(optimizer);scaler.update()
                attempts+=1;updates+=int(scaler.get_scale()>=before);optimizer.zero_grad(set_to_none=True)
            if (index+1)%50==0 or index+1==len(loader):
                print(f'DETR cumulative{global_epoch+1}/{base_epoch+cfg["added_epochs"]} batch{index+1}/{len(loader)} loss{totals["total_loss"]/batches:.5f}',flush=True)
        if batches!=len(loader):raise RuntimeError('Incomplete extension epoch cannot be saved')
        scheduler.step()
        history.append(dict(added_epoch=added+1,epoch=global_epoch+1,
            losses={key:value/batches for key,value in totals.items()},annotation_records=len(dataset),
            batches=batches,optimizer_attempts=attempts,optimizer_updates=updates,
            amp_skipped_updates=attempts-updates,next_lr=scheduler.get_last_lr()))
        state=dict(schema_version=1,identity=run_id,base_epoch=base_epoch,added_epochs_completed=added+1,
            completed_epoch=global_epoch+1,fixed_added_epochs=cfg['added_epochs'],
            fixed_final_epoch=base_epoch+cfg['added_epochs'],base_history=base_history,extension_history=history,
            model_config=model.config.to_dict(),model=legacy.cpu_copy(model.state_dict()),
            optimizer=legacy.cpu_copy(optimizer.state_dict()),scheduler=scheduler.state_dict(),
            scaler=scaler.state_dict(),rng=legacy.rng_state())
        legacy.atomic_torch(last,state)
        legacy.atomic_json(root/'history.json',history)
        legacy.atomic_json(root/'progress.json',dict(added_epochs_completed=added+1,
            cumulative_epochs_completed=global_epoch+1,fixed_added_epochs=cfg['added_epochs'],identity=run_id))
    if start==cap:
        if saved:state=saved
        else:
            state=dict(schema_version=1,identity=run_id,base_epoch=base_epoch,added_epochs_completed=0,
                completed_epoch=base_epoch,fixed_added_epochs=cfg['added_epochs'],
                fixed_final_epoch=base_epoch+cfg['added_epochs'],base_history=base_history,extension_history=[],
                model_config=model.config.to_dict(),model=legacy.cpu_copy(model.state_dict()),
                optimizer=legacy.cpu_copy(optimizer.state_dict()),scheduler=scheduler.state_dict(),
                scaler=scaler.state_dict(),rng=legacy.rng_state())
    legacy.atomic_torch(stage,legacy._compact(state))
    del model;gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    report=_report(cfg,state,stage);legacy.atomic_json(root/'result.json',report)
    return report


def _validate_state(state,run_id,base_epoch,added_epochs,cap=None):
    added=state.get('added_epochs_completed',-1)
    if (state.get('identity')!=run_id or state.get('base_epoch')!=base_epoch
            or state.get('fixed_added_epochs')!=added_epochs or not 0<=added<=added_epochs
            or state.get('completed_epoch')!=base_epoch+added
            or state.get('fixed_final_epoch')!=base_epoch+added_epochs
            or len(state.get('extension_history',[]))!=added
            or (cap is not None and added!=cap)):
        raise ValueError('Wrong DETR extension checkpoint identity/epoch')


def export_extension_proposals(config,checkpoint,role='validation'):
    """Export native2048 bbox proposals; no model fitting or ROI inference."""
    if role not in ('train','validation'):raise ValueError('role must be train or validation')
    cfg,cohort,fixed,source=_source_context(config);del source
    root=Path(cfg['output_root']);base=cfg['_base_config'];run_id=legacy.fingerprint(fixed)
    document=json.loads((root/'training_identity.json').read_text())
    if document!=dict(identity=run_id,protocol=fixed):raise ValueError('Extension export identity changed')
    checkpoint=Path(checkpoint);saved=load_checkpoint(checkpoint)
    _validate_state(saved,run_id,fixed['base_epoch'],cfg['added_epochs'])
    export_id=legacy.fingerprint(dict(training=run_id,checkpoint_sha256=legacy.digest(checkpoint),
        confidence=base['detr_confidence'],max_det=base['detr_max_det']))
    destination=root/'proposals'/export_id[:16]/role;destination.mkdir(parents=True,exist_ok=True)
    names=sorted({g['file_name'] for g in cohort['train_groups']}) if role=='train' else cohort['validation_names']
    proposals={};model=None
    for number,name in enumerate(names,1):
        stem=Path(name).stem;path=destination/(stem+'.json.gz');record=None
        stamp=fixed['source']['images'][name]
        if path.exists():
            try:
                record=json.loads(gzip.decompress(path.read_bytes()))
                if (record['identity'],record['image_sha256'],record['proposal_sha256'])!=(export_id,stamp,legacy.fingerprint(record['proposals'])):record=None
            except (OSError,ValueError,KeyError):record=None
        if record is None:
            if model is None:
                model=install_fp32_bbox_head(legacy.build_detector(base,pretrained=False,model_config=saved['model_config']).to(legacy.device_for(cfg['device'])))
                model.load_state_dict(saved['model']);model.eval()
            pixels=cv2.imread(str(Path(base['images'])/name),cv2.IMREAD_GRAYSCALE)
            rows=infer_extension_image(model,pixels,cfg)
            record=dict(identity=export_id,image_sha256=stamp,file_name=name,proposals=rows,
                proposal_sha256=legacy.fingerprint(rows),completed_epoch=saved['completed_epoch'],
                added_epochs_completed=saved['added_epochs_completed'])
            pending=path.with_name(path.name+'.pending')
            pending.write_bytes(gzip.compress(json.dumps(record,separators=(',',':')).encode()));pending.replace(path)
        proposals[stem]=record['proposals']
        if number%25==0 or number==len(names):print(f'Extended DETR proposals {role} {number}/{len(names)}',flush=True)
    legacy.atomic_json(destination/'manifest.json',dict(identity=export_id,role=role,names=names,
        checkpoint_sha256=legacy.digest(checkpoint),cumulative_epochs_completed=saved['completed_epoch'],
        added_epochs_completed=saved['added_epochs_completed'],confidence=base['detr_confidence'],
        max_det=base['detr_max_det'],mask_area_filter=False,proposal_count=sum(map(len,proposals.values()))))
    if model is not None:del model;gc.collect()
    return proposals


def infer_extension_image(model,pixels,config):
    """Native2048 boxes from the continuation resolution and explicit precision."""
    cfg=config if '_base_config' in config else normalize_config(config)
    base=cfg['_base_config'];device=next(model.parameters()).device;precision=_precision(cfg,device)
    inputs=legacy.prepare_pixels(pixels,cfg['continuation_image_size'],base['preprocessing'])[None].to(device)
    model.eval()
    with torch.inference_mode(),torch.autocast(device_type=device.type,dtype=precision['dtype'],enabled=precision['autocast']):
        result=model(pixel_values=inputs)
    scores=result.logits.float().softmax(-1)[0,:,:-1].max(-1).values.cpu().numpy()
    boxes=result.pred_boxes.float()[0].cpu().numpy();rows=[]
    for index in sorted(range(len(scores)),key=lambda i:(-float(scores[i]),i)):
        if scores[index]<base['detr_confidence']:continue
        cx,cy,w,h=map(float,boxes[index]);box=np.clip([(cx-w/2)*2048,(cy-h/2)*2048,(cx+w/2)*2048,(cy+h/2)*2048],0,2048)
        if not np.isfinite(box).all() or box[2]<=box[0] or box[3]<=box[1]:continue
        rows.append(dict(box=list(map(float,box)),score=float(scores[index])))
        if len(rows)>=base['detr_max_det']:break
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    p.add_argument('--run-until-added',type=int,default=30);p.add_argument('--export-validation',action='store_true')
    args=p.parse_args();cfg=json.loads(Path(args.config).read_text())
    result=train_extension(cfg,args.run_until_added)
    if args.export_validation:export_extension_proposals(cfg,result['checkpoint'])
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
