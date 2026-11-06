"""DETR60 -> at most cumulative150, with durable per-epoch validation stopping.

The completed DETR60 and ROI30 runs are read-only. This new stage retains AdamW
moments but explicitly restarts a lower cosine LR schedule over 90 epochs.
Validation selects checkpoints and stops training; it never supplies gradients.
"""
from __future__ import annotations

import argparse
import fcntl
import gc
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import bbox_detr_continuation as previous
import filament_detr as legacy
from native_instance_data import load_cohort

SOURCE60_SHA = '97155a443d0c161d4306625dcf5eecb68d0c5cf50dfa69b11c645c548fb92819'
SOURCE60_IDENTITY = 'a18cf39e98c9afc38a3e4229cc53612360806069e8a69d1a4c7b3446d104d557'
DEFAULTS = dict(max_epoch=150, patience=10, min_delta=.001,
    continuation_lr=5e-6, continuation_backbone_lr=5e-7,
    continuation_lr_floor=.2, continuation_image_size=1280,
    continuation_batch=2, continuation_accumulate=2,
    continuation_precision='bf16', roi_eval_batch=8)
FIXED_MONITOR = dict(confidence_grid=[.6,.8,.9], mask_threshold=.5, min_area=300,
                     quality_threshold=.6, mask_nms=.5)


def normalize_config(config):
    cfg = dict(DEFAULTS, **config)
    source = Path(cfg['source_root']).resolve()
    destination = Path(cfg['output_root']).resolve()
    source_config = json.loads((source/'experiment_config.json').read_text())
    original = Path(source_config['base_root']).resolve()
    cache = Path(cfg.get('cache_root', '/content/solar_detr_longrun_cache')).resolve()
    for protected in (source, original):
        if destination == protected or destination in protected.parents or protected in destination.parents:
            raise ValueError('New output must be separate from both read-only source runs')
    if any(cache == p or cache in p.parents or p in cache.parents for p in (source, original, destination)):
        raise ValueError('Disposable cache must be separate from source and output')
    base = legacy.normalize_config(dict(json.loads((original/'experiment_config.json').read_text()),
                                        output_root=str(original)))
    cfg.update(source_root=str(source), output_root=str(destination),
               original_base_root=str(original), cache_root=str(cache))
    cfg['device'] = cfg.get('device', base['device'])
    cfg['strict'] = base['strict']
    for key in ('max_epoch', 'patience', 'continuation_image_size', 'continuation_batch',
                'continuation_accumulate', 'roi_eval_batch'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(key+' must be a positive integer')
    if cfg['continuation_image_size'] % 32:
        raise ValueError('Continuation image size must be a multiple of32')
    if (not math.isfinite(cfg['min_delta']) or cfg['min_delta'] <= 0
            or not 0 < cfg['continuation_backbone_lr'] <= cfg['continuation_lr'] < .1
            or not 0 < cfg['continuation_lr_floor'] <= 1):
        raise ValueError('Invalid learning rate or early-stopping delta')
    if cfg['continuation_precision'] not in ('bf16', 'fp16', 'fp32'):
        raise ValueError('Invalid continuation precision')
    if cfg['strict'] and (cfg['max_epoch'] != 150 or cfg['patience'] != 10 or cfg['min_delta'] != .001):
        raise ValueError('Reviewed production plan uses cumulative150, patience10, delta.001')
    cfg['detector_image_size'] = cfg['continuation_image_size']
    cfg['detector_precision'] = cfg['continuation_precision']
    cfg['_base_config'] = dict(base, device=cfg['device'], cache_root=cfg['cache_root'])
    cfg['_training_config'] = dict(cfg['_base_config'], detr_image_size=cfg['continuation_image_size'],
        detr_batch=cfg['continuation_batch'], detr_accumulate=cfg['continuation_accumulate'])
    return cfg


def _source_context(config):
    cfg = normalize_config(config)
    folder = Path(cfg['source_root'])/'detr_extension'
    doc = json.loads((folder/'training_identity.json').read_text())
    protocol = doc['protocol']
    if legacy.fingerprint(protocol) != doc['identity']:
        raise ValueError('Source protocol fingerprint is inconsistent')
    epoch = protocol['base_epoch']+protocol['added_epochs']
    compact = folder/f'epoch{epoch:03d}.pt'
    full = folder/'last_full.pt'
    compact_sha = legacy.digest(compact)
    if cfg['strict'] and (epoch != 60 or compact_sha != SOURCE60_SHA or doc['identity'] != SOURCE60_IDENTITY):
        raise ValueError('Source must be the reviewed completed DETR60 checkpoint')
    if cfg['max_epoch'] <= epoch:
        raise ValueError('max_epoch must exceed the completed source epoch')
    base = cfg['_base_config']; cohort = load_cohort(base)
    original_protocol = legacy.input_identity(base, cohort)
    runtime = original_protocol['runtime']
    original_protocol['runtime'] = protocol['source_protocol']['runtime']
    if original_protocol != protocol['source_protocol']:
        raise ValueError('Original source data, code, or settings changed')
    if protocol['extension_sources'] != dict(legacy.source_hashes(), bbox_detr_continuation=legacy.digest(previous.__file__)):
        raise ValueError('Frozen continuation source code changed')
    if (protocol['source']['metadata'] != original_protocol['metadata']
            or protocol['source']['images'] != original_protocol['images']
            or protocol['source']['config_sha256'] != legacy.digest(Path(cfg['original_base_root'])/'experiment_config.json')):
        raise ValueError('Source continuation provenance changed')
    first = previous.load_checkpoint(compact); state = previous.load_checkpoint(full)
    for checkpoint in (first, state):
        previous._validate_state(checkpoint, doc['identity'], protocol['base_epoch'], protocol['added_epochs'], protocol['added_epochs'])
    if (not previous._tensor_states_equal(first['model'], state['model'])
            or first['model_config'] != state['model_config']
            or first['base_history'] != state['base_history']
            or first['extension_history'] != state['extension_history']):
        raise ValueError('Source full and compact checkpoint disagree')
    if (not state.get('optimizer') or len(state['optimizer']['param_groups']) != 2
            or state.get('scaler') is None or not state.get('rng')
            or len(state['base_history']) != protocol['base_epoch']):
        raise ValueError('Source full optimizer, scaler, RNG and history are required')
    source_files = dict(legacy.source_hashes(), bbox_detr_continuation=legacy.digest(previous.__file__),
                        detr_longrun=legacy.digest(__file__))
    evaluator = Path(__file__).with_name('detr_longrun_eval.py')
    if evaluator.exists(): source_files['detr_longrun_eval'] = legacy.digest(evaluator)
    fixed = dict(stage='DETR-only, fixed ROI, per-epoch fixed-validation PQ stopping',
        base_epoch=epoch, max_epoch=cfg['max_epoch'], added_epochs=cfg['max_epoch']-epoch,
        schedule=dict(kind='explicit cosine restart retaining AdamW moments',
            learning_rates=[cfg['continuation_lr'],cfg['continuation_backbone_lr']],
            floor_fraction=cfg['continuation_lr_floor'], equivalent_to_uninterrupted150=False),
        training={key:cfg[key] for key in DEFAULTS if key.startswith('continuation_')},
        early_stopping=dict(metric='fixed validation PQ', patience=cfg['patience'], min_delta=cfg['min_delta'],
            significant_anchor='accumulating increase from last significant PQ',
            best_checkpoint='any strict raw PQ improvement, independent of stopping delta',
            baseline='source checkpoint re-evaluated before any training', monitor=FIXED_MONITOR,
            roi_eval_batch=cfg['roi_eval_batch']),
        source=dict(identity=doc['identity'], compact_sha256=compact_sha,
            full_state_sha256=legacy.digest(full), config_sha256=legacy.digest(Path(cfg['source_root'])/'experiment_config.json'),
            protocol=protocol), sources=source_files,
        runtime=runtime, source_runtime=protocol['runtime'],
        runtime_transition='new stage allowed; subsequent resume requires identical runtime',
        epoch_unit='one pass over all training annotation/rater records',
        training_annotation_records=len(cohort['train_groups']),
        validation_selects_checkpoint=True, validation_supplies_gradients=False,
        bbox_head_precision='fp32 before final sigmoid', bitwise_GPU_resume_guaranteed=False)
    del first
    return cfg, cohort, fixed, state


def monitor_update(control, epoch, pq, min_delta, patience):
    """Pure monitor; small gains can accumulate, while raw best always advances."""
    if not math.isfinite(pq) or not 0 <= pq <= 1:
        raise ValueError('Validation PQ must be finite and between0 and1')
    if control is None:
        return dict(best_epoch=epoch, best_pq=pq, significant_anchor_pq=pq,
                    significant_anchor_epoch=epoch, bad_epochs=0), True
    result = {key:control[key] for key in ('best_epoch','best_pq','significant_anchor_pq',
                                          'significant_anchor_epoch','bad_epochs')}
    is_best = pq > result['best_pq']
    if is_best: result.update(best_epoch=epoch, best_pq=pq)
    if pq-result['significant_anchor_pq'] >= min_delta-1e-12:
        result.update(significant_anchor_pq=pq, significant_anchor_epoch=epoch, bad_epochs=0)
    else: result['bad_epochs'] += 1
    return result, is_best


def _recover_best(root, control):
    """Recover interruption after control commit but before pending-best rename."""
    best = root/'best.pt'; pending = root/'best_pending.pt'
    expected = control['best_checkpoint_sha256']
    if best.exists() and legacy.digest(best) == expected:
        pending.unlink(missing_ok=True)
        return
    if pending.exists() and legacy.digest(pending) == expected:
        pending.replace(best); return
    raise ValueError('Committed best checkpoint is missing or changed; refusing to reset selection')


def _validate_state(state, run_id, fixed):
    epoch = state.get('completed_epoch', -1)
    count = epoch-fixed['base_epoch']
    if (state.get('identity') != run_id or state.get('base_epoch') != fixed['base_epoch']
            or state.get('fixed_final_epoch') != fixed['max_epoch']
            or not 0 <= count <= fixed['added_epochs']
            or len(state.get('training_history', [])) != count
            or [x['epoch'] for x in state['training_history']] != list(range(fixed['base_epoch']+1,epoch+1))
            or not state.get('optimizer') or state.get('scheduler') is None
            or state.get('scaler') is None or not state.get('rng')):
        raise ValueError('Wrong long-run checkpoint identity, epoch or optimizer state')


def _validate_control(control, state, fixed, checkpoint_sha):
    if control is None:
        if state['completed_epoch'] != fixed['base_epoch']:
            raise ValueError('Missing control state after training; refusing to reset patience')
        return
    if control.get('identity') != state['identity']:
        raise ValueError('Wrong validation control identity')
    epoch = control.get('last_validated_epoch',-1)
    history = control.get('history', [])
    if (epoch not in (state['completed_epoch'],state['completed_epoch']-1)
            or [row['epoch'] for row in history] != list(range(fixed['base_epoch'],epoch+1))):
        raise ValueError('Validation control is ahead of checkpoint or has a gap')
    derived = None
    for row in history:
        derived,_ = monitor_update(derived,row['epoch'],row['score']['pq'],
                                    fixed['early_stopping']['min_delta'],fixed['early_stopping']['patience'])
    if derived is None or any(control.get(key) != value for key,value in derived.items()):
        raise ValueError('Saved patience/best history is inconsistent; refusing to reset it')
    if epoch == state['completed_epoch'] and control['last_validated_checkpoint_sha256'] != checkpoint_sha:
        raise ValueError('Validation belongs to different checkpoint weights')


def _report(cfg, fixed, control):
    root=Path(cfg['output_root']); epoch=control['last_validated_epoch']
    status = ('early_stopped' if control['bad_epochs'] >= cfg['patience'] else
              'complete' if epoch >= cfg['max_epoch'] else 'paused')
    return dict(status=status,identity=control['identity'],completed_epoch=epoch,
        cumulative_epochs_completed=epoch,base_epoch=fixed['base_epoch'],max_epoch=cfg['max_epoch'],
        added_epochs_completed=epoch-fixed['base_epoch'],best_epoch=control['best_epoch'],
        best_pq=control['best_pq'],baseline_pq=control['history'][0]['score']['pq'],
        significant_anchor_pq=control['significant_anchor_pq'],bad_epochs=control['bad_epochs'],
        patience=cfg['patience'],min_delta=cfg['min_delta'],
        best_checkpoint=str(root/'best.pt'),best_checkpoint_sha256=control['best_checkpoint_sha256'],
        best_selected_confidence=control.get('best_selected_confidence'),
        current_checkpoint=str(root/'last_full.pt'),
        current_checkpoint_sha256=control['last_validated_checkpoint_sha256'],
        best_evaluation_dir=control.get('best_evaluation_dir'),
        current_evaluation_dir=control.get('current_evaluation_dir'),
        history_path=str(root/'history.json'),validation_is_development_set=True,
        bitwise_GPU_resume_guaranteed=False)


def _optimizer_to(optimizer, device):
    for values in optimizer.state.values():
        for key,value in values.items():
            if isinstance(value,torch.Tensor) and key != 'step': values[key]=value.to(device)


def _train_epoch(model, optimizer, scheduler, scaler, dataset, cfg, epoch, precision, device):
    base=cfg['_base_config']; training=cfg['_training_config']; zero_epoch=epoch-1
    seed=base['seed']+10007*zero_epoch;legacy.seed_everything(seed);dataset.set_epoch(zero_epoch)
    loader=DataLoader(dataset,batch_size=training['detr_batch'],shuffle=True,num_workers=0,
        collate_fn=legacy.collate,generator=torch.Generator().manual_seed(seed))
    model.train();optimizer.zero_grad(set_to_none=True);totals={};batches=attempts=updates=0
    for index,(pixels,targets) in enumerate(loader):
        pixels=pixels.to(device);targets=[{key:value.to(device) for key,value in t.items()} for t in targets]
        start=(index//training['detr_accumulate'])*training['detr_accumulate']
        window=min(training['detr_accumulate'],len(loader)-start)
        with torch.autocast(device_type=device.type,dtype=precision['dtype'],enabled=precision['autocast']):
            output=model(pixel_values=pixels,labels=targets);loss=output.loss
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss; retry from completed epoch')
        scaler.scale(loss/window).backward();batches+=1
        for key,value in dict(output.loss_dict,total_loss=loss).items():
            totals[key]=totals.get(key,0.)+float(value.detach().cpu())
        if (index+1)%training['detr_accumulate']==0 or index+1==len(loader):
            scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),base['detr_clip_norm'])
            if not precision['scaler'] and not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradients')
            before=scaler.get_scale();scaler.step(optimizer);scaler.update()
            attempts+=1;updates+=int(scaler.get_scale()>=before);optimizer.zero_grad(set_to_none=True)
        if (index+1)%50==0 or index+1==len(loader):
            print(f'DETR epoch {epoch}/{cfg["max_epoch"]} batch {index+1}/{len(loader)} loss {totals["total_loss"]/batches:.5f}',flush=True)
    if batches != len(loader) or not batches: raise RuntimeError('An incomplete epoch cannot be saved')
    scheduler.step()
    return dict(epoch=epoch,losses={key:value/batches for key,value in totals.items()},
        annotation_records=len(dataset),batches=batches,optimizer_attempts=attempts,
        optimizer_updates=updates,amp_skipped_updates=attempts-updates,next_lr=scheduler.get_last_lr())


def run(config, run_until=150, evaluate_callback=None):
    """Train to an absolute pause cap; terminal runs never reset or train again.

    evaluate_callback is a test/integration seam: (cfg,path,epoch)->report with
    score.pq and checkpoint_sha256. Production always uses the fixed evaluator.
    """
    cfg=normalize_config(config);root=Path(cfg['output_root']);root.mkdir(parents=True,exist_ok=True)
    lockpath=Path('/tmp')/('detr_longrun_'+legacy.fingerprint(str(root))[:16]+'.lock')
    with lockpath.open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Another cell is running this DETR study')
        return _run_locked(config,run_until,evaluate_callback)


def _run_locked(config, run_until, evaluate_callback):
    cfg,cohort,fixed,source=_source_context(config)
    if type(run_until) is not int or not fixed['base_epoch'] <= run_until <= cfg['max_epoch']:
        raise ValueError('run_until is an absolute epoch between source epoch and max_epoch')
    root=Path(cfg['output_root']);run_id=legacy.fingerprint(fixed)
    identity_path=root/'training_identity.json';last=root/'last_full.pt';control_path=root/'control_state.json'
    document=dict(identity=run_id,protocol=fixed)
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != document:
            raise ValueError('Source, data, runtime, code or schedule changed; use a new output root')
    elif any(p.exists() for p in (last,control_path,root/'best.pt',root/'best_pending.pt')):
        raise ValueError('Unidentified previous run; refusing to reset it')
    legacy.atomic_json(identity_path,document)
    # The Colab launcher authenticates the original config bytes for runtime
    # reset/resume. Never rewrite that bootstrap-owned file while normalizing.
    legacy.atomic_json(root/'resolved_config.json',{key:value for key,value in cfg.items() if not key.startswith('_')})
    state=previous.load_checkpoint(last) if last.exists() else None
    control=json.loads(control_path.read_text()) if control_path.exists() else None
    if state is not None:
        _validate_state(state,run_id,fixed);_validate_control(control,state,fixed,legacy.digest(last))
        if control is not None:_recover_best(root,control)
    elif control is not None:raise ValueError('Missing training checkpoint; cannot restart validation history')
    evaluator=None;context=None
    if evaluate_callback is None:
        import detr_longrun_eval as evaluator
        context=evaluator.prepare_context(cfg)
        evaluate_callback=lambda config,path,epoch:evaluator.evaluate_checkpoint(config,path,epoch,context=context)
    elif cfg['strict']:
        raise ValueError('Production runs require the fixed validation evaluator')

    def finish():
        report=_report(cfg,fixed,control)
        legacy.atomic_json(root/'history.json',control['history'])
        legacy.atomic_json(root/'result.json',report)
        if evaluator is not None:
            evaluator.retain_evaluations(cfg,[control.get('best_evaluation_dir'),control.get('current_evaluation_dir')])
            exported=evaluator.export_results(cfg,[control['best_evaluation_dir'],control['current_evaluation_dir']])
            if exported is not None:
                report['exports']=exported
                legacy.atomic_json(root/'latest_results.json',dict(report,archives=exported))
        return report

    if control is not None and control['last_validated_epoch']==state['completed_epoch']:
        if control['bad_epochs']>=cfg['patience'] or state['completed_epoch']>=cfg['max_epoch']:
            return finish()
        if state['completed_epoch']>=run_until:return finish()
    device=legacy.device_for(cfg['device']);torch.backends.cudnn.benchmark=False
    precision=previous._precision(cfg,device);base=cfg['_base_config']
    legacy.seed_everything(base['seed'])
    model=previous.install_fp32_bbox_head(legacy.build_detector(base,pretrained=False,model_config=source['model_config']))
    regular=[];backbone=[]
    for name,param in model.named_parameters():
        if param.requires_grad:(backbone if 'backbone' in name else regular).append(param)
    rates=[cfg['continuation_lr'],cfg['continuation_backbone_lr']]
    optimizer=torch.optim.AdamW([dict(params=regular,lr=rates[0]),dict(params=backbone,lr=rates[1])],
                                weight_decay=base['detr_weight_decay'])
    initial=state if state is not None else source
    model.load_state_dict(initial['model']);optimizer.load_state_dict(initial['optimizer'])
    for group,rate in zip(optimizer.param_groups,rates):group.update(lr=rate,initial_lr=rate)
    schedule=lambda e:cfg['continuation_lr_floor']+(1-cfg['continuation_lr_floor'])*(1+math.cos(math.pi*e/fixed['added_epochs']))/2
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    scaler=torch.amp.GradScaler('cuda',enabled=precision['scaler'])
    if state is not None:
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
    elif precision['scaler'] and source['scaler']:scaler.load_state_dict(source['scaler'])
    legacy.restore_rng(initial['rng'])
    if state is None:
        state=dict(schema_version=1,identity=run_id,base_epoch=fixed['base_epoch'],
            completed_epoch=fixed['base_epoch'],fixed_final_epoch=cfg['max_epoch'],
            source_history=source['base_history']+source['extension_history'],training_history=[],
            model_config=model.config.to_dict(),model=legacy.cpu_copy(model.state_dict()),
            optimizer=legacy.cpu_copy(optimizer.state_dict()),scheduler=scheduler.state_dict(),
            scaler=scaler.state_dict(),rng=legacy.rng_state())
        legacy.atomic_torch(last,state)
    del initial,source;gc.collect()
    dataset=legacy.DetectionDataset(cohort['train_groups'],base['images'],cfg['_training_config'],
                                   fixed['source']['protocol']['source']['images'])
    while True:
        epoch=state['completed_epoch']
        if control is None or control['last_validated_epoch']<epoch:
            # Training is already durable. A validation interruption replays this
            # checkpoint first, without another optimizer step or a patience reset.
            model.cpu();_optimizer_to(optimizer,torch.device('cpu'));gc.collect()
            if device.type=='cuda':torch.cuda.empty_cache()
            checkpoint_sha=legacy.digest(last)
            evaluated=evaluate_callback(cfg,last,epoch)
            legacy.restore_rng(state['rng'])
            if evaluated.get('checkpoint_sha256')!=checkpoint_sha:
                raise ValueError('Evaluation checkpoint SHA does not match pending epoch')
            if legacy.digest(last)!=checkpoint_sha:raise ValueError('Checkpoint changed during validation')
            scores=dict(evaluated['score']);pq=float(scores['pq'])
            monitor,is_best=monitor_update(control,epoch,pq,cfg['min_delta'],cfg['patience'])
            record=dict(epoch=epoch,score=scores,checkpoint_sha256=checkpoint_sha,
                selected_confidence=evaluated.get('selected_confidence'),
                fixed_08_score=evaluated.get('fixed_08_score'),
                training=None if epoch==fixed['base_epoch'] else state['training_history'][-1],
                significant_anchor_pq=monitor['significant_anchor_pq'],bad_epochs=monitor['bad_epochs'])
            next_control=dict(identity=run_id,**monitor,last_validated_epoch=epoch,
                last_validated_checkpoint_sha256=checkpoint_sha,
                history=([] if control is None else control['history'])+[record],
                current_evaluation_dir=evaluated.get('evaluation_dir'),
                best_selected_confidence=evaluated.get('selected_confidence') if is_best else control.get('best_selected_confidence'),
                best_evaluation_dir=evaluated.get('evaluation_dir') if is_best else control.get('best_evaluation_dir'))
            if is_best:
                pending=root/'best_pending.pt'
                selected=dict(legacy._compact(state),validation_selection=dict(
                    epoch=epoch,score=scores,selected_confidence=evaluated.get('selected_confidence'),
                    fixed_08_score=evaluated.get('fixed_08_score'),monitor=FIXED_MONITOR,
                    evaluated_full_checkpoint_sha256=checkpoint_sha))
                legacy.atomic_torch(pending,selected)
                next_control['best_checkpoint_sha256']=legacy.digest(pending)
            else:next_control['best_checkpoint_sha256']=control['best_checkpoint_sha256']
            # Commit controller BEFORE promoting pending-best and pruning exports.
            # Both pre- and post-commit crashes have a recoverable best checkpoint.
            legacy.atomic_json(control_path,next_control);control=next_control
            _recover_best(root,control)
            legacy.atomic_json(root/'history.json',control['history'])
            if evaluator is not None:
                evaluator.retain_evaluations(cfg,[control.get('best_evaluation_dir'),control.get('current_evaluation_dir')])
            print(f'Validation epoch {epoch}: PQ {pq:.6f}, best {control["best_pq"]:.6f} at {control["best_epoch"]}, no significant improvement {control["bad_epochs"]}/{cfg["patience"]}',flush=True)
        if control['bad_epochs']>=cfg['patience'] or epoch>=cfg['max_epoch'] or epoch>=run_until:
            del model,optimizer;gc.collect()
            if device.type=='cuda':torch.cuda.empty_cache()
            return finish()
        model.to(device);_optimizer_to(optimizer,device)
        entry=_train_epoch(model,optimizer,scheduler,scaler,dataset,cfg,epoch+1,precision,device)
        state=dict(state,completed_epoch=epoch+1,training_history=state['training_history']+[entry],
            model=legacy.cpu_copy(model.state_dict()),optimizer=legacy.cpu_copy(optimizer.state_dict()),
            scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),rng=legacy.rng_state())
        legacy.atomic_torch(last,state)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--run-until',type=int,default=150)
    args=parser.parse_args()
    result=run(json.loads(Path(args.config).read_text()),args.run_until)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
