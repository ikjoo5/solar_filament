"""Full-image-coverage native training with verified epoch and tile-cursor resume.

The full-pass dataset enumerates each physical image's four 1024 tiles once per
epoch. It supplies the shuffle; this engine never samples a replacement crop or
limits the number of batches. General ImageNet weights initialize a new solar
run, while only this run's verified checkpoint may initialize a continuation.
"""
from __future__ import annotations

from collections import Counter
import gc
import json
import math
from pathlib import Path
import random
import time
import zipfile

import numpy as np
import torch
from torch.utils.data import Subset

import swefil_mixed_training as previous

base = previous.base
original = previous.original
atomic_json = previous.atomic_json
digest = previous.digest
VERSION = 'native_all_images_four_tiles_v1'


def _load_remote(store, record, identity):
    return previous._load_remote(store, record, identity)


def _config_guard(saved, current):
    """A time-limited invocation changes the cap, never the training recipe."""
    keys = ('encoder', 'pretrained', 'crop', 'batch', 'accumulate', 'epochs', 'lr',
            'weight_decay', 'minimum_lr_ratio', 'clip_grad_norm', 'spine_weight',
            'seed', 'workers', 'amp_dtype', 'thresholds', 'min_areas', 'overlap',
            'validate_every', 'preprocessing', 'samples_per_epoch')
    changed = [key for key in keys if saved.get(key) != current.get(key)]
    if changed:
        raise ValueError('Full-pass resume settings changed: '+', '.join(changed))


def publish_results(output, state, cfg, identity, prepared=None):
    """Keep each model's artifacts separate; the launcher aggregates both."""
    output = Path(output)
    result = dict(status='complete' if state['epoch'] == cfg['epochs'] else 'paused',
        model='native_unet', epochs_completed=state['epoch'], epochs_target=cfg['epochs'],
        best_epoch=state['best_epoch'], best_development_pq=state['best_pq'], identity=identity,
        initialization='ImageNet encoder and new U-Net decoder; no solar warm start',
        scope=cfg.get('validation_scope', 'Fixed development validation, not public LB.'),
        samples_per_epoch=cfg['samples_per_epoch'], batch_size=cfg['batch'],
        optimizer_update_attempts_per_epoch=math.ceil(math.ceil(cfg['samples_per_epoch']/cfg['batch'])/cfg['accumulate']),
        coverage='Every physical training image, four nonoverlapping 1024 tiles, once per epoch.',
        last_checkpoint_record=state['last'], best_checkpoint_record=state['best'],
        last_checkpoint_url='https://drive.google.com/file/d/'+state['last']['file_id']+'/view',
        best_checkpoint_url='https://drive.google.com/file/d/'+state['best']['file_id']+'/view',
        persistence='Verified Drive API completed epochs and optimizer-boundary progress checkpoints.',
        resume_interval_optimizer_updates=cfg.get('checkpoint_interval_updates', 100),
        checkpoint_weights_in_zip=False, output_root=str(output))
    atomic_json(output/'summary.json', result)
    archive = output/'fullpass_native_results.zip'
    pending = archive.with_suffix('.pending.zip')
    with zipfile.ZipFile(pending, 'w', zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(output.iterdir()):
            if path.is_file() and path.suffix in ('.json', '.csv'):
                zipped.write(path, 'unet/'+path.name)
        zipped.write(Path(__file__), 'code/'+Path(__file__).name)
        if prepared is not None:
            zipped.writestr('prepared_summary.json', json.dumps(previous.serial(prepared), indent=2))
    pending.replace(archive)
    result.update(archive=str(archive), archive_sha256=digest(archive))
    atomic_json(output/'latest_result.json', result)
    return result


def _validate_completed(checkpoint, state, cfg):
    if (checkpoint['epoch'] != state['epoch'] or len(checkpoint['history']) != state['epoch']
            or checkpoint['history'][-1]['epoch'] != state['epoch']
            or checkpoint['best_epoch'] != state['best_epoch']
            or checkpoint['best_pq'] != state['best_pq']):
        raise ValueError('Completed checkpoint and committed epoch metadata differ')
    _config_guard(checkpoint['config'], cfg)


def run_native(cfg, dataset, validator, output, identity, checkpoint_store,
               run_until=None, *, model_factory=None, prepared=None):
    """Train all tiles, with CPU-testable factories and optional verified progress.

    Progress saves occur only after an optimizer boundary, so no half-accumulated
    gradients are discarded. A resumed dataset starts directly at the next tile;
    earlier samples are not fetched again. Workers stay at zero so all RNG state
    affecting the run is captured by the checkpoint.
    """
    cfg = dict(cfg)
    count = len(dataset)
    if count < 1 or cfg.get('samples_per_epoch', count) != count:
        raise ValueError('Full-pass sample count must equal the complete dataset length')
    cfg['samples_per_epoch'] = count
    if cfg.get('initial_checkpoint') or cfg.get('resume_dir'):
        raise ValueError('No solar checkpoint warm start; only this run resumes automatically')
    if int(cfg.get('workers', 0)) != 0:
        raise ValueError('Full-pass exact tile-cursor resume requires workers=0')
    if min(int(cfg['batch']), int(cfg['accumulate'])) < 1:
        raise ValueError('Batch and accumulation must be positive')
    target = cfg['epochs'] if run_until is None else int(run_until)
    if not 1 <= target <= cfg['epochs']:
        raise ValueError('run-until must be within the unchanged full training schedule')
    interval = int(cfg.get('checkpoint_interval_updates', 100))
    if interval < 0:
        raise ValueError('checkpoint_interval_updates cannot be negative')
    if interval and not all(hasattr(checkpoint_store, name) for name in ('latest_progress', 'publish_progress', 'clear_progress')):
        raise TypeError('Tile-cursor resume requires the full-pass verified checkpoint store')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg['device'])
    amp = torch.float16 if device.type == 'cuda' else None
    state = checkpoint_store.latest_state()
    progress = checkpoint_store.latest_progress() if interval else None
    if state is None and progress is None and (output/'state.json').exists():
        raise RuntimeError('Local state exists but verified remote commits are missing; refusing fresh restart')
    completed = int(state['epoch']) if state else 0
    if state and completed >= target:
        checkpoint = _load_remote(checkpoint_store, state['last'], identity)
        _validate_completed(checkpoint, state, cfg)
        best = _load_remote(checkpoint_store, state['best'], identity)
        if best['epoch'] != state['best_epoch']:
            raise ValueError('Completed best checkpoint epoch differs')
        base.export_best_predictions(output, best)
        atomic_json(output/'history.json', checkpoint['history'])
        original.write_envelope(output/'state.json', identity, state)
        print(f'Native epoch {completed} already completed; no repeated training.', flush=True)
        return publish_results(output, state, cfg, identity, prepared)
    if progress and int(progress['epoch']) <= completed:
        checkpoint_store.clear_progress(completed_epoch=completed)
        progress = None
    if progress and (int(progress['epoch']) != completed+1 or int(progress['base_epoch']) != completed):
        raise ValueError('Progress cursor does not follow the latest completed epoch')
    checkpoint_store.preflight(required_free_bytes=int(cfg.get('checkpoint_free_bytes', 2_000_000_000)))
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg['seed'])
    factory = model_factory or (lambda fresh: base.build_model(cfg['encoder'], pretrained=fresh))
    model = factory(state is None and progress is None).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'], eta_min=cfg['lr']*cfg['minimum_lr_ratio'])
    scaler = torch.amp.GradScaler('cuda', enabled=amp is not None)
    history, best_pq, best_epoch, best_record = [], -1., 0, None
    resume_step, resume_losses, resume_attempts, resume_successes, resume_elapsed = 0, [], 0, 0, 0.
    checkpoint = None
    if progress:
        checkpoint = _load_remote(checkpoint_store, progress['checkpoint'], identity)
        if (checkpoint.get('checkpoint_kind') != 'progress' or checkpoint['epoch'] != completed+1
                or checkpoint['step'] != int(progress['step']) or checkpoint['base_epoch'] != completed):
            raise ValueError('Progress checkpoint and remote cursor differ')
        _config_guard(checkpoint['config'], cfg)
        if len(checkpoint['history']) != completed:
            raise ValueError('Progress history does not match completed epochs')
        total_batches = math.ceil(count/cfg['batch'])
        resume_step = int(checkpoint['step'])
        if not 0 < resume_step < total_batches or resume_step % cfg['accumulate']:
            raise ValueError('Progress must be an optimizer boundary within the epoch')
        resume_losses = checkpoint['losses']
        resume_attempts, resume_successes = checkpoint['attempts'], checkpoint['successes']
        resume_elapsed = float(checkpoint['elapsed_seconds'])
        if len(resume_losses) != resume_step or resume_attempts != resume_step//cfg['accumulate']:
            raise ValueError('Progress losses/update counts do not match its tile cursor')
        best_record = checkpoint['best_record']
        if state and (checkpoint['best_epoch'] != state['best_epoch'] or checkpoint['best_pq'] != state['best_pq'] or best_record != state['best']):
            raise ValueError('Progress best checkpoint differs from committed epoch')
    elif state:
        checkpoint = _load_remote(checkpoint_store, state['last'], identity)
        _validate_completed(checkpoint, state, cfg)
        best_record = state['best']
    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        history = checkpoint['history']
        best_pq, best_epoch = float(checkpoint['best_pq']), int(checkpoint['best_epoch'])
        if best_record:
            best = _load_remote(checkpoint_store, best_record, identity)
            if best['epoch'] != best_epoch:
                raise ValueError('Restored best epoch differs')
            base.export_best_predictions(output, best)
            del best
        base.restore_rng(checkpoint['rng_state'])
        del checkpoint
    atomic_json(output/'config.json', cfg)
    local = Path(cfg['cache_root'])/'checkpoint_staging'/identity[:16]
    if local.resolve().is_relative_to('/content/drive'):
        raise ValueError('Serialize checkpoints on local disk, not Drive FUSE')
    local.mkdir(parents=True, exist_ok=True)
    print(f'Native full pass: {count} tiles/epoch; completed epoch {completed}; '
          f'next epoch {completed+1}, completed batches {resume_step}; target {target}/{cfg["epochs"]}.', flush=True)

    def training_payload(epoch):
        return dict(schema_version=1, identity=identity, model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(), scheduler_state_dict=scheduler.state_dict(),
            scaler_state_dict=scaler.state_dict(), config=cfg.copy(), epoch=epoch, history=history,
            best_pq=best_pq, best_epoch=best_epoch, rng_state=base.capture_rng())

    for epoch in range(completed+1, target+1):
        dataset.epoch = epoch
        total_batches = math.ceil(count/cfg['batch'])
        start_step = resume_step if epoch == completed+1 else 0
        remaining = Subset(dataset, range(start_step*cfg['batch'], count))
        remaining.epoch = epoch
        loader = base.make_epoch_loader(remaining, cfg, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = list(resume_losses) if start_step else []
        attempts, successes = (resume_attempts, resume_successes) if start_step else (0, 0)
        elapsed_before = resume_elapsed if start_step else 0.
        started = time.monotonic()
        domains = Counter(dataset.sample_record(i)['domain'] for i in range(count)) if hasattr(dataset, 'sample_record') else {}
        for offset, (x, y, valid) in enumerate(loader):
            step = start_step+offset
            x, y, valid = (value.to(device, non_blocking=True) for value in (x, y, valid))
            group_start = (step//cfg['accumulate'])*cfg['accumulate']
            divisor = min(cfg['accumulate'], total_batches-group_start)
            with torch.autocast(device_type=device.type, dtype=amp or torch.float32, enabled=amp is not None):
                loss = base.mixed_segmentation_loss(model(x), y, valid, cfg['spine_weight'])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss; last verified progress checkpoint remains resumable')
            scaler.scale(loss/divisor).backward()
            losses.append(float(loss.detach()))
            boundary = (step+1) % cfg['accumulate'] == 0 or step+1 == total_batches
            if boundary:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_grad_norm'])
                scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                successes += int(scaler.get_scale() >= scale)
                attempts += 1
                optimizer.zero_grad(set_to_none=True)
                if interval and attempts % interval == 0 and step+1 < total_batches:
                    payload = training_payload(epoch)
                    payload.update(checkpoint_kind='progress', base_epoch=epoch-1, step=step+1,
                        losses=losses, attempts=attempts, successes=successes, best_record=best_record,
                        elapsed_seconds=elapsed_before+time.monotonic()-started)
                    path = local/'progress.pt'
                    original.save_checkpoint(path, payload)
                    checkpoint_store.publish_progress(path, metadata=dict(epoch=epoch, step=step+1, base_epoch=epoch-1))
                    print(f'Native progress verified: epoch {epoch}, batch {step+1}/{total_batches}.', flush=True)
                    del payload
            if (step+1) % 25 == 0 or step+1 == total_batches:
                print(f'Native epoch {epoch}/{cfg["epochs"]} tile batch {step+1}/{total_batches} '
                      f'loss {losses[-1]:.5f}', flush=True)
        if len(losses) != total_batches:
            raise RuntimeError('An incomplete tile pass cannot commit as a completed epoch')
        if successes == 0:
            raise FloatingPointError('All optimizer updates were skipped; refusing to commit an untrained epoch')
        row = dict(epoch=epoch, train_loss=float(np.mean(losses)), lr=float(optimizer.param_groups[0]['lr']),
            microbatches=total_batches, tiles_seen=count, optimizer_updates=successes,
            optimizer_update_attempts=attempts, amp_skipped_updates=attempts-successes,
            domain_tiles=dict(domains), coverage='each physical image tile exactly once',
            elapsed_seconds=elapsed_before+time.monotonic()-started)
        scheduler.step()
        new_best = None
        if epoch == 1 or epoch % cfg['validate_every'] == 0 or epoch == cfg['epochs']:
            best, grid, predictions = validator(model, device, amp, cfg)
            row.update(validation=best, validation_grid=grid)
            fixed = [item for item in grid if item['threshold'] == .5 and item['min_area'] == 100]
            if fixed:
                row['fixed_limb_threshold05_area100'] = fixed[0]
            if best['pq'] > best_pq:
                best_pq, best_epoch = float(best['pq']), epoch
                new_best = dict(schema_version=1, identity=identity, model_state_dict=model.state_dict(),
                    config=cfg.copy(), epoch=epoch, validation=best, validation_grid=grid,
                    validation_predictions=predictions)
                path = local/'best.pt'
                original.save_checkpoint(path, new_best)
                best_record = checkpoint_store.publish_checkpoint(path, f'checkpoints/best_epoch{epoch:03}.pt',
                    metadata=dict(epoch=epoch, kind='best'))
        history.append(row)
        payload = training_payload(epoch)
        path = local/'last.pt'
        original.save_checkpoint(path, payload)
        last_record = checkpoint_store.publish_checkpoint(path, f'checkpoints/last_epoch{epoch:03}.pt',
            metadata=dict(epoch=epoch, kind='last'))
        next_state = dict(epoch=epoch, last=last_record, best=best_record, best_pq=best_pq, best_epoch=best_epoch)
        state = checkpoint_store.commit_state(next_state, expected_previous_epoch=epoch-1)
        if interval:
            checkpoint_store.clear_progress(completed_epoch=epoch)
        original.write_envelope(output/'state.json', identity, state)
        if new_best is not None:
            base.export_best_predictions(output, new_best)
        atomic_json(output/'history.json', history)
        print(f'Native completed epoch {epoch} confirmed on Drive: {last_record["file_id"]}', flush=True)
        print(json.dumps(row), flush=True)
        publish_results(output, state, cfg, identity, prepared)
        del payload, new_best, loader
        gc.collect()
    if state is None:
        raise RuntimeError('No completed epoch state')
    result = publish_results(output, state, cfg, identity, prepared)
    del model, optimizer, scheduler, scaler
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result
