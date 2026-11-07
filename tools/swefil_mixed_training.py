"""Fresh best84-recipe native training: GONG + all annotated SWEFil + CHASE shapes.

One 100-epoch run, never a continuation of the lost CHASE checkpoint. The old
training sources are imported unchanged. Remote, verified checkpoint commits are
the sole resume authority; Drive-mounted JSON files are human-readable mirrors.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import gc
import json
import os
from pathlib import Path
import random
import time
import zipfile

os.environ.setdefault('TORCH_CUDNN_V8_API_DISABLED', '1')
import cv2
import numpy as np
import torch

import chase_shape_training as original
from chase_shape_training import atomic_json, digest, fingerprint, serial

base = original.base
VERSION = 'swefil_all_gong_chase_native_v1'
DEFAULTS = dict(original.DEFAULTS, synthetic_ratio=.1, swefil_ratio=.6)
SOURCE_FILES = tuple(dict.fromkeys((
    'swefil_mixed_training.py', 'swefil_training_data.py',
    'verified_drive_checkpoints.py', 'chase_shape_cache.py',
) + original.SOURCE_FILES))


def normalize_config(value):
    """Only paths and execution caps vary; an epoch cap never changes cosine T_max."""
    for key, expected in DEFAULTS.items():
        if key in value and value[key] != expected:
            raise ValueError('Fixed best84 training setting changed: '+key)
    if value.get('initial_checkpoint') or value.get('resume_dir'):
        raise ValueError('Start from ImageNet/new decoder; only this new run resumes automatically.')
    project = Path(value['base']).expanduser().resolve()
    previous = Path(value.get('chase_training_root', project/'fixed_split_experiments/chase_shape_best84_recipe_t4_v1')).resolve()
    old = json.loads((previous/'experiment_config.json').read_text())
    # Authenticate the old recipe without changing any of its files or identity.
    original.normalize_config(old)
    root = Path(value['output_root']).expanduser().resolve()
    cache = Path(value.get('cache_root', '/content/swefil_all_mixed_native_t4_cache')).resolve()
    if root == previous or root.is_relative_to(previous) or previous.is_relative_to(root):
        raise ValueError('The SWEFil run must use a separate output directory.')
    if cache.is_relative_to('/content/drive') or cache.is_relative_to(root) or root.is_relative_to(cache):
        raise ValueError('Use a separate local directory for caches/checkpoint staging.')
    cfg = dict(old, **DEFAULTS)
    cfg.update(base=str(project), chase_training_root=str(previous), output_root=str(root),
        cache_root=str(cache), samples_per_epoch=400, effective_batch=4, updates_per_epoch=100,
        sampling='400 image-first random crops per epoch; not a full-image pass.',
        comparison='Fresh best84 architecture and optimizer; user-selected GONG 30%, SWEFil 60%, CHASE shape 10% during mixed epochs.',
        loss='Foreground BCE+Dice; 0.1 genuine GONG spine BCE. No spine supervision for SWEFil/CHASE.',
        external_pool='All 250 annotated SWEFil observations; processed version once; no date/overlap exclusion.',
        validation_scope='Fixed 91-image development set. External temporal overlap is reported, not excluded.',
        drive_checkpoint_path_parts=value.get('drive_checkpoint_path_parts', [project.name,
            'fixed_split_experiments', root.name, 'verified_checkpoints']))
    return cfg


def swefil_coco(records):
    """Preserve every filament instance/RLE, while sunspots remain background."""
    images, annotations = [], []
    seen = set()
    for index, record in enumerate(records):
        source_name = Path(record['file_name'])
        name = source_name.name
        if name in seen or '..' in source_name.parts:
            raise ValueError('Duplicate/unsafe SWEFil physical image name')
        seen.add(name)
        if record.get('spine_valid', False):
            raise ValueError('SWEFil has no genuine spine supervision')
        group = 'swefil_'+str(index)
        images.append(dict(id=group, file_name=name, width=record['width'], height=record['height']))
        for ann in record['annotations']:
            category = int(ann['category_id'])
            if category == 3:
                continue
            if category not in (0, 1, 2):
                raise ValueError('Unknown SWEFil category')
            annotations.append(dict(ann, id='swefil_'+str(len(annotations)), image_id=group))
    return dict(images=images, annotations=annotations)


class ThreeDomainCropDataset(original.RLEMixedCropDataset):
    """Original crop/augmentation/loss contract with an explicit third domain."""
    def __init__(self, data, train_ids, real_store, synthetic_data, synthetic_store,
                 swefil_data, swefil_store, *, swefil_ratio=.6, **kwargs):
        kwargs.setdefault('synthetic_ratio', DEFAULTS['synthetic_ratio'])
        super().__init__(data, train_ids, real_store, synthetic_data, synthetic_store, **kwargs)
        if not 0 <= swefil_ratio <= 1 or self.synthetic_ratio+swefil_ratio > 1:
            raise ValueError('Invalid three-domain probabilities')
        self.swefil_ratio = float(swefil_ratio)
        # Here `synthetic=True` is the inherited RLE/no-spine target adapter.
        # SWEFil remains real external observations in all user-facing records.
        self.domains['swefil'] = self._domain(swefil_data, None, True, False)
        self.stores['swefil'] = swefil_store
        domain = self.domains['swefil']
        self.domain_info['swefil'] = dict(physical_images=len(domain['files']),
            annotation_groups=len(domain['images']), annotation_instances=sum(map(len, domain['annotations'].values())),
            excluded_blank_groups=0, small_area=domain['small_area'], source='external real observations', spine_valid=False)

    def domain_probabilities(self, epoch=None):
        epoch = self.epoch if epoch is None else int(epoch)
        mixed = self.warmup_epochs < epoch <= self.total_epochs-self.real_finish_epochs
        return dict(real=round(1-self.synthetic_ratio-self.swefil_ratio,12) if mixed else 1.,
                    swefil=self.swefil_ratio if mixed else 0., synthetic=self.synthetic_ratio if mixed else 0.)

    def sample_record(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, int(index)]))
        probabilities = self.domain_probabilities()
        draw = rng.random()
        name = ('real' if draw < probabilities['real'] else
                'swefil' if draw < probabilities['real']+probabilities['swefil'] else 'synthetic')
        domain = self.domains[name]
        filename = domain['files'][int(rng.integers(len(domain['files'])))]
        variants = domain['variants'][filename]
        group = variants[int(rng.integers(len(variants)))]
        anns = domain['annotations'].get(group, [])
        choice = rng.random()
        candidates = [a for a in anns if a['area'] <= domain['small_area']] if choice < .5 else anns
        box = None
        if choice < .75 and candidates:
            box = candidates[int(rng.integers(len(candidates)))]['bbox']
        elif choice < .5 and anns:
            box = anns[int(rng.integers(len(anns)))]['bbox']
        top, left = base.crop_origin(2048, 2048, self.crop, rng, box)
        return dict(domain=name, file_name=filename, group_id=group, top=top, left=left, rng=rng)


class VerifiedSWEFilStore:
    """The same one-time limb transform as GONG; exact source bytes are bound."""
    def __init__(self, records, cache):
        self.stores = {}
        for record in records:
            path = Path(record['image_path'])
            name = Path(record['file_name']).name
            if path.name != name or name in self.stores:
                raise ValueError('SWEFil image paths/names must be unique and aligned')
            expected = record.get('image_sha256') or digest(path)
            self.stores[name] = original.VerifiedImageStore(path.parent, cache, {name: expected})

    def load(self, name):
        return self.stores[name].load(name)


def diagnostic_scores(predictions, groups):
    saved = base.GT
    result = {}
    try:
        for label, names in groups['groups'].items():
            stems = {Path(name).stem for name in names}
            base.GT = {key: truth for key, truth in saved.items() if key.split('-', 1)[1] in stems}
            result[label] = dict(images=len(names), annotation_groups=len(base.GT),
                metrics=base.score(predictions) if base.GT else None)
    finally:
        base.GT = saved
    return result


def detection_rates(row):
    """Recall/precision at the very same strict-IoU matching and operating point."""
    tp,fp,fn = int(row['tp_pairs']),int(row['fp']),int(row['fn'])
    return dict(recall=tp/(tp+fn) if tp+fn else 0.,
                precision=tp/(tp+fp) if tp+fp else 0.)


def prepare_dataset(cfg, prepared, swefil, image_hashes):
    data = json.loads(Path(cfg['annotations']).read_text())
    base.TRAIN=Path(cfg['train_split']); base.VAL=Path(cfg['val_split']); base.IMAGES=Path(cfg['images'])
    train_ids, names, base.GT = base.prepare_split(data)
    synthetic = json.loads(Path(prepared['annotations_path']).read_text())
    syn_hashes = {row['file_name']: digest(Path(prepared['images_dir'])/row['file_name']) for row in synthetic['images']}
    cache = Path(cfg['cache_root'])/'training_images'
    real_store = original.VerifiedImageStore(cfg['images'], cache/'gong', image_hashes)
    synthetic_store = original.VerifiedImageStore(prepared['images_dir'], cache/'chase', syn_hashes)
    swefil_store = VerifiedSWEFilStore(swefil['records'], cache/'swefil')
    dataset = ThreeDomainCropDataset(data, train_ids, real_store, synthetic, synthetic_store,
        swefil_coco(swefil['records']), swefil_store, crop=cfg['crop'], length=400, seed=cfg['seed'],
        synthetic_ratio=cfg['synthetic_ratio'], swefil_ratio=cfg['swefil_ratio'],
        warmup_epochs=cfg['warmup_epochs'], real_finish_epochs=cfg['real_finish_epochs'],
        total_epochs=cfg['epochs'], cache_dir=cache/'targets', include_empty_synthetic=True, augment=True)
    return dataset, real_store, names


def publish_results(output, state, cfg, identity, prepared=None):
    output = Path(output); root = output.parent
    result = dict(status='complete' if state['epoch']==cfg['epochs'] else 'paused',
        epochs_completed=state['epoch'], epochs_target=cfg['epochs'], best_epoch=state['best_epoch'],
        best_development_pq=state['best_pq'], identity=identity, initialization=cfg['initialization'],
        scope=cfg.get('validation_scope', 'Development validation, not public LB.'),
        last_checkpoint_record=state['last'], best_checkpoint_record=state['best'],
        last_checkpoint_url='https://drive.google.com/file/d/'+state['last']['file_id']+'/view',
        best_checkpoint_url='https://drive.google.com/file/d/'+state['best']['file_id']+'/view',
        weights_location='New experiment/verified_checkpoints (Drive API files); not the old train/checkpoints folder.',
        persistence='Immutable Drive API generations; verified server acknowledgement before each epoch commit.',
        checkpoint_weights_in_zip=False, output_root=str(root))
    atomic_json(output/'summary.json', result)
    archive = root/'swefil_mixed_native_results.zip'
    pending = archive.with_suffix('.pending.zip')
    with zipfile.ZipFile(pending, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(output.iterdir()):
            if path.is_file() and path.suffix in ('.json', '.csv'):
                z.write(path, 'train/'+path.name)
        for name in SOURCE_FILES:
            path = Path(__file__).with_name(name)
            if path.is_file():
                z.write(path, 'code/'+name)
        if prepared:
            z.writestr('prepared_summary.json', json.dumps(serial(prepared), indent=2))
    pending.replace(archive)
    result.update(archive=str(archive), archive_sha256=digest(archive))
    atomic_json(root/'latest_result.json', result)
    return result


def _load_remote(store, record, identity):
    path = store.restore(record)
    if digest(path) != record['sha256']:
        raise ValueError('Restored checkpoint SHA differs')
    value = torch.load(path, map_location='cpu', weights_only=True)
    if value.get('identity') != identity:
        raise ValueError('Restored checkpoint belongs to another training identity')
    return value


def _run_epochs(cfg, dataset, model_factory, validator, output, identity, checkpoint_store,
                *, run_until=None, prepared=None):
    """CPU-testable engine. A failed remote commit prevents the next epoch."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg['device']); amp = torch.float16 if device.type=='cuda' else None
    target = cfg['epochs'] if run_until is None else int(run_until)
    if not 1 <= target <= cfg['epochs']:
        raise ValueError('run-until must lie within the unchanged 100-epoch schedule')
    state = checkpoint_store.latest_state()
    # An old FUSE pointer can never silently trigger a fresh run after remote loss.
    if state is None and (output/'state.json').exists():
        raise RuntimeError('Local state exists but the verified remote commit is missing; refusing restart.')
    if state and state['epoch'] >= target:
        checkpoint = _load_remote(checkpoint_store, state['last'], identity)
        if (checkpoint['epoch'] != state['epoch'] or len(checkpoint['history']) != state['epoch']
                or checkpoint['best_epoch'] != state['best_epoch'] or checkpoint['best_pq'] != state['best_pq']):
            raise ValueError('Completed checkpoint does not match the remote state')
        best = _load_remote(checkpoint_store, state['best'], identity)
        if best['epoch'] != state['best_epoch']:
            raise ValueError('Completed best epoch differs')
        base.export_best_predictions(output, best)
        atomic_json(output/'history.json', checkpoint['history'])
        original.write_envelope(output/'state.json', identity, state)
        print(f'Requested epoch {target} already completed; verified epoch {state["epoch"]}. No training.', flush=True)
        return publish_results(output, state, cfg, identity, prepared)
    completed = state['epoch'] if state else 0
    required_free = max(2_000_000_000, int(50_000_000_000*(cfg['epochs']-completed)/cfg['epochs']))
    checkpoint_store.preflight(required_free_bytes=required_free)
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg['seed'])
    model = model_factory(state is None).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'], eta_min=cfg['lr']*cfg['minimum_lr_ratio'])
    scaler = torch.amp.GradScaler('cuda', enabled=amp is not None)
    history=[]; best_pq=-1.; best_epoch=0; start=1; best_record=None
    if state:
        checkpoint = _load_remote(checkpoint_store, state['last'], identity)
        if checkpoint['epoch'] != state['epoch'] or len(checkpoint['history']) != state['epoch'] or checkpoint['history'][-1]['epoch'] != state['epoch']:
            raise ValueError('Checkpoint history and committed epoch differ')
        if checkpoint['best_epoch'] != state['best_epoch'] or checkpoint['best_pq'] != state['best_pq']:
            raise ValueError('Committed best checkpoint metadata differs')
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        history=checkpoint['history']; best_pq=float(state['best_pq']); best_epoch=int(state['best_epoch'])
        best_record=state['best']; start=state['epoch']+1
        best = _load_remote(checkpoint_store, best_record, identity)
        if best['epoch'] != best_epoch:
            raise ValueError('Restored best epoch differs')
        base.export_best_predictions(output, best)
        base.restore_rng(checkpoint['rng_state'])
        del checkpoint, best
    atomic_json(output/'config.json', cfg)
    local = Path(cfg['cache_root'])/'checkpoint_staging'/identity[:16]
    if local.resolve().is_relative_to('/content/drive'):
        raise ValueError('Checkpoint serialization must use local disk, not Drive FUSE')
    print(f'Fresh ImageNet run; verified completed epoch {start-1}; run through {target}/{cfg["epochs"]}.', flush=True)
    for epoch in range(start, target+1):
        dataset.epoch=epoch; loader=base.make_epoch_loader(dataset, cfg, device)
        model.train(); optimizer.zero_grad(set_to_none=True)
        losses=[]; attempts=successes=0; started=time.monotonic()
        domains=Counter(dataset.sample_record(i)['domain'] for i in range(len(dataset))) if hasattr(dataset, 'sample_record') else {}
        for step, (x,y,valid) in enumerate(loader):
            x,y,valid=(v.to(device, non_blocking=True) for v in (x,y,valid))
            group_start=(step//cfg['accumulate'])*cfg['accumulate']
            divisor=min(cfg['accumulate'], len(loader)-group_start)
            with torch.autocast(device_type=device.type, dtype=amp or torch.float32, enabled=amp is not None):
                loss=base.mixed_segmentation_loss(model(x), y, valid, cfg['spine_weight'])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss; last verified epoch remains resumable')
            scaler.scale(loss/divisor).backward(); losses.append(float(loss.detach()))
            if (step+1)%cfg['accumulate']==0 or step+1==len(loader):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_grad_norm'])
                previous=scaler.get_scale(); scaler.step(optimizer); scaler.update()
                successes+=int(scaler.get_scale()>=previous); attempts+=1; optimizer.zero_grad(set_to_none=True)
            if (step+1)%25==0 or step+1==len(loader):
                print(f'epoch {epoch}/{cfg["epochs"]} crop batch {step+1}/{len(loader)} loss {losses[-1]:.5f}', flush=True)
        row=dict(epoch=epoch, train_loss=float(np.mean(losses)), lr=float(optimizer.param_groups[0]['lr']),
            microbatches=len(loader), optimizer_updates=successes, optimizer_update_attempts=attempts,
            amp_skipped_updates=attempts-successes, sampled_domains=dict(domains),
            domain_probabilities=dataset.domain_probabilities() if hasattr(dataset, 'domain_probabilities') else None,
            elapsed_seconds=time.monotonic()-started)
        scheduler.step(); new_best=None
        if epoch==1 or epoch%cfg['validate_every']==0 or epoch==cfg['epochs']:
            best,grid,predictions=validator(model, device, amp, cfg)
            row.update(validation=best, validation_grid=grid)
            fixed=[r for r in grid if r['threshold']==.5 and r['min_area']==100]
            if fixed: row['fixed_limb_threshold05_area100']=fixed[0]
            if best['pq']>best_pq:
                best_pq,best_epoch=float(best['pq']),epoch
                new_best=dict(schema_version=1, identity=identity, model_state_dict=model.state_dict(),
                    config=cfg.copy(), epoch=epoch, validation=best, validation_grid=grid, validation_predictions=predictions)
                path=local/'checkpoints'/f'best_epoch{epoch:03}.pt'
                original.save_checkpoint(path, new_best)
                best_record=checkpoint_store.publish_checkpoint(path, f'checkpoints/{path.name}', metadata=dict(epoch=epoch, kind='best'))
        history.append(row)
        payload=dict(schema_version=1, identity=identity, model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(), scheduler_state_dict=scheduler.state_dict(),
            scaler_state_dict=scaler.state_dict(), config=cfg.copy(), epoch=epoch, history=history,
            best_pq=best_pq, best_epoch=best_epoch, rng_state=base.capture_rng())
        path=local/'checkpoints'/f'last_epoch{epoch:03}.pt'
        original.save_checkpoint(path, payload)
        last_record=checkpoint_store.publish_checkpoint(path, f'checkpoints/{path.name}', metadata=dict(epoch=epoch, kind='last'))
        next_state=dict(epoch=epoch, last=last_record, best=best_record, best_pq=best_pq, best_epoch=best_epoch)
        state=checkpoint_store.commit_state(next_state, expected_previous_epoch=epoch-1)
        # These mirrors are disposable; resuming always queries server-verified commits.
        original.write_envelope(output/'state.json', identity, state)
        if new_best is not None: base.export_best_predictions(output, new_best)
        atomic_json(output/'history.json', history)
        print(f'Checkpoint confirmed on Drive: epoch {epoch}; last file ID {last_record["file_id"]}', flush=True)
        print(json.dumps(row), flush=True)
        publish_results(output, state, cfg, identity, prepared)
        del payload,new_best,loader; gc.collect()
    if state is None: raise RuntimeError('No completed epoch state')
    atomic_json(output/'history.json', history)
    result=publish_results(output, state, cfg, identity, prepared)
    del model,optimizer,scheduler,scaler
    if device.type=='cuda': torch.cuda.empty_cache()
    return result


def run(config, phase='all', run_until=None):
    if phase not in ('all', 'train', 'prepare'):
        raise ValueError('Unsupported phase')
    cfg=normalize_config(config)
    root=Path(cfg['output_root']); root.mkdir(parents=True, exist_ok=True)
    output=root/'train'; output.mkdir(exist_ok=True)
    cv2.setNumThreads(2)
    with (root/'.training.lock').open('a+') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Another process is using this training output')
        import chase_shape_cache
        restored=chase_shape_cache.restore(cfg['chase_training_root'])
        if restored['status'] not in ('local_complete', 'restored'):
            raise FileNotFoundError('Existing CHASE synthetic PNG archive is required; no resynthesis performed: '+str(restored))
        prepared=json.loads((Path(cfg['chase_training_root'])/'prepared_summary.json').read_text())
        data=json.loads(Path(cfg['annotations']).read_text())
        base.TRAIN=Path(cfg['train_split']); base.VAL=Path(cfg['val_split']); base.IMAGES=Path(cfg['images'])
        _,names,base.GT=base.prepare_split(data)
        from swefil_training_data import prepare_swefil
        swefil=prepare_swefil(Path(cfg['base']), Path(cfg['cache_root'])/'swefil_source', val_names=names)
        if len(swefil['records']) != 250:
            raise ValueError('The pinned full annotated SWEFil pool must contain 250 physical observations')
        summary=dict(chase_synthesis_identity=prepared['identity'], chase_cache=restored,
            swefil={key:value for key,value in swefil.items() if key!='records'},
            temporal_diagnostic=dict(groups=dict(near_swefil_7d=swefil['validation_proximity']['near'],
                far_swefil_7d=swefil['validation_proximity']['far']),
                exclusions=0, source=swefil['validation_proximity'],
                note='Timestamp proximity only; no training exclusion and no independent subset tuning.'))
        atomic_json(root/'prepared_summary.json', summary)
        if phase=='prepare':
            result=dict(status='prepared', output_root=str(root), **summary)
            atomic_json(root/'latest_result.json', result)
            return result
        if not torch.cuda.is_available(): raise RuntimeError('Select a Colab T4 GPU runtime for training')
        original.configure_cuda(); cv2.setNumThreads(2)
        source,image_hashes=original.source_identity(cfg, prepared, torch.device(cfg['device']))
        source.update(version=VERSION, settings={key:cfg[key] for key in DEFAULTS},
            code={name:digest(Path(__file__).with_name(name)) for name in SOURCE_FILES},
            swefil_manifest_sha256=swefil['manifest_sha256'], external_filter='none; all 250 annotated observations')
        identity=fingerprint(source)
        identity_path=output/'identity.json'
        if identity_path.exists() and original.read_envelope(identity_path, identity)!=source:
            raise ValueError('Training identity changed')
        original.write_envelope(identity_path, identity, source)
        atomic_json(output/'temporal_diagnostics.json', summary['temporal_diagnostic'])
        historical=Path(cfg['chase_training_root'])/'train/baseline_metrics.json'
        if historical.is_file():
            atomic_json(output/'historical_best84_reference.json', dict(source_sha256=digest(historical),
                historical_only=True, source=json.loads(historical.read_text()),
                note='Previously measured native best84; not rerun and not training initialization.'))
        dataset,store,names=prepare_dataset(cfg, prepared, swefil, image_hashes)
        cfg['data_domains']=dataset.domain_info
        import verified_drive_checkpoints as durable
        service=durable.make_drive_service()
        folder_id=durable.resolve_experiment_folder(service, cfg['drive_checkpoint_path_parts'], identity,
            create_from=3, visibility_attempts=6, visibility_wait_seconds=2)
        checkpoint_store=durable.DriveCheckpointStore(service, folder_id, identity,
            Path(cfg['cache_root'])/'remote_checkpoints')
        def validator(model, device, amp, settings):
            best,grid,predictions=base.validate_native(model, store, names, device, amp, settings)
            for row in grid:
                row.update(detection_rates(row))
                row['temporal_diagnostics']=diagnostic_scores(predictions[row['threshold'],row['min_area']], summary['temporal_diagnostic'])
            return best,grid,predictions
        return _run_epochs(cfg, dataset, lambda fresh:base.build_model(cfg['encoder'], pretrained=fresh),
            validator, output, identity, checkpoint_store, run_until=run_until, prepared=summary)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--phase', choices=('prepare','train','all'), default='all')
    parser.add_argument('--run-until', type=int)
    args=parser.parse_args()
    result=run(json.loads(args.config.read_text()), args.phase, args.run_until)
    print(json.dumps(result, indent=2), flush=True)
    if result.get('best_checkpoint_url'):
        print('Verified best weights:', result['best_checkpoint_url'], flush=True)
        print('Verified resumable last weights:', result['last_checkpoint_url'], flush=True)


if __name__=='__main__': main()
