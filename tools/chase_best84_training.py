"""Isolated CHASE transfer for the deployed two-head best84 U-Net.

YOLO weights and production finalizer are fixed in this native-model arm.
The default trains only on CHASE foreground labels; GONG is validation-only.
An explicitly selected mixed mode retains the earlier replay experiment.
Resume always restores a completed
epoch, including optimizer, schedule, AMP and RNG state. By default each epoch
uses every CHASE photograph once, with one crop per photograph.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import zipfile

os.environ.setdefault('TORCH_CUDNN_V8_API_DISABLED', '1')
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

INITIAL_SHA = 'ae6b209f72355fd52d2cb07182e69028e97c703a1977ad1e7b38b27cf8cebe69'
INPUT_SHA = {
    'annotations': '5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1',
    'train_split': 'aa0733424815731c5c108ada91e45525eb17ca3a42b377bf19e3089d86aede7e',
    'val_split': '54c6a9aac5a3b70b3ecc6721109a8b0fe8ec020e90cb56e3848fccc92eadc61f',
}
DEFAULTS = dict(training_domain='chase_only', epoch_sampling='full_pass',
    epochs=60, crop=1024, batch=1, accumulate=4,
    steps_per_epoch=695, lr=2e-5, weight_decay=.01, spine_weight=.1,
    chase_ratio=1., real_finish_epochs=0, validate_every=5, seed=42,
    device='cuda', native_batch=1, workers=0, amp=True)
SOURCES = ('chase_best84_training.py', 'chase_transfer_data.py',
    'chase_best84_evaluation.py', 'solar_a100_train_standalone.py',
    'submission_inference.py', 'submission_driver.py',
    'submission_yolo_recipe.py', 'submission_native_recipe.py')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ready(value):
    if isinstance(value, Path): return str(value)
    if isinstance(value, dict): return {str(k): ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [ready(v) for v in value]
    if isinstance(value, np.generic): return value.item()
    return value


def fingerprint(value):
    return hashlib.sha256(json.dumps(ready(value), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name+'.pending')
    try:
        pending.write_text(json.dumps(ready(value), ensure_ascii=False,
            indent=2, allow_nan=False)+'\n', encoding='utf-8')
        pending.replace(path)
    finally: pending.unlink(missing_ok=True)


def atomic_torch(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name+'.pending')
    try:
        torch.save(value, pending); pending.replace(path)
    finally: pending.unlink(missing_ok=True)


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed % 2**32); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def capture_rng():
    state = np.random.get_state()
    return dict(python=random.getstate(), torch=torch.get_rng_state(),
        numpy=(state[0], torch.from_numpy(state[1].astype(np.int64)),
               int(state[2]), int(state[3]), float(state[4])),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python']); torch.set_rng_state(state['torch'].cpu())
    n = state['numpy']
    np.random.set_state((n[0], n[1].cpu().numpy().astype(np.uint32), n[2], n[3], n[4]))
    if state['cuda'] and torch.cuda.is_available():
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('Resume CUDA device count differs')
        torch.cuda.set_rng_state_all([item.cpu() for item in state['cuda']])


def configure_cuda():
    torch.backends.cudnn.benchmark = False
    if not getattr(nn.Conv2d._conv_forward, '_chase_depthwise_safe', False):
        original = nn.Conv2d._conv_forward
        def safe(self, inputs, weight, bias):
            if inputs.is_cuda and self.groups > 1 and self.groups == self.in_channels:
                with torch.backends.cudnn.flags(enabled=False):
                    return original(self, inputs, weight, bias)
            return original(self, inputs, weight, bias)
        safe._chase_depthwise_safe = True
        nn.Conv2d._conv_forward = safe


def masked_segmentation_loss(logits, target, valid, spine_weight=.1):
    """Pixel ignores affect BCE and Dice; absent CHASE spine labels give no loss."""
    if logits.ndim != 4 or logits.shape[1] != 2 or logits.shape != target.shape or logits.shape != valid.shape:
        raise ValueError('Expected matching [B,2,H,W] logits, targets and valid masks')
    if not torch.isfinite(target).all() or not torch.isfinite(valid).all():
        raise ValueError('Nonfinite supervision')
    if torch.any((target < 0) | (target > 1)) or torch.any((valid != 0) & (valid != 1)):
        raise ValueError('Targets must be in [0,1] and valid masks binary')
    pred, truth, keep = logits.float(), target.float(), valid.float()
    fg, fg_truth, fg_valid = pred[:, 0], truth[:, 0], keep[:, 0]
    count = fg_valid.sum()
    if count.item() == 0:
        raise ValueError('A training batch has no supervised foreground pixels')
    bce = (F.binary_cross_entropy_with_logits(fg, fg_truth, reduction='none')*fg_valid).sum()/count
    p = fg.sigmoid()*fg_valid
    y = fg_truth*fg_valid
    active = (fg_valid.sum((1, 2)) > 0).float()
    dice_each = 1-(2*(p*y).sum((1, 2))+1)/(p.sum((1, 2))+y.sum((1, 2))+1)
    dice = (dice_each*active).sum()/active.sum()
    line, line_truth, line_valid = pred[:, 1], truth[:, 1], keep[:, 1]
    errors = F.binary_cross_entropy_with_logits(line, line_truth, reduction='none')
    positive, negative = line_truth*line_valid, (1-line_truth)*line_valid
    neg = (errors*negative).sum()/negative.sum().clamp_min(1)
    pos_count = positive.sum()
    spine = .5*((errors*positive).sum()/pos_count.clamp_min(1)+neg) if pos_count.item() > 0 else neg
    total = bce+dice+float(spine_weight)*spine
    return total, dict(foreground_bce=bce.detach(), foreground_dice=dice.detach(),
                      spine_loss=spine.detach())


class ReplayDataset(Dataset):
    """One CHASE crop per photograph, or an explicitly requested legacy budget."""
    def __init__(self, gong, chase, cfg):
        self.gong, self.chase, self.cfg = gong, chase, cfg
        self.training_domain = cfg['training_domain']
        self.epoch_sampling = cfg['epoch_sampling']
        if self.epoch_sampling not in ('full_pass', 'bounded'):
            raise ValueError('epoch_sampling must be full_pass or bounded')
        if self.epoch_sampling == 'full_pass' and (self.training_domain != 'chase_only' or cfg['batch'] != 1):
            raise ValueError('Full-pass training requires CHASE-only and batch1')
        if self.training_domain not in ('chase_only', 'mixed'):
            raise ValueError('Unknown training domain')
        if self.training_domain == 'chase_only' and (gong is not None or
                cfg['chase_ratio'] != 1. or cfg['real_finish_epochs'] != 0):
            raise ValueError('CHASE-only mode forbids a GONG training dataset or replay schedule')
        if self.training_domain == 'mixed' and gong is None:
            raise ValueError('Mixed mode requires a GONG training dataset')
        self.length = int(cfg['steps_per_epoch'])*int(cfg['batch'])
        self.chase_ids = [str(row['id']) for row in chase.records]
        if not self.chase_ids or len(set(self.chase_ids)) != len(self.chase_ids):
            raise ValueError('CHASE replay requires unique physical observation IDs')
        if self.epoch_sampling == 'full_pass' and self.length != len(self.chase_ids):
            raise ValueError('Full-pass steps_per_epoch must equal the number of CHASE photographs')
        generator = np.random.default_rng(np.random.SeedSequence([cfg['seed'], 695, 20261104]))
        self.chase_order = generator.permutation(len(self.chase_ids))
        self.chase_per_epoch = int(round(self.length*cfg['chase_ratio']))
        if self.chase_per_epoch < 1:
            raise ValueError('The crop budget must include at least one CHASE observation')
        self.set_epoch(1)

    def __len__(self): return self.length

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        if self.gong is not None: self.gong.epoch = self.epoch
        self.chase.set_epoch(self.epoch)
        if self.epoch_sampling == 'full_pass':
            generator = np.random.default_rng(np.random.SeedSequence([self.cfg['seed'], epoch, 20261104]))
            order = generator.permutation(len(self.chase_ids))
            self.chase_indices = set(range(self.length))
            self.chase_record_indices = {index: int(record) for index, record in enumerate(order)}
            self.chase_observation_ids = [self.chase_ids[int(record)] for record in order]
            return
        ratio = self.cfg['chase_ratio'] if epoch <= self.cfg['epochs']-self.cfg['real_finish_epochs'] else 0.
        count = int(round(self.length*ratio))
        generator = np.random.default_rng(np.random.SeedSequence([self.cfg['seed'], epoch, 20261104]))
        self.chase_indices = set(map(int, generator.permutation(self.length)[:count]))
        offset = (epoch-1)*self.chase_per_epoch
        self.chase_record_indices = {index: int(self.chase_order[(offset+position) % len(self.chase_ids)])
                                     for position, index in enumerate(sorted(self.chase_indices))}
        self.chase_observation_ids = [self.chase_ids[self.chase_record_indices[index]]
                                      for index in sorted(self.chase_indices)]

    def __getitem__(self, index):
        if index in self.chase_indices:
            x, y, valid = self.chase.sample(index, record_index=self.chase_record_indices[index])
            return tuple(torch.as_tensor(np.ascontiguousarray(v) if isinstance(v, np.ndarray) else v,
                                         dtype=torch.float32) for v in (x, y, valid))
        if self.gong is None: raise AssertionError('A CHASE-only sample attempted to use GONG')
        x, y, spine_valid = self.gong[index]
        valid = torch.ones_like(y); valid[1] *= float(spine_valid)
        return x, y, valid


def normalize_config(config):
    cfg = dict(DEFAULTS, **config)
    if cfg['training_domain'] not in ('chase_only', 'mixed'):
        raise ValueError('training_domain must be chase_only or explicit mixed')
    if cfg['training_domain'] == 'mixed':
        if 'epoch_sampling' not in config: cfg['epoch_sampling'] = 'bounded'
        if 'chase_ratio' not in config: cfg['chase_ratio'] = .2
        if 'real_finish_epochs' not in config: cfg['real_finish_epochs'] = 6
    if cfg['epoch_sampling'] not in ('full_pass', 'bounded'):
        raise ValueError('epoch_sampling must be full_pass or bounded')
    if cfg['epoch_sampling'] == 'full_pass' and (cfg['training_domain'] != 'chase_only' or cfg['batch'] != 1):
        raise ValueError('Full-pass training requires CHASE-only and batch1')
    for key in ('epochs', 'crop', 'batch', 'accumulate', 'steps_per_epoch', 'validate_every', 'native_batch'):
        if not isinstance(cfg[key], int) or isinstance(cfg[key], bool) or cfg[key] < 1:
            raise ValueError(key+' must be a positive integer')
    if cfg['crop'] != 1024: raise ValueError('This experiment keeps best84 native crop1024')
    if cfg['training_domain'] == 'chase_only' and (cfg['chase_ratio'] != 1. or cfg['real_finish_epochs'] != 0):
        raise ValueError('CHASE-only mode requires chase_ratio=1 and real_finish_epochs=0')
    if not 0 < cfg['chase_ratio'] <= 1 or not 0 <= cfg['real_finish_epochs'] < cfg['epochs']:
        raise ValueError('Invalid CHASE replay schedule')
    if not 0 < cfg['lr'] <= 1e-3 or cfg['weight_decay'] < 0 or cfg['spine_weight'] < 0:
        raise ValueError('Invalid optimizer/loss settings')
    if cfg['workers'] != 0: raise ValueError('This resumable implementation fixes workers=0')
    for key in ('output_root', 'initial_checkpoint', 'images', 'annotations', 'train_split',
                'val_split', 'chase_root', 'chase_manifest', 'cache_root', 'yolo_cache'):
        if not cfg.get(key): raise ValueError('Missing path: '+key)
        cfg[key] = str(Path(cfg[key]).expanduser())
    out, old = Path(cfg['output_root']).resolve(), Path(cfg['initial_checkpoint']).resolve()
    if old.is_relative_to(out) or out == Path(cfg['images']).resolve():
        raise ValueError('New output folder must not contain the original best84 or image folder')
    if Path(cfg['cache_root']).resolve().is_relative_to(Path('/content/drive')):
        raise ValueError('Processed caches belong on Colab local disk')
    return cfg


def prepare_inputs(cfg):
    import solar_a100_train_standalone as legacy
    from chase_transfer_data import prepare_chase, ChaseDataset
    for key, expected in INPUT_SHA.items():
        if digest(cfg[key]) != expected: raise ValueError('Frozen GONG source changed: '+key)
    if digest(cfg['initial_checkpoint']) != INITIAL_SHA:
        raise ValueError('Initial checkpoint is not the deployed best84')
    gong = None
    if cfg['training_domain'] == 'chase_only':
        from chase_best84_evaluation import load_cohort
        cohort = load_cohort(cfg)
        val_names = cohort['names']; groups = cohort['groups']
        names = val_names  # No access to any GONG training photograph.
        validation_groups = len(groups)
        validation_instances = sum(len(row['masks']) for row in groups.values())
    else:
        legacy.TRAIN, legacy.VAL, legacy.IMAGES = map(Path, (cfg['train_split'], cfg['val_split'], cfg['images']))
        data = json.loads(Path(cfg['annotations']).read_text())
        train_ids, val_names, truth = legacy.prepare_split(data)
        names = sorted({row['file_name'] for row in data['images']})
        validation_groups = len(truth); validation_instances = sum(map(len, truth.values()))
    image_hashes = {}
    for index, name in enumerate(names, 1):
        image_hashes[name] = digest(Path(cfg['images'])/name)
        if index % 100 == 0 or index == len(names): print(f'GONG 검증/참조 이미지 검사 {index}/{len(names)}', flush=True)
    records, chase_report = prepare_chase(cfg['chase_root'], cfg['cache_root'], cfg['chase_manifest'])
    if cfg['epoch_sampling'] == 'full_pass' and cfg['steps_per_epoch'] != len(records):
        raise ValueError('Full-pass steps_per_epoch must equal the number of CHASE photographs')
    cache = Path(cfg['cache_root'])
    length = cfg['steps_per_epoch']*cfg['batch']
    if cfg['training_domain'] == 'mixed':
        store = legacy.ImageStore(cfg['images'], cache/'gong_images', 'limb_only')
        gong = legacy.MixedCropDataset(data, train_ids, store, crop=cfg['crop'], length=length,
            seed=cfg['seed'], synthetic_ratio=0., warmup_epochs=0, real_finish_epochs=0,
            total_epochs=cfg['epochs'], cache_dir=cache/'gong_targets', augment=False)
    chase = ChaseDataset(records, crop=cfg['crop'], length=length, seed=cfg['seed']+173,
                         cache_dir=cache/'chase_aligned')
    chase_preflight = chase.preflight()
    metadata = dict(gong_sha256=INPUT_SHA, initial_sha256=INITIAL_SHA,
        gong_image_sha256=image_hashes, chase_manifest_sha256=digest(cfg['chase_manifest']),
        chase_records=records, yolo_cache_sha256=digest(cfg['yolo_cache']),
        training_domain=cfg['training_domain'], train_photos=len(gong.files) if gong is not None else 0,
        epoch_sampling=cfg['epoch_sampling'], effective_steps_per_epoch=cfg['steps_per_epoch'],
        gong_train_photos_used=len(gong.files) if gong is not None else 0,
        chase_train_photos=len(records), gong_image_hash_scope='validation_only' if gong is None else 'train_and_validation',
        validation_photos=len(val_names), validation_groups=validation_groups,
        validation_instances=validation_instances,
        chase_report=str(chase_report), chase_preflight=chase_preflight)
    return ReplayDataset(gong, chase, cfg), ready(metadata)


def runtime_versions():
    result = dict(python=sys.version.split()[0], torch=torch.__version__,
        cuda=torch.version.cuda, opencv=cv2.__version__,
        **{p: importlib.metadata.version(p) for p in ('numpy', 'scipy',
            'segmentation-models-pytorch', 'timm', 'pycocotools')})
    for package in ('opencv-python', 'opencv-python-headless'):
        try: result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: result[package] = None
    return result


def source_hashes():
    return {name: digest(Path(__file__).with_name(name)) for name in SOURCES}


def state_payload(model, optimizer, scheduler, scaler, cfg, identity, epoch, history,
                  best_pq, best_epoch, pending_eval):
    return dict(schema_version=1, identity=identity, config=cfg, epoch=int(epoch),
        model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
        scheduler_state_dict=scheduler.state_dict(), scaler_state_dict=scaler.state_dict(),
        rng_state=capture_rng(), history=history, best_pq=best_pq,
        best_epoch=best_epoch, pending_eval=pending_eval)


def restore_state(saved, model, optimizer, scheduler, scaler, identity, cap):
    required = {'identity', 'epoch', 'history', 'rng_state', 'model_state_dict',
        'optimizer_state_dict', 'scheduler_state_dict', 'scaler_state_dict',
        'best_pq', 'best_epoch', 'pending_eval'}
    if not required <= set(saved) or saved['identity'] != identity:
        raise ValueError('Checkpoint is incomplete or has a different experiment identity')
    epoch, history = saved['epoch'], saved['history']
    if not isinstance(epoch, int) or not 0 <= epoch <= cap or len(history) != epoch:
        raise ValueError('Completed epoch/history do not match the requested stage')
    if [r['epoch'] for r in history] != list(range(1, epoch+1)):
        raise ValueError('Checkpoint history is not consecutive')
    if saved['pending_eval'] not in (None, epoch):
        raise ValueError('Only the current completed checkpoint can have pending validation')
    model.load_state_dict(saved['model_state_dict'], strict=True)
    optimizer.load_state_dict(saved['optimizer_state_dict']); scheduler.load_state_dict(saved['scheduler_state_dict'])
    scaler.load_state_dict(saved['scaler_state_dict']); restore_rng(saved['rng_state'])
    return epoch, history, saved['best_pq'], saved['best_epoch'], saved['pending_eval']


def run_training(model, dataset, cfg, run_id, cap, evaluate, *, device):
    """Factored engine permits CPU lifecycle tests with a tiny injected model."""
    root = Path(cfg['output_root']); last = root/'last.pt'
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda e: .1+.9*(1+math.cos(math.pi*e/cfg['epochs']))/2)
    enabled = bool(cfg['amp'] and device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=enabled)
    epoch, history, best_pq, best_epoch, pending = 0, [], None, None, 0
    if last.exists():
        saved = torch.load(last, map_location=device, weights_only=True)
        if saved.get('config', {}).get('training_domain') != cfg['training_domain']:
            raise ValueError('Cannot resume a checkpoint from a different training domain')
        if saved.get('config', {}).get('epoch_sampling', 'bounded') != cfg['epoch_sampling']:
            raise ValueError('Cannot resume a checkpoint with a different epoch sampling mode')
        epoch, history, best_pq, best_epoch, pending = restore_state(
            saved, model, optimizer, scheduler, scaler, run_id, cap)

    def save():
        atomic_torch(last, state_payload(model, optimizer, scheduler, scaler, cfg,
            run_id, epoch, history, best_pq, best_epoch, pending))
        atomic_json(root/'history.json', history)

    def validate_pending():
        nonlocal best_pq, best_epoch, pending
        if pending is None: return
        metric = evaluate(model, cfg, f'epoch{epoch:03d}')
        pq = float(metric['pq'])
        if not math.isfinite(pq): raise ValueError('Validation returned nonfinite PQ')
        if best_pq is None or pq > best_pq:
            best_pq, best_epoch = pq, epoch
            atomic_torch(root/'best.pt', dict(schema_version=1, identity=run_id,
                epoch=epoch, model_state_dict=model.state_dict(),
                config=dict(cfg, encoder='tu-convnext_tiny', preprocessing='limb_only', overlap=.25),
                validation=metric, scope='Updated native model; YOLO and finalizer frozen'))
        if epoch: history[-1]['validation'] = metric
        else: atomic_json(root/'baseline_validation.json', metric)
        pending = None; save()
        atomic_json(root/'best_epoch.json', dict(epoch=best_epoch, pq=best_pq))
        print(f'전체 파이프라인 epoch{epoch}: PQ={pq:.6f}, 최고 epoch{best_epoch}={best_pq:.6f}', flush=True)

    if not last.exists(): save()
    validate_pending()
    while epoch < cap:
        next_epoch = epoch+1
        seed_everything(cfg['seed']+10007*next_epoch); dataset.set_epoch(next_epoch)
        generator = torch.Generator().manual_seed(cfg['seed']+10007*next_epoch)
        loader = DataLoader(dataset, batch_size=cfg['batch'], shuffle=False, num_workers=0,
            pin_memory=device.type == 'cuda', generator=generator)
        if len(loader) != cfg['steps_per_epoch']: raise ValueError('Wrong sampled-crop epoch budget')
        started = time.monotonic(); totals = {}; successes = attempts = 0
        model.train(); optimizer.zero_grad(set_to_none=True)
        for index, (inputs, target, valid) in enumerate(loader):
            inputs, target, valid = (v.to(device, non_blocking=True) for v in (inputs, target, valid))
            group_start = index//cfg['accumulate']*cfg['accumulate']
            group_length = min(cfg['accumulate'], len(loader)-group_start)
            context = torch.autocast(device_type='cuda', dtype=torch.float16) if enabled else nullcontext()
            with context:
                loss, terms = masked_segmentation_loss(model(inputs), target, valid, cfg['spine_weight'])
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss; last completed epoch preserved')
            scaler.scale(loss/group_length).backward()
            for key, value in dict(loss=loss.detach(), **terms).items(): totals[key] = totals.get(key, 0.)+float(value.cpu())
            if (index+1) % cfg['accumulate'] == 0 or index+1 == len(loader):
                before = scaler.get_scale(); scaler.step(optimizer); scaler.update()
                attempts += 1; successes += int(scaler.get_scale() >= before); optimizer.zero_grad(set_to_none=True)
            if (index+1) % 50 == 0 or index+1 == len(loader):
                print(f'epoch {next_epoch}/{cfg["epochs"]} batch {index+1}/{len(loader)} loss {totals["loss"]/(index+1):.5f}', flush=True)
        if successes == 0: raise FloatingPointError('All optimizer updates overflowed; completed checkpoint retained')
        scheduler.step(); epoch = next_epoch
        history.append(dict(epoch=epoch, **{k: v/len(loader) for k, v in totals.items()},
            seconds=time.monotonic()-started, optimizer_steps=successes, optimizer_attempts=attempts,
            chase_crops=len(dataset.chase_indices), gong_crops=len(dataset)-len(dataset.chase_indices),
            chase_observation_ids=dataset.chase_observation_ids,
            lr=float(optimizer.param_groups[0]['lr'])))
        observed = {name for row in history for name in row['chase_observation_ids']}
        history[-1]['chase_unique_observations_seen'] = len(observed)
        pending = epoch if epoch % cfg['validate_every'] == 0 or epoch == cfg['epochs'] else None
        save(); validate_pending()
    status = 'complete' if epoch == cfg['epochs'] else 'paused'
    summary = dict(status=status, epochs_completed=epoch, fixed_final_epoch=cfg['epochs'],
        best_epoch=best_epoch, best_full_pipeline_pq=best_pq, identity=run_id,
        initial_best84_preserved=True, yolo_trained=False, rule_refiner_trained=False,
        independent_arm='native', training_domain=cfg['training_domain'],
        epoch_sampling=cfg['epoch_sampling'], steps_per_epoch=cfg['steps_per_epoch'],
        effective_steps_per_epoch=cfg['steps_per_epoch'],
        requested_steps_per_epoch=cfg['steps_per_epoch'],
        accepted_chase_observations=len(dataset.chase_ids),
        gong_training_crops=sum(row['gong_crops'] for row in history),
        scope='Same fixed91 development split; native update with original YOLO, not leaderboard PQ',
        last=str(last), best=str(root/'best.pt'))
    exposures = {}
    for row in history:
        for name in row['chase_observation_ids']: exposures[name] = exposures.get(name, 0)+1
    summary['chase_unique_observations_seen'] = len(exposures)
    summary['chase_total_observations'] = len(dataset.chase_ids)
    atomic_json(root/'chase_exposure.json', dict(per_observation_crop_counts=exposures,
        unique_observations_seen=len(exposures), total_observations=len(dataset.chase_ids),
        epoch_sampling=cfg['epoch_sampling'],
        first_cycle_epochs=math.ceil(len(dataset.chase_ids)/dataset.chase_per_epoch),
        scope='Sampled observation coverage; not every annotated pixel or filament seen'))
    atomic_json(root/'result.json', summary)
    return summary


def package_results(root):
    root = Path(root); archive = root/'chase_best84_results.zip'; pending = archive.with_suffix('.pending.zip')
    with zipfile.ZipFile(pending, 'w', zipfile.ZIP_DEFLATED) as output:
        for file in sorted(root.rglob('*')):
            rel = file.relative_to(root)
            if file.is_file() and file.suffix in ('.json', '.csv', '.log') and 'cache' not in rel.parts:
                output.write(file, str(rel))
    pending.replace(archive); return str(archive)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--run-until', type=int)
    parser.add_argument('--prepare-only', action='store_true'); args = parser.parse_args(argv)
    cfg = normalize_config(json.loads(Path(args.config).read_text()))
    cap = cfg['epochs'] if args.run_until is None else args.run_until
    if not 0 <= cap <= cfg['epochs']: raise ValueError('run-until must fit the fixed total epoch schedule')
    device = torch.device(cfg['device'])
    if not args.prepare_only and (device.type != 'cuda' or not torch.cuda.is_available()):
        raise RuntimeError('Select a Colab T4/A100 GPU runtime')
    cv2.setNumThreads(1)
    root = Path(cfg['output_root']); root.mkdir(parents=True, exist_ok=True)
    with (root/'run.lock').open('a+') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('The same experiment is already running')
        dataset, metadata = prepare_inputs(cfg)
        protocol = dict(config=cfg, data=metadata, sources=source_hashes(), runtime=runtime_versions(),
            trainable=('best84 encoder, decoder and foreground; no spine targets or GONG training samples'
                       if cfg['training_domain'] == 'chase_only' else
                       'best84 encoder, decoder and foreground; GONG replay also supervises spine'),
            fixed='YOLO, finalizer, thresholds, validation split', chase_preprocessing='alignment only',
            photometric_augmentation=False,
            chase_sampling=('each physical observation exactly once per epoch'
                if cfg['epoch_sampling'] == 'full_pass' else 'rotating physical observation permutation'),
            epoch_unit=('one crop from every CHASE photograph' if cfg['epoch_sampling'] == 'full_pass'
                else 'fixed sampled crops'), bitwise_GPU_resume_guaranteed=False)
        run_id = fingerprint(protocol); manifest = root/'training_identity.json'
        if manifest.exists() and json.loads(manifest.read_text())['identity'] != run_id:
            raise ValueError('Experiment inputs/code/settings/runtime changed; keep this run and use a new folder')
        if (root/'last.pt').exists() and not manifest.exists(): raise ValueError('Unidentified last.pt preserved')
        atomic_json(manifest, dict(identity=run_id, protocol=protocol))
        atomic_json(root/'experiment_config.json', cfg); atomic_json(root/'data_audit.json', metadata)
        if args.prepare_only:
            print(json.dumps(dict(status='prepared', identity=run_id, data_audit=str(root/'data_audit.json')))); return
        configure_cuda(); seed_everything(cfg['seed'])
        import solar_a100_train_standalone as legacy
        from chase_best84_evaluation import evaluate_candidate
        model = legacy.build_model(encoder='tu-convnext_tiny', pretrained=False).to(device)
        if not (root/'last.pt').exists():
            old = torch.load(cfg['initial_checkpoint'], map_location='cpu', weights_only=True)
            if old.get('epoch') != 84 or old.get('config', {}).get('encoder') != 'tu-convnext_tiny':
                raise ValueError('Expected deployed two-head epoch84 checkpoint')
            model.load_state_dict(old['model_state_dict'], strict=True); del old
        print(f'best84 추가 학습: 전체 {cfg["epochs"]}epoch 고정, 이번 {cap}까지. 이 native 학습 단계에서는 YOLO/후처리 고정.', flush=True)
        if cfg['training_domain'] == 'chase_only':
            detail = '전체 사진에서 각각 crop 하나' if cfg['epoch_sampling'] == 'full_pass' else 'sampled crops'
            print(f'epoch당 CHASE {len(dataset)}개: {detail}. GONG은 검증에만 사용합니다.', flush=True)
        else:
            print(f'epoch당 {len(dataset)} sampled crops. CHASE {cfg["chase_ratio"]:.0%}, 마지막 {cfg["real_finish_epochs"]}epoch GONG-only.', flush=True)
        summary = run_training(model, dataset, cfg, run_id, cap, evaluate_candidate, device=device)
        summary['archive'] = str(root/'chase_best84_results.zip')
        atomic_json(root/'result.json', summary); package_results(root)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__': main()
