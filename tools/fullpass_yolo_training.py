"""Fresh COCO YOLO26s-seg training on every source image's four 1024 tiles.

The immutable Drive store is authoritative. Interrupted epochs are replayed from
its last completed epoch; no old solar checkpoint initializes a new experiment.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import time
from urllib.request import urlopen

import numpy as np
import torch
from torch.utils.data import Dataset

from chase_yolo_training import configure_cuda
from targeted_synthetic_training import (ULTRALYTICS_VERSION, atomic_json,
    atomic_torch, capture_rng, cpu_copy, digest, fingerprint, restore_rng,
    seed_epoch)

PRETRAINED_NAME = 'yolo26s-seg.pt'
PRETRAINED_URL = 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-seg.pt'
PRETRAINED_SHA256 = '3da1d83e31caec96f9300eb4064f4f62882c133c7c264d63dfe61a7c197837a4'
DEFAULTS = dict(epochs=100, seed=42, yolo_imgsz=1024, yolo_batch=1,
    yolo_accumulate=4, yolo_lr=1e-4, yolo_weight_decay=.01,
    yolo_minimum_lr_ratio=.05, yolo_workers=0, yolo_amp=True,
    validate_every=2, device=0)
TILE_ORIGINS = frozenset(((0, 0), (0, 1024), (1024, 0), (1024, 1024)))


def download_generic_weights(folder):
    """Only the pinned official 80-class COCO asset can initialize a fresh run."""
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    target = folder / PRETRAINED_NAME
    if target.is_file() and digest(target) == PRETRAINED_SHA256:
        return target
    pending = target.with_suffix('.download')
    try:
        with urlopen(PRETRAINED_URL, timeout=120) as response, pending.open('wb') as stream:
            while block := response.read(1024 * 1024): stream.write(block)
        if digest(pending) != PRETRAINED_SHA256:
            raise ValueError('Generic COCO checkpoint hash mismatch; refusing unverified initialization')
        pending.replace(target)
    finally:
        pending.unlink(missing_ok=True)
    return target


def record_key(row):
    return (row['domain'], row['file_name'], str(row['group_id']), int(row['top']), int(row['left']))


def selected_group(groups, seed, domain, filename, epoch):
    """Same cycle as the native full-pass loader, including all observer variants."""
    values = sorted(map(str, groups))
    if not values or epoch < 1: raise ValueError('Nonempty groups and a positive epoch are required')
    encoded = json.dumps([int(seed), domain, filename], separators=(',', ':')).encode()
    offset = int(hashlib.sha256(encoded).hexdigest(), 16) % len(values)
    return values[(offset + epoch - 1) % len(values)]


class FullPassYoloDataset(Dataset):
    """Select one annotation group per image, then all four tiles, each epoch."""
    def __init__(self, dataset, records, seed=42):
        self.dataset, self.records, self.seed = dataset, list(records), int(seed)
        aliases = {str(Path(path).absolute()): index for index, path in enumerate(dataset.im_files)}
        expected = [str(Path(row['image']).absolute()) for row in self.records]
        if len(aliases) != len(dataset.im_files) or len(set(expected)) != len(expected):
            raise ValueError('Each observer/tile record requires its own image alias')
        if set(aliases) != set(expected) or len(dataset.im_files) != len(self.records):
            raise ValueError('Ultralytics discarded or added image/label records; refusing incomplete coverage')
        self.by_key = {}
        self.groups = defaultdict(set)
        for row in self.records:
            key = record_key(row)
            if key in self.by_key: raise ValueError('Duplicate physical image/group/tile record')
            self.by_key[key] = aliases[str(Path(row['image']).absolute())]
            self.groups[key[:2]].add(key[2])
        if not self.groups: raise ValueError('The full-pass training pool is empty')
        for (domain, filename), groups in self.groups.items():
            for group in groups:
                origins = {(k[3], k[4]) for k in self.by_key if k[:3] == (domain, filename, group)}
                if origins != TILE_ORIGINS:
                    raise ValueError('Every annotation group must have all four nonoverlapping 1024 tiles')
        self.set_epoch(0)

    def __len__(self): return 4 * len(self.groups)

    def __getattr__(self, name):
        if name == 'dataset': raise AttributeError(name)
        return getattr(self.dataset, name)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        if self.epoch < 0: raise ValueError('Epoch index must be nonnegative')
        keys = []
        for (domain, filename), groups in sorted(self.groups.items()):
            group = selected_group(groups, self.seed, domain, filename, self.epoch + 1)
            keys.extend((domain, filename, group, top, left) for top, left in sorted(TILE_ORIGINS))
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch + 1, 5928]))
        self.sample_keys = [keys[index] for index in rng.permutation(len(keys))]
        self.plan = [self.by_key[key] for key in self.sample_keys]
        self.domain_counts = dict(Counter(key[0] for key in self.sample_keys))
        if len(set(self.plan)) != len(self): raise ValueError('A full-pass epoch must visit every tile exactly once')

    def __getitem__(self, index): return self.dataset[self.plan[index]]


def training_options(config, dataset_yaml, local_root):
    cfg = dict(DEFAULTS, **config)
    if (cfg['yolo_batch'], cfg['yolo_workers']) != (1, 0):
        raise ValueError('The resumable T4 recipe requires YOLO batch1 and workers0')
    if cfg['yolo_imgsz'] != 1024 or cfg['yolo_accumulate'] != 4:
        raise ValueError('The full-pass recipe requires 1024 tiles and four microbatches per update')
    device = cfg['device']
    if str(device).startswith('cuda'): device = str(device).partition(':')[2] or 0
    return dict(data=str(dataset_yaml), epochs=int(cfg['epochs']), imgsz=1024,
        batch=1, nbs=4, workers=0, device=device, cache=False,
        project=str(local_root), name='ultralytics', exist_ok=True,
        optimizer='AdamW', lr0=cfg['yolo_lr'], weight_decay=cfg['yolo_weight_decay'],
        lrf=cfg['yolo_minimum_lr_ratio'], cos_lr=True, warmup_epochs=0.,
        amp=cfg['yolo_amp'], seed=cfg['seed'], deterministic=True, channels_last=False,
        patience=1000, augmentations=[], fliplr=.5, flipud=.5,
        degrees=0., translate=0., scale=0., shear=0., perspective=0.,
        hsv_h=0., hsv_s=0., hsv_v=.1, mosaic=0., close_mosaic=0, mixup=0.,
        cutmix=0., copy_paste=0., multi_scale=0., mask_ratio=4, overlap_mask=True,
        max_det=100, save=True, save_period=-1, val=False, plots=False, pretrained=True)


class AccumulationWindow:
    """Library calls every microbatch; this owns the exact update boundaries.

    Ultralytics' absolute-batch accumulation counter restarts its phase when an
    invocation resumes. Keeping its internal accumulation at one and averaging
    each explicit window avoids that off-by-three update on the resumed epoch.
    """
    def __init__(self, target=4):
        if target < 1: raise ValueError('Accumulation must be positive')
        self.target, self.pending = int(target), 0

    def step(self, parameters, perform_update, *, final=False):
        self.pending += 1
        if self.pending < self.target and not final: return False
        divisor = self.pending
        for parameter in parameters:
            if parameter.grad is not None: parameter.grad.div_(divisor)
        perform_update()
        self.pending = 0
        return True


def _checked_resume(store, state, identity):
    path = store.restore(state['last'])
    # This is only called after Drive has verified the owned checkpoint's bytes.
    value = torch.load(path, map_location='cpu', weights_only=False)
    if value.get('identity') != identity or value.get('completed_epoch') != state['epoch']:
        raise ValueError('YOLO checkpoint identity or completed epoch differs from the verified state')
    if value.get('epoch') != state['epoch'] - 1:
        raise ValueError('Ultralytics zero-based epoch metadata differs')
    if [row['epoch'] for row in value.get('history', [])] != list(range(1, state['epoch'] + 1)):
        raise ValueError('YOLO checkpoint history is incomplete')
    if value.get('best_epoch') != state.get('best_epoch', 0) or value.get('best_pq') != state.get('best_pq', -1.):
        raise ValueError('YOLO best checkpoint metadata differs from the verified state')
    for key in ('live_state', 'ema_full_state', 'optimizer', 'scheduler_state', 'rng_state'):
        if value.get(key) is None: raise ValueError('Incomplete resumable YOLO state: ' + key)
    return path, value


def _inference_payload(trainer, identity, completed):
    ema = deepcopy(trainer.ema.ema).float().cpu().to(memory_format=torch.contiguous_format)
    if hasattr(ema, 'criterion'): ema.criterion = None
    for name, tensor in ema.state_dict().items():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise FloatingPointError('Nonfinite EMA tensor cannot be checkpointed: ' + name)
    return dict(epoch=completed-1, completed_epoch=completed, identity=identity,
        model=None, ema=ema, updates=trainer.ema.updates, train_args=vars(trainer.args).copy(),
        version=ULTRALYTICS_VERSION, optimizer=None, best_fitness=0.,
        train_metrics={}, train_results={}, schema_version=1)


def _export_predictions(output, payload):
    """Mirror selected validation predictions without making them checkpoint authority."""
    predictions = payload.get('validation_predictions')
    if predictions is None: return
    import solar_a100_train_standalone as base
    base.export_best_predictions(output, dict(payload, epoch=payload['completed_epoch']))


def _result(output, state, cfg, identity):
    value = dict(status='completed' if state['epoch'] >= cfg['epochs'] else 'paused',
        identity=identity, model='YOLO26s-seg', epochs_completed=state['epoch'],
        fixed_final_epoch=cfg['epochs'], epoch_sampling='all_physical_images_four_tiles',
        best_pq=state.get('best_pq'), best_epoch=state.get('best_epoch'),
        last_checkpoint=state['last'], best_checkpoint=state.get('best'),
        initialization='official COCO yolo26s-seg.pt; no prior solar weights',
        initial_checkpoint_sha256=PRETRAINED_SHA256,
        resume_granularity='last server-verified completed epoch',
        checkpoint_selection='official validation PQ on fixed91; EMA weights',
        batch=1, gradient_accumulation=4)
    atomic_json(Path(output)/'result.json', value)
    return value


def run_yolo(config, dataset_yaml, metadata, output, identity, checkpoint_store,
             run_until=None, validator=None):
    """Train with fullpass data and a caller-provided official-PQ validator.

    ``validator(independent_yolo_wrapper, cfg)`` returns ``best, grid, predictions``
    in the same format as native validation. It may run inference safely: this
    wrapper contains a separate FP32 EMA snapshot, never the live training model.
    """
    from ultralytics import YOLO, __version__
    from ultralytics.models.yolo.segment.train import SegmentationTrainer
    from ultralytics.data.build import build_dataloader

    if __version__ != ULTRALYTICS_VERSION:
        raise ValueError('Use ultralytics==' + ULTRALYTICS_VERSION)
    if validator is None: raise ValueError('YOLO training requires the fixed91 official-PQ validator')
    cfg = dict(DEFAULTS, **config)
    target = cfg['epochs'] if run_until is None else int(run_until)
    if not 1 <= target <= cfg['epochs']: raise ValueError('run-until must lie within the fixed full schedule')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    local = Path(cfg['cache_root'])/'yolo_checkpoints'/identity[:16]
    if local.resolve().is_relative_to('/content/drive'):
        raise ValueError('Checkpoint serialization belongs on local disk, not Drive FUSE')
    local.mkdir(parents=True, exist_ok=True)
    options = training_options(cfg, dataset_yaml, local)
    state = checkpoint_store.latest_state()
    if state is None and (output/'state.json').exists():
        raise RuntimeError('Local YOLO state exists but verified remote state is missing; refusing a fresh restart')
    saved = None; restored_path = None
    if state:
        restored_path, saved = _checked_resume(checkpoint_store, state, identity)
        atomic_json(output/'history.json', saved['history'])
        if state.get('best'):
            best = torch.load(checkpoint_store.restore(state['best']), map_location='cpu', weights_only=False)
            if best.get('identity') != identity or best.get('completed_epoch') != state['best_epoch']:
                raise ValueError('Restored best YOLO checkpoint differs')
            _export_predictions(output, best); del best
        if state['epoch'] >= target:
            atomic_json(output/'state.json', state)
            print(f'YOLO epoch {state["epoch"]} already verified; no further training requested.', flush=True)
            return _result(output, state, cfg, identity)
    checkpoint_store.preflight(required_free_bytes=3_000_000_000)
    configure_cuda()
    initial_path = restored_path or download_generic_weights(Path(cfg['cache_root'])/'generic_weights')
    model = YOLO(str(initial_path), task='segment')
    if model.task != 'segment' or (saved is None and len(model.names) != 80):
        raise ValueError('Fresh initialization must be the pinned 80-class generic segmentation model')
    history = list(saved['history']) if saved else []
    best_pq = float(state.get('best_pq', -1.)) if state else -1.
    best_epoch = int(state.get('best_epoch', 0)) if state else 0
    best_record = state.get('best') if state else None

    class FullPassTrainer(SegmentationTrainer):
        def get_dataloader(self, dataset_path, batch_size=1, rank=-1, mode='train'):
            if rank != -1: raise ValueError('The T4 protocol is single GPU')
            source = self.build_dataset(dataset_path, mode, batch_size)
            if mode == 'train': source = FullPassYoloDataset(source, metadata['records'], cfg['seed'])
            return build_dataloader(source, batch=batch_size, workers=0, shuffle=False,
                rank=-1, device=self.device)

        def validate(self): return {}, 0.
        def final_eval(self): pass

        def _build_train_pipeline(self):
            if getattr(self, '_fullpass_pipeline_built', False):
                raise RuntimeError('OOM retry stopped; resume the last verified epoch without changing coverage')
            self._fullpass_pipeline_built = True
            return super()._build_train_pipeline()

        def _handle_nan_recovery(self, epoch):
            if self.loss is not None and not torch.isfinite(self.loss).all():
                raise FloatingPointError('Nonfinite YOLO loss; last verified completed epoch retained')
            return False

        def optimizer_step(self):
            def update():
                previous = float(self.scaler.get_scale())
                super(FullPassTrainer, self).optimizer_step()
                self._synthetic_step_attempts += 1
                self._synthetic_steps_completed += int(float(self.scaler.get_scale()) >= previous)
            self._fullpass_window.step(self.model.parameters(), update,
                final=self._synthetic_completed_batches + 1 == len(self.train_loader))

        def save_model(self):
            nonlocal state, best_pq, best_epoch, best_record
            completed = int(self.epoch) + 1
            if self._synthetic_completed_batches != len(self.train_loader) or self._fullpass_window.pending:
                raise RuntimeError('An incomplete epoch cannot become a resumable checkpoint')
            if self._synthetic_steps_completed == 0:
                raise FloatingPointError('All optimizer updates were skipped; retain previous completed epoch')
            expected_updates = (len(self.train_loader) + cfg['yolo_accumulate'] - 1)//cfg['yolo_accumulate']
            if self._synthetic_step_attempts != expected_updates:
                raise RuntimeError('YOLO accumulation did not cover exactly the full epoch')
            inference = _inference_payload(self, identity, completed)
            infer_path = local/'validation_ema.pt'
            atomic_torch(infer_path, inference)
            row = dict(epoch=completed,
                train_losses={k:float(v) for k,v in self.label_loss_items(self.tloss, prefix='train').items()},
                learning_rates=[float(g['lr']) for g in self.optimizer.param_groups],
                microbatches=len(self.train_loader), optimizer_updates=self._synthetic_steps_completed,
                optimizer_update_attempts=self._synthetic_step_attempts,
                amp_skipped_updates=self._synthetic_step_attempts-self._synthetic_steps_completed,
                sampled_domains=self.train_loader.dataset.domain_counts,
                physical_images=len(self.train_loader.dataset.groups),
                epoch_sampling='all_physical_images_four_tiles',
                elapsed_seconds=time.monotonic()-self._fullpass_started)
            new_best = None
            if completed == 1 or completed % cfg['validate_every'] == 0 or completed == cfg['epochs']:
                snapshot = YOLO(str(infer_path), task='segment')
                try:
                    best, grid, predictions = validator(snapshot, cfg)
                finally:
                    del snapshot; gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                if not np.isfinite(best['pq']): raise ValueError('Validation PQ must be finite')
                row.update(validation=best, validation_grid=grid)
                if best['pq'] > best_pq:
                    best_pq, best_epoch = float(best['pq']), completed
                    new_best = dict(inference, validation=best, validation_grid=grid,
                        validation_predictions=predictions, config=cfg.copy())
                    path = local/'best.pt'; atomic_torch(path, new_best)
                    best_record = checkpoint_store.publish_checkpoint(path,
                        f'checkpoints/best_epoch{completed:03}.pt', metadata=dict(kind='best', epoch=completed))
                    self.best.parent.mkdir(parents=True, exist_ok=True)
                    atomic_torch(self.best, new_best)
            history.append(row)
            payload = dict(inference, fixed_final_epoch=cfg['epochs'],
                live_state=cpu_copy(self.model.state_dict()),
                ema_full_state=cpu_copy(self.ema.ema.state_dict()),
                optimizer=cpu_copy(self.optimizer.state_dict()), scaler=cpu_copy(self.scaler.state_dict()),
                scheduler_state=cpu_copy(self.scheduler.state_dict()), rng_state=capture_rng(),
                history=history, best_pq=best_pq, best_epoch=best_epoch,
                initial_checkpoint_sha256=PRETRAINED_SHA256)
            path = local/'last.pt'; atomic_torch(path, payload)
            record = checkpoint_store.publish_checkpoint(path,
                f'checkpoints/last_epoch{completed:03}.pt', metadata=dict(kind='last', epoch=completed))
            state = checkpoint_store.commit_state(dict(epoch=completed, last=record,
                best=best_record, best_pq=best_pq, best_epoch=best_epoch), expected_previous_epoch=completed-1)
            # These files are mirrors only. Future resumes query server commits.
            atomic_json(output/'state.json', state); atomic_json(output/'history.json', history)
            if new_best is not None: _export_predictions(output, new_best)
            self.last.parent.mkdir(parents=True, exist_ok=True)
            atomic_torch(self.last, payload)
            _result(output, state, cfg, identity)
            print(f'YOLO checkpoint confirmed on Drive: epoch {completed}; {record["file_id"]}', flush=True)
            print(json.dumps(row), flush=True)
            return False  # All saving is complete; do not fire unrelated model-save callbacks.

    def restore(trainer):
        if trainer.batch_size != 1: raise ValueError('Expected physical batch size one')
        if saved is not None:
            trainer.model.load_state_dict(saved['live_state'], strict=True)
            trainer.ema.ema.load_state_dict(saved['ema_full_state'], strict=True)
            trainer.ema.updates = saved['updates']
            trainer.optimizer.load_state_dict(saved['optimizer'])
            trainer.scaler.load_state_dict(saved['scaler'])
            trainer.scheduler.load_state_dict(saved['scheduler_state'])
            restore_rng(saved['rng_state'])
        # Weight-decay setup used nbs4/accumulate4; the microbatch hook now owns
        # averaging and boundaries rather than the library's invocation counter.
        trainer.accumulate = 1

    def begin_epoch(trainer):
        trainer.train_loader.dataset.set_epoch(trainer.epoch)
        seed_epoch(trainer, cfg['seed'])
        trainer._fullpass_window = AccumulationWindow(cfg['yolo_accumulate'])
        trainer._fullpass_started = time.monotonic()
        print(f'YOLO epoch {trainer.epoch+1}/{cfg["epochs"]}: '
            f'{len(trainer.train_loader.dataset)} tiles; {trainer.train_loader.dataset.domain_counts}', flush=True)

    model.add_callback('on_train_start', restore)
    model.add_callback('on_train_epoch_start', begin_epoch)
    model.add_callback('on_train_batch_end', lambda t:setattr(t, '_synthetic_completed_batches', t._synthetic_completed_batches+1))
    model.add_callback('on_fit_epoch_end', lambda t:setattr(t, 'stop', t.epoch+1 >= target))
    atomic_json(output/'config.json', dict(identity=identity, options=options,
        initial_checkpoint_sha256=PRETRAINED_SHA256, records=len(metadata['records'])))
    print(f'YOLO fresh COCO run: verified completed epoch {state["epoch"] if state else 0}; '
        f'run through {target}/{cfg["epochs"]}.', flush=True)
    model.train(trainer=FullPassTrainer, **options, resume=str(restored_path) if saved is not None else False)
    if not state or state['epoch'] < target:
        raise RuntimeError('YOLO exited before the requested epoch was committed')
    result = _result(output, state, cfg, identity)
    del model, saved; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result
