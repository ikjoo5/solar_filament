"""CHASE/GONG continuation of the deployed YOLO26s segmentation detector.

CHASE connected components are proxy instances, never measured instance truth.
Only explicitly annotated overlap exclusions can remove a CHASE observation.
Each full-pass epoch visits every accepted CHASE image once; completed-epoch
state is durable. An explicitly selected bounded mode supports old experiments.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import fcntl
import gc
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('TORCH_CUDNN_V8_API_DISABLED', '1')
import cv2
import numpy as np
from pycocotools import mask as mu
import torch
from torch.utils.data import Dataset

from targeted_synthetic_training import (ORIGINAL_SHA, INPUT_SHA, ULTRALYTICS_VERSION,
    atomic_json, atomic_torch, capture_rng, compact_checkpoint, cpu_copy, digest,
    fingerprint, restore_rng, runtime_versions, seed_epoch)

DEFAULTS = dict(training_domain='chase_only', epochs=60, imgsz=2048, batch=1, nbs=1, workers=0,
    epoch_sampling='full_pass', steps_per_epoch=695, chase_ratio=1., real_finish_epochs=0,
    lr=2e-5, weight_decay=.01, seed=42, device=0, amp=True)
SOURCES = ('chase_yolo_training.py', 'chase_transfer_data.py',
    'targeted_synthetic_training.py', 'solar_a100_train_standalone.py')


def configure_cuda():
    """Match the reviewed T4/CUDA13 depthwise-convolution workaround."""
    torch.backends.cudnn.benchmark = False
    if not getattr(torch.nn.Conv2d._conv_forward, '_chase_depthwise_safe', False):
        original = torch.nn.Conv2d._conv_forward
        def safe(self, inputs, weight, bias):
            if inputs.is_cuda and self.groups > 1 and self.groups == self.in_channels:
                with torch.backends.cudnn.flags(enabled=False):
                    return original(self, inputs, weight, bias)
            return original(self, inputs, weight, bias)
        safe._chase_depthwise_safe = True
        torch.nn.Conv2d._conv_forward = safe


def normalize_config(config):
    cfg = dict(DEFAULTS, **config)
    if cfg['training_domain'] not in ('mixed_replay', 'chase_only'):
        raise ValueError('training_domain must be mixed_replay or chase_only')
    if cfg['epoch_sampling'] not in ('full_pass', 'bounded'):
        raise ValueError('epoch_sampling must be full_pass or bounded')
    if cfg['epoch_sampling'] == 'full_pass' and cfg['training_domain'] != 'chase_only':
        raise ValueError('Full-pass epochs require CHASE-only training')
    if cfg['training_domain'] == 'chase_only':
        cfg['chase_ratio'] = config.get('chase_ratio', 1.)
        cfg['real_finish_epochs'] = config.get('real_finish_epochs', 0)
    else:
        cfg['chase_ratio'] = config.get('chase_ratio', .2)
        cfg['real_finish_epochs'] = config.get('real_finish_epochs', 6)
    for key in ('epochs', 'imgsz', 'batch', 'nbs', 'steps_per_epoch'):
        if not isinstance(cfg[key], int) or isinstance(cfg[key], bool) or cfg[key] < 1:
            raise ValueError(key+' must be a positive integer')
    if cfg['batch'] != 1 or cfg['nbs'] != 1 or cfg['workers'] != 0:
        raise ValueError('Completed-epoch protocol requires batch1, nbs1, workers0')
    if cfg['imgsz'] != 2048:
        raise ValueError('This experiment keeps the native 2048 YOLO input')
    if cfg['training_domain'] == 'chase_only':
        if cfg['chase_ratio'] != 1. or cfg['real_finish_epochs'] != 0:
            raise ValueError('CHASE-only requires ratio1 and no GONG-only finishing epochs')
    elif not 0 < cfg['chase_ratio'] < 1 or not 0 <= cfg['real_finish_epochs'] < cfg['epochs']:
        raise ValueError('Invalid CHASE replay schedule')
    if not 0 < cfg['lr'] <= 1e-3 or cfg['weight_decay'] < 0:
        raise ValueError('Invalid optimizer settings')
    paths = ['output_root', 'yolo_checkpoint', 'train_split', 'val_split',
             'chase_root', 'chase_manifest', 'cache_root']
    if cfg['training_domain'] == 'mixed_replay': paths += ['images', 'annotations']
    for key in paths:
        if not cfg.get(key): raise ValueError('Missing path: '+key)
        cfg[key] = str(Path(cfg[key]).expanduser())
    out = Path(cfg['output_root']).resolve()
    if Path(cfg['yolo_checkpoint']).resolve().is_relative_to(out) or (cfg.get('images') and out == Path(cfg['images']).resolve()):
        raise ValueError('Use a new output folder that cannot replace source weights/images')
    if Path(cfg['cache_root']).resolve().is_relative_to(Path('/content/drive')):
        raise ValueError('Processed data caches belong on Colab local disk')
    return cfg


def read_training_groups(data, train_path, val_path):
    """Verify physical-image disjointness and keep all rater groups together."""
    by_id = {str(row['id']): row for row in data['images']}
    if len(by_id) != len(data['images']): raise ValueError('Duplicate annotation image ID')
    roles, ids = {}, {}
    for role, path in (('train', train_path), ('validation', val_path)):
        with Path(path).open(newline='', encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        selected = [row['image_id'] for row in rows]
        if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(by_id):
            raise ValueError('Empty, duplicate or unknown split annotation ID')
        if any(row['file_name'] != by_id[row['image_id']]['file_name'] for row in rows):
            raise ValueError('Split filename mismatch')
        names = {row['file_name'] for row in rows}
        if any(Path(name).name != name or Path(name).suffix.lower() != '.jpeg' for name in names):
            raise ValueError('Expected original JPEG basenames')
        if set(selected) != {sid for sid, row in by_id.items() if row['file_name'] in names}:
            raise ValueError('All annotator siblings must be in the same split')
        ids[role], roles[role] = set(selected), names
    if ids['train'] & ids['validation'] or roles['train'] & roles['validation']:
        raise ValueError('Train/validation leakage')
    if ids['train'] | ids['validation'] != set(by_id): raise ValueError('Incomplete split union')
    if (len(roles['train']), len(roles['validation'])) != (616, 91):
        raise ValueError('Expected frozen 616/91 physical-image split')
    groups = defaultdict(list)
    for sid in sorted(ids['train']): groups[by_id[sid]['file_name']].append(sid)
    return dict(groups), by_id


def decode_annotation(annotation, height=2048, width=2048):
    seg = annotation['segmentation']
    if isinstance(seg, list): value = mu.merge(mu.frPyObjects(seg, height, width))
    elif isinstance(seg['counts'], list): value = mu.frPyObjects(seg, height, width)
    else:
        value = dict(seg)
        if isinstance(value['counts'], str): value['counts'] = value['counts'].encode('ascii')
    mask = mu.decode(value).astype(np.uint8)
    if mask.shape != (height, width): raise ValueError('Wrong annotation dimensions')
    return mask


def mask_polygon(mask):
    """One polygon per supplied instance, including tiny/disconnected positives.

    Doubling the grid gives one-pixel objects nonzero polygon area. External
    contours fill holes; disconnected pieces use the library's bridge encoding.
    Neither operation is evidence of separate true CHASE filament instances.
    """
    from ultralytics.data.converter import merge_multi_segment
    mask = np.ascontiguousarray(mask > 0, dtype=np.uint8)
    if mask.ndim != 2 or not mask.any(): raise ValueError('Nonempty 2-D mask required')
    enlarged = cv2.resize(mask, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(enlarged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    parts = [c.reshape(-1, 2).astype(np.float64)/2 for c in contours]
    if not parts or any(len(p) < 3 or cv2.contourArea(p.astype(np.float32)) <= 0 for p in parts):
        raise ValueError('A positive component could not be represented; never silently drop it')
    xy = np.concatenate(merge_multi_segment([p.reshape(-1).tolist() for p in parts]), axis=0) if len(parts) > 1 else parts[0]
    # Normalized text and float32 roundtrip must not round integer positions down.
    xy += .001
    height, width = mask.shape
    xy = np.clip(xy/np.array([width, height]), 0, 1)
    line = '0 '+' '.join(f'{v:.9f}' for v in xy.reshape(-1))
    restored = polygon_mask(line, mask.shape)
    intersection = int((restored & mask).sum()); union = int((restored | mask).sum())
    report = dict(source_area=int(mask.sum()), polygon_area=int(restored.sum()),
        added_pixels=int((restored & (1-mask)).sum()),
        lost_pixels=int((mask & (1-restored)).sum()),
        representation_iou=intersection/union, contours=len(parts))
    return line, report


def polygon_mask(line, shape):
    coords = np.asarray(list(map(float, line.split()[1:])), np.float64).reshape(-1, 2)
    coords *= np.array([shape[1], shape[0]])
    result = np.zeros(shape, np.uint8)
    cv2.fillPoly(result, [coords.astype(np.int32)], 1)
    return result


def binary_proxy_polygons(mask):
    """Preserve all connected regions, including single pixels; no morphology filter."""
    count, labels = cv2.connectedComponents(np.ascontiguousarray(mask > 0, dtype=np.uint8), connectivity=8)
    lines, reports = [], []
    for label in range(1, count):
        line, report = mask_polygon(labels == label)
        lines.append(line); reports.append(report)
    return lines, reports


def prepare_dataset(cfg):
    from chase_transfer_data import prepare_chase, ChaseDataset, load_manual_exclusions
    chase_only = cfg['training_domain'] == 'chase_only'
    # In CHASE-only mode even parsing GONG object annotations or loading GONG
    # pixels is unnecessary. The immutable split CSV hashes remain audit guards.
    verified_inputs = {key:value for key,value in INPUT_SHA.items()
                       if not chase_only or key in ('train_split', 'val_split')}
    for key, expected in verified_inputs.items():
        if digest(cfg[key]) != expected: raise ValueError('Frozen GONG source changed: '+key)
    groups = {}
    anns = defaultdict(list)
    if not chase_only:
        from solar_a100_train_standalone import limb
        data = json.loads(Path(cfg['annotations']).read_text())
        groups, _ = read_training_groups(data, cfg['train_split'], cfg['val_split'])
        train_ids = {sid for values in groups.values() for sid in values}
        for ann in data['annotations']:
            if str(ann['image_id']) in train_ids: anns[str(ann['image_id'])].append(ann)
    records, report_path = prepare_chase(cfg['chase_root'], cfg['cache_root'], cfg['chase_manifest'])
    manual = load_manual_exclusions(cfg['manual_exclusions']) if cfg.get('manual_exclusions') else {}
    if not set(manual) <= {r['id'] for r in records}: raise ValueError('Unknown explicitly excluded CHASE ID')
    # YOLO segmentation has no pixel-ignore loss. Exclude the entire explicitly
    # marked observation instead of turning its uncertain pixels into negatives.
    accepted = [r for r in records if not manual.get(r['id'])]
    if not accepted: raise ValueError('No CHASE observations remain after explicit exclusions')
    source_hashes = {name: digest(Path(cfg['images'])/name) for name in sorted(groups)}
    key = fingerprint(dict(training_domain=cfg['training_domain'], inputs=verified_inputs,
        images=source_hashes, records=records, manual=manual,
        exporter=digest(__file__), alignment=digest(Path(__file__).with_name('chase_transfer_data.py'))))
    root = Path(cfg['cache_root'])/'chase_yolo_dataset'/key[:16]
    images = root/'images/train'; labels = root/'labels/train'
    images.mkdir(parents=True, exist_ok=True); labels.mkdir(parents=True, exist_ok=True)
    manifest_path = root/'manifest.json'; yaml = root/'dataset.yaml'
    if manifest_path.is_file() and yaml.is_file():
        previous = json.loads(manifest_path.read_text())
        if previous['identity'] == key and all(Path(r['image']).is_file() and
                Path(r['label']).is_file() and digest(r['image']) == r.get('image_sha256')
                and digest(r['label']) == r.get('label_sha256') for r in previous['records']):
            return yaml, previous
    if not chase_only: (root/'physical_gong').mkdir(exist_ok=True)
    chase = ChaseDataset(accepted, crop=2048, length=1, cache_dir=Path(cfg['cache_root'])/'chase_aligned')
    exported, audits = [], []
    for index, name in enumerate(sorted(groups), 1):
        physical = root/'physical_gong'/(Path(name).stem+'.png')
        # Rebuild from hashed originals whenever the prepared-cache audit fails.
        # Do not trust an existing, possibly damaged, processed PNG here.
        original = cv2.imread(str(Path(cfg['images'])/name), cv2.IMREAD_GRAYSCALE)
        if original is None or original.shape != (2048, 2048):
            raise ValueError('Wrong original GONG image: '+name)
        if not cv2.imwrite(str(physical), limb(original)): raise OSError('Failed to write GONG cache')
        for sid in groups[name]:
            alias = 'gong_'+fingerprint(sid)[:20]
            image_path, label_path = images/(alias+'.png'), labels/(alias+'.txt')
            if not image_path.exists(): image_path.symlink_to(physical.resolve())
            lines, component_reports = [], []
            for ann in anns[sid]:
                mask = decode_annotation(ann)
                if not mask.any(): raise ValueError('Empty GONG instance annotation: '+str(ann['id']))
                line, report = mask_polygon(mask)
                lines.append(line); component_reports.append(report)
            label_path.write_text('\n'.join(lines)+('\n' if lines else ''))
            exported.append(dict(domain='gong', physical=name, group=sid,
                image=str(image_path.absolute()), label=str(label_path.absolute()),
                image_sha256=digest(image_path), label_sha256=digest(label_path)))
            audits.append(dict(domain='gong', id=sid, proxy_instances=False, objects=component_reports))
        if index % 100 == 0 or index == len(groups): print(f'YOLO GONG 준비 {index}/{len(groups)}', flush=True)
    for index, record in enumerate(accepted, 1):
        image, foreground, valid = chase.aligned(record)
        if np.any(foreground & (1-valid)):
            raise ValueError('CHASE foreground lies outside supervised alignment: '+record['id'])
        # Empty warped pixels are deterministic black background, not a removed
        # filament. No source positive is filtered by component area/shape.
        alias = 'chase_'+record['id']; image_path, label_path = images/(alias+'.png'), labels/(alias+'.txt')
        if not cv2.imwrite(str(image_path), image): raise OSError('Failed to write CHASE cache')
        lines, component_reports = binary_proxy_polygons(foreground)
        label_path.write_text('\n'.join(lines)+('\n' if lines else ''))
        exported.append(dict(domain='chase', physical=record['id'], group=record['id'],
            image=str(image_path.absolute()), label=str(label_path.absolute()),
            image_sha256=digest(image_path), label_sha256=digest(label_path)))
        audits.append(dict(domain='chase', id=record['id'], proxy_instances=True, objects=component_reports))
        if index % 50 == 0 or index == len(accepted): print(f'YOLO CHASE 준비 {index}/{len(accepted)}', flush=True)
    expected = {Path(row['image']).name for row in exported}
    if {p.name for p in images.glob('*.png')} != expected: raise ValueError('Unexpected images in training cache')
    train_list = root/'train.txt'; train_list.write_text('\n'.join(row['image'] for row in exported)+'\n')
    # Library constructs a validator even when disabled. Its placeholder uses
    # one active-domain TRAIN image (CHASE for CHASE-only), never a held-out image.
    placeholder = root/'unused_validation.txt'; placeholder.write_text(exported[0]['image']+'\n')
    yaml.write_text(json.dumps(dict(path=str(root.absolute()), train=str(train_list.absolute()),
        val=str(placeholder.absolute()), names={0:'filament'})))
    all_audits = [item for row in audits for item in row['objects']]
    metadata = dict(identity=key, training_domain=cfg['training_domain'], records=exported,
        inputs=verified_inputs, source_images_sha256=source_hashes,
        train_photos=len(groups)+len(accepted), gong_training_photos=len(groups),
        validation_photos=91, validation_used_for_training=False,
        chase_count=len(accepted), chase_source_count=len(records),
        chase_source_records=records, chase_report=str(report_path),
        explicit_excluded_ids=sorted(k for k,v in manual.items() if v),
        heuristic_overlap_filter=False, photometric_chase_preprocessing=False,
        gong_preprocessing=None if chase_only else 'original limb_only', annotation_audit=audits,
        proxy_instance_warning='CHASE binary connected regions are approximations, not true instance IDs.',
        polygon_limitations='External holes are filled and disconnected GONG pieces bridged; YOLO additionally resamples polygons and downsamples masks4x.',
        polygon_added_pixels=sum(r['added_pixels'] for r in all_audits),
        polygon_lost_pixels=sum(r['lost_pixels'] for r in all_audits))
    atomic_json(manifest_path, metadata)
    return yaml, metadata


def resolve_epoch_sampling(cfg, metadata):
    """Bind effective full-pass length to accepted physical observations.

    A declared 695-source schedule may shrink only by explicit exclusions.
    An unrelated bounded budget such as400 is an error in full-pass mode.
    """
    cfg = dict(cfg)
    observations = [r['physical'] for r in metadata['records'] if r['domain'] == 'chase']
    accepted = len(set(observations))
    source_count = int(metadata.get('chase_source_count',
        len(metadata.get('chase_source_records', [])) or accepted))
    if not accepted or source_count < accepted: raise ValueError('Invalid accepted CHASE observation count')
    cfg['requested_steps_per_epoch'] = int(cfg['steps_per_epoch'])
    cfg['accepted_chase_observations'] = accepted
    if cfg['epoch_sampling'] == 'full_pass':
        if cfg['training_domain'] != 'chase_only' or any(r['domain'] != 'chase' for r in metadata['records']):
            raise ValueError('Full-pass training cannot contain GONG records')
        if len(observations) != accepted:
            raise ValueError('Full-pass training needs exactly one record per physical CHASE observation')
        excluded = metadata.get('explicit_excluded_ids', [])
        if source_count-accepted != len(set(excluded)):
            raise ValueError('An observation disappeared without an explicit exclusion')
        if cfg['steps_per_epoch'] not in (source_count, accepted):
            raise ValueError('Full-pass steps_per_epoch must equal the source or accepted observation count; use epoch_sampling=bounded for a budget')
        if cfg['steps_per_epoch'] != accepted:
            print(f'명시적 제외 {source_count-accepted}장 반영: epoch당 {cfg["steps_per_epoch"]} → {accepted}장 전체 순회.', flush=True)
        cfg['steps_per_epoch'] = accepted
    cfg['effective_steps_per_epoch'] = int(cfg['steps_per_epoch'])
    return cfg


class BalancedEpochDataset(Dataset):
    """Full-pass CHASE permutation, or an explicitly selected bounded budget."""
    def __init__(self, dataset, records, cfg):
        self.dataset, self.cfg = dataset, cfg
        paths = {str(Path(row['image']).absolute()): row for row in records}
        self.groups = {'gong': defaultdict(list), 'chase': defaultdict(list)}
        for index, name in enumerate(dataset.im_files):
            row = paths.get(str(Path(name).absolute()))
            if row is None: raise ValueError('Unexpected YOLO training image: '+name)
            if cfg['training_domain'] == 'chase_only' and row['domain'] != 'chase':
                raise ValueError('CHASE-only training cannot contain GONG records')
            self.groups[row['domain']][row['physical']].append(index)
        if len(dataset.im_files) != len(records):
            raise ValueError('Ultralytics discarded records; inspect invalid image/label warnings')
        if cfg['epoch_sampling'] == 'full_pass':
            if (cfg['training_domain'] != 'chase_only' or self.groups['gong']
                    or len(self.groups['chase']) != cfg['steps_per_epoch']
                    or any(len(indices) != 1 for indices in self.groups['chase'].values())):
                raise ValueError('Full-pass epoch length must equal the unique accepted CHASE image count')
        self.order = {}
        for i, domain in enumerate(('gong', 'chase')):
            names = sorted(self.groups[domain])
            if not names and not (cfg['training_domain'] == 'chase_only' and domain == 'gong'):
                raise ValueError('Empty replay domain: '+domain)
            rng = np.random.default_rng(np.random.SeedSequence([cfg['seed'], i, 1701]))
            self.order[domain] = [names[j] for j in rng.permutation(len(names))]
        self.set_epoch(0)

    def __len__(self): return int(self.cfg['steps_per_epoch'])

    def __getattr__(self, name):
        if name == 'dataset': raise AttributeError(name)
        return getattr(self.dataset, name)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        budget = len(self); mixed_epochs = self.cfg['epochs']-self.cfg['real_finish_epochs']
        if self.cfg['training_domain'] == 'chase_only':
            counts = {'gong':0, 'chase':budget}
            starts = {'gong':0, 'chase':epoch*budget}
        else:
            chase_budget = round(budget*self.cfg['chase_ratio'])
            chase_count = chase_budget if epoch < mixed_epochs else 0
            counts = {'gong': budget-chase_count, 'chase': chase_count}
            starts = {'chase': min(epoch, mixed_epochs)*chase_budget,
                'gong': min(epoch, mixed_epochs)*(budget-chase_budget)+max(0, epoch-mixed_epochs)*budget}
        plan, names = [], []
        for domain in ('gong', 'chase'):
            order = self.order[domain]
            for offset in range(counts[domain]):
                position = starts[domain]+offset
                physical = order[position % len(order)]
                variants = self.groups[domain][physical]
                plan.append(variants[(position//len(order)) % len(variants)])
                names.append((domain, physical))
        rng = np.random.default_rng(np.random.SeedSequence([self.cfg['seed'], epoch, 814]))
        permutation = rng.permutation(budget)
        self.plan = [plan[i] for i in permutation]
        self.sample_names = [names[i] for i in permutation]
        self.domain_counts = counts

    def __getitem__(self, index): return self.dataset[self.plan[index]]


def training_options(cfg, dataset_yaml):
    return dict(data=str(dataset_yaml), epochs=cfg['epochs'], imgsz=cfg['imgsz'],
        batch=1, nbs=1, workers=0, device=cfg['device'], cache=False,
        project=str(Path(cfg['cache_root'])/'chase_yolo_training'), name='continuation', exist_ok=True,
        optimizer='AdamW', lr0=cfg['lr'], weight_decay=cfg['weight_decay'], lrf=.1, cos_lr=True,
        warmup_epochs=0., amp=cfg['amp'], seed=cfg['seed'], deterministic=True,
        channels_last=False, patience=1000, augmentations=[], fliplr=0., flipud=0.,
        degrees=0., translate=0., scale=0., shear=0., perspective=0.,
        hsv_h=0., hsv_s=0., hsv_v=0., mosaic=0., close_mosaic=0, mixup=0.,
        cutmix=0., copy_paste=0., multi_scale=0., mask_ratio=4, overlap_mask=True,
        max_det=100, save=True, save_period=-1, val=False, plots=False, pretrained=True)


def train(config, run_until=None):
    cfg = normalize_config(config)
    folder = Path(cfg['output_root']); folder.mkdir(parents=True, exist_ok=True)
    with (folder/'.training.lock').open('a') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc: raise RuntimeError('Another YOLO process is using this output folder') from exc
        return _train_locked(cfg, run_until)


def _train_locked(cfg, run_until):
    from ultralytics import YOLO, __version__
    from ultralytics.models.yolo.segment.train import SegmentationTrainer
    from ultralytics.data.build import build_dataloader
    configure_cuda()
    if __version__ != ULTRALYTICS_VERSION: raise ValueError('Use ultralytics=='+ULTRALYTICS_VERSION)
    if digest(cfg['yolo_checkpoint']) != ORIGINAL_SHA: raise ValueError('Use the deployed original YOLO best.pt')
    epochs = cfg['epochs']; cap = epochs if run_until is None else int(run_until)
    if not 1 <= cap <= epochs: raise ValueError('run_until must be within the fixed epoch schedule')
    dataset_yaml, metadata = prepare_dataset(cfg)
    cfg = resolve_epoch_sampling(cfg, metadata)
    folder = Path(cfg['output_root']); atomic_json(folder/'dataset_report.json', metadata)
    options = training_options(cfg, dataset_yaml)
    fixed = dict(dataset_identity=metadata['identity'], options=options,
        sampler={k:cfg[k] for k in ('training_domain','epoch_sampling','steps_per_epoch',
            'requested_steps_per_epoch','effective_steps_per_epoch','accepted_chase_observations',
            'chase_ratio','real_finish_epochs','epochs','seed')},
        initial_checkpoint_sha256=ORIGINAL_SHA, sources={n:digest(Path(__file__).with_name(n)) for n in SOURCES},
        runtime=runtime_versions(), safe_depthwise_cuda=True,
        checkpoint_selection='fixed completed epoch, not mAP or validation PQ',
        scope=cfg['training_domain']+'; original YOLO continuation; fixed91 held out; native/finalizer separate',
        resume='full precision live/EMA/optimizer/scaler/scheduler/RNG at completed epoch',
        bitwise_GPU_trajectory_guaranteed=False)
    identity = fingerprint(fixed); config_path = folder/'training_config.json'
    if config_path.exists() and json.loads(config_path.read_text())['identity'] != identity:
        raise ValueError('Training data/code/runtime/settings changed; use a new output_root')
    atomic_json(config_path, dict(identity=identity, settings=fixed))
    last = folder/'last_full.pt'; stage = folder/f'epoch{cap:03d}.pt'
    saved = torch.load(last, map_location='cpu', weights_only=False) if last.exists() else None
    if saved is not None:
        if saved.get('chase_yolo_identity') != identity: raise ValueError('Wrong resume checkpoint identity')
        if [r['epoch'] for r in saved['history']] != list(range(1, saved['completed_epoch']+1)):
            raise ValueError('Checkpoint history is not consecutive')
        atomic_json(folder/'history.json', saved['history'])
        if saved['completed_epoch'] > cap and not stage.exists(): raise ValueError('Earlier requested stage is unavailable')
        if saved['completed_epoch'] == cap and not stage.exists(): atomic_torch(stage, compact_checkpoint(saved))

    def result():
        payload = torch.load(stage, map_location='cpu', weights_only=False)
        if payload.get('chase_yolo_identity') != identity or payload.get('completed_epoch') != cap:
            raise ValueError('Wrong completed stage')
        value = dict(status='complete' if cap == epochs else 'paused', epochs_completed=cap,
            fixed_final_epoch=epochs, checkpoint=str(stage), checkpoint_sha256=digest(stage), identity=identity,
            training_domain=cfg['training_domain'],
            epoch_sampling=cfg['epoch_sampling'],
            requested_steps_per_epoch=cfg['requested_steps_per_epoch'],
            effective_steps_per_epoch=cfg['effective_steps_per_epoch'],
            accepted_chase_observations=cfg['accepted_chase_observations'],
            checkpoint_selection='fixed completed epoch; no checkpoint selection by mAP',
            steps_per_epoch=cfg['steps_per_epoch'], batch=1, optimizer_accumulation=1)
        atomic_json(folder/'result.json', value); return value
    if stage.exists(): return result()

    class FixedTrainer(SegmentationTrainer):
        def get_dataloader(self, dataset_path, batch_size=1, rank=-1, mode='train'):
            if rank != -1: raise ValueError('This fixed protocol is single-GPU only')
            dataset = self.build_dataset(dataset_path, mode, batch_size)
            if mode == 'train': dataset = BalancedEpochDataset(dataset, metadata['records'], cfg)
            return build_dataloader(dataset, batch=batch_size, workers=0, shuffle=False,
                rank=-1, device=self.device)
        def validate(self): return {}, 0.
        def final_eval(self): pass
        def _build_train_pipeline(self):
            if getattr(self, '_chase_pipeline_built', False):
                raise RuntimeError('OOM retry stopped; resume the last completed epoch without silent data/batch changes')
            self._chase_pipeline_built = True
            return super()._build_train_pipeline()
        def _handle_nan_recovery(self, epoch):
            if self.loss is not None and not torch.isfinite(self.loss).all():
                raise FloatingPointError('Nonfinite loss; last completed checkpoint retained')
            return False
        def optimizer_step(self):
            before = float(self.scaler.get_scale()); super().optimizer_step()
            self._synthetic_step_attempts += 1
            self._synthetic_steps_completed += int(float(self.scaler.get_scale()) >= before)

    model = YOLO(str(last if saved is not None else cfg['yolo_checkpoint']), task='segment')
    if model.task != 'segment' or len(model.names) != 1: raise ValueError('Expected original single-class YOLO segment model')
    def restore(trainer):
        if saved is not None:
            trainer.model.load_state_dict(saved['live_state'])
            trainer.ema.ema.load_state_dict(saved['ema_full_state']); trainer.ema.updates = saved['updates']
            trainer.optimizer.load_state_dict(saved['optimizer']); trainer.scaler.load_state_dict(saved['scaler'])
            trainer.scheduler.load_state_dict(saved['scheduler_state']); restore_rng(saved['rng_state'])
        if trainer.accumulate != 1 or trainer.batch_size != 1: raise ValueError('Expected batch1, accumulation1')
    def begin_epoch(trainer):
        trainer.train_loader.dataset.set_epoch(trainer.epoch)
        seed_epoch(trainer, cfg['seed'])
        print(f'YOLO epoch {trainer.epoch+1}/{epochs}: {trainer.train_loader.dataset.domain_counts}', flush=True)
    def save(trainer):
        if trainer._synthetic_completed_batches != len(trainer.train_loader):
            raise RuntimeError('Incomplete epoch cannot become a resumable checkpoint')
        if trainer._synthetic_steps_completed == 0:
            raise FloatingPointError('Every optimizer update was skipped; retain the previous completed epoch')
        payload = torch.load(trainer.last, map_location='cpu', weights_only=False)
        completed = int(trainer.epoch)+1
        history_path = folder/'history.json'
        history = json.loads(history_path.read_text()) if history_path.exists() else []
        losses = trainer.label_loss_items(trainer.tloss, prefix='train')
        losses = {k:float(v) for k,v in losses.items()}
        history = [row for row in history if row['epoch'] < completed]
        plan = trainer.train_loader.dataset
        history.append(dict(epoch=completed, train_losses=losses, learning_rates=[g['lr'] for g in trainer.optimizer.param_groups],
            optimizer_updates=trainer._synthetic_steps_completed, optimizer_step_attempts=trainer._synthetic_step_attempts,
            amp_skipped_updates=trainer._synthetic_step_attempts-trainer._synthetic_steps_completed,
            epoch_sampling=cfg['epoch_sampling'], sample_counts=plan.domain_counts,
            unique_chase_observations=len({n for d,n in plan.sample_names if d == 'chase'}),
            samples=[dict(domain=d, id=n) for d,n in plan.sample_names]))
        payload.update(chase_yolo_identity=identity, completed_epoch=completed, fixed_final_epoch=epochs,
            live_state=cpu_copy(trainer.model.state_dict()), ema_full_state=cpu_copy(trainer.ema.ema.state_dict()),
            optimizer=cpu_copy(trainer.optimizer.state_dict()), scaler=cpu_copy(trainer.scaler.state_dict()),
            scheduler_state=trainer.scheduler.state_dict(), rng_state=capture_rng(), history=history)
        atomic_torch(last, payload); atomic_json(history_path, history)
        atomic_json(folder/'progress.json', dict(epoch=completed, fixed_final_epoch=epochs, identity=identity))
        if completed == cap: atomic_torch(stage, compact_checkpoint(payload))
    model.add_callback('on_train_start', restore)
    model.add_callback('on_train_epoch_start', begin_epoch)
    model.add_callback('on_train_batch_end', lambda t:setattr(t, '_synthetic_completed_batches', t._synthetic_completed_batches+1))
    model.add_callback('on_model_save', save)
    model.add_callback('on_fit_epoch_end', lambda t:setattr(t, 'stop', t.epoch+1 >= cap))
    print(f'YOLO 완료 epoch {saved["completed_epoch"] if saved else 0} → {cap}; 최종 {epochs}, 배치1·누적없음, {cfg["steps_per_epoch"]}샘플/epoch.', flush=True)
    model.train(trainer=FixedTrainer, **options, resume=str(last) if saved is not None else False)
    if not stage.exists(): raise RuntimeError('Training exited before the requested completed epoch')
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-until', type=int)
    args = parser.parse_args(argv)
    value = train(json.loads(args.config.read_text()), args.run_until)
    print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__': main()
