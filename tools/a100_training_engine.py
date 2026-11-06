"""Bounded-checkpoint training and completed-epoch resume for Colab.

The notebook builder injects the reviewed model, data and official evaluation
helpers into this module's namespace. These functions also work as a normal
module when those dependencies are assigned explicitly by an integration test.
"""
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import time
import uuid

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader


RESUME_KEYS = (
    'encoder', 'crop', 'batch', 'accumulate', 'steps_per_epoch', 'epochs',
    'lr', 'spine_weight', 'synthetic_ratio', 'warmup_epochs',
    'real_finish_epochs', 'include_empty_synthetic', 'seed', 'amp_dtype',
    'source_sha256', 'thresholds', 'min_areas', 'overlap', 'validate_every', 'preprocessing',
)


def atomic_json_save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.pending')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def atomic_torch_save(path, value):
    """Keep the previous complete checkpoint until its replacement is written."""
    path = Path(path)
    temporary = path.with_name(path.name + '.pending')
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def capture_rng():
    # NumPy's ndarray is converted to a tensor so weights_only=True can load it.
    numpy_state = np.random.get_state()
    return dict(
        python=random.getstate(), torch=torch.get_rng_state(),
        numpy=(numpy_state[0], torch.from_numpy(numpy_state[1].astype(np.int64)),
               int(numpy_state[2]), int(numpy_state[3]), float(numpy_state[4])),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )


def restore_rng(state):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'].cpu())
    numpy_state = state['numpy']
    np.random.set_state((numpy_state[0], numpy_state[1].cpu().numpy().astype(np.uint32),
                         numpy_state[2], numpy_state[3], numpy_state[4]))
    if state.get('cuda') and torch.cuda.is_available():
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('Resume CUDA device count differs from the saved run')
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def check_resume_config(saved, current):
    differences = {key: dict(saved=saved.get(key), current=current.get(key))
                   for key in RESUME_KEYS if saved.get(key) != current.get(key)}
    if differences:
        raise ValueError('Resume configuration differs: ' + json.dumps(differences, ensure_ascii=False))


def checkpoint_payload(model, optimizer, scheduler, scaler, cfg, epoch, history,
                       best_pq, best_epoch, elapsed_seconds):
    return dict(schema_version=1, model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        scheduler_state_dict=scheduler.state_dict(), scaler_state_dict=scaler.state_dict(),
        config=cfg.copy(), epoch=int(epoch), history=history,
        best_pq=float(best_pq), best_epoch=int(best_epoch),
        elapsed_seconds=float(elapsed_seconds), rng_state=capture_rng())


def restore_training_state(checkpoint, model, optimizer, scheduler, scaler, cfg):
    check_resume_config(checkpoint['config'], cfg)
    required = {'optimizer_state_dict', 'scheduler_state_dict', 'scaler_state_dict',
                'rng_state', 'history', 'best_pq', 'best_epoch'}
    if not required <= set(checkpoint):
        raise ValueError('This checkpoint has no complete optimizer resume state')
    epoch = int(checkpoint['epoch'])
    history = checkpoint['history']
    if len(history) != epoch or (history and int(history[-1]['epoch']) != epoch):
        raise ValueError('Saved epoch and completed history disagree')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    restore_rng(checkpoint['rng_state'])
    return dict(start_epoch=epoch + 1, history=history,
                best_pq=float(checkpoint['best_pq']), best_epoch=int(checkpoint['best_epoch']),
                elapsed_seconds=float(checkpoint.get('elapsed_seconds', 0.)))


def make_epoch_loader(dataset, cfg, device):
    workers = int(cfg.get('workers', 2))
    if workers < 0:
        raise ValueError('workers must be nonnegative')
    generator = torch.Generator().manual_seed(int(cfg['seed']) + int(dataset.epoch))
    kwargs = dict(batch_size=int(cfg['batch']), shuffle=False, num_workers=workers,
                  pin_memory=device.type == 'cuda', persistent_workers=False,
                  generator=generator)
    if workers:
        kwargs['multiprocessing_context'] = 'spawn'
    return DataLoader(dataset, **kwargs)


def storage_estimate(model):
    model_bytes = sum(value.numel() * value.element_size()
                      for value in model.state_dict().values())
    return dict(model_mib=model_bytes / 1024**2,
        full_last_estimated_mib=3 * model_bytes / 1024**2,
        persistent_best_and_last_estimated_mib=4 * model_bytes / 1024**2,
        atomic_save_peak_estimated_mib=7 * model_bytes / 1024**2,
        note='Model plus Adam moments estimate; Drive FUSE free-space is not a quota guarantee')


def export_best_predictions(output, best_payload):
    """A complete best checkpoint can repair interrupted CSV/metadata exports."""
    predictions = best_payload.get('validation_predictions')
    if predictions is None:
        return
    for (threshold, area), predicted in predictions.items():
        write_csv(Path(output) / f'best_t{threshold:g}_a{area}.csv', predicted)
    best = best_payload['validation']
    write_csv(Path(output) / 'best_validation.csv',
              predictions[best['threshold'], best['min_area']])
    atomic_json_save(Path(output) / 'best_epoch.json',
                     dict(epoch=best_payload['epoch'], best=best,
                          grid=best_payload['validation_grid']))


def normalize_training_config(config):
    cfg = dict(encoder='tu-convnext_tiny', pretrained=True, crop=1024, batch=2,
        accumulate=2, steps_per_epoch=200, epochs=30, validate_every=2, lr=1e-4,
        spine_weight=.1, seed=42, workers=2, synthetic_ratio=.2,
        warmup_epochs=3, real_finish_epochs=6, include_empty_synthetic=True,
        thresholds=[.35, .5, .65], min_areas=[100, 200, 300], overlap=.25,
        device='cuda', resume_dir=None, cache_root='/content/filament_a100_cache',
        preprocessing='limb_only')
    cfg.update(config)
    paths = [('images', 'IMAGES'), ('annotations', 'ANN'), ('train_split', 'TRAIN'),
             ('val_split', 'VAL'), ('synthetic_images', 'SYNTH_IMAGE_DIR'),
             ('synthetic_annotations', 'SYNTH_ANN_PATH'),
             ('synthetic_split', 'SYNTH_SPLIT_INFO_PATH')]
    for key, name in paths:
        value = cfg.get(key, globals().get(name))
        if value is not None:
            cfg[key] = str(value)
    cfg['output_root'] = str(cfg.get('output_root', Path(globals().get('RUN', '.')) / 'native_a100_runs'))
    cfg['cache_root'] = str(cfg['cache_root'])
    if cfg['resume_dir']:
        cfg['resume_dir'] = str(cfg['resume_dir'])
    if cfg['crop'] % 32 or not 64 <= cfg['crop'] <= 2048:
        raise ValueError('crop must be a multiple of32 in [64,2048]')
    if min(cfg[key] for key in ('batch', 'accumulate', 'steps_per_epoch', 'epochs', 'validate_every')) <= 0:
        raise ValueError('Training batch, accumulation, epoch and interval must be positive')
    if not 0 <= cfg['synthetic_ratio'] < 1 or min(cfg['warmup_epochs'], cfg['real_finish_epochs']) < 0:
        raise ValueError('Invalid synthetic probability or real-only schedule')
    if cfg['preprocessing'] != 'limb_only':
        raise ValueError('This runner uses the reviewed limb_only preprocessing')
    cache = Path(cfg['cache_root']).resolve()
    if cache.is_relative_to(Path('/content/drive')):
        raise ValueError('Processed image/target caches must stay on Colab local disk')
    return cfg


def check_synthetic_source_pool(split_info, data, train_ids, val_names):
    """Check donor-pool physical files without redoing the fixed real split.

    Source val IDs can have annotator siblings intentionally promoted to real
    train by the earlier notebook. Only its source TRAIN files are donors.
    """
    by_id = {str(row['id']): row['file_name'] for row in data['images']}
    source_train = {str(value) for value in split_info['train_image_ids']}
    source_val = {str(value) for value in split_info['val_image_ids']}
    if source_train & source_val or source_train | source_val != set(by_id):
        raise ValueError('Synthetic source split IDs do not match the original annotation images')
    source_files = {by_id[value] for value in source_train}
    real_train_files = {by_id[str(value)] for value in train_ids}
    if not source_files <= real_train_files or source_files & set(val_names):
        raise ValueError('Synthetic donor pool overlaps the fixed validation files')
    return dict(source_train_ids=len(source_train), source_train_physical_files=len(source_files),
                fixed_train_physical_files=len(real_train_files), fixed_val_physical_files=len(val_names),
                donor_validation_overlap=0, fixed_split_changed=False)


def preflight_synthetic_images(synthetic_data, image_directory):
    """Check every referenced file, then decode at most three native samples."""
    directory = Path(image_directory)
    images = synthetic_data.get('images', [])
    if not images:
        raise ValueError('Synthetic metadata contains no image records')
    records = {str(row['id']): row for row in images}
    if len(records) != len(images):
        raise ValueError('Duplicate synthetic image IDs')
    counts = Counter(str(row['image_id']) for row in synthetic_data.get('annotations', []))
    if not set(counts) <= set(records):
        raise ValueError('Synthetic annotation references an unknown image ID')
    filenames = {key: Path(row['file_name']).name for key, row in records.items()}
    if any(name in {'', '.', '..'} for name in filenames.values()):
        raise ValueError('Invalid synthetic filename')
    missing = sorted({name for name in filenames.values() if not (directory / name).is_file()})
    if missing:
        raise ValueError(f'Synthetic image files missing ({len(missing)}): {missing[:10]}')
    positives = [key for key in records if counts[key] > 0]
    blanks = [key for key in records if counts[key] == 0]
    if not positives:
        raise ValueError('Synthetic domain has no positive annotations')
    selected = list(dict.fromkeys([positives[0], positives[-1]] + blanks[:1]))
    samples = []
    for key in selected:
        path = directory / filenames[key]
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.shape != (2048, 2048) or gray.dtype != np.uint8:
            shape = None if gray is None else list(gray.shape)
            raise ValueError(f'Synthetic native uint8 2048-square image required: {path}, shape={shape}')
        samples.append(dict(file_name=filenames[key], shape=list(gray.shape),
                            dtype=str(gray.dtype), label_count=int(counts[key])))
    return dict(image_records=len(images), physical_image_files=len(set(filenames.values())),
                filenames_checked=len(set(filenames.values())), positive_images=len(positives),
                blank_images=len(blanks), missing_files=0, decoded_samples=samples)


def run_training(config):
    """Run a fresh or resumed train; helpers are provided by the notebook builder."""
    global GT, IMAGES, ANN, TRAIN, VAL
    cfg = normalize_training_config(config)
    device = torch.device(cfg['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('Select the A100 GPU runtime in Colab')
    cv2.setNumThreads(2)
    for key, name in [('images', 'IMAGES'), ('annotations', 'ANN'),
                      ('train_split', 'TRAIN'), ('val_split', 'VAL')]:
        globals()[name] = Path(cfg[key])
    source_keys = ['annotations', 'train_split', 'val_split']
    for key in ('synthetic_annotations', 'synthetic_split'):
        if cfg.get(key) and Path(cfg[key]).is_file():
            source_keys.append(key)
        elif cfg['synthetic_ratio']:
            raise ValueError(f'Synthetic metadata is required for mixing: {cfg.get(key, key)}')
    cfg['source_sha256'] = {}
    for key in source_keys:
        path = Path(cfg[key])
        with path.open('rb') as handle:
            cfg['source_sha256'][key] = hashlib.file_digest(handle, 'sha256').hexdigest()
    expected_hashes = cfg.get('expected_source_sha256', {})
    mismatches = {key: cfg['source_sha256'].get(key) for key, value in expected_hashes.items()
                  if cfg['source_sha256'].get(key) != value}
    if mismatches:
        raise ValueError('Input source hashes differ from the verified files: ' + json.dumps(mismatches))
    data = json.loads(Path(cfg['annotations']).read_text(encoding='utf-8-sig'))
    train_ids, val_names, GT = prepare_split(data)
    synthetic_data = (json.loads(Path(cfg['synthetic_annotations']).read_text(encoding='utf-8-sig'))
                      if cfg['synthetic_ratio'] else None)
    if synthetic_data is not None:
        split_info = json.loads(Path(cfg['synthetic_split']).read_text(encoding='utf-8-sig'))
        cfg['synthetic_source_pool'] = check_synthetic_source_pool(split_info, data, train_ids, val_names)
        cfg['synthetic_image_preflight'] = preflight_synthetic_images(synthetic_data, cfg['synthetic_images'])
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    amp_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device.type == 'cuda' else None
    cfg['amp_dtype'] = str(amp_dtype)
    cfg['gpu'] = torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'
    output = Path(cfg['resume_dir']) if cfg['resume_dir'] else Path(cfg['output_root']) / (
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8])
    if cfg['resume_dir']:
        if not (output / 'last.pt').is_file():
            raise ValueError(f'Resume last.pt not found: {output}')
        checkpoint = torch.load(output / 'last.pt', map_location='cpu', weights_only=True)
        check_resume_config(checkpoint['config'], cfg)
        for name in ('best.pt.pending', 'last.pt.pending'):
            (output / name).unlink(missing_ok=True)
    else:
        output.mkdir(parents=True, exist_ok=False)
        checkpoint = None
    store = ImageStore(Path(cfg['images']), Path(cfg['cache_root']) / 'images_real', 'limb_only')
    synthetic_store = (ImageStore(Path(cfg['synthetic_images']), Path(cfg['cache_root']) / 'images_synthetic', 'limb_only')
                       if synthetic_data is not None else None)
    dataset = MixedCropDataset(data, train_ids, store, synthetic_data=synthetic_data,
        synthetic_store=synthetic_store, crop=cfg['crop'], length=cfg['steps_per_epoch'] * cfg['batch'],
        seed=cfg['seed'], synthetic_ratio=cfg['synthetic_ratio'], warmup_epochs=cfg['warmup_epochs'],
        real_finish_epochs=cfg['real_finish_epochs'], total_epochs=cfg['epochs'],
        cache_dir=Path(cfg['cache_root']) / 'targets',
        include_empty_synthetic=cfg['include_empty_synthetic'], augment=True)
    cfg.update(train_physical_images=len(dataset.files), validation_physical_images=len(val_names),
        validation_annotation_groups=len(GT), small_object_area_threshold_train_only=float(dataset.small_area),
        data_domains=dataset.domain_info, checkpoint_policy='one best plus one full last; caches local only',
        effective_batch=cfg['batch'] * cfg['accumulate'], samples_per_epoch=len(dataset),
        loss='foreground BCE+Dice + balanced spine BCE on valid supervision only',
        initialization='completed-epoch optimizer/RNG resume' if checkpoint is not None else 'ImageNet encoder with new native decoder',
        native_output_heads=['foreground', 'spine'],
        validation_selection='Official global all-pairs PQ with strict IoU>.5',
        versions={name: importlib.metadata.version(name) for name in
                  ('torch', 'segmentation-models-pytorch', 'timm', 'numpy', 'scipy', 'pycocotools')})
    model = build_model(cfg['encoder'], pretrained=bool(cfg['pretrained'] and checkpoint is None)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'], eta_min=cfg['lr'] * .05)
    scaler = torch.amp.GradScaler('cuda', enabled=amp_dtype == torch.float16)
    if checkpoint is not None:
        restored = restore_training_state(checkpoint, model, optimizer, scheduler, scaler, cfg)
        history, best_pq, best_epoch = restored['history'], restored['best_pq'], restored['best_epoch']
        start_epoch, elapsed_before = restored['start_epoch'], restored['elapsed_seconds']
        best_path = output / 'best.pt'
        if best_path.is_file():
            best_payload = torch.load(best_path, map_location='cpu', weights_only=True)
            if best_payload['validation']['pq'] >= best_pq:
                best_pq, best_epoch = float(best_payload['validation']['pq']), int(best_payload['epoch'])
                export_best_predictions(output, best_payload)
            del best_payload
        del checkpoint
    else:
        history, best_pq, best_epoch, start_epoch, elapsed_before = [], -1., 0, 1, 0.
    cfg['storage_estimate'] = storage_estimate(model)
    source = globals().get('__file__')
    if source and Path(source).is_file():
        current_source = Path(source).read_bytes()
        original_source = output / 'run_source.py'
        if not original_source.is_file():
            original_source.write_bytes(current_source)
        cfg['run_source_sha256'] = hashlib.sha256(original_source.read_bytes()).hexdigest()
        if hashlib.sha256(current_source).hexdigest() != cfg['run_source_sha256']:
            (output / 'resume_source.py').write_bytes(current_source)
            cfg['resume_source_sha256'] = hashlib.sha256(current_source).hexdigest()
    atomic_json_save(output / 'config.json', cfg)
    print(json.dumps(dict(output=str(output), start_epoch=start_epoch, config=cfg), ensure_ascii=False, indent=2), flush=True)
    started = time.monotonic()
    for epoch in range(start_epoch, cfg['epochs'] + 1):
        dataset.epoch = epoch
        loader = make_epoch_loader(dataset, cfg, device)
        model.train(); optimizer.zero_grad(set_to_none=True)
        losses = []
        domains = Counter(dataset.sample_record(i)['domain'] for i in range(len(dataset)))
        for step, (x, y, spine_valid) in enumerate(loader):
            x, y, spine_valid = (value.to(device, non_blocking=True) for value in (x, y, spine_valid))
            with torch.autocast(device_type=device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
                logits = model(x)
            loss = mixed_segmentation_loss(logits, y, spine_valid, cfg['spine_weight'])
            if not torch.isfinite(loss):
                raise ValueError(f'Nonfinite loss at epoch{epoch} step{step}')
            group = min(cfg['accumulate'], len(loader) - (step // cfg['accumulate']) * cfg['accumulate'])
            scaler.scale(loss / group).backward()
            if (step + 1) % cfg['accumulate'] == 0 or step + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
            if step == 0 or (step + 1) % max(1, len(loader) // 10) == 0:
                print(f'epoch{epoch}/{cfg["epochs"]} step{step+1}/{len(loader)} loss{losses[-1]:.5f}', flush=True)
        row = dict(epoch=epoch, train_loss=float(np.mean(losses)),
            lr=float(optimizer.param_groups[0]['lr']), microbatches=len(loader),
            optimizer_updates=math.ceil(len(loader) / cfg['accumulate']),
            synthetic_probability=float(dataset.synthetic_probability()), sampled_domains=dict(domains),
            elapsed_seconds=elapsed_before + time.monotonic() - started)
        scheduler.step()
        if epoch == 1 or epoch % cfg['validate_every'] == 0 or epoch == cfg['epochs']:
            best, grid, predictions = validate_native(model, store, val_names, device, amp_dtype, cfg)
            row.update(validation=best, validation_grid=grid)
            if best['pq'] > best_pq:
                best_pq, best_epoch = float(best['pq']), epoch
                best_payload = dict(schema_version=1, model_state_dict=model.state_dict(), config=cfg.copy(),
                    epoch=epoch, validation=best, validation_grid=grid, validation_predictions=predictions)
                atomic_torch_save(output / 'best.pt', best_payload)
                export_best_predictions(output, best_payload)
                del best_payload
        history.append(row)
        payload = checkpoint_payload(model, optimizer, scheduler, scaler, cfg, epoch, history,
                                     best_pq, best_epoch, row['elapsed_seconds'])
        atomic_torch_save(output / 'last.pt', payload)
        del payload
        atomic_json_save(output / 'history.json', history)
        summary = dict(status='complete' if epoch == cfg['epochs'] else 'running',
            best_development_pq=best_pq, best_epoch=best_epoch, epochs_completed=epoch,
            target_pq=.5, target_exceeded=best_pq > .5, checkpoint_directory=str(output), config=cfg,
            scope='Same fixed development split and official global all-pairs PQ; not Kaggle LB')
        if 'validation' in row:
            archive = package_results(output, summary)
            print('자동 저장 ZIP:', archive, flush=True)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        del loader
        gc.collect()
    summary = dict(status='complete', best_development_pq=best_pq, best_epoch=best_epoch,
        epochs_completed=len(history), target_pq=.5, target_exceeded=best_pq > .5,
        checkpoint_directory=str(output), config=cfg,
        scope='Same fixed development split; completed-epoch optimizer resume supported; not Kaggle LB')
    archive = package_results(output, summary)
    return dict(output=str(output), archive=str(archive), summary=summary)
