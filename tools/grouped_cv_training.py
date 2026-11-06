"""Fresh native-model CV with a fixed final epoch and completed-epoch resume.

The old reviewed runtime supplies model/data helpers only. Its fixed616/91
split and best-validation training loop are intentionally never called.
"""
import argparse
from collections import defaultdict
import csv
from datetime import date
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import time
import zipfile

import cv2
import numpy as np
from pycocotools import mask as mu
import torch
from torch import nn

RUNTIME_SHA = '6d60b5705fb0fbde03a6fd2922c8b83839fa49b5a78cc3434e7dcf4a1b4a08b4'
DEFAULTS = dict(encoder='tu-convnext_tiny', pretrained=True, epochs=100,
    run_until_epoch=100, crop=1024, batch=1, accumulate=4, steps_per_epoch=400,
    lr=1e-4, spine_weight=.1, workers=0, infer_batch=1, overlap=.25,
    validate_every=20, seed=42, device='cuda', threshold=.5, min_area=300,
    synthetic_ratio=0., warmup_epochs=0, real_finish_epochs=0,
    include_empty_synthetic=False, preprocessing='limb_only', preflight_backward=True)
IDENTITY_KEYS = tuple(key for key in DEFAULTS if key not in ('run_until_epoch', 'device', 'preflight_backward'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name+'.pending')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    pending.replace(path)


def load_runtime(path):
    path = Path(path)
    if path.name != 'base_runtime.py' or digest(path) != RUNTIME_SHA:
        raise ValueError('Restore the reviewed original runtime as base_runtime.py')
    spec = importlib.util.spec_from_file_location('base_runtime', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['base_runtime'] = module
    spec.loader.exec_module(module)
    return module


def prepare_inputs(cfg):
    data = json.loads(Path(cfg['annotations']).read_text())
    plan = json.loads(Path(cfg['fold_plan']).read_text())
    if plan.get('training_performed') is not False or plan.get('block_days') != 27 or plan.get('embargo_days') != 3:
        raise ValueError('Expected the reviewed27-day/3-day metadata-only fold plan')
    if plan.get('sources_sha256', {}).get('annotations') != digest(cfg['annotations']):
        raise ValueError('Annotation bytes differ from the fold plan')
    fold = int(cfg['fold'])
    if not 0 <= fold < plan['folds']:
        raise ValueError('Invalid fold index')
    fold_info = next(row for row in plan['fold_summaries'] if row['fold'] == fold)
    roles = fold_info['roles']
    records = {row['physical_identifier']: row for row in plan['observations']}
    by_id = {str(row['id']): row for row in data['images']}
    if len(by_id) != len(data['images']) or set(roles) != set(records):
        raise ValueError('Duplicate annotation IDs or incomplete fold roles')
    metadata_ids = [sid for row in records.values() for sid in row['annotation_ids']]
    if len(metadata_ids) != len(set(metadata_ids)) or set(metadata_ids) != set(by_id):
        raise ValueError('Fold metadata omits/duplicates annotation variants')
    for key, row in records.items():
        if any(by_id[sid]['file_name'] != row['file_name'] for sid in row['annotation_ids']):
            raise ValueError('Fold image names differ from annotations')
        if Path(row['file_name']).name != row['file_name']:
            raise ValueError('Image basename required')
        if (plan['assignments'][key] == fold) != (roles[key] == 'validation'):
            raise ValueError('Outer fold assignment differs from roles')
    split_ids, split_names = {}, {}
    for role, field in [('train', 'train_split'), ('validation', 'val_split')]:
        with Path(cfg[field]).open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        ids = [row['image_id'] for row in rows]
        expected = {sid for key, row in records.items() if roles[key] == role for sid in row['annotation_ids']}
        if len(ids) != len(set(ids)) or set(ids) != expected:
            raise ValueError('CSV does not include every annotator of its assigned images')
        if any(row['file_name'] != by_id[row['image_id']]['file_name'] for row in rows):
            raise ValueError('CSV image names differ from annotations')
        split_ids[role] = set(ids)
        split_names[role] = {row['file_name'] for row in rows}
    if not all(split_ids.values()) or split_names['train'] & split_names['validation']:
        raise ValueError('Empty or overlapping train/validation images')
    days = {role: {date.fromisoformat(row['capture_day']) for key, row in records.items() if roles[key] == role}
            for role in ('train', 'validation')}
    if min(abs((a-b).days) for a in days['train'] for b in days['validation']) <= 3:
        raise ValueError('Temporal embargo was violated')
    if {records[key]['block'] for key in roles if roles[key] == 'train'} & {
            records[key]['block'] for key in roles if roles[key] == 'validation'}:
        raise ValueError('A temporal block crosses train/validation')
    missing = [name for name in split_names['train'] | split_names['validation']
               if not (Path(cfg['images'])/name).is_file()]
    if missing:
        raise FileNotFoundError(f'Missing original images ({len(missing)}): {missing[:3]}')
    truth = {sid: [] for sid in split_ids['validation']}
    for ann in data['annotations']:
        sid = str(ann['image_id'])
        if sid not in by_id:
            raise ValueError('Annotation has unknown image_id')
        if sid in truth:
            seg = ann['segmentation']
            if isinstance(seg, list):
                value = mu.merge(mu.frPyObjects(seg, 2048, 2048))
            elif isinstance(seg.get('counts'), list):
                value = mu.frPyObjects(seg, 2048, 2048)
            else:
                value = dict(seg)
                if isinstance(value['counts'], str):
                    value['counts'] = value['counts'].encode()
            if value['size'] != [2048, 2048] or int(mu.area(value)) <= 0:
                raise ValueError('Invalid validation mask')
            truth[sid].append(value)
    group_to_image = {sid: Path(by_id[sid]['file_name']).stem for sid in truth}
    return data, split_ids['train'], sorted(split_names['validation']), truth, group_to_image, fold_info


def metric(truth, predictions, group_to_image):
    rows = []
    blank_fp = 0
    for sid, gt in sorted(truth.items()):
        pred = predictions[group_to_image[sid]]
        if not gt:
            blank_fp += len(pred)
            continue  # Competition CSV scorer has no group for an empty GT image.
        if pred:
            matrix = np.asarray(mu.iou(gt, pred, [0]*len(pred)))
            hits = matrix > .5
            tp, fp, fn = int(hits.sum()), int((~hits.any(0)).sum()), int((~hits.any(1)).sum())
            iou = float(matrix[hits].sum())
        else:
            tp, fp, fn, iou = 0, 0, len(gt), 0.
        rows.append(dict(group_id=sid, image=group_to_image[sid], tp_pairs=tp, fp=fp, fn=fn, matched_iou_sum=iou))
    total = {key: sum(row[key] for row in rows) for key in ('tp_pairs', 'fp', 'fn', 'matched_iou_sum')}
    den = total['tp_pairs']+.5*(total['fp']+total['fn'])
    return dict(**total, pq=total['matched_iou_sum']/den if den else 0.,
        sq=total['matched_iou_sum']/total['tp_pairs'] if total['tp_pairs'] else 0.,
        rq=total['tp_pairs']/den if den else 0., per_group=rows,
        blank_gt_groups=sum(not values for values in truth.values()), fp_on_blank_gt_groups=blank_fp)


def components(runtime, probability, threshold, minimum_area):
    result = []
    for mask in runtime.iter_components(probability, threshold, minimum_area):
        value = mu.encode(np.asfortranarray(mask.astype(np.uint8)))
        result.append(dict(area=int(mu.area(value)), fg_mean=float(probability[mask].mean()),
            rle=dict(size=[2048, 2048], counts=value['counts'].decode('ascii'))))
    return result


def predict(runtime, model, store, names, cfg, device, amp, checkpoint_sha=None, output=None):
    predictions, records = {}, []
    for index, name in enumerate(names, 1):
        image = Path(name).stem
        source_sha = digest(Path(cfg['images'])/name)
        path = output/'records'/(image+'.json') if output else None
        identity = dict(cv_identity=cfg['cv_identity'], checkpoint_sha256=checkpoint_sha,
                        image_sha256=source_sha, file_name=name)
        if path and path.is_file():
            record = json.loads(path.read_text())
            if any(record.get(key) != value for key, value in identity.items()):
                raise ValueError('Cached OOF record provenance changed: '+name)
        else:
            probability = runtime.predict_foreground(model, store.load(name), device,
                cfg['crop'], cfg['overlap'], cfg['infer_batch'], amp)
            if probability.shape != (2048, 2048) or not np.isfinite(probability).all():
                raise ValueError('Invalid native probability image')
            thresholds = sorted({.35, .5, .65, cfg['threshold']}) if output else [cfg['threshold']]
            component_min_area = min(100, cfg['min_area']) if output else cfg['min_area']
            record = dict(**identity, fold=cfg['fold'], selected_epoch=cfg['epochs'],
                component_min_area=component_min_area,
                components={f'{threshold:g}': components(runtime, probability, threshold, component_min_area)
                            for threshold in thresholds})
            if path:
                atomic_json(path, record)
        values = [row['rle'] for row in record['components'][f"{cfg['threshold']:g}"]
                  if row['area'] >= cfg['min_area']]
        predictions[image] = values
        records.append(record)
        print(f"{'OOF' if output else 'diagnostic validation'} [{index}/{len(names)}] {name}", flush=True)
    return predictions, records


def configure_cuda():
    torch.backends.cudnn.benchmark = False
    if not getattr(nn.Conv2d._conv_forward, '_solar_cv_safe', False):
        original = nn.Conv2d._conv_forward
        def safe(self, inputs, weight, bias):
            if inputs.is_cuda and self.groups > 1 and self.groups == self.in_channels:
                with torch.backends.cudnn.flags(enabled=False):
                    return original(self, inputs, weight, bias)
            return original(self, inputs, weight, bias)
        safe._solar_cv_safe = True
        nn.Conv2d._conv_forward = safe


def package(output, result):
    atomic_json(output/'result.json', result)
    path = output/f"fold{result['fold']}_cv_results.zip"
    pending = path.with_name(path.name+'.pending')
    with zipfile.ZipFile(pending, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in ('config.json', 'history.json', 'result.json', 'oof_summary.json',
                     'oof_predictions.csv', 'oof_records.jsonl', 'fold_plan.json',
                     'train_split.csv', 'val_split.csv'):
            source = output/name
            if source.is_file():
                archive.write(source, name)
    pending.replace(path)
    return dict(result, archive=str(path))


def run(config, _runtime=None):
    cfg = dict(DEFAULTS, **config)
    for key in ('initial_checkpoint', 'warm_start', 'checkpoint', 'resume_dir'):
        if cfg.get(key):
            raise ValueError('CV accepts only fresh generic initialization or its own automatic fold resume')
    if cfg['synthetic_ratio'] != 0 or cfg['warmup_epochs'] != 0 or cfg['real_finish_epochs'] != 0:
        raise ValueError('Existing synthetic pool is not verified for the new folds')
    if cfg['crop'] % 32 or not 64 <= cfg['crop'] <= 2048:
        raise ValueError('Crop must be a multiple of32 in[64,2048]')
    if any(int(cfg[key]) < 1 for key in ('epochs', 'run_until_epoch', 'batch', 'accumulate',
                                        'steps_per_epoch', 'validate_every', 'infer_batch')):
        raise ValueError('Positive training/inference settings required')
    if int(cfg['workers']) < 0 or not 0 <= cfg['overlap'] < 1:
        raise ValueError('Invalid workers/overlap')
    cfg['fold'] = int(cfg['fold'])
    device = torch.device(cfg['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('Select a Colab T4 GPU runtime')
    if Path(cfg['cache_root']).resolve().is_relative_to(Path('/content/drive')):
        raise ValueError('Decoded training caches must remain on local /content disk')
    runtime = _runtime or load_runtime(cfg['base_runtime'])
    data, train_ids, val_names, truth, group_to_image, fold_info = prepare_inputs(cfg)
    amp = torch.float16 if device.type == 'cuda' else None  # T4: never BF16 emulation.
    cfg.update(amp_dtype=str(amp), thresholds=[cfg['threshold']], min_areas=[cfg['min_area']],
        gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu',
        source_sha256={key: digest(cfg[key]) for key in ('annotations', 'train_split', 'val_split', 'fold_plan', 'base_runtime')})
    cfg['source_sha256']['cv_runner'] = digest(__file__)
    identity = dict(settings={key: cfg[key] for key in IDENTITY_KEYS}, fold=cfg['fold'],
                    sources=cfg['source_sha256'], precision=cfg['amp_dtype'],
                    torch_version=str(torch.__version__), final_selection='fixed_epoch')
    cfg['cv_identity'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cfg['cv_identity_payload'] = identity
    cfg['cv_fold'] = cfg['fold']
    cfg['validation_selection'] = 'Fixed final epoch; diagnostics never choose the OOF checkpoint'
    output = Path(cfg['output_root'])/f"fold{cfg['fold']}"
    output.mkdir(parents=True, exist_ok=True)
    if (output/'config.json').is_file():
        old = json.loads((output/'config.json').read_text())
        if old['cv_identity'] != cfg['cv_identity']:
            raise ValueError('This output belongs to different fold/settings/sources; existing files preserved')
    elif (output/'last.pt').exists():
        raise ValueError('Unidentified checkpoint exists in output; preserved')
    for name, source in [('cv_runner.py', Path(__file__)), ('base_runtime.py', Path(cfg['base_runtime'])),
                         ('fold_plan.json', Path(cfg['fold_plan'])), ('train_split.csv', Path(cfg['train_split'])),
                         ('val_split.csv', Path(cfg['val_split']))]:
        target = output/name
        if target.is_file() and target.read_bytes() != source.read_bytes():
            raise ValueError('Saved source changed; refusing to overwrite it')
        if not target.exists():
            target.write_bytes(source.read_bytes())
    runtime.RESUME_KEYS = tuple(runtime.RESUME_KEYS)+('cv_fold', 'cv_identity')
    cv2.setNumThreads(2)
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed']); random.seed(cfg['seed'])
    if device.type == 'cuda':
        configure_cuda()
    last_path = output/'last.pt'
    checkpoint = torch.load(last_path, map_location='cpu', weights_only=True) if last_path.is_file() else None
    if checkpoint is not None and checkpoint['config'].get('cv_identity') != cfg['cv_identity']:
        raise ValueError('Checkpoint provenance differs from this fold')
    model = runtime.build_model(cfg['encoder'], pretrained=bool(cfg['pretrained'] and checkpoint is None)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    if checkpoint is not None:
        restored = runtime.restore_training_state(checkpoint, model, optimizer, scheduler, scaler, cfg)
        start, history, elapsed_before = restored['start_epoch'], restored['history'], restored['elapsed_seconds']
        del checkpoint
    else:
        start, history, elapsed_before = 1, [], 0.
    atomic_json(output/'config.json', cfg)
    store = runtime.ImageStore(Path(cfg['images']), Path(cfg['cache_root'])/'images_real', 'limb_only')
    dataset = runtime.MixedCropDataset(data, train_ids, store, crop=cfg['crop'],
        length=cfg['steps_per_epoch']*cfg['batch'], seed=cfg['seed'], synthetic_ratio=0.,
        warmup_epochs=0, real_finish_epochs=0, total_epochs=cfg['epochs'],
        cache_dir=Path(cfg['cache_root'])/'targets', augment=True)
    if device.type == 'cuda' and cfg['preflight_backward'] and start <= min(cfg['epochs'], cfg['run_until_epoch']):
        state = runtime.capture_rng()
        model.eval()
        x = torch.zeros((cfg['batch'], 3, cfg['crop'], cfg['crop']), device=device)
        with torch.autocast('cuda', dtype=amp):
            loss = model(x).float().square().mean()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite loss during GPU preflight')
        loss.backward()
        if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise ValueError('Nonfinite gradient during GPU preflight')
        model.zero_grad(set_to_none=True)
        runtime.restore_rng(state)
        del x, loss
        print('T4 FP16 forward/backward preflight passed; no optimizer update', flush=True)
    print(f"Fold{cfg['fold']}: train{fold_info['train_physical']} val{len(val_names)}; "
          f"start{start}, fixed-final{cfg['epochs']}, this-run-cap{cfg['run_until_epoch']}", flush=True)
    started = time.monotonic()
    cap = min(cfg['epochs'], cfg['run_until_epoch'])
    for epoch in range(start, cap+1):
        dataset.epoch = epoch
        loader = runtime.make_epoch_loader(dataset, cfg, device)
        model.train(); optimizer.zero_grad(set_to_none=True)
        losses = []
        for step, batch in enumerate(loader):
            x, target, spine_valid = (value.to(device, non_blocking=True) for value in batch)
            with torch.autocast(device.type, dtype=amp or torch.float32, enabled=amp is not None):
                logits = model(x)
            loss = runtime.mixed_segmentation_loss(logits, target, spine_valid, cfg['spine_weight'])
            if not torch.isfinite(loss):
                raise ValueError(f'Nonfinite training loss at epoch{epoch}/step{step}')
            group = min(cfg['accumulate'], len(loader)-(step//cfg['accumulate'])*cfg['accumulate'])
            scaler.scale(loss/group).backward()
            if (step+1) % cfg['accumulate'] == 0 or step+1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
            if step == 0 or (step+1) % max(1, len(loader)//10) == 0:
                print(f'epoch{epoch}/{cfg["epochs"]} step{step+1}/{len(loader)} loss{losses[-1]:.5f}', flush=True)
        scheduler.step()
        row = dict(epoch=epoch, train_loss=float(np.mean(losses)), lr=optimizer.param_groups[0]['lr'],
                   optimizer_updates=math.ceil(len(loader)/cfg['accumulate']))
        if epoch % cfg['validate_every'] == 0 and epoch < cfg['epochs']:
            predictions, _ = predict(runtime, model, store, val_names, cfg, device, amp)
            measured = metric(truth, predictions, group_to_image)
            row['diagnostic_validation'] = {key: value for key, value in measured.items() if key != 'per_group'}
        row['elapsed_seconds'] = elapsed_before+time.monotonic()-started
        history.append(row)
        payload = runtime.checkpoint_payload(model, optimizer, scheduler, scaler, cfg, epoch,
            history, -1., 0, row['elapsed_seconds'])
        runtime.atomic_torch_save(last_path, payload)
        atomic_json(output/'history.json', history)
        print(json.dumps(row), flush=True)
        del payload, loader
        gc.collect()
    completed = history[-1]['epoch'] if history else 0
    result = dict(status='paused' if completed < cfg['epochs'] else 'complete', fold=cfg['fold'],
        epochs_completed=completed, fixed_final_epoch=cfg['epochs'], cv_identity=cfg['cv_identity'],
        output=str(output), scope='Single native model, fresh fold training; not the old YOLO+native ensemble',
        oof_score_available=False, diagnostic_scores_select_checkpoint=False)
    if completed == cfg['epochs']:
        checkpoint_sha = digest(last_path)
        predictions, records = predict(runtime, model, store, val_names, cfg, device, amp, checkpoint_sha, output)
        measured = metric(truth, predictions, group_to_image)
        csv_pending = output/'oof_predictions.csv.pending'
        with csv_pending.open('w', newline='') as stream:
            writer = csv.writer(stream); writer.writerow(['filament_id', 'segmentation_rle'])
            writer.writerows((f'{image}_{index}', value['counts'])
                             for image, values in sorted(predictions.items()) for index, value in enumerate(values, 1))
        csv_pending.replace(output/'oof_predictions.csv')
        jsonl_pending = output/'oof_records.jsonl.pending'
        jsonl_pending.write_text(''.join(json.dumps(record)+'\n' for record in records))
        jsonl_pending.replace(output/'oof_records.jsonl')
        summary = dict(**result, checkpoint_sha256=checkpoint_sha, selected_epoch=completed,
            source_sha256=cfg['source_sha256'], threshold=cfg['threshold'], min_area=cfg['min_area'],
            image_coverage=len(records), image_names=val_names, metric=measured,
            oof_csv_sha256=digest(output/'oof_predictions.csv'), oof_records_sha256=digest(output/'oof_records.jsonl'),
            initialization='Generic ImageNet encoder or random initialization; no old solar weights',
            train_physical_identifiers=sorted(key for key, role in fold_info['roles'].items() if role == 'train'))
        summary['oof_score_available'] = True
        atomic_json(output/'oof_summary.json', summary)
        result.update(oof_score_available=True, selected_epoch=completed,
                      metric={key: value for key, value in measured.items() if key != 'per_group'})
    return package(output, result)


if __name__ == '__main__':
    import fcntl
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--result', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    lock_key = hashlib.sha256(str(Path(config['output_root']).resolve()).encode()).hexdigest()[:16]
    with (Path('/tmp')/f'solar_cv_{lock_key}.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another training process for this CV experiment is still running') from error
        result = run(config)
        atomic_json(args.result, result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
