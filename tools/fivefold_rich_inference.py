"""Resumable prediction-only exports for equally budgeted fivefold study stages.

Limb correction is provided by the reviewed image store. Four geometric views
share one model; the identity prediction is reused as the base prediction.
No annotation file, ground-truth mask, optimizer or scorer is accessed here.
"""
import csv
import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np
from pycocotools import mask as mu
from scipy import ndimage
import torch

SIZE = 2048
COMPONENT_THRESHOLDS = (.35, .5, .65, .8)
UNION_THRESHOLDS = (.2, .35, .5, .65, .8)
VIEWS = (('base', ()), ('hflip', (-1,)), ('vflip', (-2,)), ('hvflip', (-2, -1)))
PROFILES = ('tta', 'tta_clahe')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def content_sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name+'.pending')
    pending.write_bytes(canonical(value)+b'\n')
    pending.replace(path)


def encode(mask):
    value = mu.encode(np.asfortranarray(np.asarray(mask, np.uint8)))
    return dict(size=[SIZE, SIZE], counts=value['counts'].decode('ascii'))


def checked_probability(value):
    value = np.asarray(value, np.float32)
    if value.shape != (SIZE, SIZE) or not np.isfinite(value).all():
        raise ValueError('Expected a finite native2048-square probability image')
    if value.min() < 0 or value.max() > 1:
        raise ValueError('Foreground probabilities must be in[0,1]')
    return value


def branch_record(probability, name, model_sha):
    """Fields used by the previously reviewed native consensus postprocessing."""
    probability = checked_probability(probability)
    components = {}
    for threshold in COMPONENT_THRESHOLDS:
        labels, number = ndimage.label(probability >= threshold,
                                       structure=np.ones((3, 3), np.uint8))
        areas = np.bincount(labels.ravel())
        rows = []
        for label in range(1, number+1):
            area = int(areas[label])
            if area < 100:
                continue
            mask = labels == label
            rle = encode(mask)
            pixels = probability[mask]
            mean = float(pixels.mean())
            rows.append(dict(rle=rle, area=area,
                bbox=[int(v) for v in mu.toBbox(dict(size=rle['size'], counts=rle['counts'].encode()))],
                fg_mean=mean, stats=dict(foreground=dict(mean=mean, max=float(pixels.max()),
                                                       p90=float(np.percentile(pixels, 90))))))
        components[f'{threshold:g}'] = rows
    return dict(schema_version=1, file_name=name, size=[SIZE, SIZE], checkpoint_sha256=model_sha,
        component_min_area=100, components=components,
        unions=dict(foreground={f'{t:g}': encode(probability >= t) for t in UNION_THRESHOLDS}))


def validation_names(csv_path):
    with Path(csv_path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        if 'file_name' not in (reader.fieldnames or []):
            raise ValueError('Validation CSV must contain file_name')
        names = sorted({row['file_name'] for row in reader})
    if not names or any(Path(name).name != name or '\\' in name for name in names):
        raise ValueError('Complete validation image basenames are required')
    if len({Path(name).stem for name in names}) != len(names):
        raise ValueError('Validation image stems must be unique')
    return names


def inference_settings(cfg):
    profile = cfg.get('inference_profile', cfg.get('profile', 'tta_clahe'))
    if profile not in PROFILES:
        raise ValueError('Inference profile must be tta or tta_clahe')
    crop, batch, overlap = int(cfg.get('crop', 1024)), int(cfg.get('infer_batch', 1)), float(cfg.get('overlap', .25))
    if crop % 32 or not 64 <= crop <= SIZE or batch < 1 or not 0 <= overlap < 1:
        raise ValueError('Invalid native tiling settings')
    return dict(profile=profile, crop=crop, infer_batch=batch, overlap=overlap,
        preprocessing='limb_only', views=[name for name, _ in VIEWS],
        component_thresholds=list(COMPONENT_THRESHOLDS), union_thresholds=list(UNION_THRESHOLDS),
        component_min_area=100, view_foreground_threshold=.5,
        clahe=dict(clip_limit=2., tile_grid_size=[8, 8]) if profile == 'tta_clahe' else None,
        dtype='torch.float16' if str(cfg.get('device', 'cuda')).startswith('cuda') else 'None')


def validate_cached(record, identity, name, image_sha, model_sha):
    saved_checksum = record.get('record_sha256')
    actual = content_sha({key: value for key, value in record.items() if key != 'record_sha256'})
    if saved_checksum != actual:
        raise ValueError('Cached rich prediction content changed; existing file preserved: '+name)
    expected = dict(inference_identity=identity, file_name=name,
                    image_sha256=image_sha, checkpoint_sha256=model_sha)
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError('Cached rich prediction provenance changed; existing file preserved: '+name)


def export_stage(runtime, model, cfg, stage_dir, provenance):
    """Export every held-out image without reading labels or fitting parameters."""
    cfg, provenance = dict(cfg), dict(provenance)
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    stage_epoch = int(cfg['stage_epoch'])
    if stage_epoch < 1 or ('stage_epoch' in provenance and provenance['stage_epoch'] != stage_epoch):
        raise ValueError('Stage epoch differs from provenance')
    model_sha = provenance.get('model_sha256', provenance.get('checkpoint_sha256'))
    if not isinstance(model_sha, str) or not re.fullmatch('[0-9a-f]{64}', model_sha):
        raise ValueError('Compact stage-model SHA256 is required')
    names = validation_names(cfg['val_split'])
    settings = inference_settings(cfg)
    identity_payload = dict(schema_version=1, stage_epoch=stage_epoch, provenance=provenance,
        settings=settings, validation_csv_sha256=digest(cfg['val_split']),
        exporter_sha256=digest(__file__), runtime_sha256=digest(runtime.__file__),
        versions=dict(torch=str(torch.__version__), numpy=np.__version__,
                      opencv=cv2.__version__))
    identity = content_sha(identity_payload)
    identity_path = stage_dir/'inference_identity.json'
    stored_identity = dict(identity=identity, payload=identity_payload)
    if identity_path.exists() and json.loads(identity_path.read_text()) != stored_identity:
        raise ValueError('Stage inference settings/sources changed; existing export preserved')
    if not identity_path.exists():
        atomic_json(identity_path, stored_identity)
    records_dir = stage_dir/'records'
    records_dir.mkdir(exist_ok=True)
    device = torch.device(cfg.get('device', 'cuda'))
    amp = torch.float16 if device.type == 'cuda' else None
    store = runtime.ImageStore(Path(cfg['images']), Path(cfg['cache_root'])/'images_real', 'limb_only')
    model.eval()
    combined, newly_inferred, cached, forward_calls = [], 0, 0, 0
    for index, name in enumerate(names, 1):
        image_path = Path(cfg['images'])/name
        image_sha = digest(image_path)
        path = records_dir/(Path(name).stem+'.json')
        reused = path.exists()
        if reused:
            record = json.loads(path.read_text())
            validate_cached(record, identity, name, image_sha, model_sha)
            cached += 1
        else:
            image = store.load(name)
            if not isinstance(image, np.ndarray) or image.shape != (SIZE, SIZE) or image.dtype != np.uint8:
                raise ValueError('Image store must return native2048-square uint8 limb-corrected pixels')
            base, total, view_unions = None, None, {}
            with torch.inference_mode():
                for view, axes in VIEWS:
                    pixels = np.ascontiguousarray(np.flip(image, axis=axes)) if axes else image
                    probability = checked_probability(runtime.predict_foreground(model, pixels, device,
                        settings['crop'], settings['overlap'], settings['infer_batch'], amp))
                    forward_calls += 1
                    if axes:
                        probability = np.ascontiguousarray(np.flip(probability, axis=axes))
                    if base is None:
                        base, total = probability, probability.copy()
                    else:
                        total += probability
                    view_unions[view] = encode(probability >= .5)
                record = branch_record(base, name, model_sha)
                record['tta'] = branch_record(total/np.float32(len(VIEWS)), name, model_sha)
                record['view_unions'] = view_unions
                record['preprocess_views'] = {}
                if settings['profile'] == 'tta_clahe':
                    pixels = cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8)).apply(image)
                    probability = checked_probability(runtime.predict_foreground(model, pixels, device,
                        settings['crop'], settings['overlap'], settings['infer_batch'], amp))
                    forward_calls += 1
                    record['preprocess_views']['clahe'] = branch_record(probability, name, model_sha)
            if digest(image_path) != image_sha:
                raise ValueError('Source image changed during inference: '+name)
            record.update(inference_identity=identity, image_sha256=image_sha, provenance=provenance,
                          stage_epoch=stage_epoch, fold=cfg.get('fold', provenance.get('fold')),
                          study_identity=provenance.get('study_identity'),
                          scope='Prediction-only held-out fold; no label selection or model fitting')
            record['record_sha256'] = content_sha(record)
            atomic_json(path, record)
            newly_inferred += 1
        combined.append(record)
        print(f"Stage{stage_epoch} rich [{index}/{len(names)}] {name} "
              f"{'cache' if reused else 'inferred'}", flush=True)
    rich_path = stage_dir/'rich_records.jsonl'
    pending = rich_path.with_name(rich_path.name+'.pending')
    with pending.open('wb') as stream:
        for record in combined:
            stream.write(canonical(record)+b'\n')
    pending.replace(rich_path)
    summary = dict(schema_version=1, status='complete', image_coverage=len(combined),
        validation_image_names=names, validation_csv_sha256=digest(cfg['val_split']),
        stage_epoch=stage_epoch, fold=cfg.get('fold', provenance.get('fold')),
        provenance=provenance, inference_identity=identity, settings=settings,
        rich_predictions=str(rich_path), rich_predictions_sha256=digest(rich_path),
        rich_records=str(rich_path), rich_records_sha256=digest(rich_path),
        records_sha256={row['file_name']: row['record_sha256'] for row in combined},
        newly_inferred_images=newly_inferred, cached_images=cached, forward_calls=forward_calls,
        predictions_per_new_image=4 if settings['profile'] == 'tta' else 5,
        ground_truth_read=False, scoring_performed=False, training_performed=False)
    summary_path = stage_dir/'export_summary.json'
    atomic_json(summary_path, summary)
    return dict(rich_predictions=str(rich_path), rich_records=str(rich_path),
        summary=str(summary_path), export_summary=str(summary_path),
        image_coverage=len(combined), newly_inferred_images=newly_inferred,
        cached_images=cached, forward_calls=forward_calls, inference_identity=identity)
