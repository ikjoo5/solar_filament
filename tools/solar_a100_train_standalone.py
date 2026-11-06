from __future__ import annotations

import csv, hashlib, importlib.metadata, json, math, random, sys, time, uuid, zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import cv2, numpy as np, torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from pycocotools import mask as mu
from scipy import ndimage
from scipy.interpolate import UnivariateSpline

yy, xx = np.ogrid[:2048, :2048]
rr = (np.sqrt((xx - 1024) ** 2 + (yy - 1024) ** 2) / 900).astype(np.float32)
disk = rr <= 1
radii = (np.arange(100) + 0.5) / 100
bins = np.minimum((rr[disk] * 100).astype(np.int32), 99)

def limb(image):
    assert image is not None and image.shape == (2048, 2048) and (image.dtype == np.uint8), '2048 원본 JPEG가 필요합니다.'
    hist = np.bincount(bins * 256 + image[disk].astype(np.int32), minlength=25600).reshape(100, 256)
    count = hist.sum(axis=1)
    assert np.all(count > 0), '빈 radial bin'
    med = np.argmax(hist.cumsum(axis=1) >= ((count + 1) // 2)[:, None], axis=1).astype(float)
    spline = UnivariateSpline(radii, med, k=3, s=400)
    ref = float(np.median(spline(radii)[radii <= 0.15]))
    gain = np.clip(ref / np.maximum(spline(rr[disk]), 1), 0.25, 4)
    result = np.zeros_like(image)
    result[disk] = np.clip(image[disk].astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return result
preprocess_limb_only = limb
def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def build_model(encoder='tu-convnext_tiny', pretrained=True):
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name=encoder, encoder_weights='imagenet' if pretrained else None, in_channels=3, classes=2, activation=None)

def model_input(image: np.ndarray) -> np.ndarray:
    x = np.repeat(image[..., None], 3, axis=2).astype(np.float32) / 255.0
    return ((x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)).transpose(2, 0, 1)

def apply_d4(image, target, k, flip):
    image = np.rot90(image, k, axes=(-2, -1))
    target = np.rot90(target, k, axes=(-2, -1))
    if flip:
        image, target = (image[..., ::-1], target[..., ::-1])
    return (np.ascontiguousarray(image), np.ascontiguousarray(target))

def crop_origin(height, width, crop, rng, bbox=None):
    if crop > min(height, width) or crop <= 0:
        raise ValueError('crop must fit inside the native image')
    if bbox is None:
        return (int(rng.integers(height - crop + 1)), int(rng.integers(width - crop + 1)))
    x, y, w, h = bbox
    cx, cy = (x + w / 2, y + h / 2)
    left = int(np.clip(round(cx - crop / 2 + rng.uniform(-crop / 4, crop / 4)), 0, width - crop))
    top = int(np.clip(round(cy - crop / 2 + rng.uniform(-crop / 4, crop / 4)), 0, height - crop))
    return (top, left)

class ImageStore:

    def __init__(self, images, cache, preprocessing):
        self.images, self.cache = (Path(images), Path(cache))
        self.preprocessing = preprocessing
        self.cache.mkdir(parents=True, exist_ok=True)

    def load(self, filename):
        source = self.images / filename
        stat = source.stat()
        signature = hashlib.sha256(f'{source.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{self.preprocessing}:v1'.encode()).hexdigest()
        target = self.cache / f'{signature}.png'
        if target.is_file():
            image = cv2.imread(str(target), cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape != (2048, 2048):
                raise ValueError(f'invalid processed cache: {target}')
            return image
        image = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (2048, 2048) or image.dtype != np.uint8:
            raise ValueError(f'expected native uint8 2048-square JPEG: {source}')
        if self.preprocessing == 'limb_only':
            image = preprocess_limb_only(image)
        temporary = self.cache / f'{signature}.{uuid.uuid4().hex}.png'
        if not cv2.imwrite(str(temporary), image):
            raise OSError(f'failed writing image cache {temporary}')
        temporary.replace(target)
        metadata = dict(file_name=filename, source_sha256=digest(source), preprocessing=self.preprocessing)
        (self.cache / f'{signature}.json').write_text(json.dumps(metadata), encoding='utf-8')
        return image

def tile_positions(size, tile, overlap=0.25):
    if tile <= 0 or tile > size or (not 0 <= overlap < 1):
        raise ValueError('invalid tile size or overlap')
    stride = max(1, int(round(tile * (1 - overlap))))
    positions = list(range(0, size - tile + 1, stride))
    if positions[-1] != size - tile:
        positions.append(size - tile)
    return positions

def blend_window(tile):
    one = np.hanning(tile).astype(np.float32)
    return np.maximum(one[:, None] * one[None, :], 0.001)

@torch.inference_mode()
def predict_foreground(model, image, device, tile=768, overlap=0.25, batch=1, amp_dtype=None):
    model.eval()
    height, width = image.shape
    coordinates = [(y, x) for y in tile_positions(height, tile, overlap) for x in tile_positions(width, tile, overlap)]
    weight = blend_window(tile)
    numerator, denominator = (np.zeros((height, width), np.float32), np.zeros((height, width), np.float32))
    for start in range(0, len(coordinates), batch):
        selected = coordinates[start:start + batch]
        inputs = torch.from_numpy(np.stack([model_input(image[y:y + tile, x:x + tile]) for y, x in selected])).to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
            output = model(inputs)
        if output.shape[-2:] != (tile, tile) or output.shape[1] != 2:
            raise ValueError(f'decoder must output native 2-channel masks, got {tuple(output.shape)}')
        probabilities = output[:, 0].float().sigmoid().cpu().numpy()
        for (y, x), p in zip(selected, probabilities):
            numerator[y:y + tile, x:x + tile] += p * weight
            denominator[y:y + tile, x:x + tile] += weight
    if np.any(denominator <= 0):
        raise ValueError('tile stitching left uncovered pixels')
    return numerator / denominator

def iter_components(probability, threshold, min_area):
    labels, count = ndimage.label(probability >= threshold, structure=np.ones((3, 3), np.uint8))
    areas = np.bincount(labels.ravel())
    for i in range(1, count + 1):
        if areas[i] >= min_area:
            yield (labels == i)

def score(pred):
    tp = fp = fn = 0
    total_iou = 0.0
    for group, truth in GT.items():
        predicted = pred.get(group.split('-', 1)[1], [])
        if not predicted:
            fn += len(truth)
            continue
        iou = np.asarray(mu.iou(truth, predicted, [0] * len(predicted)))
        hit = iou > 0.5
        tp += int(hit.sum())
        fp += int((~hit.any(axis=0)).sum())
        fn += int((~hit.any(axis=1)).sum())
        total_iou += float(iou[hit].sum())
    denominator = tp + 0.5 * fp + 0.5 * fn
    return dict(PQ=total_iou / denominator if denominator else 0.0, SQ=total_iou / tp if tp else 0.0, RQ=tp / denominator if denominator else 0.0, TP=tp, FP=fp, FN=fn)
def prepare_split(data):
    by_id = {row['id']: row for row in data['images']}
    assert len(by_id) == len(data['images']), '중복 annotation image ID'

    def read(path):
        with path.open(newline='', encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        ids = {row['image_id'] for row in rows}
        assert len(ids) == len(rows) and ids <= set(by_id), 'split ID 오류'
        assert all((by_id[row['image_id']]['file_name'] == row['file_name'] for row in rows)), 'split 파일명 불일치'
        return (ids, {row['file_name'] for row in rows})
    train_ids, train_files = read(TRAIN)
    val_ids, val_files = read(VAL)
    assert not train_ids & val_ids and (not train_files & val_files), 'train/val 중복'
    assert val_ids == {row['id'] for row in data['images'] if row['file_name'] in val_files}, 'val annotator 누락'
    assert len(train_files) == 616 and len(val_files) == 91 and (len(val_ids) == 101)
    assert all(((IMAGES / name).is_file() for name in train_files | val_files)), '원본 JPEG 누락'
    truth = defaultdict(list)
    for ann in data['annotations']:
        if ann['image_id'] in val_ids:
            truth[ann['image_id']].append(mu.merge(mu.frPyObjects(ann['segmentation'], 2048, 2048)))
    assert set(truth) == val_ids and sum(map(len, truth.values())) == 708
    return (train_ids, sorted(val_files), dict(truth))

def write_csv(path, predicted):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filament_id', 'segmentation_rle'])
        for stem, instances in predicted.items():
            writer.writerows(((f'{stem}_{i}', rle['counts'].decode('ascii')) for i, rle in enumerate(instances, 1)))

def package_results(output, summary):
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    archive = output.parent / f'{output.name}_native_unet_results.zip'
    temporary = archive.with_suffix('.tmp.zip')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(output.iterdir()):
            if path.suffix in ('.json', '.csv', '.py'):
                z.write(path, arcname=path.name)
    temporary.replace(archive)
    return archive

def validate_native(model, store, names, device, amp_dtype, cfg):
    thresholds, areas = cfg['thresholds'], cfg['min_areas']
    predictions = {(t, a): {} for t in thresholds for a in areas}
    for index, name in enumerate(names, 1):
        probability = predict_foreground(model, store.load(name), device,
            cfg.get('validation_tile', cfg['crop']), cfg['overlap'],
            cfg.get('infer_batch', cfg['batch']), amp_dtype)
        for threshold in thresholds:
            rles = [mu.encode(np.asfortranarray(mask.astype(np.uint8)))
                    for mask in iter_components(probability, threshold, min(areas))]
            for area in areas:
                predictions[threshold, area][Path(name).stem] = [
                    rle for rle in rles if int(mu.area(rle)) >= area]
        print(f'validation [{index}/{len(names)}] {name}', flush=True)
    results = []
    for (threshold, area), predicted in predictions.items():
        metric = score(predicted)
        results.append(dict(threshold=threshold, min_area=area,
            pq=metric['PQ'], sq=metric['SQ'], rq=metric['RQ'],
            tp_pairs=metric['TP'], fp=metric['FP'], fn=metric['FN']))
    return max(results, key=lambda row: row['pq']), results, predictions

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import uuid
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from pycocotools import mask as mu

def annotation_mask(annotation, synthetic=False, shape=(2048, 2048)):
    """Rasterize one instance without changing the native annotation geometry."""
    height, width = shape
    segmentation = annotation['segmentation']
    if synthetic:
        if not isinstance(segmentation, list):
            raise ValueError('Synthetic segmentation must be a polygon list')
        mask = np.zeros(shape, np.uint8)
        for polygon in segmentation:
            points = np.asarray(polygon, np.float32)
            if points.size < 6 or points.size % 2 or (not np.isfinite(points).all()):
                raise ValueError('Invalid synthetic polygon')
            cv2.fillPoly(mask, [np.rint(points.reshape(-1, 2)).astype(np.int32)], 1)
        return mask
    if not isinstance(segmentation, list):
        raise ValueError('Real segmentation must be a COCO polygon list')
    rles = mu.frPyObjects(segmentation, height, width)
    if not rles:
        return np.zeros(shape, np.uint8)
    return mu.decode(mu.merge(rles)).astype(np.uint8)

def genuine_spine(annotation):
    """Return finite native xy points, or None when supervision is absent."""
    value = annotation.get('spine')
    if value is None:
        return None
    try:
        xy = np.asarray(value, np.float64)
    except (ValueError, TypeError):
        return None
    if xy.size < 4 or xy.size % 2 or (not np.isfinite(xy).all()):
        return None
    if xy.ndim not in (1, 2) or (xy.ndim == 2 and xy.shape[1] != 2):
        return None
    return xy.reshape(-1, 2)

class MixedCropDataset(Dataset):

    def __init__(self, data, train_ids, real_store, synthetic_data=None, synthetic_store=None, crop=1024, length=400, seed=42, synthetic_ratio=0.2, warmup_epochs=3, real_finish_epochs=6, total_epochs=30, cache_dir=None, include_empty_synthetic=False, augment=True):
        if not isinstance(crop, int) or not 1 <= crop <= 2048 or length <= 0:
            raise ValueError('Crop must fit the native image and length must be positive')
        if not 0 <= synthetic_ratio <= 1 or min(warmup_epochs, real_finish_epochs) < 0:
            raise ValueError('Invalid synthetic schedule')
        if total_epochs <= 0 or warmup_epochs + real_finish_epochs > total_epochs:
            raise ValueError('Warmup and real-only finish must fit the training budget')
        self.crop, self.length, self.seed = (crop, int(length), int(seed))
        self.synthetic_ratio = float(synthetic_ratio)
        self.warmup_epochs, self.real_finish_epochs = (int(warmup_epochs), int(real_finish_epochs))
        self.total_epochs, self.augment, self.epoch = (int(total_epochs), bool(augment), 0)
        self.stores = {'real': real_store, 'synthetic': synthetic_store}
        self.cache = Path(cache_dir) if cache_dir is not None else None
        if self.cache is not None:
            if self.cache.resolve().is_relative_to(Path('/content/drive')):
                raise ValueError('Native target cache must use local Colab storage')
            self.cache.mkdir(parents=True, exist_ok=True)
        ids = {str(value) for value in train_ids}
        known = {str(row['id']) for row in data['images']}
        if not ids or not ids <= known:
            raise ValueError('Unknown or empty real training IDs')
        self.domains = {'real': self._domain(data, ids, False, False)}
        if synthetic_data is not None:
            if synthetic_store is None:
                raise ValueError('Synthetic data requires a separate image store')
            self.domains['synthetic'] = self._domain(synthetic_data, None, True, not include_empty_synthetic)
        self.small_area = self.domains['real']['small_area']
        self.files = self.domains['real']['files']
        self.domain_info = {name: dict(physical_images=len(domain['files']), annotation_groups=len(domain['images']), annotation_instances=sum(map(len, domain['annotations'].values())), excluded_blank_groups=domain['excluded_blank_groups'], small_area=domain['small_area']) for name, domain in self.domains.items()}

    def _domain(self, data, allowed, synthetic, exclude_blank):
        images = {}
        for row in data['images']:
            key = str(row['id'])
            if allowed is not None and key not in allowed:
                continue
            if key in images:
                raise ValueError('Duplicate annotation image ID within a domain')
            if row.get('width', 2048) != 2048 or row.get('height', 2048) != 2048:
                raise ValueError('Expected native 2048-square metadata')
            image = dict(row)
            image['file_name'] = Path(row['file_name']).name if synthetic else row['file_name']
            if Path(image['file_name']).name != image['file_name']:
                raise ValueError('Real image filename must not contain a directory')
            images[key] = image
        annotations = defaultdict(list)
        for source in data['annotations']:
            key = str(source['image_id'])
            if key not in images:
                continue
            ann = dict(source)
            area, box = (ann.get('area'), ann.get('bbox'))
            valid_box = isinstance(box, (list, tuple)) and len(box) == 4 and np.isfinite(np.asarray(box, float)).all() and (box[2] > 0) and (box[3] > 0)
            if not isinstance(area, (int, float)) or not np.isfinite(area) or area <= 0 or (not valid_box):
                native = annotation_mask(ann, synthetic)
                area = int(native.sum())
                if area == 0:
                    raise ValueError('Empty annotated instance')
                ann['area'] = area
                ann['bbox'] = list(cv2.boundingRect(native))
            if not synthetic and genuine_spine(ann) is None:
                raise ValueError('Real annotation is missing genuine spine supervision')
            annotations[key].append(ann)
        blanks = [key for key in images if not annotations[key]]
        if exclude_blank:
            for key in blanks:
                images.pop(key)
                annotations.pop(key, None)
        variants = defaultdict(list)
        for key, image in images.items():
            variants[image['file_name']].append(key)
        if not variants:
            raise ValueError('No eligible images in training domain')
        areas = [ann['area'] for key in images for ann in annotations[key]]
        if not areas:
            raise ValueError('Training domain contains no positive annotations')
        return dict(images=images, annotations=dict(annotations), variants=dict(variants), files=sorted(variants), small_area=float(np.quantile(areas, 0.25)), synthetic=synthetic, excluded_blank_groups=len(blanks) if exclude_blank else 0)

    def __len__(self):
        return self.length

    def synthetic_probability(self, epoch=None):
        epoch = self.epoch if epoch is None else int(epoch)
        if 'synthetic' not in self.domains or epoch <= self.warmup_epochs or epoch > self.total_epochs - self.real_finish_epochs:
            return 0.0
        return self.synthetic_ratio

    def sample_record(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, int(index)]))
        name = 'synthetic' if rng.random() < self.synthetic_probability() else 'real'
        domain = self.domains[name]
        filename = domain['files'][int(rng.integers(len(domain['files'])))]
        variants = domain['variants'][filename]
        group = variants[int(rng.integers(len(variants)))]
        anns = domain['annotations'][group]
        choice = rng.random()
        candidates = [a for a in anns if a['area'] <= domain['small_area']] if choice < 0.5 else anns
        box = None
        if choice < 0.75 and candidates:
            box = candidates[int(rng.integers(len(candidates)))]['bbox']
        elif choice < 0.5 and anns:
            box = anns[int(rng.integers(len(anns)))]['bbox']
        top, left = crop_origin(2048, 2048, self.crop, rng, box)
        return dict(domain=name, file_name=filename, group_id=group, top=top, left=left, rng=rng)

    def _target(self, domain_name, group):
        domain = self.domains[domain_name]
        anns = domain['annotations'][group]
        source = dict(version='native_mixed_target_v1_cvsynthetic_cocoreal', domain=domain_name, image=domain['images'][group], annotations=anns)
        key = hashlib.sha256(json.dumps(source, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()
        path = self.cache / f'{key}.png' if self.cache is not None else None
        if path is not None and path.is_file():
            target = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if target is None or target.shape != (2048, 2048, 3) or target.dtype != np.uint8:
                raise ValueError(f'Invalid native target cache: {path}')
            return (target[:, :, 0], target[:, :, 1], bool(target[0, 0, 2]))
        fg = np.zeros((2048, 2048), np.uint8)
        spine = np.zeros_like(fg)
        spine_valid = not domain['synthetic'] or bool(anns)
        for ann in anns:
            fg |= annotation_mask(ann, domain['synthetic'])
            xy = genuine_spine(ann)
            if xy is None:
                spine_valid = False
            else:
                cv2.polylines(spine, [np.rint(xy).astype(np.int32)], False, 1, thickness=3)
        spine &= fg
        if path is not None:
            target = np.stack([fg, spine, np.full_like(fg, int(spine_valid))], axis=-1)
            temporary = path.with_name(f'{key}.{uuid.uuid4().hex}.png')
            if not cv2.imwrite(str(temporary), target, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                raise OSError(f'Failed writing target cache: {temporary}')
            temporary.replace(path)
        return (fg, spine, spine_valid)

    def __getitem__(self, index):
        row = self.sample_record(index)
        top, left = (row['top'], row['left'])
        gray = self.stores[row['domain']].load(row['file_name'])
        if gray.dtype != np.uint8 or gray.shape != (2048, 2048):
            raise ValueError('Image store must return native uint8 2048-square pixels')
        image = gray[top:top + self.crop, left:left + self.crop].copy()
        foreground, spine, valid = self._target(row['domain'], row['group_id'])
        target = np.stack([foreground[top:top + self.crop, left:left + self.crop], spine[top:top + self.crop, left:left + self.crop]]).astype(np.float32)
        rng = row['rng']
        if self.augment:
            values = np.clip(image.astype(np.float32) / 255.0, 0, 1)
            image = (np.clip(values ** rng.uniform(0.9, 1.1) * rng.uniform(0.85, 1.15), 0, 1) * 255).astype(np.uint8)
        x = model_input(image)
        if self.augment:
            x, target = apply_d4(x, target, int(rng.integers(4)), bool(rng.integers(2)))
        return (torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(target)), torch.tensor(float(valid), dtype=torch.float32))

def mixed_segmentation_loss(logits, target, spine_valid, spine_weight=0.1):
    """Learn all foreground labels; never invent negative missing spine labels."""
    if logits.shape != target.shape or logits.ndim != 4 or logits.shape[1] != 2:
        raise ValueError('Expected matching B,2,H,W logits and target')
    logits, target = (logits.float(), target.float())
    valid = torch.as_tensor(spine_valid, device=logits.device, dtype=logits.dtype)
    if valid.numel() != logits.shape[0] or torch.any((valid != 0) & (valid != 1)):
        raise ValueError('spine_valid must contain one binary flag per sample')
    valid = valid.reshape(-1, 1, 1)
    fg, truth = (logits[:, 0], target[:, 0])
    probability = fg.sigmoid()
    dice = 1 - ((2 * (probability * truth).sum((1, 2)) + 1) / (probability.sum((1, 2)) + truth.sum((1, 2)) + 1)).mean()
    foreground_loss = F.binary_cross_entropy_with_logits(fg, truth) + dice
    line, spine = (logits[:, 1], target[:, 1])
    losses = F.binary_cross_entropy_with_logits(line, spine, reduction='none')
    positives, negatives = ((spine * valid).sum(), ((1 - spine) * valid).sum())
    negative_loss = (losses * (1 - spine) * valid).sum() / negatives.clamp_min(1)
    if positives.item() > 0:
        spine_loss = 0.5 * ((losses * spine * valid).sum() / positives) + 0.5 * negative_loss
    else:
        spine_loss = negative_loss
    return foreground_loss + spine_weight * spine_loss
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
RESUME_KEYS = ('encoder', 'crop', 'batch', 'accumulate', 'steps_per_epoch', 'epochs', 'lr', 'spine_weight', 'synthetic_ratio', 'warmup_epochs', 'real_finish_epochs', 'include_empty_synthetic', 'seed', 'amp_dtype', 'source_sha256', 'thresholds', 'min_areas', 'overlap', 'validate_every', 'preprocessing')

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
    numpy_state = np.random.get_state()
    return dict(python=random.getstate(), torch=torch.get_rng_state(), numpy=(numpy_state[0], torch.from_numpy(numpy_state[1].astype(np.int64)), int(numpy_state[2]), int(numpy_state[3]), float(numpy_state[4])), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

def restore_rng(state):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'].cpu())
    numpy_state = state['numpy']
    np.random.set_state((numpy_state[0], numpy_state[1].cpu().numpy().astype(np.uint32), numpy_state[2], numpy_state[3], numpy_state[4]))
    if state.get('cuda') and torch.cuda.is_available():
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('Resume CUDA device count differs from the saved run')
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])

def check_resume_config(saved, current):
    differences = {key: dict(saved=saved.get(key), current=current.get(key)) for key in RESUME_KEYS if saved.get(key) != current.get(key)}
    if differences:
        raise ValueError('Resume configuration differs: ' + json.dumps(differences, ensure_ascii=False))

def checkpoint_payload(model, optimizer, scheduler, scaler, cfg, epoch, history, best_pq, best_epoch, elapsed_seconds):
    return dict(schema_version=1, model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(), scheduler_state_dict=scheduler.state_dict(), scaler_state_dict=scaler.state_dict(), config=cfg.copy(), epoch=int(epoch), history=history, best_pq=float(best_pq), best_epoch=int(best_epoch), elapsed_seconds=float(elapsed_seconds), rng_state=capture_rng())

def restore_training_state(checkpoint, model, optimizer, scheduler, scaler, cfg):
    check_resume_config(checkpoint['config'], cfg)
    required = {'optimizer_state_dict', 'scheduler_state_dict', 'scaler_state_dict', 'rng_state', 'history', 'best_pq', 'best_epoch'}
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
    return dict(start_epoch=epoch + 1, history=history, best_pq=float(checkpoint['best_pq']), best_epoch=int(checkpoint['best_epoch']), elapsed_seconds=float(checkpoint.get('elapsed_seconds', 0.0)))

def make_epoch_loader(dataset, cfg, device):
    workers = int(cfg.get('workers', 2))
    if workers < 0:
        raise ValueError('workers must be nonnegative')
    generator = torch.Generator().manual_seed(int(cfg['seed']) + int(dataset.epoch))
    kwargs = dict(batch_size=int(cfg['batch']), shuffle=False, num_workers=workers, pin_memory=device.type == 'cuda', persistent_workers=False, generator=generator)
    if workers:
        kwargs['multiprocessing_context'] = 'spawn'
    return DataLoader(dataset, **kwargs)

def storage_estimate(model):
    model_bytes = sum((value.numel() * value.element_size() for value in model.state_dict().values()))
    return dict(model_mib=model_bytes / 1024 ** 2, full_last_estimated_mib=3 * model_bytes / 1024 ** 2, persistent_best_and_last_estimated_mib=4 * model_bytes / 1024 ** 2, atomic_save_peak_estimated_mib=7 * model_bytes / 1024 ** 2, note='Model plus Adam moments estimate; Drive FUSE free-space is not a quota guarantee')

def export_best_predictions(output, best_payload):
    """A complete best checkpoint can repair interrupted CSV/metadata exports."""
    predictions = best_payload.get('validation_predictions')
    if predictions is None:
        return
    for (threshold, area), predicted in predictions.items():
        write_csv(Path(output) / f'best_t{threshold:g}_a{area}.csv', predicted)
    best = best_payload['validation']
    write_csv(Path(output) / 'best_validation.csv', predictions[best['threshold'], best['min_area']])
    atomic_json_save(Path(output) / 'best_epoch.json', dict(epoch=best_payload['epoch'], best=best, grid=best_payload['validation_grid']))

def normalize_training_config(config):
    cfg = dict(encoder='tu-convnext_tiny', pretrained=True, crop=1024, batch=2, accumulate=2, steps_per_epoch=200, epochs=30, validate_every=2, lr=0.0001, spine_weight=0.1, seed=42, workers=2, synthetic_ratio=0.2, warmup_epochs=3, real_finish_epochs=6, include_empty_synthetic=True, thresholds=[0.35, 0.5, 0.65], min_areas=[100, 200, 300], overlap=0.25, device='cuda', resume_dir=None, cache_root='/content/filament_a100_cache', preprocessing='limb_only')
    cfg.update(config)
    paths = [('images', 'IMAGES'), ('annotations', 'ANN'), ('train_split', 'TRAIN'), ('val_split', 'VAL'), ('synthetic_images', 'SYNTH_IMAGE_DIR'), ('synthetic_annotations', 'SYNTH_ANN_PATH'), ('synthetic_split', 'SYNTH_SPLIT_INFO_PATH')]
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
    if min((cfg[key] for key in ('batch', 'accumulate', 'steps_per_epoch', 'epochs', 'validate_every'))) <= 0:
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
    return dict(source_train_ids=len(source_train), source_train_physical_files=len(source_files), fixed_train_physical_files=len(real_train_files), fixed_val_physical_files=len(val_names), donor_validation_overlap=0, fixed_split_changed=False)

def preflight_synthetic_images(synthetic_data, image_directory):
    """Check every referenced file, then decode at most three native samples."""
    directory = Path(image_directory)
    images = synthetic_data.get('images', [])
    if not images:
        raise ValueError('Synthetic metadata contains no image records')
    records = {str(row['id']): row for row in images}
    if len(records) != len(images):
        raise ValueError('Duplicate synthetic image IDs')
    counts = Counter((str(row['image_id']) for row in synthetic_data.get('annotations', [])))
    if not set(counts) <= set(records):
        raise ValueError('Synthetic annotation references an unknown image ID')
    filenames = {key: Path(row['file_name']).name for key, row in records.items()}
    if any((name in {'', '.', '..'} for name in filenames.values())):
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
        samples.append(dict(file_name=filenames[key], shape=list(gray.shape), dtype=str(gray.dtype), label_count=int(counts[key])))
    return dict(image_records=len(images), physical_image_files=len(set(filenames.values())), filenames_checked=len(set(filenames.values())), positive_images=len(positives), blank_images=len(blanks), missing_files=0, decoded_samples=samples)

def run_training(config):
    """Run a fresh or resumed train; helpers are provided by the notebook builder."""
    global GT, IMAGES, ANN, TRAIN, VAL
    cfg = normalize_training_config(config)
    device = torch.device(cfg['device'])
    if device.type == 'cuda' and (not torch.cuda.is_available()):
        raise ValueError('Select the A100 GPU runtime in Colab')
    cv2.setNumThreads(2)
    for key, name in [('images', 'IMAGES'), ('annotations', 'ANN'), ('train_split', 'TRAIN'), ('val_split', 'VAL')]:
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
    mismatches = {key: cfg['source_sha256'].get(key) for key, value in expected_hashes.items() if cfg['source_sha256'].get(key) != value}
    if mismatches:
        raise ValueError('Input source hashes differ from the verified files: ' + json.dumps(mismatches))
    data = json.loads(Path(cfg['annotations']).read_text(encoding='utf-8-sig'))
    train_ids, val_names, GT = prepare_split(data)
    synthetic_data = json.loads(Path(cfg['synthetic_annotations']).read_text(encoding='utf-8-sig')) if cfg['synthetic_ratio'] else None
    if synthetic_data is not None:
        split_info = json.loads(Path(cfg['synthetic_split']).read_text(encoding='utf-8-sig'))
        cfg['synthetic_source_pool'] = check_synthetic_source_pool(split_info, data, train_ids, val_names)
        cfg['synthetic_image_preflight'] = preflight_synthetic_images(synthetic_data, cfg['synthetic_images'])
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    amp_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device.type == 'cuda' else None
    cfg['amp_dtype'] = str(amp_dtype)
    cfg['gpu'] = torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'
    output = Path(cfg['resume_dir']) if cfg['resume_dir'] else Path(cfg['output_root']) / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8])
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
    synthetic_store = ImageStore(Path(cfg['synthetic_images']), Path(cfg['cache_root']) / 'images_synthetic', 'limb_only') if synthetic_data is not None else None
    dataset = MixedCropDataset(data, train_ids, store, synthetic_data=synthetic_data, synthetic_store=synthetic_store, crop=cfg['crop'], length=cfg['steps_per_epoch'] * cfg['batch'], seed=cfg['seed'], synthetic_ratio=cfg['synthetic_ratio'], warmup_epochs=cfg['warmup_epochs'], real_finish_epochs=cfg['real_finish_epochs'], total_epochs=cfg['epochs'], cache_dir=Path(cfg['cache_root']) / 'targets', include_empty_synthetic=cfg['include_empty_synthetic'], augment=True)
    cfg.update(train_physical_images=len(dataset.files), validation_physical_images=len(val_names), validation_annotation_groups=len(GT), small_object_area_threshold_train_only=float(dataset.small_area), data_domains=dataset.domain_info, checkpoint_policy='one best plus one full last; caches local only', effective_batch=cfg['batch'] * cfg['accumulate'], samples_per_epoch=len(dataset), loss='foreground BCE+Dice + balanced spine BCE on valid supervision only', initialization='completed-epoch optimizer/RNG resume' if checkpoint is not None else 'ImageNet encoder with new native decoder', native_output_heads=['foreground', 'spine'], validation_selection='Official global all-pairs PQ with strict IoU>.5', versions={name: importlib.metadata.version(name) for name in ('torch', 'segmentation-models-pytorch', 'timm', 'numpy', 'scipy', 'pycocotools')})
    model = build_model(cfg['encoder'], pretrained=bool(cfg['pretrained'] and checkpoint is None)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'], eta_min=cfg['lr'] * 0.05)
    scaler = torch.amp.GradScaler('cuda', enabled=amp_dtype == torch.float16)
    if checkpoint is not None:
        restored = restore_training_state(checkpoint, model, optimizer, scheduler, scaler, cfg)
        history, best_pq, best_epoch = (restored['history'], restored['best_pq'], restored['best_epoch'])
        start_epoch, elapsed_before = (restored['start_epoch'], restored['elapsed_seconds'])
        best_path = output / 'best.pt'
        if best_path.is_file():
            best_payload = torch.load(best_path, map_location='cpu', weights_only=True)
            if best_payload['validation']['pq'] >= best_pq:
                best_pq, best_epoch = (float(best_payload['validation']['pq']), int(best_payload['epoch']))
                export_best_predictions(output, best_payload)
            del best_payload
        del checkpoint
    else:
        history, best_pq, best_epoch, start_epoch, elapsed_before = ([], -1.0, 0, 1, 0.0)
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
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        domains = Counter((dataset.sample_record(i)['domain'] for i in range(len(dataset))))
        for step, (x, y, spine_valid) in enumerate(loader):
            x, y, spine_valid = (value.to(device, non_blocking=True) for value in (x, y, spine_valid))
            with torch.autocast(device_type=device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
                logits = model(x)
            loss = mixed_segmentation_loss(logits, y, spine_valid, cfg['spine_weight'])
            if not torch.isfinite(loss):
                raise ValueError(f'Nonfinite loss at epoch{epoch} step{step}')
            group = min(cfg['accumulate'], len(loader) - step // cfg['accumulate'] * cfg['accumulate'])
            scaler.scale(loss / group).backward()
            if (step + 1) % cfg['accumulate'] == 0 or step + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
            if step == 0 or (step + 1) % max(1, len(loader) // 10) == 0:
                print(f"epoch{epoch}/{cfg['epochs']} step{step + 1}/{len(loader)} loss{losses[-1]:.5f}", flush=True)
        row = dict(epoch=epoch, train_loss=float(np.mean(losses)), lr=float(optimizer.param_groups[0]['lr']), microbatches=len(loader), optimizer_updates=math.ceil(len(loader) / cfg['accumulate']), synthetic_probability=float(dataset.synthetic_probability()), sampled_domains=dict(domains), elapsed_seconds=elapsed_before + time.monotonic() - started)
        scheduler.step()
        if epoch == 1 or epoch % cfg['validate_every'] == 0 or epoch == cfg['epochs']:
            best, grid, predictions = validate_native(model, store, val_names, device, amp_dtype, cfg)
            row.update(validation=best, validation_grid=grid)
            if best['pq'] > best_pq:
                best_pq, best_epoch = (float(best['pq']), epoch)
                best_payload = dict(schema_version=1, model_state_dict=model.state_dict(), config=cfg.copy(), epoch=epoch, validation=best, validation_grid=grid, validation_predictions=predictions)
                atomic_torch_save(output / 'best.pt', best_payload)
                export_best_predictions(output, best_payload)
                del best_payload
        history.append(row)
        payload = checkpoint_payload(model, optimizer, scheduler, scaler, cfg, epoch, history, best_pq, best_epoch, row['elapsed_seconds'])
        atomic_torch_save(output / 'last.pt', payload)
        del payload
        atomic_json_save(output / 'history.json', history)
        summary = dict(status='complete' if epoch == cfg['epochs'] else 'running', best_development_pq=best_pq, best_epoch=best_epoch, epochs_completed=epoch, target_pq=0.5, target_exceeded=best_pq > 0.5, checkpoint_directory=str(output), config=cfg, scope='Same fixed development split and official global all-pairs PQ; not Kaggle LB')
        if 'validation' in row:
            archive = package_results(output, summary)
            print('자동 저장 ZIP:', archive, flush=True)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        del loader
        gc.collect()
    summary = dict(status='complete', best_development_pq=best_pq, best_epoch=best_epoch, epochs_completed=len(history), target_pq=0.5, target_exceeded=best_pq > 0.5, checkpoint_directory=str(output), config=cfg, scope='Same fixed development split; completed-epoch optimizer resume supported; not Kaggle LB')
    archive = package_results(output, summary)
    return dict(output=str(output), archive=str(archive), summary=summary)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--result', required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    result = run_training(cfg)
    Path(args.result).write_text(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()

