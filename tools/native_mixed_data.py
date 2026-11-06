"""Native real/synthetic crop sampling and masked auxiliary spine supervision.

Synthetic labels preserve the original generator's rounded OpenCV polygon
rasterization. Real annotations use the competition's COCO rasterization.
"""
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

from train_filament_unet import apply_d4, crop_origin, model_input


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
            if points.size < 6 or points.size % 2 or not np.isfinite(points).all():
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
    if xy.size < 4 or xy.size % 2 or not np.isfinite(xy).all():
        return None
    if xy.ndim not in (1, 2) or (xy.ndim == 2 and xy.shape[1] != 2):
        return None
    return xy.reshape(-1, 2)


class MixedCropDataset(Dataset):
    def __init__(self, data, train_ids, real_store, synthetic_data=None,
                 synthetic_store=None, crop=1024, length=400, seed=42,
                 synthetic_ratio=.2, warmup_epochs=3, real_finish_epochs=6,
                 total_epochs=30, cache_dir=None, include_empty_synthetic=False,
                 augment=True):
        if not isinstance(crop, int) or not 1 <= crop <= 2048 or length <= 0:
            raise ValueError('Crop must fit the native image and length must be positive')
        if not 0 <= synthetic_ratio <= 1 or min(warmup_epochs, real_finish_epochs) < 0:
            raise ValueError('Invalid synthetic schedule')
        if total_epochs <= 0 or warmup_epochs+real_finish_epochs > total_epochs:
            raise ValueError('Warmup and real-only finish must fit the training budget')
        self.crop, self.length, self.seed = crop, int(length), int(seed)
        self.synthetic_ratio = float(synthetic_ratio)
        self.warmup_epochs, self.real_finish_epochs = int(warmup_epochs), int(real_finish_epochs)
        self.total_epochs, self.augment, self.epoch = int(total_epochs), bool(augment), 0
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
            self.domains['synthetic'] = self._domain(synthetic_data, None, True,
                                                       not include_empty_synthetic)
        self.small_area = self.domains['real']['small_area']
        self.files = self.domains['real']['files']
        self.domain_info = {name: dict(physical_images=len(domain['files']),
            annotation_groups=len(domain['images']),
            annotation_instances=sum(map(len, domain['annotations'].values())),
            excluded_blank_groups=domain['excluded_blank_groups'],
            small_area=domain['small_area']) for name, domain in self.domains.items()}

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
            area, box = ann.get('area'), ann.get('bbox')
            valid_box = (isinstance(box, (list, tuple)) and len(box) == 4
                         and np.isfinite(np.asarray(box, float)).all()
                         and box[2] > 0 and box[3] > 0)
            if not isinstance(area, (int, float)) or not np.isfinite(area) or area <= 0 or not valid_box:
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
        return dict(images=images, annotations=dict(annotations), variants=dict(variants),
                    files=sorted(variants), small_area=float(np.quantile(areas, .25)),
                    synthetic=synthetic, excluded_blank_groups=len(blanks) if exclude_blank else 0)

    def __len__(self):
        return self.length

    def synthetic_probability(self, epoch=None):
        epoch = self.epoch if epoch is None else int(epoch)
        if ('synthetic' not in self.domains or epoch <= self.warmup_epochs
                or epoch > self.total_epochs-self.real_finish_epochs):
            return 0.
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
        candidates = [a for a in anns if a['area'] <= domain['small_area']] if choice < .5 else anns
        box = None
        if choice < .75 and candidates:
            box = candidates[int(rng.integers(len(candidates)))]['bbox']
        elif choice < .5 and anns:
            box = anns[int(rng.integers(len(anns)))]['bbox']
        top, left = crop_origin(2048, 2048, self.crop, rng, box)
        return dict(domain=name, file_name=filename, group_id=group, top=top, left=left, rng=rng)

    def _target(self, domain_name, group):
        domain = self.domains[domain_name]
        anns = domain['annotations'][group]
        source = dict(version='native_mixed_target_v1_cvsynthetic_cocoreal', domain=domain_name,
                      image=domain['images'][group], annotations=anns)
        key = hashlib.sha256(json.dumps(source, sort_keys=True, separators=(',', ':'),
                                        allow_nan=False).encode('utf-8')).hexdigest()
        path = self.cache/f'{key}.png' if self.cache is not None else None
        if path is not None and path.is_file():
            target = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if target is None or target.shape != (2048, 2048, 3) or target.dtype != np.uint8:
                raise ValueError(f'Invalid native target cache: {path}')
            return target[:, :, 0], target[:, :, 1], bool(target[0, 0, 2])
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
        return fg, spine, spine_valid

    def __getitem__(self, index):
        row = self.sample_record(index)
        top, left = row['top'], row['left']
        gray = self.stores[row['domain']].load(row['file_name'])
        if gray.dtype != np.uint8 or gray.shape != (2048, 2048):
            raise ValueError('Image store must return native uint8 2048-square pixels')
        image = gray[top:top+self.crop, left:left+self.crop].copy()
        foreground, spine, valid = self._target(row['domain'], row['group_id'])
        target = np.stack([foreground[top:top+self.crop, left:left+self.crop],
                           spine[top:top+self.crop, left:left+self.crop]]).astype(np.float32)
        rng = row['rng']
        if self.augment:
            values = np.clip(image.astype(np.float32)/255., 0, 1)
            image = (np.clip(values**rng.uniform(.9, 1.1)*rng.uniform(.85, 1.15), 0, 1)*255).astype(np.uint8)
        x = model_input(image)
        if self.augment:
            x, target = apply_d4(x, target, int(rng.integers(4)), bool(rng.integers(2)))
        return (torch.from_numpy(np.ascontiguousarray(x)),
                torch.from_numpy(np.ascontiguousarray(target)),
                torch.tensor(float(valid), dtype=torch.float32))


def mixed_segmentation_loss(logits, target, spine_valid, spine_weight=.1):
    """Learn all foreground labels; never invent negative missing spine labels."""
    if logits.shape != target.shape or logits.ndim != 4 or logits.shape[1] != 2:
        raise ValueError('Expected matching B,2,H,W logits and target')
    logits, target = logits.float(), target.float()
    valid = torch.as_tensor(spine_valid, device=logits.device, dtype=logits.dtype)
    if valid.numel() != logits.shape[0] or torch.any((valid != 0) & (valid != 1)):
        raise ValueError('spine_valid must contain one binary flag per sample')
    valid = valid.reshape(-1, 1, 1)
    fg, truth = logits[:, 0], target[:, 0]
    probability = fg.sigmoid()
    dice = 1-((2*(probability*truth).sum((1, 2))+1)/
             (probability.sum((1, 2))+truth.sum((1, 2))+1)).mean()
    foreground_loss = F.binary_cross_entropy_with_logits(fg, truth)+dice
    line, spine = logits[:, 1], target[:, 1]
    losses = F.binary_cross_entropy_with_logits(line, spine, reduction='none')
    positives, negatives = (spine*valid).sum(), ((1-spine)*valid).sum()
    negative_loss = (losses*(1-spine)*valid).sum()/negatives.clamp_min(1)
    if positives.item() > 0:
        spine_loss = .5*((losses*spine*valid).sum()/positives)+.5*negative_loss
    else:
        spine_loss = negative_loss
    return foreground_loss+spine_weight*spine_loss
