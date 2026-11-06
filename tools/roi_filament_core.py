"""Box-conditioned, single-instance U-Net and train-only ROI sampling.

The first channel is grayscale in[0,1], the second is an original-box prompt.
Only the image channel is ImageNet-normalized inside the model. Geometry is
shared between training and inference; outputs are restored to native pixels.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from native_instance_data import NativeInstanceDataset, annotation_rle, limb

SIZE = 2048
ENCODERS = ('resnet18', 'resnet34', 'resnet50')


def prepare_image(pixels, mode='limb'):
    """Return exact uint8 preprocessing, with no float/rescale round trip."""
    if pixels is None or pixels.dtype != np.uint8 or pixels.shape != (SIZE, SIZE):
        raise ValueError('Original grayscale uint8 2048-square image required')
    if mode not in ('limb', 'raw'):
        raise ValueError('preprocessing must be limb or raw')
    return limb(pixels) if mode == 'limb' else pixels


def _box(box, height=SIZE, width=SIZE):
    value = np.asarray(box, dtype=float)
    if value.shape != (4,) or not np.isfinite(value).all():
        raise ValueError('Finite box_xyxy with four coordinates required')
    x1, y1, x2, y2 = value.tolist()
    if x2 <= x1 or y2 <= y1:
        raise ValueError('Box must have positive width and height')
    x1, x2 = max(0., x1), min(float(width), x2)
    y1, y2 = max(0., y1), min(float(height), y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError('Box does not intersect the native image')
    return [x1, y1, x2, y2]


def _geometry(box, context, size, height=SIZE, width=SIZE):
    box = _box(box, height, width)
    if not np.isfinite(context) or context < 1:
        raise ValueError('context must be at least1')
    if not isinstance(size, int) or size < 1:
        raise ValueError('Positive integer ROI size required')
    x1, y1, x2, y2 = box
    side = max(1, int(np.ceil(max(x2-x1, y2-y1)*context)))
    left = int(np.floor((x1+x2-side)/2))
    top = int(np.floor((y1+y2-side)/2))
    return dict(box_xyxy=box, left=left, top=top, side=side, size=size,
                image_height=height, image_width=width,
                geometry='expanded_bbox_square_pad_then_resize_v1')


def _extract(array, geometry):
    side, left, top = (geometry[key] for key in ('side', 'left', 'top'))
    height, width = array.shape
    result = np.zeros((side, side), dtype=array.dtype)
    x1, x2 = max(0, left), min(width, left+side)
    y1, y2 = max(0, top), min(height, top+side)
    if x2>x1 and y2>y1:
        result[y1-top:y2-top, x1-left:x2-left] = array[y1:y2, x1:x2]
    return result


def crop_roi(image_uint8, box_xyxy, mask=None, context=1.5, size=384):
    """Crop corrected uint8 image and single-target mask using one transform.

    Expanded context is square-padded without aspect-ratio distortion. An edge
    crop is padded rather than shifted, so the prompt location stays correct.
    """
    if image_uint8 is None or image_uint8.dtype != np.uint8 or image_uint8.shape != (SIZE, SIZE):
        raise ValueError('Preprocessed grayscale uint8 2048-square image required')
    geometry = _geometry(box_xyxy, context, size)
    original_box = geometry['box_xyxy']
    prompt = np.zeros(image_uint8.shape, np.uint8)
    x1, y1 = np.floor(original_box[:2]).astype(int)
    x2, y2 = np.ceil(original_box[2:]).astype(int)
    prompt[y1:y2, x1:x2] = 1
    gray = cv2.resize(_extract(image_uint8, geometry), (size, size), interpolation=cv2.INTER_LINEAR)
    box_prompt = cv2.resize(_extract(prompt, geometry), (size, size), interpolation=cv2.INTER_NEAREST)
    inputs = np.stack([gray.astype(np.float32)/255., box_prompt.astype(np.float32)])
    target = None
    if mask is not None:
        target_array = np.asarray(mask)
        if target_array.shape != image_uint8.shape:
            raise ValueError('Target must use the same original image coordinates')
        target_array = (target_array > 0).astype(np.uint8)
        target = cv2.resize(_extract(target_array, geometry), (size, size),
                            interpolation=cv2.INTER_NEAREST)[None].astype(np.float32)
    return inputs, target, geometry


def restore_mask(probability, transform, threshold=.5):
    """Invert ROI geometry before thresholding; return native COCO RLE orNone."""
    array = np.asarray(probability)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    expected = int(transform['size'])
    if array.shape != (expected, expected) or not np.isfinite(array).all():
        raise ValueError('Finite ROI probability must match the saved input size')
    if np.any(array < 0) or np.any(array > 1) or not 0 < threshold < 1:
        raise ValueError('Probability/threshold must lie within valid ranges')
    side, left, top = (int(transform[key]) for key in ('side', 'left', 'top'))
    height, width = int(transform['image_height']), int(transform['image_width'])
    if side < 1 or min(height, width) < 1:
        raise ValueError('Invalid saved ROI geometry')
    resized = cv2.resize(array.astype(np.float32), (side, side), interpolation=cv2.INTER_LINEAR)
    result = np.zeros((height, width), np.uint8)
    x1, x2 = max(0, left), min(width, left+side)
    y1, y2 = max(0, top), min(height, top+side)
    if x2>x1 and y2>y1:
        result[y1:y2, x1:x2] = resized[y1-top:y2-top, x1-left:x2-left] >= threshold
    if not result.any():
        return None
    return mu.encode(np.asfortranarray(result))


class RoiFilamentNet(nn.Module):
    """Image+box prompt U-Net; output is one requested object's mask logit."""
    def __init__(self, encoder='resnet34', encoder_weights='imagenet'):
        super().__init__()
        if encoder not in ENCODERS:
            raise ValueError('Unsupported ROI encoder')
        import segmentation_models_pytorch as smp
        self.encoder_name = encoder
        self.network = smp.Unet(encoder_name=encoder, encoder_weights=encoder_weights,
                               in_channels=2, classes=1, activation=None)

    def forward(self, inputs):
        if inputs.ndim != 4 or inputs.shape[1] != 2:
            raise ValueError('Expected grayscale+box prompt [B,2,H,W]')
        normalized = torch.cat([(inputs[:, :1]-.449)/.226, inputs[:, 1:2]], dim=1)
        return self.network(normalized)


def roi_loss(logits, target):
    if logits.shape != target.shape or logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError('Matching [B,1,H,W] logits and target required')
    if not torch.isfinite(target).all() or bool(((target<0)|(target>1)).any()):
        raise ValueError('Binary/soft mask targets must be finite in[0,1]')
    prediction = logits.float();target = target.float()
    bce = F.binary_cross_entropy_with_logits(prediction, target)
    probability = prediction.sigmoid()
    dims = (1,2,3)
    dice = (1.-(2*(probability*target).sum(dims)+1.) /
            (probability.sum(dims)+target.sum(dims)+1.)).mean()
    total = bce+dice
    return total, dict(loss=total.detach(), bce=bce.detach(), dice=dice.detach(),
                       positive_examples=(target.sum(dims)>0).float().sum().detach())


def _proposal_box(row):
    if isinstance(row, dict):
        value = row.get('box', row.get('box_xyxy', row.get('bbox_xyxy')))
        if value is None:
            raise ValueError('Proposal must include native box or box_xyxy')
        return _box(value)
    return _box(row)


class RoiTrainingDataset(Dataset):
    """Epoch/index deterministic single-instance ROI examples from train only.

    All alternative-rater foreground is excluded from the original box of a
    negative prompt. Expanded context may contain other objects: the target is
    empty because no object is requested inside the prompt, not because every
    context pixel is background. Unsafe negative attempts fall back to a
    positive example rather than assigning known foreground an empty label.
    """
    def __init__(self, train_groups, images, proposals_by_image=None, crop_size=384,
                 context=1.5, steps_per_epoch=256, batch_size=4, seed=42,
                 preprocessing='limb', cache_dir=None, source_sha256=None):
        self.groups = list(train_groups)
        if not self.groups:
            raise ValueError('Nonempty train groups required')
        for group in self.groups:
            if group.get('role', 'train') != 'train':
                raise ValueError('Validation groups cannot enter ROI training')
        self.size, self.context, self.seed, self.epoch = int(crop_size), float(context), int(seed), 0
        if self.size < 1 or self.context < 1 or min(steps_per_epoch,batch_size)<1:
            raise ValueError('Invalid ROI sample budget or geometry')
        self.length = int(steps_per_epoch)*int(batch_size)
        self.image_reader = NativeInstanceDataset(self.groups, images, crop=1,
            steps_per_epoch=1, batch_size=1, training=False, preprocessing=preprocessing,
            cache_dir=cache_dir, source_sha256=source_sha256)
        self.by_name = defaultdict(list)
        for group in self.groups:
            self.by_name[group['file_name']].extend(group['annotations'])
        self.names = sorted(self.by_name)
        self.positive_groups = [i for i,g in enumerate(self.groups) if g['annotations']]
        if not self.positive_groups:
            raise ValueError('At least one positive ROI training group required')
        self.proposals = {}
        for name, rows in (proposals_by_image or {}).items():
            candidates = [value for value in self.names if value == name or Path(value).stem == name]
            if not candidates:
                raise ValueError('Proposal image is not in train: '+str(name))
            self.proposals[candidates[0]] = [_proposal_box(row) for row in rows]
        self._truth = lru_cache(maxsize=4)(self._image_truth)
        self._mask = lru_cache(maxsize=8)(self._annotation_mask)

    def __len__(self):return self.length

    def set_epoch(self, epoch):
        if int(epoch)<0:raise ValueError('Epoch must be nonnegative')
        self.epoch=int(epoch)

    def _annotation_mask(self, group_index, annotation_index):
        return mu.decode(annotation_rle(self.groups[group_index]['annotations'][annotation_index])).astype(np.uint8)

    def _image_truth(self, name):
        encoded = [annotation_rle(a) for a in self.by_name[name]]
        union = mu.decode(mu.merge(encoded)).astype(bool) if encoded else np.zeros((SIZE,SIZE),bool)
        boxes = np.asarray(mu.toBbox(encoded),float).reshape(-1,4) if encoded else np.empty((0,4),float)
        if len(boxes):boxes[:,2:]+=boxes[:,:2]
        return union, boxes

    def _safe_negative(self, name, box):
        union, truth_boxes = self._truth(name)
        x1,y1=np.floor(box[:2]).astype(int);x2,y2=np.ceil(box[2:]).astype(int)
        # Zero known foreground is stronger than a 5% coverage allowance.
        if union[y1:y2,x1:x2].any():return False
        if len(truth_boxes):
            low=np.maximum(truth_boxes[:,:2],box[:2]);high=np.minimum(truth_boxes[:,2:],box[2:])
            intersection=np.maximum(high-low,0).prod(axis=1)
            other=(truth_boxes[:,2:]-truth_boxes[:,:2]).prod(axis=1)
            area=(box[2]-box[0])*(box[3]-box[1])
            if np.any(intersection/(other+area-intersection)>=.1):return False
        return True

    def sample_record(self, index):
        if not 0<=int(index)<len(self):raise IndexError(index)
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,int(index)]))
        want_negative=bool(rng.random()<.2)
        record=None
        if want_negative:
            name=self.names[int(rng.integers(len(self.names)))]
            rows=self.proposals.get(name,[])
            for position in rng.permutation(len(rows)):
                box=rows[int(position)]
                if self._safe_negative(name,box):
                    record=dict(file_name=name,box=box,positive=False,negative_source='detector_proposal')
                    break
            if record is None:
                for _ in range(64):
                    w,h=rng.uniform(24,192,2);x,y=rng.uniform(0,SIZE-np.asarray([w,h]))
                    box=[float(x),float(y),float(x+w),float(y+h)]
                    if self._safe_negative(name,box):
                        record=dict(file_name=name,box=box,positive=False,negative_source='random_safe_background')
                        break
        if record is None:
            gi=int(rng.choice(self.positive_groups));group=self.groups[gi]
            ai=int(rng.integers(len(group['annotations'])))
            rle=annotation_rle(group['annotations'][ai]);x,y,w,h=map(float,mu.toBbox(rle))
            cx=x+w/2+rng.uniform(-.1,.1)*w;cy=y+h/2+rng.uniform(-.1,.1)*h
            jw,jh=np.asarray([w,h])*rng.uniform(.8,1.2,2)
            box=_box([cx-jw/2,cy-jh/2,cx+jw/2,cy+jh/2])
            record=dict(file_name=group['file_name'],box=box,positive=True,
                        group_index=gi,annotation_index=ai,group_id=group['group_id'],
                        negative_requested=want_negative)
        record.update(rotation=int(rng.integers(4)),flip_x=bool(rng.integers(2)),
                      flip_y=bool(rng.integers(2)),gain=float(rng.uniform(.9,1.1)),gamma=float(rng.uniform(.9,1.1)))
        return record

    def __getitem__(self,index):
        record=self.sample_record(index)
        image=self.image_reader._load_image(record['file_name'])
        mask=self._mask(record['group_index'],record['annotation_index']) if record['positive'] else np.zeros((SIZE,SIZE),np.uint8)
        inputs,target,_=crop_roi(image,record['box'],mask,self.context,self.size)
        rotation=record['rotation']
        if rotation:inputs=np.rot90(inputs,rotation,axes=(-2,-1));target=np.rot90(target,rotation,axes=(-2,-1))
        if record['flip_x']:inputs=inputs[...,::-1];target=target[...,::-1]
        if record['flip_y']:inputs=inputs[...,::-1,:];target=target[...,::-1,:]
        inputs=np.ascontiguousarray(inputs)
        inputs[0]=np.clip(inputs[0]**record['gamma']*record['gain'],0,1)
        return torch.from_numpy(inputs),torch.from_numpy(np.ascontiguousarray(target))
