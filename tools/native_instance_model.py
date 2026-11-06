"""U-Net heads and conservative instance decoding for fixed-split experiments.

Inputs are one-channel tensors normalized by the caller. The three channels are
foreground, instance boundary, and normalized per-instance distance.
Semantic training uses only foreground; the other channels must not be decoded
as trained predictions in that ablation.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from pycocotools import mask as mask_utils
from scipy import ndimage as ndi
import torch
from torch import nn
from torch.nn import functional as F


ENCODERS = ('resnet18', 'resnet34', 'tu-convnext_tiny')
HEAD_MODES = ('semantic', 'instance')


class NativeInstanceNet(nn.Module):
    """One-channel SMP U-Net with three output logits at input resolution."""

    def __init__(self, encoder='resnet34', encoder_weights='imagenet', head_mode='instance'):
        super().__init__()
        if encoder not in ENCODERS:
            raise ValueError(f'encoder must be one of {ENCODERS}')
        if head_mode not in HEAD_MODES:
            raise ValueError(f'head_mode must be one of {HEAD_MODES}')
        import segmentation_models_pytorch as smp
        self.encoder_name = encoder
        self.head_mode = head_mode
        self.network = smp.Unet(encoder_name=encoder, encoder_weights=encoder_weights,
                                in_channels=1, classes=3, activation=None)

    def forward(self, inputs):
        if inputs.ndim != 4 or inputs.shape[1] != 1:
            raise ValueError('Expected normalized grayscale [B,1,H,W] inputs')
        return self.network(inputs)


def _instance_array(instances):
    array = np.asarray(instances)
    if array.ndim != 2 or not array.size:
        raise ValueError('Instance IDs must be a nonempty [H,W] array')
    if not np.issubdtype(array.dtype, np.integer):
        if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
            raise ValueError('Instance IDs must be integers')
    if np.any(array < -1):
        raise ValueError('IDs are -1 for ignored pixels, 0 for background, positive for instances')
    return array.astype(np.int64, copy=False)


def make_targets(instances_np):
    """Preserve thin foreground and derive outer and touching-instance edges.

    ``boundary`` marks the outer foreground contour and both sides of an
    8-neighbor contact between different positive IDs. This boundary may cover
    a one-pixel filament, but its foreground target is never eroded or masked.
    Ignored IDs (-1) have zero valid mask.
    Each visible instance gets its own padded-ROI EDT normalized to maximum 1.
    """
    labels = _instance_array(instances_np)
    height, width = labels.shape
    foreground = labels > 0
    boundary = foreground & ~ndi.binary_erosion(foreground, structure=np.ones((3, 3), np.uint8), border_value=0)
    for dy, dx in ((1, 0), (0, 1), (1, 1), (1, -1)):
        y0, y1 = max(0, -dy), min(height, height-dy)
        x0, x1 = max(0, -dx), min(width, width-dx)
        first = labels[y0:y1, x0:x1]
        second = labels[y0+dy:y1+dy, x0+dx:x1+dx]
        contact = (first > 0) & (second > 0) & (first != second)
        boundary[y0:y1, x0:x1] |= contact
        boundary[y0+dy:y1+dy, x0+dx:x1+dx] |= contact
    distance = np.zeros(labels.shape, np.float32)
    for instance_id in np.unique(labels[foreground]):
        yy, xx = np.nonzero(labels == instance_id)
        top, bottom = int(yy.min()), int(yy.max())+1
        left, right = int(xx.min()), int(xx.max())+1
        local = labels[top:bottom, left:right] == instance_id
        field = ndi.distance_transform_edt(np.pad(local, 1))[1:-1, 1:-1]
        maximum = float(field.max())
        if maximum <= 0:
            raise AssertionError('A nonempty instance must have positive EDT')
        target = distance[top:bottom, left:right]
        target[local] = (field[local] / maximum).astype(np.float32)
    return dict(foreground=foreground.astype(np.float32), boundary=boundary.astype(np.float32),
                distance=distance, valid=(labels >= 0).astype(np.float32))


def loss_fn(logits, instances_tensor, head_mode='instance'):
    """Foreground Dice+BCE, boundary Dice+BCE, foreground-only EDT SmoothL1.

    Semantic mode has exactly the same network tensor shape but no gradient
    from interface/distance logits. Target construction never erodes foreground.
    """
    if head_mode not in HEAD_MODES:
        raise ValueError(f'head_mode must be one of {HEAD_MODES}')
    if logits.ndim != 4 or logits.shape[1] != 3:
        raise ValueError('Expected three logits in [B,3,H,W]')
    if instances_tensor.ndim != 3 or tuple(instances_tensor.shape) != (logits.shape[0], *logits.shape[2:]):
        raise ValueError('Instance IDs must match logit batch and spatial shape')
    items = [make_targets(row) for row in instances_tensor.detach().cpu().numpy()]
    targets = {name: torch.as_tensor(np.stack([item[name] for item in items]),
                                    device=logits.device, dtype=torch.float32)
               for name in ('foreground', 'boundary', 'distance', 'valid')}
    prediction = logits.float()
    foreground = targets['foreground']
    valid = targets['valid']
    valid_count = valid.sum().clamp_min(1.)
    bce = (F.binary_cross_entropy_with_logits(prediction[:, 0], foreground, reduction='none') * valid).sum() / valid_count
    probabilities = prediction[:, 0].sigmoid() * valid
    dims = (1, 2)
    dice_per_image = 1. - (2.*(probabilities*foreground).sum(dims)+1.) / (
        probabilities.sum(dims)+(foreground*valid).sum(dims)+1.)
    active = (valid.sum(dims) > 0).float()
    dice = (dice_per_image*active).sum() / active.sum().clamp_min(1.)
    # Tensor zeros keep metrics well-defined even for semantic/blank batches.
    boundary_bce = prediction[:, 1].sum()*0.
    boundary_dice = prediction[:, 1].sum()*0.
    distance_loss = prediction[:, 2].sum()*0.
    if head_mode == 'instance':
        interface = targets['boundary']
        positive = (interface*valid).sum()
        negative = ((1.-interface)*valid).sum()
        positive_weight = (negative / positive.clamp_min(1.)).clamp(1., 20.)
        boundary_bce = (F.binary_cross_entropy_with_logits(
            prediction[:, 1], interface, pos_weight=positive_weight, reduction='none')*valid).sum()/valid_count
        boundary_probability = prediction[:, 1].sigmoid()*valid
        boundary_dice_per_image = 1. - (2.*(boundary_probability*interface).sum(dims)+1.) / (
            boundary_probability.sum(dims)+(interface*valid).sum(dims)+1.)
        boundary_dice = (boundary_dice_per_image*active).sum()/active.sum().clamp_min(1.)
        weight = foreground*valid
        distance_loss = (F.smooth_l1_loss(prediction[:, 2].sigmoid(), targets['distance'],
                                         beta=.1, reduction='none')*weight).sum()/weight.sum().clamp_min(1.)
    interface_loss = boundary_bce+boundary_dice
    loss = bce + dice + .25*interface_loss + .25*distance_loss
    metrics = dict(loss=loss.detach(), foreground_bce=bce.detach(), foreground_dice=dice.detach(),
                   boundary_loss=interface_loss.detach(), boundary_bce=boundary_bce.detach(),
                   boundary_dice=boundary_dice.detach(), distance_loss=distance_loss.detach())
    return loss, metrics


def _encode(mask):
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))


def _seeds(mask, distance, boundary, minimum_area):
    """At most one distance maximum per connected low-interface core.

    Multiple distance peaks along the same thin/curved filament are collapsed.
    A predicted interface must separate cores before multiple markers are used.
    """
    cores, count = ndi.label(mask & (boundary < .5), structure=np.ones((3, 3), np.uint8))
    sizes = np.bincount(cores.ravel())
    minimum_core = min(25, max(1, int(minimum_area*.05)))
    maxima = distance >= ndi.maximum_filter(distance, size=5, mode='constant', cval=-1.)
    seeds = np.zeros(mask.shape, np.int32)
    marker = 0
    for label in range(1, count+1):
        if sizes[label] < minimum_core:
            continue
        core = cores == label
        peaks = core & maxima
        permitted = peaks if np.any(peaks) else core
        priority = np.where(permitted, distance*(1.-boundary), -1.)
        marker += 1
        seeds[np.unravel_index(int(priority.argmax()), mask.shape)] = marker
    return seeds, marker


def decode_instances(probability_np, threshold=.5, min_area=300, mode='watershed'):
    """Decode arbitrary-size probabilities into disjoint COCO RLE instances.

    Watershed uses trained interface/distance predictions. For a semantic-only
    model call ``mode='connected'`` because its other heads are not trained.
    No-seed or one-core components are returned intact, preserving thin objects.
    """
    probability = np.asarray(probability_np)
    if probability.ndim != 3 or probability.shape[0] != 3 or not probability.shape[1] or not probability.shape[2]:
        raise ValueError('Expected probability [3,H,W]')
    if not np.isfinite(probability).all() or np.any(probability < 0) or np.any(probability > 1):
        raise ValueError('Probabilities must be finite and within [0,1]')
    if not 0 < float(threshold) < 1 or int(min_area) != min_area or min_area < 1:
        raise ValueError('threshold must be in (0,1); min_area must be a positive integer')
    if mode not in ('connected', 'watershed'):
        raise ValueError('mode must be connected or watershed')
    foreground = probability[0] >= threshold
    labels, count = ndi.label(foreground, structure=np.ones((3, 3), np.uint8))
    sizes = np.bincount(labels.ravel())
    instances = []
    for label, region in enumerate(ndi.find_objects(labels), 1):
        if sizes[label] < min_area:
            continue
        mask = labels[region] == label
        decoded = [mask]
        if mode == 'watershed':
            distance, boundary = probability[2][region], probability[1][region]
            seeds, markers = _seeds(mask, distance, boundary, min_area)
            if markers > 1:
                elevation = np.rint((.55*(1.-distance)+.45*boundary)*254).astype(np.uint8)
                elevation[~mask] = 255
                seeds[~mask] = -1
                partition = ndi.watershed_ift(elevation, seeds, structure=np.ones((3, 3), np.uint8))
                decoded = [(partition == marker) & mask for marker in range(1, markers+1)]
                if not np.array_equal(np.logical_or.reduce(decoded), mask):
                    raise AssertionError('Watershed did not allocate every foreground pixel')
                # A qualifying parent must not disappear or lose pixels solely
                # because its predicted interface creates undersized children.
                # Accept only complete partitions into admissible instances.
                if any(int(child.sum()) < min_area for child in decoded):
                    decoded = [mask]
        for local in decoded:
            if int(local.sum()) < min_area:
                continue
            full = np.zeros(foreground.shape, np.uint8)
            full[region] = local
            instances.append(_encode(full))
    return sorted(instances, key=lambda value: -int(mask_utils.area(value)))
