"""Prediction-only DETR routing of the frozen best84 native U-Net tiles.

The whole native image is limb corrected once. Every original full-grid tile
is forwarded once with batch one and is shared by the reference and all three
predeclared routing arms. Routing changes Hann blending, not input pixel scale.
Primary arms fill unobserved pixels from the same view's full-grid prediction.
This is a quality comparison, not a measured inference-cost reduction.
"""
from __future__ import annotations

from itertools import combinations, product
import math
import time

import numpy as np
import torch

import submission_inference as frozen


ARM_SPECS = {
    'detr70_c08': dict(epoch='70', confidence=.8),
    'detr80_c06': dict(epoch='80', confidence=.6),
    'detr80_c08': dict(epoch='80', confidence=.8),
}
MARGIN = 1.5
TILE = 1024
OVERLAP = .25


def checked_box(box, shape):
    """Return a finite clipped xyxy box, independently of any annotation."""
    height, width = map(int, shape)
    values = np.asarray(box, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError('Expected finite bbox xyxy coordinates')
    x1, y1, x2, y2 = values.tolist()
    if x2 <= x1 or y2 <= y1:
        raise ValueError('A bbox must have positive extent')
    x1, y1 = max(0., x1), max(0., y1)
    x2, y2 = min(float(width), x2), min(float(height), y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError('A bbox must intersect the original image')
    return (x1, y1, x2, y2)


def transform_box(box, axes, shape):
    """Apply the same global image flips as the frozen four-view prediction."""
    x1, y1, x2, y2 = checked_box(box, shape)
    height, width = map(int, shape)
    if any(axis not in (-1, -2) for axis in axes) or len(set(axes)) != len(axes):
        raise ValueError('Only the fixed global horizontal/vertical flips are supported')
    if -1 in axes:
        x1, x2 = width-x2, width-x1
    if -2 in axes:
        y1, y2 = height-y2, height-y1
    return (x1, y1, x2, y2)


def expanded_box(box, shape, margin=MARGIN):
    if not math.isfinite(margin) or margin < 1:
        raise ValueError('Context margin must be finite and at least one')
    values = np.asarray(checked_box(box, shape), dtype=np.float64)
    center = (values[:2]+values[2:])/2
    half = (values[2:]-values[:2])*margin/2
    lo = np.maximum(0, np.floor(center-half))
    hi = np.minimum(np.array([shape[1], shape[0]]), np.ceil(center+half))
    return tuple(map(int, np.r_[lo, hi]))


def _axis_tiles(lo, hi, positions, tile):
    """Fewest existing grid intervals covering an axis, with fixed tie order."""
    center = (lo+hi)/2
    for count in range(1, len(positions)+1):
        possible = []
        for subset in combinations(positions, count):
            if (subset[0] <= lo and subset[-1]+tile >= hi
                    and all(right-left <= tile for left, right in zip(subset, subset[1:]))):
                possible.append((sum((start+tile/2-center)**2 for start in subset), subset))
        if possible:
            return min(possible)[1]
    raise ValueError('Native tile grid cannot cover the requested box')


def select_tiles(boxes, shape=(2048, 2048), tile=TILE, overlap=OVERLAP, margin=MARGIN):
    """Return deduplicated (top,left) tiles, with no resizing or GT input."""
    height, width = map(int, shape)
    ys = frozen.tile_positions(height, tile, overlap)
    xs = frozen.tile_positions(width, tile, overlap)
    selected = set()
    for box in boxes:
        x1, y1, x2, y2 = expanded_box(box, shape, margin)
        chosen_x = _axis_tiles(x1, x2, xs, tile)
        chosen_y = _axis_tiles(y1, y2, ys, tile)
        selected.update(product(chosen_y, chosen_x))
    return sorted(selected)


def arm_boxes(boxes_by_epoch, shape):
    """Apply the three predeclared confidence cutoffs to model proposals."""
    keys = {str(key): value for key, value in boxes_by_epoch.items()}
    if not {'70', '80'} <= set(keys):
        raise ValueError('Both DETR70 and DETR80 proposal lists are required')
    checked = {}
    for epoch in ('70', '80'):
        checked[epoch] = []
        for row in keys[epoch]:
            score = float(row['score'])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('Invalid detector confidence')
            checked[epoch].append((checked_box(row['box'], shape), score))
    return {name: [box for box, score in checked[spec['epoch']] if score >= spec['confidence']]
            for name, spec in ARM_SPECS.items()}


@torch.inference_mode()
def predict_arms(model, image, boxes_by_epoch, device, *, tile=TILE, overlap=OVERLAP,
                 margin=MARGIN, amp_dtype=None, include_selected_only=False, progress=None):
    """Infer full and routed four-view probabilities in a single tile pass.

    ``image`` is already corrected uint8 grayscale. Small images and tile sizes
    are supported here only for independent CPU fixtures. The production entry
    point below fixes original2048 pixels, native1024 and FP16 on a T4.
    Returned predictions map each arm to ``(base, mean_four_views, view_unions)``.
    Optional selected-only arms are explicitly diagnostic: missing pixels are
    zero and therefore must not be described as preserving the old recipe.
    """
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError('Expected preprocessed uint8 grayscale image')
    device = torch.device(device)
    if amp_dtype is not None and device.type != 'cuda':
        raise ValueError('CPU fixtures must explicitly use FP32')
    if amp_dtype not in (None, torch.float16):
        raise ValueError('The T4 comparison uses FP16, or explicit FP32 fixtures')
    if not np.isfinite(image).all():
        raise ValueError('Image must be finite')
    height, width = image.shape
    coordinates = [(y, x) for y in frozen.tile_positions(height, tile, overlap)
                   for x in frozen.tile_positions(width, tile, overlap)]
    boxes = arm_boxes(boxes_by_epoch, image.shape)
    names = ['full', *ARM_SPECS]
    diagnostic_names = [name+'_selected_only' for name in ARM_SPECS] if include_selected_only else []
    all_names = names+diagnostic_names
    state = {name: dict(base=None, total=None, unions={}) for name in all_names}
    union_support = {name: np.zeros(image.shape, bool) for name in ARM_SPECS}
    window = frozen.blend_window(tile)
    diagnostics = dict(tile=tile, overlap=overlap, margin=margin, batch=1,
        native_precision='fp16' if amp_dtype is torch.float16 else 'fp32',
        routing='DETR boxes select existing native tile grid; no image resizing',
        missing_pixels='same-view full-native probabilities in primary arms',
        selected_only_diagnostics=bool(include_selected_only),
        actual_tile_forwards=0, views={},
        inference_cost_note='Full reference is always inferred. Selected tile counts are not realized compute savings.',
        arm_specs=ARM_SPECS)
    model.eval()
    began = time.monotonic()
    for view, axes in frozen.VIEWS:
        transformed = np.ascontiguousarray(np.flip(image, axes)) if axes else image
        selections = {name: set(select_tiles([transform_box(box, axes, image.shape) for box in values],
                       image.shape, tile, overlap, margin)) for name, values in boxes.items()}
        numerators = {name: np.zeros((2, height, width), np.float32) for name in names}
        denominators = {name: np.zeros(image.shape, np.float32) for name in names}
        for number, (top, left) in enumerate(coordinates, 1):
            pixels = frozen.model_input(transformed[top:top+tile, left:left+tile])
            inputs = torch.from_numpy(np.ascontiguousarray(pixels[None])).to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype or torch.float32,
                                enabled=amp_dtype is not None):
                output = model(inputs)
            if tuple(output.shape) != (1, 2, tile, tile):
                raise ValueError('best84 must return one two-head native tile')
            probability = output.float().sigmoid().cpu().numpy()[0]
            if not np.isfinite(probability).all():
                raise FloatingPointError('Nonfinite best84 probabilities')
            region = np.s_[top:top+tile, left:left+tile]
            weighted = probability*window
            for name in names:
                if name == 'full' or (top, left) in selections[name]:
                    numerators[name][:, region[0], region[1]] += weighted
                    denominators[name][region] += window
            diagnostics['actual_tile_forwards'] += 1
            if progress is not None:
                progress(view, number, len(coordinates))
        if np.any(denominators['full'] <= 0):
            raise ValueError('Full native tile grid left uncovered pixels')
        full = numerators['full']/denominators['full']
        view_heads = {'full': full}
        view_diagnostics = {}
        for name in ARM_SPECS:
            denominator = denominators[name]
            support = denominator > 0
            selected = numerators[name]/np.where(support, denominator, np.float32(1.))
            if include_selected_only:
                view_heads[name+'_selected_only'] = selected.copy()
            selected[:, ~support] = full[:, ~support]
            view_heads[name] = selected
            restored_support = np.flip(support, axes) if axes else support
            union_support[name] |= restored_support
            view_diagnostics[name] = dict(selected_tiles=[list(p) for p in sorted(selections[name])],
                selected_tile_count=len(selections[name]),
                observed_pixel_fraction=float(support.mean()),
                full_fallback_pixel_fraction=float((~support).mean()),
                proposal_count=len(boxes[name]))
        diagnostics['views'][view] = view_diagnostics
        for name, heads in view_heads.items():
            if not np.isfinite(heads).all() or np.any(heads < 0) or np.any(heads > 1):
                raise FloatingPointError('Invalid assembled best84 probabilities')
            restored = np.flip(heads, axes) if axes else heads
            target = state[name]
            if target['base'] is None:
                # Copies detach base/total from this view's temporary arrays.
                target['base'], target['total'] = restored.copy(), restored.copy()
            else:
                target['total'] += restored
            target['unions'][view] = frozen.rle(restored[0] >= .5)
        del view_heads, numerators, denominators, full
    predictions = {name: (value['base'], value['total']/np.float32(len(frozen.VIEWS)), value['unions'])
                   for name, value in state.items()}
    diagnostics['arms'] = {name: dict(
        selected_tile_forwards=sum(diagnostics['views'][view][name]['selected_tile_count'] for view, _ in frozen.VIEWS),
        observed_union_pixel_fraction=float(union_support[name].mean())) for name in ARM_SPECS}
    diagnostics['elapsed_seconds'] = time.monotonic()-began
    return dict(predictions=predictions, diagnostics=diagnostics)


def infer_native_variants(model, image_uint8, boxes_by_epoch, config, device):
    """Production entry point; original pixels in, frozen-format rich records out."""
    if image_uint8 is None or image_uint8.shape != (2048, 2048) or image_uint8.dtype != np.uint8:
        raise ValueError('Original uint8 2048-square image is required')
    if config.get('native_batch', 1) != 1:
        raise ValueError('T4 comparison fixes native batch size to one')
    precision = config.get('native_precision', 'fp16')
    if precision not in ('fp16', 'fp32'):
        raise ValueError('Use FP16 on T4 or explicit FP32 CPU fixtures')
    file_name = config['file_name']
    corrected = frozen.limb(image_uint8)
    output = predict_arms(model, corrected, boxes_by_epoch, device,
        amp_dtype=torch.float16 if precision == 'fp16' else None,
        include_selected_only=bool(config.get('include_selected_only', False)),
        progress=config.get('progress'))
    rich = {}
    for arm, (base, average, unions) in output['predictions'].items():
        record = frozen.prediction_record(base, file_name, config.get('image_sha256', ''),
                                          config.get('identity', 'detr-best84'))
        record['tta'] = frozen.prediction_record(average, file_name, config.get('image_sha256', ''),
                                                config.get('identity', 'detr-best84'))
        record['view_unions'] = unions
        rich[arm] = record
    output['diagnostics']['whole_image_limb_calls'] = 1
    return dict(rich=rich, diagnostics=output['diagnostics'])
