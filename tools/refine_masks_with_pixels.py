"""A fixed, small grid of instance-preserving, pixel-guided mask refinements.

Parameters never inspect GT. Ground truth is used only after each complete
prediction has been built. Missing tile coverage retains the original mask.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from pycocotools import mask as mu
from scipy import ndimage

from experiment_mask_postprocess import group_rows, overlap_stats, read_rows
from experiment_view_combinations import encode_roi
from filament_eval import Filament, evaluate


class TilePixels:
    """Read lossless native-coordinate PNG tiles, validating their provenance."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        if self.manifest['schema_version'] != 1:
            raise ValueError('Unsupported pixel manifest version')
        self.shape = tuple(self.manifest['image_size'])
        if len(self.shape) != 2 or any(type(v) is not int or v <= 0 for v in self.shape):
            raise ValueError('Invalid native image shape')
        self.missing = Counter()

    def load(self, stem):
        info = self.manifest['images'][stem]
        image = np.zeros(self.shape, np.uint8)
        covered = np.zeros(self.shape, bool)
        for tile in info['tiles']:
            path = (self.root / tile['path']).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError('Tile path escapes the pixel directory')
            if not path.is_file():
                self.missing[stem] += 1
                continue
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != tile['sha256']:
                raise ValueError(f'Tile checksum differs: {path}')
            pixels = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
            x, y, w, h = [tile[k] for k in ('x', 'y', 'width', 'height')]
            if (any(type(v) is not int for v in (x, y, w, h)) or x < 0 or y < 0
                    or w <= 0 or h <= 0 or x + w > self.shape[1] or y + h > self.shape[0]
                    or pixels is None or pixels.shape != (h, w) or pixels.dtype != np.uint8):
                raise ValueError(f'Invalid tile geometry or pixels: {path}')
            region = np.s_[y:y+h, x:x+w]
            if np.any(covered[region] & (image[region] != pixels)):
                raise ValueError('Overlapping tiles contain different pixel values')
            image[region] = pixels
            covered[region] = True
        return image, covered


def _core(mask, mode):
    if mode == 'erode3':
        return cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if mode == 'relative':
        distance = ndimage.distance_transform_edt(mask)
        return mask & (distance >= max(1., .6 * distance.max()))
    raise ValueError(f'Unknown core mode: {mode}')


def _robust(values):
    median = float(np.median(values))
    sigma = max(3., 1.4826 * float(np.median(np.abs(values - median))))
    return median, sigma


def refine_mask(gray, original, other_original=None, *, method='grabcut', band=3,
                core_mode='relative', prior=1., coverage=None, seed=0):
    """Return one native ROI mask and an explicit success/fallback reason.

Bayes' signed-distance prior is normalized by band width, so its weight has
the same meaning for band3 and band5. All safeguards depend on pixels/masks.
"""
    original = np.asarray(original, bool)
    gray = np.asarray(gray)
    if gray.shape != original.shape or gray.dtype != np.uint8 or gray.ndim != 2:
        raise ValueError('Expected matching grayscale uint8 and binary 2D mask')
    if method not in {'grabcut', 'bayes'} or type(band) is not int or band <= 0:
        raise ValueError('Unknown refinement method or band')
    if not np.isfinite(prior) or prior < 0 or not original.any():
        raise ValueError('Invalid prior or empty original instance')
    other = np.zeros_like(original) if other_original is None else np.asarray(other_original, bool)
    if other.shape != original.shape or np.any(other & original):
        raise ValueError('Original instances must be disjoint and have matching geometry')
    if coverage is not None:
        coverage = np.asarray(coverage, bool)
        if coverage.shape != original.shape:
            raise ValueError('Coverage shape differs')
        if not coverage.all():
            return original.copy(), 'fallback_missing_pixels'
    inside = ndimage.distance_transform_edt(original)
    outside = ndimage.distance_transform_edt(~original)
    allowed = (outside <= band) & ~other
    core = _core(original, core_mode)
    ring = ~original & allowed
    if core.sum() < 6 or ring.sum() < 12 or np.all(allowed):
        return original.copy(), 'fallback_insufficient_samples'
    foreground, fg_sigma = _robust(gray[core].astype(np.float64))
    background, bg_sigma = _robust(gray[ring].astype(np.float64))
    if foreground >= background - 1.:
        return original.copy(), 'fallback_low_dark_contrast'
    if method == 'grabcut':
        labels = np.full(gray.shape, cv2.GC_BGD, np.uint8)
        labels[allowed] = cv2.GC_PR_BGD
        labels[original] = cv2.GC_PR_FGD
        labels[core] = cv2.GC_FGD
        cv2.setRNGSeed(int(seed))
        try:
            cv2.grabCut(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), labels, None,
                        np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64),
                        3, cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            return original.copy(), 'fallback_grabcut_fit'
        changed = (labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)
    else:
        variance = .5 * (fg_sigma**2 + bg_sigma**2)
        pixels = gray.astype(np.float64)
        likelihood = ((pixels - background)**2 - (pixels - foreground)**2) / (2 * variance)
        signed_prior = (inside - outside) / band
        changed = (likelihood + prior * signed_prior > 0) & allowed
        changed[core] = True
    changed &= allowed
    # Preserve all pieces anchored in a core; one input instance stays one row.
    labels, _ = ndimage.label(changed, structure=np.ones((3, 3), bool))
    anchored = np.unique(labels[core & changed])
    changed = np.isin(labels, anchored[anchored != 0])
    area = int(changed.sum())
    ratio = area / int(original.sum())
    union = np.count_nonzero(changed | original)
    iou = np.count_nonzero(changed & original) / union if union else 0.
    if not .5 <= ratio <= 1.5 or iou < .5:
        return original.copy(), 'fallback_excessive_change'
    return changed, 'changed' if np.any(changed != original) else 'unchanged'


def assign_nearest(candidates, originals):
    """Resolve new-pixel collisions by distance; preserve original ownership."""
    if len(candidates) != len(originals):
        raise ValueError('Candidate/original instance count differs')
    if not originals:
        return []
    shape = originals[0].shape
    owner = np.full(shape, -1, np.int32)
    for index, mask in enumerate(originals):
        if mask.shape != shape or np.any(mask & (owner >= 0)):
            raise ValueError('Original masks overlap or have different shapes')
        owner[mask] = index
    output = [np.asarray(c, bool) & ((owner < 0) | (owner == i)) for i, c in enumerate(candidates)]
    count = np.zeros(shape, np.uint16)
    for mask in output:
        count += mask
    disputed = count > 1
    if not disputed.any():
        return output
    best = np.full(shape, np.inf, np.float64)
    winner = np.full(shape, -1, np.int32)
    for index, (mask, original) in enumerate(zip(output, originals)):
        claim = disputed & mask
        if not claim.any():
            continue
        ys, xs = np.nonzero(mask | original)
        y1, y2, x1, x2 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
        region = np.s_[y1:y2, x1:x2]
        distance = ndimage.distance_transform_edt(~original[region])
        better = claim[region] & (distance < best[region])
        best[region][better] = distance[better]
        winner[region][better] = index
    for index, mask in enumerate(output):
        mask[disputed & (winner != index)] = False
    return output


def resolve_rois(candidates, originals, regions, shape):
    """Nearest original ownership without materializing full candidate masks."""
    if not len(candidates) == len(originals) == len(regions):
        raise ValueError('ROI candidate/original/region count differs')
    original_owner = np.full(shape, -1, np.int32)
    for index, (mask, region) in enumerate(zip(originals, regions)):
        if mask.shape != original_owner[region].shape or np.any(mask & (original_owner[region] >= 0)):
            raise ValueError('ROI originals overlap or have inconsistent geometry')
        original_owner[region][mask] = index
    winner = np.full(shape, -1, np.int32)
    best = np.full(shape, np.inf, np.float64)
    for index, (candidate, original, region) in enumerate(zip(candidates, originals, regions)):
        if candidate.shape != original.shape:
            raise ValueError('ROI candidate/original geometry differs')
        permitted = candidate & ((original_owner[region] < 0) | (original_owner[region] == index))
        distance = ndimage.distance_transform_edt(~original)
        better = permitted & (distance < best[region])
        winner[region][better] = index
        best[region][better] = distance[better]
    return [candidate & (winner[region] == index)
            for index, (candidate, region) in enumerate(zip(candidates, regions))]


def settings():
    grid = [dict(name='baseline', method=None)]
    for band in (3, 5, 7):
        for core in ('erode3', 'relative'):
            grid.append(dict(name=f'grabcut_b{band}_{core}', method='grabcut', band=band, core_mode=core))
    for band in (3, 5):
        for prior in (.5, 1., 2.):
            grid.append(dict(name=f'bayes_b{band}_p{prior:g}', method='bayes', band=band,
                             core_mode='relative', prior=prior))
    return grid


def run(args):
    started = time.monotonic()
    cv2.setNumThreads(1)
    store = TilePixels(args.pixels)
    if store.shape != (2048, 2048):
        raise ValueError('Competition refinement requires native 2048 square images')
    expected_val_hash = store.manifest.get('source_sha256', {}).get(Path(args.valsplit).name)
    if expected_val_hash and hashlib.sha256(Path(args.valsplit).read_bytes()).hexdigest() != expected_val_hash:
        raise ValueError('Pixel export and current validation split hashes differ')
    with Path(args.valsplit).open(newline='', encoding='utf-8-sig') as f:
        selection = list(csv.DictReader(f))
    groups = {r['image_id'] for r in selection}
    images = {Path(r['file_name']).stem for r in selection}
    if not images <= set(store.manifest['images']):
        raise ValueError('Pixel manifest omits selected validation images')
    truth = read_rows(Path(args.gtcsv), gt=True, selected=groups)
    original_rows = [r for r in read_rows(Path(args.predcsv)) if r.image_id in images]
    if not truth or {r.group_id for r in truth} != groups:
        raise ValueError('Validation selection and GT groups differ')
    by_image = group_rows(original_rows)
    if overlap_stats(original_rows)['overlap_images']:
        raise ValueError('This instance-preserving experiment expects a disjoint input')
    grid = settings()
    predictions = {s['name']: [] for s in grid}
    statuses = {s['name']: Counter() for s in grid}
    for number, stem in enumerate(sorted(images), 1):
        gray, coverage = store.load(stem)
        members = by_image.get(stem, [])
        originals = [mu.decode(r.rle(2048, 2048)).astype(bool) for r in members]
        union = np.zeros((2048, 2048), bool)
        for mask in originals:
            union |= mask
        for setting in grid:
            if setting['method'] is None:
                predictions[setting['name']].extend(members)
                continue
            candidates, regions, local_originals = [], [], []
            kwargs = {k: v for k, v in setting.items() if k != 'name'}
            for index, (row, original) in enumerate(zip(members, originals)):
                x, y, w, h = map(int, mu.toBbox(row.rle(2048, 2048)))
                padding = setting['band'] + 3
                x1, y1 = max(0, x-padding), max(0, y-padding)
                x2, y2 = min(2048, x+w+padding), min(2048, y+h+padding)
                region = np.s_[y1:y2, x1:x2]
                local, status = refine_mask(gray[region], original[region],
                    union[region] & ~original[region], coverage=coverage[region], seed=index, **kwargs)
                statuses[setting['name']][status] += 1
                candidates.append(local)
                regions.append(region)
                local_originals.append(original[region])
            candidates = resolve_rois(candidates, local_originals, regions, (2048, 2048))
            for row, candidate, region in zip(members, candidates, regions):
                ys, xs = np.nonzero(candidate)
                if len(ys) == 0:
                    raise RuntimeError('A refinement unexpectedly removed an entire instance')
                y, x = int(ys.min()), int(xs.min())
                roi = candidate[y:int(ys.max())+1, x:int(xs.max())+1]
                rle = encode_roi(roi, y + region[0].start, x + region[1].start)
                predictions[setting['name']].append(Filament(row.filament_id, stem, stem,
                    rle['counts'].decode('ascii'), int(mu.area(rle))))
        print(f'Pixel refinement {number}/{len(images)}: {stem}', flush=True)
    output = Path(args.outputdir)
    output.mkdir(parents=True, exist_ok=False)
    reports = []
    for setting in grid:
        rows = predictions[setting['name']]
        score = evaluate(truth, rows, 2048, 2048)
        overlaps = overlap_stats(rows)
        if overlaps['overlap_images']:
            raise RuntimeError('Refined predictions overlap')
        path = output / (setting['name'] + '.csv')
        with path.open('x', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filament_id', 'segmentation_rle'])
            writer.writerows((r.filament_id, r.counts) for r in rows)
        reports.append(dict(**setting, **{k: score[k] for k in ('pq', 'sq', 'rq', 'tp_pairs', 'fp', 'fn')},
            statuses=dict(statuses[setting['name']]), prediction_rows=len(rows),
            extra_matching_pairs=score['tp_pairs']-(len(truth)-score['fn']), **overlaps))
    best = max(reports, key=lambda r: r['pq'])
    report = dict(scope=store.manifest.get('scope', 'User-exported native limb pixels; same development split'),
        physical_images=len(images), annotation_groups=len(groups), gt_instances=len(truth),
        input_prediction_sha256=hashlib.sha256(Path(args.predcsv).read_bytes()).hexdigest(),
        pixel_manifest_sha256=hashlib.sha256((store.root/'manifest.json').read_bytes()).hexdigest(),
        pixel_source_sha256=store.manifest.get('source_sha256', {}),
        gt_free_rules='Fixed 13 settings; no object/image GT selection; preserve instance count; native pixels and mask safeguards only',
        missing_tile_files=dict(store.missing), experiments=reports, best=best,
        target_exceeded_in_this_scope=best['pq'] > .5, elapsed_seconds=time.monotonic()-started)
    (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(baseline=reports[0], best=best, scope=report['scope']), indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pixels', 'predcsv', 'gtcsv', 'valsplit', 'outputdir'):
        parser.add_argument('--'+name, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
