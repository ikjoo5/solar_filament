"""Exhaustive, shared GONG/SWEFil/CHASE tile plans for fresh YOLO and U-Net.

Each physical 2048-square image contributes all four 1024-square quadrants once
per epoch. Only their order is shuffled. GONG observer labels rotate by image,
so its four tiles always use one coherent observer and competing masks are not
merged. This module never applies a probabilistic domain sampling budget.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

import swefil_mixed_training as mixed

VERSION = 'fullpass_three_domain_tiles_v1'
SOURCE_SIZE = 2048
TILE_SIZE = 1024
POSITIONS = ((0, 0), (0, 1024), (1024, 0), (1024, 1024))
DOMAIN_ORDER = ('real', 'swefil', 'synthetic')
EXPECTED_IMAGES = dict(real=616, swefil=250, synthetic=616)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def group_for_image(variants, seed, domain, filename, epoch):
    """Round-robin observer choice; independent of worker count or tile order."""
    if isinstance(epoch, bool) or int(epoch) != epoch or epoch < 1:
        raise ValueError('Epoch must be a positive integer')
    choices = sorted(map(str, variants))
    if not choices or len(set(choices)) != len(choices):
        raise ValueError('Observer variants must be nonempty and unique')
    offset = int(_hash([int(seed), domain, filename]), 16) % len(choices)
    return choices[(offset + int(epoch) - 1) % len(choices)]


class FullPassTileDataset(mixed.ThreeDomainCropDataset):
    """Same tensors and loss masks as the old dataset, with exhaustive coverage."""
    def __init__(self, data, train_ids, real_store, synthetic_data, synthetic_store,
                 swefil_data, swefil_store, *, crop=TILE_SIZE, length=None, **kwargs):
        if crop != TILE_SIZE:
            raise ValueError('The full-pass plan requires four native 1024 tiles')
        for key in ('synthetic_ratio', 'swefil_ratio', 'warmup_epochs', 'real_finish_epochs'):
            if kwargs.get(key, 0) != 0:
                raise ValueError('Full-pass data has no sampling/real-only phase: '+key)
            kwargs[key] = 0
        if kwargs.get('include_empty_synthetic', True) is not True:
            raise ValueError('Empty tiles/images must remain in a full-pass epoch')
        kwargs['include_empty_synthetic'] = True
        super().__init__(data, train_ids, real_store, synthetic_data, synthetic_store,
            swefil_data, swefil_store, crop=crop, length=1, **kwargs)
        self.physical_images = [(domain, name) for domain in DOMAIN_ORDER
            for name in self.domains[domain]['files']]
        self.length = len(self.physical_images)*len(POSITIONS)
        if length is not None and length != self.length:
            raise ValueError('Full-pass length is derived from all images, not a crop budget')
        self.epoch = 1
        self._plan_epoch = None
        self._permutation = None

    def group_for_image(self, domain, filename, epoch=None):
        return group_for_image(self.domains[domain]['variants'][filename], self.seed,
            domain, filename, self.epoch if epoch is None else epoch)

    def domain_probabilities(self, epoch=None):
        """Observed coverage proportions for reporting; never used for sampling."""
        total = len(self.physical_images)
        return {name: len(self.domains[name]['files'])/total for name in DOMAIN_ORDER}

    def synthetic_probability(self, epoch=None):
        return self.domain_probabilities(epoch)['synthetic']

    def _row(self, canonical_index, epoch):
        domain, filename = self.physical_images[canonical_index//4]
        tile_id = canonical_index % 4
        top, left = POSITIONS[tile_id]
        return dict(domain=domain, file_name=filename,
            group_id=self.group_for_image(domain, filename, epoch),
            tile_id=tile_id, top=top, left=left,
            physical_key=domain+'/'+filename, canonical_index=canonical_index)

    def _epoch_permutation(self, epoch):
        if isinstance(epoch, bool) or int(epoch) != epoch or epoch < 1:
            raise ValueError('Epoch must be a positive integer')
        if self._plan_epoch != epoch:
            self._permutation = np.random.default_rng(np.random.SeedSequence(
                [self.seed, int(epoch), 0xF011])).permutation(self.length)
            self._plan_epoch = int(epoch)
        return self._permutation

    def sample_record(self, index):
        if isinstance(index, bool) or int(index) != index or not 0 <= index < self.length:
            raise IndexError('Tile index is outside the exhaustive epoch')
        canonical_index = int(self._epoch_permutation(self.epoch)[int(index)])
        row = self._row(canonical_index, self.epoch)
        row['rng'] = np.random.default_rng(np.random.SeedSequence(
            [self.seed, int(self.epoch), canonical_index, 0xA096]))
        return row

    def tile_rows(self, epoch=None, shuffled=True):
        epoch = self.epoch if epoch is None else epoch
        order = self._epoch_permutation(epoch) if shuffled else range(self.length)
        # Validate even the unshuffled path.
        self.group_for_image(*self.physical_images[0], epoch)
        return [self._row(int(index), epoch) for index in order]

    def all_variant_rows(self):
        rows = []
        for index, (domain, filename) in enumerate(self.physical_images):
            for group in sorted(self.domains[domain]['variants'][filename]):
                for tile_id, (top, left) in enumerate(POSITIONS):
                    rows.append(dict(domain=domain, file_name=filename, group_id=group,
                        tile_id=tile_id, top=top, left=left,
                        physical_key=domain+'/'+filename, canonical_index=index*4+tile_id))
        return rows

    def coverage_manifest(self):
        domains = {name: dict(physical_images=len(self.domains[name]['files']),
            tiles_per_epoch=4*len(self.domains[name]['files']),
            annotation_groups=len(self.domains[name]['images']),
            files=list(self.domains[name]['files'])) for name in DOMAIN_ORDER}
        return dict(version=VERSION, source_size=SOURCE_SIZE, tile_size=TILE_SIZE,
            physical_images=len(self.physical_images), tiles_per_epoch=self.length,
            tiles_per_image=4, domains=domains,
            observer_policy='One observer per physical image per epoch; deterministic round-robin; all four tiles share it.',
            blank_tiles='Included; no foreground, size, or brightness filtering.',
            sampling='Exhaustive without replacement; only tile order is shuffled.',
            augmentations='Intensity and D4 only; no rescaling, translation or random crops.')

    def epoch_manifest(self, epoch=None):
        epoch = self.epoch if epoch is None else epoch
        rows = self.tile_rows(epoch)
        return dict(self.coverage_manifest(), epoch=epoch, seed=self.seed,
            domain_tile_counts=dict(Counter(row['domain'] for row in rows)),
            order_sha256=_hash(rows), records=rows)

    def instances(self, row):
        """Yield each clipped instance independently, including disconnected parts."""
        from chase_yolo_training import decode_annotation
        domain = self.domains[row['domain']]
        group = str(row['group_id'])
        if domain['images'][group]['file_name'] != row['file_name']:
            raise ValueError('Tile observer does not belong to this physical image')
        top, left = int(row['top']), int(row['left'])
        if (top, left) not in POSITIONS:
            raise ValueError('Unexpected tile position')
        for ann in domain['annotations'].get(group, []):
            mask = decode_annotation(ann)
            crop = mask[top:top+TILE_SIZE, left:left+TILE_SIZE].copy()
            if crop.any():
                yield crop


def prepare_dataset(cfg, prepared, swefil, image_hashes):
    """Use existing verified images/synthesis without changing the frozen split."""
    original, base = mixed.original, mixed.base
    data = json.loads(Path(cfg['annotations']).read_text())
    base.TRAIN=Path(cfg['train_split']); base.VAL=Path(cfg['val_split']); base.IMAGES=Path(cfg['images'])
    train_ids, names, base.GT = base.prepare_split(data)
    synthetic = json.loads(Path(prepared['annotations_path']).read_text())
    syn_hashes = {row['file_name']: original.digest(Path(prepared['images_dir'])/row['file_name'])
        for row in synthetic['images']}
    cache = Path(cfg['cache_root'])/'fullpass_images'
    real_store = original.VerifiedImageStore(cfg['images'], cache/'gong', image_hashes)
    synthetic_store = original.VerifiedImageStore(prepared['images_dir'], cache/'chase', syn_hashes)
    swefil_store = mixed.VerifiedSWEFilStore(swefil['records'], cache/'swefil')
    dataset = FullPassTileDataset(data, train_ids, real_store, synthetic, synthetic_store,
        mixed.swefil_coco(swefil['records']), swefil_store, seed=cfg['seed'],
        total_epochs=cfg['epochs'], cache_dir=cache/'targets', augment=True)
    actual = {name:len(dataset.domains[name]['files']) for name in DOMAIN_ORDER}
    if actual != EXPECTED_IMAGES:
        raise ValueError('The full-pass image pool changed: '+str(actual))
    if set(dataset.domains['real']['files']) & set(names):
        raise ValueError('Official validation images entered the training pool')
    return dataset, real_store, names


def _store_identity(store, files):
    if hasattr(store, 'hashes'):
        return {name:store.hashes[name] for name in files}
    if hasattr(store, 'stores'):
        return {name:_store_identity(store.stores[name], [name])[name] for name in files}
    # Useful for small in-memory/test stores; production stores have bound hashes.
    return {name:hashlib.sha256(store.load(name).tobytes()).hexdigest() for name in files}


def _write_png(path, image):
    temporary = path.with_name(path.stem+'.pending.png')
    if not cv2.imwrite(str(temporary), image, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise OSError('Failed writing tile: '+str(path))
    temporary.replace(path)


def export_yolo_dataset(dataset, root):
    """Export local YOLO tiles once; retain all observer variants for epoch rotation.

The returned records use one symlink per observer/tile for Ultralytics' implicit
image-to-label mapping. Every physical tile PNG is stored once. The epoch sampler
must select exactly the group returned by ``group_for_image`` for each image.
YOLO's polygon representation fills holes/connects components; added/lost pixel
counts are recorded rather than silently treating it as an exact raster mask.
"""
    from chase_yolo_training import decode_annotation, mask_polygon
    root = Path(root).expanduser().resolve()
    if root.is_relative_to('/content/drive'):
        raise ValueError('YOLO tiles and symlinks must use local Colab storage')
    source = dict(version=VERSION, code=mixed.digest(__file__), seed=dataset.seed,
        polygon_code=mixed.digest(Path(__file__).with_name('chase_yolo_training.py')),
        domains={name:dict(images=dataset.domains[name]['images'],
            annotations=dataset.domains[name]['annotations'],
            source_hashes=_store_identity(dataset.stores[name], dataset.domains[name]['files']))
            for name in DOMAIN_ORDER})
    identity = _hash(source)
    manifest_path, yaml_path = root/'manifest.json', root/'dataset.yaml'
    placeholder_path = root/'unused_validation.txt'
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text())
        if previous.get('identity') != identity:
            raise ValueError('Existing YOLO tile cache belongs to different sources')
        records = previous.get('records', [])
        physical = {r['image']:r for r in previous.get('physical_tiles', [])}
        expected_rows = {(r['domain'],r['file_name'],r['group_id'],r['tile_id']) for r in dataset.all_variant_rows()}
        actual_rows = {(r['domain'],r['file_name'],r['group_id'],r['tile_id']) for r in records}
        expected_tiles = {(domain,name,tile) for domain,name in dataset.physical_images for tile in range(4)}
        actual_tiles = {(r['domain'],r['file_name'],r['tile_id']) for r in physical.values()}
        valid = (len(records)==len(expected_rows) and actual_rows==expected_rows
            and len(physical)==len(expected_tiles) and actual_tiles==expected_tiles
            and yaml_path.is_file() and mixed.digest(yaml_path)==previous.get('yaml_sha256')
            and placeholder_path.is_file() and mixed.digest(placeholder_path)==previous.get('validation_placeholder_sha256'))
        valid = valid and all(Path(path).is_file() and mixed.digest(path)==r['image_sha256'] for path,r in physical.items())
        valid = valid and all(Path(r['image']).is_symlink() and Path(r['image']).resolve()==Path(r['physical_image'])
            and Path(r['label']).is_file() and mixed.digest(r['label'])==r['label_sha256'] for r in records)
        if valid:
            return yaml_path, previous
    images, labels = root/'images/train', root/'labels/train'
    images.mkdir(parents=True, exist_ok=True); labels.mkdir(parents=True, exist_ok=True)
    exported, physical_tiles, audits = [], [], []
    for image_index, (domain, filename) in enumerate(dataset.physical_images, 1):
        gray = dataset.stores[domain].load(filename)
        if gray.shape != (SOURCE_SIZE, SOURCE_SIZE) or gray.dtype != np.uint8:
            raise ValueError('Expected native uint8 source pixels')
        directory = root/'physical'/domain; directory.mkdir(parents=True, exist_ok=True)
        physical = {}
        for tile_id, (top, left) in enumerate(POSITIONS):
            path = directory/(_hash([domain, filename, tile_id])[:24]+'.png')
            _write_png(path, gray[top:top+TILE_SIZE,left:left+TILE_SIZE])
            physical[tile_id] = path
            physical_tiles.append(dict(domain=domain,file_name=filename,tile_id=tile_id,
                image=str(path),image_sha256=mixed.digest(path)))
        for group in sorted(dataset.domains[domain]['variants'][filename]):
            lines_by_tile = {tile:[] for tile in range(4)}
            for ann in dataset.domains[domain]['annotations'].get(group, []):
                native = decode_annotation(ann)
                for tile_id, (top, left) in enumerate(POSITIONS):
                    mask = native[top:top+TILE_SIZE,left:left+TILE_SIZE]
                    if not mask.any():
                        continue
                    line, report = mask_polygon(mask)
                    lines_by_tile[tile_id].append(line)
                    audits.append(dict(domain=domain,file_name=filename,group_id=group,
                        tile_id=tile_id,annotation_id=ann.get('id'),**report))
            for tile_id, (top, left) in enumerate(POSITIONS):
                stem = _hash([domain,filename,group,tile_id])[:24]
                image_path, label_path = images/(stem+'.png'), labels/(stem+'.txt')
                if image_path.exists() or image_path.is_symlink():
                    image_path.unlink()
                image_path.symlink_to(os.path.relpath(physical[tile_id], images))
                content = '\n'.join(lines_by_tile[tile_id])
                label_path.write_text(content+'\n' if content else '', encoding='utf-8')
                exported.append(dict(domain=domain,file_name=filename,group_id=group,
                    tile_id=tile_id,top=top,left=left,physical_key=domain+'/'+filename,
                    image=str(image_path),label=str(label_path),physical_image=str(physical[tile_id]),
                    image_sha256=mixed.digest(physical[tile_id]),label_sha256=mixed.digest(label_path),
                    instances=len(lines_by_tile[tile_id])))
        if image_index%25==0 or image_index==len(dataset.physical_images):
            print(f'YOLO full-pass tiles: {image_index}/{len(dataset.physical_images)} physical images', flush=True)
    # Built-in YOLO validation is disabled. Final/full-field validation uses the
    # frozen official development set through the project's PQ evaluator.
    # Ultralytics can construct its validator even when validation is disabled;
    # one unused image avoids rescanning every observer/tile a second time.
    placeholder_path.write_text(exported[0]['image']+'\n', encoding='utf-8')
    yaml_path.write_text('path: '+json.dumps(str(root))+'\ntrain: images/train\nval: unused_validation.txt\nnames:\n  0: filament\n', encoding='utf-8')
    manifest = dict(identity=identity, seed=dataset.seed, coverage=dataset.coverage_manifest(),
        records=exported, physical_tiles=physical_tiles,
        variants={name:dataset.domains[name]['variants'] for name in DOMAIN_ORDER},
        polygon_audit=dict(instances=len(audits), blank_variant_tiles=sum(not r['instances'] for r in exported),
            added_pixels=sum(r['added_pixels'] for r in audits),lost_pixels=sum(r['lost_pixels'] for r in audits),
            multipart_instances=sum(r['contours']>1 for r in audits),
            minimum_representation_iou=min((r['representation_iou'] for r in audits),default=1.),
            records=audits),
        validation='One unused training-image placeholder in YAML only; val=False. No validation images are exported.',
        yaml_sha256=mixed.digest(yaml_path), validation_placeholder_sha256=mixed.digest(placeholder_path))
    mixed.atomic_json(manifest_path, manifest)
    return yaml_path, manifest
