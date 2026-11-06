"""Real-only fixed-split native crops for semantic/instance U-Net comparisons.

Original COCO masks stay authoritative. Annotation groups (alternative raters)
are separate records, never merged. Overlapping instance pixels within one
group have deterministic smaller-instance ownership because an integer label
image can assign only one ID per pixel; cropped partial instances are retained.
"""
from collections import defaultdict
from functools import lru_cache
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil

import cv2
import numpy as np
from pycocotools import mask as mu
from scipy.interpolate import UnivariateSpline
import torch
from torch.utils.data import Dataset

SIZE = 2048
EXPECTED = dict(
    annotations='5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1',
    train_split='aa0733424815731c5c108ada91e45525eb17ca3a42b377bf19e3089d86aede7e',
    val_split='54c6a9aac5a3b70b3ecc6721109a8b0fe8ec020e90cb56e3848fccc92eadc61f',
)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def annotation_rle(annotation):
    """Use the competition COCO polygon/RLE rasterization, without rescaling."""
    segmentation = annotation['segmentation']
    if isinstance(segmentation, list):
        if not segmentation:
            raise ValueError('Empty polygon annotation')
        encoded = mu.merge(mu.frPyObjects(segmentation, SIZE, SIZE))
    elif isinstance(segmentation, dict):
        if list(segmentation.get('size', [])) != [SIZE, SIZE]:
            raise ValueError('Annotation RLE must use native2048 coordinates')
        if isinstance(segmentation['counts'], list):
            encoded = mu.frPyObjects(segmentation, SIZE, SIZE)
        else:
            encoded = dict(segmentation)
            if isinstance(encoded['counts'], str):
                encoded['counts'] = encoded['counts'].encode('ascii')
    else:
        raise ValueError('Expected COCO polygon list or RLE dictionary')
    if list(encoded['size']) != [SIZE, SIZE] or int(mu.area(encoded)) <= 0:
        raise ValueError('Invalid or empty native instance annotation')
    return encoded


def _read_splits(data, train_path, val_path, expected_counts=None):
    """Same complete-rater and coverage checks as reviewed read_splits()."""
    images = {str(row['id']): row for row in data['images']}
    if len(images) != len(data['images']):
        raise ValueError('Duplicate annotation image IDs')
    ids, names = {}, {}
    for role, path in [('train', train_path), ('validation', val_path)]:
        with Path(path).open(newline='', encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        selected = [row['image_id'] for row in rows]
        if not selected or len(selected) != len(set(selected)) or not set(selected) <= set(images):
            raise ValueError('Empty, duplicate, or unknown split annotation IDs')
        if any(row['file_name'] != images[row['image_id']]['file_name'] for row in rows):
            raise ValueError('Split filename differs from annotation metadata')
        filenames = {row['file_name'] for row in rows}
        if any(Path(name).name != name or Path(name).suffix.lower() != '.jpeg' for name in filenames):
            raise ValueError('Expected original JPEG basenames')
        if set(selected) != {sid for sid, row in images.items() if row['file_name'] in filenames}:
            raise ValueError('Every annotator sibling must stay in its physical image split')
        ids[role], names[role] = set(selected), sorted(filenames)
    if ids['train'] & ids['validation'] or set(names['train']) & set(names['validation']):
        raise ValueError('Train/validation leakage')
    if ids['train'] | ids['validation'] != set(images):
        raise ValueError('Split union must cover every original annotation group')
    if expected_counts is not None and tuple(len(names[r]) for r in ('train', 'validation')) != tuple(expected_counts):
        raise ValueError('Expected fixed616 train and91 validation physical images')
    return ids, names


def load_cohort(cfg):
    """Read the original fixed split; only train groups are returned for fitting.

    ``strict=False`` is for explicit small test fixtures. Production defaults
    require the three reviewed file hashes and exactly616/91 physical images.
    Input digests cover metadata; the trainer owns image-pixel fingerprints.
    """
    strict = cfg.get('strict', True)
    input_sha256 = {key: digest(cfg[key]) for key in EXPECTED}
    if strict and input_sha256 != EXPECTED:
        raise ValueError('Use the original reviewed annotations and fixed616/91 split CSVs')
    data = json.loads(Path(cfg['annotations']).read_text(encoding='utf-8'))
    ids, names = _read_splits(data, cfg['train_split'], cfg['val_split'],
                              expected_counts=(616, 91) if strict else None)
    by_id = {str(row['id']): row for row in data['images']}
    for row in by_id.values():
        if row.get('width', SIZE) != SIZE or row.get('height', SIZE) != SIZE:
            raise ValueError('Original image metadata must be2048 square')
    image_root = Path(cfg['images'])
    if not image_root.is_dir():
        raise FileNotFoundError(str(image_root))
    missing = [name for role in names.values() for name in role if not (image_root/name).is_file()]
    if missing:
        raise FileNotFoundError(f'Missing original images ({len(missing)}): {missing[:5]}')
    annotations = defaultdict(list)
    annotation_ids = set()
    for annotation in data['annotations']:
        group = str(annotation['image_id'])
        if group not in by_id:
            raise ValueError('Annotation refers to unknown image group')
        if 'id' in annotation:
            key = str(annotation['id'])
            if key in annotation_ids:
                raise ValueError('Duplicate annotation IDs')
            annotation_ids.add(key)
        annotations[group].append(annotation)
    train_groups = [dict(group_id=sid, file_name=by_id[sid]['file_name'],
                         annotations=list(annotations[sid]), role='train')
                    for sid in sorted(ids['train'])]
    truth = {sid: [annotation_rle(a) for a in annotations[sid]] for sid in sorted(ids['validation'])}
    mapping = {sid: Path(by_id[sid]['file_name']).stem for sid in truth}
    return dict(train_groups=train_groups, validation_names=names['validation'], truth=truth,
                mapping=mapping, input_sha256=input_sha256,
                counts=dict(train_physical_images=len(names['train']),
                            training_annotation_groups=len(train_groups),
                            training_instances=sum(len(g['annotations']) for g in train_groups),
                            validation_physical_images=len(names['validation']),
                            validation_annotation_groups=len(truth),
                            validation_instances=sum(map(len, truth.values()))),
                strict_original_split=bool(strict),
                instance_overlap_ownership='smaller area wins; annotation order resolves exact ties')


@lru_cache(maxsize=1)
def limb_geometry():
    yy, xx = np.ogrid[:SIZE, :SIZE]
    rr = (np.sqrt((xx-1024)**2+(yy-1024)**2)/900).astype(np.float32)
    disk = rr <= 1
    radii = (np.arange(100)+.5)/100
    bins = np.minimum((rr[disk]*100).astype(np.int32), 99)
    return rr, disk, radii, bins


def limb(image):
    """Exact reviewed radial correction; zero outside the solar disk."""
    if image is None or image.shape != (SIZE, SIZE) or image.dtype != np.uint8:
        raise ValueError('Original grayscale uint8 2048-square image required')
    rr, disk, radii, bins = limb_geometry()
    hist = np.bincount(bins*256+image[disk].astype(np.int32), minlength=25600).reshape(100, 256)
    count = hist.sum(axis=1)
    if np.any(count <= 0):
        raise ValueError('Empty radial bin')
    med = np.argmax(hist.cumsum(axis=1)>=((count+1)//2)[:, None], axis=1).astype(float)
    spline = UnivariateSpline(radii, med, k=3, s=400)
    reference = float(np.median(spline(radii)[radii<=.15]))
    gain = np.clip(reference/np.maximum(spline(rr[disk]), 1), .25, 4)
    result = np.zeros_like(image)
    result[disk] = np.clip(image[disk].astype(np.float32)*gain, 0, 255).astype(np.uint8)
    return result


def preprocess_image(image, mode='limb'):
    """Shared training/inference preprocessing, float32 native image in[0,1]."""
    if image is None or image.shape != (SIZE, SIZE) or image.dtype != np.uint8:
        raise ValueError('Original grayscale uint8 2048-square image required')
    if mode not in ('raw', 'limb'):
        raise ValueError('preprocessing must be limb or raw')
    corrected = limb(image) if mode == 'limb' else image
    return corrected.astype(np.float32)/255.


def instance_labels(annotations):
    """Rasterize native instance IDs; smaller GT retains overlapping pixels."""
    encoded = [annotation_rle(annotation) for annotation in annotations]
    areas = [int(mu.area(value)) for value in encoded]
    labels = np.zeros((SIZE, SIZE), np.int32)
    for index in sorted(range(len(encoded)), key=lambda i: (-areas[i], i)):
        labels[mu.decode(encoded[index]).astype(bool)] = index+1
    return labels


class NativeInstanceDataset(Dataset):
    """Deterministic train-group sampling; epoch is part of every sample seed.

    A sample chooses one annotation/rater group uniformly, then a positive
    foreground-centered crop with probability0.7 or a uniform crop otherwise.
    Random crops may include foreground. All partial objects retain their IDs.
    ``training=False`` disables augmentation, not crop selection.
    """
    def __init__(self, groups, images, crop=768, steps_per_epoch=256, batch_size=2,
                 seed=42, training=True, preprocessing='limb', augment=None,
                 positive_fraction=.7, cache_dir=None, source_sha256=None):
        if not groups or not isinstance(crop, int) or not 1<=crop<=SIZE:
            raise ValueError('Nonempty training groups and native-size crop required')
        if int(steps_per_epoch)<=0 or int(batch_size)<=0 or not 0<=positive_fraction<=1:
            raise ValueError('Positive epoch/batch sizes and valid crop fraction required')
        if preprocessing not in ('limb', 'raw'):
            raise ValueError('preprocessing must be limb or raw')
        self.groups = list(groups)
        if len({str(group['group_id']) for group in self.groups}) != len(self.groups):
            raise ValueError('Duplicate dataset annotation groups')
        for group in self.groups:
            if Path(group['file_name']).name != group['file_name']:
                raise ValueError('Image basename required')
            if training and group.get('role', 'train') != 'train':
                raise ValueError('Validation groups cannot enter the training dataset')
        self.images = Path(images)
        self.crop, self.seed, self.epoch = crop, int(seed), 0
        self.length = int(steps_per_epoch)*int(batch_size)
        self.training, self.preprocessing = bool(training), preprocessing
        self.augment = bool(training) if augment is None else bool(augment)
        self.positive_fraction = float(positive_fraction)
        self.source_sha256 = dict(source_sha256 or {})
        self._verified_cache_images = set()
        self._verified_processed_images = set()
        self._cache_source_sha256 = {}
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            if self.cache_dir.resolve() == self.images.resolve():
                raise ValueError('Image cache must differ from the original image directory')
        self._data_source_sha256 = digest(Path(__file__))
        # Bounded per-dataset caches:8 uint8 images (~32MiB) and2 int32 labels
        # (~32MiB), avoiding an entire-Drive image or target cache in RAM.
        self._load_image = lru_cache(maxsize=8)(self._read_image)
        self._load_labels = lru_cache(maxsize=2)(self._read_labels)

    def __len__(self):
        return self.length

    def set_epoch(self, epoch):
        if int(epoch)<0:
            raise ValueError('Epoch must be nonnegative')
        self.epoch = int(epoch)

    def _read_image(self, name):
        source = self.images/name
        if not source.is_file():
            raise FileNotFoundError(str(source))
        selected = source
        if self.cache_dir is not None:
            selected = self.cache_dir/name
            # Verify cached bytes against original on each first read in a new
            # dataset/runtime. No source deletion or Drive cache is performed.
            if name not in self._verified_cache_images:
                expected = self.source_sha256.get(name) or digest(source)
                self._cache_source_sha256[name] = expected
                if not selected.is_file() or digest(selected) != expected:
                    pending = selected.with_name(selected.name+f'.pending.{os.getpid()}')
                    shutil.copyfile(source, pending)
                    if digest(pending) != expected:
                        pending.unlink()
                        raise ValueError('Source image changed after pixel fingerprinting: '+name)
                    pending.replace(selected)
                self._verified_cache_images.add(name)
            return self._processed_image(name, selected)
        image = cv2.imread(str(selected), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (SIZE, SIZE) or image.dtype != np.uint8:
            raise ValueError('Original grayscale uint8 2048-square image required: '+str(selected))
        return limb(image) if self.preprocessing == 'limb' else image

    def _processed_image(self, name, selected):
        """Local uint8 NPY cache avoids repeating radial correction per crop.

        Every new dataset/runtime checks its first cached read against source
        pixels, preprocessing mode, code bytes and the saved NPY SHA256. Local
        cache loss is harmless: regenerate from the original verified JPEG.
        """
        folder = self.cache_dir/'processed'/self.preprocessing
        folder.mkdir(parents=True, exist_ok=True)
        array_path = folder/(name+'.npy')
        meta_path = folder/(name+'.json')
        if name not in self._verified_processed_images:
            expected = dict(source_sha256=self._cache_source_sha256[name],
                            preprocessing=self.preprocessing,
                            code_sha256=self._data_source_sha256, dtype='uint8', shape=[SIZE, SIZE])
            valid = array_path.is_file() and meta_path.is_file()
            if valid:
                try:
                    meta = json.loads(meta_path.read_text())
                    valid = meta['identity'] == expected and meta['array_sha256'] == digest(array_path)
                    if valid:
                        image = np.load(array_path, allow_pickle=False)
                        valid = image.shape == (SIZE, SIZE) and image.dtype == np.uint8
                except (OSError, KeyError, TypeError, ValueError):
                    valid = False
            if not valid:
                image = cv2.imread(str(selected), cv2.IMREAD_GRAYSCALE)
                if image is None or image.shape != (SIZE, SIZE) or image.dtype != np.uint8:
                    raise ValueError('Original grayscale uint8 2048-square image required: '+str(selected))
                image = limb(image) if self.preprocessing == 'limb' else image
                pending = array_path.with_name(array_path.name+f'.pending.{os.getpid()}')
                with pending.open('wb') as stream:
                    np.save(stream, image, allow_pickle=False)
                pending.replace(array_path)
                metadata = dict(identity=expected, array_sha256=digest(array_path))
                pending_meta = meta_path.with_name(meta_path.name+f'.pending.{os.getpid()}')
                pending_meta.write_text(json.dumps(metadata, sort_keys=True)+'\n')
                pending_meta.replace(meta_path)
            self._verified_processed_images.add(name)
            return image
        image = np.load(array_path, allow_pickle=False)
        if image.shape != (SIZE, SIZE) or image.dtype != np.uint8:
            raise ValueError('Corrupted preprocessed native cache: '+str(array_path))
        return image

    def _read_labels(self, index):
        return instance_labels(self.groups[index]['annotations'])

    def sample_record(self, index):
        if not 0<=int(index)<len(self):
            raise IndexError(index)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, int(index)]))
        group_index = int(rng.integers(len(self.groups)))
        labels = self._load_labels(group_index)
        positive = bool(rng.random()<self.positive_fraction and labels.any())
        if positive:
            ids = np.unique(labels)
            instance_id = int(rng.choice(ids[ids>0]))
            ys, xs = np.nonzero(labels==instance_id)
            point = int(rng.integers(len(ys)))
            offset_low, offset_high = self.crop//4, max(self.crop//4+1, (3*self.crop)//4+1)
            y = int(np.clip(int(ys[point])-int(rng.integers(offset_low, offset_high)), 0, SIZE-self.crop))
            x = int(np.clip(int(xs[point])-int(rng.integers(offset_low, offset_high)), 0, SIZE-self.crop))
        else:
            instance_id = None
            y, x = (int(rng.integers(SIZE-self.crop+1)) for _ in range(2))
        rotation = int(rng.integers(4)) if self.augment else 0
        flip_x = bool(rng.integers(2)) if self.augment else False
        flip_y = bool(rng.integers(2)) if self.augment else False
        gain = float(rng.uniform(.9, 1.1)) if self.augment else 1.
        gamma = float(rng.uniform(.9, 1.1)) if self.augment else 1.
        return dict(group_index=group_index, group_id=self.groups[group_index]['group_id'],
                    file_name=self.groups[group_index]['file_name'], y=y, x=x,
                    foreground_centered=positive, centered_instance_id=instance_id,
                    rotation=rotation, flip_x=flip_x, flip_y=flip_y, gain=gain, gamma=gamma)

    def __getitem__(self, index):
        record = self.sample_record(index)
        image = self._load_image(record['file_name'])
        labels = self._load_labels(record['group_index'])
        region = np.s_[record['y']:record['y']+self.crop, record['x']:record['x']+self.crop]
        image = image[region].astype(np.float32)/255.
        labels = labels[region]
        if record['rotation']:
            image, labels = np.rot90(image, record['rotation']), np.rot90(labels, record['rotation'])
        if record['flip_x']:
            image, labels = image[:, ::-1], labels[:, ::-1]
        if record['flip_y']:
            image, labels = image[::-1], labels[::-1]
        image = np.clip(image**record['gamma']*record['gain'], 0., 1.)
        return (torch.from_numpy(np.ascontiguousarray(image[None], dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(labels, dtype=np.int64)))
