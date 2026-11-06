"""Pinned AHAS2024–2025 binary supervision for the deployed two-head best84.

Only similarity alignment is applied to delivered pixels. Only explicitly
supplied overlap metadata excludes regions; no morphology-based screening is
performed. Excluded foreground stays foreground and is ignored by the loss.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import time
import zipfile
import zlib

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


SCHEMA = 1
IMAGE_SIDE = 2048
SCREEN_VERSION = 'explicit_overlap_metadata_only_v2'
SOURCE_GRID_VERSION = 'morden_whole_frame_to_mask_grid_bilinear_v1'
SOURCE_GRID_REFERENCE = ('https://github.com/irisaltHu/MORDEN/blob/'
    'e8186480a37395272ba414a0143b2b1232294ed1/segmentation/inference.py#L102')
POLICY = ('Only explicitly supplied overlap-exclusion boxes are ignored by loss. '
          'With no overlap metadata, retain all supplied foreground. No junction, '
          'branch, component-size or morphology heuristic excludes annotations. '
          'No instance IDs are invented by this binary dataset.')
DATE_NAME = re.compile(r'^RSM(202[45]\d{4})T\d{6}_\d{4}_HA\.png$')


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name+'.pending')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    pending.replace(path)


def validate_manifest(value):
    if value.get('schema_version') != SCHEMA or value.get('expected_pairs') != 695:
        raise ValueError('Expected the frozen695-observation AHAS manifest')
    archive = value['archive']
    if archive['url'] != 'https://zenodo.org/api/records/21582245/files/data.zip/content':
        raise ValueError('Unexpected external archive URL')
    if archive['size'] != 1198936482 or archive['md5'] != '498e00c9837cd01d9b3763b86e98d9d0':
        raise ValueError('Unexpected AHAS archive identity')
    pairs = value['pairs']
    if len(pairs) != 695 or len({p['id'] for p in pairs}) != len(pairs):
        raise ValueError('Missing/duplicate physical observations')
    for pair in pairs:
        name = pair['id']+'.png'
        if not DATE_NAME.fullmatch(name):
            raise ValueError('Only dated2024–2025 AHAS observations are allowed')
        for kind in ('image', 'mask'):
            info = pair[kind]
            expected = f"AHAS/train/{'images' if kind == 'image' else 'masks'}/{name}"
            if info['member'] != expected or any(x in ('..', '') for x in PurePosixPath(info['member']).parts):
                raise ValueError('Unexpected or unsafe archive member')
            if not 0 < info['size'] < 20*1024**2 or not 0 <= info['crc32'] <= 0xffffffff:
                raise ValueError('Invalid archive member size/CRC')
    return value


def _archive_valid(path, archive):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != archive['size']:
        return False
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'md5').hexdigest() == archive['md5']


def download_archive(archive, cache_root, session=None):
    """Resume on local disk; accept cached data only after exact size+MD5."""
    import requests
    root = Path(cache_root); root.mkdir(parents=True, exist_ok=True)
    target = root/'ahas_21582245_data.zip'
    pending = target.with_name(target.name+'.part')
    if _archive_valid(target, archive):
        return target
    if target.exists():
        raise ValueError('Cached archive checksum failed; preserve it for inspection: '+str(target))
    client = session or requests.Session()
    expected = int(archive['size'])
    for attempt in range(3):
        offset = pending.stat().st_size if pending.exists() else 0
        if offset > expected:
            raise ValueError('Partial archive is larger than the declared source')
        if offset == expected:
            if not _archive_valid(pending, archive):
                raise ValueError('Complete downloaded archive has the wrong checksum')
            pending.replace(target); return target
        headers = {'Accept-Encoding': 'identity'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        try:
            with client.get(archive['url'], headers=headers, stream=True, timeout=(20, 90)) as response:
                response.raise_for_status()
                if response.status_code == 206:
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                    if not match or tuple(map(int, match.groups())) != (offset, expected-1, expected):
                        raise ValueError('Download range does not match the requested archive remainder')
                    mode = 'ab' if offset else 'wb'
                elif response.status_code == 200:
                    mode = 'wb'; offset = 0  # Server ignored Range: restart safely.
                else:
                    raise ValueError('Unexpected download HTTP status')
                total = offset
                announced = total // (128*1024**2)
                with pending.open(mode) as handle:
                    for chunk in response.iter_content(1024*1024):
                        if not chunk:
                            continue
                        total += len(chunk)
                        if total > expected:
                            raise ValueError('Server sent more bytes than the pinned archive')
                        handle.write(chunk)
                        if total // (128*1024**2) > announced:
                            announced = total // (128*1024**2)
                            print(f'CHASE ZIP 다운로드 {total/1024**2:.0f}/{expected/1024**2:.0f} MiB', flush=True)
                if total != expected:
                    raise requests.ConnectionError('Incomplete archive; retained for resume')
            if not _archive_valid(pending, archive):
                raise ValueError('Downloaded archive MD5 mismatch')
            pending.replace(target); return target
        except (requests.RequestException, OSError):
            if attempt == 2:
                raise
            time.sleep(1+attempt)
    raise AssertionError('Unreachable download state')


def _verified_bytes(path, info):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != info['size']:
        return None
    data = path.read_bytes()
    return data if zlib.crc32(data) & 0xffffffff == info['crc32'] else None


def _decode(data, mask=False, *, source='PNG bytes'):
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    actual = 'undecodable' if image is None else f'shape={image.shape}, dtype={image.dtype}'
    if image is None or image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError(f'Expected grayscale uint8 PNG; {source}: {actual}')
    if mask and image.shape != (IMAGE_SIDE, IMAGE_SIDE):
        raise ValueError(f'Expected published2048 mask grid; {source}: {actual}')
    if not mask and (min(image.shape) < 64 or max(image.shape) > 8192):
        raise ValueError(f'Unexpected CHASE source image dimensions; {source}: {actual}')
    if mask and not np.isin(image, (0, 255)).all():
        raise ValueError(f'Expected binary mask pixels0/255; no implicit thresholding; {source}: {actual}')
    return image


def prepare_chase(root, cache_root, manifest_path):
    """Persist only695 dated image/mask pairs; no MHAS duplicate observations.

    Existing source PNGs are validated against frozen archive CRC/size, then
    decoded. A completed export therefore needs no network access on resume.
    The large archive stays in cache_root, not the persistent Drive root.
    """
    root, cache_root = Path(root), Path(cache_root)
    if cache_root.resolve().is_relative_to(root.resolve()) or cache_root.resolve().is_relative_to(Path('/content/drive')):
        raise ValueError('The large CHASE archive cache must be outside the persistent Drive source folder')
    root.mkdir(parents=True, exist_ok=True)
    manifest = validate_manifest(json.loads(Path(manifest_path).read_text()))
    identity = _identity(manifest)
    identity_path = root/'source_identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text())['manifest_identity'] != identity:
        raise ValueError('Existing CHASE folder belongs to a different source manifest')
    _atomic_json(identity_path, dict(manifest_identity=identity, archive=manifest['archive']))
    todo, records = [], []
    for pair in manifest['pairs']:
        record = dict(id=pair['id'], date=pair['date'], source='CHASE_AHAS_2024_2025',
                      manifest_identity=identity)
        for kind in ('image', 'mask'):
            path = root/('images' if kind == 'image' else 'masks')/(pair['id']+'.png')
            path.parent.mkdir(parents=True, exist_ok=True)
            data = _verified_bytes(path, pair[kind])
            record[kind+'_path'] = str(path)
            if data is None:
                todo.append((path, pair[kind], record, kind))
            else:
                pixels = _decode(data, mask=kind == 'mask', source=str(path))
                record[kind+'_shape'] = list(pixels.shape)
                record[kind+'_sha256'] = hashlib.sha256(data).hexdigest()
        records.append(record)
    if todo:
        print(f'CHASE 원본 복원: 필요한 파일{len(todo)}/1390; 대용량 ZIP은 로컬 캐시에 저장', flush=True)
        archive = download_archive(manifest['archive'], cache_root)
        with zipfile.ZipFile(archive) as source:
            infos = {i.filename: i for i in source.infolist()}
            if len(infos) != len(source.infolist()):
                raise ValueError('Duplicate archive members')
            for number, (path, expected, record, kind) in enumerate(todo, 1):
                info = infos.get(expected['member'])
                if info is None or (info.file_size, info.CRC) != (expected['size'], expected['crc32']):
                    raise ValueError('Archive member identity differs from the pinned manifest')
                data = source.read(info)  # zipfile also verifies CRC.
                pixels = _decode(data, mask=kind == 'mask', source=expected['member'])
                record[kind+'_shape'] = list(pixels.shape)
                pending = path.with_name(path.name+'.pending')
                pending.write_bytes(data); pending.replace(path)
                if number % 100 == 0 or number == len(todo):
                    print(f'CHASE 원본 저장 {number}/{len(todo)}', flush=True)
    for record in records:
        for kind in ('image', 'mask'):
            record.setdefault(kind+'_sha256', sha256(record[kind+'_path']))
    report = dict(schema_version=SCHEMA, manifest_identity=identity, physical_images=len(records),
                  years=[2024, 2025], source='AHAS automatic masks manually screened by dataset authors',
                  excludes_MHAS_duplicates=True, instance_labels_available=False,
                  photometric_preprocessing=False, ambiguity_policy=POLICY,
                  source_grid_policy=SOURCE_GRID_VERSION, source_grid_reference=SOURCE_GRID_REFERENCE,
                  image_to_mask_resize_count=sum(r['image_shape'] != r['mask_shape'] for r in records),
                  records=records)
    report_path = root/'source_report.json'; _atomic_json(report_path, report)
    return records, report_path


def estimate_disk(image):
    """Fit only alignment geometry; the blurred threshold image is not output."""
    image = np.asarray(image)
    if image.ndim != 2 or image.dtype != np.uint8 or min(image.shape) < 64:
        raise ValueError('Disk estimation needs a grayscale uint8 image')
    blurred = cv2.GaussianBlur(image, (0, 0), 3.)
    threshold, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if not 1 <= threshold <= 240:
        raise ValueError('Disk/background separation is unreliable')
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError('No solar disk contour found')
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 20:
        raise ValueError('Insufficient disk boundary')
    (cx, cy), (a, b), angle = cv2.fitEllipse(contour)
    radius = (a+b)/4.
    height, width = image.shape
    circularity = min(a, b)/max(a, b)
    filling = cv2.contourArea(contour)/(np.pi*radius**2)
    if not (.30*min(height, width) < radius < .51*min(height, width)
            and .85 < circularity <= 1 and .85 < filling < 1.1
            and abs(cx-width/2) < .12*width and abs(cy-height/2) < .12*height):
        raise ValueError('Solar disk fit failed geometry checks')
    return dict(cx=float(cx), cy=float(cy), radius=float(radius),
                ellipse_axes=[float(a), float(b)], circularity=float(circularity),
                filling=float(filling), threshold=float(threshold))


def align_and_screen(image, mask, *, manual_boxes=(), target_size=2048, target_radius=900,
                     source_geometry=None):
    """Align on the published mask grid; manual boxes also use that grid."""
    image, mask = np.asarray(image), np.asarray(mask)
    if image.ndim != 2 or image.dtype != np.uint8 or mask.ndim != 2:
        raise ValueError('Expected source grayscale image and 2-D mask')
    if not np.isin(mask, (0, 1, 255)).all():
        raise ValueError('Binary CHASE mask required')
    source_grid = dict(policy=SOURCE_GRID_VERSION, reference=SOURCE_GRID_REFERENCE,
        image_shape=list(image.shape), mask_shape=list(mask.shape),
        resized=image.shape != mask.shape, mask_resized=False)
    if image.shape != mask.shape:
        if source_geometry is not None:
            raise ValueError('Geometry overrides require matching image/mask grids')
        # The authors resize the entire source intensity image to the output
        # mask grid before DenseCRF (inference.py lines102–103, output2048 at143).
        # 2025 sources are2313x2304 while published masks remain2048x2048.
        # Preserve those mask coordinates; do not crop or resize the labels.
        image = cv2.resize(image, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
        source_grid['interpolation'] = 'cv2.INTER_LINEAR (authors default)'
    foreground = mask > 0
    geometry = estimate_disk(image) if source_geometry is None else dict(source_geometry)
    cx, cy, radius = (float(geometry[k]) for k in ('cx', 'cy', 'radius'))
    if not np.isfinite([cx, cy, radius]).all() or radius <= 0 or not 0 < target_radius < target_size/2:
        raise ValueError('Invalid similarity geometry')
    scale = target_radius/radius
    matrix = np.array([[scale, 0, target_size/2-scale*cx], [0, scale, target_size/2-scale*cy]], np.float64)
    boxes = []
    for box in manual_boxes:
        values = [int(box[key]) for key in ('x0', 'y0', 'x1', 'y1')]
        x0, y0, x1, y1 = values
        if not 0 <= x0 < x1 <= image.shape[1] or not 0 <= y0 < y1 <= image.shape[0]:
            raise ValueError('Manual exclusion box outside source image')
        boxes.append(dict(zip(('x0', 'y0', 'x1', 'y1'), values), reason='explicit_overlap_metadata'))
    source_valid = np.ones(image.shape, np.uint8)
    for box in boxes:
        source_valid[box['y0']:box['y1'], box['x0']:box['x1']] = 0
    # INTER_LINEAR performs resampling only; no photometric filtering is added.
    aligned = cv2.warpAffine(image, matrix, (target_size, target_size), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    aligned_fg = cv2.warpAffine(foreground.astype(np.uint8), matrix, (target_size, target_size),
                                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = cv2.warpAffine(source_valid, matrix, (target_size, target_size), flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if not (aligned_fg & valid).any():
        # Explicit metadata may ignore all targets. Retain true background
        # without pretending ignored foreground disappeared.
        usable_foreground = 0
    else:
        usable_foreground = int((aligned_fg & valid).sum())
    report = dict(schema_version=SCHEMA, geometry=geometry, affine=matrix.tolist(),
        source_grid=source_grid,
        target_size=target_size, target_radius=target_radius, source_foreground=int(foreground.sum()),
        aligned_foreground=int(aligned_fg.sum()), usable_foreground=usable_foreground,
        ignored_foreground=int((aligned_fg & (valid == 0)).sum()),
        valid_pixels=int(valid.sum()), exclusion_boxes=boxes, screen_version=SCREEN_VERSION,
        ambiguity_policy=POLICY, photometric_preprocessing=False,
        interpolation='image:bilinear; mask and validity:nearest; outside observed support ignored')
    return aligned, aligned_fg.astype(np.float32), valid.astype(np.float32), report


def load_manual_exclusions(path):
    result = {}
    path = Path(path)
    if not path.exists():
        return result
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if not {'id', 'x0', 'y0', 'x1', 'y1'} <= set(reader.fieldnames or []):
            raise ValueError('Manual exclusions need id,x0,y0,x1,y1 columns')
        for row in reader:
            result.setdefault(row['id'], []).append({k: int(row[k]) for k in ('x0', 'y0', 'x1', 'y1')})
    return result


class ChaseDataset(Dataset):
    """Deterministic crop sampling; shared legacy ImageNet input convention."""

    def __init__(self, records, crop=1024, length=400, seed=42, cache_dir=None,
                 manual_exclusions=None):
        if not records or not 32 <= int(crop) <= IMAGE_SIDE or int(crop) % 32 or int(length) <= 0:
            raise ValueError('Invalid CHASE crop/length or empty records')
        self.records = list(records); self.crop = int(crop); self.length = int(length)
        self.seed = int(seed); self.epoch = 0
        self.cache = Path(cache_dir) if cache_dir is not None else None
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)
        self.manual = dict(manual_exclusions or {})
        if not set(self.manual) <= {r['id'] for r in self.records}:
            raise ValueError('Manual exclusions refer to an unknown CHASE observation')
        self._memory = {}

    def __len__(self):
        return self.length

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def preflight(self):
        """Validate and cache every observation before any optimizer update.

        There is no implicit per-image quality/morphology filtering. A broken
        source or impossible disk geometry raises with the original ID rather
        than silently changing the training pool.
        """
        reports = []
        for number, record in enumerate(self.records, 1):
            try:
                image, foreground, valid = self._aligned(record)
            except (ValueError, OSError) as error:
                raise ValueError(f"CHASE preflight failed for {record['id']}: {error}") from error
            if not valid.any():
                raise ValueError('No observed supervised pixels after explicit exclusions: '+record['id'])
            reports.append(dict(id=record['id'], source_image_sha256=record['image_sha256'],
                source_mask_sha256=record['mask_sha256'], aligned_foreground=int(foreground.sum()),
                usable_foreground=int((foreground & valid).sum()), valid_pixels=int(valid.sum()),
                ignored_foreground=int((foreground & (valid == 0)).sum()),
                explicit_exclusion_count=len(self.manual.get(record['id'], []))))
            if number % 50 == 0 or number == len(self.records):
                print(f'CHASE 원반 정렬 검증 {number}/{len(self.records)}', flush=True)
        report = dict(schema_version=SCHEMA, observations=len(reports),
            records=reports, overlap_exclusion_policy=POLICY,
            photometric_preprocessing=False, retained_all_source_observations=True)
        if self.cache:
            _atomic_json(self.cache/'preflight.json', report)
        return report

    def _aligned(self, record):
        token = _identity(dict(image=record['image_sha256'], mask=record['mask_sha256'],
            policy=SCREEN_VERSION, manual=self.manual.get(record['id'], []),
            source_grid=SOURCE_GRID_VERSION,
            alignment='ellipse_similarity_bilinear_v1', target_radius=900))
        if token in self._memory:
            return self._memory[token]
        path = self.cache/(token+'.npz') if self.cache else None
        report_path = path.with_suffix('.json') if path else None
        if path and path.is_file() and report_path.is_file():
            try:
                report = json.loads(report_path.read_text())
                if report['cache_identity'] != token or report['cache_sha256'] != sha256(path):
                    raise ValueError('Aligned CHASE cache hash mismatch')
                with np.load(path, allow_pickle=False) as value:
                    result = tuple(value[k] for k in ('image', 'foreground', 'valid'))
                if any(v.shape != (IMAGE_SIDE, IMAGE_SIDE) for v in result):
                    raise ValueError('Wrong cached image geometry')
                self._memory = {token: result}; return result
            except (OSError, ValueError, KeyError, EOFError):
                pass  # Rebuild disposable local cache from immutable source.
        for kind in ('image', 'mask'):
            if sha256(record[kind+'_path']) != record[kind+'_sha256']:
                raise ValueError('CHASE source image/mask changed: '+record['id'])
        image = _decode(Path(record['image_path']).read_bytes(), source=record['image_path'])
        mask = _decode(Path(record['mask_path']).read_bytes(), mask=True, source=record['mask_path'])
        image, foreground, valid, report = align_and_screen(image, mask, manual_boxes=self.manual.get(record['id'], []))
        result = (image, foreground.astype(np.uint8), valid.astype(np.uint8))
        if path:
            pending = path.with_name(path.name+'.pending')
            with pending.open('wb') as handle:
                np.savez_compressed(handle, image=result[0], foreground=result[1], valid=result[2])
            pending.replace(path)
            report.update(cache_identity=token, cache_sha256=sha256(path), source_id=record['id'])
            _atomic_json(report_path, report)
        self._memory = {token: result}
        return result

    def aligned(self, record):
        """Public image/mask/validity access for the independent YOLO exporter."""
        return self._aligned(record)

    def sample(self, index, epoch=None, record_index=None):
        epoch = self.epoch if epoch is None else int(epoch)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, int(index)]))
        sampled_index = int(rng.integers(len(self.records)))
        if record_index is not None:
            if (isinstance(record_index, (bool, np.bool_))
                    or not isinstance(record_index, (int, np.integer))
                    or not 0 <= int(record_index) < len(self.records)):
                raise ValueError('Explicit CHASE record_index is outside the source pool')
            sampled_index = int(record_index)
        # Consume the ordinary image-selection draw even for a forced record,
        # preserving the established crop RNG sequence and default behavior.
        record = self.records[sampled_index]
        image, foreground, valid = self._aligned(record)
        positives = np.flatnonzero((foreground > 0) & (valid > 0))
        limit = IMAGE_SIDE-self.crop
        if len(positives) and rng.random() < .75:
            cy, cx = np.unravel_index(int(positives[int(rng.integers(len(positives)))]), foreground.shape)
            top = int(rng.integers(max(0, cy-self.crop+1), min(cy, limit)+1))
            left = int(rng.integers(max(0, cx-self.crop+1), min(cx, limit)+1))
        else:
            top, left = (int(v) for v in rng.integers(0, limit+1, size=2))
        region = np.s_[top:top+self.crop, left:left+self.crop]
        if not valid[region].any():
            eligible = np.flatnonzero(valid > 0)
            if not len(eligible):
                raise ValueError('No valid CHASE loss pixels: '+record['id'])
            cy, cx = np.unravel_index(int(eligible[int(rng.integers(len(eligible)))]), valid.shape)
            top = int(rng.integers(max(0, cy-self.crop+1), min(cy, limit)+1))
            left = int(rng.integers(max(0, cx-self.crop+1), min(cx, limit)+1))
            region = np.s_[top:top+self.crop, left:left+self.crop]
        pixels = np.repeat(image[region][None], 3, axis=0).astype(np.float32)/255.
        pixels = (pixels-np.array([.485, .456, .406], np.float32)[:, None, None])/np.array([.229, .224, .225], np.float32)[:, None, None]
        target = np.stack([foreground[region], np.zeros((self.crop, self.crop), np.uint8)]).astype(np.float32)
        loss_valid = np.stack([valid[region], np.zeros((self.crop, self.crop), np.uint8)]).astype(np.float32)
        return tuple(torch.from_numpy(np.ascontiguousarray(v)) for v in (pixels, target, loss_valid))

    def __getitem__(self, index):
        return self.sample(index)
