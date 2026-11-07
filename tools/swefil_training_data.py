"""Verified SWEFil human annotations for the all-observation native experiment.

The public SWEFil train/test partitions are pooled as external training data.
Exactly one delivered processed GONG JPEG is used per annotated observation;
the raw counterpart is not another sample. No temporal purge is applied.
Validation proximity is reported only as a diagnostic, never a train filter.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import time
import uuid

import cv2
import numpy as np
from pycocotools import mask as mu


REVISION = 'bc818c69a79266ad0fa91a539359485af57967cf'
REPOSITORY = 'antonio-reche/SWEFil'
VARIANT = 'processed'
SIDE = 2048
SCHEMA = 1
IMAGE_INDEX_SHA256 = 'd12b9273c5ad30d083195e0e08cd40a9f499bd605b1ee9d79a468a1939df9bfe'
SOURCE_SPECS = {
    'train.json': {'size': 11014423, 'sha256':
                   '41856b9a30b336b102b2cf09173118c76e2a3ff53431315f306176f292f80ebc'},
    'test.json': {'size': 2212664, 'sha256':
                  '3f56ffb3f63be31911d12135fb9e31e837c2b871466fed4f53e176dd230c5b92'},
}
CATEGORIES = {0: 'QRF', 1: 'IRF', 2: 'ARF', 3: 'SUNSPOT'}
EXPECTED_COUNTS = {'physical_observations': 277, 'annotated_observations': 250,
                   'excluded_unannotated': 27, 'filament_positive_observations': 248,
                   'sunspot_only_observations': 2, 'filament_instances': 1670,
                   'sunspot_instances': 402,
                   'categories': {'0': 1129, '1': 285, '2': 256, '3': 402},
                   'annotated_by_public_partition': {'train': 209, 'test': 41}}
OBSERVATION = re.compile(r'^(\d{14}[A-Za-z]{2})(_raw)?\.jpg$')


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + '.' + uuid.uuid4().hex + '.pending')
    try:
        with pending.open('w', encoding='utf-8') as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def validate_revision(revision):
    if revision != REVISION:
        raise ValueError('This experiment requires the pinned SWEFil revision ' + REVISION)


def _matches(path, spec):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != spec['size']:
        return False
    if 'sha256' in spec:
        return digest(path) == spec['sha256']
    sha = hashlib.sha1(b'blob ' + str(spec['size']).encode() + b'\0')
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest() == spec['git_blob_sha1']


def _quarantine(path):
    path = Path(path)
    if path.exists():
        path.replace(path.with_name(path.name + '.corrupt-' + uuid.uuid4().hex[:12]))


def download_verified(url, path, spec, session):
    """Resume a partial transfer and publish only a verified complete file."""
    import requests
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if _matches(path, spec):
        return path
    _quarantine(path)
    partial = path.with_name(path.name + '.part')
    expected = int(spec['size'])
    if expected <= 0:
        raise ValueError('Invalid expected download size')
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset >= expected:
            if offset == expected and _matches(partial, spec):
                partial.replace(path); return path
            _quarantine(partial); offset = 0
        headers = {'Accept-Encoding': 'identity'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        try:
            with session.get(url, headers=headers, stream=True, timeout=(20, 90)) as response:
                response.raise_for_status()
                if response.status_code == 206:
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                    if not match or tuple(map(int, match.groups())) != (offset, expected - 1, expected):
                        raise ValueError('Unexpected HTTP download range')
                    mode = 'ab' if offset else 'wb'
                elif response.status_code == 200:
                    mode = 'wb'; offset = 0
                else:
                    raise ValueError('Unexpected HTTP download status')
                total = offset
                with partial.open(mode) as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        if not chunk:
                            continue
                        total += len(chunk)
                        if total > expected:
                            raise ValueError('Download exceeds pinned size')
                        handle.write(chunk)
                    handle.flush(); os.fsync(handle.fileno())
                if total != expected:
                    raise requests.ConnectionError('Incomplete download; partial bytes retained')
            if not _matches(partial, spec):
                raise ValueError('Downloaded source checksum mismatch: ' + path.name)
            partial.replace(path)
            return path
        except (requests.RequestException, OSError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    raise AssertionError('Unreachable download state')


def _local_copy(source, destination, spec):
    destination = Path(destination)
    if _matches(destination, spec):
        return destination
    if not _matches(source, spec):
        raise ValueError('Persistent SWEFil cache failed verification: ' + str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    _quarantine(destination)
    pending = destination.with_name(destination.name + '.pending')
    shutil.copyfile(source, pending)
    if not _matches(pending, spec):
        raise OSError('Local SWEFil copy checksum mismatch')
    pending.replace(destination)
    return destination


def select_observations(documents, *, expected_counts=EXPECTED_COUNTS):
    """Choose annotated processed views without merging distinct instances."""
    selected = []; all_observations = set(); category_counts = Counter()
    annotated_by_partition = {}; positives = 0; sunspot_only = 0
    for partition, data in sorted(documents.items()):
        categories = {int(c['id']): c['name'] for c in data['categories']}
        if categories != CATEGORIES:
            raise ValueError('Unexpected SWEFil category mapping')
        images = {}; views = {}; annotations = {}; annotation_ids = set()
        for entry in data['images']:
            image_id = entry['id']; file_name = entry['file_name']
            if image_id in images or PurePosixPath(file_name).parent.as_posix() != 'images':
                raise ValueError('Duplicate image ID or unsafe SWEFil image path')
            match = OBSERVATION.fullmatch(PurePosixPath(file_name).name)
            if not match or entry['width'] != SIDE or entry['height'] != SIDE:
                raise ValueError('Expected dated native 2048 SWEFil observations')
            observation, raw = match.group(1), bool(match.group(2))
            datetime.strptime(observation[:14], '%Y%m%d%H%M%S')
            if (observation, raw) in views:
                raise ValueError('Duplicate physical SWEFil view')
            views[observation, raw] = image_id; images[image_id] = entry
        for ann in data['annotations']:
            if ann['id'] in annotation_ids or ann['image_id'] not in images:
                raise ValueError('Duplicate annotation ID or unknown image reference')
            annotation_ids.add(ann['id'])
            if ann['category_id'] not in CATEGORIES or ann.get('iscrowd', 0) != 0:
                raise ValueError('Unexpected category or crowd label')
            annotations.setdefault(ann['image_id'], []).append(ann)
        observations = sorted({observation for observation, raw in views})
        if all_observations.intersection(observations):
            raise ValueError('Physical observation duplicated across public partitions')
        all_observations.update(observations); accepted = 0
        for observation in observations:
            if (observation, False) not in views or (observation, True) not in views:
                raise ValueError('Expected paired raw and processed SWEFil views')
            image_id = views[observation, False]
            anns = annotations.get(image_id, [])
            # Check that using one view neither drops labels nor doubles them.
            fields = ('category_id', 'segmentation', 'bbox', 'area', 'iscrowd')
            signatures = lambda rows: sorted(fingerprint({k: a.get(k, 0) for k in fields}) for a in rows)
            if signatures(anns) != signatures(annotations.get(views[observation, True], [])):
                raise ValueError('Raw/processed annotation mismatch')
            if not anns:
                continue  # Unknown annotation status is not a background target.
            if len(set(signatures(anns))) != len(anns):
                raise ValueError('Duplicate instance annotations within processed view')
            accepted += 1
            counts = Counter(a['category_id'] for a in anns); category_counts.update(counts)
            positive = any(counts[k] for k in (0, 1, 2))
            positives += int(positive); sunspot_only += int(not positive)
            selected.append(dict(observation_id=observation, file_name=images[image_id]['file_name'],
                                 public_partition=partition, width=SIDE, height=SIDE,
                                 source_image_id=image_id, annotations=anns, spine_valid=False))
        annotated_by_partition[partition] = accepted
    counts = dict(physical_observations=len(all_observations), annotated_observations=len(selected),
                  excluded_unannotated=len(all_observations) - len(selected),
                  filament_positive_observations=positives, sunspot_only_observations=sunspot_only,
                  filament_instances=sum(category_counts[k] for k in (0, 1, 2)),
                  sunspot_instances=category_counts[3],
                  categories={str(k): category_counts[k] for k in CATEGORIES},
                  annotated_by_public_partition=annotated_by_partition)
    if not selected:
        raise ValueError('No annotated SWEFil observations found')
    if expected_counts is not None and counts != expected_counts:
        raise ValueError('Pinned SWEFil source count contract changed: ' + json.dumps(counts))
    return sorted(selected, key=lambda row: row['observation_id']), counts


def canonical_annotation(ann, height=SIDE, width=SIDE):
    """Rasterize all polygons of one COCO object as one instance union."""
    polygons = ann['segmentation']
    if not isinstance(polygons, list) or not polygons:
        raise ValueError('Expected nonempty COCO polygon segmentation')
    for polygon in polygons:
        if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2:
            raise ValueError('Invalid COCO polygon')
        if any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in polygon):
            raise ValueError('Nonfinite polygon coordinates')
        # Polygon vertices may lie on the outer pixel-grid edge.
        if any(x < 0 or x > width for x in polygon[::2]) or any(y < 0 or y > height for y in polygon[1::2]):
            raise ValueError('Polygon lies outside the native image grid')
    rle = mu.merge(mu.frPyObjects(polygons, height, width))
    area = int(mu.area(rle))
    if area <= 0:
        raise ValueError('Empty rasterized instance mask')
    return dict(id=ann['id'], category_id=ann['category_id'],
                segmentation={'size': [height, width], 'counts': rle['counts'].decode('ascii')},
                area=area, bbox=list(map(float, mu.toBbox(rle))),
                source_polygon_count=len(polygons), source_area=ann['area'],
                source_bbox=ann['bbox'])


def validation_proximity(val_names, observations, days=7):
    """UTC timestamp proximity, diagnostic only; it cannot prove independence."""
    external = [(r['observation_id'], datetime.strptime(r['observation_id'][:14], '%Y%m%d%H%M%S'))
                for r in observations]
    if not external:
        raise ValueError('Proximity diagnostics need external observations')
    records = []; near = []; far = []
    for name in sorted(set(val_names or [])):
        stem = Path(name).stem
        try:
            date = datetime.strptime(stem[:14], '%Y%m%d%H%M%S')
        except ValueError as exc:
            raise ValueError('Validation filename has no observation timestamp: ' + name) from exc
        observation, seconds = min(((obs, abs((date - dt).total_seconds())) for obs, dt in external),
                                   key=lambda pair: pair[1])
        group = 'near' if seconds <= days * 86400 else 'far'
        (near if group == 'near' else far).append(name)
        records.append(dict(file_name=name, group=group, nearest_external_observation=observation,
                            absolute_gap_days=seconds / 86400))
    return dict(days=days, near=near, far=far, records=records, training_exclusions=0,
                limitation='Temporal proximity is diagnostic, not proof of distinct solar features; no image-hash comparison.')


def _image_index(root, selected, session):
    path = Path(root) / 'image_index.json'
    if path.is_file():
        index = json.loads(path.read_text())
        if fingerprint(index) == IMAGE_INDEX_SHA256:
            return {row['path']: row for row in index}
        _quarantine(path)
    url = f'https://huggingface.co/api/datasets/{REPOSITORY}/tree/{REVISION}?recursive=true&limit=1000'
    with session.get(url, timeout=(20, 90)) as response:
        response.raise_for_status()
        if 'rel="next"' in response.headers.get('Link', ''):
            raise ValueError('Unexpected paginated pinned SWEFil tree')
        tree = response.json()
    wanted = {row['file_name'] for row in selected}; index = []
    for entry in tree:
        if entry.get('path') in wanted:
            index.append(dict(path=entry['path'], size=entry['size'], sha256=entry['lfs']['oid']))
    index.sort(key=lambda row: row['path'])
    if len(index) != 250 or fingerprint(index) != IMAGE_INDEX_SHA256:
        raise ValueError('Pinned processed SWEFil image index checksum changed')
    atomic_json(path, index)
    return {row['path']: row for row in index}


def prepare_swefil(base, local_cache, val_names=None, *, revision=REVISION):
    """Download/verify all250 annotated processed views and prepare native RLEs."""
    import requests
    validate_revision(revision)
    root = Path(base) / 'external_SWEFil' / REVISION
    local = Path(local_cache); root.mkdir(parents=True, exist_ok=True); local.mkdir(parents=True, exist_ok=True)
    source_records = {}; documents = {}
    with requests.Session() as session:
        for name, spec in SOURCE_SPECS.items():
            url = f'https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{name}'
            path = download_verified(url, root / name, spec, session)
            source_records[name] = dict(path=str(path), sha256=digest(path), size=path.stat().st_size,
                                        url=url, expected=spec)
            documents[Path(name).stem] = json.loads(path.read_text())
        selected, counts = select_observations(documents)
        index = _image_index(root, selected, session)
        records = []
        print('SWEFil: all250 annotated processed observations; 27 unknown-label observations excluded.', flush=True)
        for position, row in enumerate(selected, 1):
            name = row['file_name']; spec = index[name]
            url = f'https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{name}'
            persistent = download_verified(url, root / name, spec, session)
            image_path = _local_copy(persistent, local / name, spec)
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape != (SIDE, SIDE) or image.dtype != np.uint8:
                raise ValueError('Expected decodable native 2048 uint8 SWEFil image: ' + str(image_path))
            record = dict(row, image_path=str(image_path), persistent_image_path=str(persistent),
                          image_sha256=spec['sha256'], image_bytes=spec['size'], source_url=url)
            record['annotations'] = [canonical_annotation(ann) for ann in row['annotations']]
            records.append(record)
            if position % 25 == 0 or position == len(selected):
                print(f'SWEFil images verified {position}/{len(selected)}', flush=True)
    proximity = validation_proximity(val_names, records)
    # Store cache-relative paths: the manifest identity must survive runtime relocation.
    manifest_records = [{k: v for k, v in row.items() if k not in ('image_path', 'persistent_image_path')}
                        for row in records]
    manifest = dict(schema_version=SCHEMA, repository=REPOSITORY, revision=REVISION, variant=VARIANT,
                    source_counts=counts, source_files=source_records, records=manifest_records,
                    image_index_sha256=IMAGE_INDEX_SHA256, training_temporal_filter='none',
                    public_partitions_pooled=True, foreground_categories=[0, 1, 2], sunspot_category=3,
                    spine_valid=False, pixel_processing='Delivered processed JPEG, native2048; no additional preprocessing here.',
                    variant_comparison='Processed is the declared default; raw/processed performance has not been compared.',
                    label_policy='Human polygons; all parts of each annotation unioned as one instance. No pseudo labels.',
                    license='CC-BY-4.0')
    # Paths differ by machine; they are reported separately from the source identity.
    for info in manifest['source_files'].values():
        info.pop('path', None)
    manifest_path = root / 'training_manifest.json'
    atomic_json(manifest_path, manifest)
    diagnostics_path = root / ('validation_proximity_' + fingerprint(proximity)[:16] + '.json')
    atomic_json(diagnostics_path, proximity)
    return dict(records=records, manifest_path=str(manifest_path), manifest_sha256=digest(manifest_path),
                source_counts=counts, validation_proximity=proximity,
                validation_proximity_path=str(diagnostics_path), persistent_root=str(root),
                local_cache=str(local), revision=REVISION, variant=VARIANT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--local-cache', type=Path, default=Path('/content/swefil_' + REVISION[:12]))
    parser.add_argument('--val-names', type=Path, help='JSON list of validation filenames or CSV with file_name column')
    parser.add_argument('--revision', default=REVISION)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args(); names = None
    if args.val_names:
        if args.val_names.suffix.lower() == '.csv':
            import csv
            with args.val_names.open() as handle:
                names = [row['file_name'] for row in csv.DictReader(handle)]
        else:
            names = json.loads(args.val_names.read_text())
            if not isinstance(names, list):
                raise ValueError('--val-names JSON must contain a filename list')
    prepared = prepare_swefil(args.base, args.local_cache, names, revision=args.revision)
    report = {k: v for k, v in prepared.items() if k != 'records'}
    if args.report:
        atomic_json(args.report, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
