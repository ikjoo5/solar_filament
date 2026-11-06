"""Persist/recover raw CHASE synthetic PNGs without importing ML libraries.

The authoritative synthesis metadata determines every expected filename/hash.
This optional cache does not change a training recipe, checkpoint or source image.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import re
import stat
import uuid
import zipfile

SCHEMA = 'chase_shape_raw_png_cache_v1'
BLOCK = 1024 * 1024


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 128 * BLOCK:
        raise ValueError('Missing, oversized or unsafe cache metadata: '+str(path))
    return json.loads(path.read_text(encoding='utf-8'))


def _envelope(path, identity):
    value = _json(path)
    if value.get('identity') != identity or value.get('payload_sha256') != _fingerprint(value.get('payload')):
        raise ValueError('Synthesis metadata checksum/identity differs: '+str(path))
    return value['payload']


def archive_path(output_root):
    return Path(output_root).resolve()/'synthetic_images_cache.zip'


def _metadata(output_root):
    root = Path(output_root).resolve()
    cfg = _json(root/'experiment_config.json')
    prepared = _json(root/'prepared_summary.json')
    identity = prepared.get('identity', '')
    if not isinstance(identity, str) or not re.fullmatch('[0-9a-f]{64}', identity):
        raise ValueError('Invalid synthesis identity')
    if Path(cfg.get('output_root', '')).resolve() != root:
        raise ValueError('Configuration belongs to another output directory')
    cache_raw = Path(cfg['cache_root'])
    if not cache_raw.is_absolute():
        raise ValueError('Configured local cache must be absolute')
    cache = cache_raw.resolve()
    if cache.is_relative_to(root) or root.is_relative_to(cache) or cache.is_relative_to('/content/drive'):
        raise ValueError('Raw PNG cache must be separate from persistent Drive output')
    expected_images = cache/'chase_shape'/identity[:16]/'images'
    images = Path(prepared['images_dir'])
    if not images.is_absolute() or images.resolve() != expected_images:
        raise ValueError('Synthetic images directory differs from the configured local cache')
    for path in (expected_images, expected_images.parent, expected_images.parent.parent):
        if path.is_symlink():
            raise ValueError('Symlink in synthetic cache destination')
    synthesis = root/'synthesis'
    if synthesis.is_symlink():
        raise ValueError('Unsafe synthesis metadata directory')
    manifest_path = synthesis/'manifest.json'
    annotation_path = synthesis/'synthetic_annotations.json'
    for key, expected in [('manifest_path', manifest_path), ('annotations_path', annotation_path)]:
        if Path(prepared[key]).resolve() != expected:
            raise ValueError('Prepared metadata path differs: '+key)
    source = _json(synthesis/'source_identity.json')
    if source.get('identity') != identity or _fingerprint(source.get('settings')) != identity:
        raise ValueError('Source identity checksum differs')
    manifest = _envelope(manifest_path, identity)
    if manifest.get('identity') != identity or manifest.get('settings') != source['settings']:
        raise ValueError('Manifest source settings differ')
    annotation_hash = _digest(annotation_path)
    if annotation_hash != manifest.get('annotations_sha256'):
        raise ValueError('Synthetic annotation checksum differs')
    annotation = _json(annotation_path)
    if annotation.get('info', {}).get('identity') != identity:
        raise ValueError('Synthetic annotation identity differs')
    rows = annotation['images']
    records = manifest['records']
    if not rows or len(rows) != len(records) or len(rows) != manifest.get('synthetic_variants'):
        raise ValueError('Incomplete synthetic cache inventory')
    by_index = {}
    for record in records:
        seed = record.get('seed_components', [])
        index = seed[1] if len(seed) == 3 else None
        checksum = record.get('png_sha256', '')
        if type(index) is not int or not 0 <= index < len(rows) or index in by_index:
            raise ValueError('Invalid or repeated synthesis record index')
        if not isinstance(checksum, str) or not re.fullmatch('[0-9a-f]{64}', checksum):
            raise ValueError('Invalid synthetic image checksum')
        by_index[index] = checksum
    entries = []
    for index, row in enumerate(rows):
        name = row['file_name']
        if name != f'chase_shape_{index:06d}.png':
            raise ValueError('Unsafe or unexpected synthetic image filename')
        entries.append(dict(file_name=name, sha256=by_index[index]))
    cache_manifest = dict(schema=SCHEMA, identity=identity, annotations_sha256=annotation_hash,
                          source_manifest_sha256=_digest(manifest_path), files=entries)
    return root, expected_images, cache_manifest


@contextmanager
def _locked(root):
    path = root/'.synthetic_cache.lock'
    if path.is_symlink():
        raise ValueError('Unsafe synthetic cache lock')
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another process is saving/restoring the synthetic PNG cache') from error
        yield


def _inspect(archive, expected):
    infos = archive.infolist()
    names = [info.filename for info in infos]
    wanted = {'cache_manifest.json'} | {'images/'+row['file_name'] for row in expected['files']}
    if len(names) != len(set(names)) or set(names) != wanted:
        raise ValueError('Unexpected, duplicate or unsafe ZIP members')
    for info in infos:
        if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16) or info.compress_type != zipfile.ZIP_STORED:
            raise ValueError('Unsupported ZIP member type')
        if info.file_size > 64 * BLOCK:
            raise ValueError('Oversized raw-image ZIP member')
    manifest_info = archive.getinfo('cache_manifest.json')
    if manifest_info.file_size > 2 * BLOCK:
        raise ValueError('Oversized cache manifest')
    if json.loads(archive.read(manifest_info)) != expected:
        raise ValueError('Saved ZIP identity/inventory differs from current synthesis metadata')


def _copy_checked(source, destination, expected_hash):
    checksum = hashlib.sha256()
    count = 0
    while block := source.read(BLOCK):
        checksum.update(block)
        count += len(block)
        if count > 64 * BLOCK:
            raise ValueError('Raw synthetic PNG unexpectedly large')
        if destination is not None:
            destination.write(block)
    if checksum.hexdigest() != expected_hash:
        raise ValueError('Synthetic PNG checksum differs; existing committed files preserved')
    return count


def _verify_archive(path, expected):
    if path.is_symlink():
        raise ValueError('Unsafe synthetic ZIP path')
    with zipfile.ZipFile(path) as archive:
        _inspect(archive, expected)
        print(f'Checking saved synthetic ZIP: {len(expected["files"])} PNGs', flush=True)
        for index, row in enumerate(expected['files'], 1):
            with archive.open('images/'+row['file_name']) as source:
                _copy_checked(source, None, row['sha256'])
            if index % 100 == 0 or index == len(expected['files']):
                print(f'Synthetic ZIP verification {index}/{len(expected["files"])}', flush=True)


def backup(output_root):
    """Save verified raw PNGs once; interruption leaves an older ZIP untouched."""
    root, images, expected = _metadata(output_root)
    target = archive_path(root)
    result = dict(archive=str(target), identity=expected['identity'], images=len(expected['files']))
    with _locked(root):
        if target.exists() or target.is_symlink():
            _verify_archive(target, expected)
            print(f'Synthetic PNG backup already verified: {target}', flush=True)
            return dict(result, status='already_backed_up')
        total_bytes = 0
        for row in expected['files']:
            path = images/row['file_name']
            if path.is_symlink() or not path.is_file():
                raise ValueError('Raw synthetic PNG missing/unsafe: '+str(path))
            total_bytes += path.stat().st_size
        print(f'Synthetic PNG backup preflight: {len(expected["files"])} files; '
              f'{total_bytes:,} bytes ({total_bytes / BLOCK:.1f} MiB)', flush=True)
        pending = target.with_name(target.name+'.'+uuid.uuid4().hex+'.pending')
        try:
            with zipfile.ZipFile(pending, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
                archive.writestr('cache_manifest.json', json.dumps(expected, sort_keys=True, separators=(',', ':'), allow_nan=False))
                for index, row in enumerate(expected['files'], 1):
                    path = images/row['file_name']
                    if path.is_symlink() or not path.is_file():
                        raise ValueError('Raw synthetic PNG missing/unsafe: '+str(path))
                    with path.open('rb') as source, archive.open('images/'+row['file_name'], 'w', force_zip64=True) as destination:
                        _copy_checked(source, destination, row['sha256'])
                    if index % 50 == 0 or index == len(expected['files']):
                        print(f'Synthetic PNG backup {index}/{len(expected["files"])}', flush=True)
            # ZIP CRC and each source-bound SHA are checked before publication.
            _verify_archive(pending, expected)
            pending.replace(target)
        finally:
            pending.unlink(missing_ok=True)
    return dict(result, status='backed_up', bytes=target.stat().st_size)


def restore(output_root):
    """Restore only missing/corrupt raw PNGs; a valid local cache needs no ZIP."""
    root = Path(output_root).resolve()
    if not (root/'prepared_summary.json').is_file():
        return dict(status='not_prepared', restored=0, archive=str(archive_path(root)))
    root, images, expected = _metadata(root)
    target = archive_path(root)
    result = dict(archive=str(target), identity=expected['identity'], images=len(expected['files']))
    with _locked(root):
        missing = []
        for index, row in enumerate(expected['files'], 1):
            path = images/row['file_name']
            if path.is_symlink():
                raise ValueError('Unsafe local PNG destination')
            if not path.is_file() or _digest(path) != row['sha256']:
                missing.append(row)
            if index % 100 == 0 or index == len(expected['files']):
                print(f'Local synthetic PNG verification {index}/{len(expected["files"])}', flush=True)
        if not missing:
            return dict(result, status='local_complete', restored=0)
        if target.is_symlink():
            raise ValueError('Unsafe synthetic ZIP path')
        if not target.exists():
            return dict(result, status='archive_missing', restored=0, missing=len(missing))
        with zipfile.ZipFile(target) as archive:
            _inspect(archive, expected)
            images.mkdir(parents=True, exist_ok=True)
            for index, row in enumerate(missing, 1):
                path = images/row['file_name']
                pending = path.with_name(path.name+'.'+uuid.uuid4().hex+'.pending')
                try:
                    with archive.open('images/'+row['file_name']) as source, pending.open('wb') as destination:
                        _copy_checked(source, destination, row['sha256'])
                    pending.replace(path)
                finally:
                    pending.unlink(missing_ok=True)
                if index % 50 == 0 or index == len(missing):
                    print(f'Synthetic PNG restore {index}/{len(missing)}', flush=True)
    return dict(result, status='restored', restored=len(missing))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--action', choices=('backup', 'restore'), required=True)
    args = parser.parse_args()
    print(json.dumps(globals()[args.action](args.root), indent=2), flush=True)


if __name__ == '__main__':
    main()
