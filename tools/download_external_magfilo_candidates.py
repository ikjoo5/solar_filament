"""Acquire only filename-disjoint MAGFiLO candidates; never claim pixel independence.

Public labels are checksum-verified but never decoded/scored. Downloaded original
GONG JPEGs need a separate comparison to all competition pixels on the user's Drive.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

from colab_inventory_external_magfilo import load_verified_annotations, physical_key

MAX_IMAGE_BYTES = 16 * 1024 * 1024
NAME = re.compile(r'\d{14}[A-Za-z]h\.jpe?g', re.I)


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def official_identifiers(manifest, public_ids):
    """Require complete named splits; reject duplicate/unknown observations."""
    if not isinstance(manifest, dict) or set(manifest) != {'train_images', 'test_images'}:
        raise ValueError('Expected official train_images and test_images filename lists')
    splits = {}
    for split, names in manifest.items():
        if not isinstance(names, list) or not names or any(not isinstance(n, str) or not NAME.fullmatch(n) for n in names):
            raise ValueError(f'Nonempty canonical official filename list required: {split}')
        keys = [physical_key(n) for n in names]
        if len(set(keys)) != len(keys):
            raise ValueError(f'Duplicate official physical identifiers: {split}')
        if set(keys) - public_ids:
            raise ValueError(f'Official filenames absent from pinned public release: {split}')
        splits[split] = set(keys)
    if splits['train_images'] & splits['test_images']:
        raise ValueError('Official train/test physical identifiers overlap')
    return splits


def checked_url(url, key):
    parsed = urllib.parse.urlsplit(url)
    expected = re.compile(r'/HA/hag/\d{6}/\d{8}/\d{14}[A-Za-z]h\.jpg')
    if (parsed.scheme != 'https' or parsed.hostname != 'gong2.nso.edu'
            or parsed.netloc != 'gong2.nso.edu' or parsed.query or parsed.fragment
            or not expected.fullmatch(parsed.path) or physical_key(parsed.path) != key):
        raise ValueError(f'Unexpected public GONG image URL: {url}')
    if parsed.path.split('/')[3] != key[:6] or parsed.path.split('/')[4] != key[:8]:
        raise ValueError('GONG URL date does not match observation identifier')
    return url


def candidate_records(data, manifest):
    public_ids = {physical_key(row['file_name']) for row in data['images']}
    splits = official_identifiers(manifest, public_ids)
    official = set.union(*splits.values())
    grouped = defaultdict(list)
    for row in data['images']:
        key = physical_key(row['file_name'])
        if key not in official:
            grouped[key].append(row)
    records = []
    for key, rows in sorted(grouped.items()):
        names = {row['file_name'] for row in rows}
        urls = {row['url'] for row in rows}
        if len(names) != 1 or len(urls) != 1:
            raise ValueError(f'Conflicting public source names/URLs for {key}')
        name, url = next(iter(names)), next(iter(urls))
        if not NAME.fullmatch(name) or any(row.get('height') != 2048 or row.get('width') != 2048 for row in rows):
            raise ValueError(f'Unexpected public image name/dimensions for {key}')
        records.append(dict(physical_identifier=key, file_name=name,
                            source_url=checked_url(url, key),
                            annotation_image_ids=sorted(str(row['id']) for row in rows)))
    return records, {key: len(value) for key, value in splits.items()}


def fingerprint(raw):
    """Decode rather than trust HTTP status or JPEG suffix; require native dimensions."""
    import cv2
    import numpy as np
    if not 0 < len(raw) <= MAX_IMAGE_BYTES:
        raise ValueError('Image byte length outside bounded range')
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != (2048, 2048) or image.dtype != np.uint8:
        raise ValueError('Expected decodable uint8 2048x2048 image')
    return dict(bytes=len(raw), file_sha256=sha256(raw), pixel_sha256=sha256(image.tobytes()),
                shape=[2048, 2048], dtype='uint8', pixel_decoder=f'opencv-{cv2.__version__}-IMREAD_GRAYSCALE')


class RestrictedRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        checked_url(newurl, physical_key(req.full_url))
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_bytes(record):
    opener = urllib.request.build_opener(RestrictedRedirect())
    request = urllib.request.Request(record['source_url'], headers={'User-Agent': 'SolarFilamentEvaluation/1.0'})
    with opener.open(request, timeout=20) as response:
        checked_url(response.url, record['physical_identifier'])
        if response.status != 200:
            raise ValueError(f'Expected HTTP200; received {response.status}')
        return response.read(MAX_IMAGE_BYTES + 1)


def acquire_one(record, folder, previous=None, fetch=download_bytes, retries=2):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / record['file_name']
    if target.exists() and previous:
        info = fingerprint(target.read_bytes())
        if (info['file_sha256'] != previous.get('file_sha256')
                or info['pixel_sha256'] != previous.get('pixel_sha256')
                or record['source_url'] != previous.get('source_url')):
            raise ValueError(f'Existing image changed; preserved: {target}')
        return dict(record, **info, path=str(target.resolve()), acquisition='verified_cache')
    raw = None
    for attempt in range(retries + 1):
        try:
            raw = fetch(record)
            info = fingerprint(raw)
            break
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            if attempt == retries:
                raise
            time.sleep(min(attempt + 1, 2))
    # Existing images without provenance are compared to a new successful fetch.
    if target.exists():
        if target.read_bytes() != raw:
            raise ValueError(f'Existing file differs from public source; preserved: {target}')
        return dict(record, **info, path=str(target.resolve()), acquisition='cache_remote_reverified')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=folder, prefix=target.name+'.', suffix='.pending', delete=False) as stream:
            stream.write(raw)
            temporary = Path(stream.name)
        try:
            os.link(temporary, target)  # Atomic creation without replacing an existing image.
        except FileExistsError:
            if target.read_bytes() != raw:
                raise ValueError(f'Concurrent different image appeared; preserved: {target}')
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return dict(record, **info, path=str(target.resolve()), acquisition='downloaded')


def write_report(path, report):
    pending = path.with_suffix('.json.pending')
    pending.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    pending.replace(path)


def run(annotations, official_manifest, output, workers=4):
    if not 1 <= workers <= 4:
        raise ValueError('Download workers must be between1 and4')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data, public_provenance = load_verified_annotations(annotations)
    manifest_raw = Path(official_manifest).read_bytes()
    manifest = json.loads(manifest_raw)
    candidates, counts = candidate_records(data, manifest)
    report_path = output/'acquisition.json'
    previous = json.loads(report_path.read_text()) if report_path.is_file() else {}
    if previous and (previous.get('official_manifest_sha256') != sha256(manifest_raw)
                     or previous.get('public_provenance', {}).get('sha256') != public_provenance['sha256']):
        raise ValueError('Existing acquisition belongs to different inputs; use a new output folder')
    cache = {row['physical_identifier']: row for row in previous.get('records', [])}
    report = dict(data_role='external_evaluation_candidates_only', official_counts=counts,
        official_manifest_sha256=sha256(manifest_raw), public_provenance=public_provenance,
        candidate_physical_images=len(candidates), downloaded_or_verified_physical_images=0,
        candidate_annotation_groups=sum(len(row['annotation_image_ids']) for row in candidates),
        actual_eligible_external_N=None, status='acquiring_filename_disjoint_candidates',
        records=[], errors=[], file_duplicate_groups=[], decoded_pixel_duplicate_groups=[],
        training_performed=False, masks_decoded=False, scoring_performed=False,
        official_images_downloaded=0,
        next_step='Compare these file and decoded-pixel hashes with every official train/test image on Drive before scoring; filename exclusion alone does not establish independence.')
    write_report(report_path, report)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(acquire_one, row, output/'images', cache.get(row['physical_identifier'])): row for row in candidates}
        for done in as_completed(pending):
            row = pending[done]
            try:
                result = done.result()
                report['records'].append(result)
                print(f"[{len(report['records']) + len(report['errors'])}/{len(candidates)}] {row['file_name']} {result['acquisition']}", flush=True)
            except Exception as error:
                report['errors'].append(dict(row, error_type=type(error).__name__, error=str(error)))
                print(f"ERROR {row['file_name']}: {type(error).__name__}: {error}", flush=True)
            report['records'].sort(key=lambda r: r['physical_identifier'])
            report['downloaded_or_verified_physical_images'] = len(report['records'])
            write_report(report_path, report)
    for hash_name, result_name in [('file_sha256', 'file_duplicate_groups'), ('pixel_sha256', 'decoded_pixel_duplicate_groups')]:
        groups = defaultdict(list)
        for row in report['records']:
            groups[row[hash_name]].append(row['physical_identifier'])
        report[result_name] = [values for values in groups.values() if len(values) > 1]
    report['total_image_bytes'] = sum(row['bytes'] for row in report['records'])
    report['status'] = 'acquired_pending_official_file_pixel_overlap_audit' if not report['errors'] else 'partial_acquisition_pending_retry'
    write_report(report_path, report)
    print(json.dumps({key: value for key, value in report.items() if key not in {'records', 'errors', 'public_provenance'}}, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--official-manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    result = run(args.annotations, args.official_manifest, args.output, args.workers)
    raise SystemExit(1 if result['errors'] else 0)
