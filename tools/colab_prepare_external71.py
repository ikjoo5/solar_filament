"""Colab CPU cell: prepare external evaluation data; no training or scoring."""
from collections import Counter, defaultdict
from datetime import date
import hashlib
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request
import zipfile

import cv2
import numpy as np

BASE = Path('/content/drive/MyDrive/filament-segmentation-2026')
PUBLIC_URL = 'https://dataverse.harvard.edu/api/access/datafile/10407987'
PUBLIC_BYTES = 60602626
PUBLIC_MD5 = 'bfa414567b4dfa2a6b4a9eccedb6aa20'
EXPECTED_IDS = {
    'train': (707, 'a37a8ecd7c3bb1b7c2d7f5bcc18a4c2171b24d6d89da8b9098ccd5e06da824c2'),
    'test': (180, '77652684f3d6dc8b4ad30f31eec330720f91ac0ce692a0c5a49981197d7f1644'),
}
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def key(name):
    match = re.search(r'(\d{14}[a-z]h)', str(name), re.I)
    if not match:
        raise ValueError(f'Unknown observation filename: {name}')
    return match.group(1).lower()


def day(identifier):
    return date(int(identifier[:4]), int(identifier[4:6]), int(identifier[6:8]))


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)+'\n').encode()


def preserve(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise FileExistsError(f'Existing different file preserved: {path}')
        return path
    temporary = path.with_name(path.name+'.pending')
    temporary.write_bytes(raw)
    if path.exists():
        temporary.unlink()
        raise FileExistsError(f'Concurrent file creation; existing file preserved: {path}')
    temporary.replace(path)
    return path


def image_info(raw):
    if not raw:
        raise ValueError('Expected a native 2048 x 2048 image; received empty bytes')
    try:
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    except cv2.error as error:
        raise ValueError('Expected a native 2048 x 2048 decodable image') from error
    if image is None or image.shape != (2048, 2048) or image.dtype != np.uint8:
        raise ValueError('Expected a native 2048 x 2048 uint8 image')
    return dict(file_sha256=sha(raw), pixel_sha256=sha(image.tobytes()), bytes=len(raw))


def validate_archive_url(url):
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != 'https' or parsed.netloc != 'gong2.nso.edu' or parsed.query or parsed.fragment
            or not re.fullmatch(r'/HA/hag/\d{6}/\d{8}/\d{14}[A-Za-z]h\.jpg', parsed.path)):
        raise ValueError(f'Unexpected archive URL: {url}')


class ArchiveRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_archive_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch(url, limit, opener=None, image_request=False):
    if image_request:
        validate_archive_url(url)
    opener = opener or (urllib.request.build_opener(ArchiveRedirect()).open
                        if image_request else urllib.request.urlopen)
    for attempt in range(2):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'SolarFilamentEvaluation/1.0'})
            with opener(request, timeout=35) as response:
                if image_request:
                    validate_archive_url(response.geturl())
                raw = response.read(limit+1)
            if len(raw) > limit:
                raise ValueError('Download exceeds the expected size limit')
            return raw
        except (OSError, TimeoutError):
            if attempt:
                raise


def archive_image(path, url, opener=None):
    """Resume verified downloads; never silently replace a different local image."""
    path = Path(path)
    validate_archive_url(url)
    receipt = path.with_name(path.name+'.source.json')
    if path.is_file() and receipt.is_file():
        raw = path.read_bytes()
        info = image_info(raw)
        saved = json.loads(receipt.read_text())
        if saved.get('url') != url or saved.get('file_sha256') != info['file_sha256']:
            raise ValueError(f'Cached image/source changed; preserved: {path}')
        return info
    raw = fetch(url, 16*1024*1024, opener, image_request=True)
    info = image_info(raw)
    preserve(path, raw)
    preserve(receipt, json_bytes(dict(url=url, **info)))
    return info


def official_inventory(base, expected=EXPECTED_IDS):
    inventory = {}
    for split in ('train', 'test'):
        folder = Path(base)/'MAGFiLO_1.0_Kaggle_2026'/split/f'{split}_images'
        paths = sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
        ids = [key(p.name) for p in paths]
        count, digest = expected[split]
        if len(ids) != count or len(set(ids)) != count or sha('\n'.join(sorted(ids)).encode()) != digest:
            raise ValueError(f'{split}: full image inventory differs from the supplied official manifest')
        inventory[split] = []
        for index, (path, identifier) in enumerate(zip(paths, ids), 1):
            inventory[split].append(dict(file_name=path.name, physical_identifier=identifier,
                                         **image_info(path.read_bytes())))
            if index % 50 == 0 or index == len(paths):
                print(f'Official {split}: {index}/{len(paths)} fingerprinted', flush=True)
    train = {r['physical_identifier'] for r in inventory['train']}
    if train & {r['physical_identifier'] for r in inventory['test']}:
        raise ValueError('Official train/test identifiers overlap')
    return inventory


def assign_splits(records):
    for record in records:
        block = str((date.fromisoformat(record['capture_day'])-date(2000, 1, 1)).days//27)
        record['time_block_27days'] = block
        record['split'] = 'external_confirm' if int(sha(f'42:{block}'.encode())[:16], 16)/2**64 < .3 else 'external_dev'
    confirm = [date.fromisoformat(r['capture_day']) for r in records if r['split'] == 'external_confirm']
    for record in records:
        if record['split'] == 'external_dev' and confirm:
            if min(abs((date.fromisoformat(record['capture_day'])-d).days) for d in confirm) <= 3:
                record['split'] = 'external_embargo'
    return records


def select_records(groups, acquired, inventory, images):
    """Exclude official duplicates and ambiguous duplicate/conflicting external observations."""
    official = inventory['train']+inventory['test']
    files = {r['file_sha256'] for r in official}
    pixels = {r['pixel_sha256'] for r in official}
    dates = {day(r['physical_identifier']) for r in official}
    counts = Counter(r['pixel_sha256'] for r in acquired.values())
    records, excluded = [], []
    for identifier, info in sorted(acquired.items()):
        reason = ('official_file_or_decoded_pixel_duplicate' if info['file_sha256'] in files or info['pixel_sha256'] in pixels
                  else 'duplicate_external_decoded_pixels' if counts[info['pixel_sha256']] > 1 else None)
        if reason:
            excluded.append(dict(physical_identifier=identifier, reason=reason))
            continue
        rows = groups[identifier]
        nearest = min(abs((day(identifier)-d).days) for d in dates)
        records.append(dict(evaluation_key=identifier, source_image_ids=sorted(str(r['id']) for r in rows),
            path=str((images/rows[0]['file_name']).resolve()), file_name=rows[0]['file_name'],
            capture_day=day(identifier).isoformat(), observatory=identifier[-2:],
            nearest_official_day_distance=nearest, time_clean_3days=nearest > 3, **info))
    return assign_splits(records), excluded


def run(base=BASE):
    base = Path(base)
    external = base/'external_MAGFiLO'
    manifests, images = external/'manifests', external/'external71_images'
    annotation_path = manifests/'magfilo_2024_v1.0.json'
    raw = annotation_path.read_bytes() if annotation_path.exists() else fetch(PUBLIC_URL, PUBLIC_BYTES)
    if len(raw) != PUBLIC_BYTES or hashlib.md5(raw).hexdigest() != PUBLIC_MD5:
        raise ValueError('Public annotation size/checksum differs; existing file preserved')
    preserve(annotation_path, raw)
    data = json.loads(raw)
    inventory = official_inventory(base)
    official_ids = {r['physical_identifier'] for rows in inventory.values() for r in rows}
    groups = defaultdict(list)
    for row in data['images']:
        identifier = key(row['file_name'])
        if identifier not in official_ids:
            groups[identifier].append(row)
    if len(groups) != 71:
        raise ValueError(f'Expected 71 nonofficial candidate observations; found {len(groups)}')
    acquired, failures = {}, []
    for index, (identifier, rows) in enumerate(sorted(groups.items()), 1):
        try:
            names, urls = {r['file_name'] for r in rows}, {r['url'] for r in rows}
            if len(names) != 1 or len(urls) != 1:
                raise ValueError('Conflicting versions of the same physical observation')
            if any((r.get('width'), r.get('height')) != (2048, 2048) for r in rows):
                raise ValueError('Annotation native dimensions differ from 2048 square')
            name, url = next(iter(names)), next(iter(urls))
            if Path(name).name != name or key(urllib.parse.urlsplit(url).path) != identifier:
                raise ValueError('Unsafe filename or mismatched archive observation')
            acquired[identifier] = archive_image(images/name, url)
        except (OSError, ValueError) as error:
            failures.append(dict(physical_identifier=identifier, error=str(error)))
        print(f'External {index}/71: cached/downloaded {len(acquired)}, failed {len(failures)}', flush=True)
    records, excluded = select_records(groups, acquired, inventory, images)
    allowed = {sid for r in records for sid in r['source_image_ids']}
    eligible = {k: data[k] for k in ('info', 'licenses', 'categories') if k in data}
    eligible.update(images=[r for r in data['images'] if str(r['id']) in allowed],
                    annotations=[r for r in data['annotations'] if str(r['image_id']) in allowed])
    eligible_raw = json_bytes(eligible)
    eligible_path = preserve(manifests/f'external71_annotations_{sha(eligible_raw)[:12]}.json', eligible_raw)
    summary = dict(candidate_physical_images=71, downloaded_or_verified_images=len(acquired),
        eligible_physical_images=len(records), eligible_annotation_groups=len(allowed),
        excluded_physical_images=len(excluded), acquisition_failures=len(failures), complete_acquisition=not failures,
        split_physical_counts=dict(Counter(r['split'] for r in records)),
        split_time_clean_3day_counts=dict(Counter(r['split'] for r in records if r['time_clean_3days'])),
        official_image_counts={k: len(v) for k, v in inventory.items()})
    plan = dict(schema_version=1, data_role='external_evaluation_only', allow_training=False,
        allow_pseudo_label_training=False, allow_test_time_parameter_updates=False, scores_computed=False,
        annotations=str(eligible_path), annotations_sha256=sha(eligible_raw), images=str(images),
        public_annotation_sha256=sha(raw), summary=summary, records=records, excluded=excluded,
        acquisition_failures=failures, official_fingerprints=inventory,
        split_policy=dict(seed=42, group='27-day blocks across all observatories', confirm_fraction=.3,
            dev_confirm_embargo_days=3, labels_used_for_assignment=False, default_evaluation='external_dev',
            confirmation_scores_not_computed_during_preparation=True))
    plan_raw = json_bytes(plan)
    plan_path = preserve(manifests/f'external71_plan_{sha(plan_raw)[:12]}.json', plan_raw)
    archive = manifests/f'external71_audit_{sha(plan_raw)[:12]}.zip'
    if not archive.exists():
        temporary = archive.with_name(archive.name+'.pending')
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as z:
            z.writestr('external71_plan.json', plan_raw)
        temporary.replace(archive)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if failures:
        print('일부 이미지 확보가 실패했습니다. 같은 셀을 다시 실행하면 성공한 다운로드를 재사용합니다. '
              '이 결과는 부분 준비 상태이며 전체 외부 평가 수가 확정된 것은 아닙니다.', flush=True)
    print('학습·추론·PQ 계산은 실행하지 않았습니다. 준비 기록:\n', plan_path, flush=True)
    return archive


if __name__ == '__main__':
    from google.colab import drive, files
    drive.mount('/content/drive')
    files.download(str(run()))
