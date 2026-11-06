"""Export existing CHASE comparison caches; never load models or infer."""
from pathlib import Path
from datetime import datetime, timezone
import gzip, hashlib, json, uuid, zipfile


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def export_cache(root, limit=24 * 1024**2):
    root = Path(root)
    entries = []

    def add(name, path, expected=None):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError('저장된 캐시가 없습니다: ' + str(path))
        data = path.read_bytes()
        sha = _sha(data)
        if expected is not None and sha != expected:
            raise ValueError('캐시 해시가 다릅니다: ' + str(path))
        entries.append(dict(name=name, path=path, sha256=sha, bytes=len(data)))
        return data

    comparison = json.loads(add('comparison/epoch060_comparison.json', root/'comparison/epoch060_comparison.json'))
    roles = {r['role']: r for r in comparison['results']}
    if len(roles) != 4 or comparison['epoch_per_model'] != 60:
        raise ValueError('60에폭 네 조합 결과가 필요합니다.')
    config = json.loads(add('configs/comparison.json', root/'configs/comparison.json'))
    protocols = {}
    for role, row in roles.items():
        folder = 'comparison/validation/' + row['label']
        if Path(row['label']).name != row['label']:
            raise ValueError('잘못된 비교 이름')
        metadata = json.loads(add(folder+'/identity.json', root/folder/'identity.json'))
        if metadata['identity'] != row['identity'] or _sha(_json(metadata['protocol'])) != row['identity']:
            raise ValueError('비교 identity가 다릅니다: ' + role)
        protocols[role] = metadata['protocol']
        add(folder+'/metrics.json', root/folder/'metrics.json')
        add(folder+'/predictions.csv', root/folder/'predictions.csv', row['csv_sha256'])
    image_hashes = protocols['old_yolo_old_native']['source_images']
    if len(image_hashes) != 91 or any(p['source_images'] != image_hashes for p in protocols.values()):
        raise ValueError('동일한 검증 이미지 91장이 필요합니다.')
    for arm, role in [('old', 'old_yolo_old_native'), ('new', 'old_yolo_new_native')]:
        identity = roles[role]['native_identity']
        if len(identity) != 64 or any(c not in '0123456789abcdef' for c in identity):
            raise ValueError('잘못된 native identity')
        source = root/'native_validation_cache'/identity
        metadata = json.loads(add('native/'+arm+'/identity.json', source/'identity.json'))
        if (metadata['identity'] != identity or _sha(_json(metadata['protocol'])) != identity
                or metadata['protocol']['source_images'] != image_hashes):
            raise ValueError('Native cache identity가 다릅니다: ' + arm)
        for filename, image_sha in sorted(image_hashes.items()):
            if Path(filename).name != filename:
                raise ValueError('잘못된 이미지 파일명')
            stem = Path(filename).stem
            envelope = json.loads(gzip.decompress(add('native/'+arm+'/'+stem+'.json.gz', source/(stem+'.json.gz'))))
            payload = envelope['payload']
            if (envelope.get('schema_version') != 1 or envelope['identity'] != identity
                    or envelope['image_sha256'] != image_sha
                    or _sha(_json(payload)) != envelope['payload_sha256']
                    or payload.get('file_name') != filename):
                raise ValueError('Native cache 내용 검증 실패: ' + filename)
        print('검증 완료:', arm, 'native 91장', flush=True)
    add('yolo/old.json.gz', config['yolo_cache'], protocols['old_yolo_old_native']['yolo_cache_sha256'])
    add('yolo/new.json.gz', root/'comparison/yolo_epoch060.json.gz', protocols['new_yolo_old_native']['yolo_cache_sha256'])
    manifest = dict(schema_version=1, source_root=str(root), images=91,
        input_mode='Existing limb-corrected inference; not raw-image inference',
        purpose='Cloud postprocessing/model-output comparison; no training or inference performed',
        native_identities={arm: roles[role]['native_identity'] for arm, role in
                           [('old', 'old_yolo_old_native'), ('new', 'old_yolo_new_native')]},
        files=[{k: v for k, v in entry.items() if k != 'path'} for entry in entries])
    overhead = len(_json(manifest)) + 2048
    groups, current, size = [], [], overhead
    for entry in entries:
        cost = entry['bytes'] + 2*len(entry['name'].encode()) + 200
        if cost + overhead > limit:
            raise ValueError('단일 캐시가 분할 크기를 초과합니다: ' + entry['name'])
        if current and size + cost > limit:
            groups.append(current); current, size = [], overhead
        current.append(entry); size += cost
    if current:
        groups.append(current)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:6]
    destination = root/'cloud_cache_exports'/stamp
    destination.mkdir(parents=True)
    paths = []
    for number, group in enumerate(groups, 1):
        path = destination/f'chase_model_cache_{stamp}_part{number:02d}.zip'
        pending = path.with_suffix('.pending')
        with zipfile.ZipFile(pending, 'w', compression=zipfile.ZIP_STORED) as archive:
            archive.writestr('export_manifest.json', _json(dict(manifest, part=number, parts=len(groups))))
            for entry in group:
                data = entry['path'].read_bytes()
                if _sha(data) != entry['sha256']:
                    raise ValueError('내보내는 중 캐시가 변경됐습니다: ' + entry['name'])
                archive.writestr(entry['name'], data)
        if pending.stat().st_size > limit:
            pending.unlink()
            raise ValueError('ZIP 크기 제한 초과')
        pending.replace(path); paths.append(path)
        print('내보내기:', path.name, f'({path.stat().st_size/1024**2:.2f} MiB)', flush=True)
    return paths
