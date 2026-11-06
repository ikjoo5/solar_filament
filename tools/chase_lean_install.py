"""Install the pinned CHASE runtime without changing its training identity."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import uuid
import zipfile

VERSION = '057b2b49a0cdf3e0'
ROOT_NAME = 'chase_only_both_alignment_t4_60ep_fullpass_pngfix_v1'
CACHE_ROOT = '/content/chase_only_both_alignment_60ep_fullpass_cache'


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def settings_for(root):
    root = Path(root)
    if root.name != ROOT_NAME or root.parent.name != 'fixed_split_experiments':
        raise ValueError('Use the existing PNG-fix experiment folder; do not reset this run')
    base = root.parent.parent
    drive = base.parent
    return dict(root=str(root), assets=str(root/'assets'/VERSION),
        images=str(base/'MAGFiLO_1.0_Kaggle_2026/train/train_images'),
        annotations=str(base/'MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'),
        yolo_checkpoint=str(drive/'SolarFilament_LimbOnly_YOLO26_mixed_FullRes/best.pt'),
        native_checkpoint=str(drive/'SolarFilament_NativeUNet_A100/20261028T030808Z_bedf3e21/best.pt'),
        chase_root=str(base/'external_CHASE/ahas_2024_2025'), cache_root=CACHE_ROOT)


def _safe_member(name, launcher):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or str(path) != name:
        raise ValueError('Unsafe runtime member: '+name)
    parts = path.parts
    original = (len(parts) == 3 and parts[0] in ('code', 'assets')
                and parts[1] == VERSION)
    monitor = (name == launcher and len(parts) == 3 and parts[0] == 'launchers'
               and re.fullmatch('[0-9a-f]{16}', parts[1])
               and parts[2] == 'chase_runtime_monitor.py')
    if not (original or monitor):
        raise ValueError('Unexpected runtime member: '+name)


def install_runtime(package_path, root, expected_sha256):
    """Verify all payloads and existing files before writing anything.

    Checkpoints, training configs, source code from other versions and original
    source images are never installed, overwritten or deleted by this helper.
    """
    root, package_path = Path(root), Path(package_path)
    if not re.fullmatch('[0-9a-f]{64}', expected_sha256):
        raise ValueError('A pinned archive SHA256 is required')
    with package_path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != expected_sha256:
        raise ValueError('Runtime ZIP checksum differs; upload the provided runtime ZIP')
    settings = settings_for(root)
    pending_files = {}
    with zipfile.ZipFile(package_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate runtime ZIP members')
        manifest = json.loads(archive.read('install_manifest.json'))
        if manifest.get('schema_version') != 1 or manifest.get('training_version') != VERSION:
            raise ValueError('Runtime ZIP belongs to another training experiment')
        payloads = manifest['files']; launcher = manifest['launcher']
        if launcher not in payloads or not 1 <= len(payloads) <= 40:
            raise ValueError('Invalid runtime file manifest')
        if set(names) != set(payloads) | {'install_manifest.json', 'install_runtime.py'}:
            raise ValueError('Runtime ZIP contains unexpected files')
        for name, expected in payloads.items():
            _safe_member(name, launcher)
            info = archive.getinfo(name)
            if info.file_size > 20*1024**2:
                raise ValueError('Unexpectedly large runtime payload: '+name)
            value = archive.read(name)
            if digest_bytes(value) != expected:
                raise ValueError('Runtime payload checksum differs: '+name)
            pending_files[name] = value
        source_manifest = json.loads(pending_files[f'code/{VERSION}/source_manifest.json'])
        expected_originals = {f'{folder}/{VERSION}/{name}': sha
            for folder, key in (('code', 'sources'), ('assets', 'assets'))
            for name, sha in source_manifest[key].items()}
        original_payloads = {name: sha for name, sha in payloads.items()
            if name != launcher and not name.endswith('/source_manifest.json')}
        if original_payloads != expected_originals:
            raise ValueError('Original source/asset identity manifest differs')
    settings_path = root/'bootstrap_settings.json'
    if settings_path.is_file():
        # Keep prior user-selected image/weight/cache paths exactly. The existing
        # bootstrap and training identities validate those inputs on execution.
        existing_bytes = settings_path.read_bytes()
        existing = json.loads(existing_bytes)
        if (not isinstance(existing, dict) or set(existing) != set(settings)
                or any(not isinstance(value, str) or not value for value in existing.values())
                or any(existing[key] != settings[key] for key in ('root', 'assets'))):
            raise ValueError('Preserved existing settings belong to a different or invalid runtime')
        pending_files['bootstrap_settings.json'] = existing_bytes
    else:
        pending_files['bootstrap_settings.json'] = json.dumps(settings, sort_keys=True, indent=2).encode()
    # First inspect every target, so a changed saved source/config blocks writes.
    for name, value in pending_files.items():
        target = root/name
        if target.is_symlink() or not target.resolve().is_relative_to(root.resolve()):
            raise ValueError('Runtime destination escapes its experiment folder: '+name)
        if target.exists() and (not target.is_file() or target.read_bytes() != value):
            raise ValueError('Preserved existing file differs: '+str(target))
    created = 0
    for name, value in pending_files.items():
        target = root/name
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name+'.'+uuid.uuid4().hex+'.pending')
        try:
            temporary.write_bytes(value)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        created += 1
    print(f'실행 파일 확인 완료: 새 파일 {created}개. 기존 학습 상태를 유지합니다.', flush=True)
    return str(root/launcher)
