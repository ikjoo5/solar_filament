"""Real ZIP round trips, integrity and atomic recovery without ML imports."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'tools'))
import chase_shape_cache as cache


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2)+'\n')


@pytest.fixture
def prepared(tmp_path):
    root = tmp_path/'output'
    local = tmp_path/'local'
    source_settings = dict(version='fixture', seed=42)
    identity = cache._fingerprint(source_settings)
    images = local/'chase_shape'/identity[:16]/'images'
    images.mkdir(parents=True)
    rows, records, originals = [], [], {}
    for index in range(3):
        name = f'chase_shape_{index:06d}.png'
        payload = b'\x89PNG\r\n\x1a\n'+bytes([index])*4097
        (images/name).write_bytes(payload)
        originals[name] = payload
        rows.append(dict(id=f'chase_shape_{index:06d}', file_name=name, width=2048, height=2048))
        records.append(dict(png_sha256=hashlib.sha256(payload).hexdigest(), seed_components=[42,index,20261106]))
    synthesis = root/'synthesis'
    annotation = synthesis/'synthetic_annotations.json'
    write_json(annotation, dict(images=rows, annotations=[], info=dict(identity=identity)))
    manifest = dict(identity=identity, settings=source_settings, records=records,
                    synthetic_variants=3, annotations_sha256=cache._digest(annotation))
    manifest_path = synthesis/'manifest.json'
    write_json(manifest_path, dict(identity=identity, payload=manifest, payload_sha256=cache._fingerprint(manifest)))
    write_json(synthesis/'source_identity.json', dict(identity=identity, settings=source_settings))
    write_json(root/'experiment_config.json', dict(output_root=str(root), cache_root=str(local)))
    write_json(root/'prepared_summary.json', dict(identity=identity, images_dir=str(images),
        annotations_path=str(annotation), manifest_path=str(manifest_path), physical_images=3))
    (root/'checkpoint.pt').write_bytes(b'untouched checkpoint')
    return root, images, originals


def rewrite_zip(path, mutate):
    with zipfile.ZipFile(path) as archive:
        members = {name:archive.read(name) for name in archive.namelist()}
    mutate(members)
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_STORED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


def test_backup_clear_restore_preserves_exact_bytes_and_other_outputs(prepared):
    root, images, originals = prepared
    config = (root/'experiment_config.json').read_bytes()
    result = cache.backup(root)
    assert result['status']=='backed_up' and result['images']==3
    with zipfile.ZipFile(result['archive']) as archive:
        assert all(info.compress_type==zipfile.ZIP_STORED for info in archive.infolist())
        assert len(archive.namelist())==4
    shutil.rmtree(images.parent.parent)
    restored = cache.restore(root)
    assert restored['status']=='restored' and restored['restored']==3
    assert {p.name:p.read_bytes() for p in images.glob('*.png')}==originals
    assert (root/'checkpoint.pt').read_bytes()==b'untouched checkpoint'
    assert (root/'experiment_config.json').read_bytes()==config


def test_all_local_valid_skips_even_missing_or_invalid_zip(prepared):
    root, images, _ = prepared
    cache.archive_path(root).write_bytes(b'invalid ZIP deliberately unused')
    with patch.object(cache.zipfile, 'ZipFile', side_effect=AssertionError('must not inspect archive')):
        result = cache.restore(root)
    assert result['status']=='local_complete' and result['restored']==0


def test_only_missing_corrupt_images_restored(prepared):
    root, images, originals = prepared
    cache.backup(root)
    stable = images/'chase_shape_000000.png'
    stable_time = stable.stat().st_mtime_ns
    (images/'chase_shape_000001.png').unlink()
    (images/'chase_shape_000002.png').write_bytes(b'broken local PNG')
    result = cache.restore(root)
    assert result['restored']==2 and stable.stat().st_mtime_ns==stable_time
    assert {p.name:p.read_bytes() for p in images.glob('*.png')}==originals


def test_valid_backup_is_not_rewritten(prepared):
    root, images, _ = prepared
    cache.backup(root)
    target = cache.archive_path(root)
    before = target.read_bytes(), target.stat().st_mtime_ns
    shutil.rmtree(images)
    assert cache.backup(root)['status']=='already_backed_up'
    assert (target.read_bytes(),target.stat().st_mtime_ns)==before


def test_identity_mismatch_preserves_archive_and_writes_no_images(prepared):
    root, images, _ = prepared
    cache.backup(root)
    target = cache.archive_path(root)
    def change(members):
        meta = json.loads(members['cache_manifest.json'])
        meta['identity']='0'*64
        members['cache_manifest.json']=json.dumps(meta).encode()
    rewrite_zip(target, change)
    old = target.read_bytes()
    shutil.rmtree(images)
    with pytest.raises(ValueError, match='identity/inventory'):cache.restore(root)
    with pytest.raises(ValueError, match='identity/inventory'):cache.backup(root)
    assert not images.exists() and target.read_bytes()==old


def test_traversal_or_duplicate_zip_members_rejected_before_restore(prepared):
    root, images, _ = prepared
    cache.backup(root)
    target = cache.archive_path(root)
    rewrite_zip(target, lambda members:members.update({'../escape.png':b'bad'}))
    shutil.rmtree(images)
    with pytest.raises(ValueError, match='unsafe ZIP'):cache.restore(root)
    assert not images.exists() and not (root.parent/'escape.png').exists()


def test_corrupt_zip_image_does_not_replace_existing_local_file(prepared):
    root, images, _ = prepared
    cache.backup(root)
    target = cache.archive_path(root)
    name='chase_shape_000000.png'
    rewrite_zip(target, lambda members:members.update({'images/'+name:b'wrong archive pixels'}))
    local = images/name
    local.write_bytes(b'old corrupt cache kept until replacement verified')
    before=local.read_bytes()
    with pytest.raises(ValueError, match='PNG checksum'):cache.restore(root)
    assert local.read_bytes()==before
    assert not list(images.glob('*.pending'))


def test_interrupted_backup_never_publishes_partial_archive(prepared):
    root, images, _ = prepared
    original = cache._copy_checked
    calls = 0
    def interrupted(source,destination,expected):
        nonlocal calls
        calls+=1
        if calls==2:
            destination.write(b'partial')
            raise OSError('fixture disk full')
        return original(source,destination,expected)
    with patch.object(cache,'_copy_checked',side_effect=interrupted):
        with pytest.raises(OSError,match='disk full'):cache.backup(root)
    assert not cache.archive_path(root).exists()
    assert not list(root.glob('*.pending'))
    cache.backup(root)
    target = cache.archive_path(root);saved=target.read_bytes()
    with patch.object(cache,'_verify_archive',side_effect=OSError('interrupted backup verification')):
        with pytest.raises(OSError):cache.backup(root)
    assert target.read_bytes()==saved


def test_metadata_tampering_and_unsafe_local_paths_rejected(prepared):
    root, images, _ = prepared
    path=root/'synthesis/manifest.json';original=path.read_text()
    value=json.loads(original);value['payload']['records'][0]['png_sha256']='0'*64
    write_json(path,value)
    with pytest.raises(ValueError,match='checksum/identity'):cache.backup(root)
    path.write_text(original)
    summary=root/'prepared_summary.json';value=json.loads(summary.read_text())
    value['images_dir']=str(root/'synthesis')
    write_json(summary,value)
    with pytest.raises(ValueError,match='configured local cache'):cache.restore(root)


def test_missing_cache_and_missing_preparation_are_explicit_statuses(prepared, tmp_path):
    root, images, _ = prepared
    shutil.rmtree(images)
    result=cache.restore(root)
    assert result['status']=='archive_missing' and result['missing']==3
    assert cache.restore(tmp_path/'new_output')['status']=='not_prepared'
