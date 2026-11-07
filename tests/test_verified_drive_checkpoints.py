"""Durability tests use an in-memory Drive API, never mounted Drive files."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('verified_drive_checkpoints', ROOT / 'tools/verified_drive_checkpoints.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class Request:
    def __init__(self, value):
        self.value = value

    def execute(self):
        return copy.deepcopy(self.value)


class Drive:
    def __init__(self, *, server_sha=True):
        self.server_sha = server_sha
        self.items, self.data, self.calls, self.downloads = {}, {}, [], []
        self.corrupt_kind = None
        self.quota = {'limit': str(200_000_000_000), 'usage': str(100_000_000_000)}
        self.folder('folder', 'checkpoints', 'experiment',
                    {'run_identity': 'identity', 'kind': 'checkpoint_folder'})

    def folder(self, file_id, name, parent, properties=None):
        self.items[file_id] = {'id': file_id, 'name': name, 'mimeType': worker.FOLDER_MIME,
                              'parents': [parent], 'appProperties': properties or {}, 'trashed': False}

    def files(self):
        return self

    def about(self):
        return self

    def get(self, **kwargs):
        self.calls.append(('get', kwargs))
        if kwargs.get('fields') == 'storageQuota':
            return Request({'storageQuota': self.quota})
        return Request(self.items[kwargs['fileId']])

    def create(self, *, body, **kwargs):
        self.calls.append(('create_folder', copy.deepcopy(body)))
        assert body['mimeType'] == worker.FOLDER_MIME
        file_id = str(len(self.items) + 1)
        self.items[file_id] = dict(body, id=file_id, trashed=False)
        return Request(self.items[file_id])

    def list(self, **kwargs):
        self.calls.append(('list', kwargs))
        query = kwargs['q']
        parent = re.search(r"'([^']+)' in parents", query).group(1)
        name = re.search(r"name = '([^']+)'", query)
        identity = re.search(r"key='run_identity' and value='([^']+)'", query)
        kind = re.search(r"key='kind' and value='([^']+)'", query)
        items = [item for item in self.items.values()
                 if parent in item.get('parents', []) and not item.get('trashed')
                 and (not name or item['name'] == name.group(1))
                 and (not identity or item.get('appProperties', {}).get('run_identity') == identity.group(1))
                 and (not kind or item.get('appProperties', {}).get('kind') == kind.group(1))
                 and ('mimeType =' not in query or item.get('mimeType') == worker.FOLDER_MIME)]
        return Request({'files': items})

    # No update/delete/trash methods exist: using one fails the test immediately.


def upload(service, path, metadata):
    service.calls.append(('upload', copy.deepcopy(metadata)))
    file_id = str(len(service.items) + 1)
    data = Path(path).read_bytes()
    if metadata['appProperties']['kind'] == service.corrupt_kind:
        data = bytes([data[0] ^ 1]) + data[1:]
    item = dict(copy.deepcopy(metadata), id=file_id, size=str(len(data)), trashed=False)
    if service.server_sha:
        item['sha256Checksum'] = hashlib.sha256(data).hexdigest()
    service.items[file_id], service.data[file_id] = item, data
    return item


def download(service, file_id, stream):
    service.downloads.append(file_id)
    data = service.data[file_id]
    for start in range(0, len(data), 7):
        stream.write(data[start:start + 7])


def store(tmp_path, drive=None, **kwargs):
    return worker.DriveCheckpointStore(drive or Drive(), 'folder', 'identity', tmp_path / 'cache',
                                        uploader=upload, downloader=download, **kwargs)


def checkpoint(storage, tmp_path, epoch, kind='last'):
    path = tmp_path / f'{kind}{epoch}.pt'
    path.write_bytes(f'{kind} checkpoint epoch {epoch}'.encode())
    return storage.publish_checkpoint(path, f'checkpoints/{kind}_epoch{epoch:03d}.pt',
                                      {'epoch': epoch, 'kind': kind})


def state(storage, tmp_path, epoch, best=None):
    best = best or checkpoint(storage, tmp_path, epoch, 'best')
    return {'epoch': epoch, 'last': checkpoint(storage, tmp_path, epoch),
            'best': best, 'best_epoch': int(storage.service.items[best['file_id']]['appProperties']['epoch']),
            'best_pq': 0.43}


@pytest.mark.parametrize('server_sha', [True, False])
def test_epoch_is_committed_only_after_server_verification_and_restores(tmp_path, server_sha):
    drive = Drive(server_sha=server_sha)
    storage = store(tmp_path, drive)
    preflight = storage.preflight()
    assert preflight['quota']['free'] == 100_000_000_000
    payload = state(storage, tmp_path, 1)
    result = storage.commit_state(payload, expected_previous_epoch=0)
    assert result['epoch'] == 1 and result['_commit']['file_id']
    assert storage.latest_state() == result
    restored = storage.restore(result['last'])
    assert restored.read_bytes() == b'last checkpoint epoch 1'
    assert worker.sha256_file(restored) == result['last']['sha256']
    uploaded_kinds = [call[1]['appProperties']['kind'] for call in drive.calls if call[0] == 'upload']
    assert uploaded_kinds == ['probe', 'best', 'last', 'state']
    if not server_sha:
        assert payload['last']['file_id'] in drive.downloads and payload['best']['file_id'] in drive.downloads


@pytest.mark.parametrize('server_sha', [True, False])
def test_failed_checkpoint_verification_cannot_advance_state(tmp_path, server_sha):
    drive = Drive(server_sha=server_sha)
    storage = store(tmp_path, drive)
    initial = storage.commit_state(state(storage, tmp_path, 1), expected_previous_epoch=0)
    drive.corrupt_kind = 'last'
    with pytest.raises(worker.IntegrityError, match='SHA-256'):
        checkpoint(storage, tmp_path, 2)
    assert storage.latest_state() == initial
    assert len([i for i in drive.items.values() if i.get('appProperties', {}).get('kind') == 'state']) == 1


def test_all_epoch_files_retained_and_other_runs_untouched(tmp_path):
    drive = Drive()
    drive.items['old-run'] = {'id': 'old-run', 'name': 'best_epoch082.pt',
                              'parents': ['other'], 'appProperties': {'run_identity': 'old'}}
    before = copy.deepcopy(drive.items['old-run'])
    storage = store(tmp_path, drive)
    best = None
    records = []
    for epoch in (1, 2, 3):
        payload = state(storage, tmp_path, epoch, best=best)
        best = payload['best']
        records.append(storage.commit_state(payload, expected_previous_epoch=epoch - 1))
    assert drive.items['old-run'] == before
    assert all(r['last']['file_id'] in drive.items for r in records)
    assert len([i for i in drive.items.values() if i.get('appProperties', {}).get('kind') == 'last']) == 3
    assert len([i for i in drive.items.values() if i.get('appProperties', {}).get('kind') == 'best']) == 1
    assert len([i for i in drive.items.values() if i.get('appProperties', {}).get('kind') == 'state']) == 3


def test_stale_writer_cannot_commit_after_another_epoch_advanced(tmp_path):
    storage = store(tmp_path)
    payload = state(storage, tmp_path, 1)
    storage.commit_state(payload, expected_previous_epoch=0)
    with pytest.raises(worker.StateConflict, match='changed'):
        storage.commit_state(payload, expected_previous_epoch=0)


def test_duplicate_state_generation_is_not_silently_selected(tmp_path):
    storage = store(tmp_path)
    committed = storage.commit_state(state(storage, tmp_path, 1), expected_previous_epoch=0)
    record = committed['_commit']
    other = copy.deepcopy(storage.service.items[record['file_id']])
    other['id'] = 'competing-state'
    storage.service.items[other['id']] = other
    storage.service.data[other['id']] = storage.service.data[record['file_id']]
    with pytest.raises(worker.StateConflict, match='Duplicate'):
        storage.latest_state()


def test_newest_corrupt_state_does_not_fall_back_to_older_epoch(tmp_path):
    storage = store(tmp_path)
    storage.commit_state(state(storage, tmp_path, 1), expected_previous_epoch=0)
    last = storage.commit_state(state(storage, tmp_path, 2), expected_previous_epoch=1)
    key = last['_commit']['file_id']
    storage.service.data[key] = b'corrupt'
    with pytest.raises(worker.IntegrityError, match='readback hash'):
        storage.latest_state()


def test_resume_rechecks_local_cache_and_rejects_bad_remote_download(tmp_path):
    storage = store(tmp_path)
    record = checkpoint(storage, tmp_path, 1)
    restored = storage.restore(record)
    restored.write_bytes(b'corrupted local bytes')
    assert storage.restore(record).read_bytes() == b'last checkpoint epoch 1'
    restored.unlink()
    storage.service.data[record['file_id']] = b'wrong bytes'
    with pytest.raises(worker.IntegrityError, match='Restored'):
        storage.restore(record)
    assert not restored.exists()
    assert not list(storage.local_cache.glob('*.pending'))


def test_record_from_another_run_is_rejected_even_when_bytes_match(tmp_path):
    storage = store(tmp_path)
    record = checkpoint(storage, tmp_path, 1)
    storage.service.items[record['file_id']]['parents'] = ['other-folder']
    with pytest.raises(worker.IntegrityError, match='ownership mismatch'):
        storage.restore(record)


def test_quota_guard_and_write_probe_run_before_training(tmp_path):
    storage = store(tmp_path)
    storage.service.quota['usage'] = str(199_000_000_000)
    with pytest.raises(RuntimeError, match='Insufficient verified Drive capacity'):
        storage.preflight()
    assert not any(call[0] == 'upload' for call in storage.service.calls)
    storage.service.quota = {}
    with pytest.raises(RuntimeError, match='capacity cannot be verified'):
        storage.preflight()


def test_bad_write_probe_fails_preflight_without_any_state(tmp_path):
    storage = store(tmp_path)
    storage.service.corrupt_kind = 'probe'
    with pytest.raises(worker.IntegrityError):
        storage.preflight()
    assert storage.latest_state() is None


def test_resume_can_preflight_remaining_budget_instead_of_full_run_budget(tmp_path):
    storage = store(tmp_path)
    storage.commit_state(state(storage, tmp_path, 1), expected_previous_epoch=0)
    storage.service.quota['usage'] = str(197_000_000_000)
    with pytest.raises(RuntimeError, match='Insufficient verified Drive capacity'):
        storage.preflight()
    report = storage.preflight(required_free_bytes=2_000_000_000)
    assert report['latest_epoch'] == 1
    assert report['quota']['required'] == 2_000_000_000
    assert report['quota']['free'] == 3_000_000_000
    assert storage.latest_state()['epoch'] == 1


def test_visibility_retry_does_not_create_duplicate_fuse_experiment(tmp_path, monkeypatch):
    drive = Drive()
    drive.folder('project', 'project', 'root')
    drive.folder('experiments', 'experiments', 'project')
    waits = []

    def visible_after_wait(seconds):
        waits.append(seconds)
        drive.folder('existing-run', 'run', 'experiments')

    monkeypatch.setattr(worker.time, 'sleep', visible_after_wait)
    identifier = worker.resolve_experiment_folder(drive,
        ['project', 'experiments', 'run', 'verified'], 'identity',
        create_from=3, visibility_attempts=6)
    assert waits == [2.0]
    assert drive.items[identifier]['parents'] == ['existing-run']
    created_names = [body['name'] for kind, body in drive.calls if kind == 'create_folder']
    assert created_names == ['verified']


def test_folder_lookup_requires_visible_base_and_rejects_ambiguity(tmp_path):
    drive = Drive()
    with pytest.raises(FileNotFoundError, match='Required Drive base'):
        worker.resolve_experiment_folder(drive, ['project', 'experiments', 'new', 'checkpoints'], 'identity')
    assert not any(c[0] == 'create_folder' for c in drive.calls)
    drive.folder('project', 'project', 'root')
    drive.folder('experiments', 'experiments', 'project')
    folder_id = worker.resolve_experiment_folder(drive, ['project', 'experiments', 'new', 'verified'], 'identity')
    assert drive.items[folder_id]['appProperties']['run_identity'] == 'identity'
    assert worker.resolve_experiment_folder(drive, ['project', 'experiments', 'new', 'verified'], 'identity') == folder_id
    drive.folder('duplicate', 'project', 'root')
    with pytest.raises(worker.StateConflict, match='Ambiguous'):
        worker.resolve_experiment_folder(drive, ['project', 'experiments', 'new', 'verified'], 'identity')


def test_unowned_final_folder_is_never_adopted_or_modified(tmp_path):
    drive = Drive()
    drive.folder('project', 'project', 'root')
    drive.folder('experiments', 'experiments', 'project')
    drive.folder('unknown', 'unknown', 'experiments')
    before = copy.deepcopy(drive.items['unknown'])
    with pytest.raises(worker.IntegrityError, match='not owned'):
        worker.resolve_experiment_folder(drive, ['project', 'experiments', 'unknown'], 'identity')
    assert drive.items['unknown'] == before


def test_path_escape_and_mismatched_epoch_cannot_be_committed(tmp_path):
    storage = store(tmp_path)
    path = tmp_path / 'checkpoint.pt'; path.write_bytes(b'weights')
    with pytest.raises(ValueError, match='relative'):
        storage.publish_checkpoint(path, '../another-run/best.pt', {'kind': 'best', 'epoch': 1})
    payload = state(storage, tmp_path, 2)
    payload['epoch'] = 1
    with pytest.raises(worker.IntegrityError, match='another epoch'):
        storage.commit_state(payload, expected_previous_epoch=0)
    assert storage.latest_state() is None
