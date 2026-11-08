"""Bounded retention may delete only verified, committed files of the new run."""
import copy
import hashlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import fullpass_checkpoint_store as worker
from test_verified_drive_checkpoints import Drive as OldDrive, Request, download, state


class Drive(OldDrive):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sequence = 0
        self.deleted = []

    def delete(self, **kwargs):
        file_id = kwargs['fileId']
        self.calls.append(('delete', kwargs))
        self.deleted.append(copy.deepcopy(self.items[file_id]))
        del self.items[file_id]
        del self.data[file_id]
        return Request({})


def upload(service, path, metadata):
    service.sequence += 1
    file_id = 'uploaded-' + str(service.sequence)
    data = Path(path).read_bytes()
    if metadata['appProperties']['kind'] == service.corrupt_kind:
        data = bytes([data[0] ^ 1]) + data[1:]
    item = dict(copy.deepcopy(metadata), id=file_id, size=str(len(data)), trashed=False)
    if service.server_sha:
        item['sha256Checksum'] = hashlib.sha256(data).hexdigest()
    service.items[file_id], service.data[file_id] = item, data
    service.calls.append(('upload', copy.deepcopy(metadata)))
    return item


def store(tmp_path, drive=None):
    return worker.BoundedDriveCheckpointStore(drive or Drive(), 'folder', 'identity',
        tmp_path/'cache', uploader=upload, downloader=download)


def progress(storage, tmp_path, epoch, step):
    path = tmp_path/'progress.pt'
    path.write_bytes(f'epoch={epoch} step={step}'.encode())
    return storage.publish_progress(path, metadata={'epoch':epoch, 'step':step, 'base_epoch':epoch-1})


@pytest.mark.parametrize('server_sha', [True, False])
def test_retains_latest_two_epoch_checkpoints_and_complete_state_chain(tmp_path, server_sha):
    storage = store(tmp_path, Drive(server_sha=server_sha))
    states = []
    for epoch in range(1, 5):
        states.append(storage.commit_state(state(storage, tmp_path, epoch), expected_previous_epoch=epoch-1))
    items = list(storage.service.items.values())
    counts = {kind:len([x for x in items if x.get('appProperties', {}).get('kind')==kind])
              for kind in ['last','best','state']}
    assert counts == {'last':2,'best':2,'state':4}
    assert storage.latest_state()['epoch'] == 4
    assert storage.restore(states[-2]['last']).read_bytes() == b'last checkpoint epoch 3'
    restarted = store(tmp_path, storage.service)
    assert restarted.latest_state()['last'] == states[-1]['last']
    assert restarted.restore(states[-1]['best']).read_bytes() == b'best checkpoint epoch 4'
    for item in storage.service.deleted:
        assert item['parents'] == ['folder']
        assert item['appProperties']['run_identity'] == 'identity'
        assert item['appProperties']['kind'] in ['last','best']


def test_retains_old_best_but_does_not_sweep_uncommitted_or_other_run(tmp_path):
    storage = store(tmp_path)
    first = storage.commit_state(state(storage, tmp_path, 1), expected_previous_epoch=0)
    orphan_path=tmp_path/'uncommitted.pt'; orphan_path.write_bytes(b'not yet committed')
    orphan=storage.publish_checkpoint(orphan_path, 'last999.pt', {'kind':'last','epoch':999})
    foreign = {'id':'foreign', 'name':'old_other.pt', 'parents':['other'],
               'appProperties':{'run_identity':'other','kind':'last'}}
    storage.service.items['foreign']=copy.deepcopy(foreign)
    for epoch in range(2, 5):
        storage.commit_state(state(storage, tmp_path, epoch, best=first['best']), expected_previous_epoch=epoch-1)
    assert storage.latest_state()['best']['file_id'] == first['best']['file_id']
    assert orphan['file_id'] in storage.service.items
    assert storage.service.items['foreign'] == foreign


def test_failed_epoch_state_readback_does_not_prune_old_checkpoints(tmp_path):
    storage = store(tmp_path)
    for epoch in (1,2):
        storage.commit_state(state(storage, tmp_path, epoch), expected_previous_epoch=epoch-1)
    before=set(storage.service.items)
    storage.service.corrupt_kind='state'
    with pytest.raises(worker.durable.IntegrityError):
        storage.commit_state(state(storage, tmp_path, 3), expected_previous_epoch=2)
    assert before.issubset(storage.service.items)
    assert not storage.service.deleted


@pytest.mark.parametrize('server_sha', [True, False])
def test_progress_restores_latest_cursor_and_retains_two_generations(tmp_path, server_sha):
    storage = store(tmp_path, Drive(server_sha=server_sha))
    records = [progress(storage, tmp_path, 1, step) for step in (400,800,1200)]
    restarted=store(tmp_path, storage.service)
    latest=restarted.latest_progress()
    assert {k:latest[k] for k in ['epoch','step','base_epoch']} == {'epoch':1,'step':1200,'base_epoch':0}
    assert restarted.restore(latest['checkpoint']).read_bytes() == b'epoch=1 step=1200'
    assert records[0]['checkpoint']['file_id'] not in storage.service.items
    assert records[0]['_commit']['file_id'] not in storage.service.items
    assert records[1]['checkpoint']['file_id'] in storage.service.items
    assert records[1]['_commit']['file_id'] in storage.service.items
    assert storage.latest_state() is None


def test_progress_failure_cannot_destroy_last_verified_progress(tmp_path):
    storage=store(tmp_path)
    first=progress(storage, tmp_path, 1,400)
    storage.service.corrupt_kind='progress'
    with pytest.raises(worker.durable.IntegrityError):
        progress(storage,tmp_path,1,800)
    assert storage.latest_progress()['step']==400
    assert storage.restore(first['checkpoint']).read_bytes()==b'epoch=1 step=400'
    assert not storage.service.deleted


def test_progress_clear_requires_commit_and_ignores_completed_progress_on_resume(tmp_path):
    storage=store(tmp_path)
    partial=progress(storage,tmp_path,1,400)
    with pytest.raises(worker.durable.StateConflict):
        storage.clear_progress(completed_epoch=1)
    storage.commit_state(state(storage,tmp_path,1),expected_previous_epoch=0)
    assert storage.latest_progress() is None
    newer=progress(storage,tmp_path,2,400)
    storage.clear_progress(completed_epoch=1)
    assert partial['checkpoint']['file_id'] not in storage.service.items
    assert newer['checkpoint']['file_id'] in storage.service.items
    assert storage.latest_progress()['epoch']==2


def test_progress_rejects_stale_or_duplicate_cursor(tmp_path):
    storage=store(tmp_path)
    progress(storage,tmp_path,1,400)
    for step in (300,400):
        with pytest.raises(worker.durable.StateConflict,match='advance'):
            progress(storage,tmp_path,1,step)
    storage.commit_state(state(storage,tmp_path,1),expected_previous_epoch=0)
    with pytest.raises(worker.durable.StateConflict,match='epoch changed'):
        progress(storage,tmp_path,1,800)


def test_progress_conflict_or_corrupt_newest_never_falls_back(tmp_path):
    storage=store(tmp_path)
    progress(storage,tmp_path,1,400)
    newest=progress(storage,tmp_path,1,800)
    storage.service.data[newest['_commit']['file_id']]=b'corrupt'
    with pytest.raises(worker.durable.IntegrityError):
        storage.latest_progress()


def test_duplicate_progress_generation_is_detected(tmp_path):
    storage=store(tmp_path)
    newest=progress(storage,tmp_path,1,400)
    record=newest['_commit']; item=copy.deepcopy(storage.service.items[record['file_id']])
    item['id']='duplicate'; storage.service.items['duplicate']=item
    storage.service.data['duplicate']=storage.service.data[record['file_id']]
    with pytest.raises(worker.durable.StateConflict,match='Duplicate progress'):
        storage.latest_progress()


def test_bounded_store_preflight_requires_two_gb_not_full_run_budget(tmp_path):
    storage=store(tmp_path)
    storage.service.quota['usage']=str(197_000_000_000)
    assert storage.preflight()['quota']['required']==2_000_000_000
