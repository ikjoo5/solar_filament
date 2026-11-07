"""Drive recovery uses read-only APIs and publishes only the exact weight bytes."""
import hashlib
import importlib.util
import json
from pathlib import Path
import types

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('drive_recovery',ROOT/'tools/chase_shape_drive_recovery.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)


class Request:
    def __init__(self,value):self.value=value
    def execute(self,**kwargs):
        if isinstance(self.value,Exception):raise self.value
        return self.value


class Drive:
    def __init__(self,pages=None,data=None):
        self.pages=pages or {};self.data=data or {};self.calls=[]
    def files(self):return self
    def list(self,**kwargs):
        self.calls.append(kwargs)
        trash='trashed = true' in kwargs['q']
        return Request(self.pages.get((trash,kwargs['pageToken']),{'files':[]}))


def download(service,file_id,stream):
    data=service.data[file_id]
    if isinstance(data,Exception):raise data
    stream.write(data)


@pytest.fixture
def setup(tmp_path,monkeypatch):
    payload=b'reviewed epoch82 checkpoint fixture'
    checksum=hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(worker,'CHECKPOINT_SHA',checksum)
    return tmp_path/'recovered',payload,checksum


def item(identifier,**kwargs):
    return dict(id=identifier,name=kwargs.pop('name','best_epoch082.pt'),**kwargs)


def test_paginated_active_and_trash_search_recovers_without_api_mutation(setup):
    output,payload,checksum=setup
    service=Drive({
        (False,None):{'files':[item('wrong')],'nextPageToken':'page2'},
        (False,'page2'):{'files':[item('other',name='best_epoch083.pt')]},
        (True,None):{'files':[item('correct',trashed=True,parents=['backup'])]},
    },{'wrong':b'other model','correct':payload})
    result=worker.recover(service,output,downloader=download)
    assert result['status']=='recovered' and result['search_complete']
    assert result['recovered_file']['trashed'] and result['checkpoint_sha256']==checksum
    assert Path(result['checkpoint']).read_bytes()==payload
    assert result['pages']==3 and len(service.calls)==3 and not result['api_mutations']
    assert all(c['supportsAllDrives'] and c['includeItemsFromAllDrives'] for c in service.calls)
    assert all("'root' in parents" not in c['q'] for c in service.calls)
    assert not list(output.glob('*.pending'))
    assert json.loads(Path(result['report_path']).read_text())['status']=='recovered'


def test_filename_match_is_not_enough_and_mismatched_bytes_are_removed(setup):
    output,payload,checksum=setup
    service=Drive({(False,None):{'files':[item('wrong')]}},{'wrong':b'wrong checkpoint'})
    result=worker.recover(service,output,downloader=download)
    assert result['status']=='not_found' and result['attempts'][0]['status']=='sha_mismatch'
    assert not (output/'best_epoch082.pt').exists() and not list(output.glob('*.pending'))


def test_sha_metadata_selects_renamed_model_but_download_bytes_still_checked(setup):
    output,payload,checksum=setup
    service=Drive({(False,None):{'files':[
        item('renamed',name='best.pth',sha256Checksum=checksum),
        item('skip',sha256Checksum='0'*64),
    ]}},{'renamed':payload})
    result=worker.recover(service,output,downloader=download)
    assert result['status']=='recovered' and result['recovered_file']['id']=='renamed'
    assert any(a['id']=='skip' and a['status']=='metadata_sha_mismatch' for a in result['attempts'])


def test_verified_copy_reused_without_drive_calls(setup):
    output,payload,checksum=setup;output.mkdir();target=output/'best_epoch082.pt';target.write_bytes(payload)
    service=Drive()
    result=worker.recover(service,output,downloader=lambda *_:pytest.fail('download should not run'))
    assert result['status']=='recovered' and result['reused'] and not service.calls
    assert target.read_bytes()==payload


def test_existing_unknown_file_is_preserved(setup):
    output,payload,checksum=setup;output.mkdir();target=output/'best_epoch082.pt';target.write_bytes(b'user file')
    service=Drive()
    result=worker.recover(service,output,downloader=download)
    assert result['status']=='selection_required' and target.read_bytes()==b'user file'
    assert not service.calls


def test_output_and_target_symlinks_rejected(setup,tmp_path):
    output,payload,checksum=setup;outside=tmp_path/'outside';outside.mkdir();output.symlink_to(outside)
    with pytest.raises(ValueError,match='symlink'):worker.recover(Drive(),output)
    output.unlink();output.mkdir();(output/'best_epoch082.pt').symlink_to(outside/'other.pt')
    with pytest.raises(ValueError,match='symlink'):worker.recover(Drive(),output)


def test_auth_error_is_not_reported_as_absence_or_leaked(setup):
    output,payload,checksum=setup
    error=RuntimeError('secret-token-must-not-appear');error.resp=types.SimpleNamespace(status=403)
    service=Drive({(False,None):error})
    result=worker.recover(service,output,downloader=download)
    assert result['status']=='access_error' and not result['search_complete']
    assert result['errors'][0]['http_status']==403
    assert 'secret-token' not in Path(result['report_path']).read_text()


def test_unreadable_trashed_file_keeps_link_without_auto_restore(setup):
    output,payload,checksum=setup
    error=RuntimeError('private response');error.resp=types.SimpleNamespace(status=404)
    service=Drive({(True,None):{'files':[item('trashed',trashed=True)]}},{'trashed':error})
    result=worker.recover(service,output,downloader=download)
    assert result['status']=='access_error' and result['candidates'][0]['trashed']
    assert result['candidates'][0]['webViewLink'].endswith('/trashed/view')
    assert result['attempts'][0]['http_status']==404 and not list(output.glob('*.pending'))


def test_page_and_candidate_limits_report_incomplete(setup):
    output,payload,checksum=setup
    service=Drive({(False,None):{'files':[],'nextPageToken':'more'}})
    result=worker.recover(service,output,max_pages=1,downloader=download)
    assert result['status']=='incomplete' and not result['search_complete']
    service=Drive({(False,None):{'files':[item('one'),item('two')]}},{'one':b'wrong','two':payload})
    result=worker.recover(service,output,max_candidates=1,downloader=download)
    assert result['status']=='incomplete' and not result['search_complete']


def test_size_limits_prevent_large_recovery_and_cleanup_partial_download(setup):
    output,payload,checksum=setup
    service=Drive({(False,None):{'files':[item('huge',size='1000')]}})
    result=worker.recover(service,output,max_bytes=10,downloader=lambda *_:pytest.fail('oversize download'))
    assert result['status']=='incomplete' and result['attempts'][0]['status']=='size_limit'
    service=Drive({(False,None):{'files':[item('misreported',size='1')]}},{'misreported':payload})
    result=worker.recover(service,output,max_bytes=10,downloader=download)
    assert result['status']=='access_error' and not list(output.glob('*.pending'))
    assert not (output/'best_epoch082.pt').exists()


def test_report_preserves_drive_incomplete_search_flag(setup):
    output,payload,checksum=setup
    service=Drive({(False,None):{'files':[],'incompleteSearch':True}})
    result=worker.recover(service,output,downloader=download)
    assert result['status']=='incomplete' and not result['search_complete']
