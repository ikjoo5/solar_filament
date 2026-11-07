"""Recover one reviewed checkpoint through read-only Google Drive API calls.

Active and trashed files are listed separately. No file is restored, moved,
deleted or modified through the API. Only an exact SHA-256 match is published
as a local recovery copy; a missing match is a report, not a training request.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import uuid

CHECKPOINT_SHA = '8ea1198734dd434c42889aee2c79928afd8eb8ce4491e826c13d07e41e03d330'
MAX_BYTES = 2 * 1024**3
# Drive's name-contains operator is prefix based. Include the actual saved
# filename prefix and common copied/renamed checkpoint prefixes explicitly.
NAME_QUERY = '(' + ' or '.join("name contains '"+name+"'" for name in
    ('best', 'Best', 'epoch082', 'epoch82', 'checkpoint', 'model', '082', '82', 'Copy of best')) + ')'
FIELDS = 'nextPageToken,incompleteSearch,files(id,name,size,trashed,modifiedTime,webViewLink,parents,sha256Checksum,md5Checksum)'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def atomic_json(path,value):
    pending=path.with_name(path.name+'.'+uuid.uuid4().hex+'.pending')
    try:
        pending.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
        pending.replace(path)
    finally:pending.unlink(missing_ok=True)


def safe_error(error):
    """Do not include OAuth tokens, HTTP bodies or credential-bearing URLs."""
    status=getattr(getattr(error,'resp',None),'status',None)
    return dict(error_type=type(error).__name__,http_status=status if isinstance(status,int) else None)


def candidate(item):
    name=item.get('name','')
    if Path(name).suffix.lower() not in ('.pt','.pth'):
        return False
    return (item.get('sha256Checksum')==CHECKPOINT_SHA or
            bool(re.search(r'(?<!\d)0?82(?!\d)',Path(name).stem)))


class LimitedWriter:
    def __init__(self,stream,limit):
        self.stream,self.limit,self.written=stream,limit,0

    def write(self,data):
        if self.written+len(data)>self.limit:
            raise ValueError('Checkpoint exceeds the recovery byte limit')
        count=self.stream.write(data);self.written+=count
        return count


def download_media(service,file_id,stream):
    # Imported only when authenticated Colab actually downloads a candidate.
    from googleapiclient.http import MediaIoBaseDownload
    request=service.files().get_media(fileId=file_id,supportsAllDrives=True)
    transfer=MediaIoBaseDownload(stream,request,chunksize=8*1024**2)
    done=False;previous=-1
    while not done:
        progress,done=transfer.next_chunk(num_retries=2)
        percentage=int(progress.progress()*100) if progress else 0
        if percentage//20!=previous:
            print('Checkpoint download:',percentage,'%',flush=True);previous=percentage//20


def recover(service,output,*,downloader=download_media,max_candidates=20,
            max_pages=100,max_bytes=MAX_BYTES):
    """Return recovered/not_found/incomplete/access_error/selection_required.

    ``service`` is an authenticated Drive-v3 service. Tests inject both service
    and downloader. All API methods used here are list/get_media (read-only).
    ``output`` is a new recovery directory separate from the training folder.
    """
    supplied=Path(output)
    if supplied.is_symlink():raise ValueError('Recovery output cannot be a symlink')
    output=supplied.resolve();output.mkdir(parents=True,exist_ok=True)
    target=output/'best_epoch082.pt';report_path=output/'drive_recovery_report.json'
    if report_path.is_symlink() or target.is_symlink():
        raise ValueError('Recovery output files cannot be symlinks')
    report=dict(expected_epoch=82,expected_sha256=CHECKPOINT_SHA,output=str(output),
        scope='Drive API name-prefix search, active and trashed accessible files; no folder restriction',
        query=NAME_QUERY,search_complete=False,pages=0,candidates=[],attempts=[],errors=[],
        api_mutations=False)

    def finish(status,**values):
        report.update(status=status,**values);atomic_json(report_path,report)
        report['report_path']=str(report_path)
        return report

    if target.exists():
        if target.is_file() and digest(target)==CHECKPOINT_SHA:
            return finish('recovered',checkpoint=str(target),checkpoint_sha256=CHECKPOINT_SHA,reused=True)
        return finish('selection_required',message='Existing recovery file differs and was preserved; choose another recovery directory.')
    seen=set();listing_failed=False;truncated=False
    for trashed in (False,True):
        token=None
        while True:
            if report['pages']>=max_pages:
                truncated=True;break
            try:
                page=service.files().list(q="mimeType != 'application/vnd.google-apps.folder' and "+NAME_QUERY+
                    ' and trashed = '+str(trashed).lower(),pageSize=1000,pageToken=token,
                    fields=FIELDS,spaces='drive',corpora='user',supportsAllDrives=True,
                    includeItemsFromAllDrives=True).execute(num_retries=2)
            except Exception as error:
                report['errors'].append(dict(phase='list',trashed=trashed,**safe_error(error)))
                listing_failed=True;break
            report['pages']+=1
            truncated=truncated or bool(page.get('incompleteSearch'))
            for item in page.get('files',[]):
                if item.get('id') in seen or not candidate(item):continue
                seen.add(item['id'])
                record={key:item[key] for key in ('id','name','size','trashed','modifiedTime','webViewLink',
                                                  'parents','sha256Checksum','md5Checksum') if key in item}
                record.setdefault('trashed',trashed)
                record.setdefault('webViewLink','https://drive.google.com/file/d/'+item['id']+'/view')
                report['candidates'].append(record)
            token=page.get('nextPageToken')
            if not token:break
        if listing_failed:break
    report['search_complete']=not listing_failed and not truncated
    # A metadata SHA match gets first priority, followed by the original name.
    report['candidates'].sort(key=lambda r:(r.get('sha256Checksum')!=CHECKPOINT_SHA,
        r['name']!='best_epoch082.pt',r.get('trashed',False),r['name'],r['id']))
    eligible=[]
    for item in report['candidates']:
        known_sha=item.get('sha256Checksum')
        try:size=int(item.get('size',0))
        except (ValueError,TypeError):size=0
        if known_sha and known_sha!=CHECKPOINT_SHA:
            report['attempts'].append(dict(id=item['id'],status='metadata_sha_mismatch'));continue
        if size>max_bytes:
            report['attempts'].append(dict(id=item['id'],status='size_limit'));truncated=True;continue
        eligible.append(item)
    if len(eligible)>max_candidates:truncated=True
    for index,item in enumerate(eligible[:max_candidates],1):
        pending=output/('download-'+uuid.uuid4().hex+'.pending')
        print('Checking Drive candidate',index,'/',min(len(eligible),max_candidates),item['name'],
              '(trash)' if item.get('trashed') else '(active)',flush=True)
        try:
            with pending.open('xb') as stream:
                downloader(service,item['id'],LimitedWriter(stream,max_bytes))
            actual=digest(pending)
            attempt=dict(id=item['id'],status='sha_match' if actual==CHECKPOINT_SHA else 'sha_mismatch',sha256=actual)
            report['attempts'].append(attempt)
            if actual==CHECKPOINT_SHA:
                if target.exists():
                    return finish('selection_required',message='Recovery destination appeared during download; preserved it.')
                pending.replace(target)
                return finish('recovered',checkpoint=str(target),checkpoint_sha256=actual,
                    recovered_file=dict(item),reused=False)
        except Exception as error:
            report['attempts'].append(dict(id=item['id'],status='download_error',**safe_error(error)))
            report['errors'].append(dict(phase='download',id=item['id'],trashed=item.get('trashed'),**safe_error(error)))
        finally:pending.unlink(missing_ok=True)
    if listing_failed or any(e['phase']=='download' for e in report['errors']):
        return finish('access_error',message='Some Drive files could not be read. If a listed file is in trash, inspect its Drive link and restore it manually before retrying. No Drive file was changed.')
    if truncated:
        report['search_complete']=False
        return finish('incomplete',message='Search/download limits were reached. Review listed candidates; absence has not been established.')
    return finish('not_found',message='No exact reviewed checkpoint was recovered within this name-prefix search. Arbitrarily renamed files, other accounts and permanently deleted files are outside this result.')
