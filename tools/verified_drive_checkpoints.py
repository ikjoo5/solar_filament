"""Append-only, server-verified checkpoint storage for Colab training.

Mounted Drive/FUSE files are never evidence of durability. Upload completed local
files, verify their server checksum (or streamed bytes), and only then publish a
verified immutable state record. A crash can leave harmless uncommitted files.
Nothing here deletes, trashes, updates or prunes a remote file.

The state-generation checks detect competing writers; Drive does not provide an
atomic compare-and-swap here. Run only ONE trainer per experiment folder.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import time
import uuid

FOLDER_MIME = 'application/vnd.google-apps.folder'
FIELDS = 'id,name,mimeType,size,sha256Checksum,parents,appProperties,trashed'
MINIMUM_FREE_BYTES = 50_000_000_000
MAX_STATE_BYTES = 2 * 1024 * 1024


class IntegrityError(RuntimeError):
    """Remote metadata/bytes do not prove the requested checkpoint identity."""


class StateConflict(RuntimeError):
    """Another committed generation or writer was detected."""


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def sha256_file(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def make_drive_service():
    """Colab must call google.colab.auth.authenticate_user() first."""
    import google.auth
    from googleapiclient.discovery import build
    credentials, _ = google.auth.default(scopes=['https://www.googleapis.com/auth/drive'])
    return build('drive', 'v3', credentials=credentials, cache_discovery=False)


def upload_file(service, path, metadata):
    from googleapiclient.http import MediaFileUpload
    media = MediaFileUpload(str(path), resumable=True, chunksize=8 * 1024 * 1024)
    request = service.files().create(body=metadata, media_body=media,
                                     fields=FIELDS, supportsAllDrives=True)
    result = None
    while result is None:
        _, result = request.next_chunk(num_retries=3)
    return result


def download_file(service, file_id, stream):
    from googleapiclient.http import MediaIoBaseDownload
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    transfer = MediaIoBaseDownload(stream, request, chunksize=8 * 1024 * 1024)
    done = False
    while not done:
        _, done = transfer.next_chunk(num_retries=3)


def _quote(value):
    return "'" + str(value).replace('\\', '\\\\').replace("'", "\\'") + "'"


def _list(service, query):
    token = None
    while True:
        page = service.files().list(q=query, pageSize=1000, pageToken=token,
            fields='nextPageToken,incompleteSearch,files(' + FIELDS + ')',
            supportsAllDrives=True, includeItemsFromAllDrives=True).execute()
        if page.get('incompleteSearch'):
            raise IntegrityError('Drive returned an incomplete search; no state was selected.')
        yield from page.get('files', [])
        token = page.get('nextPageToken')
        if not token:
            break


def resolve_experiment_folder(service, path_parts, identity, *, create_from=2,
                              visibility_attempts=1, visibility_wait_seconds=2):
    """Resolve an exact My Drive path; create only experiment-specific folders.

    The default requires the project and fixed_split_experiments to EXIST via
    the API. A missing/unsynced base is an error, never a duplicate project.
    The final folder is dedicated exclusively to this identity. Call before
    creating that final folder through the mounted filesystem.

    If a launcher already created the experiment folder through FUSE, pass
    create_from=3 and visibility_attempts=6 to wait briefly for API visibility.
    Exhausted visibility retries raise, rather than creating another folder.
    """
    parts = list(path_parts)
    if not parts or any(not p or p in ('.', '..') or '/' in p for p in parts):
        raise ValueError('Supply nonempty relative Drive path components.')
    parent = 'root'
    for index, name in enumerate(parts):
        query = (f'{_quote(parent)} in parents and trashed = false and '
                 f'name = {_quote(name)} and mimeType = {_quote(FOLDER_MIME)}')
        matches = list(_list(service, query))
        if not matches and index < create_from:
            for _ in range(max(0, int(visibility_attempts) - 1)):
                time.sleep(max(0, float(visibility_wait_seconds)))
                matches = list(_list(service, query))
                if matches:
                    break
        if len(matches) > 1:
            raise StateConflict('Ambiguous same-name Drive folders: ' + '/'.join(parts[:index + 1]))
        if not matches:
            if index < create_from:
                raise FileNotFoundError('Required Drive base folder is not visible through the API: '
                                        + '/'.join(parts[:index + 1]))
            body = {'name': name, 'mimeType': FOLDER_MIME, 'parents': [parent]}
            if index == len(parts) - 1:
                body['appProperties'] = {'run_identity': identity, 'kind': 'checkpoint_folder'}
            item = service.files().create(body=body, fields=FIELDS,
                                          supportsAllDrives=True).execute()
            # Two simultaneous creators can both succeed: never choose silently.
            matches = list(_list(service, query))
            if len(matches) != 1 or matches[0]['id'] != item['id']:
                raise StateConflict('Concurrent or incomplete Drive folder creation; stop other writers.')
        else:
            item = matches[0]
        if index == len(parts) - 1:
            properties = item.get('appProperties', {})
            if properties.get('run_identity') != identity or properties.get('kind') != 'checkpoint_folder':
                raise IntegrityError('Checkpoint folder is not owned by this experiment identity.')
        parent = item['id']
    return parent


class _HashWriter:
    def __init__(self, limit, stream=None):
        self.limit, self.stream, self.size = limit, stream, 0
        self.hash = hashlib.sha256()

    def write(self, data):
        if self.size + len(data) > self.limit:
            raise IntegrityError('Downloaded bytes exceed the declared artifact size.')
        if self.stream is not None:
            written = self.stream.write(data)
            if written != len(data):
                raise OSError('Short local checkpoint write.')
        self.hash.update(data)
        self.size += len(data)
        return len(data)


class DriveCheckpointStore:
    """One append-only experiment; injectable transports permit offline tests."""
    def __init__(self, service, folder_id, identity, local_cache, *,
                 minimum_free_bytes=MINIMUM_FREE_BYTES,
                 uploader=upload_file, downloader=download_file):
        self.service, self.folder_id, self.identity = service, folder_id, identity
        self.local_cache = Path(local_cache)
        if self.local_cache.is_symlink():
            raise ValueError('Checkpoint cache must not be a symlink.')
        self.local_cache.mkdir(parents=True, exist_ok=True)
        self.minimum_free_bytes = int(minimum_free_bytes)
        self.uploader, self.downloader = uploader, downloader
        self.writer_id = uuid.uuid4().hex
        self._verified = set()
        folder = self._get(folder_id)
        if (folder.get('trashed') or folder.get('mimeType') != FOLDER_MIME or
                folder.get('appProperties', {}).get('run_identity') != identity or
                folder.get('appProperties', {}).get('kind') != 'checkpoint_folder'):
            raise IntegrityError('Drive checkpoint folder ownership does not match this experiment.')

    def _get(self, file_id):
        return self.service.files().get(fileId=file_id, fields=FIELDS,
                                       supportsAllDrives=True).execute()

    def check_quota(self, required_free_bytes=None):
        quota = self.service.about().get(fields='storageQuota').execute().get('storageQuota', {})
        required = self.minimum_free_bytes if required_free_bytes is None else int(required_free_bytes)
        # usage includes trash; usageInDrive alone is not sufficient.
        if 'limit' not in quota or 'usage' not in quota:
            raise RuntimeError('Drive did not report a finite quota and usage; capacity cannot be verified.')
        free = int(quota['limit']) - int(quota['usage'])
        if free < required:
            raise RuntimeError(f'Insufficient verified Drive capacity: {free:,} bytes free; '
                               f'{required:,} bytes required. No training state was advanced.')
        return {'limit': int(quota['limit']), 'usage': int(quota['usage']),
                'free': free, 'required': required}

    def preflight(self, *, required_free_bytes=None):
        """Check capacity and API write/readback BEFORE spending GPU time."""
        state = self.latest_state()
        quota = self.check_quota(required_free_bytes)
        probe = self._publish_json({'identity': self.identity, 'writer_id': self.writer_id,
                                    'purpose': 'write_readback_probe'}, 'probe', 0)
        self._read_json(probe)
        return {'quota': quota, 'probe': probe, 'latest_epoch': state['epoch'] if state else 0}

    def _validate_record(self, record, *, kind=None):
        try:
            file_id, checksum, size = record['file_id'], record['sha256'], int(record['size'])
        except (KeyError, TypeError, ValueError) as error:
            raise IntegrityError('Invalid checkpoint record.') from error
        if size < 0 or len(checksum) != 64 or any(c not in '0123456789abcdef' for c in checksum):
            raise IntegrityError('Invalid artifact checksum or size.')
        item = self._get(file_id)
        properties = item.get('appProperties', {})
        if (item.get('trashed') or self.folder_id not in item.get('parents', []) or
                properties.get('run_identity') != self.identity or
                properties.get('sha256') != checksum or int(item.get('size', -1)) != size or
                item.get('name') != record.get('name') or
                (kind is not None and properties.get('kind') != kind)):
            raise IntegrityError('Artifact server identity, size, folder or ownership mismatch.')
        server_sha = item.get('sha256Checksum')
        if server_sha and server_sha != checksum:
            raise IntegrityError('Artifact server SHA-256 mismatch.')
        return item

    def _verify(self, record, *, kind=None):
        item = self._validate_record(record, kind=kind)
        key = (record['file_id'], record['sha256'], int(record['size']))
        if not item.get('sha256Checksum') and key not in self._verified:
            sink = _HashWriter(int(record['size']))
            self.downloader(self.service, record['file_id'], sink)
            if sink.size != int(record['size']) or sink.hash.hexdigest() != record['sha256']:
                raise IntegrityError('Artifact streamed readback SHA-256 or size mismatch.')
        self._verified.add(key)
        return item

    def _upload(self, path, relative_path, kind, epoch):
        path = Path(path)
        if path.is_symlink() or not path.is_file():
            raise ValueError('Upload requires a completed regular local file.')
        logical = PurePosixPath(relative_path)
        if logical.is_absolute() or '..' in logical.parts or not logical.name:
            raise ValueError('Checkpoint path must be relative to this experiment.')
        checksum, size = sha256_file(path), path.stat().st_size
        self.check_quota(size + 16 * 1024 * 1024)
        name = f'epoch{epoch:03d}-{kind}-{uuid.uuid4().hex}-{logical.name}'
        body = {'name': name, 'parents': [self.folder_id], 'appProperties': {
            'run_identity': self.identity, 'kind': kind, 'epoch': str(epoch), 'sha256': checksum}}
        item = self.uploader(self.service, path, body)
        record = {'path': str(logical), 'file_id': item['id'], 'sha256': checksum,
                  'size': size, 'name': name}
        if path.stat().st_size != size or sha256_file(path) != checksum:
            raise IntegrityError('Local checkpoint changed during upload; state was not committed.')
        self._verify(record, kind=kind)
        return record

    def publish_checkpoint(self, local_path, relative_path, metadata=None):
        metadata = metadata or {}
        kind, epoch = metadata.get('kind'), int(metadata.get('epoch', -1))
        if kind not in ('last', 'best') or epoch < 1:
            raise ValueError('Checkpoint metadata requires kind last/best and a positive epoch.')
        return self._upload(local_path, relative_path, kind, epoch)

    def _publish_json(self, value, kind, epoch):
        fd, name = tempfile.mkstemp(prefix='drive-state-', suffix='.json', dir=self.local_cache)
        path = Path(name)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(canonical_bytes(value)); stream.flush(); os.fsync(stream.fileno())
            return self._upload(path, kind + '.json', kind, epoch)
        finally:
            path.unlink(missing_ok=True)

    def _read_json(self, record):
        if int(record['size']) > MAX_STATE_BYTES:
            raise IntegrityError('State metadata exceeds its size limit.')
        self._validate_record(record)
        content = io.BytesIO()
        sink = _HashWriter(int(record['size']), content)
        self.downloader(self.service, record['file_id'], sink)
        if sink.size != int(record['size']) or sink.hash.hexdigest() != record['sha256']:
            raise IntegrityError('State readback hash mismatch.')
        return json.loads(content.getvalue())

    def latest_state(self):
        query = (f'{_quote(self.folder_id)} in parents and trashed = false and '
                 "appProperties has { key='run_identity' and value=" + _quote(self.identity) +
                 " } and appProperties has { key='kind' and value='state' }")
        items = list(_list(self.service, query))
        if not items:
            return None
        by_epoch = {}
        for item in items:
            try:
                epoch = int(item['appProperties']['epoch'])
            except (KeyError, ValueError) as error:
                raise IntegrityError('Malformed remote state generation.') from error
            if epoch < 1 or epoch in by_epoch:
                raise StateConflict('Duplicate/conflicting state generations; stop all experiment writers.')
            by_epoch[epoch] = item
        epoch = max(by_epoch)
        item = by_epoch[epoch]
        record = {'file_id': item['id'], 'sha256': item['appProperties']['sha256'],
                  'size': int(item['size']), 'name': item['name'], 'path': 'state.json'}
        envelope = self._read_json(record)
        payload = envelope.get('payload', {})
        if envelope.get('identity') != self.identity or payload.get('epoch') != epoch:
            raise IntegrityError('Latest state identity or epoch mismatch; refusing an older fallback.')
        previous = envelope.get('previous')
        earlier = [e for e in by_epoch if e < epoch]
        if earlier:
            prev = by_epoch[max(earlier)]
            if not previous or previous.get('file_id') != prev['id'] or previous.get('sha256') != prev['appProperties']['sha256']:
                raise StateConflict('Latest state does not extend the preceding committed generation.')
        elif previous is not None:
            raise StateConflict('Previous committed state is missing.')
        self._verify(payload['last'], kind='last')
        if payload.get('best'):
            self._verify(payload['best'], kind='best')
        return {**payload, '_commit': record}

    def commit_state(self, payload, *, expected_previous_epoch):
        payload = {k: v for k, v in payload.items() if k != '_commit'}
        epoch = int(payload['epoch'])
        if epoch != int(expected_previous_epoch) + 1:
            raise ValueError('Commit must advance exactly one completed epoch.')
        last = self._verify(payload['last'], kind='last')
        if int(last['appProperties']['epoch']) != epoch:
            raise IntegrityError('Last checkpoint belongs to another epoch.')
        if payload.get('best'):
            best = self._verify(payload['best'], kind='best')
            if int(best['appProperties']['epoch']) != int(payload['best_epoch']) or not 1 <= int(payload['best_epoch']) <= epoch:
                raise IntegrityError('Best checkpoint epoch mismatch.')
        current = self.latest_state()
        if (current['epoch'] if current else 0) != int(expected_previous_epoch):
            raise StateConflict('Committed state changed while this epoch was running.')
        previous = current['_commit'] if current else None
        envelope = {'schema_version': 1, 'identity': self.identity, 'payload': payload,
                    'previous': previous, 'writer_id': self.writer_id,
                    'created_utc': datetime.now(timezone.utc).isoformat()}
        record = self._publish_json(envelope, 'state', epoch)
        if self._read_json(record) != envelope:
            raise IntegrityError('Committed state readback differs from the requested payload.')
        # Detect an already visible competitor, without claiming an atomic lock.
        latest = self.latest_state()
        if latest['_commit']['file_id'] != record['file_id']:
            raise StateConflict('Another writer published a state during commit.')
        return latest

    def restore(self, record):
        """Return only hash-verified local bytes, never a FUSE checkpoint path."""
        self._verify(record)
        target = self.local_cache / (record['sha256'] + '.pt')
        if target.is_symlink():
            raise ValueError('Checkpoint cache target must not be a symlink.')
        if target.is_file() and target.stat().st_size == int(record['size']) and sha256_file(target) == record['sha256']:
            return target
        fd, name = tempfile.mkstemp(prefix='checkpoint-', suffix='.pending', dir=self.local_cache)
        pending = Path(name)
        try:
            with os.fdopen(fd, 'wb') as stream:
                sink = _HashWriter(int(record['size']), stream)
                self.downloader(self.service, record['file_id'], sink)
                stream.flush(); os.fsync(stream.fileno())
            if sink.size != int(record['size']) or sink.hash.hexdigest() != record['sha256']:
                raise IntegrityError('Restored checkpoint bytes failed SHA-256 verification.')
            pending.replace(target)
            return target
        finally:
            pending.unlink(missing_ok=True)
