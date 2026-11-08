"""Bounded, server-verified checkpoints for a new full-pass training run.

Small immutable completed-epoch state records are retained to preserve the audit
chain. Only checkpoints referenced by the latest two completed epochs and the
latest two within-epoch progress commits are retained. Pruning runs *after* a
new commit has passed remote readback, and touches only committed records owned
by this exact identity and checkpoint folder. Uncommitted uploads and other
experiments are never swept. Use one trainer per folder.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import verified_drive_checkpoints as durable


class BoundedDriveCheckpointStore(durable.DriveCheckpointStore):
    """A bounded variant intended exclusively for the new full-pass run."""

    def __init__(self, *args, minimum_free_bytes=2_000_000_000, **kwargs):
        super().__init__(*args, minimum_free_bytes=minimum_free_bytes, **kwargs)

    def _items(self, kind=None):
        query = (f'{durable._quote(self.folder_id)} in parents and trashed = false and '
                 "appProperties has { key='run_identity' and value=" +
                 durable._quote(self.identity) + " }")
        if kind is not None:
            query += " and appProperties has { key='kind' and value=" + durable._quote(kind) + " }"
        return list(durable._list(self.service, query))

    @staticmethod
    def _record(item, path):
        try:
            return dict(file_id=item['id'], sha256=item['appProperties']['sha256'],
                        size=int(item['size']), name=item['name'], path=path)
        except (KeyError, TypeError, ValueError) as error:
            raise durable.IntegrityError('Malformed owned remote record.') from error

    def _state_payloads(self):
        records = []
        seen = set()
        for item in self._items('state'):
            record = self._record(item, 'state.json')
            value = self._read_json(record)
            epoch = int(item['appProperties']['epoch'])
            payload = value.get('payload', {})
            if (value.get('identity') != self.identity or payload.get('epoch') != epoch or epoch < 1):
                raise durable.IntegrityError('Invalid state encountered during retention audit.')
            if epoch in seen:
                raise durable.StateConflict('Duplicate state generation during retention audit.')
            seen.add(epoch)
            records.append((epoch, payload))
        return sorted(records)

    def _remove_committed_records(self, records, *, keep_ids=(), allowed_kinds):
        """Remove only listed, verified, committed records; never glob-delete."""
        keep = set(keep_ids)
        available = {item['id']: item for item in self._items()}
        deleted = []
        seen = set()
        for record in records:
            file_id = record['file_id']
            if file_id in seen or file_id in keep or file_id not in available:
                continue
            seen.add(file_id)
            # Re-fetch immediately before deletion; parent, owner, name, size and
            # checksum must still equal the immutable record from a commit.
            item = self._verify(record)
            kind = item.get('appProperties', {}).get('kind')
            if kind not in allowed_kinds:
                raise durable.IntegrityError('Refusing to prune an unexpected artifact kind.')
            self.service.files().delete(fileId=file_id, supportsAllDrives=True).execute()
            deleted.append(file_id)
            cached = self.local_cache / (record['sha256'] + '.pt')
            if cached.is_symlink():
                raise ValueError('Checkpoint cache target must not be a symlink.')
            cached.unlink(missing_ok=True)
        return deleted

    def _prune_completed(self, committed):
        # All epoch JSONs remain: the base store can still verify the preceding
        # generation and will never fall back to an old pruned checkpoint.
        states = self._state_payloads()
        if not states or states[-1][0] != committed['epoch']:
            raise durable.StateConflict('State changed before checkpoint retention.')
        kept = [record for _, payload in states[-2:]
                for key in ('last', 'best') if (record := payload.get(key))]
        for record in kept:
            self._verify(record)
        all_records = [record for _, payload in states
                       for key in ('last', 'best') if (record := payload.get(key))]
        return self._remove_committed_records(all_records,
            keep_ids=[record['file_id'] for record in kept], allowed_kinds={'last', 'best'})

    def commit_state(self, payload, *, expected_previous_epoch):
        committed = super().commit_state(payload, expected_previous_epoch=expected_previous_epoch)
        self._prune_completed(committed)
        return committed

    def _progress_commits(self):
        commits = []
        generations = set()
        for item in self._items('progress_state'):
            record = self._record(item, 'progress_state.json')
            value = self._read_json(record)
            payload = value.get('payload', {})
            try:
                epoch, step, base_epoch = (int(payload[key]) for key in ('epoch', 'step', 'base_epoch'))
            except (KeyError, TypeError, ValueError) as error:
                raise durable.IntegrityError('Malformed progress generation.') from error
            if (value.get('identity') != self.identity or epoch < 1 or step < 1 or
                    base_epoch != epoch - 1 or int(item['appProperties']['epoch']) != epoch):
                raise durable.IntegrityError('Progress identity, epoch or cursor mismatch.')
            generation = (epoch, step)
            if generation in generations:
                raise durable.StateConflict('Duplicate progress generations; stop concurrent writers.')
            generations.add(generation)
            if not isinstance(payload.get('checkpoint'), dict):
                raise durable.IntegrityError('Progress checkpoint record is missing.')
            commits.append({**payload, '_commit': record})
        return sorted(commits, key=lambda value: (value['epoch'], value['step']))

    def latest_progress(self):
        state = self.latest_state()
        completed = int(state['epoch']) if state else 0
        active = [p for p in self._progress_commits() if p['epoch'] > completed]
        if not active:
            return None
        if any(p['base_epoch'] != completed or p['epoch'] != completed + 1 for p in active):
            raise durable.StateConflict('Progress does not extend the last completed epoch.')
        latest = active[-1]
        item = self._verify(latest['checkpoint'], kind='progress')
        if int(item['appProperties']['epoch']) != latest['epoch']:
            raise durable.IntegrityError('Progress weights belong to another epoch.')
        return latest

    def publish_progress(self, local_path, metadata=None):
        metadata = metadata or {}
        epoch, step, base_epoch = (int(metadata.get(key, -1))
                                   for key in ('epoch', 'step', 'base_epoch'))
        if epoch < 1 or step < 1 or base_epoch != epoch - 1:
            raise ValueError('Progress requires epoch=base_epoch+1 and positive completed step.')
        state = self.latest_state()
        completed = int(state['epoch']) if state else 0
        if completed != base_epoch:
            raise durable.StateConflict('Completed epoch changed before progress upload.')
        previous = self.latest_progress()
        if previous and (previous['epoch'] != epoch or previous['step'] >= step):
            raise durable.StateConflict('Progress must advance the within-epoch cursor.')
        checkpoint = self._upload(local_path,
            f'progress/epoch{epoch:03d}_step{step:06d}.pt', 'progress', epoch)
        payload = dict(epoch=epoch, step=step, base_epoch=base_epoch, checkpoint=checkpoint)
        state_after = self.latest_state()
        if (int(state_after['epoch']) if state_after else 0) != base_epoch:
            raise durable.StateConflict('Completed epoch changed while uploading progress.')
        envelope = dict(schema_version=1, identity=self.identity, payload=payload,
                        writer_id=self.writer_id,
                        created_utc=datetime.now(timezone.utc).isoformat())
        record = self._publish_json(envelope, 'progress_state', epoch)
        if self._read_json(record) != envelope:
            raise durable.IntegrityError('Progress commit readback differs from requested payload.')
        latest = self.latest_progress()
        if not latest or latest['_commit']['file_id'] != record['file_id']:
            raise durable.StateConflict('Another writer published progress during commit.')
        commits = self._progress_commits()
        current = [p for p in commits if p['epoch'] == epoch]
        keep = {r['file_id'] for p in current[-2:] for r in (p['checkpoint'], p['_commit'])}
        records = [r for p in commits for r in (p['checkpoint'], p['_commit'])]
        # Remove metadata before old checkpoint bytes. An interruption can leave
        # unused bytes, but never a visible progress commit pointing at deletion.
        records.sort(key=lambda r: r['path'] != 'progress_state.json')
        self._remove_committed_records(records, keep_ids=keep,
                                       allowed_kinds={'progress', 'progress_state'})
        return latest

    def clear_progress(self, *, completed_epoch):
        completed_epoch = int(completed_epoch)
        state = self.latest_state()
        if not state or completed_epoch > int(state['epoch']) or completed_epoch < 0:
            raise durable.StateConflict('Progress can be cleared only after a completed epoch commit.')
        commits = [p for p in self._progress_commits() if p['epoch'] <= completed_epoch]
        records = [p['_commit'] for p in commits] + [p['checkpoint'] for p in commits]
        return self._remove_committed_records(records,
                                             allowed_kinds={'progress', 'progress_state'})


# Explicit name for callers which prefer the generic store interface.
DriveCheckpointStore = BoundedDriveCheckpointStore
