"""Offline checks for durable inference snapshots and read-only Drive retrieval."""
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('swefil_test_colab', ROOT/'tools/swefil_test_colab.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()


@pytest.fixture
def layout(tmp_path):
    repository = tmp_path/'repository'
    for name in launcher.SOURCE_FILES:
        path = repository/name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name == 'tools/swefil_test_colab.py':
            shutil.copyfile(ROOT/name, path)
        else:
            path.write_text('# Test source fixture.\n')
    subprocess.run(['git', 'init', '--initial-branch=main', str(repository)], check=True, capture_output=True)
    git(repository, 'config', 'user.name', 'CPU test')
    git(repository, 'config', 'user.email', 'cpu@example.invalid')
    git(repository, 'add', '.')
    git(repository, 'commit', '-m', 'Fixture')
    training = tmp_path/'project/experiments/swefil_training'
    (training/'train').mkdir(parents=True)
    (training/'train/identity.json').write_text(json.dumps({'identity': launcher.TRAINING_IDENTITY}))
    (training/'train/config.json').write_text('{}')
    images = tmp_path/'project/test_images'
    images.mkdir()
    output = tmp_path/'project/submissions'/launcher.SCOPE
    return types.SimpleNamespace(repository=repository, training=training, images=images,
                                 output=output, cache=tmp_path/'local_cache')


def install(item, **kwargs):
    return launcher.install(item.repository, item.training, item.images, item.output, **kwargs)


def result_file(output, **changes):
    csv = output/'submission.csv'
    csv.write_text('filament_id,segmentation_rle\n')
    archive = output/'review.zip'
    archive.write_bytes(b'review fixture')
    result = dict(status='complete', images_completed=180, checkpoint_epoch=100,
                  checkpoint_sha256=launcher.CHECKPOINT_SHA, output_root=str(output),
                  csv=str(csv), csv_sha256=launcher.sha(csv), archive=str(archive),
                  archive_sha256=launcher.sha(archive))
    result.update(changes)
    (output/'latest_result.json').write_text(json.dumps(result))
    return result


class FakeDrive:
    """Only the read endpoint exists: uploads, quota checks and searches fail."""
    def __init__(self, item, data):
        self.item, self.data, self.calls = item, data, []

    def files(self):
        return self

    def get(self, **kwargs):
        self.calls.append(('get', kwargs))
        return types.SimpleNamespace(execute=lambda: copy.deepcopy(self.item))

    def download(self, service, file_id, stream):
        assert service is self and file_id == self.item['id']
        self.calls.append(('download', file_id))
        for start in range(0, len(self.data), 7):
            stream.write(self.data[start:start+7])


@pytest.fixture
def drive(layout, monkeypatch):
    data = b'Fixture reviewed epoch100 model bytes; no torch pickle needed.'
    digest = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(launcher, 'CHECKPOINT_SHA', digest)
    monkeypatch.setattr(launcher, 'CHECKPOINT_SIZE', len(data))
    record = dict(file_id='fixture-file-id', name='epoch100-best-fixture.pt',
                  path='checkpoints/best_epoch100.pt', sha256=digest, size=len(data))
    summary = dict(identity=launcher.TRAINING_IDENTITY, best_epoch=100,
                   best_checkpoint_record=record)
    (layout.training/'train/summary.json').write_text(json.dumps(summary))
    properties = dict(run_identity=launcher.TRAINING_IDENTITY, kind='best', epoch='100', sha256=digest)
    item = dict(id=record['file_id'], name=record['name'], size=str(len(data)),
                mimeType='application/octet-stream', trashed=False,
                sha256Checksum=digest, appProperties=properties)
    return FakeDrive(item, data)


def download(layout, drive, **kwargs):
    return launcher.download_checkpoint(layout.training, cache_root=layout.cache,
        service=drive, downloader=drive.download, **kwargs)


def test_read_only_download_verifies_and_atomically_caches_bytes(layout, drive):
    before = {p.name: p.read_bytes() for p in (layout.training/'train').iterdir()}
    target = download(layout, drive)
    assert target == layout.cache/(launcher.CHECKPOINT_SHA+'.pt')
    assert target.read_bytes() == drive.data and len(drive.calls) == 2
    assert drive.calls[0][0] == 'get' and drive.calls[1][0] == 'download'
    assert drive.calls[0][1]['supportsAllDrives']
    assert not list(layout.cache.glob('*.pending'))
    assert before == {p.name: p.read_bytes() for p in (layout.training/'train').iterdir()}


def test_hash_verified_cache_survives_missing_metadata_and_avoids_network(layout, drive):
    target = download(layout, drive)
    (layout.training/'train/summary.json').unlink()
    drive.calls.clear()
    assert download(layout, drive) == target
    assert drive.calls == []
    target.write_bytes(b'bad cache')
    with pytest.raises(ValueError, match='No matching reviewed'):
        download(layout, drive)
    assert drive.calls == [] and target.read_bytes() == b'bad cache'


@pytest.mark.parametrize('bad_bytes', [b'wrong bytes', b'', b'x'*80])
def test_streamed_bad_bytes_never_become_cached(layout, drive, bad_bytes):
    drive.data = bad_bytes
    with pytest.raises(ValueError, match='size|SHA-256'):
        download(layout, drive)
    assert not list(layout.cache.iterdir())


def test_stream_checksum_is_checked_when_length_matches_server(layout, drive):
    drive.data = b'x'*launcher.CHECKPOINT_SIZE
    with pytest.raises(ValueError, match='SHA-256'):
        download(layout, drive)
    assert not list(layout.cache.iterdir())


@pytest.mark.parametrize('change', [
    {'trashed': True}, {'trashed': None}, {'size': '1'}, {'size': 'bad'},
    {'id': 'different'}, {'sha256Checksum': 'f'*64}, {'name': 'different.pt'},
    {'mimeType': 'application/vnd.google-apps.folder'}, {'appProperties': {}},
    {'appProperties': {'epoch': '98'}},
])
def test_remote_metadata_refused_before_download(layout, drive, change):
    drive.item.update(change)
    with pytest.raises(ValueError, match='Drive checkpoint'):
        download(layout, drive)
    assert [call[0] for call in drive.calls] == ['get']
    assert not list(layout.cache.iterdir())


def test_missing_server_sha_still_uses_actual_stream_hash(layout, drive):
    del drive.item['sha256Checksum']
    assert download(layout, drive).read_bytes() == drive.data


def test_explicit_file_id_can_recover_stale_mirrors_and_copied_metadata(layout, drive):
    (layout.training/'train/summary.json').write_text('{}')
    drive.item.pop('appProperties')
    drive.item.pop('sha256Checksum')
    assert download(layout, drive, checkpoint_file_id=drive.item['id']).read_bytes() == drive.data


def test_explicit_file_id_cannot_override_conflicting_app_properties(layout, drive):
    drive.item['appProperties']['run_identity'] = 'unrelated-training'
    with pytest.raises(ValueError, match='appProperties'):
        download(layout, drive, checkpoint_file_id=drive.item['id'])
    assert [call[0] for call in drive.calls] == ['get']


def test_matching_state_record_is_used_when_summary_is_stale(layout, drive):
    path = layout.training/'train/summary.json'
    summary = json.loads(path.read_text())
    record = summary['best_checkpoint_record']
    summary['best_epoch'] = 98
    path.write_text(json.dumps(summary))
    (layout.training/'train/state.json').write_text(json.dumps(dict(identity=launcher.TRAINING_IDENTITY,
        payload=dict(best_epoch=100, best=record))))
    assert download(layout, drive).read_bytes() == drive.data


def test_local_override_is_hash_verified_without_drive(layout, drive):
    checkpoint = layout.training.parent/'downloaded.pt'
    checkpoint.write_bytes(drive.data)
    assert download(layout, drive, checkpoint=checkpoint) == checkpoint
    assert not drive.calls and not layout.cache.exists()
    checkpoint.write_bytes(b'x'*launcher.CHECKPOINT_SIZE)
    with pytest.raises(ValueError, match='SHA-256'):
        download(layout, drive, checkpoint=checkpoint)


def test_interrupted_download_cleans_partial_and_retry_succeeds(layout, drive):
    def interrupted(service, file_id, stream):
        stream.write(drive.data[:5])
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        launcher.download_checkpoint(layout.training, cache_root=layout.cache,
            service=drive, downloader=interrupted)
    assert not list(layout.cache.iterdir())
    assert download(layout, drive).read_bytes() == drive.data


def test_snapshot_survives_checkout_loss_and_retains_input_bytes(layout):
    before = {p.name: p.read_bytes() for p in (layout.training/'train').iterdir()}
    config, code, state = install(layout)
    assert config['inference_only'] and config['native_only']
    assert config['expected_epoch'] == 100 and config['expected_images'] == 180
    assert config['checkpoint_file_id'] is None
    assert config['checkpoint_sha256'] == launcher.CHECKPOINT_SHA
    assert state['source_commit'] == git(layout.repository, 'rev-parse', 'HEAD')
    assert not any('synthesis' in name or 'swefil_mixed_training.py' in name for name in launcher.SOURCE_FILES)
    assert all((code/name).read_bytes() == (layout.repository/name).read_bytes() for name in launcher.SOURCE_FILES)
    shutil.rmtree(layout.repository)
    assert launcher.restore(layout.output, layout.training, layout.images) == (config, code, state)
    assert before == {p.name: p.read_bytes() for p in (layout.training/'train').iterdir()}


@pytest.mark.parametrize('name', ['tools/swefil_test_inference.py', 'tools/verified_drive_checkpoints.py'])
def test_saved_source_changes_are_rejected(layout, name):
    _, code, _ = install(layout)
    (code/name).write_text('# Changed source.\n')
    with pytest.raises(ValueError, match='checksum differs'):
        install(layout)


def test_saved_settings_and_output_boundaries_cannot_drift(layout):
    install(layout)
    with pytest.raises(ValueError, match='paths/settings changed'):
        launcher.restore(layout.output, layout.training/'different', layout.images)
    with pytest.raises(ValueError, match='separate'):
        launcher.settings(layout.training, layout.images, layout.training/'predictions')
    path = layout.output/'inference_launcher_config.json'
    path.write_text('{}')
    with pytest.raises(ValueError, match='settings checksum'):
        launcher.restore(layout.output)


def test_result_requires_all_images_exact_checkpoint_safe_paths_and_hashes(layout, tmp_path):
    install(layout)
    result = result_file(layout.output)
    assert launcher.checked_result(layout.output) == result
    for changes in ({'status': 'partial'}, {'images_completed': 179},
                    {'checkpoint_epoch': 98}, {'checkpoint_sha256': '0'*64}):
        result_file(layout.output, **changes)
        with pytest.raises((ValueError, RuntimeError)):
            launcher.checked_result(layout.output)
    result_file(layout.output)
    Path(result['csv']).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        launcher.checked_result(layout.output)
    outside = tmp_path/'elsewhere.zip'
    outside.write_bytes(b'outside')
    result_file(layout.output, archive=str(outside), archive_sha256=launcher.sha(outside))
    with pytest.raises(ValueError, match='unsafe'):
        launcher.checked_result(layout.output)


def test_restore_run_uses_new_locator_without_rewriting_snapshot(layout, monkeypatch):
    config, code, state = install(layout, checkpoint_file_id='old-stale-id')
    before = (layout.output/'inference_launcher_config.json').read_bytes()
    monkeypatch.setattr(launcher, 'ensure_dependencies', lambda output: None)
    selections, commands = [], []
    checkpoint = layout.cache/'selected.pt'
    def select(training, local, file_id):
        selections.append((training, local, file_id))
        return checkpoint
    def child(command, log, cwd, env):
        commands.append(command)
        assert cwd == code/'tools' and env['TORCH_CUDNN_V8_API_DISABLED'] == '1'
        assert env['PYTHONUNBUFFERED'] == '1' and log == layout.output/'inference.log'
        result_file(layout.output)
    monkeypatch.setattr(launcher, 'download_checkpoint', select)
    monkeypatch.setattr(launcher, 'monitored', child)
    shutil.rmtree(layout.repository)
    result = launcher.run(layout.training, layout.images, layout.output, checkpoint_file_id='new-id')
    assert selections == [(layout.training, None, 'new-id')]
    assert commands[0][2] == str(code/'tools/swefil_test_inference.py')
    assert commands[0][-2:] == ['--checkpoint', str(checkpoint)]
    assert '--phase' not in commands[0] and '--run-until' not in commands[0]
    assert result['images_completed'] == 180
    assert (layout.output/'inference_launcher_config.json').read_bytes() == before
    assert launcher.restore(layout.output) == (config, code, state)


def test_installer_preserves_provided_cuda_and_numpy(layout, monkeypatch):
    install(layout)
    versions = {'torch': '2.14.1+cu130', 'torchvision': '0.27.0+cu130', 'numpy': '2.4.6',
                'segmentation-models-pytorch': '0.4.0', 'timm': '1.0.20', 'pycocotools': '2.0.11'}
    monkeypatch.setattr(importlib.metadata, 'version', lambda name: versions[name])
    monkeypatch.setattr(importlib.util, 'find_spec', lambda name: object())
    calls = []
    monkeypatch.setattr(subprocess, 'run', lambda command, **kwargs: calls.append((command, kwargs)))
    launcher.ensure_dependencies(layout.output)
    command, kwargs = calls.pop()
    assert kwargs['check'] and 'segmentation-models-pytorch==0.5.0' in command
    assert not any(value.startswith(('torch==', 'torchvision==', 'ultralytics')) for value in command)
    assert (layout.output/'keep_colab_runtime.txt').read_text().splitlines() == [
        'torch==2.14.1+cu130', 'torchvision==0.27.0+cu130', 'numpy==2.4.6']


def test_worker_failure_reaches_console_and_log(tmp_path, capsys):
    path = tmp_path/'inference.log'
    command = [sys.executable, '-u', '-c', 'print("image progress",flush=True); raise RuntimeError("fixture failure")']
    with pytest.raises(subprocess.CalledProcessError):
        launcher.monitored(command, path, tmp_path, {})
    assert 'RuntimeError: fixture failure' in path.read_text()
    assert 'fixture failure' in capsys.readouterr().out


def test_interrupt_stops_owned_process_group(tmp_path, monkeypatch):
    killed = []
    class Process:
        pid = 99999999
        calls = 0
        def poll(self):
            return None if self.calls < 2 else -15
        def wait(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                assert kwargs['timeout'] == 30
                raise KeyboardInterrupt()
            return -15
    process = Process()
    monkeypatch.setattr(subprocess, 'Popen', lambda *args, **kwargs: process)
    monkeypatch.setattr(launcher.os, 'killpg', lambda pid, sig: killed.append((pid, sig)))
    with pytest.raises(KeyboardInterrupt):
        launcher.monitored(['fixture'], tmp_path/'log', tmp_path, {})
    assert killed == [(process.pid, signal.SIGTERM)]
