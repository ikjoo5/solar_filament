"""The small launcher must preserve training identity and checkpoint bytes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'tools'))
import chase_runtime_monitor as monitor


def restored(tmp_path, monkeypatch):
    root = tmp_path/'fixed_split_experiments'/monitor.ROOT_NAME
    code, assets = root/'code'/monitor.VERSION, root/'assets'/monitor.VERSION
    source_values = {'chase_both_colab_bootstrap.py': b'# pinned fixture\n'}
    asset_values = {'train_split.csv': b'image_id,file_name\n1,example.jpeg\n'}
    hashes = []
    for folder, values in ((code, source_values), (assets, asset_values)):
        folder.mkdir(parents=True)
        for name, value in values.items(): (folder/name).write_bytes(value)
        hashes.append({name:hashlib.sha256(value).hexdigest() for name,value in values.items()})
    monkeypatch.setattr(monitor, 'SOURCES', hashes[0]); monkeypatch.setattr(monitor, 'ASSETS', hashes[1])
    (code/'source_manifest.json').write_text(json.dumps(dict(sources=hashes[0], assets=hashes[1])))
    return root, code


def test_missing_settings_restored_only_after_exact_source_validation(tmp_path, monkeypatch):
    root, code = restored(tmp_path, monkeypatch)
    checkpoint = root/'native/last.pt'; checkpoint.parent.mkdir(); checkpoint.write_bytes(b'opaque checkpoint')
    settings, returned = monitor.validate_run(root)
    assert returned == code
    assert settings == monitor.canonical_settings(root)
    first = (root/'bootstrap_settings.json').read_bytes()
    assert monitor.validate_run(root)[0] == settings
    assert (root/'bootstrap_settings.json').read_bytes() == first
    assert checkpoint.read_bytes() == b'opaque checkpoint'
    # Adding monitor code here would alter bootstrap's own source identity.
    (code/'accidental_monitor.py').write_text('pass')
    with pytest.raises(ValueError, match='Unexpected Python'):
        monitor.validate_run(root)


def test_corrupt_source_does_not_create_settings_or_touch_checkpoint(tmp_path, monkeypatch):
    root, code = restored(tmp_path, monkeypatch)
    (code/'chase_both_colab_bootstrap.py').write_text('changed')
    with pytest.raises(ValueError, match='변경'):
        monitor.validate_run(root)
    assert not (root/'bootstrap_settings.json').exists()


def test_existing_custom_input_paths_are_preserved(tmp_path, monkeypatch):
    root, _ = restored(tmp_path, monkeypatch)
    settings = monitor.canonical_settings(root)
    settings['images'] = '/content/custom/data/train_images'
    path = root/'bootstrap_settings.json'; path.write_text(json.dumps(settings))
    original = path.read_bytes()
    assert monitor.validate_run(root)[0] == settings
    assert path.read_bytes() == original
    settings['assets'] = '/content/unrelated'
    path.write_text(json.dumps(settings))
    with pytest.raises(ValueError, match='different experiment'):
        monitor.validate_run(root)


def test_real_subprocess_logging_is_complete_but_display_is_bounded(tmp_path, capsys):
    root = tmp_path/'run'; root.mkdir()
    program = "import time\nfor i in range(300): print('batch',i,flush=True)\ntime.sleep(.15)\n"
    calls = []
    def snapshot(pid, **kwargs):
        calls.append((pid, kwargs))
        return dict(time_utc='fixture', available_ram_bytes=2*1024**3,
                    total_ram_bytes=4*1024**3, boot_id='test-boot', uptime_seconds=100,
                    gpu_mib_used_total_util_percent='1, 16, 0')
    code = monitor.monitored_command([sys.executable, '-u', '-c', program], root,
        cwd=tmp_path, environment=os.environ.copy(), interval=.02, screen_interval=30,
        snapshot=snapshot)
    assert code == 0 and len(calls) >= 2
    text = (root/'training.log').read_text()
    assert all(f'batch {i}\n' in text for i in range(300))
    # Hundreds of tqdm-like updates are not all rendered in the browser.
    assert len(capsys.readouterr().out.splitlines()) <= 2
    latest = json.loads((root/'diagnostics/latest_heartbeat.json').read_text())
    assert latest['exit_code'] == 0 and latest['last_output'] == 'batch 299'
    assert latest['boot_id'] == 'test-boot'
    assert len((root/'diagnostics/runtime_resources.jsonl').read_text().splitlines()) >= 2


def test_failed_subprocess_retains_error_and_heartbeat(tmp_path):
    root = tmp_path/'run'; root.mkdir()
    def snapshot(pid, **kwargs): return dict(available_ram_bytes=1024, boot_id='fixture')
    with pytest.raises(subprocess.CalledProcessError) as error:
        monitor.monitored_command([sys.executable, '-c', "print('ValueError: fixture');raise SystemExit(3)"],
            root, cwd=tmp_path, environment=os.environ.copy(), snapshot=snapshot)
    assert error.value.returncode == 3
    assert 'ValueError: fixture' in (root/'training.log').read_text()
    assert json.loads((root/'diagnostics/latest_heartbeat.json').read_text())['exit_code'] == 3


def test_resume_uses_original_code_settings_and_keeps_checkpoint(tmp_path, monkeypatch):
    root, code = restored(tmp_path, monkeypatch)
    monitor.validate_run(root)
    history_path = root/'native/history.json'; history_path.parent.mkdir()
    history_path.write_text(json.dumps([dict(epoch=11)]))
    checkpoint = root/'native/last.pt'; checkpoint.write_bytes(b'no torch.load permitted')
    original_settings = (root/'bootstrap_settings.json').read_bytes()
    captured = []
    def command(args, run_root, **kwargs):
        captured.append((args, run_root, kwargs))
        (root/'latest_result.json').write_text(json.dumps(dict(archive='result.zip', status='complete')))
        return 0
    monkeypatch.setattr(monitor, 'monitored_command', command)
    result = monitor.run_resume(root, install_dependencies=False)
    args, selected, kwargs = captured[0]
    assert selected == root and kwargs['cwd'] == code
    assert args[2] == str(code/'chase_both_colab_bootstrap.py')
    assert args[args.index('--run-until')+1] == '60'
    assert args[args.index('--arms')+1] == 'yolo,native'
    assert args[args.index('--phase')+1] == 'all'
    assert (root/'bootstrap_settings.json').read_bytes() == original_settings
    assert checkpoint.read_bytes() == b'no torch.load permitted'
    assert json.loads(history_path.read_text()) == [dict(epoch=11)]
    assert result['archive'] == 'result.zip'


def test_existing_orchestrator_prevents_second_launch(tmp_path, monkeypatch):
    root, _ = restored(tmp_path, monkeypatch)
    with (root/'orchestrator.lock').open('a+') as held:
        monitor.fcntl.flock(held, monitor.fcntl.LOCK_EX | monitor.fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='아직 실행'):
            monitor.run_resume(root, install_dependencies=False)


def test_resource_observation_does_not_require_cuda_or_psutil():
    snapshot = monitor.resource_snapshot(disk_path='/tmp')
    assert snapshot['available_ram_bytes'] > 0
    assert snapshot['total_ram_bytes'] >= snapshot['available_ram_bytes']
    assert snapshot['boot_id'] and snapshot['uptime_seconds'] > 0
    assert snapshot['local_disk_free_bytes'] >= 0
