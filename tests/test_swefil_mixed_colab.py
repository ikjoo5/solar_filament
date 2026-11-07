"""CPU tests of durable source pinning and the single-cell Colab boundary."""
import ast
import importlib.metadata
import importlib.util
import io
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import swefil_mixed_colab as launcher

NOTEBOOK = ROOT/'notebooks/SolarFilament_SWEFil_Mixed_T4.ipynb'


def test_notebook_describes_requested_external_heavy_sampling():
    document = json.loads(NOTEBOOK.read_text())
    text = '\n'.join(''.join(cell['source']) for cell in document['cells'])
    assert 'GONG 30%, SWEFil 60%, CHASE synthetic 10%' in text
    assert 'expected 120/240/40' in text and 'realized counts vary' in text
    assert 'GONG 50%' not in text and 'SWEFil 25%' not in text
    assert 'best_checkpoint_url' in text and 'last_checkpoint_url' in text


def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()


@pytest.fixture
def fixture(tmp_path):
    repository = tmp_path/'remote'
    repository.mkdir()
    for name in launcher.SOURCE_FILES:
        path = repository/name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name in ('tools/swefil_mixed_colab.py', 'tools/chase_shape_colab.py'):
            shutil.copyfile(ROOT/name, path)
        else:
            path.write_text('{}\n' if name.endswith('.json') else '# CPU fixture\n')
    subprocess.run(['git', 'init', '--initial-branch=main', str(repository)], check=True, capture_output=True)
    git(repository, 'config', 'user.name', 'Test')
    git(repository, 'config', 'user.email', 'test@example.invalid')
    git(repository, 'add', '.')
    git(repository, 'commit', '-m', 'CPU fixture')
    base = tmp_path/'drive/MyDrive/filament-segmentation-2026'
    return types.SimpleNamespace(repository=repository, base=base,
        output=base/'fixed_split_experiments'/launcher.SCOPE,
        cache=tmp_path/'cache', commit=git(repository, 'rev-parse', 'HEAD'))


def install(item):
    return launcher.install(item.repository, item.base, item.output, item.cache)


def write_result(output):
    archive = output/'review.zip'
    archive.write_bytes(b'fixture review')
    result = dict(status='paused', epochs_completed=30, output_root=str(output),
                  archive=str(archive), archive_sha256=launcher.sha(archive))
    (output/'latest_result.json').write_text(json.dumps(result))
    return result


def test_source_snapshot_restores_without_git_checkout(fixture):
    config, path, code, state = install(fixture)
    assert state['source_commit'] == fixture.commit
    assert config['drive_checkpoint_path_parts'] == [
        'filament-segmentation-2026', 'fixed_split_experiments', launcher.SCOPE, 'verified_checkpoints']
    assert set(config) == {'base', 'output_root', 'cache_root', 'chase_training_root', 'drive_checkpoint_path_parts'}
    assert 'initial_checkpoint' not in config and 'resume_dir' not in config
    assert not (fixture.output/'verified_checkpoints').exists()
    old = Path(config['chase_training_root'])
    old.mkdir(parents=True)
    old_state = old/'state.json'
    old_state.write_text('{"epoch":51,"best_epoch":42}')
    shutil.rmtree(fixture.repository)
    assert launcher.restore(fixture.output, fixture.base, fixture.cache)[:3] == (config, path, code)
    assert old_state.read_text() == '{"epoch":51,"best_epoch":42}'


@pytest.mark.parametrize('target', ['tools/swefil_mixed_training.py', 'tools/chase_shape_colab.py',
                                   'configs/best84_training_reference.json'])
def test_changed_code_is_not_overwritten(fixture, target):
    _, _, code, _ = install(fixture)
    changed = code/target
    changed.write_text('changed')
    with pytest.raises(ValueError, match='checksum changed'):
        install(fixture)
    assert changed.read_text() == 'changed'


def test_modified_config_is_rejected(fixture):
    _, path, _, _ = install(fixture)
    path.write_text('{}')
    with pytest.raises(ValueError, match='configuration checksum'):
        launcher.restore(fixture.output)


def test_old_experiment_cannot_be_used_as_output(fixture):
    old = fixture.base/'fixed_split_experiments/chase_shape_best84_recipe_t4_v1'
    with pytest.raises(ValueError, match='separate'):
        launcher.install(fixture.repository, fixture.base, old, fixture.cache)
    assert not old.exists()
    with pytest.raises(ValueError, match='local Colab disk'):
        launcher.settings(fixture.base, fixture.output, '/content/drive/MyDrive/cache')


def test_unidentified_checkpoint_output_is_preserved(fixture):
    target = fixture.output/'verified_checkpoints'
    target.mkdir(parents=True)
    with pytest.raises(ValueError, match='no bootstrap identity'):
        install(fixture)
    assert target.exists() and not (fixture.output/'code').exists()


def test_snapshot_symlink_escape_is_rejected(fixture, tmp_path):
    _, _, code, _ = install(fixture)
    path = code/'tools/swefil_training_data.py'
    other = tmp_path/'other.py'
    other.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(other)
    with pytest.raises(ValueError, match='escapes'):
        launcher.restore(fixture.output)


def test_dependency_installer_keeps_cuda_torch_and_numpy(fixture, monkeypatch):
    fixture.output.mkdir(parents=True)
    versions = {'torch': '2.11.0+cu130', 'torchvision': '0.26.0+cu130', 'numpy': '2.1.0',
                'segmentation-models-pytorch': '0.4.0', 'timm': '1.0.20', 'pycocotools': '2.0.11'}
    monkeypatch.setattr(importlib.metadata, 'version', lambda name: versions[name])
    monkeypatch.setattr(importlib.util, 'find_spec', lambda name: object())
    calls = []
    monkeypatch.setattr(subprocess, 'run', lambda args, **kw: calls.append((args, kw)))
    launcher.ensure_dependencies(fixture.output)
    command, options = calls.pop()
    assert options['check'] and 'segmentation-models-pytorch==0.5.0' in command
    assert not any(value.startswith(('torch==', 'torchvision==', 'numpy==')) for value in command)
    assert (fixture.output/'keep_colab_runtime.txt').read_text().splitlines() == [
        'torch==2.11.0+cu130', 'torchvision==0.26.0+cu130', 'numpy==2.1.0']


@pytest.mark.parametrize('phase', ['prepare', 'train', 'all'])
def test_worker_receives_saved_config_and_stage_without_reinitializing_old_run(fixture, monkeypatch, phase):
    config, config_path, code, _ = install(fixture)
    old = Path(config['chase_training_root'])
    old.mkdir(parents=True)
    for name in ('experiment_config.json', 'prepared_summary.json'):
        (old/name).write_text('{}')
    calls = []
    monkeypatch.setattr(launcher, 'ensure_dependencies', lambda *args: None)
    def child(command, log, cwd, env):
        calls.append(command)
        assert cwd == code/'tools' and env['PYTHONUNBUFFERED'] == '1'
        assert env['TORCH_CUDNN_V8_API_DISABLED'] == '1'
        write_result(fixture.output)
    monkeypatch.setattr(launcher, 'monitored', child)
    result = launcher.run(fixture.base, fixture.output, phase, 30, fixture.cache)
    assert len(calls) == 1 and result['epochs_completed'] == 30
    assert calls[0][-6:] == ['--config', str(config_path), '--phase', phase, '--run-until', '30']
    assert calls[0][2] == str(code/'tools/swefil_mixed_training.py')


def test_worker_traceback_is_live_and_logged(tmp_path, capsys):
    command = [sys.executable, '-u', '-c', 'print("epoch progress",flush=True);raise RuntimeError("inner failure")']
    logfile = tmp_path/'training.log'
    with pytest.raises(subprocess.CalledProcessError):
        launcher.monitored(command, logfile, tmp_path, {})
    assert 'epoch progress' in capsys.readouterr().out
    assert 'RuntimeError: inner failure' in logfile.read_text()


def test_interrupt_stops_owned_process_group(tmp_path, monkeypatch):
    killed = []
    class Interrupted(io.StringIO):
        def __next__(self):
            raise KeyboardInterrupt()
    class Process:
        pid = 9876543
        stdout = Interrupted()
        finished = False
        def poll(self):
            return -15 if self.finished else None
        def wait(self, **kwargs):
            self.finished = True
            return -15
    process = Process()
    monkeypatch.setattr(subprocess, 'Popen', lambda *args, **kwargs: process)
    monkeypatch.setattr(launcher.previous.os, 'killpg', lambda pid, sig: killed.append((pid, sig)))
    with pytest.raises(KeyboardInterrupt):
        launcher.monitored(['fixture'], tmp_path/'log', tmp_path, {})
    assert killed == [(process.pid, signal.SIGTERM)] and process.stdout.closed


def test_cli_sigterm_unwinds_the_monitor_instead_of_orphaning_worker(tmp_path, monkeypatch):
    handlers, killed = {}, []
    class Process:
        pid = 9876543
        finished = False
        def poll(self):
            return -15 if self.finished else None
        def wait(self, **kwargs):
            self.finished = True
            return -15
    class Interrupted(io.StringIO):
        def __next__(self):
            handlers[signal.SIGTERM](signal.SIGTERM, None)
    process = Process()
    process.stdout = Interrupted()
    monkeypatch.setattr(launcher.signal, 'signal', lambda signum, handler: handlers.update({signum: handler}))
    monkeypatch.setattr(sys, 'argv', ['swefil_mixed_colab.py'])
    monkeypatch.setattr(subprocess, 'Popen', lambda *args, **kwargs: process)
    monkeypatch.setattr(launcher.os, 'killpg', lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.setattr(launcher, 'run', lambda *args, **kwargs:
                        launcher.monitored(['fixture'], tmp_path/'log', tmp_path, {}))
    with pytest.raises(SystemExit) as stopped:
        launcher.main()
    assert stopped.value.code == 128+signal.SIGTERM
    assert killed == [(process.pid, signal.SIGTERM)] and process.stdout.closed


@pytest.fixture
def notebook(fixture, tmp_path, monkeypatch):
    document = json.loads(NOTEBOOK.read_text())
    cells = [cell for cell in document['cells'] if cell['cell_type'] == 'code']
    assert len(cells) == 1 and cells[0]['outputs'] == [] and cells[0]['execution_count'] is None
    source = ''.join(cells[0]['source'])
    assert 'base64' not in source.lower() and len(source) < 9000
    assert 'RUN_UNTIL_EPOCH = 100' in source and 'auth.authenticate_user()' in source
    checkout = tmp_path/'checkout'
    source = source.replace('/content/drive/MyDrive/filament-segmentation-2026', str(fixture.base))
    source = source.replace('/content/swefil_all_mixed_native_t4_cache', str(fixture.cache))
    source = source.replace('https://github.com/ikjoo5/solar_filament.git', str(fixture.repository))
    source = source.replace('/content/solar_swefil_mixed_github', str(checkout))
    downloads, commands, auth_calls = [], [], []
    colab = types.ModuleType('google.colab')
    colab.drive = types.SimpleNamespace(mount=lambda *args: None)
    colab.auth = types.SimpleNamespace(authenticate_user=lambda: auth_calls.append(True))
    colab.files = types.SimpleNamespace(download=lambda path: downloads.append(path))
    monkeypatch.setitem(sys.modules, 'google.colab', colab)
    real_popen = subprocess.Popen
    def popen(command, **kwargs):
        if len(command) > 2 and command[2].endswith('swefil_mixed_colab.py'):
            commands.append(command)
            assert kwargs['stdout'] == subprocess.PIPE and kwargs['stderr'] == subprocess.STDOUT
            assert kwargs['start_new_session'] and kwargs['text'] and kwargs['bufsize'] == 1
            launcher.install(Path(command[2]).parents[1], fixture.base, fixture.output, fixture.cache)
            write_result(fixture.output)
            return types.SimpleNamespace(wait=lambda **kw: 0, poll=lambda: 0, pid=999999,
                stdout=io.StringIO('Epoch and verified upload progress\n'))
        return real_popen(command, **kwargs)
    monkeypatch.setattr(subprocess, 'Popen', popen)
    return types.SimpleNamespace(fixture=fixture, checkout=checkout, downloads=downloads,
        commands=commands, auth_calls=auth_calls, execute=lambda: exec(compile(source, 'colab_cell', 'exec'), {}))


def test_same_cell_resumes_after_checkout_and_remote_disappear(notebook):
    notebook.execute()
    assert len(notebook.downloads) == 1 and len(notebook.auth_calls) == 1
    shutil.rmtree(notebook.checkout)
    shutil.rmtree(notebook.fixture.repository)
    notebook.execute()
    assert not notebook.checkout.exists() and len(notebook.downloads) == 2
    assert str(notebook.fixture.output/'code') in notebook.commands[-1][2]


def test_notebook_checks_imported_helpers_before_executing_launcher(notebook):
    notebook.execute()
    _, _, code, _ = launcher.restore(notebook.fixture.output)
    (code/'tools/chase_shape_colab.py').write_text('raise RuntimeError("must not run")')
    with pytest.raises(ValueError, match='Saved launcher/source changed'):
        notebook.execute()
    assert len(notebook.commands) == 1


@pytest.mark.parametrize('exit_code', [0, 7])
def test_notebook_relays_actual_child_stdout_and_stderr(capsys, exit_code):
    source = ''.join(next(cell['source'] for cell in json.loads(NOTEBOOK.read_text())['cells'] if cell['cell_type'] == 'code'))
    tree = ast.parse(source)
    index = next(index for index, node in enumerate(tree.body) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == 'process' for target in node.targets))
    block = ast.Module(body=tree.body[index:index+2], type_ignores=[])
    namespace = dict(subprocess=subprocess, os=launcher.os, signal=signal,
        command=[sys.executable, '-u', '-c',
                 'import sys; print("epoch progress",flush=True);'
                 'print("stderr progress",file=sys.stderr,flush=True);sys.exit('+str(exit_code)+')'])
    if exit_code:
        with pytest.raises(subprocess.CalledProcessError):
            exec(compile(block, 'relay', 'exec'), namespace)
    else:
        exec(compile(block, 'relay', 'exec'), namespace)
    captured = capsys.readouterr().out
    assert 'epoch progress' in captured and 'stderr progress' in captured
    assert namespace['process'].stdout.closed
