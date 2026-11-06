"""Exercise the shipped notebook with local Git and a fake GPU child.

No Drive, public network, image dataset or model weights are needed.
"""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks/SolarFilament_GitHub_Colab.ipynb'
OFFICIAL_SHA = 'f9b2272771d3f6666f6ef812f79968a27f3d448ac2f277eda46c854ad38d598f'


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def notebook(tmp_path, monkeypatch):
    document = json.loads(NOTEBOOK.read_text())
    cells = [''.join(c['source']) for c in document['cells'] if c['cell_type'] == 'code']
    assert len(cells) == 3
    for cell in cells:
        compile(cell, str(NOTEBOOK), 'exec')
    assert all(not c.get('outputs') and c.get('execution_count') is None
               for c in document['cells'] if c['cell_type'] == 'code')
    assert OFFICIAL_SHA in cells[1]
    assert OFFICIAL_SHA in (ROOT / 'tools/r1_test_submission.py').read_text()

    source = tmp_path / 'remote'
    source.mkdir()
    subprocess.run(['git', 'init', '--initial-branch=main', str(source)], check=True, capture_output=True)
    git(source, 'config', 'user.name', 'Notebook fixture')
    git(source, 'config', 'user.email', 'fixture@example.invalid')
    (source / 'tools').mkdir()
    (source / '.gitignore').write_text('__pycache__/\n')
    shutil.copy2(ROOT / 'tools/r1_submission_colab_bootstrap.py', source / 'tools/r1_submission_colab_bootstrap.py')
    (source / 'tools/r1_test_submission.py').write_text('# GPU child replaced by test fixture\n')
    git(source, 'add', '.')
    git(source, 'commit', '-m', 'Fixture initial code')
    original_commit = git(source, 'rev-parse', 'HEAD')

    base = tmp_path / 'drive'
    images = base / 'dataset/test_images'
    images.mkdir(parents=True)
    names = [f'fixture_{i:03}.jpeg' for i in range(180)]
    for name in names:
        (images / name).write_bytes(b'not decoded by the notebook wrapper')
    weights = base / 'weights'
    weights.mkdir()
    yolo = weights / 'yolo.pt'
    native = weights / 'native.pt'
    yolo.write_bytes(b'fixture yolo')
    native.write_bytes(b'fixture native')
    output = base / 'submissions/run'
    code = tmp_path / 'checkout'
    calls = dict(downloads=[], deps=[], commands=[], killed=[])
    fake = types.ModuleType('google.colab')
    fake.drive = types.SimpleNamespace(mount=lambda *args: None)
    fake.files = types.SimpleNamespace(download=lambda path: calls['downloads'].append(str(path)))
    monkeypatch.setitem(sys.modules, 'google.colab', fake)
    namespace = {}
    exec(compile(cells[0], 'notebook_setup', 'exec'), namespace)
    namespace.update(BASE=base, TEST_IMAGES=images, YOLO_CHECKPOINT=yolo,
                     NATIVE_CHECKPOINT=native, OUTPUT=output, CODE=code,
                     REPOSITORY=str(source), REF='main')
    fixture_hash = hashlib.sha256(json.dumps(names, sort_keys=True, separators=(',', ':'),
                                              ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    prepare_source = cells[1].replace(OFFICIAL_SHA, fixture_hash)

    @contextmanager
    def stub_dependencies():
        original_spec = importlib.util.spec_from_file_location

        def load(name, location, *args, **kwargs):
            spec = original_spec(name, location, *args, **kwargs)
            if name == 'repo_bootstrap':
                loader = spec.loader

                def execute(module):
                    loader.exec_module(module)
                    module.ensure_dependencies = lambda path: calls['deps'].append(str(path))
                spec.loader = types.SimpleNamespace(create_module=loader.create_module, exec_module=execute)
            return spec

        with monkeypatch.context() as patch:
            patch.setattr(importlib.util, 'spec_from_file_location', load)
            yield

    def prepare():
        with stub_dependencies():
            exec(compile(prepare_source, 'notebook_prepare', 'exec'), namespace)

    def infer(status='complete', exit_code=0, tamper=False):
        def popen(command, **kwargs):
            calls['commands'].append(command)
            assert kwargs['cwd'] == code and kwargs['start_new_session']
            assert kwargs['env']['TORCH_CUDNN_V8_API_DISABLED'] == '1'
            assert command[2] == str(code / 'tools/r1_test_submission.py')
            kwargs['stdout'].write(b'Fixture child progress\n')
            if exit_code:
                kwargs['stdout'].write(b'Traceback: fixture_inner_failure\n')
            elif status == 'partial':
                (output / 'latest_results.json').write_text(json.dumps(dict(status='partial', completed_images=2)))
            else:
                baseline = output / 'submission_pipeline_baseline.csv'
                refined = output / 'submission_pipeline_R1.csv'
                changes = output / 'change_audit.csv'
                archive = output / 'results.zip'
                for path in (baseline, refined, changes):
                    path.write_bytes(b'fixture generated CSV\n')
                archive.write_bytes(b'fixture generated archive')
                (output / 'run_manifest.json').write_text(json.dumps(dict(
                    output_sha256={p.name: sha(p) for p in (baseline, refined, changes)})))
                (output / 'latest_results.json').write_text(json.dumps(dict(status='complete', completed_images=180,
                    baseline_csv=str(baseline), r1_csv=str(refined), archive=str(archive), archive_sha256=sha(archive))))
                if tamper == 'archive':
                    archive.write_bytes(b'modified archive')
                elif tamper:
                    refined.write_bytes(b'modified output')
            return types.SimpleNamespace(pid=987654321, args=command, wait=lambda **kw: exit_code)

        with monkeypatch.context() as patch:
            patch.setattr(subprocess, 'Popen', popen)
            patch.setattr(namespace['os'], 'killpg', lambda pid, sig: calls['killed'].append((pid, sig)))
            exec(compile(cells[2], 'notebook_infer', 'exec'), namespace)

    return types.SimpleNamespace(prepare=prepare, infer=infer, namespace=namespace, calls=calls,
        source=source, code=code, output=output, images=images, original_commit=original_commit,
        before_weights=(yolo.read_bytes(), native.read_bytes()), weights=(yolo, native))


def test_real_git_clone_pin_and_runtime_reset_resume(notebook):
    item = notebook
    item.prepare()
    pin = (item.output / 'source_commit.json').read_bytes()
    assert json.loads(pin)['commit'] == item.original_commit
    assert git(item.code, 'status', '--porcelain') == ''
    (item.source / 'later.txt').write_text('new upstream commit')
    git(item.source, 'add', 'later.txt')
    git(item.source, 'commit', '-m', 'New upstream code')
    item.prepare()
    assert git(item.code, 'rev-parse', 'HEAD') == item.original_commit
    shutil.rmtree(item.code)
    item.prepare()
    assert git(item.code, 'rev-parse', 'HEAD') == item.original_commit
    assert (item.output / 'source_commit.json').read_bytes() == pin
    assert tuple(p.read_bytes() for p in item.weights) == item.before_weights


def test_modified_local_code_and_changed_names_are_rejected(notebook):
    item = notebook
    item.prepare()
    file = item.code / 'tools/r1_test_submission.py'
    original = file.read_bytes()
    file.write_bytes(b'# user edit\n')
    with pytest.raises(ValueError, match='로컬 코드 변경'):
        item.prepare()
    assert file.read_bytes() == b'# user edit\n'
    file.write_bytes(original)
    (item.images / 'extra.jpeg').write_bytes(b'wrong set')
    with pytest.raises(ValueError, match='공식 test180'):
        item.prepare()


def test_partial_then_complete_cli_and_downloads(notebook):
    item = notebook
    item.prepare()
    config = json.loads((item.output / 'github_config.json').read_text())
    assert config['precision'] == 'fp16' and config['native_batch'] == 1
    assert config['expected_images'] == 180 and 'baseline_csv' not in config
    item.namespace['MAX_IMAGES'] = 2
    item.infer(status='partial')
    assert item.calls['downloads'] == []
    assert item.calls['commands'][-1][-2:] == ['--max-images', '2']
    item.namespace['MAX_IMAGES'] = None
    item.prepare()
    item.infer()
    assert len(item.calls['downloads']) == 3
    assert '--max-images' not in item.calls['commands'][-1]
    assert item.calls['killed']


def test_child_failure_log_and_changed_completed_output(notebook, capsys):
    item = notebook
    item.prepare()
    with pytest.raises(subprocess.CalledProcessError):
        item.infer(exit_code=7)
    assert 'fixture_inner_failure' in capsys.readouterr().out
    assert 'fixture_inner_failure' in (item.output / 'github_inference.log').read_text()
    assert item.calls['downloads'] == [] and item.calls['killed']
    with pytest.raises(ValueError, match='완료 파일'):
        item.infer(tamper=True)
    assert item.calls['downloads'] == []


def test_modified_archive_is_rejected_before_any_download(notebook):
    item = notebook
    item.prepare()
    with pytest.raises(ValueError, match='결과 ZIP'):
        item.infer(tamper='archive')
    assert item.calls['downloads'] == []
