import base64
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import types
import zipfile

import pytest

TOOLS = Path(__file__).resolve().parents[1]/'tools'


def module(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS/(name+'.py'))
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


@pytest.fixture
def builder(): return module('build_chase_independent_colab')


@pytest.fixture
def launcher(): return module('chase_independent_launcher')


def fixture_sources(tmp_path):
    source = tmp_path/'source'; source.mkdir()
    (source/'chase_independent_inference.py').write_text("print('test worker')\n")
    (source/'chase_independent_decode.py').write_text('VALUE = 1\n')
    (source/'chase_independent_launcher.py').write_text(
        "def run_experiment(root, code, expected):\n    return {'archives': []}\n")
    return source


def test_bundle_roundtrip_and_notebook(builder, tmp_path):
    source = fixture_sources(tmp_path)
    report = builder.build(tmp_path/'output', source_dir=source)
    cell = Path(report['copy_cell']).read_text()
    compile(cell, 'cell', 'exec')
    compile(Path(report['resume_cell']).read_text(), 'resume', 'exec')
    namespace = {}
    start = cell.index('BUNDLE_SHA = ')
    end = cell.index('\nraw = gzip.decompress')
    exec(cell[start:end], namespace)
    raw = gzip.decompress(base64.b64decode(namespace['PAYLOAD']))
    assert hashlib.sha256(raw).hexdigest() == report['bundle_sha256']
    bundle = json.loads(raw)
    assert set(bundle['files']) == set(builder.NAMES)
    for name in builder.NAMES:
        assert bundle['files'][name] == (source/name).read_text()
    notebook = json.loads(Path(report['notebook']).read_text())
    assert ''.join(notebook['cells'][0]['source']) == cell
    assert report['cell_bytes'] < 30*1024 and report['resume_bytes'] < 4000


def test_cell_preserves_files_and_rejects_changed_source(builder, tmp_path, monkeypatch):
    source = fixture_sources(tmp_path)
    report = builder.build(tmp_path/'output', source_dir=source)
    root = tmp_path/'drive'/'chase_only_both_alignment_t4_60ep_fullpass_pngfix_v1'
    root.mkdir(parents=True)
    weight = root/'last.pt'; weight.write_bytes(b'checkpoint bytes remain')
    before = weight.read_bytes(), weight.stat().st_mtime_ns
    mount = types.SimpleNamespace(mount=lambda path: None)
    fake = types.ModuleType('google.colab'); fake.drive = mount
    fake.files = types.SimpleNamespace(download=lambda path: None)
    monkeypatch.setitem(sys.modules, 'google.colab', fake)
    monkeypatch.setitem(sys.modules, 'chase_independent_launcher', None)
    cell = Path(report['copy_cell']).read_text().replace(
        '/content/drive/MyDrive/filament-segmentation-2026/fixed_split_experiments/chase_only_both_alignment_t4_60ep_fullpass_pngfix_v1', str(root))
    exec(compile(cell, 'cell', 'exec'), {})
    code = root/'independent_models_v1/code'/report['version']
    times = {path.name:path.stat().st_mtime_ns for path in code.glob('*.py')}
    exec(compile(cell, 'cell', 'exec'), {})
    assert times == {path.name:path.stat().st_mtime_ns for path in code.glob('*.py')}
    assert before == (weight.read_bytes(), weight.stat().st_mtime_ns)
    target = code/'chase_independent_decode.py'
    target.write_text('VALUE = 99\n')
    with pytest.raises(ValueError, match='변경된 파일'):
        exec(compile(cell, 'cell', 'exec'), {})
    assert target.read_text() == 'VALUE = 99\n'


def test_archive_parts_hashes_and_size(launcher, tmp_path):
    entries = []
    for i in range(3):
        path = tmp_path/f'frame{i}.json.gz'; path.write_bytes(bytes([i])*12000)
        entries.append(dict(name=path.name, path=path, bytes=path.stat().st_size, sha256=launcher.digest(path)))
    paths = launcher.package_results(tmp_path, entries, dict(conditions=8, models=4), limit=34000)
    assert len(paths) == 3
    seen = set()
    for path in map(Path, paths):
        assert path.stat().st_size <= 34000
        with zipfile.ZipFile(path) as z:
            manifest = json.loads(z.read('export_manifest.json'))
            assert manifest['parts'] == 3 and len(manifest['files']) == 3
            for name in set(z.namelist())-{'export_manifest.json'}:
                item = next(x for x in entries if x['name'] == name)
                assert hashlib.sha256(z.read(name)).hexdigest() == item['sha256']
                seen.add(name)
    assert seen == {x['name'] for x in entries}


def test_four_workers_produce_eight_conditions(launcher, tmp_path):
    root = tmp_path/launcher.ROOT_NAME; root.mkdir()
    base = root/'independent_models_v1'
    code = base/'code'/'fixture'; code.mkdir(parents=True)
    expected = {}
    for name in ('chase_independent_inference.py', 'chase_independent_decode.py', 'chase_independent_launcher.py'):
        (code/name).write_text('VALUE = 1\n'); expected[name] = launcher.digest(code/name)
    old = root/'native'/'last.pt'; old.parent.mkdir(); old.write_bytes(b'old checkpoint')
    before = old.read_bytes(), old.stat().st_mtime_ns
    called = []
    def worker(command, log, *, cwd, role):
        called.append(role)
        assert command[command.index('--model-role')+1] == role
        assert '--device' in command and '--batch' in command
        assert cwd == code.resolve()
        rows = []
        for mode in ('raw', 'limb'):
            images = {f'202001{n:02d}.jpeg':'a'*64 for n in range(91)}
            protocol = dict(model_role=role, input_mode=mode, source_images=images,
                            helper_sha256=expected['chase_independent_inference.py'],
                            decoder_sha256=expected['chase_independent_decode.py'])
            identity = hashlib.sha256(json.dumps(protocol, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
            folder = base/role/identity; (folder/'frames').mkdir(parents=True)
            metadata = dict(identity=identity, protocol=protocol)
            (folder/'identity.json').write_text(json.dumps(metadata))
            (folder/'predictions.csv').write_text('filament_id,segmentation_rle\n')
            for name in images:
                (folder/'frames'/(Path(name).stem+'.json.gz')).write_bytes(gzip.compress(b'{}'))
            row = dict(model_role=role, input_mode=mode, identity=identity,
                             cache_root=str(folder), metrics_path=str(folder/'metrics.json'),
                             csv_path=str(folder/'predictions.csv'),
                             csv_sha256=launcher.digest(folder/'predictions.csv'),
                             pq=.1, tp_pairs=1, fp=2, fn=3)
            rows.append(row)
            (folder/'metrics.json').write_text(json.dumps(row))
        (base/role/'role_results.json').write_text(json.dumps(dict(model_role=role,status='complete',results=rows)))
    result = launcher.run_experiment(root, code, expected, install_dependencies=False, worker=worker)
    assert called == list(launcher.ROLES)
    summary = json.loads(Path(result['summary']).read_text())
    assert len(summary['results']) == 8 and summary['conditions'] == 8
    assert set(summary['raw_minus_limb_pq']) == set(launcher.ROLES)
    assert before == (old.read_bytes(), old.stat().st_mtime_ns)
    with zipfile.ZipFile(result['archives'][0]) as z:
        assert not any(name.endswith('.pt') for name in z.namelist())


def test_failed_worker_does_not_export_partial_run(launcher, tmp_path):
    root = tmp_path/launcher.ROOT_NAME; root.mkdir()
    code = root/'independent_models_v1/code/fixture'; code.mkdir(parents=True)
    expected = {}
    for name in ('chase_independent_inference.py', 'chase_independent_decode.py', 'chase_independent_launcher.py'):
        (code/name).write_text('VALUE = 1\n'); expected[name] = launcher.digest(code/name)
    def failed(*args, **kwargs): raise RuntimeError('simulated worker failure')
    with pytest.raises(RuntimeError, match='simulated'):
        launcher.run_experiment(root, code, expected, install_dependencies=False, worker=failed)
    assert not (root/'independent_models_v1/exports').exists()


def test_real_child_output_and_failure_are_saved(launcher, tmp_path, capsys):
    log = tmp_path/'inference.log'
    command = [sys.executable, '-u', '-c', "print('\\n'.join('line'+str(i) for i in range(1000)))"]
    launcher.run_worker(command, log, cwd=tmp_path, role='fixture', interval=.01)
    assert 'line999' in log.read_text() and 'line0' in log.read_text()
    assert len(capsys.readouterr().out.splitlines()) < 30
    heartbeat = json.loads(log.with_suffix('.heartbeat.json').read_text())
    assert heartbeat['exit_code'] == 0
    with pytest.raises(subprocess.CalledProcessError):
        launcher.run_worker([sys.executable, '-u', '-c', "raise ValueError('visible failure')"],
                            log, cwd=tmp_path, role='fixture', interval=.01)
    assert 'visible failure' in capsys.readouterr().out
    assert json.loads(log.with_suffix('.heartbeat.json').read_text())['exit_code'] != 0
