import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import uuid
import zipfile

import pytest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
spec = importlib.util.spec_from_file_location('plain_builder', TOOLS / 'build_chase_independent_plain_colab.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_plain_source_roundtrip(tmp_path):
    result = builder.build(tmp_path)
    cell = Path(result['copy_cell']).read_text()
    tree = ast.parse(cell)
    literals = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == 'WORKER'}
    source = (TOOLS / 'chase_independent_plain_worker.py').read_text()
    assert literals['WORKER'] == source
    assert hashlib.sha256(source.encode()).hexdigest() == result['worker_sha256']
    assert len(cell.encode()) < 24 * 1024
    assert 'base64' not in cell
    compile(cell, 'cell', 'exec')
    notebook = json.loads((tmp_path / (builder.STEM + '_OneCell.ipynb')).read_text())
    assert ''.join(notebook['cells'][0]['source']) == cell


def test_plain_export_eight_conditions_split_and_integrity(tmp_path):
    base = tmp_path / 'independent_models_plain_v1'
    script = base / 'code' / 'worker.py'
    script.parent.mkdir(parents=True)
    script.write_text('print("mock")\n')
    digest = hashlib.sha256(script.read_bytes()).hexdigest()
    ns = dict(Path=Path, json=json, hashlib=hashlib, datetime=datetime, timezone=timezone,
              uuid=uuid, zipfile=zipfile, BASE=base, SCRIPT=script, WORKER_SHA=digest)
    tree = ast.parse(builder.END)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    exec(compile(ast.Module(body=functions, type_ignores=[]), 'functions', 'exec'), ns)
    roles = ('old_yolo', 'new_yolo', 'old_native', 'new_native')
    weight = tmp_path / 'last.pt'
    weight.write_bytes(b'unchanged original weight')
    before = (weight.read_bytes(), weight.stat().st_mtime_ns)
    for role in roles:
        rows = []
        for mode in ('raw', 'limb'):
            protocol = dict(helper_sha256=digest, model_role=role, input_mode=mode,
                            source_images={f'image_{i:03}.jpeg': 'mockhash' for i in range(91)})
            identity = hashlib.sha256(json.dumps(protocol, ensure_ascii=False,
                                                  separators=(',', ':'), allow_nan=False).encode()).hexdigest()
            folder = base / role / identity
            (folder / 'frames').mkdir(parents=True)
            for name in protocol['source_images']:
                (folder / 'frames' / (Path(name).stem + '.json.gz')).write_bytes(b'0' * 43000)
            (folder / 'identity.json').write_text(json.dumps(dict(identity=identity, protocol=protocol)))
            (folder / 'predictions.csv').write_bytes(b'ImageId,EncodedPixels\n')
            row = dict(input_mode=mode, identity=identity, cache_root=str(folder), pq=.25,
                       tp_pairs=100, fp=200, fn=300,
                       csv_sha256=hashlib.sha256((folder / 'predictions.csv').read_bytes()).hexdigest())
            rows.append(row)
            (folder / 'metrics.json').write_text(json.dumps(row))
        (base / role / 'role_results.json').write_text(json.dumps(dict(
            status='complete', model_role=role, results=rows)))
    archives = ns['export_results'](roles)
    assert len(archives) == 2
    exported = set()
    for path in archives:
        assert path.stat().st_size <= 24 * 1024**2
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read('export_manifest.json'))
            assert manifest['parts'] == 2 and manifest['conditions'] == 8
            expected = {entry['name']: entry for entry in manifest['files']}
            for name in archive.namelist():
                if name == 'export_manifest.json':
                    continue
                assert name not in exported
                exported.add(name)
                assert not name.endswith('.pt')
                assert hashlib.sha256(archive.read(name)).hexdigest() == expected[name]['sha256']
    summary = json.loads((base / 'summary.json').read_text())
    assert len(summary['results']) == 8 and not summary['training']
    assert len([n for n in exported if n.endswith('.json.gz')]) == 728
    assert before == (weight.read_bytes(), weight.stat().st_mtime_ns)
    bad = base / roles[0] / 'role_results.json'
    changed = json.loads(bad.read_text())
    changed['results'][0]['identity'] = 'wrong'
    bad.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='캐시 경로'):
        ns['export_results'](roles)
