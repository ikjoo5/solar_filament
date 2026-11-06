"""Lossless, native-coordinate CPU tile export and cache provenance checks."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

import cv2
import numpy as np
from pycocotools import mask as mu

ROOT = Path(__file__).resolve().parents[1]


def exporter_namespace():
    tree = ast.parse((ROOT/'tools/colab_export_pixel_tiles_cell.py').read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) or (
        isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {'prefix', 'capture'})]
    namespace = {'ast': ast}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<exporter>', 'exec'), namespace)
    return namespace


def original_source():
    tree = ast.parse((ROOT/'tools/colab_paste_evaluate.py').read_text())
    return next(ast.literal_eval(node.value) for node in tree.body
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == 'SCRIPT')


def restored_source():
    tree = ast.parse((ROOT/'tools/colab_restore_pixel_export_session_cell.py').read_text())
    return next(ast.literal_eval(node.value) for node in tree.body
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == 'SCRIPT')


def worker_for(data, run, payload_limit=None):
    tree = ast.parse(original_source())
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
            if key in {'DATA', 'RUN'}:
                node.value = ast.Call(func=ast.Name(id='Path', ctx=ast.Load()),
                                      args=[ast.Constant(str(data if key == 'DATA' else run))], keywords=[])
    worker = exporter_namespace()['build_worker'](ast.unparse(ast.fix_missing_locations(tree)))
    if payload_limit is not None:
        parsed = ast.parse(worker)
        for node in parsed.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'payload_limit':
                node.value = ast.Constant(payload_limit)
        worker = ast.unparse(ast.fix_missing_locations(parsed))
    return worker


def cached_record(stem, view, boxes, confidence=.5):
    instances = []
    for x, y, w, h in boxes:
        mask = np.zeros((2048, 2048), np.uint8)
        mask[y:y+h, x:x+w] = 1
        rle = mu.encode(np.asfortranarray(mask))
        instances.append(dict(confidence=confidence, rle=rle['counts'].decode('ascii')))
    return dict(schema_version=1, image=stem, view=view, shape=[2048, 2048],
                coordinate_space='original', instances=instances)


def setup_fixture(folder):
    data = folder/'data'; run = folder/'run'
    images = data/'train/train_images'; images.mkdir(parents=True); run.mkdir()
    rng = np.random.default_rng(617)
    image = rng.integers(70, 170, (2048, 2048), np.uint8)
    # The exporter reads image content, independent of the JPEG filename extension.
    ok, encoded = cv2.imencode('.png', image); assert ok
    (images/'sample.jpeg').write_bytes(encoded.tobytes())
    annotation = data/'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'
    annotation.write_text('not parseable JSON: annotation is hashed, never used to choose pixels')
    (run/'best.pt').write_bytes(b'not a model: CPU tile export never loads it')
    (run/'train_split.csv').write_text('image_id,file_name\n')
    (run/'val_split.csv').write_text('image_id,file_name\nA-sample,sample.jpeg\n')
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        annotation, run/'best.pt', run/'train_split.csv', run/'val_split.csv')}
    old = run/'official_pq_validation/20260101_old'; new = run/'official_pq_validation/20260102_new'
    old.mkdir(parents=True); new.mkdir()
    old_rows = [cached_record('sample', 'base', [(1000, 1000, 25, 25)]),
                cached_record('sample', 'scale', [(500, 500, 8, 8)], confidence=.14)]
    new_rows = [cached_record('sample', 'limb', [(2030, 2030, 18, 18)], confidence=.25),
                cached_record('sample', 'clahe', [(700, 700, 8, 8)], confidence=.24)]
    for output, name, rows in [(old, 'raw_views.jsonl', old_rows), (new, 'raw_preprocess_views.jsonl', new_rows)]:
        (output/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
        (output/'summary.json').write_text(json.dumps(dict(source_sha256=hashes)))
    return data, run


def run_worker(folder, data, run, payload_limit=None):
    script = folder/'worker.py'; result = folder/'archives.json'
    script.write_text(worker_for(data, run, payload_limit))
    process = subprocess.run([sys.executable, str(script), str(result)], text=True, capture_output=True)
    return process, result


class PixelTileExportTests(unittest.TestCase):
    def test_fresh_session_restores_identical_cpu_worker(self):
        build = exporter_namespace()['build_worker']
        # Same literal paths, native geometry, preprocessing and capture logic.
        self.assertEqual(build(restored_source()), build(original_source()))
        self.assertNotIn('torch', restored_source())
        self.assertNotIn('YOLO', restored_source().replace('SolarFilament_LimbOnly_YOLO26_mixed_FullRes', 'run'))

    def test_worker_extracts_only_cpu_preprocessing(self):
        worker = exporter_namespace()['build_worker'](original_source())
        tree = ast.parse(worker)
        imports = {alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
                   for alias in node.names}
        self.assertNotIn('torch', imports); self.assertNotIn('ultralytics', imports)
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        self.assertNotIn('GT', names); self.assertNotIn('YOLO', names)
        self.assertNotIn('annotations', names)

    def test_padded_bbox_selection_png_losslessness_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); data, run = setup_fixture(folder)
            process, result = run_worker(folder, data, run)
            self.assertEqual(process.returncode, 0, process.stderr)
            paths = [Path(p) for p in json.loads(result.read_text())]
            self.assertEqual(len(paths), 1)
            with zipfile.ZipFile(paths[0]) as archive:
                manifest = json.loads(archive.read('manifest.json'))
                info = manifest['images']['sample']
                positions = {(t['y'], t['x']) for t in info['tiles']}
                self.assertEqual(positions, {(896, 896), (896, 1024), (1024, 896), (1024, 1024), (1920, 1920)})
                namespace = {}; exec(worker_for(data, run).split('cv2.setNumThreads')[0], namespace)
                gray = cv2.imread(str(data/'train/train_images/sample.jpeg'), cv2.IMREAD_GRAYSCALE)
                expected = namespace['limb'](gray)
                self.assertEqual(info['limb_sha256'], hashlib.sha256(expected.tobytes()).hexdigest())
                for tile in info['tiles']:
                    payload = archive.read(tile['path'])
                    self.assertEqual(hashlib.sha256(payload).hexdigest(), tile['sha256'])
                    actual = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED)
                    top, left = tile['y'], tile['x']
                    np.testing.assert_array_equal(actual, expected[top:top+128, left:left+128])
                    self.assertEqual(actual.dtype, np.uint8)

    def test_shards_repeat_identical_complete_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); data, run = setup_fixture(folder)
            cache = next((run/'official_pq_validation').glob('*/raw_views.jsonl'))
            cache.write_text(json.dumps(cached_record('sample', 'base', [(900, 900, 300, 300)]))+'\n')
            process, result = run_worker(folder, data, run, payload_limit=70000)
            self.assertEqual(process.returncode, 0, process.stderr)
            paths = [Path(p) for p in json.loads(result.read_text())]
            self.assertGreater(len(paths), 1)
            manifests = []; names = []
            for path in paths:
                self.assertLess(path.stat().st_size, 24*1024*1024)
                with zipfile.ZipFile(path) as archive:
                    manifests.append(archive.read('manifest.json'))
                    names.extend(n for n in archive.namelist() if n != 'manifest.json')
            self.assertTrue(all(raw == manifests[0] for raw in manifests))
            manifest = json.loads(manifests[0])
            self.assertEqual(manifest['parts'], [p.name for p in paths])
            self.assertEqual(set(names), {t['path'] for t in manifest['images']['sample']['tiles']})
            self.assertEqual(len(names), len(set(names)))

    def test_current_input_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); data, run = setup_fixture(folder)
            (run/'best.pt').write_bytes(b'changed checkpoint')
            process, _ = run_worker(folder, data, run)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn('best.pt', process.stderr)
            self.assertFalse((run/'pixel_tile_exports').exists())

    def test_non_native_cache_coordinates_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); data, run = setup_fixture(folder)
            path = next((run/'official_pq_validation').glob('*/raw_views.jsonl'))
            path.write_text(path.read_text().replace('"original"', '"network"'))
            process, _ = run_worker(folder, data, run)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn('원본 좌표', process.stderr)


if __name__ == '__main__':
    unittest.main()
