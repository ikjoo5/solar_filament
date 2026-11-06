import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest
import zipfile

SPEC = importlib.util.spec_from_file_location('cache_export', Path(__file__).parents[1]/'tools/chase_cloud_cache_export.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def fixture(root):
    names = {f'image{i:03d}.jpeg': sha(str(i).encode()) for i in range(91)}
    identities = {}
    for arm in ('old', 'new'):
        protocol = dict(source_images=names, model=arm)
        identity = sha(encode(protocol)); identities[arm] = identity
        base = root/'native_validation_cache'/identity
        write(base/'identity.json', encode(dict(identity=identity, protocol=protocol)))
        for number, (name, image_sha) in enumerate(names.items()):
            rng = random.Random(number)
            payload = dict(file_name=name, padding=rng.randbytes(2500).hex())
            envelope = dict(schema_version=1, identity=identity, image_sha256=image_sha,
                            payload_sha256=sha(encode(payload)), payload=payload)
            write(base/(Path(name).stem+'.json.gz'), gzip.compress(encode(envelope)))
    old_yolo = root/'assets/old.json.gz'; new_yolo = root/'comparison/yolo_epoch060.json.gz'
    write(old_yolo, gzip.compress(b'old yolo')); write(new_yolo, gzip.compress(b'new yolo'))
    rows = []
    for yolo, native in [('old', 'old'), ('new', 'old'), ('old', 'new'), ('new', 'new')]:
        role = f'{yolo}_yolo_{native}_native'; label = 'epoch060_'+role
        protocol = dict(source_images=names, model=native,
                        yolo_cache_sha256=sha((old_yolo if yolo == 'old' else new_yolo).read_bytes()))
        identity = sha(encode(protocol))
        base = root/'comparison/validation'/label
        write(base/'identity.json', encode(dict(identity=identity, protocol=protocol)))
        write(base/'metrics.json', b'{}'); write(base/'predictions.csv', b'ImageId,EncodedPixels\n')
        rows.append(dict(role=role, label=label, identity=identity, native_identity=identities[native],
                         csv_sha256=sha((base/'predictions.csv').read_bytes())))
    write(root/'comparison/epoch060_comparison.json', encode(dict(epoch_per_model=60, results=rows)))
    write(root/'configs/comparison.json', encode(dict(yolo_cache=str(old_yolo))))
    return identities


class ExportTests(unittest.TestCase):
    def test_parts_content_and_input_preservation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fixture(root)
            original = {p: (sha(p.read_bytes()), p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}
            outputs = MODULE.export_cache(root, limit=128*1024)
            self.assertGreater(len(outputs), 1)
            actual, expected = {}, None
            for part in outputs:
                self.assertLessEqual(part.stat().st_size, 128*1024)
                with zipfile.ZipFile(part) as archive:
                    manifest = json.loads(archive.read('export_manifest.json'))
                    expected = {r['name']: r['sha256'] for r in manifest['files']}
                    self.assertEqual(manifest['parts'], len(outputs))
                    for name in archive.namelist():
                        if name != 'export_manifest.json':
                            self.assertNotIn(name, actual)
                            actual[name] = sha(archive.read(name))
            self.assertEqual(actual, expected)
            self.assertEqual(len([n for n in actual if n.startswith('native/') and n.endswith('.json.gz')]), 182)
            self.assertEqual(original, {p: (sha(p.read_bytes()), p.stat().st_mtime_ns) for p in original})

    def test_bad_payload_rejected_before_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); identities = fixture(root)
            path = root/'native_validation_cache'/identities['new']/'image000.json.gz'
            value = json.loads(gzip.decompress(path.read_bytes())); value['payload']['padding'] = 'changed'
            path.write_bytes(gzip.compress(encode(value)))
            with self.assertRaisesRegex(ValueError, '내용 검증'):
                MODULE.export_cache(root)
            self.assertFalse((root/'cloud_cache_exports').exists())

    def test_missing_file_rejected_before_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); identities = fixture(root)
            (root/'native_validation_cache'/identities['new']/'image000.json.gz').unlink()
            with self.assertRaises(FileNotFoundError):
                MODULE.export_cache(root)
            self.assertFalse((root/'cloud_cache_exports').exists())

    def test_yolo_hash_rejected_before_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fixture(root)
            (root/'comparison/yolo_epoch060.json.gz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, '해시'):
                MODULE.export_cache(root)
            self.assertFalse((root/'cloud_cache_exports').exists())


if __name__ == '__main__':
    unittest.main()
