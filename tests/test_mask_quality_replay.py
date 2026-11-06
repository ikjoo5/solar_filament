"""Portable fixtures for a label-free frozen-proposal export."""
import ast
from contextlib import redirect_stdout
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import export_mask_quality_replay as replay
import build_mask_quality_replay_colab as builder


def write_json(path, value):
    path.write_text(json.dumps(value))


class ReplayTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)/'experiment'; root.mkdir()
        (root/'proposals').mkdir()
        identity = dict(inputs=replay.INPUTS, code=replay.CODE, runtime=dict(python='fixture'))
        corpus_id = replay.fingerprint(identity)
        name = '20110101000000Ch.jpeg'; stem = Path(name).stem
        raw = dict(identity=dict(corpus=corpus_id, image_sha='a'*64, name=name, role='validation'),
            instances=[dict(confidence=.3, rle='PPPP', target=[0., 0.]),
                       dict(confidence=.7, rle='QQQQ', target=[.8, 1.])])
        payload = dict(raw, checksum=replay.fingerprint(raw))
        data = gzip.compress(json.dumps(payload).encode())
        path = root/'proposals'/f'{stem}.json.gz'; path.write_bytes(data)
        records = [dict(name='20100101000000Ch.jpeg', role='train', proposal_count=1, roi_count=1, selected=[0], raw_sha='b'*64),
                   dict(name=name, role='validation', proposal_count=2, roi_count=2, selected=[0, 1], raw_sha=replay.sha(data))]
        baseline = dict(pq=.4, fp=2, fn=3)
        summary = dict(status='complete', epochs_completed=2, final_epoch=2, best_development_pq=.35,
            baseline=baseline, training_images=1, validation_images=1, training_proposals=1, validation_proposals=2)
        values = {'summary.json': summary, 'baseline.json': baseline, 'corpus_identity.json': identity,
            'training_config.json': dict(corpus=corpus_id, epochs=2),
            'corpus.json': dict(identity=corpus_id, records=records, training_images=1, validation_images=1),
            'history.json': [dict(epoch=1, validation=dict(selected=dict(pq=.35))),
                             dict(epoch=2, validation=dict(selected=dict(pq=.3)))],
            'best_validation_scores.json': dict(epoch=1, scores={stem: [.2, .6]})}
        for name, value in values.items(): write_json(root/name, value)
        (root/'best_validation_predictions.csv').write_text('filament_id,segmentation_rle\n')
        # Sensitive or large unrelated files must never enter the package.
        (root/'best.pt').write_bytes(b'untouched model')
        (root/'original.jpeg').write_bytes(b'untouched image')
        train = root/'proposals/20100101000000Ch.json.gz'; train.write_bytes(b'not for export')
        return root, path

    def collect(self, root):
        with redirect_stdout(io.StringIO()):
            return replay.collect(root, expected_counts=(1, 1), expected_proposals=2)

    def refresh_raw_sha(self, root, path):
        corpus = json.loads((root/'corpus.json').read_text())
        corpus['records'][1]['raw_sha'] = replay.sha(path.read_bytes())
        write_json(root/'corpus.json', corpus)

    def test_exact_bytes_only_validation_scope_and_originals_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root, raw = self.fixture(directory)
            before = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            with redirect_stdout(io.StringIO()):
                result = replay.export(root, root/'exports', expected_counts=(1, 1), expected_proposals=2)
            with zipfile.ZipFile(result['archive']) as z:
                expected = set(replay.FILES)|{'proposals/'+raw.name, 'replay_manifest.json'}
                self.assertEqual(set(z.namelist()), expected)
                self.assertIsNone(z.testzip())
                self.assertEqual(z.read('proposals/'+raw.name), raw.read_bytes())
                manifest = json.loads(z.read('replay_manifest.json'))
                for name, value in manifest['entries'].items():
                    self.assertEqual(hashlib.sha256(z.read(name)).hexdigest(), value['sha256'])
                self.assertEqual(manifest['protocol']['planned_prediction_CSVs'], 28)
                self.assertFalse(manifest['training_or_inference_performed'])
            self.assertTrue(all((root/name).read_bytes() == value for name, value in before.items()))

    def test_tampered_gzip_SHA_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, path = self.fixture(directory)
            path.write_bytes(path.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError, 'file SHA'):
                self.collect(root)

    def test_changed_payload_even_with_updated_file_SHA_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, path = self.fixture(directory)
            raw = json.loads(gzip.decompress(path.read_bytes()))
            raw['instances'][0]['confidence'] = .9
            path.write_bytes(gzip.compress(json.dumps(raw).encode()))
            self.refresh_raw_sha(root, path)
            with self.assertRaisesRegex(ValueError, 'checksum'):
                self.collect(root)

    def test_missing_scores_invalid_scores_and_capped_validation_are_rejected(self):
        for change, message in [('missing', 'coverage'), ('nan', 'quality scores'), ('selected', 'all proposals')]:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root, _ = self.fixture(directory)
                if change == 'selected':
                    value = json.loads((root/'corpus.json').read_text()); value['records'][1]['selected'] = [1]; value['records'][1]['roi_count'] = 1
                    write_json(root/'corpus.json', value)
                else:
                    value = json.loads((root/'best_validation_scores.json').read_text())
                    if change == 'missing': value['scores'] = {}
                    else: next(iter(value['scores'].values()))[0] = float('nan')
                    write_json(root/'best_validation_scores.json', value)
                with self.assertRaisesRegex(ValueError, message): self.collect(root)

    def test_wrong_epoch_source_and_split_duplicates_are_rejected(self):
        for change, message in [('epoch', 'epoch mismatch'), ('source', 'source hashes'), ('duplicate', 'duplicate corpus')]:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root, _ = self.fixture(directory)
                if change == 'epoch':
                    value = json.loads((root/'best_validation_scores.json').read_text()); value['epoch'] = 2
                    write_json(root/'best_validation_scores.json', value)
                elif change == 'source':
                    value = json.loads((root/'corpus_identity.json').read_text()); value['code'] = {}
                    write_json(root/'corpus_identity.json', value)
                else:
                    value = json.loads((root/'corpus.json').read_text()); value['records'][0]['name'] = value['records'][1]['name']
                    write_json(root/'corpus.json', value)
                with self.assertRaisesRegex(ValueError, message): self.collect(root)

    def test_default_counts_cannot_accept_partial_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = self.fixture(directory)
            with self.assertRaisesRegex(ValueError, '616/91'):
                replay.collect(root)

    def test_copycell_payload_is_exact_stdlib_exporter(self):
        import base64,zlib
        info = builder.build()
        cell = Path(info['copy_cell']).read_text(); compile(cell, 'copy-cell', 'exec')
        payload = next(node.args[0].value for node in ast.walk(ast.parse(cell))
                       if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                       and node.func.attr == 'b64decode' and isinstance(node.args[0], ast.Constant))
        source = zlib.decompress(base64.b64decode(payload))
        self.assertEqual(source, Path(replay.__file__).read_bytes())
        self.assertEqual(replay.sha(source), info['source_sha256'])
        imports = {node.names[0].name.split('.')[0] for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Import)}
        self.assertTrue(imports <= sys.stdlib_module_names)


if __name__ == '__main__':
    unittest.main()
