"""Held-out rich exports reuse identity views and resume without reading labels."""
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from pycocotools import mask as mu
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import fivefold_rich_inference as rich


class ExportTests(unittest.TestCase):
    def fixture(self, root, names=('20200101000000Bh.jpeg',)):
        images = root/'images'; images.mkdir()
        split = root/'val.csv'
        with split.open('w', newline='') as stream:
            writer = csv.writer(stream); writer.writerow(['image_id', 'file_name'])
            for name in names:
                (images/name).write_bytes(('original '+name).encode())
                writer.writerow(['rater1-'+Path(name).stem, name])
                writer.writerow(['rater2-'+Path(name).stem, name])
        pixels = np.zeros((2048, 2048), np.uint8)
        pixels[17:37, 61:81] = np.arange(20, dtype=np.uint8)[None, :]+180
        calls = []
        class Store:
            def __init__(self, *args):
                self.args = args
                assert args[-1] == 'limb_only'
            def load(self, name):
                return pixels.copy() if '0101' in name else np.zeros_like(pixels)
        def predict(model, image, device, crop, overlap, batch, dtype):
            self.assertFalse(model.training)
            self.assertFalse(torch.is_grad_enabled())
            self.assertEqual((crop, overlap, batch, dtype), (1024, .25, 1, None))
            calls.append(image.copy())
            return image.astype(np.float32)/np.float32(255)
        runtime = types.SimpleNamespace(__file__=__file__, ImageStore=Store, predict_foreground=predict)
        cfg = dict(images=str(images), val_split=str(split), cache_root=str(root/'cache'), crop=1024,
            infer_batch=1, overlap=.25, device='cpu', stage_epoch=20, fold=0, profile='tta_clahe')
        provenance = dict(model_sha256='a'*64, study_identity='study', fold=0, stage_epoch=20)
        return runtime, torch.nn.Identity(), cfg, provenance, pixels, calls

    def test_four_flip_alignment_identity_reuse_clahe_and_cache_resume(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            root = Path(folder)
            runtime, model, cfg, provenance, pixels, calls = self.fixture(root)
            result = rich.export_stage(runtime, model, cfg, root/'stage', provenance)
            self.assertEqual((result['image_coverage'], result['forward_calls']), (1, 5))
            self.assertEqual(len(calls), 5)  # Base plus3 flips plusCLAHE, not six calls.
            self.assertTrue(np.array_equal(calls[0], pixels))
            self.assertTrue(np.array_equal(calls[1], np.flip(pixels, -1)))
            record = json.loads(Path(result['rich_predictions']).read_text())
            self.assertEqual(record['components'], record['tta']['components'])
            self.assertEqual(record['unions'], record['tta']['unions'])
            self.assertEqual(len(record['view_unions']), 4)
            expected = pixels >= 128
            for value in record['view_unions'].values():
                self.assertTrue(np.array_equal(mu.decode(value).astype(bool), expected))
            self.assertIn('clahe', record['preprocess_views'])
            from submission_native_recipe import _refine_boundaries, _final_admission_pruning
            originals = [dict(size=[2048,2048], counts=c['rle']['counts'].encode())
                         for c in record['components']['0.5'] if c['area'] >= 300]
            refined, _ = _refine_boundaries(originals, record)
            final, _ = _final_admission_pruning(refined, record)
            self.assertEqual(len(final), 1)
            self.assertTrue(np.array_equal(mu.decode(final[0]), mu.decode(originals[0])))
            self.assertEqual(Path(result['rich_records']).name, 'rich_records.jsonl')
            first = Path(result['rich_predictions']).read_bytes()
            runtime.predict_foreground = lambda *args: self.fail('Cached inference must not predict again')
            resumed = rich.export_stage(runtime, model, cfg, root/'stage', provenance)
            self.assertEqual((resumed['cached_images'], resumed['forward_calls']), (1, 0))
            self.assertEqual(Path(resumed['rich_predictions']).read_bytes(), first)
            self.assertFalse(json.loads(Path(resumed['summary']).read_text())['ground_truth_read'])

    def test_zero_prediction_images_remain_in_coverage_and_profile_can_skip_clahe(self):
        names = ('20200101000000Bh.jpeg', '20200102000000Bh.jpeg')
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            root = Path(folder)
            runtime, model, cfg, provenance, _, calls = self.fixture(root, names)
            cfg['profile'] = 'tta'
            result = rich.export_stage(runtime, model, cfg, root/'stage', provenance)
            self.assertEqual((result['image_coverage'], len(calls)), (2, 8))
            records = [json.loads(line) for line in Path(result['rich_predictions']).read_text().splitlines()]
            blank = next(record for record in records if record['file_name'] == names[1])
            self.assertEqual(blank['components']['0.5'], [])
            self.assertEqual(blank['tta']['components']['0.5'], [])
            self.assertEqual(int(mu.area(blank['tta']['unions']['foreground']['0.2'])), 0)
            self.assertEqual(blank['preprocess_views'], {})

    def test_changed_cache_model_or_input_is_preserved_and_rejected(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            root = Path(folder)
            runtime, model, cfg, provenance, _, _ = self.fixture(root)
            result = rich.export_stage(runtime, model, cfg, root/'stage', provenance)
            path = next((root/'stage/records').glob('*.json'))
            original = path.read_bytes()
            changed = json.loads(original); changed['fold'] = 4
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'content changed'):
                rich.export_stage(runtime, model, cfg, root/'stage', provenance)
            self.assertNotEqual(path.read_bytes(), original)
            path.write_bytes(original)
            with self.assertRaisesRegex(ValueError, 'settings/sources changed'):
                rich.export_stage(runtime, model, cfg, root/'stage', dict(provenance, model_sha256='b'*64))
            source = Path(cfg['images'])/'20200101000000Bh.jpeg'
            source.write_bytes(b'changed source')
            with self.assertRaisesRegex(ValueError, 'provenance changed'):
                rich.export_stage(runtime, model, cfg, root/'stage', provenance)
            self.assertEqual(path.read_bytes(), original)

    def test_invalid_shapes_never_write_a_prediction_record(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            root = Path(folder)
            runtime, model, cfg, provenance, _, _ = self.fixture(root)
            runtime.predict_foreground = lambda *args: np.zeros((16, 16), np.float32)
            with self.assertRaisesRegex(ValueError, '2048-square'):
                rich.export_stage(runtime, model, cfg, root/'stage', provenance)
            self.assertEqual(list((root/'stage/records').glob('*.json')), [])


if __name__ == '__main__':
    unittest.main()
