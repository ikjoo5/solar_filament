"""Inference-only Colab export: seam parity, real stats, and verified restart."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

import numpy as np
from pycocotools import mask as mu
import torch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cell = load('native_export_cell_test_module', ROOT/'tools/colab_export_native_predictions_cell.py')
export = types.ModuleType('native_export_worker_test_module')
exec(compile(cell.EXPORT_SOURCE, '<native-export-worker-fixture>', 'exec'), export.__dict__)
source = load('native_export_source_test_module', ROOT/'tools/solar_a100_train_standalone.py')


class TwoHead(torch.nn.Module):
    def forward(self, inputs):
        return torch.stack([inputs[:, 0]*.7+.3, inputs[:, 1]*-.2-.6], dim=1)


class PositionedTwoHead(torch.nn.Module):
    def forward(self, inputs):
        y = torch.linspace(-.7, .9, inputs.shape[-2])[None, :, None]
        x = torch.linspace(.8, -.3, inputs.shape[-1])[None, None, :]
        return torch.stack([inputs[:, 0]*.7+y+x, inputs[:, 1]*-.2+y-x], dim=1)


class NativeExportTests(unittest.TestCase):
    def test_tile_blending_matches_original_foreground_and_pointwise_spine(self):
        image = np.random.default_rng(4).integers(0, 256, (48, 64), np.uint8)
        model = TwoHead()
        device = torch.device('cpu')
        heads = export.predict_heads(source, model, image, device, 32, .25, 2, None)
        old = source.predict_foreground(model, image, device, 32, .25, 2, None)
        expected = model(torch.from_numpy(source.model_input(image)[None])).sigmoid()[0].numpy()
        np.testing.assert_array_equal(heads[0], old)
        np.testing.assert_allclose(heads, expected, atol=2e-7, rtol=1e-6)
        self.assertEqual(heads.shape, (2, 48, 64))
        self.assertTrue(np.isfinite(heads).all())

    def test_four_view_tta_explicit_inverse_and_probability_average(self):
        image = np.random.default_rng(8).integers(0, 256, (48, 64), np.uint8)
        model, device = PositionedTwoHead(), torch.device('cpu')
        progress = []
        base, average, unions = export.predict_views(source, model, image, device, 32, .25, 2, None,
                                                     progress.append)
        expected = []
        for view, axes in [('base', ()), ('hflip', (-1,)), ('vflip', (-2,)), ('hvflip', (-2, -1))]:
            changed = np.ascontiguousarray(np.flip(image, axes)) if axes else image
            predicted = export.predict_heads(source, model, changed, device, 32, .25, 2, None)
            restored = np.flip(predicted, axes) if axes else predicted
            expected.append(restored)
            np.testing.assert_array_equal(mu.decode(unions[view]), restored[0] >= .5)
        np.testing.assert_array_equal(base, expected[0])
        np.testing.assert_allclose(average, sum(expected)/4, atol=1e-7)
        self.assertGreater(float(np.max(np.abs(base-average))), .01)
        self.assertEqual(progress, ['base', 'hflip', 'vflip', 'hvflip'])

    def test_union_rle_and_component_statistics_use_actual_probabilities(self):
        fg = np.zeros((24, 32), np.float32)
        fg[3:15, 4:14] = np.linspace(.36, .96, 120, dtype=np.float32).reshape(12, 10)
        fg[19:21, 25:27] = .9  # Small component remains in union, excluded from candidates.
        spine = np.linspace(.01, .99, fg.size, dtype=np.float32).reshape(fg.shape)
        record = export.prediction_record(np.stack([fg, spine]), 'sample.jpeg', 'image-hash', 'run-id')
        self.assertEqual(set(record['unions']['foreground']), {'0.2', '0.35', '0.5', '0.65', '0.8'})
        self.assertEqual(set(record['unions']['spine']), {'0.3', '0.5', '0.7'})
        for key, encoded in record['unions']['foreground'].items():
            np.testing.assert_array_equal(mu.decode(encoded), fg >= float(key))
        instances = record['components']['0.35']
        self.assertEqual(len(instances), 1)
        component = instances[0]
        selected = mu.decode(component['rle']).astype(bool)
        self.assertEqual(component['area'], 120)
        self.assertEqual(component['bbox'], [4, 3, 10, 12])
        for name, probability in [('foreground', fg), ('spine', spine)]:
            values = probability[selected]
            self.assertEqual(component['stats'][name]['mean'], float(values.mean()))
            self.assertEqual(component['stats'][name]['max'], float(values.max()))
            self.assertEqual(component['stats'][name]['p90'], float(np.percentile(values, 90)))
        self.assertEqual(record['file_name'], 'sample.jpeg')
        self.assertEqual(record['size'], [24, 32])
        self.assertNotIn('ground_truth', record)

    def test_atomic_record_restart_rejects_changes_and_recovers_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'image.jsonl'
            record = dict(identity='run', file_name='a.jpeg', image_sha256='pixels', unions={})
            export.atomic_text(path, export.compact(record)+'\n')
            entry = dict(record_sha256=export.sha(path), image_sha256='pixels')
            self.assertEqual(export.verified_record(path, entry, 'run', 'a.jpeg', 'pixels'), record)
            self.assertIsNone(export.verified_record(path, entry, 'different-run', 'a.jpeg', 'pixels'))
            self.assertIsNone(export.verified_record(path, entry, 'run', 'a.jpeg', 'changed-pixels'))
            path.with_name(path.name+'.pending').write_text('interrupted')
            self.assertEqual(export.verified_record(path, entry, 'run', 'a.jpeg', 'pixels'), record)
            path.write_text('corrupt')
            self.assertIsNone(export.verified_record(path, entry, 'run', 'a.jpeg', 'pixels'))
            export.atomic_text(path, export.compact(record)+'\n')
            self.assertEqual(export.verified_record(path, entry, 'run', 'a.jpeg', 'pixels'), record)
            self.assertFalse(path.with_name(path.name+'.pending').exists())

    def test_readable_cell_is_portable_and_never_calls_training(self):
        text = (ROOT/'tools/colab_export_native_predictions_cell.py').read_text()
        compile(text, '<one-paste-export-cell>', 'exec')
        compile(cell.EXPORT_SOURCE, '<inference-subprocess-worker>', 'exec')
        self.assertLessEqual(len(text.splitlines()), 250)
        self.assertNotIn('/workspace', text)
        self.assertNotIn('base64', text)
        self.assertNotIn('.run_training(', text)
        self.assertNotIn('.backward(', text)
        self.assertNotIn('optimizer.step', text)
        self.assertIn('weights_only=True', text)
        self.assertIn('subprocess.run([sys.executable', text)
        self.assertEqual(export.MIN_AREA, 100)


if __name__ == '__main__':
    unittest.main()
