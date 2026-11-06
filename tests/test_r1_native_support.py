"""Frozen full-rich native decoding and isolated worker behavior, without GPU/models."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import cv2

TOOLS = Path(__file__).resolve().parents[1]/'tools'
sys.path.insert(0, str(TOOLS))
import r1_test_submission as runner


class NativeSupportTests(unittest.TestCase):
    def test_prepare_uses_frozen_grayscale_decode_for_tinted_original_jpeg(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); images = root/'images'; images.mkdir()
            pixels = np.full((2048, 2048, 3), 100, np.uint8)
            pixels[:, :, 0] = 120
            source = images/'original.jpeg'
            self.assertTrue(cv2.imwrite(str(source), pixels))
            decoded = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
            self.assertFalse(np.array_equal(decoded[:, :, 0], decoded[:, :, 1]))
            (root/'names.json').write_text(json.dumps(['original.jpeg']))
            (root/'yolo.pt').write_bytes(b'test yolo'); (root/'native.pt').write_bytes(b'test native')
            contract = runner.Contract(images=1, yolo_sha=runner.sha(root/'yolo.pt'), native_sha=runner.sha(root/'native.pt'))
            config = dict(images=str(images), expected_names_path=str(root/'names.json'), output_root=str(root/'out'),
                          yolo_checkpoint=str(root/'yolo.pt'), native_checkpoint=str(root/'native.pt'), device='cpu')
            _, manifest, _ = runner.prepare(config, contract)
            self.assertEqual(manifest[0]['sha256'], runner.sha(source))
            self.assertEqual(manifest[0]['name'], 'original.jpeg')

    def test_full_record_is_identical_to_frozen_decoder(self):
        from submission_inference import prediction_record, rle
        base = np.zeros((2, 2048, 2048), np.float32)
        average = base.copy()
        base[0, 40:65, 100:140] = np.float32(.35)
        base[0, 70:80, 100:120] = np.nextafter(np.float32(.35), np.float32(0))
        base[0, 81:91, 100:120] = np.nextafter(np.float32(.35), np.float32(1))
        average[0, 42:67, 102:142] = np.float32(.35)
        average[0, 200:225, 240:265] = .9
        views = {name:rle(base[0] >= .5) for name in ('base', 'hflip', 'vflip', 'hvflip')}
        support = runner.native_record_from_heads(base, average, views, file_name='photo.jpeg', image_sha256='image', identity='job')
        expected_base = prediction_record(base, 'photo.jpeg', 'image', 'job')
        expected_tta = prediction_record(average, 'photo.jpeg', 'image', 'job')
        self.assertEqual(support['unions']['foreground']['0.35'], expected_base['unions']['foreground']['0.35'])
        self.assertEqual(support['tta']['unions']['foreground']['0.35'], expected_tta['unions']['foreground']['0.35'])
        self.assertEqual(support['components'], expected_base['components'])
        self.assertEqual(support['tta'], expected_tta)
        self.assertEqual(support['view_unions'], views)

    def test_import_does_not_initialize_torch_or_ultralytics(self):
        code = (f'import sys;sys.path.insert(0,{str(TOOLS)!r});import r1_test_submission;'
                'assert "torch" not in sys.modules;assert "ultralytics" not in sys.modules;print("ok")')
        result = subprocess.check_output([sys.executable, '-c', code], text=True)
        self.assertEqual(result.strip(), 'ok')

    def test_server_forces_fp16_and_preserves_protocol_and_metadata(self):
        import torch
        import submission_inference
        import native_instance_training
        calls = []
        pixels = np.zeros((2048, 2048), np.uint8)
        heads = np.zeros((2, 2048, 2048), np.float32)
        heads[0, 100:130, 100:130] = .5
        class Provider:
            def __init__(self, *args, **kwargs):
                self.device = torch.device('cuda')
                calls.append(('init', args, kwargs))
            def _native(self):return 'native_model'
        def predict(model, image, device, **kwargs):
            calls.append(('predict', kwargs))
            self.assertEqual(model, 'native_model')
            np.testing.assert_array_equal(image, pixels)
            views = {name:submission_inference.rle(heads[0] >= .5) for name in ('base', 'hflip', 'vflip', 'hvflip')}
            return heads, heads, views
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'input.npy'; target = root/'out.json'
            np.save(source, pixels, allow_pickle=False)
            requests = [dict(protocol='solar-native-v1', command='native', id=1, image_id='photo',
                input_path=str(source), output_path=str(target), file_name='photo.jpeg', image_sha256='pixels', identity='job'),
                dict(protocol='solar-native-v1', command='close')]
            stdin = io.StringIO(''.join(json.dumps(row)+'\n' for row in requests)); stdout = io.StringIO()
            with patch.object(submission_inference, 'InferenceProvider', Provider), \
                    patch.object(submission_inference, 'limb', lambda value: value), \
                    patch.object(submission_inference, 'predict_views', predict), \
                    patch.object(native_instance_training, 'configure_cuda') as configure, \
                    patch('sys.stdin', stdin), redirect_stdout(stdout):
                runner.native_support_server(['--native-support-server', '--checkpoint', 'unused', '--device', 'cuda', '--batch', '1'])
            response = json.loads(stdout.getvalue())
            self.assertEqual(response['status'], 'ok')
            configure.assert_called_once()
            self.assertEqual(calls[-1][1]['amp_dtype'], torch.float16)
            self.assertEqual(calls[-1][1]['batch'], 1)
            output = json.loads(target.read_text())
            self.assertEqual(output['file_name'], 'photo.jpeg')
            self.assertEqual(output['image_sha256'], 'pixels')
            self.assertEqual(output['identity'], 'job')
            self.assertIn('components', output)
            self.assertEqual(set(output['view_unions']), {'base', 'hflip', 'vflip', 'hvflip'})


if __name__ == '__main__':unittest.main()
