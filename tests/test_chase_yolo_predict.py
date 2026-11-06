from contextlib import redirect_stdout
import csv
import gzip
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import chase_yolo_predict as yp


class YoloExportTests(unittest.TestCase):
    def test_explicit_checkpoint_integrity_precedes_model_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = root / 'trained.pt'
            checkpoint.write_bytes(b'fixture checkpoint')
            with self.assertRaisesRegex(ValueError, 'SHA256 differs'):
                yp.generate_yolo_cache(checkpoint, '0' * 64, {}, root/'output.gz')

    def test_stage_resume_and_baseline_protection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            images = root/'images'
            images.mkdir()
            base_image = images/'frame000.jpeg'
            cv2.imwrite(str(base_image), np.full((2048, 2048), 120, np.uint8))
            names = [f'frame{i:03d}.jpeg' for i in range(91)]
            for name in names[1:]:
                os.link(base_image, images/name)
            image_sha = yp.digest(base_image)
            val = root/'val.csv'
            with val.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['image_id', 'file_name'])
                writer.writeheader()
                writer.writerows(dict(image_id=f'a-{Path(name).stem}', file_name=name) for name in names)
                writer.writerows(dict(image_id=f'b-{Path(name).stem}', file_name=name) for name in names[:10])
            old = root/'baseline.json.gz'
            old.write_bytes(gzip.compress(json.dumps(dict(schema_version=1,
                frames={Path(name).stem: {} for name in names},
                image_sha256={name: image_sha for name in names})).encode()))
            checkpoint = root/'new.pt'
            checkpoint.write_bytes(b'trained fixture')
            cfg = dict(images=str(images), val_split=str(val), yolo_cache=str(old), device='cpu')
            output = root/'new_yolo.json.gz'
            calls = []
            interrupt = [True]

            class FakeProvider:
                def __init__(self, yolo, native, **kwargs):
                    assert Path(yolo) == checkpoint
                    assert native is None and kwargs['verify_weights'] is False

                def _stage(self, stem, name, count):
                    calls.append((stem, name))
                    if interrupt[0] and name == 'preprocessing':
                        interrupt[0] = False
                        raise KeyboardInterrupt()
                    return [dict(image=stem, view=f'{name}{i}', instances=[],
                        shape=[2048, 2048], coordinate_space='original') for i in range(count)]

                def geometric(self, stem, pixels):
                    return self._stage(stem, 'geometric', 4)
                def preprocessing(self, stem, pixels):
                    return self._stage(stem, 'preprocessing', 8)
                def tiled(self, stem, pixels):
                    return self._stage(stem, 'tiled', 18)

            with patch.dict(yp.SPLIT_SHA256, val_split=yp.digest(val)), redirect_stdout(io.StringIO()):
                with self.assertRaises(KeyboardInterrupt):
                    yp.generate_yolo_cache(checkpoint, yp.digest(checkpoint), cfg, output, provider_factory=FakeProvider)
                result = yp.generate_yolo_cache(checkpoint, yp.digest(checkpoint), cfg, output, provider_factory=FakeProvider)
                self.assertEqual(result['images'], 91)
                self.assertEqual(calls.count(('frame000', 'geometric')), 1)
                count = len(calls)
                repeated = yp.generate_yolo_cache(checkpoint, yp.digest(checkpoint), cfg, output, provider_factory=FakeProvider)
                self.assertEqual(count, len(calls))
                self.assertEqual(result['sha256'], repeated['sha256'])
                output.write_bytes(b'corrupted output')
                repaired = yp.generate_yolo_cache(checkpoint, yp.digest(checkpoint), cfg, output, provider_factory=FakeProvider)
                self.assertEqual(result['sha256'], repaired['sha256'])
                self.assertEqual(count, len(calls))
                protected = root/'different_run.json.gz'
                protected.write_bytes(gzip.compress(json.dumps({'identity': 'other run'}).encode()))
                with self.assertRaisesRegex(ValueError, 'Existing YOLO output belongs'):
                    yp.generate_yolo_cache(checkpoint, yp.digest(checkpoint), cfg, protected, provider_factory=FakeProvider)
                with self.assertRaisesRegex(ValueError, 'Preserve the baseline'):
                    yp.generate_yolo_cache(checkpoint, yp.digest(checkpoint), cfg, old, provider_factory=FakeProvider)
                checkpoint.write_bytes(b'different trained fixture')
                with self.assertRaisesRegex(ValueError, 'checkpoint/inputs/code/runtime changed'):
                    yp.generate_yolo_cache(checkpoint, yp.digest(checkpoint), cfg, output, provider_factory=FakeProvider)


if __name__ == '__main__':
    unittest.main()
