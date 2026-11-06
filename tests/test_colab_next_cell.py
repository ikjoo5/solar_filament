"""Check the paste-in update against the previous cell without Colab or a GPU."""
import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import cv2
import numpy as np
from pycocotools import mask as mu


ROOT = Path(__file__).resolve().parents[1]


def patch_parts():
    tree = ast.parse((ROOT/'tools/colab_next_inference_cell.py').read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) or (
        isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in {'INITIALIZE', 'POSTPROCESS', 'INFERENCE'})]
    namespace = {'ast': ast}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<reviewed-patch>', 'exec'), namespace)
    return namespace


class Tensor:
    def __init__(self, data): self.data = np.asarray(data)
    def float(self): return self
    def cpu(self): return self
    def numpy(self): return self.data


class ColabUpdateTests(unittest.TestCase):
    def test_previous_cell_patches_once_and_keeps_final_export(self):
        original = ast.parse((ROOT/'tools/colab_paste_evaluate.py').read_text())
        script = next(ast.literal_eval(n.value) for n in original.body
                      if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                      and n.targets[0].id == 'SCRIPT')
        patch = patch_parts()['Patch']()
        updated = ast.unparse(ast.fix_missing_locations(patch.visit(ast.parse(script))))
        compile(updated, '<updated-worker>', 'exec')
        self.assertEqual(patch.counts, dict(init=1, post=1, loop=1, summary=1))
        self.assertIn('raw_views.jsonl', updated)
        self.assertIn('low_conf_4view_v1', updated)
        self.assertIn('result', updated)

    def test_all_views_return_original_coordinates_and_real_jsonl(self):
        parts = patch_parts()
        image = np.zeros((2048, 2048), np.uint8)
        image[25:40, 100:115] = 255
        image[500:510, 900:910] = 255

        class Model:
            def predict(self, source, **kwargs):
                mask = source[:, :, 0] > 0
                return [SimpleNamespace(masks=SimpleNamespace(data=Tensor(mask[None])),
                                        boxes=SimpleNamespace(conf=Tensor([.8])))]

        with tempfile.TemporaryDirectory() as folder:
            namespace = dict(np=np, cv2=cv2, mu=mu, json=json, Path=Path,
                             val_names=['sample.jpeg'], IMAGES=Path(folder), output=Path(folder),
                             model=Model(), limb=lambda x: image)
            # No actual image read is necessary for this geometry-only fixture.
            (Path(folder)/'sample.jpeg').write_bytes(b'placeholder')
            exec(parts['INITIALIZE']+'\n'+parts['POSTPROCESS'], namespace)
            self.assertEqual(len(namespace['predictions']), 22)
            exec(parts['INFERENCE'], namespace)
            records = [json.loads(line) for line in (Path(folder)/'raw_views.jsonl').read_text().splitlines()]
            self.assertEqual(len(records), 4)
            for record in records:
                self.assertEqual(record['instances'][0]['confidence'], .8)
                decoded = mu.decode({'size': [2048,2048],
                                     'counts': record['instances'][0]['rle'].encode('ascii')})
                np.testing.assert_array_equal(decoded, image > 0)
            for (mode, threshold), predicted in namespace['predictions'].items():
                self.assertIn('sample', predicted)
                expected_count = 1 if mode.endswith('_200') else 2
                self.assertEqual(len(predicted['sample']), expected_count)


if __name__ == '__main__':
    unittest.main()
