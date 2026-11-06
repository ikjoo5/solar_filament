"""Current minimal-config compatibility and native tiled mask coordinate checks."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import cv2
import numpy as np
from pycocotools import mask as mu

ROOT = Path(__file__).resolve().parents[1]


def parts():
    tree = ast.parse((ROOT/'tools/colab_tiled_inference_cell.py').read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) or (
        isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {'PREFIX', 'HELPERS', 'RUN'})]
    namespace = {'ast': ast}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<tiled-cell>', 'exec'), namespace)
    return namespace


def source_from(filename):
    tree = ast.parse((ROOT/'tools'/filename).read_text())
    return next(ast.literal_eval(node.value) for node in tree.body
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == 'SCRIPT')


def helpers():
    namespace = dict(np=np, cv2=cv2, mu=mu)
    exec(parts()['HELPERS'], namespace)
    return namespace


class Tensor:
    def __init__(self, array): self.array = np.asarray(array)
    def float(self): return self
    def cpu(self): return self
    def numpy(self): return self.array


class TiledInferenceTests(unittest.TestCase):
    def test_current_minimal_script_builds_without_old_gt_worker(self):
        patch = parts()
        source = source_from('colab_restore_pixel_export_session_cell.py')
        worker = patch['build_worker'](source)
        compile(worker, '<tiled-worker>', 'exec')
        tree = ast.parse(worker)
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        self.assertNotIn('GT', names); self.assertNotIn('image_map', names)
        self.assertNotIn('annotations', names); self.assertNotIn('score', names)
        self.assertIn('torch.cuda.is_available()', worker)
        self.assertIn('raw_tiled_views.jsonl', worker)
        self.assertIn('worker_source.py', worker)
        self.assertIn('81c2a27d95821b5ede283169c0da8c08d360ecc4c2a60a1b9c6baf96d76bcccc', worker)
        self.assertIn("'source_sha256'", worker)
        # Also accepts the earlier full SCRIPT while extracting no GT code.
        old_worker = patch['build_worker'](source_from('colab_paste_evaluate.py'))
        self.assertNotIn('GT', {n.id for n in ast.walk(ast.parse(old_worker)) if isinstance(n, ast.Name)})

    def test_18_views_preserve_confidence_offsets_holes_and_edge_flags(self):
        namespace = helpers()
        image = np.broadcast_to(np.arange(2048, dtype=np.uint16)[None, :] % 256,
                                (2048, 2048)).astype(np.uint8)
        masks = np.zeros((2, 1024, 1024), np.uint8)
        masks[0, 6:11, :2] = 1
        masks[1, 500:505, 400:405] = 1; masks[1, 502, 402] = 0
        confidence = np.array([.832745183, .05123456789], np.float64)
        calls = []

        class Model:
            def predict(self, pixels, **kwargs):
                calls.append((pixels.shape, int(pixels[0, 0, 0]), kwargs))
                np.testing.assert_array_equal(pixels[:, :, 0], pixels[:, :, 1])
                return [SimpleNamespace(masks=SimpleNamespace(data=Tensor(masks)),
                                        boxes=SimpleNamespace(conf=Tensor(confidence)))]

        rows = list(namespace['infer_image'](image, 'example', Model()))
        self.assertEqual(len(rows), 18); self.assertEqual(len(calls), 18)
        self.assertEqual(len({row['view'] for row in rows}), 18)
        self.assertEqual({row['input_size'] for row in rows}, {1536, 2048})
        self.assertEqual({tuple(row['crop_bbox']) for row in rows},
                         {(x, y, 1024, 1024) for x in (0, 512, 1024) for y in (0, 512, 1024)})
        for row, (shape, first_pixel, kwargs) in zip(rows, calls):
            left, top, width, height = row['crop_bbox']
            self.assertEqual(shape, (1024, 1024, 3)); self.assertEqual(first_pixel, int(image[top, left]))
            self.assertEqual(kwargs, dict(imgsz=row['input_size'], conf=.05, iou=.7,
                retina_masks=True, max_det=100, device=0, verbose=False))
            self.assertEqual(row['shape'], [2048, 2048]); self.assertEqual(row['coordinate_space'], 'original')
            self.assertEqual(row['view'], f'tile{row["input_size"]}_y{top}_x{left}')
            self.assertFalse(row['max_det_reached']); self.assertEqual(row['raw_box_count'], 2)
            for index, instance in enumerate(row['instances']):
                self.assertEqual(instance['confidence'], float(confidence[index]))
                decoded = mu.decode(dict(size=[2048, 2048], counts=instance['rle'].encode('ascii')))
                expected = np.zeros((2048, 2048), np.uint8)
                expected[top:top+1024, left:left+1024] = masks[index]
                np.testing.assert_array_equal(decoded, expected)
            self.assertEqual(row['instances'][0]['edge_clipped'], left > 0)
            self.assertEqual(row['instances'][0]['edge_sides'], ['left'] if left > 0 else [])
            self.assertFalse(row['instances'][1]['edge_clipped'])
            self.assertEqual(int(mu.area(dict(size=[2048, 2048],
                counts=row['instances'][1]['rle'].encode('ascii')))), 24)

    def test_only_crop_borders_inside_full_image_are_flagged(self):
        edge = helpers()['edge_sides']
        mask = np.zeros((1024, 1024), bool); mask[0, 0] = True; mask[-1, -1] = True
        self.assertEqual(edge(mask, 0, 0), ['bottom', 'right'])
        self.assertEqual(edge(mask, 512, 512), ['top', 'left', 'bottom', 'right'])
        self.assertEqual(edge(mask, 1024, 1024), ['top', 'left'])
        mask[:] = False; mask[100, 4] = True
        self.assertEqual(edge(mask, 512, 512), [])
        mask[100, 3] = True
        self.assertEqual(edge(mask, 512, 512), ['left'])

    def test_retina_shape_mismatch_fails_without_resizing(self):
        result = SimpleNamespace(masks=SimpleNamespace(data=Tensor(np.ones((1, 768, 768)))),
                                 boxes=SimpleNamespace(conf=Tensor([.5])))
        with self.assertRaises(AssertionError):
            helpers()['encode_result'](result, 'example', 1536, 512, 512)

    def test_cap_flag_counts_raw_boxes_even_when_masks_empty(self):
        masks = np.broadcast_to(np.zeros((1, 1024, 1024), np.uint8), (100, 1024, 1024))
        result = SimpleNamespace(masks=SimpleNamespace(data=Tensor(masks)),
                                 boxes=SimpleNamespace(conf=Tensor(np.full(100, .5))))
        row = helpers()['encode_result'](result, 'example', 2048, 1024, 1024)
        self.assertTrue(row['max_det_reached']); self.assertEqual(row['raw_box_count'], 100)
        self.assertEqual(row['empty_mask_count'], 100); self.assertEqual(row['instances'], [])
        empty = SimpleNamespace(masks=None, boxes=None)
        row = helpers()['encode_result'](empty, 'example', 1536, 0, 0)
        self.assertFalse(row['max_det_reached']); self.assertEqual(row['instances'], [])

    def test_native_roi_encoder_preserves_image_edge_and_holes(self):
        encode = helpers()['encode_native_roi']
        mask = np.ones((3, 7), bool); mask[1, 1:5] = False
        rle = encode(mask, 2045, 2041)
        expected = np.zeros((2048, 2048), np.uint8); expected[2045:, 2041:] = mask
        np.testing.assert_array_equal(mu.decode(rle), expected)


if __name__ == '__main__':
    unittest.main()
