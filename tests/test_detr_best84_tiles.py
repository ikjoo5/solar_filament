"""Numerical and geometric checks for fixed DETR-routed best84 inference."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import submission_inference as frozen
import detr_best84_tiles as routed


class AsymmetricModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, inputs):
        self.calls += 1
        y = torch.linspace(-.7, .5, inputs.shape[-2], device=inputs.device)[None, :, None]
        x = torch.linspace(.1, 1.2, inputs.shape[-1], device=inputs.device)[None, None, :]
        return torch.stack([inputs[:, 0]*.4+y, inputs[:, 1]*-.3+x], dim=1)


def all_tiles_boxes(shape, tile):
    boxes = [dict(box=[x+tile/2-1, y+tile/2-1, x+tile/2+1, y+tile/2+1], score=.95)
             for y in frozen.tile_positions(shape[0], tile)
             for x in frozen.tile_positions(shape[1], tile)]
    return {'70': boxes, '80': boxes}


class RoutedNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_full_grid_reference_and_all_arms_match_frozen_four_views(self):
        shape = (128, 128); tile = 64
        image = np.random.default_rng(41).integers(0, 256, shape, dtype=np.uint8)
        model = AsymmetricModel(); device = torch.device('cpu')
        expected = frozen.predict_views(model, image, device, tile, .25, 1, None)
        model.calls = 0
        # Flipped grid starts are different. Force all tiles separately in all
        # views by using the original and inverse-mapped grid-center boxes.
        proposals = all_tiles_boxes(shape, tile)
        boxes = proposals['70'][:]
        for _, axes in frozen.VIEWS:
            boxes += [dict(box=routed.transform_box(row['box'], axes, shape), score=.95)
                      for row in proposals['70']]
        result = routed.predict_arms(model, image, {'70':boxes, '80':boxes}, device, tile=tile)
        self.assertEqual(model.calls, 36)
        self.assertEqual(result['diagnostics']['actual_tile_forwards'], 36)
        for name, (base, average, unions) in result['predictions'].items():
            np.testing.assert_array_equal(base, expected[0], err_msg=name)
            np.testing.assert_array_equal(average, expected[1], err_msg=name)
            self.assertEqual(unions, expected[2])
            self.assertEqual(frozen.prediction_record(average, 'sample.jpeg'),
                             frozen.prediction_record(expected[1], 'sample.jpeg'))
        self.assertTrue(all(v['selected_tile_forwards']==36 for v in result['diagnostics']['arms'].values()))

    def test_subset_blending_inverse_flips_and_missing_pixel_fallback(self):
        image = np.random.default_rng(73).integers(0, 256, (64, 64), dtype=np.uint8)
        boxes = {'70':[dict(box=[2, 4, 10, 12], score=.85)],
                 '80':[dict(box=[2, 4, 10, 12], score=.7)]}
        model = AsymmetricModel(); device = torch.device('cpu')
        result = routed.predict_arms(model, image, boxes, device, tile=32, include_selected_only=True)
        self.assertEqual(model.calls, 36)
        full = result['predictions']['full']
        empty_arm = result['predictions']['detr80_c08']
        np.testing.assert_array_equal(empty_arm[0], full[0])
        np.testing.assert_array_equal(empty_arm[1], full[1])
        self.assertEqual(empty_arm[2], full[2])
        self.assertFalse(result['predictions']['detr80_c08_selected_only'][1].any())
        # Independent single-tile assembly: the chosen native32 tile is the
        # corresponding corner in each global view; Hann divides out exactly
        # as in the reference calculation.
        expected_views, expected_unions = [], {}
        for view, axes in frozen.VIEWS:
            transformed = np.ascontiguousarray(np.flip(image, axes)) if axes else image
            expected = frozen.predict_heads(model, transformed, device, 32, .25, 1, None)
            y, x = (32 if -2 in axes else 0), (32 if -1 in axes else 0)
            crop = transformed[y:y+32, x:x+32]
            probability = model(torch.from_numpy(frozen.model_input(crop)[None])).float().sigmoid().detach().numpy()[0]
            weight = frozen.blend_window(32)
            expected[:, y:y+32, x:x+32] = probability*weight/weight
            expected = np.flip(expected, axes) if axes else expected
            expected_views.append(expected)
            expected_unions[view] = frozen.rle(expected[0]>=.5)
            self.assertEqual(result['diagnostics']['views'][view]['detr70_c08']['selected_tiles'], [[y,x]])
        expected_mean = expected_views[0].copy()
        for array in expected_views[1:]: expected_mean += array
        expected_mean /= np.float32(4)
        for name in ('detr70_c08', 'detr80_c06'):
            actual = result['predictions'][name]
            np.testing.assert_array_equal(actual[0], expected_views[0])
            np.testing.assert_array_equal(actual[1], expected_mean)
            self.assertEqual(actual[2], expected_unions)
        self.assertGreater(np.abs(expected_mean-full[1]).max(), .01)
        np.testing.assert_array_equal(result['predictions']['detr70_c08'][1][:, 40:, 40:], full[1][:, 40:, 40:])

    def test_empty_boxes_are_exact_full_fallback_without_extra_forwards(self):
        image = np.zeros((64,64), np.uint8); model = AsymmetricModel()
        result = routed.predict_arms(model, image, {'70':[], '80':[]}, 'cpu', tile=32)
        expected = result['predictions']['full']
        self.assertEqual(model.calls, 36)
        for arm in routed.ARM_SPECS:
            actual = result['predictions'][arm]
            np.testing.assert_array_equal(actual[0], expected[0])
            np.testing.assert_array_equal(actual[1], expected[1])
            self.assertEqual(actual[2], expected[2])
            self.assertEqual(result['diagnostics']['arms'][arm]['selected_tile_forwards'], 0)
            self.assertEqual(result['diagnostics']['arms'][arm]['observed_union_pixel_fraction'], 0)

    def test_104_edge_boxes_are_covered_and_flips_are_involutions(self):
        rng = np.random.default_rng(87)
        boxes = [[0,0,1,1],[2047,2047,2048,2048],[0,0,2048,2048],[768,20,1792,1024]]
        for _ in range(100):
            lo = rng.uniform(0, 2047, 2)
            hi = lo+rng.uniform(.01, 1, 2)*(2048-lo)
            boxes.append([lo[0],lo[1],hi[0],hi[1]])
        self.assertEqual(len(boxes), 104)
        for box in boxes:
            for _, axes in frozen.VIEWS:
                transformed = routed.transform_box(box, axes, (2048,2048))
                np.testing.assert_allclose(routed.transform_box(transformed, axes, (2048,2048)), box, atol=1e-10)
                selected = routed.select_tiles([transformed])
                cover = np.zeros((2048,2048), bool)
                for y,x in selected:
                    self.assertIn(y,(0,768,1024)); self.assertIn(x,(0,768,1024))
                    cover[y:y+1024,x:x+1024] = True
                x1,y1,x2,y2 = routed.expanded_box(transformed,(2048,2048))
                self.assertTrue(cover[y1:y2,x1:x2].all())
                self.assertEqual(len(selected), len(set(selected)))
        self.assertEqual(routed.transform_box([768,0,1792,1024],(-1,),(2048,2048)), (256,0,1280,1024))
        self.assertEqual(routed.select_tiles([[0,0,2048,2048]]),[(0,0),(0,1024),(1024,0),(1024,1024)])

    def test_production_entry_uses_one_whole_image_limb_and_frozen_record_format(self):
        original = np.zeros((2048,2048), np.uint8)
        corrected = np.full_like(original, 12)
        heads = np.zeros((2,16,16), np.float32); heads[0,3:15,4:14]=.9
        predictions = {'full':(heads,heads.copy(),dict(base=frozen.rle(heads[0]>=.5)))}
        fake = dict(predictions=predictions, diagnostics={})
        cfg = dict(file_name='fixture.jpeg',image_sha256='pixels',identity='run',native_precision='fp32')
        with patch.object(frozen, 'limb', return_value=corrected) as limb, patch.object(routed, 'predict_arms', return_value=fake) as predict:
            result = routed.infer_native_variants(None, original, {'70':[], '80':[]}, cfg, 'cpu')
        limb.assert_called_once_with(original)
        self.assertIs(predict.call_args.args[1], corrected)
        expected = frozen.prediction_record(heads,'fixture.jpeg','pixels','run')
        expected['tta'] = frozen.prediction_record(heads,'fixture.jpeg','pixels','run')
        expected['view_unions'] = predictions['full'][2]
        self.assertEqual(result['rich']['full'], expected)
        self.assertEqual(result['diagnostics']['whole_image_limb_calls'], 1)

    def test_invalid_boxes_precision_and_nonfinite_model_are_rejected(self):
        image = np.zeros((64,64),np.uint8); empty = {'70':[], '80':[]}
        with self.assertRaisesRegex(ValueError,'FP32'):
            routed.predict_arms(AsymmetricModel(), image, empty, 'cpu', tile=32,amp_dtype=torch.float16)
        with self.assertRaisesRegex(ValueError,'confidence'):
            routed.arm_boxes({'70':[dict(box=[0,0,10,10],score=float('nan'))],'80':[]},image.shape)
        for box in ([1,1,1,8],[100,100,110,110],[0,0,float('inf'),5]):
            with self.assertRaises(ValueError): routed.select_tiles([box],image.shape,tile=32)
        class Bad(torch.nn.Module):
            def forward(self,x): return torch.full((1,2,32,32),float('nan'))
        with self.assertRaisesRegex(FloatingPointError,'Nonfinite'):
            routed.predict_arms(Bad(),image,empty,'cpu',tile=32)


if __name__ == '__main__': unittest.main()
