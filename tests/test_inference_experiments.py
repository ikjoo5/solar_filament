"""Coordinate, raw cache and binary agreement tests, without a real YOLO model."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np
from pycocotools import mask as mu

TOOLS = Path(__file__).resolve().parents[1]/"tools"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location("experiment_yolo_inference", TOOLS/"experiment_yolo_inference.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class Tensor:
    def __init__(self, data): self.data = np.asarray(data)
    def detach(self): return self
    def float(self): return self
    def cpu(self): return self
    def numpy(self): return self.data


class FakeModel:
    def __init__(self): self.calls = []
    def predict(self, **kwargs):
        self.calls.append(kwargs)
        mask = kwargs['source'][:, :, 0] > 0
        return [SimpleNamespace(masks=SimpleNamespace(data=Tensor(mask[None].astype(float))),
                                boxes=SimpleNamespace(conf=Tensor([.82])))]


class InferenceExperimentTests(unittest.TestCase):
    def image(self):
        image = np.zeros((4, 5), dtype=np.uint8)
        image[0, 1] = 255
        image[2, 4] = 180
        return image

    def test_flips_have_expected_coordinates(self):
        image = self.image()
        horizontal = runner.apply_view(image, 'hflip2048')
        self.assertEqual(horizontal[0, 3], 255)
        self.assertEqual(horizontal[2, 0], 180)
        vertical = runner.apply_view(image, 'vflip2048')
        self.assertEqual(vertical[3, 1], 255)
        self.assertEqual(vertical[1, 4], 180)
        self.assertTrue(horizontal.flags.c_contiguous)
        self.assertTrue(vertical.flags.c_contiguous)

    def test_each_view_inverts_to_original_mask(self):
        for view in runner.DEFAULT_VIEWS:
            with self.subTest(view=view):
                model = FakeModel()
                result = runner.infer_view(model, self.image(), view, 'cpu')
                np.testing.assert_array_equal(result[0]['mask'], self.image() > 0)
                self.assertEqual(model.calls[0]['imgsz'], runner.VIEW_SIZES[view])
                self.assertEqual(model.calls[0]['conf'], .05)
                self.assertEqual(result[0]['confidence'], .82)

    def test_invalid_view_and_dimension_rejected(self):
        with self.assertRaises(runner.RunnerError):
            runner.apply_view(self.image(), 'rotation13')
        with self.assertRaises(runner.RunnerError):
            runner.apply_view(self.image()[None], 'base2048')

    def test_one_view_counts_once_even_with_duplicate_instances(self):
        mask = self.image() > 0
        raw = [{'mask': mask, 'confidence': .2}, {'mask': mask, 'confidence': .3}]
        first = runner.view_foreground(raw, .1, mask.shape)
        blank = np.zeros_like(mask)
        voted = runner.vote_foreground([first, blank, blank, blank], .5)
        self.assertFalse(voted.any())
        np.testing.assert_array_equal(runner.vote_foreground([first, first, blank, blank], .5), mask)
        self.assertFalse(runner.vote_foreground([first, first, blank, blank], .75).any())

    def test_foreground_threshold_includes_exact_confidence(self):
        raw = [{'mask': self.image() > 0, 'confidence': .15}]
        self.assertEqual(int(runner.view_foreground(raw, .15, (4, 5)).sum()), 2)
        self.assertFalse(runner.view_foreground(raw, .1501, (4, 5)).any())
        with self.assertRaises(runner.RunnerError):
            runner.view_foreground(raw, .1, (5, 4))

    def test_empty_view_and_voting_validation(self):
        self.assertFalse(runner.view_foreground([], .1, (4, 5)).any())
        for fraction in (0, -1, 1.1, float('nan')):
            with self.assertRaises(runner.RunnerError):
                runner.vote_foreground([self.image() > 0], fraction)
        with self.assertRaises(runner.RunnerError):
            runner.vote_foreground([], .5)

    def test_components_are_eight_connected_and_area_filtered(self):
        foreground = np.zeros((4, 5), dtype=bool)
        foreground[0, 0] = foreground[1, 1] = True
        foreground[3, 4] = True
        masks = runner.component_masks(foreground, 2)
        self.assertEqual(len(masks), 1)
        self.assertEqual(int(masks[0].sum()), 2)
        self.assertEqual(runner.component_masks(foreground, 3), [])

    def test_raw_rle_cache_preserves_original_coordinates_and_confidence(self):
        raw = [{'mask': self.image() > 0, 'confidence': .234}]
        record = runner.cache_record('sample.jpeg', 'hflip2048', raw)
        self.assertEqual(record['coordinate_space'], 'original image')
        self.assertEqual(record['instances'][0]['confidence'], .234)
        rle = {'size': record['shape'], 'counts': record['instances'][0]['segmentation_rle'].encode('ascii')}
        np.testing.assert_array_equal(mu.decode(rle), self.image() > 0)

    def test_predefined_plan_is_bounded_and_contains_prior_baseline(self):
        plan = runner.experiment_plan()
        self.assertEqual(len(plan), 22)
        self.assertEqual(len({p['name'] for p in plan}), 22)
        self.assertTrue(any(p['kind'] == 'base' and p['confidence'] == .25 and p['min_area'] == 20 for p in plan))
        self.assertEqual(sum(p['kind'] == 'base' for p in plan), 10)
        self.assertEqual(sum(p['kind'] == 'binary_vote' for p in plan), 12)


if __name__ == '__main__':
    unittest.main()
