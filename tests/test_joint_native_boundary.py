"""Boundary refinement must preserve instance identity and disjointness."""
from pathlib import Path
import sys
import unittest

import numpy as np
from pycocotools import mask as mu

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import experiment_joint_native_boundary as experiment
from filament_eval import Filament


class JointBoundaryTests(unittest.TestCase):
    def test_supported_boundary_change_cannot_enter_another_original_instance(self):
        original = np.zeros((48, 48), bool)
        original[10:30, 10:30] = True
        peer = np.zeros_like(original)
        peer[10:30, 11:31] = True
        other = np.zeros_like(original)
        setting = experiment.plan()[0]
        candidate, status = experiment.refine_roi(original, peer, peer, other, setting)
        self.assertEqual(status, 'changed')
        self.assertTrue(np.array_equal(candidate, peer))
        other[10:30, 30] = True
        candidate, status = experiment.refine_roi(original, peer, peer, other, setting)
        self.assertEqual(status, 'fallback_other_original_intrusion')
        self.assertTrue(np.array_equal(candidate, original))

    def test_component_split_and_large_geometry_change_fall_back(self):
        original = np.zeros((60, 60), bool)
        original[10:30, 5:25] = True
        original[10:30, 30:50] = True
        original[20, 25:30] = True
        peer = original.copy()
        peer[20, 25:30] = False
        candidate, status = experiment.refine_roi(
            original, peer, peer, np.zeros_like(original), experiment.plan()[0])
        self.assertEqual(status, 'fallback_component_count')
        self.assertTrue(np.array_equal(candidate, original))
        for setting in experiment.plan():
            candidate, status = experiment.refine_roi(
                original, np.zeros_like(original), np.zeros_like(original),
                np.zeros_like(original), setting)
            self.assertEqual(status, 'fallback_geometry')
            self.assertTrue(np.array_equal(candidate, original))

    def test_expansions_colliding_in_old_background_roll_back_both_instances(self):
        def row(index, left, width):
            rle = experiment.encode_roi(np.ones((20, width), bool), 10, left)
            return Filament(f'image_{index}', 'image', 'image',
                            rle['counts'].decode(), int(mu.area(rle)))
        originals = [row(1, 10, 20), row(2, 40, 20)]
        expanded = [row(1, 10, 26), row(2, 34, 26)]
        result, rollbacks = experiment.revert_collisions(expanded, originals)
        self.assertEqual(rollbacks, 2)
        self.assertEqual(result, originals)
        # Both fixed fusion methods are exact identities when peers agree.
        original = np.zeros((40, 40), bool)
        original[10:30, 10:30] = True
        for setting in experiment.plan():
            result, status = experiment.refine_roi(
                original, original, original, np.zeros_like(original), setting)
            self.assertEqual(status, 'unchanged')
            self.assertTrue(np.array_equal(result, original))


if __name__ == '__main__':
    unittest.main()
