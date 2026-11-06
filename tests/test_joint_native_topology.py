"""Regrouping may change instance identity, but must conserve every FG pixel."""
from pathlib import Path
import sys
import unittest

import numpy as np
from pycocotools import mask as mu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from experiment_joint_native_topology import merge_masks, partition_by_seeds, plan, split_masks


def encode(mask):
    return mu.encode(np.asfortranarray(mask.astype(np.uint8)))


class JointTopologyTests(unittest.TestCase):
    def fixture(self):
        left = np.zeros((2048, 2048), bool)
        right = np.zeros_like(left)
        left[30:60, 20:50] = True
        right[30:60, 70:100] = True
        return left, right

    def test_fixed_plan_and_partition_conserve_foreground(self):
        self.assertEqual([s['name'] for s in plan()],
                         ['split_tta0.5', 'split_tta0.65', 'merge_tta0.35', 'merge_tta0.5'])
        foreground = np.ones((30, 90), bool)
        seeds = [np.zeros_like(foreground), np.zeros_like(foreground)]
        seeds[0][10:20, 5:15] = True
        seeds[1][10:20, 70:80] = True
        pieces = partition_by_seeds(foreground, seeds, 300)
        self.assertEqual(len(pieces), 2)
        self.assertTrue(np.array_equal(sum(pieces), foreground.astype(int)))
        self.assertIsNone(partition_by_seeds(foreground, [seeds[0], seeds[0]], 300))
        self.assertIsNone(partition_by_seeds(foreground, seeds, 2000))

    def test_split_requires_both_models_and_preserves_full_original(self):
        left, right = self.fixture()
        original = left | right
        original[43:47, 50:70] = True
        source = encode(original)
        peers = [encode(left), encode(right)]
        result, events, _ = split_masks([source], peers, peers, plan()[0])
        self.assertEqual(len(result), 2)
        self.assertEqual(len(events), 1)
        self.assertEqual(int(mu.area(mu.merge(result))), int(original.sum()))
        self.assertEqual(sum(int(mu.area(r)) for r in result), int(original.sum()))
        self.assertTrue(np.array_equal(mu.decode(mu.merge(result)).astype(bool), original))
        rejected, events, _ = split_masks([source], peers, peers[:1], plan()[0])
        self.assertEqual(len(rejected), 1)
        self.assertFalse(events)
        self.assertEqual(rejected[0]['counts'], source['counts'])

    def test_merge_requires_two_model_agreement_and_keeps_other_objects(self):
        left, right = self.fixture()
        other = np.zeros_like(left)
        other[100:130, 100:130] = True
        originals = [encode(left), encode(right), encode(other)]
        witness = encode(left | right)
        result, events, _ = merge_masks(originals, [witness], [witness], plan()[2])
        self.assertEqual(len(result), 2)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['original_indices'], [0, 1])
        self.assertTrue(np.array_equal(mu.decode(mu.merge(result)), mu.decode(mu.merge(originals))))
        self.assertEqual(sum(int(mu.area(r)) for r in result), int(mu.area(mu.merge(originals))))
        rejected, events, _ = merge_masks(originals, [witness], [encode(other)], plan()[2])
        self.assertEqual(len(rejected), 3)
        self.assertFalse(events)


if __name__ == '__main__':
    unittest.main()
