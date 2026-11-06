"""Target geometry, trained-head loss and conservative instance decoding."""
from pathlib import Path
import sys
import unittest

import numpy as np
from pycocotools import mask as mask_utils
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from native_instance_model import NativeInstanceNet, decode_instances, loss_fn, make_targets


class NativeInstanceModelTests(unittest.TestCase):
    def test_one_pixel_filament_remains_positive_even_when_all_boundary(self):
        ids = np.zeros((32, 48), np.int32)
        ids[16, 4:44] = 1
        target = make_targets(ids)
        self.assertEqual(int(target['foreground'].sum()), 40)
        self.assertEqual(int(target['boundary'].sum()), 40)
        np.testing.assert_array_equal(target['distance'][16, 4:44], np.ones(40))
        probability = np.stack([target[k] for k in ('foreground', 'boundary', 'distance')])
        decoded = decode_instances(probability, min_area=1)
        self.assertEqual(len(decoded), 1)
        np.testing.assert_array_equal(mask_utils.decode(decoded[0]), ids > 0)

    def test_touching_instances_have_interface_and_split_without_foreground_loss(self):
        ids = np.zeros((48, 64), np.int32)
        ids[8:40, 8:32] = 1
        ids[8:40, 32:56] = 2
        target = make_targets(ids)
        self.assertTrue(np.all(target['boundary'][8:40, 31:33]))
        self.assertEqual(int(target['foreground'].sum()), 32*48)
        self.assertEqual(float(target['distance'][ids == 1].max()), 1.)
        self.assertEqual(float(target['distance'][ids == 2].max()), 1.)
        probability = np.stack([target[k] for k in ('foreground', 'boundary', 'distance')])
        self.assertEqual(len(decode_instances(probability, min_area=20, mode='connected')), 1)
        decoded = decode_instances(probability, min_area=20, mode='watershed')
        self.assertEqual(len(decoded), 2)
        areas = mask_utils.area(decoded)
        self.assertEqual(int(areas.sum()), int((ids > 0).sum()))
        self.assertEqual(int(mask_utils.area(mask_utils.merge(decoded))), int(areas.sum()))
        overlap = mask_utils.iou([mask_utils.encode(np.asfortranarray((ids == i).astype(np.uint8))) for i in (1, 2)], decoded, [0]*2)
        self.assertTrue(np.all(overlap.max(axis=1) > .9))

    def test_multiple_distance_peaks_do_not_split_one_curved_filament(self):
        probability = np.zeros((3, 64, 64), np.float32)
        probability[0, 10:54, 28:32] = 1.
        probability[0, 48:54, 10:32] = 1.
        probability[2, 15, 29] = 1.
        probability[2, 35, 29] = 1.
        probability[2, 50, 15] = 1.
        decoded = decode_instances(probability, min_area=10)
        self.assertEqual(len(decoded), 1)
        self.assertEqual(int(mask_utils.area(decoded[0])), int(probability[0].sum()))

    def test_all_boundary_no_seed_falls_back_and_blank_output_is_empty(self):
        probability = np.zeros((3, 24, 24), np.float32)
        self.assertEqual(decode_instances(probability), [])
        probability[0, 4:20, 4:20] = .9
        probability[1] = .9
        self.assertEqual(len(decode_instances(probability, min_area=50)), 1)

    def test_undersized_watershed_children_preserve_qualifying_parent(self):
        ids = np.zeros((32, 32), np.int32)
        ids[6:26, 6:16] = 1
        ids[6:26, 16:26] = 2
        targets = make_targets(ids)
        probability = np.stack([targets[k] for k in ('foreground', 'boundary', 'distance')])
        self.assertEqual(len(decode_instances(probability, min_area=100)), 2)
        decoded = decode_instances(probability, min_area=300)
        self.assertEqual(len(decoded), 1)
        self.assertEqual(int(mask_utils.area(decoded[0])), 400)
        np.testing.assert_array_equal(mask_utils.decode(decoded[0]), ids > 0)

    def test_loss_handles_empty_and_ignored_pixels_and_semantic_unused_heads(self):
        ids = torch.zeros((2, 16, 16), dtype=torch.long)
        ids[0, 3:12, 3:12] = 1
        ids[0, 12, 3:12] = 2
        ids[1] = -1
        logits = torch.randn((2, 3, 16, 16), requires_grad=True)
        loss, metrics = loss_fn(logits, ids)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(value) for value in metrics.values()))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(float(logits.grad[1].abs().sum()), 0.)
        logits = torch.randn((2, 3, 16, 16), requires_grad=True)
        loss, metrics = loss_fn(logits, ids, head_mode='semantic')
        loss.backward()
        self.assertEqual(float(logits.grad[:, 1:].abs().sum()), 0.)
        empty_logits = torch.zeros((1, 3, 16, 16), requires_grad=True)
        empty_loss, _ = loss_fn(empty_logits, torch.zeros((1, 16, 16), dtype=torch.long))
        empty_loss.backward()
        self.assertTrue(torch.isfinite(empty_loss))

    def test_actual_resnet18_model_optimizer_step_without_weight_download(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            torch.manual_seed(23)
            model = NativeInstanceNet(encoder='resnet18', encoder_weights=None)
            inputs = torch.randn((2, 1, 64, 64))
            ids = torch.zeros((2, 64, 64), dtype=torch.long)
            ids[:, 10:40, 10:30] = 1
            ids[:, 10:40, 30:50] = 2
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            before = model.network.segmentation_head[0].weight.detach().clone()
            logits = model(inputs)
            self.assertEqual(tuple(logits.shape), (2, 3, 64, 64))
            loss, _ = loss_fn(logits, ids)
            loss.backward()
            optimizer.step()
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(float((model.network.segmentation_head[0].weight.detach()-before).abs().max()), 0.)
        finally:
            torch.set_num_threads(previous_threads)

    def test_sparse_boundary_dice_supplies_positive_edge_gradient(self):
        ids = torch.zeros((1, 128, 128), dtype=torch.long)
        ids[:, 60:68, 60:68] = 1
        targets = make_targets(ids[0].numpy())
        edges = torch.as_tensor(targets['boundary']).bool()
        self.assertLess(float(edges.float().mean()), .002)
        logits = torch.zeros((1, 3, 128, 128), requires_grad=True)
        with torch.no_grad():
            logits[:, 1] = -5.
        loss, metrics = loss_fn(logits, ids)
        self.assertGreater(float(metrics['boundary_dice']), .9)
        self.assertAlmostEqual(float(metrics['boundary_loss']),
                               float(metrics['boundary_bce']+metrics['boundary_dice']), places=6)
        loss.backward()
        self.assertTrue(torch.all(logits.grad[0, 1][edges] < 0))
        improved = logits.detach().clone()
        improved[0, 1][edges] = 5.
        _, better = loss_fn(improved, ids)
        self.assertLess(float(better['boundary_loss']), float(metrics['boundary_loss']))

    def test_rejects_invalid_probabilities_and_noninteger_instance_ids(self):
        with self.assertRaises(ValueError):
            make_targets(np.full((8, 8), .5))
        with self.assertRaises(ValueError):
            decode_instances(np.full((3, 8, 8), np.nan))
        with self.assertRaises(ValueError):
            decode_instances(np.zeros((3, 8, 8)), min_area=0)


if __name__ == '__main__':
    unittest.main()
