"""CPU fixtures: annotation separation, geometric quality, and epoch resume."""
from contextlib import redirect_stdout
import csv
import io
import json
import gzip
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from pycocotools import mask as mu
import torch
from torch import nn
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import fixed_split_mask_quality as quality


def rectangle(x, y, width, height, size=2048):
    value = np.zeros((size, size), np.uint8)
    value[y:y+height, x:x+width] = 1
    return mu.encode(np.asfortranarray(value))


def proposal(value, confidence):
    return dict(rle=value['counts'].decode(), confidence=confidence)


class TinyQuality(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Dropout(.2))
        self.head = nn.Linear(5, 2)

    def forward(self, inputs):
        return self.head(self.backbone(inputs))


class QualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_annotation_siblings_are_indivisible(self):
        data = dict(images=[dict(id='a1', file_name='a.jpeg'), dict(id='a2', file_name='a.jpeg'),
                            dict(id='b1', file_name='b.jpeg')])
        with tempfile.TemporaryDirectory() as folder:
            train, val = Path(folder)/'train.csv', Path(folder)/'val.csv'
            for path, rows in [(train, [('a1', 'a.jpeg'), ('a2', 'a.jpeg')]), (val, [('b1', 'b.jpeg')])]:
                with path.open('w', newline='') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(['image_id', 'file_name'])
                    writer.writerows(rows)
            roles = quality.read_splits(data, train, val, expected_counts=(1, 1))
            self.assertEqual(roles, dict(train=['a.jpeg'], validation=['b.jpeg']))
            train.write_text('image_id,file_name\na1,a.jpeg\n')
            with self.assertRaisesRegex(ValueError, 'sibling'):
                quality.read_splits(data, train, val, expected_counts=(1, 1))

    def test_soft_targets_equal_rater_weight_strict_half_and_empty_group(self):
        p = rectangle(100, 100, 20, 20)
        half = rectangle(100, 100, 10, 20)
        unrelated = rectangle(400, 400, 20, 20)
        values = quality.soft_targets([p], [[p, unrelated, unrelated], [half], []])
        np.testing.assert_allclose(values, [[1/3, 1/3]])
        # Adding objects to one rater must not change the weight of that rater.
        np.testing.assert_allclose(quality.soft_targets([p], [[p], [half], []]), values)
        no = quality.soft_targets([], [[p], []])
        self.assertEqual(no.shape, (0, 2))

    def test_roi_mask_and_image_share_letterbox_coordinates_at_border(self):
        mask = rectangle(0, 17, 12, 80, size=128)
        decoded = mu.decode(mask)
        gray = decoded*173
        bbox = quality.roi_box(decoded)
        self.assertEqual(bbox[0], 0)
        self.assertEqual(bbox[1], 0)
        self.assertEqual(bbox[3], 123)
        result = quality.roi_array(gray, mask)
        self.assertEqual(result.shape, (2, 256, 256))
        binary = result[1] > 0
        self.assertGreater(float(result[0][binary].mean()), 155)
        visible = result[0] > 86
        self.assertGreater((visible & binary).sum()/(visible | binary).sum(), .95)
        self.assertEqual(set(np.unique(result[1])), {0, 255})
        self.assertTrue(np.all(result[:, :, :50] == 0))

    def test_four_channel_start_preserves_pretrained_rgb_and_ignores_mask(self):
        backbone = SimpleNamespace(conv1=nn.Conv2d(3, 4, 3, padding=1, bias=False))
        old = backbone.conv1
        rgb = torch.randn(2, 3, 12, 12)
        expected = old(rgb).detach()
        quality.expand_conv(backbone)
        with torch.no_grad():
            observed = backbone.conv1(torch.cat([rgb, torch.randn(2, 1, 12, 12)], 1))
        torch.testing.assert_close(expected, observed)
        self.assertEqual(int(torch.count_nonzero(backbone.conv1.weight[:, 3])), 0)

    def test_actual_timm_resnet_has_two_outputs_and_trainable_mask_channel(self):
        model = quality.QualityModel(pretrained=False).eval()
        inputs = torch.rand(2, 4, 64, 64)
        prediction = model(inputs)
        self.assertEqual(tuple(prediction.shape), (2, 2))
        prediction.sum().backward()
        self.assertGreater(float(model.backbone.conv1.weight.grad[:, 3].abs().sum()), 0)

    def test_quality_order_changes_ownership_and_duplicate_is_suppressed(self):
        first, second = rectangle(100, 100, 30, 30), rectangle(120, 100, 30, 30)
        items = [proposal(first, .9), proposal(second, .1), proposal(first, .8)]
        baseline = quality.decode(items, [.9, .1, .8], .05, min_area=1)
        learned = quality.decode(items, [.1, .9, .1], .05, min_area=1)
        self.assertEqual(len(baseline), 2)
        self.assertEqual(len(learned), 2)
        self.assertEqual(int(mu.area(baseline[0])), 900)
        self.assertEqual(int(mu.area(baseline[1])), 600)
        self.assertEqual(int(mu.area(learned[0])), 900)
        self.assertEqual(int(mu.area(learned[1])), 600)
        self.assertEqual(int((mu.decode(baseline[0]) & mu.decode(baseline[1])).sum()), 0)
        self.assertEqual(int(mu.decode(baseline[0])[110, 125]), 1)
        self.assertEqual(int(mu.decode(learned[1])[110, 125]), 0)
        self.assertEqual(int(mu.decode(learned[0])[110, 125]), 1)

    def test_train_balancing_never_limits_validation_selection(self):
        instances = [dict(confidence=1-i/100, target=[.8, 1] if i >= 80 else [0, 0]) for i in range(100)]
        selected = quality.select_training(instances, 10)
        self.assertEqual(len(selected), 10)
        self.assertEqual(sum(instances[i]['target'][1] > 0 for i in selected), 5)

    def test_default_schedule_is_twenty_and_cpu_precision_is_float32(self):
        args = quality.parser().parse_args(['--phase', 'train'])
        self.assertEqual((args.epochs, args.run_until), (20, 20))
        with quality.autocast_for(torch.device('cpu')):
            value = nn.Linear(3, 2)(torch.ones(1, 3))
        self.assertEqual(value.dtype, torch.float32)

    def test_proposal_payload_modification_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'raw.json.gz'
            value = dict(identity=dict(role='train'), instances=[dict(target=[.8, 1])])
            quality.atomic_raw(path, value)
            self.assertEqual(quality.read_raw(path), value)
            with gzip.open(path, 'rt') as stream:
                changed = json.load(stream)
            changed['instances'][0]['target'] = [0, 0]
            with gzip.open(path, 'wt') as stream:
                json.dump(changed, stream)
            with self.assertRaisesRegex(ValueError, 'checksum'):
                quality.read_raw(path)

    def test_confidence_threshold_grid_can_outperform_fixed025(self):
        gt = rectangle(100, 100, 30, 30)
        wrong = rectangle(400, 400, 30, 30)
        items = [proposal(gt, .2), proposal(wrong, .27)]
        reports = quality.threshold_reports({'a': items}, {'a': [.2, .27]}, {'rater': [gt]}, {'rater': 'a'})
        self.assertEqual([row['threshold'] for row in reports], quality.THRESHOLDS)
        self.assertGreater(max(row['pq'] for row in reports), next(row['pq'] for row in reports if row['threshold'] == .25))

    def test_threshold_prefix_optimization_matches_independent_ownership_decodes(self):
        items = [proposal(rectangle(100+x, 100, width, 30), score) for x, width, score in
                 [(0, 30, .35), (30, 30, .3), (20, 20, .25), (45, 30, .15), (0, 30, .2), (60, 30, .11)]]
        scores = [row['confidence'] for row in items]
        candidates = quality.decode(items, scores, .1, return_scores=True)
        self.assertNotIn(.25, [score for _, score in candidates])  # Entirely clipped by two higher masks.
        for threshold in quality.THRESHOLDS:
            independent = quality.decode(items, scores, threshold)
            prefix = [value for value, score in candidates if score >= threshold]
            self.assertEqual([value['counts'] for value in independent], [value['counts'] for value in prefix])

    def test_prepare_resumes_raw_cache_and_keeps_validation_proposals_uncapped(self):
        import submission_inference
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output, cache, images = root/'drive', root/'local', root/'images'
            images.mkdir()
            import cv2
            for name in ['a.jpeg', 'b.jpeg']:
                self.assertTrue(cv2.imwrite(str(images/name), np.full((2048, 2048), 100, np.uint8)))
            data = dict(images=[dict(id='a1', file_name='a.jpeg'), dict(id='b1', file_name='b.jpeg')],
                annotations=[dict(image_id='a1', segmentation=[[100, 100, 130, 100, 130, 130, 100, 130]]),
                             dict(image_id='b1', segmentation=[[100, 100, 130, 100, 130, 130, 100, 130]])])
            ann = root/'ann.json'
            ann.write_text(json.dumps(data))
            args = SimpleNamespace(output=str(output), cache=str(cache), annotations=str(ann), images=str(images),
                                   train_split='unused', val_split='unused', yolo='frozen', device='cpu', train_cap=2, seed=42)
            views = [dict(view='base', max_det_reached=False, instances=[proposal(rectangle(x, 100, 30, 30), .3)
                                                                       for x in [100, 200, 300, 400]])]
            views[0]['instances'].append(proposal(mu.encode(np.zeros((2048, 2048), np.uint8, order='F')), .1))
            with patch.object(quality, 'source_identity', return_value={'fixture': 1}), \
                 patch.object(quality, 'read_splits', return_value=dict(train=['a.jpeg'], validation=['b.jpeg'])), \
                 patch.object(submission_inference.InferenceProvider, 'geometric', return_value=views) as detector, \
                 redirect_stdout(io.StringIO()):
                quality.prepare(args)
                self.assertEqual(detector.call_count, 2)
                quality.prepare(args)
                self.assertEqual(detector.call_count, 2)
            corpus = json.loads((output/'corpus.json').read_text())
            train = quality.QualityDataset(corpus['records'], output, cache, 'train')
            val = quality.QualityDataset(corpus['records'], output, cache, 'validation')
            self.assertEqual((len(train), len(val)), (2, 4))
            self.assertEqual(train[0][0].shape, (4, 256, 256))
            self.assertEqual(list(output.glob('*.npy')), [])
            self.assertEqual(quality.read_raw(output/'proposals/a.json.gz')['empty_skipped'], 1)

    def test_real_optimizer_resume_matches_uninterrupted_two_epochs(self):
        inputs = torch.arange(48, dtype=torch.float32).reshape(12, 4)/48
        targets = torch.stack([inputs[:, 0], (inputs[:, 1] > .5).float()], 1)
        dataset = TensorDataset(inputs, targets)
        def reset():
            random.seed(31)
            np.random.seed(31)
            torch.manual_seed(31)
            return TinyQuality()
        def evaluate(model, epoch):
            # No validation optimizer step; deterministic fixture checkpoint score.
            with torch.no_grad():
                result = float(model(inputs).sigmoid()[:, 0].mean())
            return dict(selected=dict(pq=result, threshold=.2))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            full, resumed = root/'full', root/'resumed'
            full.mkdir()
            resumed.mkdir()
            args = SimpleNamespace(output=str(full), lr=1e-3, epochs=2, run_until=2, batch_size=4,
                                   freeze_epochs=1, seed=31, validate_every=1)
            with redirect_stdout(io.StringIO()):
                model_full = reset()
                quality.fit(model_full, dataset, args, torch.device('cpu'), dict(fixture=1), evaluate)
                args.output, args.run_until = str(resumed), 1
                model_one = reset()
                quality.fit(model_one, dataset, args, torch.device('cpu'), dict(fixture=1), evaluate)
                args.run_until = 2
                model_resumed = TinyQuality()  # New process would initialize a new model too.
                quality.fit(model_resumed, dataset, args, torch.device('cpu'), dict(fixture=1), evaluate)
            for key, expected in model_full.state_dict().items():
                torch.testing.assert_close(expected, model_resumed.state_dict()[key], atol=0, rtol=0)
            checkpoint = torch.load(resumed/'last.pt', weights_only=False)
            self.assertEqual(checkpoint['epoch'], 2)
            self.assertEqual([row['proposals_seen'] for row in checkpoint['history']], [12, 12])
            self.assertEqual(checkpoint['scheduler']['last_epoch'], 2)
            with self.assertRaisesRegex(ValueError, 'settings/source'):
                quality.fit(TinyQuality(), dataset, args, torch.device('cpu'), dict(fixture=2), evaluate)

    def test_completed_epoch_restores_best_after_interruption_between_last_and_best(self):
        dataset = TensorDataset(torch.ones(4, 4), torch.zeros(4, 2))
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(output=folder, lr=1e-3, epochs=2, run_until=1, batch_size=4,
                                   freeze_epochs=0, seed=42, validate_every=1)
            evaluate = lambda model, epoch: dict(selected=dict(pq=.4, threshold=.2))
            original = quality.atomic_torch
            def interrupted(path, value):
                if path.name == 'best.pt':
                    raise KeyboardInterrupt('session stopped after completed last.pt')
                return original(path, value)
            with patch.object(quality, 'atomic_torch', side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt):
                    quality.fit(TinyQuality(), dataset, args, torch.device('cpu'), {}, evaluate)
            last = torch.load(Path(folder)/'last.pt', weights_only=False)
            self.assertEqual(last['epoch'], 1)
            # This ahead-of-commit best must also be replaced from the committed last.
            torch.save(dict(epoch=99, selected=dict(pq=.9)), Path(folder)/'best.pt')
            with redirect_stdout(io.StringIO()):
                history, pq = quality.fit(TinyQuality(), dataset, args, torch.device('cpu'), {}, evaluate)
            best = torch.load(Path(folder)/'best.pt', weights_only=False)
            self.assertEqual(best['epoch'], 1)
            self.assertEqual(pq, .4)
            self.assertEqual(len(history), 1)
            self.assertEqual(json.loads((Path(folder)/'history.json').read_text())[0]['epoch'], 1)


if __name__ == '__main__':
    unittest.main()
