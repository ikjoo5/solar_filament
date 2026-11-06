from pathlib import Path
import gzip
import json
import sys

import cv2
import numpy as np
from pycocotools import mask as mu
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import chase_best84_evaluation as ev


def encoded(region, shape=(16, 16)):
    array = np.zeros(shape, np.uint8)
    array[region] = 1
    return mu.encode(np.asfortranarray(array))


def test_official_score_repeats_predictions_for_independent_annotators():
    a = encoded(np.s_[1:5, 1:5])
    b = encoded(np.s_[9:13, 9:13])
    groups = {'annotator-a': dict(image='frame', masks=[a]),
              'annotator-b': dict(image='frame', masks=[a, b])}
    result = ev.score_predictions(groups, {'frame': [a]})
    assert result['tp_pairs'] == 2 and result['fn'] == 1 and result['fp'] == 0
    assert result['pq'] == .8
    assert result['matched_iou_sum'] == 2


def test_official_score_threshold_is_strict_and_empty_predictions_count_fn():
    a = encoded(np.s_[0:4, 0:4])
    half = encoded(np.s_[0:2, 0:4])
    groups = {'a-frame': dict(image='frame', masks=[a])}
    result = ev.score_predictions(groups, {'frame': [half]})
    assert (result['tp_pairs'], result['fp'], result['fn']) == (0, 1, 1)
    assert ev.score_predictions(groups, {'frame': []})['fn'] == 1
    with unittest.TestCase().assertRaises(ValueError):
        ev.score_predictions(groups, {})


def test_official_all_pairs_is_not_hungarian():
    # Duplicate GT across this diagnostic group makes the distinction explicit.
    a = encoded(np.s_[0:4, 0:4])
    result = ev.score_predictions({'group': dict(image='frame', masks=[a, a])}, {'frame': [a]})
    assert result['tp_pairs'] == 2


def test_model_digest_tracks_parameters_and_buffers_without_mutation():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
    model.train()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    first = ev.model_digest(model)
    assert first == ev.model_digest(model)
    assert model.training
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
    with torch.no_grad():
        model[1].num_batches_tracked.add_(1)
    assert first != ev.model_digest(model)


def cache_bundle(filename, image_sha):
    stem = Path(filename).stem
    frames = {}
    for family, views in {
        'geometric': ['base', 'scale', 'hflip', 'vflip'],
        'preprocessing': ['limb', 'clahe16', 'clahe32', 'percentile', 'unsharp', 'gaussian', 'bilateral', 'blackhat'],
        'tiled': [f'tile{i}' for i in range(18)],
    }.items():
        frames[family] = [dict(image=stem, view=view, instances=[], shape=[2048, 2048],
                               coordinate_space='original') for view in views]
    return dict(schema_version=1, frames={stem: frames}, image_sha256={filename: image_sha},
                sources={'fixture': 'not production'})


def test_frozen_cache_rejects_cross_image_views_and_missing_image_hash(tmp_path):
    path = tmp_path / 'cache.json.gz'
    data = cache_bundle('frame.jpeg', 'a' * 64)
    path.write_bytes(gzip.compress(json.dumps(data).encode()))
    assert ev.load_yolo_cache(path, ['frame.jpeg'])['image_sha256']['frame.jpeg'] == 'a' * 64
    data['frames']['frame']['geometric'][0]['image'] = 'other'
    path.write_bytes(gzip.compress(json.dumps(data).encode()))
    with unittest.TestCase().assertRaisesRegex(ValueError, 'views'):
        ev.load_yolo_cache(path, ['frame.jpeg'])
    data = cache_bundle('frame.jpeg', 'a' * 64)
    data['image_sha256'] = {}
    path.write_bytes(gzip.compress(json.dumps(data).encode()))
    with unittest.TestCase().assertRaisesRegex(ValueError, 'hash missing'):
        ev.load_yolo_cache(path, ['frame.jpeg'])


def test_same_model_resume_and_model_input_change_rejection(tmp_path, monkeypatch):
    images = tmp_path / 'images'
    images.mkdir()
    image_path = images / 'frame.jpeg'
    assert cv2.imwrite(str(image_path), np.full((2048, 2048), 120, np.uint8))
    image_sha = ev.digest(image_path)
    path = tmp_path / 'cache.json.gz'
    path.write_bytes(gzip.compress(json.dumps(cache_bundle(image_path.name, image_sha)).encode()))
    truth = encoded(np.s_[10:20, 10:20], (2048, 2048))
    cohort = dict(names=[image_path.name], groups={'a-frame': dict(image='frame', masks=[truth])},
                  source_sha256={'fixture': 'synthetic-only'})
    monkeypatch.setattr(ev, 'load_cohort', lambda cfg: cohort)
    monkeypatch.setattr(ev, 'finalize_frame', lambda stem, stages: [truth])
    calls = []

    class FakeProvider:
        def __init__(self, *args, **kwargs):
            assert kwargs['native_model'] is model

        def native(self, stem, pixels, **kwargs):
            calls.append(stem)
            return dict(**kwargs, size=[2048, 2048], unions={}, components={}, tta={}, view_unions={})

    model = torch.nn.Conv2d(3, 2, 1).train()
    cfg = dict(images=str(images), yolo_cache=str(path), output_root=str(tmp_path / 'out'), device='cpu')
    report = ev.evaluate_candidate(model, cfg, 'epoch000', provider_factory=FakeProvider)
    assert report['pq'] == 1 and calls == ['frame'] and model.training
    second = ev.evaluate_candidate(model, cfg, 'epoch000', provider_factory=FakeProvider)
    assert second['pq'] == 1 and second['reused_images'] == 1 and calls == ['frame']
    # The identical native model is reused for a different YOLO combination.
    different = cache_bundle(image_path.name, image_sha)
    different['sources'] = {'new_yolo': 'fixture'}
    other_cache = tmp_path / 'new_yolo.json.gz'
    other_cache.write_bytes(gzip.compress(json.dumps(different).encode()))
    third = ev.evaluate_candidate(model, dict(cfg, yolo_cache=str(other_cache),
        comparison_role='new_yolo_old_native'), 'new_yolo', provider_factory=FakeProvider)
    assert third['native_identity'] == report['native_identity']
    assert third['identity'] != report['identity'] and calls == ['frame']
    # A damaged completed cache can be reconstructed from saved native output.
    completed = tmp_path / 'out/validation/epoch000/completed/frame.json.gz'
    completed.write_bytes(b'corrupt')
    ev.evaluate_candidate(model, cfg, 'epoch000', provider_factory=FakeProvider)
    assert calls == ['frame']
    with torch.no_grad():
        model.bias.add_(.1)
    with unittest.TestCase().assertRaisesRegex(ValueError, 'model/data/code/runtime changed'):
        ev.evaluate_candidate(model, cfg, 'epoch000', provider_factory=FakeProvider)
    assert cv2.imwrite(str(image_path), np.full((2048, 2048), 121, np.uint8))
    with unittest.TestCase().assertRaisesRegex(ValueError, 'image differs'):
        ev.evaluate_candidate(model, cfg, 'other', provider_factory=FakeProvider)


def test_interrupted_evaluation_restores_model_mode(tmp_path, monkeypatch):
    images = tmp_path / 'images'
    images.mkdir()
    image_path = images / 'frame.jpeg'
    assert cv2.imwrite(str(image_path), np.full((2048, 2048), 120, np.uint8))
    bundle = cache_bundle(image_path.name, ev.digest(image_path))
    path = tmp_path / 'cache.json'
    path.write_text(json.dumps(bundle))
    monkeypatch.setattr(ev, 'load_cohort', lambda cfg: dict(names=[image_path.name],
        groups={}, source_sha256={'fixture': 'synthetic-only'}))

    class InterruptedProvider:
        def __init__(self, *args, **kwargs):
            pass
        def native(self, *args, **kwargs):
            raise KeyboardInterrupt()

    model = torch.nn.Linear(1, 1).train()
    cfg = dict(images=str(images), yolo_cache=str(path), output_root=str(tmp_path/'out'), device='cpu')
    with unittest.TestCase().assertRaises(KeyboardInterrupt):
        ev.evaluate_candidate(model, cfg, 'epoch000', provider_factory=InterruptedProvider)
    assert model.training


class _PatchContext:
    def __init__(self, stack):
        self.stack = stack

    def setattr(self, obj, name, value):
        self.stack.enter_context(patch.object(obj, name, value))


class EvaluationTests(unittest.TestCase):
    def test_official_annotators(self):
        test_official_score_repeats_predictions_for_independent_annotators()

    def test_strict_threshold_and_empty(self):
        test_official_score_threshold_is_strict_and_empty_predictions_count_fn()

    def test_all_pairs(self):
        test_official_all_pairs_is_not_hungarian()

    def test_tensor_hash(self):
        test_model_digest_tracks_parameters_and_buffers_without_mutation()

    def test_cache_origin(self):
        with tempfile.TemporaryDirectory() as root:
            test_frozen_cache_rejects_cross_image_views_and_missing_image_hash(Path(root))

    def test_resume_and_bindings(self):
        with tempfile.TemporaryDirectory() as root, ExitStack() as stack:
            test_same_model_resume_and_model_input_change_rejection(Path(root), _PatchContext(stack))

    def test_interrupt_restores_mode(self):
        with tempfile.TemporaryDirectory() as root, ExitStack() as stack:
            test_interrupted_evaluation_restores_model_mode(Path(root), _PatchContext(stack))


if __name__ == '__main__':
    unittest.main()
