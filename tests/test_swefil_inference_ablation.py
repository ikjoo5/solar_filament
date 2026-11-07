"""CPU contracts for fixed-weight inference, geometry, scoring and safe resume."""
import csv
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from pycocotools import mask as mu
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import swefil_inference_ablation as worker


def probability(rectangles=()):
    out = np.zeros((2048, 2048), np.float32)
    for y, x, h, w, value in rectangles:
        out[y:y+h, x:x+w] = value
    return out


def rle(y=100, x=200, h=20, w=20):
    return mu.encode(np.asfortranarray(probability([(y, x, h, w, 1)]).astype(np.uint8)))


def test_recipe_selection_requires_baseline_and_rejects_unknown_duplicates():
    assert worker.select_recipes('raw,tta4') == ('baseline', 'raw', 'tta4')
    for value in ('', 'tta8', 'baseline,baseline'):
        with pytest.raises(ValueError, match='distinct recipes'):
            worker.select_recipes(value)


@pytest.mark.parametrize('view', ['base', 'h', 'v', 'hv'])
def test_tta_restores_native_coordinates_exactly(view):
    image = np.arange(35, dtype=np.float32).reshape(5, 7)
    restored = worker.transform(worker.transform(image, view), view)
    assert np.array_equal(restored, image)
    assert restored.flags.c_contiguous


def test_cache_is_lossless_and_committed_identity_checksum_are_required(tmp_path):
    path = tmp_path / 'view.npz'
    value = np.random.default_rng(42).random((7, 5), dtype=np.float32)
    actual, reused = worker.cached_probability(path, 'protocol', lambda: value, value.shape)
    assert not reused and np.array_equal(value, actual)
    actual, reused = worker.cached_probability(path, 'protocol', lambda: pytest.fail('cached'), value.shape)
    assert reused and np.array_equal(value, actual)
    with pytest.raises(ValueError, match='identity/checksum'):
        worker.cached_probability(path, 'changed-weights', lambda: value, value.shape)
    path.write_bytes(b'changed bytes')
    with pytest.raises(ValueError, match='missing or changed'):
        worker.cached_probability(path, 'protocol', lambda: value, value.shape)


def test_interrupted_probability_commit_recomputes_view(tmp_path, monkeypatch):
    path = tmp_path / 'view.npz'
    value = np.full((3, 5), .42, np.float32)
    envelope = worker.reviewed.envelope
    def interrupted(*args, **kwargs):
        raise RuntimeError('interrupt before commit')
    monkeypatch.setattr(worker.reviewed, 'envelope', interrupted)
    with pytest.raises(RuntimeError, match='before commit'):
        worker.cached_probability(path, 'identity', lambda: value, value.shape)
    assert path.is_file() and not path.with_suffix('.json').exists()
    assert not list(tmp_path.glob('*.pending'))
    monkeypatch.setattr(worker.reviewed, 'envelope', envelope)
    calls = []
    def produce():
        calls.append('recomputed')
        return value
    _, reused = worker.cached_probability(path, 'identity', produce, value.shape)
    assert not reused and calls == ['recomputed']


def test_cache_npz_failure_leaves_no_completed_record(tmp_path, monkeypatch):
    path = tmp_path / 'view.npz'
    def fail(stream, **kwargs):
        stream.write(b'incomplete')
        raise RuntimeError('interrupted array write')
    monkeypatch.setattr(worker.np, 'savez_compressed', fail)
    with pytest.raises(RuntimeError, match='array write'):
        worker.cached_probability(path, 'identity', lambda: np.zeros((2, 2), np.float32), (2, 2))
    assert not path.exists() and not path.with_suffix('.json').exists()
    assert not list(tmp_path.glob('*.pending'))


def test_decoder_preserves_native_coordinates_and_inclusive_area_threshold():
    value = probability([(1700, 1800, 15, 20, .65), (50, 50, 10, 20, .649)])
    masks = worker.masks_at_threshold(value, .65, 300)
    assert len(masks) == 1 and int(mu.area(masks[0])) == 300
    restored = mu.decode(masks[0])
    assert restored[1700:1715, 1800:1820].all()
    assert not restored[1800:1820, 1700:1715].any()
    diagonal = probability([(10, 10, 10, 15, .9), (20, 25, 10, 15, .9)])
    assert len(worker.masks_at_threshold(diagonal, .65, 300)) == 1


def test_score_uses_every_annotator_group_and_blank_image():
    groups = {'a-one': dict(image='one', masks=[rle()]),
              'b-one': dict(image='one', masks=[rle()]),
              'a-empty': dict(image='empty', masks=[rle()])}
    result = worker.score_predictions(groups, {'one': [rle()], 'empty': []})
    assert result['tp_pairs'] == 2 and result['fn'] == 1 and result['fp'] == 0
    assert result['pq'] == .8 and result['sq'] == 1
    assert result['prediction_instances'] == 1 and result['physical_images'] == 2
    assert len(result['per_group']) == 3
    with pytest.raises(ValueError, match='Every validation image'):
        worker.score_predictions(groups, {'one': [rle()]})


def test_iou_exactly_half_does_not_match():
    groups = {'g': dict(image='one', masks=[rle(h=20, w=20)])}
    result = worker.score_predictions(groups, {'one': [rle(h=20, w=40)]})
    assert result['tp_pairs'] == 0 and result['fp'] == 1 and result['fn'] == 1


def test_baseline_view_reused_in_tta_and_averaging_is_in_original_coordinates(tmp_path, monkeypatch):
    raw = np.zeros((2048, 2048), np.uint8)
    raw[10:30, 90:110] = 128
    monkeypatch.setattr(worker.reviewed.base, 'limb', lambda image: image)
    calls = []
    def predict(image, overlap):
        calls.append(overlap)
        result = image.astype(np.float32) / 255
        result[100:120, 200:220] += .1  # A view-frame perturbation must be unflipped.
        return result
    sources = {'a.jpeg': 'verified-image-sha'}
    baseline, reused = worker.probability_for_recipe(raw, 'a.jpeg', sources, 'baseline', tmp_path,
                                                    'protocol', predict)
    assert reused == 0 and len(calls) == 1
    averaged, reused = worker.probability_for_recipe(raw, 'a.jpeg', sources, 'tta4', tmp_path,
                                                    'protocol', predict)
    assert reused == 1 and len(calls) == 4
    expected = raw.astype(np.float32) / 255
    bump = probability([(100, 200, 20, 20, .1)])
    for view in ('base', 'h', 'v', 'hv'):
        expected += worker.transform(bump, view) / 4
    assert np.allclose(averaged, expected, atol=1e-7)
    again, reused = worker.probability_for_recipe(raw, 'a.jpeg', sources, 'baseline', tmp_path,
                                                 'protocol', lambda *_: pytest.fail('baseline changed'))
    assert reused == 1 and np.array_equal(again, baseline)


@pytest.fixture
def run_fixture(tmp_path, monkeypatch):
    images = tmp_path / 'images'
    images.mkdir()
    for name in ('one.jpeg', 'empty.jpeg'):
        (images / name).write_bytes(name.encode())
    sources = {p.name: worker.reviewed.digest(p) for p in sorted(images.iterdir())}
    groups = {'a-one': dict(image='one', masks=[rle()]),
              'b-one': dict(image='one', masks=[rle()]),
              'a-empty': dict(image='empty', masks=[rle()])}
    metadata = dict(config={}, checkpoint_selection=dict(selected_epoch=100))
    monkeypatch.setattr(worker.reviewed, 'training_metadata', lambda *_: metadata)
    monkeypatch.setattr(worker, 'validation_inputs', lambda *_a, **_k: (images, sources, groups, {}))
    monkeypatch.setattr(worker.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(worker.reviewed, 'configure_cuda', lambda: None)
    monkeypatch.setattr(worker.reviewed, 'runtime_identity', lambda: dict(device='cpu_fixture'))
    monkeypatch.setattr(worker.reviewed, 'read_image', lambda path, _sha:
                        np.full((2048, 2048), int(path.stem == 'one'), np.uint8))
    monkeypatch.setattr(worker.reviewed.base, 'limb', lambda image: image)
    monkeypatch.setattr(worker.reviewed, 'REVIEWED_PQ', .8)
    calls = []
    monkeypatch.setattr(worker.reviewed, 'load_reviewed_checkpoint', lambda *_:
                        dict(model_state_dict={'fixture': 1}))
    class Model:
        def load_state_dict(self, state, strict):
            assert state == {'fixture': 1} and strict
        def to(self, device):
            return self
        def eval(self):
            return self
    def build(encoder, pretrained):
        assert encoder == 'tu-convnext_tiny' and not pretrained
        calls.append('model')
        return Model()
    monkeypatch.setattr(worker.reviewed.base, 'build_model', build)
    def inference(_model, image, _device, **kwargs):
        assert kwargs['tile'] == 1024 and kwargs['batch'] == 1
        assert kwargs['amp_dtype'] is worker.torch.float16
        calls.append(('predict', kwargs['overlap']))
        return probability([(100, 200, 20, 20, .8)]) if image[0, 0] else probability()
    monkeypatch.setattr(worker.reviewed.base, 'predict_foreground', inference)
    return (tmp_path / 'training', tmp_path / 'checkpoint.pt', tmp_path / 'output'), calls


def test_run_outputs_all_decoders_blank_images_and_resume_uses_cache(run_fixture):
    args, calls = run_fixture
    result = worker.run(*args)
    assert result['status'] == 'complete' and result['configurations_scored'] == 36
    assert result['baseline']['pq'] == .8 and result['baseline']['fn'] == 1
    assert result['completed_recipes'] == ['baseline', 'dense', 'raw', 'tta4']
    assert result['recipes'] == ['baseline', 'tta4', 'dense', 'raw']
    assert calls.count('model') == 1
    assert len([c for c in calls if isinstance(c, tuple)]) == 12
    with zipfile.ZipFile(result['archive']) as archive:
        assert len([name for name in archive.namelist() if name.startswith('masks/')]) == 36
        assert not any(name.startswith('probabilities/') or name.endswith('.pt') for name in archive.namelist())
        assert 'validation_gt.csv' in archive.namelist()
        assert len(json.loads(archive.read('scores.json'))) == 36
    calls.clear()
    resumed = worker.run(*args)
    assert calls == [] and resumed['best'] == result['best']
    changed = dict(json.loads((args[2] / 'identity.json').read_text())['payload'])
    changed['runtime'] = {'device': 'other'}
    with pytest.raises(ValueError, match='identity/checksum'):
        worker.run(*args, recipes=('baseline', 'tta4'))


def test_mismatched_baseline_stops_before_other_inference(run_fixture, monkeypatch):
    args, calls = run_fixture
    monkeypatch.setattr(worker.reviewed, 'REVIEWED_PQ', .4)
    result = worker.run(*args)
    assert result['status'] == 'baseline_mismatch'
    assert result['completed_recipes'] == ['baseline']
    assert result['configurations_scored'] == 9
    assert len([c for c in calls if isinstance(c, tuple)]) == 2
    assert Path(result['archive']).is_file()


def test_validation_source_hash_rejected_before_loading_annotations(tmp_path):
    training = tmp_path / 'training'
    (training / 'train').mkdir(parents=True)
    annotations = tmp_path / 'annotations.json'
    annotations.write_text('this is intentionally not json')
    identity = dict(sources=dict(annotations='different-hash'))
    worker.reviewed.envelope(training / 'train/identity.json', worker.reviewed.TRAINING_IDENTITY, identity)
    with pytest.raises(ValueError, match='validation source changed'):
        worker.validation_inputs(training, dict(annotations=str(annotations), train_split='a', val_split='b'))
