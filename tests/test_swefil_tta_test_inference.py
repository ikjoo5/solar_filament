"""CPU contracts for the measured four-view test recipe and shared persistence."""
import csv
import json
from pathlib import Path
import sys
import zipfile

import cv2
import numpy as np
from pycocotools import mask as mu
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import swefil_tta_test_inference as worker
import swefil_inference_ablation as ablation

shared = worker.shared


def probability(nonempty=True):
    value = np.zeros((2048, 2048), np.float32)
    if nonempty:
        value[1700:1715, 1800:1820] = .65
    return value


def test_inverse_alignment_and_float_mean_not_mask_voting(monkeypatch):
    raw = np.zeros((2048, 2048), np.uint8)
    raw[40:60, 90:110] = 64
    raw[1100:1131, 1300:1353] = 96
    monkeypatch.setattr(shared.base, 'limb', lambda image: image)
    calls = []

    def predictor(image):
        view = worker.RECIPE['views'][len(calls)]
        assert np.array_equal(image, ablation.transform(raw, view))
        offset = len(calls) * .1
        calls.append(view)
        return image.astype(np.float32) / 255 + np.float32(offset)

    actual = worker.tta_probability(raw, predictor)
    expected = raw.astype(np.float32) / 255
    expected = ((expected + (expected + .1)) + (expected + .2)) + (expected + .3)
    expected /= 4
    assert calls == ['base', 'h', 'v', 'hv']
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32


def test_tta_matches_reviewed_validation_averaging_byte_exact(tmp_path, monkeypatch):
    raw = np.zeros((2048, 2048), np.uint8)
    raw[40:77, 100:151] = 70
    monkeypatch.setattr(shared.base, 'limb', lambda image: image)
    calls = []

    def predict(image):
        offset = np.float32(.1 * len(calls)); calls.append(1)
        return image.astype(np.float32) / 255 + offset

    test_result = worker.tta_probability(raw, predict)
    calls.clear()
    val_result, reused = ablation.probability_for_recipe(
        raw, 'fixture.jpeg', {'fixture.jpeg': 'fixture_sha'}, 'tta4',
        tmp_path, 'fixture_identity', lambda image, overlap: predict(image))
    assert reused == 0 and len(calls) == 4
    np.testing.assert_array_equal(test_result, val_result)


@pytest.fixture
def scenario(tmp_path):
    images = tmp_path / 'images'; images.mkdir()
    for name in ('a.jpeg', 'b.jpeg'):
        (images / name).write_bytes(name.encode())
    sources = {p.name: shared.digest(p) for p in sorted(images.iterdir())}
    protocol = dict(version='cpu_tta_fixture', recipe=worker.RECIPE.copy(), source_images=sources,
                    checkpoint_sha256=worker.CHECKPOINT_SHA, runtime={'device': 'cpu'},
                    validation_reference=worker.VALIDATION_REFERENCE,
                    checkpoint_native_validation={'pq': shared.REVIEWED_PQ})
    output = tmp_path / 'output'

    def reader(path, checksum):
        assert shared.digest(path) == checksum
        return path.name

    return images, output, protocol, reader


def invoke(scenario, predictor, **kwargs):
    images, output, protocol, reader = scenario
    return shared._run_images(images, output, protocol, predictor, reader,
                              profile=worker.PROFILE, **kwargs)


def test_tta_export_provenance_empty_images_and_completed_resume(scenario):
    result = invoke(scenario, lambda name: probability(name == 'a.jpeg'))
    assert result['recipe'] == worker.RECIPE
    assert result['images_completed'] == result['images_total'] == 2
    assert result['empty_images'] == ['b.jpeg'] and result['prediction_instances'] == 1
    assert result['overlap_pixels'] == 0
    assert result['validation_reference']['pq'] == worker.REVIEWED_PQ
    assert result['checkpoint_native_validation']['pq'] == shared.REVIEWED_PQ
    assert Path(result['csv']).name == worker.CSV_NAME
    assert Path(result['archive']).name == worker.ARCHIVE_NAME
    assert not (scenario[1] / shared.CSV_NAME).exists()
    with zipfile.ZipFile(result['archive']) as archive:
        manifest = json.loads(archive.read('submission_manifest.json'))
        assert manifest['recipe'] == worker.RECIPE
        assert worker.CSV_NAME in archive.namelist()
        assert 'predictions/b.jpeg.json' in archive.namelist()
        assert 'code/swefil_tta_test_inference.py' in archive.namelist()
        assert 'code/swefil_inference_ablation.py' in archive.namelist()
        assert not any(x.endswith(('.npz', '.pt', '.jpeg')) for x in archive.namelist())
    with Path(result['csv']).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    masks = shared.validate_masks([row['segmentation_rle'] for row in rows])
    assert len(masks) == 1 and int(mu.area(masks[0])) == 300
    again = invoke(scenario, lambda *_: pytest.fail('Completed images must not be inferred again'))
    assert again['csv_sha256'] == result['csv_sha256']
    assert shared.RECIPE['tta'] is False and shared.CSV_NAME != worker.CSV_NAME


def test_interruption_only_repeats_unfinished_image(scenario):
    def interrupted(name):
        if name == 'b.jpeg':
            raise RuntimeError('interrupted view')
        return probability()
    with pytest.raises(RuntimeError, match='interrupted view'):
        invoke(scenario, interrupted)
    assert not (scenario[1] / worker.CSV_NAME).exists()
    assert not (scenario[1] / 'latest_result.json').exists()
    calls = []
    def resumed(name):
        calls.append(name)
        return probability(False)
    result = invoke(scenario, resumed)
    assert calls == ['b.jpeg'] and result['images_completed'] == 2


def test_baseline_prediction_cache_cannot_be_reused_for_tta(scenario):
    images, output, protocol, reader = scenario
    baseline = dict(protocol, recipe=shared.RECIPE.copy())
    shared._run_images(images, output, baseline, lambda _: probability(), reader)
    before = (output / shared.CSV_NAME).read_bytes()
    with pytest.raises(ValueError, match='identity/checksum'):
        invoke(scenario, lambda *_: pytest.fail('Different recipe must not run'))
    assert not (output / worker.CSV_NAME).exists()
    assert (output / shared.CSV_NAME).read_bytes() == before


@pytest.mark.parametrize('key,value', [
    ('runtime', {'device': 'different'}),
    ('checkpoint_sha256', 'different_weight'),
    ('source_images', {'other.jpeg': 'changed'}),
    ('source_code', {'swefil_tta_test_inference.py': 'changed-code'}),
])
def test_changed_prediction_identity_cannot_resume(scenario, key, value):
    invoke(scenario, lambda _: probability())
    images, output, protocol, reader = scenario
    with pytest.raises(ValueError, match='identity/checksum'):
        invoke((images, output, dict(protocol, **{key: value}), reader), lambda _: probability())


def test_profile_recipe_must_match_protocol(scenario):
    images, output, protocol, reader = scenario
    protocol = dict(protocol, recipe=dict(worker.RECIPE, threshold=.5))
    with pytest.raises(ValueError, match='profile identity/checksum'):
        invoke((images, output, protocol, reader), lambda _: probability())
    assert not output.exists()


def test_export_profile_must_match_protocol(scenario):
    images, output, protocol, reader = scenario
    protocol = dict(protocol, export_profile=dict(csv_name='wrong.csv',
        archive_name=worker.ARCHIVE_NAME, source_files=list(worker.SOURCE_FILES)))
    with pytest.raises(ValueError, match='Export profile identity/checksum'):
        invoke((images, output, protocol, reader), lambda _: probability())
    assert not output.exists()


def test_default_inventory_rejects_incomplete_180_image_test_set(tmp_path):
    for i in range(179):
        (tmp_path / f'image{i:03}.jpeg').write_bytes(b'fixture')
    with pytest.raises(ValueError, match='exact official 180'):
        shared.inventory(tmp_path)


def test_model_boundary_uses_four_views_and_cached_run_never_builds_model(tmp_path, monkeypatch):
    images = tmp_path / 'images'; images.mkdir()
    raw = np.full((2048, 2048), 100, np.uint8)
    raw[1700:1715, 1800:1820] = 0
    image = images / 'fixture.png'; cv2.imwrite(str(image), raw)
    sources = {image.name: shared.digest(image)}
    metadata = dict(config={'test_images': str(images)}, checkpoint=str(tmp_path / 'weights.pt'),
                    training_runtime={'device': 'cuda'},
                    validation={'pq': shared.REVIEWED_PQ, 'threshold': .65, 'min_area': 300},
                    checkpoint_selection={'selected_epoch': 100})
    monkeypatch.setattr(shared, 'training_metadata', lambda *_: metadata)
    monkeypatch.setattr(shared, 'inventory', lambda _: sources)
    monkeypatch.setattr(worker.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(shared, 'configure_cuda', lambda: None)
    monkeypatch.setattr(shared, 'runtime_identity', lambda: {'device': 'cuda', 'gpu': 'Tesla T4'})
    calls = []
    monkeypatch.setattr(shared.base, 'limb', lambda image: calls.append('limb') or image)
    monkeypatch.setattr(shared, 'load_reviewed_checkpoint',
                        lambda _: calls.append('load') or {'model_state_dict': {'fixture': 1}})
    class Model:
        def load_state_dict(self, state, strict):
            assert state == {'fixture': 1} and strict
        def to(self, device):
            assert str(device) == 'cuda'; return self
        def eval(self):
            return self
    def build(encoder, pretrained):
        assert encoder == 'tu-convnext_tiny' and pretrained is False
        calls.append('build'); return Model()
    monkeypatch.setattr(shared.base, 'build_model', build)
    def inference(model, image, device, **options):
        assert options == dict(tile=1024, overlap=.25, batch=1, amp_dtype=worker.torch.float16)
        assert str(device) == 'cuda'
        calls.append('predict')
        return (image < 50).astype(np.float32) * np.float32(.8)
    monkeypatch.setattr(shared.base, 'predict_foreground', inference)
    args = (tmp_path / 'training', images, tmp_path / 'output', metadata['checkpoint'])
    result = worker.run(*args)
    assert calls == ['limb', 'load', 'build', 'predict', 'predict', 'predict', 'predict']
    assert result['prediction_instances'] == 1 and result['recipe']['tta'] is True
    record = shared.read_envelope(args[2] / 'predictions/fixture.png.json', result['identity'])
    decoded = mu.decode(shared.validate_masks(record['masks'])[0])
    assert int(decoded.sum()) == 300 and decoded[1700:1715, 1800:1820].all()
    calls.clear()
    again = worker.run(*args)
    assert not calls and again['csv_sha256'] == result['csv_sha256']
