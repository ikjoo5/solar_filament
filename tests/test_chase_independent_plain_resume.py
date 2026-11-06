from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import torch
from pycocotools import mask as mu

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
import chase_independent_plain_worker as worker
import chase_best84_evaluation as evaluation
import chase_best84_training as training
import solar_a100_train_standalone as legacy
import submission_inference as inference


def test_full_mock_inference_resume_and_corrupt_frame(tmp_path, monkeypatch):
    root = tmp_path / 'chase_only_both_alignment_t4_60ep_fullpass_pngfix_v1'
    frozen = Path('/workspace/artifacts/chase_cache_review_20261105/replay_root')
    for folder in ('code', 'assets'):
        target = root / folder / worker.VERSION
        target.mkdir(parents=True)
        for path in (frozen / folder / worker.VERSION).iterdir():
            if path.is_file():
                (target / path.name).symlink_to(path)
    (root / 'configs').mkdir()
    (root / 'comparison/validation/epoch060_old_yolo_old_native').mkdir(parents=True)
    names = [f'{index:03}.jpeg' for index in range(91)]
    images = root / 'images'
    images.mkdir()
    for name in names:
        (images / name).write_bytes(b'mocked JPEG')
    image_hashes = {name: worker.sha(images / name) for name in names}
    data_sha = {'annotations': 'mock annotation', 'train_split': 'mock train', 'val_split': 'mock val'}
    protocol = dict(data_sha256=data_sha, source_images=image_hashes)
    original_id = evaluation.fingerprint(protocol)
    (root / 'comparison/validation/epoch060_old_yolo_old_native/identity.json').write_text(
        json.dumps(dict(identity=original_id, protocol=protocol)))

    class Tiny(torch.nn.Linear):
        def __init__(self):
            super().__init__(1, 1)

        def to(self, *args, **kwargs):
            return self

    model = Tiny()
    weight = root / 'original.pt'
    torch.save(dict(epoch=84, model_state_dict=model.state_dict()), weight)
    before = (weight.read_bytes(), weight.stat().st_mtime_ns)
    (root / 'configs/comparison.json').write_text(json.dumps(dict(images=str(images), initial_checkpoint=str(weight))))
    (root / 'comparison/epoch060_comparison.json').write_text(json.dumps(dict(results=[dict(
        role='old_yolo_old_native', identity=original_id, native_checkpoint_sha256=worker.sha(weight),
        model_sha256=evaluation.model_digest(model))])))
    binary = np.zeros((2048, 2048), np.uint8)
    binary[10:30, 10:30] = 1
    mask = mu.encode(np.asfortranarray(binary))
    record = {'components': {'0.5': [{'rle': mask}]}}
    groups = {name: dict(image=Path(name).stem, masks=[mask]) for name in names}
    monkeypatch.setattr(evaluation, 'load_cohort', lambda cfg: dict(names=names, source_sha256=data_sha, groups=groups))
    monkeypatch.setattr(evaluation, 'runtime_identity', lambda device: {'gpu': 'mock', 'torch': 'mock'})
    monkeypatch.setattr(training, 'configure_cuda', lambda: None)
    monkeypatch.setattr(legacy, 'build_model', lambda **kw: Tiny())
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda: False)
    import cv2
    gray = np.zeros((2048, 2048), np.uint8)
    corrected = np.ones_like(gray)
    monkeypatch.setattr(cv2, 'imdecode', lambda *a: gray)
    monkeypatch.setattr(inference, 'limb', lambda value: corrected)
    calls = []

    def predict(selected, pixels, device, **kwargs):
        assert not torch.is_grad_enabled() and not selected.training
        assert kwargs == dict(tile=1024, overlap=.25, batch=1, amp_dtype=torch.float16)
        calls.append('raw' if pixels is gray else 'limb')
        return np.zeros((2, 2, 2), np.float32)

    monkeypatch.setattr(inference, 'predict_heads', predict)
    monkeypatch.setattr(inference, 'prediction_record', lambda *a: dict(record))
    monkeypatch.setattr(inference, 'rle', lambda value: mask)
    worker.run(root, 'old_native')
    assert calls.count('raw') == calls.count('limb') == 91
    result_path = root / 'independent_models_plain_v1/old_native/role_results.json'
    results = json.loads(result_path.read_text())['results']
    assert {row['input_mode'] for row in results} == {'raw', 'limb'}
    assert all(row['pq'] == 1 and row['reused_images'] == 0 for row in results)
    calls.clear()
    worker.run(root, 'old_native')
    assert calls == []
    assert all(row['reused_images'] == 91 for row in json.loads(result_path.read_text())['results'])
    frame = Path(results[0]['cache_root']) / 'frames/000.json.gz'
    frame.write_bytes(b'interrupted/corrupt cache')
    worker.run(root, 'old_native')
    assert calls == ['raw']
    assert before == (weight.read_bytes(), weight.stat().st_mtime_ns)
