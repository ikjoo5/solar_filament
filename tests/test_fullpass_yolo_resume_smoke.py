"""Opt-in real Ultralytics CPU check of fresh conversion and exact epoch resume.

RUN_ULTRALYTICS_RESUME_SMOKE=1 python -m pytest -q tests/test_fullpass_yolo_resume_smoke.py
"""
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import fullpass_yolo_training as worker
from test_swefil_mixed_training import FakeVerifiedStore


@pytest.mark.skipif(os.environ.get('RUN_ULTRALYTICS_RESUME_SMOKE') != '1',
                    reason='Opt-in actual Ultralytics CPU optimizer/resume check')
def test_fresh_80_class_conversion_and_resumed_training_match_continuous(tmp_path):
    from ultralytics import YOLO

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(37)
        model = YOLO('yolo26n-seg.yaml', task='segment')
        assert len(model.names) == 80
        initial = tmp_path/'generic_fixture.pt'
        model.save(initial)
        del model
        records = []
        for domain in ('real', 'swefil', 'synthetic'):
            for tile_id, (top, left) in enumerate(sorted(worker.TILE_ORIGINS)):
                image = tmp_path/'data/images/train'/f'{domain}_{tile_id}.png'
                label = tmp_path/'data/labels/train'/f'{domain}_{tile_id}.txt'
                image.parent.mkdir(parents=True, exist_ok=True)
                label.parent.mkdir(parents=True, exist_ok=True)
                pixels = np.full((64, 64, 3), 128, np.uint8)
                # Keep one genuinely blank tile in each domain.
                if tile_id:
                    cv2.rectangle(pixels, (15+tile_id, 15), (40, 35), (20, 20, 20), -1)
                    label.write_text('0 0.23 0.23 0.64 0.23 0.64 0.56 0.23 0.56\n')
                else:
                    label.write_text('')
                assert cv2.imwrite(str(image), pixels)
                records.append(dict(domain=domain, file_name=domain+'.png',
                    group_id=domain+'_observer', top=top, left=left,
                    image=str(image), label=str(label)))
        dataset_yaml = tmp_path/'data/dataset.yaml'
        dataset_yaml.write_text(json.dumps(dict(path=str(tmp_path/'data'),
            train='images/train', val='images/train', names={0:'filament'})))
        options = worker.training_options

        def tiny_options(cfg, yaml, local):
            value = options(cfg, yaml, local)
            value.update(imgsz=64, verbose=False)
            return value

        def validator(snapshot, _):
            assert len(snapshot.names) == 1
            row = dict(pq=float(next(snapshot.model.parameters()).detach().float().mean().sigmoid()),
                       threshold=.5, min_area=100, tp_pairs=1, fp=0, fn=0)
            return row, [row], {(.5,100): {'fixture':[]}}

        def run(name, cap):
            root = tmp_path/name
            store = FakeVerifiedStore(root/'server')
            cfg = dict(worker.DEFAULTS, epochs=2, device='cpu', yolo_amp=False,
                cache_root=str(root/'cache'), validate_every=1)
            result = worker.run_yolo(cfg, dataset_yaml, dict(records=records), root/'output',
                'fixture-v1', store, run_until=cap, validator=validator)
            payload = torch.load(store.restore(store.latest_state()['last']),
                                 map_location='cpu', weights_only=False)
            return result, payload

        with patch.object(worker, 'download_generic_weights', return_value=initial) as download, \
             patch.object(worker, 'training_options', side_effect=tiny_options):
            paused, _ = run('resumed', 1)
            assert paused['status'] == 'paused'
            _, resumed = run('resumed', 2)
            _, uninterrupted = run('uninterrupted', 2)
            assert download.call_count == 2  # Never reinitialize the resumed solar run.

        assert len(resumed['ema'].names) == 1
        for key in ('live_state', 'ema_full_state'):
            assert set(resumed[key]) == set(uninterrupted[key])
            for name in resumed[key]:
                torch.testing.assert_close(resumed[key][name], uninterrupted[key][name], rtol=0, atol=0)
        assert resumed['scheduler_state'] == uninterrupted['scheduler_state']
        assert resumed['updates'] == uninterrupted['updates'] == 6
        for row in resumed['history']:
            assert row['microbatches'] == 12
            assert row['optimizer_update_attempts'] == row['optimizer_updates'] == 3
            assert row['sampled_domains'] == dict(real=4, swefil=4, synthetic=4)
            assert row['physical_images'] == 3
    finally:
        torch.set_num_threads(previous_threads)
