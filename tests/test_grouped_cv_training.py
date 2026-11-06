"""Real optimizer/RNG resume, strict data roles, and final-only OOF export."""
from contextlib import redirect_stdout
from datetime import date, timedelta
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import grouped_cv_training as cv
from prepare_grouped_fivefold import build_plan, write_outputs


class TinyDataset(Dataset):
    def __init__(self, *args, length, **kwargs):
        self.length, self.epoch = length, 0
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([42, self.epoch, index]))
        x = torch.from_numpy(rng.normal(size=(3, 8, 8)).astype(np.float32))
        target = torch.zeros((2, 8, 8))
        target[0, 2:6, 2:6] = 1
        target[1, 3:5, 3:5] = 1
        return x, target, torch.tensor(1.)


class TinyStore:
    def __init__(self, *args):
        pass
    def load(self, name):
        return name


def fixture(root):
    images, annotations = [], []
    original = root/'images'; original.mkdir()
    for offset in (1, 2, 26, 27, 28, 54, 55, 81, 108, 135, 162, 189):
        name = (date(2000, 1, 1)+timedelta(days=offset)).strftime('%Y%m%d')+'000000Bh.jpeg'
        (original/name).write_bytes(('fixture '+name).encode())
        for rater in (1, 2):
            identifier = f'{rater}-{name[:-5]}'
            images.append(dict(id=identifier, file_name=name))
            annotations.append(dict(image_id=identifier, segmentation=[[10,10,30,10,30,30,10,30]]))
    data = dict(images=images, annotations=annotations)
    ann = root/'annotations.json'; ann.write_text(json.dumps(data))
    plan = build_plan(data)
    plan['sources_sha256'] = dict(annotations=cv.digest(ann))
    split = root/'splits'; write_outputs(plan, split)
    runtime = root/'base_runtime.py'
    # The committed standalone differs from the pinned original by its leading newline.
    runtime.write_bytes(b'\n'+(ROOT/'tools/solar_a100_train_standalone.py').read_bytes())
    assert cv.digest(runtime) == cv.RUNTIME_SHA
    return dict(base_runtime=str(runtime), fold_plan=str(split/'plan.json'), fold=0,
        images=str(original), annotations=str(ann), train_split=str(split/'fold0/train_split.csv'),
        val_split=str(split/'fold0/val_split.csv'), cache_root=str(root/'cache'),
        output_root=str(root/'out'), epochs=3, run_until_epoch=3, steps_per_epoch=2,
        crop=64, batch=1, accumulate=2, workers=0, infer_batch=1, validate_every=2,
        device='cpu', preflight_backward=False)


class CVTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_resumed_optimizer_matches_continuous_and_only_final_is_oof(self):
        calls = []
        def build(encoder, pretrained=True):
            calls.append(pretrained)
            return nn.Sequential(nn.Conv2d(3, 2, 1), nn.Dropout2d(.2))
        def predict(model, image, *args):
            model.eval()
            p = np.zeros((2048, 2048), np.float32)
            p[10:30, 10:30] = .9
            return p
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            root = Path(folder); cfg = fixture(root)
            runtime = cv.load_runtime(cfg['base_runtime'])
            with patch.multiple(runtime, build_model=build, ImageStore=TinyStore,
                                MixedCropDataset=TinyDataset, predict_foreground=predict):
                whole = cv.run(dict(cfg, output_root=str(root/'whole')), _runtime=runtime)
                paused = cv.run(dict(cfg, run_until_epoch=1), _runtime=runtime)
                self.assertEqual(paused['status'], 'paused')
                self.assertFalse(paused['oof_score_available'])
                self.assertFalse((root/'out/fold0/oof_predictions.csv').exists())
                resumed = cv.run(cfg, _runtime=runtime)
                a = torch.load(root/'whole/fold0/last.pt', weights_only=True)
                b = torch.load(root/'out/fold0/last.pt', weights_only=True)
                for key in a['model_state_dict']:
                    self.assertTrue(torch.equal(a['model_state_dict'][key], b['model_state_dict'][key]))
                self.assertEqual(a['scheduler_state_dict'], b['scheduler_state_dict'])
                for key, state in a['optimizer_state_dict']['state'].items():
                    for field, value in state.items():
                        self.assertTrue(torch.equal(value, b['optimizer_state_dict']['state'][key][field]))
                self.assertEqual(calls, [True, True, False])
                self.assertTrue(resumed['oof_score_available'])
                self.assertEqual(resumed['selected_epoch'], 3)
                self.assertEqual((root/'whole/fold0/oof_predictions.csv').read_bytes(),
                                 (root/'out/fold0/oof_predictions.csv').read_bytes())
                self.assertFalse((root/'out/fold0/best.pt').exists())
                from evaluate_grouped_oof import evaluate_oof
                audited = evaluate_oof(cfg['annotations'], cfg['fold_plan'], root/'out')
                self.assertEqual(audited['status'], 'partial_folds')
                self.assertEqual(audited['completed_folds'], [0])
                self.assertAlmostEqual(audited['pooled_completed_folds']['pq'], resumed['metric']['pq'])
                with patch.object(runtime, 'predict_foreground', side_effect=AssertionError('Cached export should resume')):
                    cv.run(cfg, _runtime=runtime)
                for changes in (dict(epochs=4), dict(threshold=.65)):
                    with self.assertRaisesRegex(ValueError, 'different fold/settings/sources'):
                        cv.run(dict(cfg, **changes), _runtime=runtime)

    def test_missing_annotator_and_unsafe_warm_start_fail_before_model(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = fixture(Path(folder))
            with self.assertRaisesRegex(ValueError, 'fresh generic'):
                cv.run(dict(cfg, initial_checkpoint='old_best84.pt'))
            with self.assertRaisesRegex(ValueError, 'synthetic'):
                cv.run(dict(cfg, synthetic_ratio=.2))
            path = Path(cfg['val_split'])
            lines = path.read_text().splitlines()
            path.write_text('\n'.join(lines[:-1])+'\n')
            with self.assertRaisesRegex(ValueError, 'every annotator'):
                cv.prepare_inputs(cfg)

    def test_metric_is_strict_half_and_metadata_maps_all_raters(self):
        from pycocotools import mask as mu
        a = np.zeros((2048,2048), np.uint8); a[0:20,0:20] = 1
        b = a.copy(); b[:,10:20] = 0
        ar, br = [mu.encode(np.asfortranarray(value)) for value in (a,b)]
        score = cv.metric({'raterA':[ar], 'raterB':[ar]}, {'image':[br]},
                          {'raterA':'image','raterB':'image'})
        self.assertEqual((score['tp_pairs'],score['fp'],score['fn']), (0,2,2))


if __name__ == '__main__':
    unittest.main()
