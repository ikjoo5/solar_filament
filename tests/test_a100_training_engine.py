"""Completed-epoch resume and bounded checkpoint mechanics on CPU."""
import ast
from contextlib import redirect_stdout
import csv
import importlib
import json
import io
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import cv2
from pycocotools import mask as mu
import torch
from torch.utils.data import Dataset, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import a100_training_engine as engine
from native_mixed_data import mixed_segmentation_loss


def io_helpers():
    tree = ast.parse((ROOT / 'tools/colab_train_native_unet_cell.py').read_text())
    training = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                    and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'TRAINING')
    functions = [node for node in ast.parse(training).body
                 if isinstance(node, ast.FunctionDef) and node.name in {'write_csv', 'package_results'}]
    namespace = dict(Path=Path, csv=csv, json=json, zipfile=zipfile)
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<reviewed-io>', 'exec'), namespace)
    return namespace


class StubStore:
    def __init__(self, *args, **kwargs): pass


class StubMixedDataset(Dataset):
    """Small deterministic crops exercise engine state, not model quality."""
    def __init__(self, data, ids, store, *, length, **kwargs):
        self.length, self.epoch = length, 0
        self.files, self.small_area, self.domain_info = ['train.jpeg'], 50., {'real': 1, 'synthetic': 1}
    def __len__(self): return self.length
    def sample_record(self, index): return {'domain': 'synthetic' if index % 2 else 'real'}
    def synthetic_probability(self): return .2
    def __getitem__(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([42, self.epoch, index]))
        x = torch.from_numpy(rng.normal(size=(3, 8, 8)).astype(np.float32))
        y = torch.zeros((2, 8, 8)); y[0, 2:6, 2:6] = 1; y[1, 3:5, 3:5] = 1
        return x, y, torch.tensor(float(index % 2 == 0))


class EngineTests(unittest.TestCase):
    def configuration(self, **updates):
        cfg = engine.normalize_training_config(dict(device='cpu', crop=64, batch=1,
            accumulate=1, steps_per_epoch=2, epochs=2, validate_every=1, workers=0))
        cfg.update(amp_dtype='None', source_sha256={'a': 'hash'})
        cfg.update(updates)
        return cfg

    def test_rng_roundtrip_is_weights_only_safe(self):
        random.seed(21); np.random.seed(22); torch.manual_seed(23)
        state = engine.capture_rng()
        expected = (random.random(), np.random.random(), torch.rand(5))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'state.pt'; engine.atomic_torch_save(path, state)
            loaded = torch.load(path, map_location='cpu', weights_only=True)
            engine.restore_rng(loaded)
        self.assertEqual(random.random(), expected[0]); self.assertEqual(np.random.random(), expected[1])
        self.assertTrue(torch.equal(torch.rand(5), expected[2]))

    def test_resume_rejects_changed_sources_crop_and_schedule(self):
        cfg = self.configuration()
        engine.check_resume_config(cfg, cfg.copy())
        for key, value in [('source_sha256', {'a': 'changed'}), ('crop', 96),
                           ('synthetic_ratio', .4), ('real_finish_epochs', 2), ('epochs', 3)]:
            with self.assertRaisesRegex(ValueError, key):
                engine.check_resume_config(cfg, dict(cfg, **{key: value}))

    def test_atomic_save_failure_preserves_previous_complete_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'last.pt'; engine.atomic_torch_save(path, {'epoch': 1})
            def failure(value, destination):
                Path(destination).write_bytes(b'partial'); raise OSError('disk full')
            with patch.object(engine.torch, 'save', failure):
                with self.assertRaisesRegex(OSError, 'disk full'):
                    engine.atomic_torch_save(path, {'epoch': 2})
            self.assertEqual(torch.load(path, weights_only=True), {'epoch': 1})
            self.assertEqual({p.name for p in Path(folder).iterdir()}, {'last.pt'})

    def test_source_pool_allows_annotator_promotion_but_no_validation_donor(self):
        data = dict(images=[dict(id='a', file_name='train.jpeg'), dict(id='b', file_name='train.jpeg'),
                            dict(id='c', file_name='val.jpeg')])
        info = dict(train_image_ids=['a'], val_image_ids=['b', 'c'])
        result = engine.check_synthetic_source_pool(info, data, {'a', 'b'}, ['val.jpeg'])
        self.assertEqual(result['donor_validation_overlap'], 0)
        self.assertFalse(result['fixed_split_changed'])
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            engine.check_synthetic_source_pool(dict(train_image_ids=['a', 'c'], val_image_ids=['b']),
                                               data, {'a', 'b'}, ['val.jpeg'])

    def test_synthetic_preflight_decodes_positive_ends_and_blank(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            native = np.full((2048, 2048), 130, np.uint8)
            for name in ('first.png', 'last.png', 'blank.png'):
                self.assertTrue(cv2.imwrite(str(folder/name), native))
            data = dict(images=[dict(id=i, file_name=name) for i, name in
                                enumerate(('first.png', 'last.png', 'blank.png'))],
                        annotations=[dict(image_id=0), dict(image_id=1)])
            audit = engine.preflight_synthetic_images(data, folder)
            self.assertEqual(audit['filenames_checked'], 3)
            self.assertEqual({s['file_name'] for s in audit['decoded_samples']},
                             {'first.png', 'last.png', 'blank.png'})
            self.assertEqual(audit['blank_images'], 1)
            self.assertTrue(cv2.imwrite(str(folder/'last.png'), np.zeros((32, 32), np.uint8)))
            with self.assertRaisesRegex(ValueError, '2048-square'):
                engine.preflight_synthetic_images(data, folder)

    def test_loader_spawn_and_epoch_rng_are_separate_from_training_rng(self):
        data = TensorDataset(torch.arange(6), torch.arange(6), torch.ones(6))
        data.epoch = 3
        cfg = self.configuration(batch=2, workers=1)
        loader = engine.make_epoch_loader(data, cfg, torch.device('cpu'))
        self.assertEqual(loader.multiprocessing_context.get_start_method(), 'spawn')
        self.assertFalse(loader.persistent_workers)
        torch.manual_seed(77); state = torch.get_rng_state()
        actual = torch.cat([row[0] for row in loader])
        self.assertTrue(torch.equal(actual, torch.arange(6)))
        self.assertTrue(torch.equal(torch.get_rng_state(), state))

    def test_interrupted_epoch_resume_matches_uninterrupted_optimizer_training(self):
        helpers = io_helpers(); calls = []
        def build_model(encoder, pretrained=True):
            calls.append(pretrained)
            return torch.nn.Sequential(torch.nn.Conv2d(3, 2, 1), torch.nn.Dropout2d(.2))
        whole = mu.encode(np.asfortranarray(np.ones((8, 8), np.uint8)))
        def prepare_split(data): return {'t'}, ['val.jpeg'], {'A-val': [whole]}
        def validate(model, store, names, device, amp_dtype, cfg):
            prediction = {'val': [whole]}
            grid = [dict(threshold=t, min_area=a, pq=.4, sq=.5, rq=.8, tp_pairs=1, fp=0, fn=0)
                    for t in (.35, .5, .65) for a in (100, 200, 300)]
            return grid[0], grid, {(r['threshold'], r['min_area']): prediction for r in grid}
        dependencies = dict(ImageStore=StubStore, MixedCropDataset=StubMixedDataset,
            mixed_segmentation_loss=mixed_segmentation_loss, build_model=build_model,
            prepare_split=prepare_split, validate_native=validate,
            write_csv=helpers['write_csv'], package_results=helpers['package_results'])
        class Interrupted(RuntimeError): pass
        with tempfile.TemporaryDirectory() as folder, patch.multiple(engine, create=True, **dependencies), redirect_stdout(io.StringIO()):
            folder = Path(folder)
            paths = {}
            payloads = dict(annotations=dict(images=[dict(id='t', file_name='train.jpeg'),
                                                       dict(id='v', file_name='val.jpeg')], annotations=[]),
                            synthetic_annotations=dict(images=[dict(id='s', file_name='synthetic.png')],
                                                       annotations=[dict(image_id='s')]),
                            synthetic_split=dict(train_image_ids=['t'], val_image_ids=['v']),
                            train_split={}, val_split={})
            for key, payload in payloads.items():
                path = folder/(key+'.json'); path.write_text(json.dumps(payload)); paths[key] = str(path)
            cfg = self.configuration(**paths, images=str(folder), synthetic_images=str(folder),
                                     cache_root=str(folder/'cache'), output_root=str(folder/'reference'))
            with self.assertRaisesRegex(ValueError, 'Synthetic image files missing'):
                engine.run_training(cfg)
            self.assertEqual(calls, [], 'Missing synthetic files must fail before any model is constructed')
            self.assertTrue(cv2.imwrite(str(folder/'synthetic.png'), np.full((2048, 2048), 130, np.uint8)))
            reference = engine.run_training(cfg)
            reference_state = torch.load(Path(reference['output'])/'last.pt', weights_only=True)
            original_package = helpers['package_results']
            def stop_after_validation(output, summary):
                result = original_package(output, summary)
                if summary['epochs_completed'] == 1: raise Interrupted('simulated session expiry')
                return result
            engine.package_results = stop_after_validation
            interrupted_cfg = dict(cfg, output_root=str(folder/'interrupted'))
            with self.assertRaises(Interrupted): engine.run_training(interrupted_cfg)
            output = next(path for path in (folder/'interrupted').iterdir() if path.is_dir())
            first_state = torch.load(output/'last.pt', weights_only=True)
            self.assertEqual(first_state['epoch'], 1)
            engine.package_results = original_package
            # Repair an interrupted export using the payload saved in best.pt.
            (output/'best_validation.csv').unlink()
            resumed = engine.run_training(dict(interrupted_cfg, resume_dir=str(output)))
            state = torch.load(output/'last.pt', weights_only=True)
            self.assertEqual(state['epoch'], 2); self.assertEqual(len(state['history']), 2)
            for key, expected in reference_state['model_state_dict'].items():
                self.assertTrue(torch.equal(state['model_state_dict'][key], expected), key)
            self.assertEqual(state['scheduler_state_dict'], reference_state['scheduler_state_dict'])
            self.assertTrue((output/'best_validation.csv').is_file())
            self.assertEqual(calls, [True, True, False], 'Resume must not download pretrained weights')
            with zipfile.ZipFile(resumed['archive']) as archive:
                self.assertEqual(json.loads(archive.read('summary.json'))['status'], 'complete')
                self.assertNotIn('best.pt', archive.namelist()); self.assertNotIn('last.pt', archive.namelist())
            self.assertEqual({p.name for p in output.glob('*.pt')}, {'best.pt', 'last.pt'})


if __name__ == '__main__':
    unittest.main()
