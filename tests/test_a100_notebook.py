"""Delivered notebook/worker integration, with CPU fixtures and native masks."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import cv2
import numpy as np
from pycocotools import mask as mu
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from build_a100_colab import CONFIG, worker_source


def config_values(ratio=.2):
    tree = ast.parse(CONFIG)
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'SYNTHETIC_RATIO':
                node.value = ast.Constant(ratio)
            nodes.append(node)
    namespace = {'Path': Path}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 '<notebook-config>', 'exec'), namespace)
    return namespace['CONFIG']


def load_worker(path):
    name = 'a100_flat_worker_fixture'
    path.write_text(worker_source())
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class A100NotebookTests(unittest.TestCase):
    def test_delivered_notebook_matches_readable_portable_worker(self):
        notebook = json.loads((ROOT/'notebooks/SolarFilament_A100_RealSynthetic.ipynb').read_text())
        self.assertEqual(notebook['nbformat'], 4)
        ids = [cell['id'] for cell in notebook['cells']]
        self.assertEqual(len(ids), len(set(ids)))
        extracted = None
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code':
                continue
            source = ''.join(cell['source'])
            compile(source, '<delivered-notebook-cell>', 'exec')
            for node in ast.parse(source).body:
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'WORKER_SOURCE':
                    extracted = ast.literal_eval(node.value)
        self.assertEqual(extracted.strip(), worker_source().strip())
        self.assertEqual((ROOT/'tools/solar_a100_train_standalone.py').read_text().strip(), extracted.strip())
        self.assertNotIn('/workspace', extracted)
        self.assertNotIn('base64', extracted)
        local_modules = {'native_mixed_data', 'a100_training_engine', 'train_filament_unet'}
        self.assertFalse(any(isinstance(node, ast.ImportFrom) and node.module in local_modules
                             for node in ast.walk(ast.parse(extracted))))
        compile((ROOT/'notebooks/SolarFilament_A100_CopyCell.py').read_text(), '<copy-cell>', 'exec')

    def test_known_paths_mixture_and_real_only_ablation(self):
        cfg = config_values()
        self.assertEqual(cfg['crop'], 1024)
        self.assertEqual(cfg['batch']*cfg['accumulate'], 4)
        self.assertEqual(cfg['batch']*cfg['steps_per_epoch'], 400)
        self.assertEqual(cfg['workers'], 2)
        self.assertEqual(cfg['synthetic_ratio'], .2)
        self.assertTrue(cfg['include_empty_synthetic'])
        self.assertEqual(len(cfg['expected_source_sha256']), 5)
        self.assertTrue(cfg['synthetic_images'].endswith('synthetic_copy_paste_v2/images'))
        self.assertTrue(cfg['cache_root'].startswith('/content/'))
        self.assertFalse(cfg['cache_root'].startswith('/content/drive/'))
        real = config_values(0.)
        self.assertEqual(len(real['expected_source_sha256']), 3)

    def test_flat_worker_native_preprocessing_and_masked_spine(self):
        with tempfile.TemporaryDirectory() as directory:
            module = load_worker(Path(directory)/'worker.py')
            from evaluate_yolo_baseline import preprocess_limb_only
            pixels = np.full((2048, 2048), 125, np.uint8)
            pixels[900:950, 1000:1030] = 80
            np.testing.assert_array_equal(module.limb(pixels), preprocess_limb_only(pixels))
            logits = torch.zeros((2, 2, 12, 12), requires_grad=True)
            target = torch.zeros_like(logits); target[:, 0, 3:8, 4:9] = 1
            target[0, 1, 4:7, 6] = 1
            loss = module.mixed_segmentation_loss(logits, target, torch.tensor([1., 0.]))
            loss.backward()
            self.assertGreater(float(logits.grad[1, 0].abs().sum()), 0)
            self.assertEqual(float(logits.grad[1, 1].abs().sum()), 0)

    def test_flat_worker_real_synthetic_training_and_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); module = load_worker(root/'worker.py')
            real = root/'real'; syn = root/'synthetic'; real.mkdir(); syn.mkdir()
            rows = []
            anns = []
            for index, name in enumerate(('train1', 'train2', 'val'), 1):
                pixels = np.full((2048, 2048), 125, np.uint8); pixels[90:110, 90:110] = 80
                cv2.imwrite(str(real/(name+'.png')), pixels)
                rows.append(dict(id=name, file_name=name+'.png', width=2048, height=2048))
                anns.append(dict(id=index, image_id=name, area=400, bbox=[90, 90, 20, 20],
                    segmentation=[[90,90,110,90,110,110,90,110]], spine=[92,100,108,100]))
            data = dict(images=rows, annotations=anns)
            (root/'ann.json').write_text(json.dumps(data))
            synthetic = dict(images=[dict(id=1, file_name='s.png', width=2048, height=2048)],
                             annotations=[dict(id=1, image_id=1, area=441, bbox=[90,90,21,21],
                                 segmentation=[[90,90,110,90,110,110,90,110]])])
            cv2.imwrite(str(syn/'s.png'), np.full((2048, 2048), 125, np.uint8))
            (root/'syn.json').write_text(json.dumps(synthetic))
            (root/'split.json').write_text(json.dumps(dict(train_image_ids=['train1','train2'], val_image_ids=['val'])))
            for label, names in [('train', ['train1','train2']), ('val', ['val'])]:
                (root/(label+'.csv')).write_text('image_id,file_name\n'+''.join(f'{name},{name}.png\n' for name in names))
            # Only fixture size and model change; use the delivered stores,
            # dataset, losses, official scorer, state writer and ZIP path.
            truth = mu.merge(mu.frPyObjects(anns[-1]['segmentation'], 2048, 2048))
            module.prepare_split = lambda loaded: ({'train1','train2'}, ['val.png'], {'A-val':[truth]})
            module.build_model = lambda encoder, pretrained: torch.nn.Conv2d(3, 2, 1)
            cfg = dict(images=str(real), annotations=str(root/'ann.json'), train_split=str(root/'train.csv'),
                val_split=str(root/'val.csv'), synthetic_images=str(syn), synthetic_annotations=str(root/'syn.json'),
                synthetic_split=str(root/'split.json'), output_root=str(root/'runs'), cache_root=str(root/'cache'),
                device='cpu', pretrained=False, crop=64, validation_tile=2048, batch=1, accumulate=1,
                steps_per_epoch=4, epochs=2, workers=0, infer_batch=1, synthetic_ratio=.5,
                warmup_epochs=0, real_finish_epochs=0, validate_every=1)
            result = module.run_training(cfg)
            output = Path(result['output'])
            history = json.loads((output/'history.json').read_text())
            self.assertEqual(len(history), 2)
            self.assertGreater(sum(row['sampled_domains'].get('synthetic', 0) for row in history), 0)
            self.assertEqual(len(history[-1]['validation_grid']), 9)
            self.assertTrue((output/'best.pt').is_file() and (output/'last.pt').is_file())
            checkpoint = torch.load(output/'last.pt', map_location='cpu', weights_only=True)
            self.assertEqual(checkpoint['epoch'], 2)
            with zipfile.ZipFile(result['archive']) as archive:
                self.assertEqual(json.loads(archive.read('summary.json'))['status'], 'complete')
                self.assertEqual(sum(name.startswith('best_t') and name.endswith('.csv') for name in archive.namelist()), 9)
                self.assertNotIn('best.pt', archive.namelist())
                self.assertNotIn('last.pt', archive.namelist())


if __name__ == '__main__':
    unittest.main()
