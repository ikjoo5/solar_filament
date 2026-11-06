"""Standalone native-training mechanics, without GPU or full dataset training."""
import ast
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
import zipfile

import numpy as np
from pycocotools import mask as mu
import torch

ROOT=Path(__file__).resolve().parents[1]


def parts():
    tree=ast.parse((ROOT/'tools/colab_train_native_unet_cell.py').read_text())
    nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) or (
        isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name)
        and node.targets[0].id in {'PREFIX','CORE','TRAINING'})]
    namespace={'ast':ast}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<native-training-cell>','exec'),namespace)
    return namespace


def original_config():
    tree=ast.parse((ROOT/'tools/colab_restore_pixel_export_session_cell.py').read_text())
    return next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign)
                and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='SCRIPT')


def mechanics():
    module=ModuleType('native_training_mechanics')
    sys.modules[module.__name__]=module
    p=parts()
    exec(p['PREFIX']+'\n'+p['CORE']+'\n'+p['TRAINING'],module.__dict__)
    return module


class NativeTrainingCellTests(unittest.TestCase):
    def test_current_minimal_config_is_standalone_and_pinned(self):
        p=parts();worker=p['build_worker'](original_config())
        compile(worker,'<standalone-native-training>','exec')
        self.assertLess(len(worker.splitlines()),400)
        imports=[node for node in ast.walk(ast.parse(worker)) if isinstance(node,ast.ImportFrom)]
        self.assertFalse(any(node.module in {'train_filament_unet','filament_eval','convert_annotations','evaluate_yolo_baseline'} for node in imports))
        self.assertNotIn('/workspace',worker)
        self.assertNotIn('base64',worker)
        self.assertIn('5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1',worker)
        self.assertIn('native_unet_runs',worker)
        self.assertIn("Path('/content/filament_native_cache')",worker)

    def test_gpu_defaults_preserve_effective_batch_and_sampling_budget(self):
        module=mechanics()
        for memory,crop,batch,accumulate,steps in [(16,768,1,4,400),(40,1024,2,2,200)]:
            cfg=module.default_config(memory*1024**3)
            self.assertEqual((cfg['crop'],cfg['batch'],cfg['accumulate'],cfg['steps_per_epoch']),
                             (crop,batch,accumulate,steps))
            self.assertEqual(cfg['epochs'],30);self.assertEqual(cfg['validate_every'],2)
            self.assertEqual(cfg['batch']*cfg['accumulate'],4)
            self.assertEqual(cfg['batch']*cfg['steps_per_epoch'],400)

    def test_one_native_crop_optimizer_step_and_stitched_validation_grid(self):
        module=mechanics()
        image=np.full((2048,2048),140,np.uint8);image[80:100,80:100]=80
        class Store:
            def load(self,filename):return image
        ann=dict(image_id='A-image',area=400,bbox=[80,80,20,20],
                 segmentation=[[80,80,100,80,100,100,80,100]],spine=[82,90,98,90])
        data=dict(images=[dict(id='A-image',file_name='image.jpeg')],annotations=[ann])
        dataset=module.CropDataset(data,{'A-image'},Store(),64,1,42,1.,0.,False)
        x,y=dataset[0]
        self.assertEqual(tuple(x.shape),(3,64,64));self.assertEqual(tuple(y.shape),(2,64,64))
        self.assertGreater(float(y[0].sum()),0);self.assertTrue(torch.all(y[1]<=y[0]))
        model=torch.nn.Conv2d(3,2,1);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        before=model.weight.detach().clone();loss=module.segmentation_loss(model(x[None]),y[None],.1)
        self.assertTrue(torch.isfinite(loss));loss.backward();optimizer.step()
        self.assertFalse(torch.equal(before,model.weight.detach()))
        class Constant(torch.nn.Module):
            def forward(self,x):return torch.full((len(x),2,x.shape[-2],x.shape[-1]),2.,device=x.device)
        class SmallStore:
            def load(self,filename):return np.full((64,64),140,np.uint8)
        whole=mu.encode(np.asfortranarray(np.ones((64,64),np.uint8)))
        module.GT={'A-image':[whole]}
        best,grid,predictions=module.validate_native(Constant(),SmallStore(),['image.jpeg'],torch.device('cpu'),None,dict(crop=64,batch=1))
        self.assertEqual(len(grid),9);self.assertEqual(len(predictions),9)
        self.assertEqual(best['pq'],1.);self.assertEqual(best['fn'],0);self.assertEqual(best['fp'],0)

    def test_official_strict_half_iou_and_global_annotator_repetition(self):
        module=mechanics()
        whole=np.ones((10,10),np.uint8);half=np.zeros_like(whole);half[:,:5]=1
        encode=lambda mask:mu.encode(np.asfortranarray(mask))
        module.GT={'A-image':[encode(whole)],'B-image':[encode(whole)]}
        metric=module.score({'image':[encode(half)]})
        self.assertEqual(metric,dict(PQ=0.,SQ=0.,RQ=0.,TP=0,FP=2,FN=2))
        metric=module.score({'image':[encode(whole)]})
        self.assertEqual(metric['PQ'],1.);self.assertEqual(metric['TP'],2)

    def test_running_zip_replaced_with_complete_zip_and_no_checkpoints(self):
        module=mechanics()
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'run';output.mkdir()
            (output/'config.json').write_text('{}')
            (output/'history.json').write_text('[]')
            (output/'run_source.py').write_text('# plain reviewed code\n')
            (output/'best.pt').write_bytes(b'large checkpoint placeholder')
            (output/'last.pt').write_bytes(b'optimizer checkpoint placeholder')
            whole=mu.encode(np.asfortranarray(np.ones((12,12),np.uint8)))
            for threshold in (.35,.5,.65):
                for area in (100,200,300):
                    module.write_csv(output/f'best_t{threshold:g}_a{area}.csv',{'image':[whole]})
            archive=module.package_results(output,dict(status='running',epochs_completed=2))
            with zipfile.ZipFile(archive) as z:
                self.assertEqual(json.loads(z.read('summary.json'))['status'],'running')
                self.assertEqual(sum(name.endswith('.csv') for name in z.namelist()),9)
                self.assertNotIn('best.pt',z.namelist());self.assertNotIn('last.pt',z.namelist())
            repeated=module.package_results(output,dict(status='complete',epochs_completed=30))
            self.assertEqual(archive,repeated)
            with zipfile.ZipFile(archive) as z:self.assertEqual(json.loads(z.read('summary.json'))['status'],'complete')


if __name__=='__main__':unittest.main()
