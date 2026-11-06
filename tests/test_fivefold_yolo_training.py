"""Fold labels and full YOLO checkpoint persistence; no GPU training claims."""
import csv
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'));sys.path.insert(0,str(ROOT/'tests'))
import fivefold_yolo_training as yolo
import grouped_cv_training as native
from test_grouped_cv_training import fixture


class Store:
    def __init__(self,*args):pass
    def load(self,name):return np.full((2048,2048),100,np.uint8)


class YoloStudyTests(unittest.TestCase):
    def test_only_fold_training_annotations_enter_dataset(self):
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()):
            cfg=fixture(Path(folder));runtime=native.load_runtime(cfg['base_runtime'])
            with patch.object(runtime,'ImageStore',Store):path=yolo.prepare_dataset(cfg,runtime)
            metadata=json.loads((path.parent/'data_manifest.json').read_text())
            with Path(cfg['train_split']).open() as stream:train={r['image_id'] for r in csv.DictReader(stream)}
            with Path(cfg['val_split']).open() as stream:valid={r['image_id'] for r in csv.DictReader(stream)}
            self.assertEqual(set(metadata['train_annotation_ids']),train)
            self.assertFalse(set(metadata['train_annotation_ids'])&valid)
            dataset=json.loads(path.read_text());self.assertEqual(dataset['names'],{'0':'filament'})
            self.assertEqual(dataset['val'],dataset['train'])
            self.assertEqual(len(list((path.parent/'labels/train').glob('*.txt'))),len(train))

    def test_callback_keeps_optimizer_live_weights_rng_and_selected_stage(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg=fixture(Path(folder));runtime=native.load_runtime(cfg['base_runtime']);root=Path(folder)
            model=torch.nn.Conv2d(3,2,1);optimizer=torch.optim.AdamW(model.parameters())
            model(torch.ones(1,3,4,4)).sum().backward();optimizer.step()
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,100)
            local=root/'local.pt';torch.save(dict(epoch=0,ema=deepcopy(model),optimizer=None,scaler={}),local)
            target=root/'yolo';target.mkdir();stage=root/'stage.pt'
            trainer=types.SimpleNamespace(last=local,epoch=0,model=model,optimizer=optimizer,scheduler=scheduler)
            yolo.checkpoint_callback(target,runtime,'study',1,stage)(trainer)
            full=torch.load(target/'last_full.pt',weights_only=False)
            compact=torch.load(stage,weights_only=False)
            self.assertEqual(full['cv_epoch'],1);self.assertTrue(full['optimizer']['state'])
            self.assertIn('cv_rng',full);self.assertEqual(full['cv_identity'],'study')
            self.assertIsNone(compact['optimizer'])
            for key,value in model.state_dict().items():self.assertTrue(torch.equal(value.cpu(),full['cv_live_state_dict'][key]))
            before=(target/'last_full.pt').read_bytes()
            with patch.object(runtime,'atomic_torch_save',side_effect=OSError('disk full')):
                with self.assertRaises(OSError):yolo.checkpoint_callback(target,runtime,'study',1,stage)(trainer)
            self.assertEqual((target/'last_full.pt').read_bytes(),before)

    def test_old_solar_checkpoint_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'generic'):
            yolo.run(dict(yolo_pretrained='old/best.pt'))
        with self.assertRaisesRegex(ValueError,'Old solar'):
            yolo.run(dict(initial_checkpoint='best84.pt'))


if __name__=='__main__':unittest.main()
