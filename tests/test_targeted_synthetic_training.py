"""Contract tests and actual tiny CPU continuation/resume verification."""
from contextlib import redirect_stdout,redirect_stderr
import csv
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import cv2
import numpy as np
from pycocotools import mask as mu
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import targeted_synthetic_training as training


def rectangle(x,y,w,h):
    pixels=np.zeros((2048,2048),np.uint8);pixels[y:y+h,x:x+w]=1
    return mu.encode(np.asfortranarray(pixels))

class TrainingContracts(unittest.TestCase):
    def test_unique_gt_recovery_and_precision_are_not_pair_counts(self):
        first=rectangle(0,0,20,20);second=rectangle(50,50,30,30)
        result=training.matched_counts({'a':[first,second]},{'a':'image'},
            {'image':[first,second]},{'image':[first]})
        self.assertEqual(result['recovered_GT'],1);self.assertEqual(result['lost_GT'],0)
        self.assertEqual(result['matched_GT'],2);self.assertEqual(result['precision'],1.)
        self.assertEqual(result['small_GT']['GT'],1)

    def test_raw_pool_preserves_order_highest_conf_and_skips_empty(self):
        a=rectangle(0,0,20,20)['counts'].decode();b=rectangle(100,100,20,20)['counts'].decode()
        empty=mu.encode(np.zeros((2048,2048),np.uint8,order='F'))['counts'].decode()
        rows=training.raw_items([{'instances':[{'rle':a,'confidence':.2},{'rle':b,'confidence':.4}]},
                                 {'instances':[{'rle':a,'confidence':.6},{'rle':empty,'confidence':.9}]}])
        self.assertEqual(rows,[{'rle':a,'confidence':.6},{'rle':b,'confidence':.4}])

    def test_compact_export_does_not_destroy_resume_state(self):
        payload=dict(optimizer={'x':torch.tensor([1.])},live_state={'w':torch.ones(2)},
            ema_full_state={'w':torch.ones(2)*2},scaler={'x':1},scheduler_state={'n':3},rng_state={'x':4},
            ema='inference-model',completed_epoch=30)
        compact=training.compact_checkpoint(payload)
        self.assertIsNone(compact['optimizer']);self.assertIsNone(compact['ema_full_state'])
        self.assertEqual(compact['ema'],'inference-model')
        self.assertIsNotNone(payload['optimizer']);self.assertIsNotNone(payload['live_state'])

    def test_atomic_checkpoint_failure_keeps_old_completed_file(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'last_full.pt';training.atomic_torch(p,{'epoch':1})
            before=p.read_bytes()
            with patch.object(training.torch,'save',side_effect=OSError('disk full')):
                with self.assertRaises(OSError):training.atomic_torch(p,{'epoch':2})
            self.assertEqual(p.read_bytes(),before)

    def test_real30_schedule_and_no_accumulation(self):
        cfg=dict(cache_root='/tmp/test')
        options=training.training_options(cfg,'data.yaml','control_v2')
        self.assertEqual(options['epochs'],30)
        self.assertEqual(options['imgsz'],2048)
        self.assertEqual((options['batch'],options['nbs'],options['workers']),(1,1,0))
        self.assertEqual(options['warmup_epochs'],0)
        self.assertFalse(options['val'])
        self.assertEqual(options['mosaic'],0)

    def test_missing_native_is_an_explicit_full_ensemble_error(self):
        with patch.object(training,'INPUT_SHA',{}),patch.object(training,'digest',return_value=training.ORIGINAL_SHA):
            with self.assertRaisesRegex(FileNotFoundError,'full_ensemble=True'):
                training.evaluate_checkpoints({'yolo':'fake','full_ensemble':True,'native':None},{})

    def test_export_omits_weights_images_and_splits_large_caches(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'evaluation/cache').mkdir(parents=True)
            (root/'training_config.json').write_text('{"fixed_final_epoch":30}')
            (root/'last_full.pt').write_bytes(b'not exported')
            (root/'source_image.png').write_bytes(b'not exported')
            for i in range(3):(root/f'evaluation/cache/{i}.json.gz').write_bytes(os.urandom(10*1024*1024))
            archives=training.export_results({'output_root':str(root)})
            self.assertEqual(len(archives),2)
            union=set();manifests=[]
            for path in archives:
                self.assertLess(path.stat().st_size,28*1024*1024)
                with zipfile.ZipFile(path) as archive:
                    self.assertIsNone(archive.testzip());names=set(archive.namelist());union|=names
                    self.assertIn('training_config.json',names)
                    manifests.append(json.loads(archive.read('export_manifest.json')))
            self.assertNotIn('last_full.pt',union);self.assertNotIn('source_image.png',union)
            self.assertEqual(manifests[0]['entries'],manifests[1]['entries'])
            self.assertEqual(len(manifests[0]['entries']),4)


def make_cpu_fixture(root):
    """A random one-class YOLO26n checkpoint and4 generated64px samples."""
    from ultralytics import YOLO
    from ultralytics.nn.tasks import yaml_model_load
    import yaml
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    model_cfg=yaml_model_load('yolo26n-seg.yaml');model_cfg['nc']=1
    arch=root/'yolo26n-seg.yaml';arch.write_text(yaml.safe_dump(model_cfg))
    torch.manual_seed(19);model=YOLO(str(arch),task='segment');checkpoint=root/'initial.pt';model.save(checkpoint)
    image_root=root/'data/images/train';label_root=root/'data/labels/train'
    image_root.mkdir(parents=True);label_root.mkdir(parents=True)
    for i in range(4):
        image=np.full((64,64,3),150,np.uint8);image[12+i:30+i,15:42]=70
        cv2.imwrite(str(image_root/f'{i}.png'),image)
        (label_root/f'{i}.txt').write_text('0 0.234375 0.1875 0.65625 0.1875 0.65625 0.46875 0.234375 0.46875\n')
    dataset=root/'data.yaml';dataset.write_text(yaml.safe_dump({'path':str(root/'data'),'train':'images/train','val':'images/train','names':{0:'filament'}}))
    cfg=dict(yolo=str(checkpoint),epochs=2,imgsz=64,device='cpu',amp=False,seed=42,lr=1e-4,cudnn_enabled=False,
        cache_root=str(root/'cache'),output_root=str(root/'continued'))
    return cfg,dataset


class ActualCPUResume(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('SOLAR_RUN_YOLO_CPU_SMOKE')=='1','Opt-in actual CPU YOLO smoke')
    def test_continuous_two_epochs_equal_pause_and_resume(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory(dir='/workspace/artifacts') as folder:
            root=Path(folder);cfg,dataset=make_cpu_fixture(root)
            initial=training.digest(cfg['yolo'])
            with patch.object(training,'ORIGINAL_SHA',initial):
                continuous=training.train_arm(cfg,'control_v2',dataset,'fixture-data',2)
                resumed_cfg=dict(cfg,output_root=str(root/'resumed'),cache_root=str(root/'resume_cache'))
                first=training.train_arm(resumed_cfg,'control_v2',dataset,'fixture-data',1)
                second=training.train_arm(resumed_cfg,'control_v2',dataset,'fixture-data',2)
                paths=[Path(x['output_root'])/'arms/control_v2/last_full.pt' for x in (cfg,resumed_cfg)]
                states=[torch.load(path,map_location='cpu',weights_only=False) for path in paths]
                for state in states:
                    self.assertEqual(state['completed_epoch'],2)
                    self.assertEqual(state['train_args']['epochs'],2)
                    self.assertTrue(state['optimizer']['state'])
                    self.assertEqual(len(state['history']),2)
                differences={kind:max(float((states[0][kind][k].float()-states[1][kind][k].float()).abs().max()) for k in states[0][kind]) for kind in ('live_state','ema_full_state')}
                self.assertEqual(differences,{'live_state':0.,'ema_full_state':0.})
                before=paths[1].read_bytes()
                again=training.train_arm(resumed_cfg,'control_v2',dataset,'fixture-data',2)
                self.assertEqual(paths[1].read_bytes(),before);self.assertEqual(again['checkpoint'],second['checkpoint'])
                with self.assertRaisesRegex(ValueError,'changed'):
                    training.train_arm(dict(resumed_cfg,lr=2e-4),'control_v2',dataset,'fixture-data',2)
                report=dict(status='passed_actual_cpu_training_and_resume',fixture='YOLO26n-seg random one-class,4synthetic64px images',
                    production='original YOLO26s continuation,2048px,30epochs eacharm; GPU not executed',
                    model_and_ema_max_absolute_difference=differences,full_optimizer_preserved=True,
                    completed_epoch=2,finished_rerun_idempotent=True,changed_settings_rejected=True,
                    source_sha256=training.digest(training.__file__),first_stage_checkpoint=Path(first['checkpoint']).name,
                    GPU_training_verified=False,competition_PQ_measured=False)
                training.atomic_json(ROOT/'docs/evidence/targeted_synthetic_training_cpu_smoke.json',report)


if __name__=='__main__':unittest.main()
