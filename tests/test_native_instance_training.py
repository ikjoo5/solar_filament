"""Actual CPU continuation plus shared-preprocessing/evaluation integration."""
from pathlib import Path
from contextlib import redirect_stdout
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import cv2
import numpy as np
from pycocotools import mask as mu
import torch
from torch import nn
from torch.utils.data import Dataset

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import native_instance_training as training
import native_instance_model as architecture
import native_instance_data as data


class TinyModel(nn.Module):
    def __init__(self,**kwargs):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(1,4,3,padding=1),nn.ReLU(),nn.Dropout2d(.1),nn.Conv2d(4,3,1))
    def forward(self,x):return self.net(x)


class CropFixture(Dataset):
    def __init__(self,groups,images,crop,steps_per_epoch,batch_size,seed,**kwargs):
        self.crop=crop;self.length=steps_per_epoch*batch_size;self.seed=seed;self.epoch=0
    def set_epoch(self,epoch):self.epoch=epoch
    def __len__(self):return self.length
    def __getitem__(self,index):
        generator=np.random.default_rng(self.seed+self.epoch*1000+index)
        pixels=generator.uniform(.2,.8,(1,self.crop,self.crop)).astype(np.float32)
        target=np.zeros((self.crop,self.crop),np.int64);target[6:18,7:20]=1;target[18:25,7:20]=2
        return torch.from_numpy(pixels),torch.from_numpy(target)


def fixture(root,head_mode='instance'):
    images=root/'images';images.mkdir(parents=True)
    for name,value in [('train.png',100),('val.png',150)]:
        pixels=np.full((32,32),value,np.uint8);pixels[8:20,10:24]=40
        assert cv2.imwrite(str(images/name),pixels)
    mask=np.zeros((32,32),np.uint8);mask[8:20,10:24]=1
    cohort=dict(train_groups=[dict(group_id='train',file_name='train.png',annotations=[])],
                validation_names=['val.png'],truth={'val_group':[mu.encode(np.asfortranarray(mask))]},
                mapping={'val_group':'val'},input_sha256={'fixture':'same'})
    cfg=dict(images=str(images),annotations='fixture',train_split='fixture',val_split='fixture',output_root=str(root/'continuous'),
             cache_root=str(root/'cache'),strict=False,device='cpu',pretrained=False,encoder='resnet18',head_mode=head_mode,
             epochs=2,crop=32,batch=2,steps_per_epoch=3,accumulate=2,amp=False,seed=17,lr=1e-3,
             preprocessing='raw',eval_min_areas=[1,10])
    return cfg,cohort


class NativeTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_actual_cpu_resume_matches_uninterrupted_and_partial_epoch_is_not_saved(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);cfg,cohort=fixture(root)
            with patch.object(data,'load_cohort',return_value=cohort),patch.object(data,'NativeInstanceDataset',CropFixture), \
                 patch.object(architecture,'NativeInstanceNet',TinyModel),patch.object(training,'source_hashes',return_value={'fixture':'stable'}),redirect_stdout(io.StringIO()):
                continuous=training.train(cfg,2)
                resumed=dict(cfg,output_root=str(root/'resumed'))
                first=training.train(resumed,1)
                last=Path(resumed['output_root'])/'last.pt';first_bytes=last.read_bytes()
                original_loss=architecture.loss_fn;calls=0
                def interrupted_loss(*args,**kwargs):
                    nonlocal calls
                    calls+=1
                    if calls==2:raise RuntimeError('fixture interrupted mid epoch')
                    return original_loss(*args,**kwargs)
                with patch.object(architecture,'loss_fn',side_effect=interrupted_loss):
                    with self.assertRaisesRegex(RuntimeError,'interrupted'):
                        training.train(resumed,2)
                self.assertEqual(last.read_bytes(),first_bytes)
                self.assertFalse((last.parent/'epoch002.pt').exists())
                second=training.train(resumed,2)
                a=torch.load(Path(cfg['output_root'])/'last.pt',map_location='cpu',weights_only=False)
                b=torch.load(last,map_location='cpu',weights_only=False)
                self.assertEqual(a['completed_epoch'],b['completed_epoch'],2)
                for key in a['model']:torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
                self.assertTrue(b['optimizer']['state']);self.assertEqual(len(b['history']),2)
                self.assertEqual([x['optimizer_attempts'] for x in b['history']],[2,2])
                self.assertEqual(b['scheduler'],a['scheduler'])
                before=last.read_bytes();training.train(resumed,2);self.assertEqual(last.read_bytes(),before)
                with self.assertRaisesRegex(ValueError,'changed'):
                    training.train(dict(resumed,lr=2e-3),2)
                evidence=dict(status='passed',fixture='actual CPU AdamW3minibatches/epoch, uneven accumulation3/2, dropout, instance loss',
                              completed_epoch=2,live_model_max_difference=0.,optimizer_and_scheduler_restored=True,
                              partial_epoch_did_not_replace_checkpoint=True,finished_stage_idempotent=True,
                              GPU_training_executed=False,competition_PQ_measured=False,
                              source_sha256=training.digest(training.__file__))
                training.atomic_json(ROOT/'docs/evidence/native_instance_training_cpu_smoke.json',evidence)

    def test_shared_real_preprocessing_and_normalization_in_tiled_inference(self):
        # This integration catches accidental /255 twice or train/infer mismatch.
        cfg=training.normalize_config(dict(output_root='/tmp/fixture',cache_root='/tmp/fixture_cache',
             crop=768,device='cpu',pretrained=False,amp=False,preprocessing='raw',overlap=.25))
        pixels=np.full((2048,2048),128,np.uint8);seen=[]
        class Echo(nn.Module):
            def forward(self,inputs):seen.append(inputs.detach().clone());return inputs.expand(-1,3,-1,-1)
        actual=training.predict_image(Echo(),pixels,cfg,torch.device('cpu'))
        expected=(float(data.preprocess_image(pixels,'raw')[0,0])-cfg['input_mean'])/cfg['input_std']
        self.assertTrue(seen);self.assertGreater(expected,.1)
        for inputs in seen:torch.testing.assert_close(inputs,torch.full_like(inputs,expected),rtol=1e-6,atol=1e-6)
        np.testing.assert_allclose(actual,torch.sigmoid(torch.tensor(expected)).item(),rtol=1e-6,atol=1e-6)
        self.assertEqual(actual.shape,(3,2048,2048))

    def test_semantic_evaluation_never_uses_untrained_aux_heads_and_cache_recovers(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);cfg,cohort=fixture(root,head_mode='semantic')
            with patch.object(data,'SIZE',32),patch.object(data,'load_cohort',return_value=cohort),patch.object(data,'NativeInstanceDataset',CropFixture), \
                 patch.object(architecture,'NativeInstanceNet',TinyModel),patch.object(training,'source_hashes',return_value={'fixture':'stable'}),redirect_stdout(io.StringIO()):
                trained=training.train(cfg,2)
                calls=[];original_decoder=architecture.decode_instances
                def decoder(*args,**kwargs):calls.append(kwargs['mode']);return original_decoder(*args,**kwargs)
                with patch.object(architecture,'decode_instances',side_effect=decoder):report=training.evaluate(cfg,trained['checkpoint'],2)
                self.assertEqual(calls,['connected','connected']);self.assertTrue(report['semantic_watershed_disabled'])
                self.assertEqual(set(report['results']),{'connected_area1','connected_area10'})
                hashes=report['prediction_csv_sha256']
                shutil.rmtree(cfg['cache_root'])
                with patch.object(architecture,'NativeInstanceNet',side_effect=AssertionError('should use persistent RLE records')):
                    repeated=training.evaluate(cfg,trained['checkpoint'],2)
                self.assertEqual(repeated['prediction_csv_sha256'],hashes)
                victim=Path(cfg['images'])/'val.png';assert cv2.imwrite(str(victim),np.full((32,32),200,np.uint8))
                with self.assertRaisesRegex(ValueError,'differ'):
                    training.evaluate(cfg,trained['checkpoint'],2)

    def test_area_specific_watershed_keeps_qualifying_parent(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);cfg,cohort=fixture(root);cfg['eval_min_areas']=[150,300]
            labels=np.zeros((32,32),np.int32);labels[6:26,6:16]=1;labels[6:26,16:26]=2
            targets=architecture.make_targets(labels)
            probability=np.stack([targets[key] for key in ('foreground','boundary','distance')])
            with patch.object(data,'load_cohort',return_value=cohort),patch.object(data,'NativeInstanceDataset',CropFixture), \
                 patch.object(architecture,'NativeInstanceNet',TinyModel),patch.object(training,'source_hashes',return_value={'fixture':'stable'}), \
                 patch.object(training,'predict_image',return_value=probability),redirect_stdout(io.StringIO()):
                stage=training.train(cfg,2);report=training.evaluate(cfg,stage['checkpoint'],2)
            record_path=next((Path(report['evaluation_root'])/'records').glob('*.json.gz'))
            import gzip
            record=json.loads(gzip.decompress(record_path.read_bytes()))
            self.assertEqual(len(record['predictions']['watershed_area150']),2)
            larger=record['predictions']['watershed_area300'];self.assertEqual(len(larger),1)
            self.assertEqual(int(mu.area(dict(size=larger[0]['size'],counts=larger[0]['counts'].encode()))),400)

    def test_atomic_failed_save_preserves_previous_complete_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'last.pt';training.atomic_torch(p,{'epoch':1});before=p.read_bytes()
            def bad_save(payload,path):Path(path).write_bytes(b'incomplete');raise OSError('fixture quota')
            with patch.object(training.torch,'save',side_effect=bad_save):
                with self.assertRaises(OSError):training.atomic_torch(p,{'epoch':2})
            self.assertEqual(p.read_bytes(),before);self.assertFalse(p.with_name('last.pt.pending').exists())

    def test_strict_iou_threshold_and_small_recall(self):
        gt=np.zeros((8,8),np.uint8);gt[2:4,2:4]=1
        doubled=gt.copy();doubled[2:4,4:6]=1
        encoded=lambda x:mu.encode(np.asfortranarray(x))
        truth={'g':[encoded(gt)]};mapping={'g':'image'}
        miss=training.score_predictions(truth,mapping,{'image':[encoded(doubled)]})
        self.assertEqual((miss['score']['pq'],miss['score']['fn'],miss['score']['fp']),(0.,1,1))
        good=training.score_predictions(truth,mapping,{'image':[encoded(gt)]})
        self.assertEqual(good['score']['pq'],1.);self.assertEqual(good['score']['small_GT']['recall'],1.)

    def test_result_archives_split_and_omit_weights_and_source_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'validation/records').mkdir(parents=True)
            (root/'result.json').write_text('{"status":"fixture"}')
            (root/'last.pt').write_bytes(b'weights');(root/'source.png').write_bytes(b'pixels')
            for index in range(3):(root/f'validation/records/{index}.json.gz').write_bytes(os.urandom(10*1024**2))
            archives=training.export_results({'output_root':str(root)});self.assertEqual(len(archives),2)
            union=set()
            for path in archives:
                self.assertLess(Path(path).stat().st_size,28*1024**2)
                with zipfile.ZipFile(path) as archive:self.assertIsNone(archive.testzip());union.update(archive.namelist())
            self.assertNotIn('last.pt',union);self.assertNotIn('source.png',union)


if __name__=='__main__':unittest.main()
