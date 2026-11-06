"""Native geometry, domain sampling, and missing-spine loss behavior."""
from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from native_mixed_data import MixedCropDataset, annotation_mask, mixed_segmentation_loss
from train_filament_unet import segmentation_loss


class Store:
    def __init__(self):
        self.image = np.full((2048, 2048), 140, np.uint8)
        self.image[1000:1021, 1000:1021] = 80
        self.loaded = []

    def load(self, filename):
        self.loaded.append(filename)
        return self.image


def fixture():
    images = [dict(id=key, file_name=name, width=2048, height=2048) for key, name in
              [('A1','a.jpeg'), ('A2','a.jpeg'), ('B1','b.jpeg'), ('V1','validation.jpeg')]]
    anns = [dict(id=i, image_id=row['id'], area=400, bbox=[1000,1000,20,20],
                 segmentation=[[1000,1000,1020,1000,1020,1020,1000,1020]],
                 spine=[1002,1010,1018,1010]) for i, row in enumerate(images)]
    real = dict(images=images, annotations=anns)
    # Missing area/bbox deliberately exercise generator-native derivation.
    synthetic = dict(images=[dict(id=1,file_name='synthetic_00000.png',width=2048,height=2048),
                             dict(id=2,file_name='blank.png',width=2048,height=2048)],
                     annotations=[dict(id=1,image_id=1,
                          segmentation=[[1000,1000,1020,1000,1020,1020,1000,1020]])])
    return real, synthetic


def dataset(**kwargs):
    real, synthetic = fixture()
    options = dict(crop=64, length=10000, synthetic_data=synthetic,
                   synthetic_store=Store(), augment=False)
    options.update(kwargs)
    return MixedCropDataset(real, {'A1','A2','B1'}, Store(), **options)


class NativeMixedDataTests(unittest.TestCase):
    def test_image_first_train_only_sampling_and_exact_domain_schedule(self):
        d = dataset()
        self.assertEqual(d.domain_info['real']['physical_images'], 2)
        self.assertEqual(d.domain_info['synthetic']['physical_images'], 1)
        self.assertEqual(d.domain_info['synthetic']['excluded_blank_groups'], 1)
        for epoch in (0,1,2,3,25,26,30):
            d.epoch = epoch
            self.assertEqual(d.synthetic_probability(), 0.)
            self.assertTrue(all(d.sample_record(i)['domain']=='real' for i in range(100)))
        d.epoch = 4
        self.assertEqual(d.synthetic_probability(), .2)
        counts = Counter()
        real_groups = set()
        for i in range(10000):
            row = d.sample_record(i); counts[row['domain']] += 1
            self.assertNotEqual(row['file_name'], 'validation.jpeg')
            if row['domain']=='real':
                real_groups.add(row['group_id']); counts[row['file_name']] += 1
        self.assertLess(abs(counts['synthetic']/10000-.2), .02)
        self.assertLess(abs(counts['a.jpeg']/counts['real']-.5), .025)
        self.assertEqual(real_groups, {'A1','A2','B1'})
        self.assertEqual(d.synthetic_probability(24), .2)
        self.assertEqual(d.synthetic_probability(25), 0.)

    def test_synthetic_rasterization_preserves_generator_cv_pixels(self):
        _, synthetic = fixture(); ann = synthetic['annotations'][0]
        generated = annotation_mask(ann, synthetic=True)
        competition = annotation_mask(ann, synthetic=False)
        self.assertEqual(int(generated.sum()), 441)
        self.assertEqual(int(competition.sum()), 400)
        self.assertTrue(generated[1020,1020])
        d=dataset()
        prepared=d.domains['synthetic']['annotations']['1'][0]
        self.assertEqual(prepared['area'], 441)
        self.assertEqual(prepared['bbox'], [1000,1000,21,21])

    def test_native_crop_alignment_and_invalid_synthetic_spine_flag(self):
        d=dataset(); d.epoch=4
        index=next(i for i in range(1000) if d.sample_record(i)['domain']=='synthetic'
                   and d.sample_record(i)['top']<=1000<d.sample_record(i)['top']+64
                   and d.sample_record(i)['left']<=1000<d.sample_record(i)['left']+64)
        row=d.sample_record(index); x,y,valid=d[index]
        self.assertEqual(tuple(x.shape),(3,64,64)); self.assertEqual(tuple(y.shape),(2,64,64))
        expected=np.zeros((64,64),np.float32)
        whole=annotation_mask(fixture()[1]['annotations'][0],True)
        expected[:]=whole[row['top']:row['top']+64,row['left']:row['left']+64]
        np.testing.assert_array_equal(y[0].numpy(), expected)
        self.assertGreater(float(y[0].sum()),0)
        self.assertEqual(float(valid),0.); self.assertEqual(float(y[1].sum()),0.)
        d.epoch=1
        index=next(i for i in range(1000) if d.sample_record(i)['top']<=1000<d.sample_record(i)['top']+64
                   and d.sample_record(i)['left']<=1000<d.sample_record(i)['left']+64)
        _,target,valid=d[index]
        self.assertEqual(float(valid),1.)
        self.assertTrue(torch.all(target[1]<=target[0]))

    def test_missing_spine_never_creates_negative_auxiliary_gradient(self):
        target=torch.zeros((2,2,12,12)); target[:,0,3:9,3:9]=1; target[0,1,5:7,4:8]=1
        logits=torch.zeros_like(target,requires_grad=True)
        loss=mixed_segmentation_loss(logits,target,torch.tensor([1.,0.]))
        self.assertTrue(torch.isfinite(loss)); loss.backward()
        self.assertGreater(float(logits.grad[0,1].abs().sum()),0)
        self.assertEqual(float(logits.grad[1,1].abs().sum()),0)
        self.assertGreater(float(logits.grad[1,0].abs().sum()),0)
        logits=torch.zeros_like(target,requires_grad=True)
        loss=mixed_segmentation_loss(logits,target,torch.zeros(2));loss.backward()
        self.assertEqual(float(logits.grad[:,1].abs().sum()),0)
        self.assertGreater(float(logits.grad[:,0].abs().sum()),0)
        with self.assertRaises(ValueError):
            mixed_segmentation_loss(logits,target,torch.tensor([1.,.5]))

    def test_all_valid_supervision_matches_reviewed_original_loss(self):
        generator=torch.Generator().manual_seed(17)
        logits=torch.randn((2,2,13,11),generator=generator)
        target=(torch.rand((2,2,13,11),generator=generator)>.8).float()
        target[:,1]*=target[:,0]
        original=segmentation_loss(logits,target,.1)
        masked=mixed_segmentation_loss(logits,target,torch.ones(2),.1)
        torch.testing.assert_close(masked,original,rtol=0,atol=0)

    def test_intentional_blank_synthetic_is_a_valid_foreground_negative(self):
        d=dataset(include_empty_synthetic=True,synthetic_ratio=1.,warmup_epochs=0,
                  real_finish_epochs=0)
        d.epoch=1
        index=next(i for i in range(100) if d.sample_record(i)['file_name']=='blank.png')
        _,target,valid=d[index]
        self.assertEqual(float(target.sum()),0.)
        self.assertEqual(float(valid),0.)
        self.assertEqual(d.domain_info['synthetic']['physical_images'],2)
        self.assertEqual(d.domain_info['synthetic']['excluded_blank_groups'],0)

    def test_local_target_cache_matches_fresh_and_is_annotation_keyed(self):
        with tempfile.TemporaryDirectory() as folder:
            d=dataset(cache_dir=folder)
            fg,spine,valid=d._target('real','A1')
            paths=list(Path(folder).glob('*.png')); self.assertEqual(len(paths),1)
            self.assertTrue(valid);self.assertTrue(np.all(spine<=fg))
            before=paths[0].stat().st_mtime_ns
            again=d._target('real','A1')
            self.assertEqual(paths[0].stat().st_mtime_ns,before)
            np.testing.assert_array_equal(again[0],fg);np.testing.assert_array_equal(again[1],spine)
            # A changed source annotation cannot accidentally reuse the old cache.
            d.domains['real']['annotations']['A1'][0]['spine']=[1000,1001,1018,1001]
            d._target('real','A1')
            self.assertEqual(len(list(Path(folder).glob('*.png'))),2)
            self.assertFalse(list(Path(folder).glob('*.tmp*')))

    def test_missing_real_spine_and_invalid_schedule_are_rejected(self):
        real,_=fixture();real['annotations'][0].pop('spine')
        with self.assertRaises(ValueError):MixedCropDataset(real,{'A1'},Store(),crop=64)
        with self.assertRaises(ValueError):dataset(warmup_epochs=20,real_finish_epochs=20)
        with self.assertRaises(ValueError):dataset(cache_dir='/content/drive/test_target_cache')


if __name__=='__main__':unittest.main()
