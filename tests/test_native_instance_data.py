"""Native real-only crops, rater separation and full-reset reproducibility."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from pycocotools import mask as mu
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from native_instance_data import (NativeInstanceDataset, annotation_rle,
                                  digest, instance_labels, limb, load_cohort,
                                  preprocess_image)


def encode(mask):
    encoded = mu.encode(np.asfortranarray(mask.astype(np.uint8)))
    return dict(size=encoded['size'], counts=encoded['counts'].decode('ascii'))


def fixture(root):
    images = root/'images';images.mkdir()
    pixels = np.full((2048, 2048), 160, np.uint8)
    pixels[990:1030, 980:1060] = 60
    mask = np.zeros_like(pixels);mask[990:1030, 980:1060]=1
    other = np.zeros_like(mask);other[1000:1020, 1000:1040]=1
    metadata = dict(images=[dict(id=sid, file_name=name, width=2048, height=2048)
                           for sid, name in [('T1','train.jpeg'), ('T2','train.jpeg'),
                                             ('B1','blank.jpeg'), ('V1','val.jpeg')]],
                    annotations=[dict(id=1, image_id='T1', segmentation=encode(mask)),
                                 dict(id=2, image_id='T1', segmentation=encode(other)),
                                 dict(id=3, image_id='T2', segmentation=[[980,990,1060,990,1060,1030,980,1030]]),
                                 dict(id=4, image_id='V1', segmentation=encode(mask))])
    for name in ('train.jpeg','blank.jpeg','val.jpeg'):
        assert cv2.imwrite(str(images/name), pixels)
    config = dict(images=str(images), strict=False)
    annotation_path=root/'annotations.json';annotation_path.write_text(json.dumps(metadata));config['annotations']=str(annotation_path)
    for role, ids in [('train',['T1','T2','B1']), ('val',['V1'])]:
        path=root/(role+'.csv')
        by_id={row['id']:row['file_name'] for row in metadata['images']}
        with path.open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['image_id','file_name'])
            writer.writerows((sid,by_id[sid]) for sid in ids)
        config[role+'_split']=str(path)
    return config, metadata, pixels, mask, other


class NativeInstanceDataTests(unittest.TestCase):
    def test_raters_stay_separate_and_only_train_groups_returned(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg,_,_,_,_=fixture(Path(directory));cohort=load_cohort(cfg)
            self.assertEqual(cohort['counts'],dict(train_physical_images=2,training_annotation_groups=3,
                             training_instances=3,validation_physical_images=1,
                             validation_annotation_groups=1,validation_instances=1))
            self.assertEqual({g['group_id'] for g in cohort['train_groups']},{'T1','T2','B1'})
            self.assertEqual(cohort['validation_names'],['val.jpeg'])
            self.assertEqual(cohort['mapping'],{'V1':'val'})
            self.assertIsInstance(cohort['truth']['V1'][0]['counts'],bytes)
            self.assertTrue(all(g['role']=='train' and g['file_name']!='val.jpeg' for g in cohort['train_groups']))
            self.assertEqual(set(cohort['input_sha256']),{'annotations','train_split','val_split'})

    def test_split_rejects_rater_sibling_leak_and_production_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg,_,_,_,_=fixture(Path(directory))
            with self.assertRaisesRegex(ValueError,'reviewed'):
                load_cohort(dict(cfg,strict=True))
            Path(cfg['train_split']).write_text('image_id,file_name\nT1,train.jpeg\nB1,blank.jpeg\n')
            Path(cfg['val_split']).write_text('image_id,file_name\nT2,train.jpeg\nV1,val.jpeg\n')
            with self.assertRaisesRegex(ValueError,'sibling'):
                load_cohort(cfg)

    def test_coco_raster_and_smaller_instance_overlap_ownership(self):
        with tempfile.TemporaryDirectory() as directory:
            _,metadata,_,mask,other=fixture(Path(directory))
            rle=annotation_rle(metadata['annotations'][0]);self.assertTrue(np.array_equal(mu.decode(rle),mask))
            labels=instance_labels(metadata['annotations'][:2])
            self.assertEqual(labels.dtype,np.int32)
            self.assertTrue(np.all(labels[other.astype(bool)]==2))
            self.assertTrue(np.all(labels[mask.astype(bool)&~other.astype(bool)]==1))
            self.assertEqual(set(np.unique(labels)),{0,1,2})
            uncompressed=dict(size=[2048,2048],counts=[2048*2048-1,1])
            self.assertEqual(int(mu.area(annotation_rle({'segmentation':uncompressed}))),1)

    def test_epoch_reset_reproduces_augmented_image_and_instance_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg,_,_,_,_=fixture(Path(directory));groups=load_cohort(cfg)['train_groups']
            options=dict(crop=128,steps_per_epoch=3,batch_size=2,seed=83,preprocessing='raw')
            first=NativeInstanceDataset(groups,cfg['images'],**options);first.set_epoch(7)
            second=NativeInstanceDataset(groups,cfg['images'],**options);second.set_epoch(7)
            self.assertEqual(len(first),6)
            for index in range(len(first)):
                before=first[index];after=second[index]
                self.assertTrue(torch.equal(before[0],after[0]))
                self.assertTrue(torch.equal(before[1],after[1]))
                self.assertEqual(before[0].shape,(1,128,128));self.assertEqual(before[1].shape,(128,128))
                self.assertEqual(before[0].dtype,torch.float32);self.assertEqual(before[1].dtype,torch.int64)
                self.assertGreaterEqual(float(before[0].min()),0);self.assertLessEqual(float(before[0].max()),1)
            first.set_epoch(8)
            self.assertNotEqual(first.sample_record(0),second.sample_record(0))

    def test_positive_crops_contain_the_centered_id_and_preserve_partial_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg,_,_,_,_=fixture(Path(directory));groups=load_cohort(cfg)['train_groups']
            train=[group for group in groups if group['group_id']=='T1']
            data=NativeInstanceDataset(train,cfg['images'],crop=16,steps_per_epoch=8,batch_size=1,
                                       seed=14,positive_fraction=1.,preprocessing='raw',augment=False)
            for i in range(len(data)):
                record=data.sample_record(i);_,labels=data[i]
                self.assertTrue(record['foreground_centered'])
                self.assertTrue((labels==record['centered_instance_id']).any())
                native=instance_labels(train[0]['annotations'])
                expected=native[record['y']:record['y']+16,record['x']:record['x']+16]
                self.assertTrue(np.array_equal(labels.numpy(),expected))

    def test_random_negative_crops_and_validation_group_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg,_,_,_,_=fixture(Path(directory));groups=load_cohort(cfg)['train_groups']
            blank=[g for g in groups if g['group_id']=='B1']
            data=NativeInstanceDataset(blank,cfg['images'],crop=64,steps_per_epoch=1,batch_size=1,
                                       preprocessing='raw',augment=False)
            self.assertFalse(data.sample_record(0)['foreground_centered'])
            self.assertEqual(int(data[0][1].sum()),0)
            with self.assertRaisesRegex(ValueError,'Validation'):
                NativeInstanceDataset([dict(blank[0],role='validation')],cfg['images'])

    def test_augmented_foreground_pixels_stay_aligned_with_instance_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg,_,_,_,_=fixture(Path(directory));groups=load_cohort(cfg)['train_groups']
            groups=[group for group in groups if group['group_id']=='T1']
            data=NativeInstanceDataset(groups,cfg['images'],crop=128,steps_per_epoch=12,batch_size=1,
                                       preprocessing='raw',positive_fraction=1.,seed=19)
            for index in range(len(data)):
                image,labels=data[index];foreground=labels>0
                self.assertTrue(foreground.any());self.assertTrue((~foreground).any())
                self.assertLess(float(image[0][foreground].mean()),float(image[0][~foreground].mean())-.2)

    def test_limb_matches_reviewed_algorithm_and_shared_preprocessing(self):
        from submission_inference import limb as reviewed_limb
        rng=np.random.default_rng(31)
        pixels=rng.integers(0,256,(2048,2048),dtype=np.uint8)
        corrected=limb(pixels)
        self.assertTrue(np.array_equal(corrected,reviewed_limb(pixels)))
        shared=preprocess_image(pixels,'limb')
        self.assertEqual(shared.dtype,np.float32)
        self.assertTrue(np.array_equal(shared,corrected.astype(np.float32)/255.))
        self.assertTrue(np.array_equal(preprocess_image(pixels,'raw'),pixels.astype(np.float32)/255.))

    def test_bounded_cache_rebuilds_from_original_after_cache_deletion(self):
        import shutil
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg,_,_,_,_=fixture(root);groups=load_cohort(cfg)['train_groups']
            options=dict(crop=64,steps_per_epoch=1,batch_size=1,preprocessing='raw',cache_dir=root/'cache')
            a=NativeInstanceDataset(groups,cfg['images'],**options);before=a[0]
            self.assertEqual(a._load_image.cache_info().maxsize,8);self.assertEqual(a._load_labels.cache_info().maxsize,2)
            shutil.rmtree(root/'cache')
            b=NativeInstanceDataset(groups,cfg['images'],**options);after=b[0]
            self.assertTrue(torch.equal(before[0],after[0]));self.assertTrue(torch.equal(before[1],after[1]))

    def test_cached_jpeg_tampering_is_repaired_using_source_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg,_,_,_,_=fixture(root);groups=load_cohort(cfg)['train_groups']
            options=dict(crop=64,steps_per_epoch=1,batch_size=1,preprocessing='raw',
                         cache_dir=root/'cache',source_sha256={name:digest(Path(cfg['images'])/name)
                             for name in ('train.jpeg','blank.jpeg')})
            a=NativeInstanceDataset(groups,cfg['images'],**options);a[0]
            name=a.sample_record(0)['file_name'];cached=root/'cache'/name
            cached.write_bytes(b'corrupted cache')
            b=NativeInstanceDataset(groups,cfg['images'],**options);b[0]
            self.assertEqual(cached.read_bytes(),(Path(cfg['images'])/name).read_bytes())

    def test_processed_cache_reuses_exact_uint8_and_recovers_after_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg,_,_,_,_=fixture(root);groups=load_cohort(cfg)['train_groups']
            options=dict(crop=64,steps_per_epoch=1,batch_size=1,preprocessing='limb',
                         cache_dir=root/'cache',source_sha256={name:digest(Path(cfg['images'])/name)
                             for name in ('train.jpeg','blank.jpeg')})
            a=NativeInstanceDataset(groups,cfg['images'],**options);before=a[0]
            name=a.sample_record(0)['file_name'];processed=root/'cache/processed/limb'/(name+'.npy')
            self.assertEqual(np.load(processed,allow_pickle=False).dtype,np.uint8)
            with patch('native_instance_data.limb',side_effect=AssertionError('Cache must avoid repeated limb correction')):
                b=NativeInstanceDataset(groups,cfg['images'],**options);cached=b[0]
            self.assertTrue(torch.equal(before[0],cached[0]));self.assertTrue(torch.equal(before[1],cached[1]))
            processed.write_bytes(b'invalid local processed image')
            c=NativeInstanceDataset(groups,cfg['images'],**options);rebuilt=c[0]
            self.assertTrue(torch.equal(before[0],rebuilt[0]));self.assertTrue(torch.equal(before[1],rebuilt[1]))

    def test_changed_source_cannot_be_copied_under_old_pixel_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg,_,_,_,_=fixture(root);groups=load_cohort(cfg)['train_groups']
            sources={name:digest(Path(cfg['images'])/name) for name in ('train.jpeg','blank.jpeg')}
            data=NativeInstanceDataset(groups,cfg['images'],crop=64,steps_per_epoch=1,batch_size=1,
                                       cache_dir=root/'cache',source_sha256=sources)
            name=data.sample_record(0)['file_name']
            cv2.imwrite(str(Path(cfg['images'])/name),np.full((2048,2048),45,np.uint8))
            with self.assertRaisesRegex(ValueError,'changed after pixel'):
                data[0]


if __name__=='__main__':
    unittest.main()
