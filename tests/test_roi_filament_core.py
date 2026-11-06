"""Single-target ROI geometry, safe negatives and a real CPU U-Net step."""
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np
from pycocotools import mask as mu
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from roi_filament_core import (RoiFilamentNet,RoiTrainingDataset,crop_roi,
                               restore_mask,roi_loss,prepare_image)


def annotation(mask, identifier):
    rle=mu.encode(np.asfortranarray(mask));rle['counts']=rle['counts'].decode()
    return dict(id=identifier,segmentation=rle)


def fixture(root):
    pixels=np.full((2048,2048),150,np.uint8)
    first=np.zeros_like(pixels);first[900:940,900:1000]=1
    second=np.zeros_like(pixels);second[950:980,960:1040]=1
    pixels[first.astype(bool)]=40;pixels[second.astype(bool)]=60
    image=root/'train.jpeg';cv2.imwrite(str(image),pixels)
    groups=[dict(group_id='r1',file_name=image.name,role='train',annotations=[annotation(first,'a'),annotation(second,'b')]),
            dict(group_id='r2',file_name=image.name,role='train',annotations=[annotation(second,'c')])]
    return pixels,first,second,groups


class RoiCoreTests(unittest.TestCase):
    def test_geometry_round_trip_and_non_square_box(self):
        image=np.full((2048,2048),170,np.uint8);target=np.zeros_like(image)
        target[400:440,600:760]=1
        inputs,mask,transform=crop_roi(image,[600,400,760,440],target,context=1.5,size=384)
        self.assertEqual(inputs.shape,(2,384,384));self.assertEqual(mask.shape,(1,384,384))
        self.assertEqual(set(np.unique(inputs[1])),{0.,1.})
        restored=restore_mask(mask,transform)
        gt=mu.encode(np.asfortranarray(target))
        self.assertGreater(float(mu.iou([gt],[restored],[0])[0,0]),.99)
        self.assertTrue(np.array_equal(inputs[1],mask[0]))

    def test_edge_padding_prompt_position_and_beyond_original_box(self):
        image=np.ones((2048,2048),np.uint8)*255
        target=np.zeros_like(image);target[0:20,0:40]=1
        inputs,mask,transform=crop_roi(image,[5,5,30,18],target,context=3.,size=192)
        self.assertLess(transform['left'],0);self.assertLess(transform['top'],0)
        self.assertEqual(float(inputs[0,0,0]),0.)
        restored=mu.decode(restore_mask(mask,transform))
        self.assertGreater(int(restored[:5,:5].sum()),0)
        self.assertGreater(float(mu.iou([mu.encode(np.asfortranarray(target))],
                                      [mu.encode(np.asfortranarray(restored))],[0])[0,0]),.9)
        self.assertIsNone(restore_mask(np.zeros((192,192),np.float32),transform))

    def test_only_requested_object_in_target(self):
        with tempfile.TemporaryDirectory() as directory:
            pixels,first,second,_=fixture(Path(directory))
            _,target,transform=crop_roi(pixels,[900,900,1000,940],first,context=3.,size=384)
            restored=mu.decode(restore_mask(target,transform))
            self.assertEqual(int((restored.astype(bool)&second.astype(bool)).sum()),0)

    def test_real_resnet18_step_and_prompt_normalization(self):
        torch.set_num_threads(2);torch.manual_seed(42)
        model=RoiFilamentNet('resnet18',encoder_weights=None)
        inputs=torch.rand(2,2,64,64);inputs[:,1]=(inputs[:,1]>.5).float()
        target=torch.zeros(2,1,64,64);target[0,:,15:40,20:45]=1
        captured=[]
        hook=model.network.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach()))
        output=model(inputs);hook.remove()
        self.assertTrue(torch.equal(captured[0][:,1],inputs[:,1]))
        self.assertTrue(torch.allclose(captured[0][:,:1],(inputs[:,:1]-.449)/.226))
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        before=model.network.segmentation_head[0].weight.detach().clone()
        loss,metrics=roi_loss(output,target);self.assertTrue(torch.isfinite(loss))
        loss.backward();optimizer.step()
        self.assertFalse(torch.equal(before,model.network.segmentation_head[0].weight))
        self.assertEqual(int(metrics['positive_examples']),1)

    def test_empty_target_loss_is_finite_and_background_gradient(self):
        logits=torch.zeros(2,1,32,32,requires_grad=True)
        loss,_=roi_loss(logits,torch.zeros_like(logits));loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue((logits.grad>0).all())

    def test_dataset_reproducible_and_epoch_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);_,_,_,groups=fixture(root)
            options=dict(crop_size=96,context=1.5,steps_per_epoch=4,batch_size=2,seed=42,preprocessing='raw')
            a=RoiTrainingDataset(groups,root,**options);b=RoiTrainingDataset(groups,root,**options)
            a.set_epoch(3);b.set_epoch(3)
            self.assertEqual(len(a),8)
            for i in range(8):
                self.assertEqual(a.sample_record(i),b.sample_record(i))
                for left,right in zip(a[i],b[i]):self.assertTrue(torch.equal(left,right))
            b.set_epoch(4);self.assertNotEqual(a.sample_record(0),b.sample_record(0))

    def test_safe_negatives_exclude_all_rater_foreground(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);_,first,second,groups=fixture(root)
            # The first rater does not label the second object's foreground.
            # Its detector proposal must still be excluded as a negative.
            groups[0]['annotations']=groups[0]['annotations'][:1]
            proposals={'train':[dict(box=[960,950,1040,980],score=.9),dict(box=[100,100,150,130],score=.8)]}
            data=RoiTrainingDataset(groups,root,proposals,crop_size=64,steps_per_epoch=100,batch_size=1,preprocessing='raw')
            negatives=0
            for i in range(len(data)):
                record=data.sample_record(i)
                if not record['positive']:
                    negatives+=1
                    x1,y1=np.floor(record['box'][:2]).astype(int);x2,y2=np.ceil(record['box'][2:]).astype(int)
                    self.assertEqual(int((first|second)[y1:y2,x1:x2].sum()),0)
                    self.assertEqual(float(data[i][1].sum()),0.)
            self.assertGreater(negatives,5)
            with self.assertRaisesRegex(ValueError,'Validation'):
                RoiTrainingDataset([dict(groups[0],role='validation')],root)
            with self.assertRaisesRegex(ValueError,'not in train'):
                RoiTrainingDataset(groups,root,{'val':[dict(box=[0,0,10,10])]})

    def test_preprocessing_range_and_geometry_validation(self):
        raw=np.ones((2048,2048),np.uint8)*125
        self.assertIs(prepare_image(raw,'raw'),raw)
        self.assertEqual(prepare_image(raw,'limb').dtype,np.uint8)
        with self.assertRaises(ValueError):crop_roi(raw,[10,10,0,20])
        with self.assertRaises(ValueError):crop_roi(raw,[-100,-100,-50,-50])
        with self.assertRaises(ValueError):crop_roi(raw,[0,0,10,10],context=.5)


if __name__=='__main__':unittest.main()
