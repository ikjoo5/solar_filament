"""Training helpers, native output resolution, and crop/annotation alignment."""
import gc
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np
import torch


TOOLS = Path(__file__).resolve().parents[1]/'tools'
if str(TOOLS) not in sys.path:
    sys.path.insert(0,str(TOOLS))
import train_filament_unet as training


def annotation(group,index,x,y):
    return {'id':f'{group}-ann{index}','image_id':group,'area':100,
            'bbox':[x,y,10,10],'segmentation':[[x,y,x+10,y,x+10,y+10,x,y+10]],
            'spine':[x+2,y+5,x+8,y+5], 'category_id':1,'iscrowd':0}


class TrainingHelperTests(unittest.TestCase):
    def test_loss_finite_with_and_without_spine_and_backward(self):
        for positive in [False,True]:
            logits = torch.zeros(2,2,16,16,requires_grad=True)
            target = torch.zeros_like(logits)
            target[:,0,4:8,4:8] = 1
            if positive:
                target[:,1,6,4:8] = 1
            loss = training.segmentation_loss(logits,target)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(torch.isfinite(logits.grad).all())
            correct = torch.where(target>0,torch.tensor(10.),torch.tensor(-10.))
            self.assertLess(training.segmentation_loss(correct,target).item(),loss.item())

    def test_loss_rejects_misaligned_outputs(self):
        with self.assertRaises(ValueError):
            training.segmentation_loss(torch.zeros(1,2,8,8),torch.zeros(1,2,4,4))

    def test_d4_transforms_preserve_pixel_alignment(self):
        source = np.arange(64).reshape(1,8,8)
        target = np.concatenate([source,source+100],axis=0)
        for k in range(4):
            for flip in [False,True]:
                image,truth = training.apply_d4(source,target,k,flip)
                np.testing.assert_array_equal(image[0],truth[0])
                np.testing.assert_array_equal(image[0]+100,truth[1])
                self.assertTrue(image.flags.c_contiguous)

    def test_physical_image_sampling_is_not_weighted_by_annotators(self):
        images = [{'id':'a-one','file_name':'one.png'}]
        images += [{'id':f'{a}-many','file_name':'many.png'} for a in ['a','b','c']]
        data = {'images':images,'annotations':[annotation(i['id'],1,100,100) for i in images]}
        dataset = training.CropDataset(data,[i['id'] for i in images],None,crop=64,length=1000)
        one = sum(dataset.sample_record(i)[0]=='one.png' for i in range(1000))
        self.assertGreater(one,400)
        self.assertLess(one,600)

    def test_small_crop_uses_one_annotator_and_native_mask_coordinates(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = np.zeros((2048,2048),np.uint8)
            image[100:110,100:110] = 255
            image[1000:1010,1000:1010] = 255
            cv2.imwrite(str(root/'img.png'),image)
            images = [{'id':f'{a}-img','file_name':'img.png'} for a in ['a','b']]
            data = {'images':images,'annotations':[annotation('a-img',1,100,100),
                                                 annotation('b-img',1,1000,1000)]}
            store = training.ImageStore(root,root/'cache','raw')
            dataset = training.CropDataset(data,[i['id'] for i in images],store,crop=64,length=8,
                                           small_probability=1.,positive_probability=0.,augment=False)
            for index in range(8):
                _,group,top,left,_ = dataset.sample_record(index)
                x,y = dataset[index]
                self.assertEqual(tuple(x.shape),(3,64,64))
                self.assertEqual(tuple(y.shape),(2,64,64))
                self.assertEqual(int(y[0].sum()),100)
                self.assertGreater(int(y[1].sum()),0)
                value = 100 if group=='a-img' else 1000
                self.assertEqual(float(y[0,value-top,value-left]),1.)
                self.assertTrue(torch.all(y[1] <= y[0]))

    def test_tile_coverage_and_constant_logit_reconstruction(self):
        class Constant(torch.nn.Module):
            def forward(self,x):
                return torch.zeros(x.shape[0],2,*x.shape[-2:],device=x.device)
        positions = training.tile_positions(150,64,.25)
        self.assertEqual(positions[0],0)
        self.assertEqual(positions[-1],86)
        self.assertGreater(float(training.blend_window(64).min()),0.)
        image = np.zeros((150,180),np.uint8)
        probability = training.predict_foreground(Constant(),image,torch.device('cpu'),64,.25,batch=3)
        np.testing.assert_allclose(probability,.5,atol=1e-6)

    def test_cc_uses_eight_connectivity_and_area_filter(self):
        probability = np.zeros((12,12),np.float32)
        probability[1:4,1:4] = 1
        probability[4:7,4:7] = 1
        masks = training.components(probability,.5,10)
        self.assertEqual(len(masks),1)
        self.assertEqual(int(masks[0].sum()),18)
        self.assertEqual(training.components(probability,.5,20),[])

    @unittest.skipUnless(importlib.util.find_spec('segmentation_models_pytorch'), 'training dependency not installed')
    def test_model_native_resolution_and_cpu_backward_without_download(self):
        torch.set_num_threads(2)
        for encoder in ['resnet18','tu-convnext_tiny']:
            model = training.build_model(encoder,pretrained=False).eval()
            x = torch.randn(1,3,64,64)
            output = model(x)
            self.assertEqual(tuple(output.shape),(1,2,64,64))
            if encoder == 'resnet18':
                target = torch.zeros_like(output)
                target[:,0,20:30,20:30] = 1
                target[:,1,25,21:29] = 1
                training.segmentation_loss(output,target).backward()
                self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            del model,output
            gc.collect()


if __name__ == '__main__':
    unittest.main()
