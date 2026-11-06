"""Actual tiny HF-DETR, native2048 target geometry, and epoch resume tests."""
from pathlib import Path
import csv
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from pycocotools import mask as mu
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import filament_detr as detr
from native_instance_data import load_cohort


def fixture(root):
    images=root/'images';images.mkdir()
    names=['20110101000000Ah.jpeg','20110201000000Ah.jpeg','20110301000000Ah.jpeg']
    for index,name in enumerate(names):
        pixels=np.full((2048,2048),110+index*10,np.uint8)
        cv2.rectangle(pixels,(320,420),(600,445),30,-1)
        cv2.imwrite(str(images/name),pixels)
    groups=[dict(id='r0',file_name=names[0]),dict(id='r1',file_name=names[0]),
            dict(id='r2',file_name=names[1]),dict(id='v0',file_name=names[2])]
    annotations=[]
    for sid in ('r0','r1','v0'):
        mask=np.zeros((2048,2048),np.uint8);mask[420:446,320:601]=1
        rle=mu.encode(np.asfortranarray(mask));rle['counts']=rle['counts'].decode()
        annotations.append(dict(id=sid+'_a',image_id=sid,segmentation=rle))
    annotation=root/'annotations.json';annotation.write_text(json.dumps(dict(images=groups,annotations=annotations)))
    for role,selected in [('train',groups[:3]),('val',groups[3:])]:
        with (root/(role+'.csv')).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['image_id','file_name']);writer.writeheader()
            for row in selected:writer.writerow(dict(image_id=row['id'],file_name=row['file_name']))
    return dict(images=str(images),annotations=str(annotation),train_split=str(root/'train.csv'),val_split=str(root/'val.csv'),
                output_root=str(root/'output'),cache_root=str(root/'cache'),strict=False,detr_tiny_fixture=True,
                detr_pretrained=False,detr_epochs=2,detr_image_size=64,detr_batch=1,detr_accumulate=2,
                detr_lr=1e-4,detr_backbone_lr=1e-5,detr_amp=False,device='cpu',preprocessing='raw',seed=42)


class DetrTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_actual_tiny_hf_detr_forward_backward_and_empty_labels(self):
        cfg=dict(strict=False,detr_tiny_fixture=True,detr_pretrained=False,detr_image_size=64,device='cpu')
        model=detr.build_detector(cfg,False)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        targets=[dict(class_labels=torch.tensor([0]),boxes=torch.tensor([[.5,.4,.3,.1]])),
                 dict(class_labels=torch.empty(0,dtype=torch.int64),boxes=torch.empty((0,4)))]
        result=model(pixel_values=torch.randn(2,3,64,64),labels=targets)
        self.assertTrue(torch.isfinite(result.loss));result.loss.backward();optimizer.step()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters()))

    def test_bbox_geometry_and_synchronized_flip_for_real2048_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg=fixture(Path(directory));cohort=load_cohort(cfg)
            dataset=detr.DetectionDataset(cohort['train_groups'],cfg['images'],cfg)
            self.assertEqual(len(dataset),3)
            self.assertEqual(cohort['counts']['train_physical_images'],2)
            pixels,target=dataset[0]
            expected=detr.box_tensor(cohort['train_groups'][0]['annotations'])
            np.testing.assert_allclose(expected.numpy(),[[(320+281/2)/2048,(420+26/2)/2048,281/2048,26/2048]],rtol=0,atol=1e-7)
            original=cv2.imread(str(Path(cfg['images'])/cohort['train_groups'][0]['file_name']),cv2.IMREAD_GRAYSCALE)
            wanted=detr.prepare_pixels(original,64,'raw')
            rng=np.random.default_rng(np.random.SeedSequence([42,0,0]))
            if rng.random()<.5:wanted=wanted.flip(-1);expected[:,0]=1-expected[:,0]
            if rng.random()<.5:wanted=wanted.flip(-2);expected[:,1]=1-expected[:,1]
            self.assertTrue(torch.equal(pixels,wanted));self.assertTrue(torch.equal(target['boxes'],expected))
            self.assertEqual(dataset[2][1]['boxes'].shape,(0,4))

    def test_real_hf_resume_exact_proposals_cache_and_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg=fixture(root)
            continuous=dict(cfg,output_root=str(root/'continuous'),cache_root=str(root/'cache_continuous'))
            resumed=dict(cfg,output_root=str(root/'resumed'),cache_root=str(root/'cache_resumed'))
            detr.train_detr(continuous,2)
            detr.train_detr(resumed,1)
            first=torch.load(root/'resumed/detr/last_full.pt',map_location='cpu',weights_only=False)
            self.assertEqual(first['completed_epoch'],1)
            self.assertEqual(first['history'][0]['optimizer_attempts'],2)
            self.assertTrue(first['optimizer']['state']);del first
            report=detr.train_detr(resumed,2)
            a=torch.load(root/'continuous/detr/last_full.pt',map_location='cpu',weights_only=False)
            b=torch.load(root/'resumed/detr/last_full.pt',map_location='cpu',weights_only=False)
            maximum=max(float((a['model'][key].float()-b['model'][key].float()).abs().max()) for key in a['model'])
            self.assertEqual(maximum,0.)
            self.assertEqual(b['history'][-1]['annotation_records'],3)
            self.assertEqual(b['history'][-1]['optimizer_updates'],2)
            self.assertEqual(b['completed_epoch'],2)
            before=detr.digest(root/'resumed/detr/last_full.pt')
            with patch.object(detr,'build_detector',side_effect=AssertionError('completed stage rebuilt')):
                detr.train_detr(resumed,2)
            self.assertEqual(detr.digest(root/'resumed/detr/last_full.pt'),before)
            with self.assertRaisesRegex(ValueError,'changed'):
                detr.train_detr(dict(resumed,detr_lr=2e-4),2)
            proposals=detr.export_proposals(resumed,report['checkpoint'],'validation')
            self.assertEqual(len(proposals),1)
            for rows in proposals.values():
                for row in rows:
                    self.assertEqual(len(row['box']),4);self.assertGreaterEqual(min(row['box']),0)
                    self.assertLessEqual(max(row['box']),2048)
                    self.assertNotIn('rle',row)
            with patch.object(detr,'build_detector',side_effect=AssertionError('completed proposals rebuilt')):
                self.assertEqual(proposals,detr.export_proposals(resumed,report['checkpoint'],'validation'))
            victim=Path(cfg['images'])/'20110301000000Ah.jpeg';victim.write_bytes(b'changed pixels')
            with self.assertRaisesRegex(ValueError,'differ'):
                detr.export_proposals(resumed,report['checkpoint'],'validation')
            evidence=dict(status='passed_actual_tiny_HF_DETR_CPU_training_resume',
                          architecture='HF DETR, native tiny ResNet backbone, 1encoder+1decoderlayer,10queries',
                          fixture='native2048 images,3trainraterrecords/2physicalimages,1validationphoto; no downloads',
                          continuous_vs_resumed_model_max_difference=maximum,completed_epoch=2,
                          remainder_accumulation_flushed=True,finished_stage_idempotent=True,
                          export_cache_reused=True,changed_input_pixels_rejected=True,
                          initial_parameter_sha256=json.loads((root/'resumed/detr/initialization.json').read_text())['initial_parameter_sha256'],
                          source_sha256=detr.source_hashes(),
                          competition_training=False,GPU_training=False)
            destination=Path(__file__).resolve().parents[1]/'docs/evidence/filament_detr_cpu_smoke.json'
            destination.write_text(json.dumps(evidence,indent=2)+'\n')

    def test_interrupted_epoch_preserves_last_completed_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg=fixture(root);detr.train_detr(cfg,1)
            path=root/'output/detr/last_full.pt';before=detr.digest(path)
            original=detr.DetectionDataset.__getitem__;count=0
            def interrupted(instance,index):
                nonlocal count
                count+=1
                if count>1:raise RuntimeError('fixture mid-epoch interruption')
                return original(instance,index)
            with patch.object(detr.DetectionDataset,'__getitem__',interrupted),self.assertRaisesRegex(RuntimeError,'mid-epoch'):
                detr.train_detr(cfg,2)
            self.assertEqual(detr.digest(path),before)
            self.assertFalse((root/'output/detr/epoch002.pt').exists())

    def test_production_defaults_are_bbox_only_true30_and_fixture_guarded(self):
        cfg=detr.normalize_config(dict(output_root='/tmp/test'))
        self.assertEqual((cfg['detr_epochs'],cfg['detr_image_size'],cfg['detr_batch'],cfg['detr_accumulate']),(30,800,1,4))
        with self.assertRaisesRegex(ValueError,'Tiny fixture'):
            detr.normalize_config(dict(detr_tiny_fixture=True))
        self.assertNotIn('min_area',cfg)


if __name__=='__main__':unittest.main()
