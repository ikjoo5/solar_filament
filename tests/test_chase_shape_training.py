"""CPU contract checks for the fixed best84 recipe and restartable new training."""
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from pycocotools import mask as mu
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import chase_shape_training as worker


def path_config(root):
    return {key:str(root/key) for key in ('images','annotations','train_split',
        'val_split','chase_root','chase_manifest','cache_root','output_root')}


class Store:
    def load(self, _):
        # Spatially variable image makes crop/augmentation equivalence meaningful.
        return np.tile(np.arange(2048,dtype=np.uint16).astype(np.uint8),(2048,1))


def domains():
    polygon=[100,100,150,100,150,140,100,140]
    real=dict(images=[dict(id=1,file_name='real.jpeg',width=2048,height=2048)],
        annotations=[dict(id=1,image_id=1,segmentation=[polygon],spine=[105,120,145,120],
            area=2000,bbox=[100,100,50,40])])
    mask=np.zeros((2048,2048),np.uint8);mask[100:140,100:150]=1
    encoded=mu.encode(np.asfortranarray(mask));encoded['counts']=encoded['counts'].decode()
    synthetic=dict(images=[dict(id=2,file_name='synthetic.png',width=2048,height=2048)],
        annotations=[dict(id=2,image_id=2,segmentation=encoded,area=int(mask.sum()),
            bbox=[100.,100.,50.,40.])])
    return real,synthetic,mask


class ToyDataset:
    epoch=0
    def __len__(self):return 6
    def __getitem__(self,index):
        # Exercise all restored RNGs, in addition to model dropout.
        x=torch.randn(3,4,4)+(random.random()+float(np.random.random()))*.01
        foreground=(x[0]>.1).float();spine=(x[1]>.6).float()*foreground
        return x,torch.stack([foreground,spine]),torch.tensor(float(index%2))


def toy_model(_):
    return nn.Sequential(nn.Conv2d(3,4,1),nn.ReLU(),nn.Dropout2d(.2),nn.Conv2d(4,2,1))


def toy_validator(model,device,amp,cfg):
    # Include validation RNG consumption, which must be in completed-epoch state.
    torch.rand(2);random.random();np.random.random()
    score=float(torch.sigmoid(next(model.parameters()).detach().mean()))
    row=dict(pq=score,threshold=.5,min_area=100,tp=1,fp=0,fn=1)
    return row,[row],{(.5,100):{'fixture.jpeg':[]}}


def toy_config(root):
    return dict(worker.DEFAULTS,device='cpu',epochs=4,batch=1,accumulate=2,
        steps_per_epoch=6,crop=4,lr=.003,validate_every=1,workers=0,
        output_root=str(root))


def train(root,cap,identity='fixture-v1'):
    return worker._run_epochs(toy_config(root),ToyDataset(),toy_model,toy_validator,
        root/'train',identity,run_until=cap)


def last_checkpoint(root):
    state=worker.read_envelope(root/'train/state.json','fixture-v1')
    return worker.load_checkpoint(root/'train',state['last'])


def assert_nested_equal(case,left,right):
    if isinstance(left,torch.Tensor):case.assertTrue(torch.equal(left,right))
    elif isinstance(left,dict):
        case.assertEqual(set(left),set(right))
        for key in left:assert_nested_equal(case,left[key],right[key])
    elif isinstance(left,(tuple,list)):
        case.assertEqual(len(left),len(right))
        for a,b in zip(left,right):assert_nested_equal(case,a,b)
    else:case.assertEqual(left,right)


class ChaseShapeTrainingTests(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory();self.root=Path(self.folder.name)
        self.addCleanup(self.folder.cleanup)
        self.old_threads=torch.get_num_threads();torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads,self.old_threads)

    def test_explicit_best84_recipe_and_user_mix_change(self):
        cfg=worker.normalize_config(path_config(self.root))
        self.assertEqual(cfg['epochs'],100)
        self.assertEqual(cfg['initialization'],'imagenet_encoder_new_decoder')
        self.assertEqual(cfg['batch']*cfg['steps_per_epoch'],400)
        self.assertEqual(cfg['batch']*cfg['accumulate'],4)
        self.assertEqual(cfg['steps_per_epoch']//cfg['accumulate'],100)
        self.assertEqual(cfg['synthetic_ratio'],.5)
        self.assertEqual(cfg['original_synthetic_ratio'],.2)
        self.assertEqual(cfg['preprocessing'],'limb_only')
        for changed in ({'epochs':60},{'lr':.01},{'initial_checkpoint':'best84.pt'},
                        {'synthetic_ratio':.2},{'encoder':'resnet34'}):
            with self.assertRaises(ValueError):
                worker.normalize_config(dict(path_config(self.root),**changed))

    def test_rle_mask_spine_ignore_and_original_real_sampler(self):
        real,synthetic,mask=domains()
        dataset=worker.RLEMixedCropDataset(real,[1],Store(),synthetic,Store(),crop=64,
            length=400,seed=42,synthetic_ratio=.5,warmup_epochs=3,real_finish_epochs=6,
            total_epochs=100,cache_dir=self.root/'targets',augment=True)
        foreground,spine,valid=dataset._target('synthetic','2')
        self.assertTrue(np.array_equal(foreground,mask));self.assertFalse(spine.any())
        self.assertFalse(valid)
        real_fg,real_spine,real_valid=dataset._target('real','1')
        self.assertTrue(real_valid);self.assertTrue(real_spine.any())
        self.assertTrue(np.all(real_spine<=real_fg))
        original=worker.base.MixedCropDataset(real,[1],Store(),crop=64,length=400,
            seed=42,total_epochs=100,augment=True)
        dataset.epoch=original.epoch=1
        for index in range(12):
            old=original.sample_record(index);new=dataset.sample_record(index)
            for key in ('domain','file_name','group_id','top','left'):
                self.assertEqual(old[key],new[key])
            for old_tensor,new_tensor in zip(original[index],dataset[index]):
                self.assertTrue(torch.equal(old_tensor,new_tensor))
        for epoch,expected in ((1,0),(3,0),(4,.5),(94,.5),(95,0),(100,0)):
            self.assertEqual(dataset.synthetic_probability(epoch),expected)
        dataset.epoch=4
        count=sum(dataset.sample_record(i)['domain']=='synthetic' for i in range(400))
        self.assertGreater(count,160);self.assertLess(count,240)
        # Both caches remain bound to exact content, not merely dimensions.
        synthetic_key=worker.fingerprint(dict(version=worker.VERSION,domain='synthetic',
            image=synthetic['images'][0],annotations=synthetic['annotations']))
        (self.root/'targets'/(synthetic_key+'.png')).write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError,'checksum'):dataset._target('synthetic','2')

    def test_synthetic_foreground_is_supervised_without_false_spine_negatives(self):
        logits=torch.zeros((2,2,4,4),requires_grad=True)
        target=torch.zeros_like(logits);target[:,:,1:3,1:3]=1
        worker.base.mixed_segmentation_loss(logits,target,torch.tensor([0.,1.]),.1).backward()
        self.assertGreater(torch.count_nonzero(logits.grad[0,0]).item(),0)
        self.assertEqual(torch.count_nonzero(logits.grad[0,1]).item(),0)
        self.assertGreater(torch.count_nonzero(logits.grad[1,1]).item(),0)

    def test_malformed_rle_area_rejected(self):
        real,synthetic,_=domains();synthetic['annotations'][0]['area']+=1
        with self.assertRaisesRegex(ValueError,'area/bbox'):
            worker.RLEMixedCropDataset(real,[1],Store(),synthetic,Store(),total_epochs=100)

    def test_limb_store_preprocesses_once_and_checks_cache_integrity(self):
        images=self.root/'images';images.mkdir();path=images/'source.png'
        cv2.imwrite(str(path),np.full((2048,2048),128,np.uint8))
        store=worker.VerifiedImageStore(images,self.root/'cache',{'source.png':worker.digest(path)})
        with patch.object(worker.base,'limb',side_effect=lambda x:x-1) as correction:
            first=store.load('source.png');second=store.load('source.png')
        self.assertEqual(correction.call_count,1)
        self.assertTrue(np.array_equal(first,second));self.assertEqual(int(first[0,0]),127)
        next((self.root/'cache').glob('*.png')).write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError,'checksum'):store.load('source.png')

    def test_resume_matches_uninterrupted_optimizer_scheduler_model_rng(self):
        whole=self.root/'whole';resumed=self.root/'resumed'
        train(whole,4);train(resumed,2);result=train(resumed,4)
        a=last_checkpoint(whole);b=last_checkpoint(resumed)
        for key in ('model_state_dict','optimizer_state_dict','scheduler_state_dict','rng_state','best_epoch','best_pq'):
            assert_nested_equal(self,a[key],b[key])
        self.assertEqual([r['train_loss'] for r in a['history']],[r['train_loss'] for r in b['history']])
        self.assertEqual([r['optimizer_updates'] for r in b['history']],[3]*4)
        self.assertEqual(result['status'],'complete')
        self.assertEqual(len(list((resumed/'train/checkpoints').glob('*.pt'))),2)
        state_before=(resumed/'train/state.json').read_bytes()
        train(resumed,4)
        self.assertEqual((resumed/'train/state.json').read_bytes(),state_before)

    def test_failed_checkpoint_does_not_commit_partial_epoch(self):
        reference=self.root/'reference';resumed=self.root/'interrupted'
        train(reference,4);train(resumed,2)
        committed=(resumed/'train/state.json').read_bytes()
        original=worker.save_checkpoint
        def fail_last(path,payload):
            if Path(path).name=='last_epoch003.pt':raise OSError('fixture interrupted write')
            return original(path,payload)
        with patch.object(worker,'save_checkpoint',side_effect=fail_last):
            with self.assertRaisesRegex(OSError,'interrupted'):train(resumed,4)
        self.assertEqual((resumed/'train/state.json').read_bytes(),committed)
        train(resumed,4)
        a=last_checkpoint(reference);b=last_checkpoint(resumed)
        for key in ('model_state_dict','optimizer_state_dict','scheduler_state_dict','rng_state'):
            assert_nested_equal(self,a[key],b[key])

    def test_wrong_identity_and_checkpoint_corruption_fail_before_training(self):
        root=self.root/'guard';train(root,2)
        original=(root/'train/state.json').read_bytes()
        with self.assertRaisesRegex(ValueError,'identity'):train(root,4,'other-source')
        self.assertEqual((root/'train/state.json').read_bytes(),original)
        state=worker.read_envelope(root/'train/state.json','fixture-v1')
        path=root/'train'/state['last']['path'];path.write_bytes(b'broken')
        with self.assertRaisesRegex(ValueError,'checksum'):train(root,4)

    def test_optional_original_checkpoint_is_evaluation_only_and_cached(self):
        checkpoint=self.root/'best84.pt'
        model=toy_model(False)
        torch.save(dict(epoch=84,model_state_dict=model.state_dict()),checkpoint)
        cfg=dict(worker.DEFAULTS,device='cpu',baseline_checkpoint=str(checkpoint))
        output=self.root/'evaluation';output.mkdir()
        row=dict(pq=.4,threshold=.5,min_area=100)
        with patch.object(worker,'BASELINE_SHA',worker.digest(checkpoint)), \
             patch.object(worker.base,'build_model',side_effect=lambda *a,**kw:toy_model(False)) as build, \
             patch.object(worker.base,'validate_native',return_value=(row,[row],{})) as validation:
            a=worker.evaluate_baseline(cfg,object(),['fixture.jpeg'],output,'input-v1')
            b=worker.evaluate_baseline(cfg,object(),['fixture.jpeg'],output,'input-v1')
            self.assertEqual(a,b);self.assertEqual(validation.call_count,1)
            build.assert_called_once_with('tu-convnext_tiny',pretrained=False)
            self.assertEqual(a['input'],'limb_only')
            self.assertFalse((output/'checkpoints').exists())
            with self.assertRaisesRegex(ValueError,'identity'):
                worker.evaluate_baseline(cfg,object(),['fixture.jpeg'],output,'changed-input')

    def test_streamed_checkpoint_digest_matches_file_and_failure_keeps_previous(self):
        path=self.root/'checkpoints/last_epoch001.pt'
        record=worker.save_checkpoint(path,dict(epoch=1,value=torch.arange(5)))
        self.assertEqual(record['sha256'],worker.digest(path));original=path.read_bytes()
        def broken(payload,stream):stream.write(b'partial');raise OSError('disk full')
        with patch.object(worker.torch,'save',side_effect=broken):
            with self.assertRaises(OSError):worker.save_checkpoint(path,dict(epoch=2))
        self.assertEqual(path.read_bytes(),original)
        self.assertFalse(path.with_name(path.name+'.pending').exists())


if __name__=='__main__':unittest.main()
