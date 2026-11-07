"""CPU contracts for three-domain targets, deterministic resume and fail-closed saves."""
from pathlib import Path
import json
import random
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from pycocotools import mask as mu
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import swefil_mixed_training as worker


def encoded(mask):
    rle=mu.encode(np.asfortranarray(mask)); rle['counts']=rle['counts'].decode()
    return dict(segmentation=rle, area=int(mask.sum()), bbox=list(map(float,mu.toBbox(dict(rle,counts=rle['counts'].encode())))))


def data_fixture():
    polygon=[100,100,150,100,150,140,100,140]
    real=dict(images=[dict(id=1,file_name='real.jpeg',width=2048,height=2048)],
        annotations=[dict(id=1,image_id=1,segmentation=[polygon],spine=[105,120,145,120],area=2000,bbox=[100,100,50,40])])
    foreground=np.zeros((2048,2048),np.uint8); foreground[100:140,100:150]=1
    synthetic=dict(images=[dict(id=2,file_name='synthetic.png',width=2048,height=2048)],
        annotations=[dict(id=2,image_id=2,**encoded(foreground))])
    # Two separated parts belong to one external filament. A sunspot is negative.
    foreground[700:740,600:650]=1
    sunspot=np.zeros_like(foreground); sunspot[300:350,300:350]=1
    records=[dict(file_name='images/swefil.jpg',observation_id='20230101120000',
        width=2048,height=2048,spine_valid=False,
        annotations=[dict(id=3,category_id=1,**encoded(foreground)),dict(id=4,category_id=3,**encoded(sunspot))]),
        dict(file_name='images/sunspot_only.jpg',observation_id='20230102120000',
        width=2048,height=2048,spine_valid=False,annotations=[dict(id=5,category_id=3,**encoded(sunspot))])]
    return real,synthetic,records,foreground,sunspot


class ImageStore:
    def load(self, _): return np.full((2048,2048),128,np.uint8)


class ToyDataset:
    epoch=0
    def __len__(self): return 6
    def __getitem__(self,index):
        x=torch.randn(3,4,4)+(random.random()+float(np.random.random()))*.01
        fg=(x[0]>.1).float(); spine=(x[1]>.6).float()*fg
        return x,torch.stack([fg,spine]),torch.tensor(float(index%2))


def toy_model(_):
    return nn.Sequential(nn.Conv2d(3,4,1),nn.ReLU(),nn.Dropout2d(.2),nn.Conv2d(4,2,1))


def toy_validator(model,device,amp,cfg):
    torch.rand(2);random.random();np.random.random()
    pq=float(torch.sigmoid(next(model.parameters()).detach().mean()))
    row=dict(pq=pq,threshold=.5,min_area=100,tp=1,fp=0,fn=1)
    return row,[row],{(.5,100):{'fixture':[]}}


class FakeVerifiedStore:
    """Disk-backed server substitute; the production store has independent API tests."""
    def __init__(self, root, identity='fixture-v1'):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
        self.identity=identity; self.fail_epoch=None; self.commits=[]; self.preflight_budgets=[]
    def preflight(self,required_free_bytes=None): self.preflight_budgets.append(required_free_bytes)
    def latest_state(self):
        path=self.root/'commit.json'
        return worker.original.read_envelope(path,self.identity) if path.exists() else None
    def publish_checkpoint(self,path,relative_path,metadata):
        checksum=worker.digest(path); key=checksum+'.pt'
        shutil.copyfile(path,self.root/key)
        return dict(file_id=key,path=relative_path,sha256=checksum,size=Path(path).stat().st_size,name=Path(path).name)
    def commit_state(self,payload,expected_previous_epoch):
        if payload['epoch']==self.fail_epoch: raise OSError('Server acknowledgement failed')
        current=self.latest_state()
        if (current['epoch'] if current else 0)!=expected_previous_epoch: raise ValueError('Concurrent epoch')
        for key in ('last','best'): self.restore(payload[key])
        worker.original.write_envelope(self.root/'commit.json',self.identity,payload)
        self.commits.append(payload['epoch'])
        return payload
    def restore(self,record):
        path=self.root/record['file_id']
        if worker.digest(path)!=record['sha256']: raise ValueError('Server checkpoint checksum differs')
        return path


def toy_config(root):
    return dict(worker.DEFAULTS,device='cpu',epochs=4,batch=1,accumulate=2,
        steps_per_epoch=6,crop=4,lr=.003,validate_every=1,workers=0,
        output_root=str(root),cache_root=str(root.parent/(root.name+'_cache')))


def train(root,cap,store=None):
    store=store or FakeVerifiedStore(root/'server')
    return worker._run_epochs(toy_config(root),ToyDataset(),toy_model,toy_validator,
        root/'train','fixture-v1',store,run_until=cap)


def last_checkpoint(root):
    store=FakeVerifiedStore(root/'server')
    return worker._load_remote(store,store.latest_state()['last'],'fixture-v1')


def assert_nested_equal(case,left,right):
    if isinstance(left,torch.Tensor): case.assertTrue(torch.equal(left,right))
    elif isinstance(left,dict):
        case.assertEqual(set(left),set(right))
        for key in left: assert_nested_equal(case,left[key],right[key])
    elif isinstance(left,(tuple,list)):
        case.assertEqual(len(left),len(right))
        for a,b in zip(left,right): assert_nested_equal(case,a,b)
    else: case.assertEqual(left,right)


class SWEFilTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        previous=torch.get_num_threads();torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads,previous)

    def dataset(self):
        real,synthetic,records,_,_=data_fixture()
        return worker.ThreeDomainCropDataset(real,[1],ImageStore(),synthetic,ImageStore(),
            worker.swefil_coco(records),ImageStore(),crop=64,length=400,seed=42,
            synthetic_ratio=.1,swefil_ratio=.6,warmup_epochs=3,real_finish_epochs=6,
            total_epochs=100,cache_dir=self.root/'targets',augment=False)

    def test_schedule_and_crop_budget(self):
        ds=self.dataset()
        self.assertEqual(len(ds),400)
        for epoch in (1,3,95,100):
            self.assertEqual(ds.domain_probabilities(epoch),dict(real=1.,swefil=0.,synthetic=0.))
        for epoch in (4,50,94):
            self.assertEqual(ds.domain_probabilities(epoch),dict(real=.3,swefil=.6,synthetic=.1))
        ds.epoch=4
        rows=[ds.sample_record(i) for i in range(400)]
        counts=worker.Counter(r['domain'] for r in rows)
        self.assertTrue(80<counts['real']<160)
        self.assertTrue(200<counts['swefil']<280)
        self.assertTrue(15<counts['synthetic']<65)
        for i,row in enumerate(rows):
            again=ds.sample_record(i)
            for key in ('domain','file_name','group_id','top','left'): self.assertEqual(row[key],again[key])

    def test_real_only_crop_matches_original_sampler(self):
        ds=self.dataset();real,_,_,_,_=data_fixture()
        old=worker.base.MixedCropDataset(real,[1],ImageStore(),crop=64,length=400,seed=42,total_epochs=100,augment=False)
        ds.epoch=old.epoch=1
        for i in range(20):
            left=ds.sample_record(i);right=old.sample_record(i)
            for key in ('domain','file_name','group_id','top','left'): self.assertEqual(left[key],right[key])

    def test_external_union_target_sunspot_negative_and_missing_spine(self):
        ds=self.dataset();_,_,_,expected,sunspot=data_fixture()
        fg,spine,valid=ds._target('swefil','swefil_0')
        self.assertTrue(np.array_equal(fg,expected));self.assertFalse(valid);self.assertFalse(spine.any())
        self.assertFalse(fg[sunspot.astype(bool)].any())
        blank,spine,valid=ds._target('swefil','swefil_1')
        self.assertFalse(blank.any());self.assertFalse(valid)
        self.assertEqual(ds.domain_info['swefil']['physical_images'],2)
        self.assertEqual(ds.domain_info['swefil']['annotation_instances'],1)
        real_fg,real_spine,real_valid=ds._target('real','1')
        self.assertTrue(real_valid);self.assertTrue(real_spine.any())

    def test_foreground_gradients_without_invented_external_spines(self):
        logits=torch.zeros((3,2,4,4),requires_grad=True)
        target=torch.zeros_like(logits);target[:,:,1:3,1:3]=1
        worker.base.mixed_segmentation_loss(logits,target,torch.tensor([1.,0.,0.]),.1).backward()
        self.assertGreater(torch.count_nonzero(logits.grad[0,1]).item(),0)
        for index in (1,2):
            self.assertEqual(torch.count_nonzero(logits.grad[index,1]).item(),0)
            self.assertGreater(torch.count_nonzero(logits.grad[index,0]).item(),0)

    def test_recall_precision_use_same_instance_match_counts(self):
        rates=worker.detection_rates(dict(tp_pairs=80,fp=20,fn=40))
        self.assertAlmostEqual(rates['precision'],.8)
        self.assertAlmostEqual(rates['recall'],2/3)
        self.assertEqual(worker.detection_rates(dict(tp_pairs=0,fp=0,fn=0)),dict(recall=0.,precision=0.))

    def test_fresh_config_and_old_directory_preserved(self):
        old=self.root/'old';old.mkdir()
        oldcfg={key:str(self.root/key) for key in ('images','annotations','train_split','val_split','chase_root','chase_manifest','cache_root','output_root')}
        worker.atomic_json(old/'experiment_config.json',oldcfg)
        before=(old/'experiment_config.json').read_bytes()
        cfg=dict(base=str(self.root),chase_training_root=str(old),output_root=str(self.root/'new'),cache_root=str(self.root/'local'))
        normalized=worker.normalize_config(cfg)
        self.assertEqual(normalized['epochs'],100);self.assertEqual(normalized['steps_per_epoch'],400)
        self.assertEqual(normalized['synthetic_ratio'],.1);self.assertEqual(normalized['swefil_ratio'],.6)
        self.assertEqual(normalized['initialization'],'imagenet_encoder_new_decoder')
        for change in ({'epochs':60},{'initial_checkpoint':'best42.pt'},{'swefil_ratio':.5},{'output_root':str(old)}):
            with self.assertRaises(ValueError):worker.normalize_config(dict(cfg,**change))
        self.assertEqual(before,(old/'experiment_config.json').read_bytes())

    def test_resumed_optimizer_scheduler_weights_and_rng_match_full_run(self):
        a=self.root/'whole';b=self.root/'resumed'
        train(a,4);train(b,2);result=train(b,4)
        left=last_checkpoint(a);right=last_checkpoint(b)
        for key in ('model_state_dict','optimizer_state_dict','scheduler_state_dict','rng_state','best_epoch','best_pq'):
            assert_nested_equal(self,left[key],right[key])
        self.assertEqual([r['train_loss'] for r in left['history']],[r['train_loss'] for r in right['history']])
        self.assertEqual([r['optimizer_updates'] for r in right['history']],[3]*4)
        self.assertEqual(result['status'],'complete')
        original=(b/'server/commit.json').read_bytes();train(b,4)
        self.assertEqual(original,(b/'server/commit.json').read_bytes())

    def test_failed_remote_commit_stops_before_next_epoch_and_resumes(self):
        reference=self.root/'reference';root=self.root/'interrupted'
        train(reference,4);store=FakeVerifiedStore(root/'server');train(root,2,store)
        original=(root/'server/commit.json').read_bytes();store.fail_epoch=3
        with self.assertRaisesRegex(OSError,'acknowledgement'):train(root,4,store)
        self.assertEqual(original,(root/'server/commit.json').read_bytes())
        self.assertEqual(store.commits,[1,2])
        store.fail_epoch=None;train(root,4,store)
        left=last_checkpoint(reference);right=last_checkpoint(root)
        for key in ('model_state_dict','optimizer_state_dict','scheduler_state_dict','rng_state'):
            assert_nested_equal(self,left[key],right[key])

    def test_remote_authority_ignores_stale_mirror_but_missing_remote_fails(self):
        root=self.root/'guard';train(root,2)
        (root/'train/state.json').write_text('{"epoch":0}')
        train(root,3)
        self.assertEqual(FakeVerifiedStore(root/'server').latest_state()['epoch'],3)
        (root/'server/commit.json').unlink()
        with self.assertRaisesRegex(RuntimeError,'remote commit is missing'):train(root,4)

    def test_no_remote_generation_is_deleted(self):
        root=self.root/'keep';store=FakeVerifiedStore(root/'server');train(root,2,store)
        initial={p.name for p in (root/'server').glob('*.pt')};train(root,4,store)
        self.assertTrue(initial<={p.name for p in (root/'server').glob('*.pt')})
        self.assertEqual(store.commits,[1,2,3,4])

    def test_completed_target_does_not_build_model_or_check_training_quota(self):
        root=self.root/'complete';store=FakeVerifiedStore(root/'server');train(root,2,store)
        self.assertEqual(store.preflight_budgets,[50_000_000_000])
        with patch.object(worker.base,'make_epoch_loader',side_effect=AssertionError('No loader')):
            result=worker._run_epochs(toy_config(root),ToyDataset(),
                lambda fresh: (_ for _ in ()).throw(AssertionError('No model')),
                toy_validator,root/'train','fixture-v1',store,run_until=2)
        self.assertEqual(result['epochs_completed'],2)
        self.assertEqual(store.preflight_budgets,[50_000_000_000])
        train(root,4,store)
        self.assertEqual(store.preflight_budgets,[50_000_000_000,25_000_000_000])


if __name__=='__main__':unittest.main()
