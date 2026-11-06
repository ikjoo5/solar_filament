"""Actual ROI resume, box-only export, and oracle/actual separation checks."""
from pathlib import Path
import copy
import csv
import json
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from pycocotools import mask as mu
import torch
from torch import nn
from torch.utils.data import Dataset

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import roi_filament_pipeline as engine
import roi_filament_core as core


def encoded(box):
    mask=np.zeros((2048,2048),np.uint8);x1,y1,x2,y2=map(int,box);mask[y1:y2,x1:x2]=1
    return mu.encode(np.asfortranarray(mask))


class TinyModel(nn.Module):
    def __init__(self,*args,**kwargs):
        super().__init__();self.conv=nn.Conv2d(2,1,1);self.dropout=nn.Dropout2d(.25)
    def forward(self,x):return self.conv(self.dropout(x))


class TinyDataset(Dataset):
    def __init__(self,*args,steps_per_epoch=3,batch_size=2,seed=42,**kwargs):self.n=steps_per_epoch*batch_size;self.seed=seed;self.epoch=0
    def __len__(self):return self.n
    def set_epoch(self,e):self.epoch=e
    def __getitem__(self,index):
        generator=torch.Generator().manual_seed(self.seed+self.epoch*1000+index)
        x=torch.rand((2,32,32),generator=generator);x[1]=(x[1]>.5).float();return x,(x[:1]>.6).float()


class RoiPipelineTests(unittest.TestCase):
    def fixture(self,directory):
        root=Path(directory);images=root/'images';images.mkdir()
        names=['20110101000000Ah.jpeg','20110201000000Ah.jpeg']
        for name in names:cv2.imwrite(str(images/name),np.full((2048,2048),125,np.uint8))
        cohort=dict(train_groups=[dict(group_id='train',file_name=names[0],annotations=[],role='train')],
                    validation_names=[names[1]],input_sha256={'annotations':'fixture'},
                    truth={'v1':[encoded([40,50,70,80])],'v2':[encoded([120,140,150,170])]},
                    mapping={'v1':Path(names[1]).stem,'v2':Path(names[1]).stem})
        cfg=dict(images=str(images),output_root=str(root/'run'),cache_root=str(root/'cache'),
                 strict=False,device='cpu',amp=False,roi_pretrained=False,roi_encoder='resnet18',
                 roi_size=32,roi_epochs=2,roi_batch=2,roi_accumulate=2,roi_steps_per_epoch=3,preprocessing='raw')
        Path(cfg['output_root']).mkdir()
        return root,cfg,cohort

    def test_actual_full_optimizer_resume_matches_continuous_and_is_idempotent(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            root,cfg,cohort=self.fixture(directory);other=dict(cfg,output_root=str(root/'resume'))
            proposals={'20110101000000Ah':[dict(box=[20,20,80,80],score=.4)]}
            with patch.object(engine,'load_cohort',return_value=cohort),patch.object(engine,'source_hashes',return_value={'fixture':'source'}), \
                 patch.object(core,'RoiFilamentNet',TinyModel),patch.object(core,'RoiTrainingDataset',TinyDataset):
                engine.train_roi(cfg,proposals,2)
                engine.train_roi(other,proposals,1)
                first=torch.load(root/'resume/roi/last_full.pt',map_location='cpu',weights_only=False)
                self.assertEqual(first['completed_epoch'],1);self.assertTrue(first['optimizer']['state'])
                engine.train_roi(other,proposals,2)
                a=torch.load(root/'run/roi/last_full.pt',map_location='cpu',weights_only=False)
                b=torch.load(root/'resume/roi/last_full.pt',map_location='cpu',weights_only=False)
                for key in a['model']:self.assertTrue(torch.equal(a['model'][key],b['model'][key]),key)
                self.assertEqual(b['history'][-1]['optimizer_attempts'],2) #3 batches/accumulate2 flushes last
                before=engine.digest(root/'resume/roi/last_full.pt');engine.train_roi(other,proposals,2)
                self.assertEqual(before,engine.digest(root/'resume/roi/last_full.pt'))
                with self.assertRaisesRegex(ValueError,'corpus changed'):
                    engine.train_roi(other,{'20110101000000Ah':[]},2)
                with self.assertRaisesRegex(ValueError,'Validation/test'):
                    engine.train_roi(other,{'20110201000000Ah':[]},2)

    def test_partial_epoch_preserves_prior_full_checkpoint(self):
        class InterruptDataset(TinyDataset):
            def __getitem__(self,index):
                if self.epoch==1 and index==2:raise KeyboardInterrupt('fixture mid-epoch')
                return super().__getitem__(index)
        with tempfile.TemporaryDirectory() as directory:
            root,cfg,cohort=self.fixture(directory)
            with patch.object(engine,'load_cohort',return_value=cohort),patch.object(engine,'source_hashes',return_value={'fixture':'source'}),patch.object(core,'RoiFilamentNet',TinyModel),patch.object(core,'RoiTrainingDataset',InterruptDataset):
                engine.train_roi(cfg,{},1);path=root/'run/roi/last_full.pt';before=engine.digest(path)
                with self.assertRaises(KeyboardInterrupt):engine.train_roi(cfg,{},2)
                self.assertEqual(before,engine.digest(path));self.assertFalse((path.parent/'epoch002.pt').exists())

    def test_actual_yolo_boxes_export_does_not_require_any_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root,cfg,cohort=self.fixture(directory);checkpoint=root/'best.pt';checkpoint.write_bytes(b'fixtureweights');cfg['yolo']=str(checkpoint)
            factory=types.SimpleNamespace();calls=[]
            class Model:
                def __init__(self,*args,**kwargs):calls.append('construct')
                def predict(self,**kwargs):
                    calls.append('predict');return [types.SimpleNamespace(boxes=types.SimpleNamespace(xyxy=torch.tensor([[10.,20.,30.,40.]]),conf=torch.tensor([.9])),masks=None)]
            factory.YOLO=Model
            with patch.dict(sys.modules,{'ultralytics':factory}),patch.object(engine,'source_hashes',return_value={'fixture':'source'}):
                stamps={name:engine.digest(Path(cfg['images'])/name) for name in ['20110101000000Ah.jpeg','20110201000000Ah.jpeg']}
                one=engine.export_yolo_proposals(cfg,'validation',cohort,stamps)
                again=engine.export_yolo_proposals(cfg,'validation',cohort,stamps)
            self.assertEqual(one,again);self.assertEqual(one['20110201000000Ah'][0]['box'],[10.,20.,30.,40.])
            self.assertEqual(calls,['construct','predict'])

    def test_gt_box_oracle_is_separate_for_alternative_raters(self):
        with tempfile.TemporaryDirectory() as directory:
            root,cfg,cohort=self.fixture(directory)
            per_group={'v1':cohort['truth']['v1'],'v2':cohort['truth']['v2']}
            score=engine.score_groups(cohort['truth'],cohort['mapping'],per_group,per_group=True)
            self.assertEqual(score['score']['pq'],1.);self.assertEqual(score['score']['physical_images'],1)
            self.assertEqual(score['score']['fp'],0)
            actual={cohort['mapping']['v1']:cohort['truth']['v1']+cohort['truth']['v2']}
            self.assertEqual(engine.score_groups(cohort['truth'],cohort['mapping'],actual)['score']['fp'],2)

    def test_mask_nms_and_ownership_output_nonoverlapping_instances(self):
        a=encoded([10,10,40,40]);b=encoded([30,10,60,40]);duplicate=encoded([11,10,41,40])
        candidates=[dict(rle=r,box_score=.8,selection_score=s) for r,s in [(a,.8),(b,.7),(duplicate,.6)]]
        result=engine.mask_predictions(candidates,.05,150)
        self.assertEqual(len(result),2);self.assertEqual(int(mu.area(mu.merge(result))),sum(map(int,mu.area(result))))

    def test_low_quality_roi_is_cached_before_optional_admission_gate(self):
        class Constant(nn.Module):
            def forward(self,x):return torch.full((x.shape[0],1,x.shape[-2],x.shape[-1]),float(torch.logit(torch.tensor(.55))))
        cfg=engine.config_defaults(dict(output_root='/tmp/roi_quality_fixture',cache_root='/tmp/roi_quality_cache',
            roi_size=32,device='cpu',amp=False,preprocessing='raw'))
        image=np.full((2048,2048),125,np.uint8)
        rows=engine.infer_rois(Constant(),image,[dict(box=[100,100,130,130],score=.9)],cfg,torch.device('cpu'))
        self.assertEqual(len(rows),1);self.assertLess(rows[0]['roi_quality'],.6)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'record.json.gz';engine.write_record(path,rows)
            cached=engine.decode_serialized_candidates(engine.read_record(path))
            self.assertEqual(len(engine.mask_predictions(cached,.05,150)),1)
            self.assertEqual(engine.mask_predictions(cached,.05,150,minimum_quality=.6),[])

    def test_actual_bbox_diagnostics_crowd_overlap_means_target_coverage(self):
        gt={'one':[encoded([100,100,120,120])]};mapping={'one':'image'}
        proposals={'image':[dict(box=[100,100,120,120],score=.9)]}
        report=engine.bbox_diagnostics(gt,mapping,proposals,1.5)['thresholds']['0.05']
        self.assertEqual(report['bbox_iou50'],1);self.assertEqual(report['expanded_crop_coverage95'],1)
        proposals={'image':[dict(box=[100,100,110,120],score=.9)]}
        report=engine.bbox_diagnostics(gt,mapping,proposals,1.)['thresholds']['0.05']
        self.assertEqual(report['bbox_iou50'],0);self.assertEqual(report['expanded_crop_coverage95'],0)
        # Square-context pixels above a thin proposed box are real image
        # context, not the narrower paper-style rectangular crop.
        gt={'one':[encoded([100,90,120,95])]}
        proposals={'image':[dict(box=[100,100,120,105],score=.9)]}
        report=engine.bbox_diagnostics(gt,mapping,proposals,1.5)['thresholds']['0.05']
        self.assertEqual(report['bbox_iou50'],0);self.assertEqual(report['expanded_crop_coverage95'],1)

    def test_compare_detr_reuses_roi_and_all_partial_detr_never_starts_roi(self):
        with tempfile.TemporaryDirectory() as directory:
            root,cfg,cohort=self.fixture(directory);cfg.update(detr_epochs=2);stem='20110201000000Ah'
            cfg=engine.config_defaults(cfg)
            roi=root/'run/roi/epoch002.pt';roi.parent.mkdir()
            fake=types.ModuleType('filament_detr');fake.train_detr=lambda *a,**k:dict(checkpoint='fixtureDETR')
            fake.export_proposals=lambda *a,**k:{stem:[]}
            with patch.dict(sys.modules,{'filament_detr':fake}),patch.object(engine,'load_cohort',return_value=cohort), \
                 patch.object(engine,'export_yolo_proposals',return_value={stem:[]}),patch.object(engine,'train_roi') as train, \
                 patch.object(engine,'evaluate_pipeline',return_value={'status':'fixtureevaluation'}) as evaluate,patch.object(engine,'export_results',return_value=[]), \
                 patch.object(engine,'source_hashes',return_value={'fixture':'source'}):
                stamps={name:engine.digest(Path(cfg['images'])/name) for name in ['20110101000000Ah.jpeg','20110201000000Ah.jpeg']}
                fixed=dict(static=engine.roi_protocol(cfg,cohort,stamps),proposal_corpus_sha256='fixture',corpus_label='yolo_train_only')
                runid=engine.identity(fixed);engine.atomic_json(roi.parent/'training_identity.json',dict(identity=runid,protocol=fixed))
                engine.atomic_torch(roi,dict(identity=runid,completed_epoch=2,fixed_final_epoch=2,model=TinyModel().state_dict()))
                before=engine.digest(roi)
                result=engine.run(cfg,'compare_detr',2,2)
                train.assert_not_called();self.assertEqual(evaluate.call_count,1)
                self.assertEqual(engine.digest(roi),before);self.assertEqual(result['status'],'complete')
                evaluate.reset_mock();result=engine.run(dict(cfg,output_root=str(root/'other')),'all',1,1)
                train.assert_not_called();evaluate.assert_not_called();self.assertEqual(result['status'],'detr_stage_complete_roi_not_started')

    def test_wrong_saved_roi_rejected_before_expensive_detr_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root,cfg,cohort=self.fixture(directory)
            roi=root/'run/roi/epoch002.pt';roi.parent.mkdir()
            engine.atomic_torch(roi,dict(identity='wrong',completed_epoch=2,fixed_final_epoch=2,model={'x':torch.zeros(1)}))
            engine.atomic_json(roi.parent/'training_identity.json',dict(identity='also_wrong',protocol={'static':{}}))
            fake=types.ModuleType('filament_detr');called=[];fake.train_detr=lambda *a,**k:called.append(True)
            with patch.dict(sys.modules,{'filament_detr':fake}),patch.object(engine,'load_cohort',return_value=cohort):
                with self.assertRaisesRegex(ValueError,'Wrong ROI'):engine.run(cfg,'compare_detr',2,30)
            self.assertEqual(called,[])

    def test_real_roi_geometry_eval_cache_and_per_rater_oracle_integration(self):
        class EchoPrompt(nn.Module):
            def __init__(self,*args,**kwargs):super().__init__();self.dummy=nn.Parameter(torch.zeros(()))
            def forward(self,inputs):return torch.where(inputs[:,1:2]>.5,10.,-10.)+self.dummy*0
        with tempfile.TemporaryDirectory() as directory:
            root,cfg,cohort=self.fixture(directory);cfg=engine.config_defaults(cfg)
            checkpoint=root/'run/roi/epoch002.pt';checkpoint.parent.mkdir()
            stamps={name:engine.digest(Path(cfg['images'])/name) for name in ['20110101000000Ah.jpeg','20110201000000Ah.jpeg']}
            with patch.object(engine,'load_cohort',return_value=cohort),patch.object(engine,'source_hashes',return_value={'fixture':'source'}),patch.object(core,'RoiFilamentNet',EchoPrompt):
                fixed=dict(static=engine.roi_protocol(cfg,cohort,stamps),proposal_corpus_sha256=engine.identity({}),corpus_label='yolo_train_only')
                runid=engine.identity(fixed);engine.atomic_json(checkpoint.parent/'training_identity.json',dict(identity=runid,protocol=fixed))
                engine.atomic_torch(checkpoint,dict(identity=runid,completed_epoch=2,fixed_final_epoch=2,model=EchoPrompt().state_dict()))
                proposals={'yolo':{'20110201000000Ah':[dict(box=[40.,50.,70.,80.],score=.9)]}}
                report=engine.evaluate_pipeline(cfg,proposals,str(checkpoint),2)
                oracle=report['gt_bbox_oracle']['results']['area300']['score']
                self.assertGreater(oracle['pq'],.9);self.assertEqual(oracle['fp'],0);self.assertEqual(oracle['fn'],0)
                actual=report['actual_detector_results']['yolo']['box0.05_area300']['score']
                self.assertEqual(actual['tp_pairs'],1);self.assertEqual(actual['fp'],1);self.assertEqual(actual['fn'],1)
                with patch.object(engine,'infer_rois',side_effect=AssertionError('cache should prevent inference')):
                    repeated=engine.evaluate_pipeline(cfg,proposals,str(checkpoint),2)
                self.assertEqual(report,repeated)


if __name__=='__main__':unittest.main()
