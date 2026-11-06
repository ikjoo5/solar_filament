"""Geometry, annotator accounting and durable-cache tests; CPU fixtures only."""
from pathlib import Path
import json
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import detr_longrun_eval as evaluation
import roi_filament_pipeline as reference
from roi_filament_core import RoiFilamentNet


class PromptMask(nn.Module):
    def forward(self, inputs):
        return (inputs[:,1:2]-.5)*16


def encoded(box):
    mask = np.zeros((2048,2048), np.uint8)
    x1,y1,x2,y2 = box
    mask[y1:y2,x1:x2] = 1
    return mu.encode(np.asfortranarray(mask))


class LongrunEvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_batched_geometry_matches_serial_edge_fractional_and_filtered_boxes(self):
        image = np.full((2048,2048), 125, np.uint8)
        rows = [dict(box=[0.,0.,40.,120.],score=.95),
                dict(box=[1999.3,1998.8,2048.,2048.],score=.9),
                dict(box=[900.,900.,1000.,1005.],score=.7),
                dict(box=[100.,100.,500.,500.],score=.59)]
        cfg = dict(proposal_confidence=.6, roi_context=1.5, roi_size=64,
                   mask_threshold=.5, amp=False)
        model = PromptMask()
        serial = reference.infer_rois(model,image,rows,cfg,torch.device('cpu'))
        for batch in (1,2,8):
            actual = evaluation.infer_rois_batched(model,image,rows,cfg,torch.device('cpu'),batch)
            self.assertEqual(len(actual), 3)
            for new,old,proposal in zip(actual,serial,rows):
                self.assertEqual(new['rle'],old['rle'])
                self.assertEqual(new['roi_quality'],old['roi_quality'])
                self.assertEqual(new['selection_score'],old['selection_score'])
                self.assertEqual(new['box'],proposal['box'])
                self.assertEqual(new['box_id'],evaluation.identity(proposal['box']))
        class FailOnForward(nn.Module):
            def forward(self, inputs): raise AssertionError('empty candidates must not run ROI')
        self.assertEqual(evaluation.infer_rois_batched(FailOnForward(),image,rows[-1:],cfg,torch.device('cpu'),8),[])

    def test_actual_resnet18_roi_cpu_batch_agrees_with_serial(self):
        torch.manual_seed(7)
        model = RoiFilamentNet(encoder='resnet18',encoder_weights=None).eval()
        image = np.random.default_rng(5).integers(0,256,(2048,2048),dtype=np.uint8)
        rows = [dict(box=[10.,20.,110.,150.],score=.9),dict(box=[1900.,1960.,2048.,2048.],score=.85)]
        cfg = dict(proposal_confidence=.6,roi_context=1.5,roi_size=64,mask_threshold=.5,amp=False)
        serial = reference.infer_rois(model,image,rows,cfg,torch.device('cpu'))
        batch = evaluation.infer_rois_batched(model,image,rows,cfg,torch.device('cpu'),8)
        self.assertEqual(len(serial),len(batch))
        self.assertGreater(len(batch),0)
        for a,b in zip(serial,batch):
            overlap = float(mu.iou([a['rle']],[b['rle']],[0])[0,0])
            self.assertGreater(overlap,.9999)
            self.assertAlmostEqual(a['roi_quality'],b['roi_quality'],places=6)

    def fixture(self, directory):
        root = Path(directory)
        images = root/'images';images.mkdir()
        names = ['20110101000000Ah.jpeg','20110201000000Ah.jpeg']
        for name in names: cv2.imwrite(str(images/name), np.full((2048,2048),125,np.uint8))
        checkpoint=root/'detector.pt';checkpoint.write_bytes(b'checkpoint fixture 1')
        roi_checkpoint=root/'roi.pt';roi_checkpoint.write_bytes(b'frozen ROI fixture')
        truth={'a':[encoded([100,100,164,164])], 'b':[encoded([300,300,364,364])],
               'c':[encoded([700,700,764,764])]}
        mapping={'a':Path(names[0]).stem,'b':Path(names[0]).stem,'c':Path(names[1]).stem}
        cohort=dict(validation_names=names,truth=truth,mapping=mapping,
            counts=dict(validation_physical_images=2,validation_annotation_groups=3,validation_instances=3))
        old=dict(images=str(images),roi_context=1.5,roi_size=96,amp=False,roi_epochs=30,
                 roi_encoder='resnet18',preprocessing='raw',strict=False)
        cfg=dict(output_root=str(root/'run'),device='cpu',detector_precision='fp32',
                 detector_image_size=64,roi_eval_batch=8)
        context=dict(base_config=old,cohort=cohort,stamps={n:evaluation.digest(images/n) for n in names},
            roi_checkpoint=roi_checkpoint,roi_sha=evaluation.digest(roi_checkpoint),identity='fixturecontext',
            protocol={'settings':evaluation.evaluation_settings(cfg,old)})
        return root,cfg,context,checkpoint

    def test_complete_rater_scoring_empty_records_resume_and_weight_bound_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root,cfg,context,checkpoint=self.fixture(directory)
            outputs=[[dict(box=[100.,100.,164.,164.],score=.85)],[]]
            torch.manual_seed(234);before=torch.get_rng_state().clone()
            def models(*args):
                torch.rand(3)  # model initialization must not change training RNG
                return nn.Linear(1,1),PromptMask()
            with patch.object(evaluation,'_load_models',side_effect=models) as load, \
                 patch.object(evaluation.probe,'infer_detr_preprocessed',side_effect=outputs) as infer:
                result=evaluation.evaluate_checkpoint(cfg,checkpoint,60,context=context)
                self.assertEqual(infer.call_count,2);self.assertEqual(load.call_count,1)
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
            score=result['score']
            self.assertEqual((score['tp_pairs'],score['fp'],score['fn']),(1,1,2))
            self.assertAlmostEqual(score['pq'],.4)
            self.assertEqual(score['annotation_groups'],3)
            self.assertEqual(result['physical_images_processed'],2)
            self.assertEqual(result['physical_images_with_empty_predictions'],1)
            self.assertEqual(result['selected_confidence'],.6)
            self.assertEqual(result['fixed_08_score'],score)
            self.assertEqual(result['confidence_grid_results']['0.9']['score']['fn'],3)
            folder=Path(result['evaluation_dir'])
            self.assertEqual(len(list((folder/'records').glob('*.json.gz'))),2)
            with patch.object(evaluation,'_load_models',side_effect=AssertionError('cache must skip models')):
                again=evaluation.evaluate_checkpoint(cfg,checkpoint,60,context=context)
                self.assertEqual(again['score'],score)
            # Interrupt-resume: a missing final photo is recomputed, not omitted.
            missing=folder/'records'/(Path(context['cohort']['validation_names'][1]).stem+'.json.gz')
            missing.unlink()
            with patch.object(evaluation,'_load_models',side_effect=models), \
                 patch.object(evaluation.probe,'infer_detr_preprocessed',return_value=[]) as infer:
                resumed=evaluation.evaluate_checkpoint(cfg,checkpoint,60,context=context)
                self.assertEqual(infer.call_count,1);self.assertEqual(resumed['score'],score)
            checkpoint.write_bytes(b'checkpoint fixture 2')
            with patch.object(evaluation,'_load_models',side_effect=models), \
                 patch.object(evaluation.probe,'infer_detr_preprocessed',side_effect=outputs) as infer:
                changed=evaluation.evaluate_checkpoint(cfg,checkpoint,60,context=context)
                self.assertEqual(infer.call_count,2)
            self.assertNotEqual(changed['identity'],result['identity'])
            self.assertNotEqual(changed['evaluation_dir'],result['evaluation_dir'])
            # Pixel mutation is refused, even when completed predictions exist.
            image=Path(context['base_config']['images'])/context['cohort']['validation_names'][0]
            image.write_bytes(b'changed image')
            with self.assertRaisesRegex(ValueError,'Validation image changed'):
                evaluation.evaluate_checkpoint(cfg,checkpoint,60,context=context)

    def test_fixed_grid_guards_and_missing_photo_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            _,cfg,context,checkpoint=self.fixture(directory)
            with self.assertRaisesRegex(ValueError,'confidence grid'):
                evaluation.evaluation_settings(dict(cfg,eval_box_thresholds=[.5,.8,.9]),context['base_config'])
            with self.assertRaisesRegex(ValueError,'fixed eval_min_area'):
                evaluation.evaluation_settings(dict(cfg,eval_min_area=150),context['base_config'])
            context['cohort']['validation_names'].pop()
            with self.assertRaisesRegex(ValueError,'Incomplete validation'):
                evaluation.evaluate_checkpoint(cfg,checkpoint,60,context=context)

    def test_retention_and_export_preserve_selected_only_and_checksums(self):
        with tempfile.TemporaryDirectory() as directory:
            root,cfg,context,checkpoint=self.fixture(directory)
            reports=[]
            for epoch in (60,61,62):
                with patch.object(evaluation,'_load_models',return_value=(nn.Linear(1,1),PromptMask())), \
                     patch.object(evaluation.probe,'infer_detr_preprocessed',return_value=[]):
                    reports.append(evaluation.evaluate_checkpoint(cfg,checkpoint,epoch,context=context))
            out=Path(cfg['output_root'])
            unknown=out/'evaluations'/'user_folder';unknown.mkdir();(unknown/'keep.txt').write_text('keep')
            incomplete=out/'evaluations'/'pending_eval';incomplete.mkdir()
            (incomplete/'owner.json').write_text(json.dumps({'owner':'detr_longrun_eval'}))
            kept=[reports[0]['evaluation_dir'],reports[-1]['evaluation_dir']]
            removed=evaluation.retain_evaluations(cfg,kept)
            self.assertEqual(removed,[reports[1]['evaluation_dir']])
            self.assertTrue(unknown.exists());self.assertTrue(incomplete.exists())
            self.assertEqual(len(list((out/'epoch_reports').glob('*.json'))),1)
            (out/'last_full.pt').write_bytes(b'no weights in export')
            (out/'result.json').write_text(json.dumps({'status':'paused'}))
            archives=evaluation.export_results(cfg,kept)
            self.assertGreaterEqual(len(archives),1)
            for name in archives:
                self.assertLessEqual(Path(name).stat().st_size,28*1024*1024)
                with zipfile.ZipFile(name) as z:
                    self.assertIsNone(z.testzip())
                    self.assertFalse(any(p.endswith('.pt') for p in z.namelist()))
                    self.assertFalse(any('user_folder' in p for p in z.namelist()))
                    manifest=json.loads(z.read('export_manifest.json'))
                    for member in z.namelist():
                        if member=='export_manifest.json':continue
                        import hashlib
                        self.assertEqual(hashlib.sha256(z.read(member)).hexdigest(),manifest['entries'][member]['sha256'])

    def test_export_replacement_failure_retains_previous_set_and_unknown_archives(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'evaluations'/'fixture';folder.mkdir(parents=True)
            (folder/'report.json').write_text(json.dumps({'status':'complete'}))
            cfg=dict(output_root=str(root))
            previous=evaluation.export_results(cfg,[str(folder)])
            pointer=root/'exports'/'current_exports.json'
            previous_pointer=pointer.read_bytes()
            unknown=root/'exports'/'detr_longrun_user_part01.zip'
            with zipfile.ZipFile(unknown,'w') as archive:
                archive.writestr('export_manifest.json',json.dumps({'owner':'user'}))
            unknown_sha=evaluation.digest(unknown)
            # Expanded-size partitioning forces two parts without large ZIPs.
            (folder/'one.json.gz').write_bytes(b'a'*(14*1024*1024))
            (folder/'two.json.gz').write_bytes(b'b'*(14*1024*1024))
            actual_zipfile=zipfile.ZipFile
            def fail_second(file,mode='r',*args,**kwargs):
                if mode=='w' and Path(file).name.endswith('_part02.pending'):
                    raise OSError('fixture second-part write failed')
                return actual_zipfile(file,mode,*args,**kwargs)
            with patch.object(evaluation.zipfile,'ZipFile',side_effect=fail_second):
                with self.assertRaisesRegex(OSError,'second-part write failed'):
                    evaluation.export_results(cfg,[str(folder)])
            self.assertEqual(pointer.read_bytes(),previous_pointer)
            self.assertTrue(all(Path(path).is_file() for path in previous))
            self.assertFalse(list((root/'exports').glob('*.pending')))
            replacement=evaluation.export_results(cfg,[str(folder)])
            self.assertEqual(len(replacement),2)
            self.assertTrue(all(Path(path).is_file() for path in replacement))
            self.assertFalse(any(Path(path).exists() for path in previous))
            self.assertEqual(json.loads(pointer.read_text())['archives'],replacement)
            self.assertEqual(evaluation.digest(unknown),unknown_sha)


if __name__=='__main__':
    unittest.main()
