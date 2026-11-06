"""Inference-only orchestration: real ROI geometry, paired scoring and resume."""
from pathlib import Path
import csv
import gzip
import hashlib
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

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import detr_best84_pipeline as pipeline
import roi_filament_pipeline as roi
import submission_inference as inference


class PromptMask(nn.Module):
    def forward(self, inputs):return (inputs[:,1:2]-.5)*16


def encoded(box):
    pixels=np.zeros((2048,2048),np.uint8);x1,y1,x2,y2=box;pixels[y1:y2,x1:x2]=1
    return mu.encode(np.asfortranarray(pixels))


def jsonable(value):return roi.json_ready(value)


def fixture(root):
    images=root/'images';images.mkdir()
    names=['20110101000000Ah.jpeg','20110201000000Ah.jpeg','20110301000000Ah.jpeg']
    for name in names:cv2.imwrite(str(images/name),np.full((2048,2048),125,np.uint8))
    rows=[dict(id='train',file_name=names[0],width=2048,height=2048),
          dict(id='val_a',file_name=names[1],width=2048,height=2048),
          dict(id='val_b',file_name=names[1],width=2048,height=2048),
          dict(id='val_c',file_name=names[2],width=2048,height=2048)]
    boxes=[[900,900,964,964],[100,100,164,164],[300,300,364,364],[700,700,764,764]]
    annotations=[]
    for i,(row,box) in enumerate(zip(rows,boxes)):
        x1,y1,x2,y2=box
        annotations.append(dict(id=i,image_id=row['id'],segmentation=[[x1,y1,x2,y1,x2,y2,x1,y2]]))
    ann=root/'annotations.json';ann.write_text(json.dumps(dict(images=rows,annotations=annotations)))
    for filename,selection in [('train.csv',rows[:1]),('val.csv',rows[1:])]:
        with (root/filename).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['image_id','file_name']);writer.writeheader()
            writer.writerows(dict(image_id=row['id'],file_name=row['file_name']) for row in selection)
    native=root/'native.pt';native.write_bytes(b'native fixture weights')
    roi_weight=root/'roi.pt';roi_weight.write_bytes(b'ROI fixture weights')
    cfg=dict(output_root=str(root/'run'),images=str(images),annotations=str(ann),
        train_split=str(root/'train.csv'),val_split=str(root/'val.csv'),
        native_checkpoint=str(native),roi_checkpoint=str(roi_weight),device='cpu',strict=False,
        native_precision='fp32',roi_precision='fp32',roi_encoder='resnet18',roi_size=64,preprocessing='raw')
    frames=[]
    for index,name in enumerate(names[1:]):
        proposals=[dict(box=list(map(float,boxes[1])),score=.85)] if index==0 else []
        frames.append(dict(file_name=name,image_sha256=pipeline.digest(images/name),
            yolo_reference=[jsonable(encoded(boxes[1]))] if index==0 else [],weak=[],
            boxes={'70':proposals,'80':proposals}))
    seed=dict(schema_version=1,scope='Explicit CPU fixture predictions only',inference_contains_GT=False,
        physical_images=2,frames=frames,metadata={k:pipeline.digest(cfg[k]) for k in pipeline.EXPECTED},
        frozen_recipe_sha256={name:pipeline.digest(Path(pipeline.__file__).with_name(name)) for name in
                             ('submission_native_recipe.py','submission_yolo_recipe.py')},
        source_checkpoints=dict(native_best84=pipeline.digest(native),roi30=pipeline.digest(roi_weight),
            detr={str(epoch):dict(available_checkpoint_sha256='fixture'+str(epoch)) for epoch in (70,80)}))
    seed_path=root/'seed.json.gz';seed_path.write_bytes(gzip.compress(json.dumps(seed).encode(),mtime=0))
    cfg.update(seed_path=str(seed_path),seed_sha256=pipeline.digest(seed_path))
    return cfg,names[1:]


def native_fixture(model,image,boxes,config,device):
    # Prediction takes only image, boxes and inference metadata, never GT paths.
    assert 'annotations' not in config and 'truth' not in config
    assert image.shape==(2048,2048)
    built={}
    for extra in (False,True):
        fg=np.zeros((2048,2048),np.float32)
        if boxes['70']:
            fg[100:164,100:164]=.99
            if extra:fg[300:364,300:364]=.99
        heads=np.stack([fg,fg])
        record=inference.prediction_record(heads,config['file_name'],config['image_sha256'],config['identity'])
        record['tta']=dict(record)
        record['view_unions']={name:inference.rle(fg>=.5) for name,_ in inference.VIEWS}
        built[extra]=record
    rich={arm:built[arm=='detr80_c08'] for arm in pipeline.NATIVE_ARMS}
    return dict(rich=rich,diagnostics={'fixture':True,'actual_tile_forwards':0})


class Best84PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def patches(self, callback=native_fixture):
        return (patch.object(pipeline,'_load_native',return_value=nn.Identity()),
                patch.object(pipeline,'_load_roi',side_effect=lambda *_:PromptMask()),
                patch.object(pipeline.native,'infer_native_variants',side_effect=callback))

    def test_partial_resume_exact_raters_candidate_scope_and_immutable_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg,names=fixture(root)
            output=Path(cfg['output_root']);output.mkdir()
            config_path=output/'experiment_config.json';original=json.dumps(cfg,sort_keys=True,indent=4).encode()
            config_path.write_bytes(original)
            before={str(p):pipeline.digest(p) for p in root.iterdir() if p.is_file()}
            p1,p2,p3=self.patches()
            with p1,p2,p3:
                partial=pipeline.run(cfg,1)
            self.assertEqual(partial['status'],'paused');self.assertFalse(partial['full_cohort_score_available'])
            self.assertEqual(partial['metrics'],{});self.assertIsNone(partial['submission_candidate'])
            cached=pipeline.digest(output/'records/native'/f'{Path(names[0]).stem}.json.gz')
            p1,p2,p3=self.patches()
            with p1 as n,p2 as r,p3 as infer:
                completed=pipeline.run(cfg)
            self.assertEqual(n.call_count,1);self.assertEqual(r.call_count,1);self.assertEqual(infer.call_count,1)
            self.assertEqual(completed['status'],'complete')
            self.assertEqual(completed['completed_images'],dict(native=2,roi=2))
            full=completed['metrics']['full']['score']
            self.assertEqual((full['tp_pairs'],full['fp'],full['fn']),(1,1,2))
            self.assertAlmostEqual(full['pq'],.4)
            self.assertEqual(completed['best_development_native_arm'],'detr80_c08')
            self.assertEqual(completed['metrics']['detr80_c08']['recovered_GT'],1)
            self.assertEqual(completed['metrics']['detr80_c08']['lost_GT'],0)
            self.assertEqual(completed['submission_candidate']['public_LB_reference'],.39)
            self.assertIsNone(completed['submission_candidate']['public_LB_for_candidate'])
            self.assertFalse(completed['submission_candidate']['seed_reusable_on_test'])
            self.assertEqual(completed['submission_candidate']['paired_photo_bootstrap_delta_pq']['physical_photo_clusters'],2)
            self.assertEqual(config_path.read_bytes(),original)
            self.assertEqual(pipeline.digest(output/'records/native'/f'{Path(names[0]).stem}.json.gz'),cached)
            self.assertTrue((output/'resolved_config.json').is_file())
            with patch.object(pipeline,'_load_native',side_effect=AssertionError('complete run reloads model')), \
                 patch.object(pipeline,'_load_roi',side_effect=AssertionError('complete run reloads model')):
                repeated=pipeline.run(cfg,1)
            self.assertEqual(repeated['status'],'complete')
            self.assertEqual(repeated['metrics'],completed['metrics'])
            self.assertEqual(before,{str(p):pipeline.digest(p) for p in root.iterdir() if p.is_file()})
            self.assertFalse(list(output.rglob('*.pt')))
            for name in repeated['archives']:
                self.assertLessEqual(Path(name).stat().st_size,28*1024*1024)
                with zipfile.ZipFile(name) as z:
                    self.assertIsNone(z.testzip());manifest=json.loads(z.read('export_manifest.json'))
                    self.assertFalse(any(x.endswith(('.pt','.jpeg')) for x in z.namelist()))
                    for member in z.namelist():
                        if member=='export_manifest.json':continue
                        content=z.read(member)
                        self.assertEqual(hashlib.sha256(content).hexdigest(),manifest['entries'][member]['sha256'])

    def test_mid_native_interrupt_replays_only_missing_photo_then_roi(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg,names=fixture(root);seen=[]
            def interrupt(model,image,boxes,config,device):
                seen.append(config['file_name'])
                if config['file_name']==names[1]:raise RuntimeError('inference interrupted')
                return native_fixture(model,image,boxes,config,device)
            p1,p2,p3=self.patches(interrupt)
            with p1,p2,p3,self.assertRaisesRegex(RuntimeError,'interrupted'):
                pipeline.run(cfg)
            self.assertEqual(seen,names)
            self.assertFalse((Path(cfg['output_root'])/'records/roi').exists())
            p1,p2,p3=self.patches()
            with p1,p2,p3 as infer:
                report=pipeline.run(cfg)
            self.assertEqual(infer.call_count,1);self.assertEqual(report['status'],'complete')

    def test_corrupt_cache_recomputed_valid_foreign_cache_refused_and_pixels_guarded(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg,names=fixture(root)
            p1,p2,p3=self.patches()
            with p1,p2,p3:pipeline.run(cfg)
            path=Path(cfg['output_root'])/'records/native'/f'{Path(names[0]).stem}.json.gz'
            good=path.read_bytes();path.write_bytes(b'bad gzip')
            p1,p2,p3=self.patches()
            with p1,p2,p3 as infer:pipeline.run(cfg)
            self.assertEqual(infer.call_count,1)
            record=roi.read_record(path);record['identity']='other run';roi.write_record(path,record)
            with self.assertRaisesRegex(RuntimeError,'different image or protocol'):pipeline.run(cfg)
            path.write_bytes(good)
            image=Path(cfg['images'])/names[0];image.write_bytes(image.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'pixels differ'):pipeline.run(cfg)

    def test_seed_weight_runtime_settings_identity_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg,names=fixture(root)
            p1,p2,p3=self.patches()
            with p1,p2,p3:pipeline.run(cfg,1)
            with self.assertRaisesRegex(ValueError,'settings changed'):
                pipeline.run(dict(cfg,roi_size=96),1)
            with self.assertRaisesRegex(ValueError,'batch one'):
                pipeline.run(dict(cfg,native_batch=2),1)
            original=Path(cfg['roi_checkpoint']).read_bytes()
            Path(cfg['roi_checkpoint']).write_bytes(original+b'changed')
            with self.assertRaisesRegex(ValueError,'Weights do not match'):pipeline.run(cfg,1)
            Path(cfg['roi_checkpoint']).write_bytes(original)
            seed=Path(cfg['seed_path']);seed.write_bytes(seed.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'seed checksum'):pipeline.run(cfg,1)

    def test_roi_matches_frozen_geometry_and_box_association(self):
        image=np.full((2048,2048),125,np.uint8)
        boxes=[dict(box=[100.,100.,164.,164.],score=.9),dict(box=[1900.,1950.,2048.,2048.],score=.7),
               dict(box=[0.,0.,30.,100.],score=.59)]
        cfg=dict(preprocessing='raw',roi_context=1.5,roi_size=64,roi_precision='fp32')
        result=pipeline.infer_roi_variants(PromptMask(),image,{'70':boxes,'80':boxes},cfg,torch.device('cpu'))
        legacy=roi.infer_rois(PromptMask(),image,boxes,dict(proposal_confidence=.6,roi_context=1.5,roi_size=64,
                             mask_threshold=.5,amp=False),torch.device('cpu'))
        self.assertEqual(len(result['candidates']['70']),1)
        self.assertEqual(len(result['candidates']['80']),len(legacy))
        for new,old in zip(result['candidates']['80'],legacy):
            self.assertEqual(new['rle'],old['rle']);self.assertEqual(new['roi_quality'],old['roi_quality'])
            self.assertEqual(new['box_id'],pipeline.identity(new['box']))

    def test_checkpoint_change_during_model_loading_is_not_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg,_=fixture(root)
            def changed(config,device):
                path=Path(config['native_checkpoint']);path.write_bytes(path.read_bytes()+b'changed')
                return nn.Identity()
            with patch.object(pipeline,'_load_native',side_effect=changed), \
                 self.assertRaisesRegex(ValueError,'Checkpoint changed during native loading'):
                pipeline.run(cfg,1)
            self.assertFalse(list((Path(cfg['output_root'])/'records').rglob('*.gz')))

    def test_photo_bootstrap_keeps_alternative_raters_together(self):
        # Both annotation groups belong to ONE photo. Resampling the photo
        # cannot manufacture a different balance of its two annotations.
        first=dict(score={'pq':.4},per_group=[
            dict(image='photo',tp_pairs=1,fp=0,fn=0,matched_iou_sum=.8),
            dict(image='photo',tp_pairs=0,fp=1,fn=1,matched_iou_sum=0.)])
        second=dict(score={'pq':.8},per_group=[
            dict(image='photo',tp_pairs=1,fp=0,fn=0,matched_iou_sum=.9),
            dict(image='photo',tp_pairs=1,fp=0,fn=0,matched_iou_sum=.7)])
        result=pipeline.paired_photo_bootstrap(first,second,['photo.jpeg'],samples=100)
        # Computed resampled scores are .4 and .8, irrespective of bootstrap.
        np.testing.assert_allclose(result['ci95'],[.4,.4])


if __name__=='__main__':unittest.main()
