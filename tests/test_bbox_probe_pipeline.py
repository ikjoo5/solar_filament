"""Coordinate, normalization, selection, and export checks for frozen-ROI probes."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from pycocotools import mask as mu
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import bbox_probe_pipeline as probe
import roi_filament_pipeline as roi


def encoded(box):
    a=np.zeros((2048,2048),np.uint8);x,y,x2,y2=box;a[y:y2,x:x2]=1
    return mu.encode(np.asfortranarray(a))


class ProbeTests(unittest.TestCase):
    def test_tiles_cover_edges_and_native_coordinate_mapping(self):
        windows=probe.tile_windows();self.assertEqual(len(windows),9)
        self.assertEqual(windows[-1],(1024,1024,2048,2048))
        covered=np.zeros((2048,2048),bool)
        for x,y,x2,y2 in windows:covered[y:y2,x:x2]=True
        self.assertTrue(covered.all())
        rows=probe.native_boxes([dict(box=[-10,50,1040,1000],score=.9),dict(box=[3,4,3,8],score=.8)],windows[-1])
        self.assertEqual(rows,[dict(box=[1024.,1074.,2048.,2024.],score=.9)])

    def test_bbox_nms_removes_duplicates_but_preserves_nearby_instances(self):
        rows=[dict(box=[10,10,50,40],score=.9),dict(box=[11,10,51,40],score=.8),dict(box=[45,10,85,40],score=.7)]
        self.assertEqual(probe.suppress_boxes(rows,.5,200),[rows[0],rows[2]])
        self.assertEqual(probe.suppress_boxes(rows,.5,1),[rows[0]])

    def test_detr_tile_normalization_and_native_shape_are_exact(self):
        class Fake(nn.Module):
            def __init__(self):super().__init__();self.anchor=nn.Parameter(torch.zeros(()));self.seen=None
            def forward(self,pixel_values):
                self.seen=pixel_values.detach().clone()
                return types.SimpleNamespace(logits=torch.tensor([[[8.,-8.]]]),pred_boxes=torch.tensor([[[.5,.25,.5,.25]]]))
        model=Fake();image=np.full((1024,1024),128,np.uint8)
        with patch.object(probe,'prepare_image',side_effect=AssertionError('Radial preprocessing must happen before tile inference')):
            rows=probe.infer_detr_preprocessed(model,image,dict(detr_image_size=64,detector_precision='fp32',detr_confidence=.05,detr_max_det=100))
        self.assertEqual(tuple(model.seen.shape),(1,3,64,64))
        expected=(128/255.-.485)/.229
        self.assertAlmostEqual(float(model.seen[0,0].mean()),expected,places=6)
        self.assertEqual(rows[0]['box'],[256.,128.,768.,384.])
        self.assertEqual(probe.native_boxes(rows,(1024,1024,2048,2048))[0]['box'],[1280.,1152.,1792.,1408.])
        with self.assertRaisesRegex(RuntimeError,'explicitly select fp32'):
            probe.infer_detr_preprocessed(model,image,dict(detr_image_size=64,detector_precision='bf16'))

    def test_sparse_mask_selector_exactly_matches_frozen_nms_ownership(self):
        masks=[encoded(b) for b in [(10,10,50,50),(11,10,51,50),(40,10,90,50),(500,600,520,620)]]
        rows=[dict(rle=m,box_score=s,roi_quality=q,selection_score=s*q) for m,s,q in zip(masks,[.9,.8,.4,.1],[.9,.99,.8,.7])]
        for nms in (.3,.5,.7):
            fast=probe.MaskSelector(rows,nms)
            for confidence,area,quality in [(.05,150,.6),(.3,300,.6),(.05,500,.9)]:
                expected=roi.mask_predictions(rows,confidence,area,nms,quality);actual=fast(confidence,area,quality)
                self.assertEqual(len(actual),len(expected))
                for a,b in zip(actual,expected):
                    self.assertEqual(int(mu.area(a)),int(mu.area(b)))
                    self.assertEqual(int(mu.area(a)),int(mu.area(mu.merge([a,b],intersect=True))))
                if actual:self.assertEqual(int(mu.area(mu.merge(actual))),sum(map(int,mu.area(actual))))

    def test_bbox_availability_is_distinct_from_one_to_one_recall(self):
        truth={'r1':[encoded([100,100,120,120]),encoded([102,102,122,122])],'r2':[encoded([100,100,120,120])]}
        mapping={'r1':'same_photo','r2':'same_photo'}
        props={'same_photo':[dict(box=[100,100,120,120],score=.9)]}
        value=probe.bbox_metrics(truth,mapping,props,[.05])['thresholds']['0.05']
        row=value['by_iou']['0.5']
        self.assertEqual(row['GT'],3);self.assertEqual(row['available_GT'],3)
        self.assertEqual(row['tp'],2);self.assertEqual(row['fn'],1);self.assertEqual(row['fp'],0)
        self.assertEqual(value['physical_proposals'],1)

    def test_different_output_root_and_bad_tile_settings_rejected(self):
        with self.assertRaisesRegex(ValueError,'disjoint'):
            probe.normalize_config(dict(base_root='/tmp/base',output_root='/tmp/base/child',cache_root='/tmp/cache'))
        with self.assertRaisesRegex(ValueError,'without gaps'):
            probe.normalize_config(dict(base_root='/tmp/base',output_root='/tmp/new',cache_root='/tmp/cache',tile_stride=1100))

    def test_export_status_and_checksums_with_csv_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'evaluation').mkdir();(root/'roi_box_cache').mkdir()
            (root/'result.json').write_text(json.dumps(dict(status='complete',phase='probes')))
            (root/'evaluation/a.csv').write_text('filament_id,segmentation_rle\n')
            (root/'roi_box_cache/duplicate.json').write_text('{}')
            (root/'weights.pt').write_bytes(b'not for export')
            paths=probe.export_results(dict(output_root=str(root)),'probes')
            self.assertEqual(len(paths),1)
            with zipfile.ZipFile(paths[0]) as archive:
                self.assertIsNone(archive.testzip());self.assertEqual(json.loads(archive.read('result.json'))['status'],'complete')
                self.assertNotIn('weights.pt',archive.namelist());self.assertNotIn('roi_box_cache/duplicate.json',archive.namelist())
                manifest=json.loads(archive.read('export_manifest.json'))
                self.assertIn('evaluation/a.csv',manifest['entries'])
                self.assertEqual(manifest['parts_total'],1)

    def test_large_csvs_are_partitioned_and_parts_share_complete_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'evaluation').mkdir()
            (root/'result.json').write_text('{"status":"complete"}')
            for name in ('one','two'):(root/'evaluation'/(name+'.csv')).write_bytes(b'x'*(14*1024*1024))
            paths=probe.export_results(dict(output_root=str(root)),'probes');self.assertEqual(len(paths),2)
            found=set();identities=set()
            for path in paths:
                with zipfile.ZipFile(path) as archive:
                    manifest=json.loads(archive.read('export_manifest.json'));identities.add(manifest['export_id'])
                    self.assertEqual(manifest['parts_total'],2)
                    self.assertEqual(len([n for n in archive.namelist() if n.endswith('.csv')]),1)
                    found.update(n for n in archive.namelist() if n.endswith('.csv'))
                    self.assertIn('result.json',archive.namelist())
            self.assertEqual(len(identities),1);self.assertEqual(found,{'evaluation/one.csv','evaluation/two.csv'})


if __name__=='__main__':unittest.main()
