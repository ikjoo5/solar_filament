"""CPU wiring fixture: exact image bytes and no reserved-confirmation target use."""
from datetime import date,timedelta
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import cv2
import numpy as np
from pycocotools import mask as mu
import torch

TOOLS=Path(__file__).resolve().parents[1]/'tools'
if str(TOOLS) not in sys.path:sys.path.insert(0,str(TOOLS))
import external_eval_runner as runner
from external_evaluation_data import assign_external_splits


class FakeNative:
    closed=False
    def __init__(self,*args,**kwargs):FakeNative.closed=False
    def close(self):FakeNative.closed=True


class ExternalRunnerTests(unittest.TestCase):
    def test_exact_byte_aliasing_dev_only_export_and_no_confirm_rasterization(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('train','test','external'):(root/name).mkdir()
            for name,value in [('train/20110101000000Ch.png',10),('test/20120101000000Bh.png',20)]:
                cv2.imwrite(str(root/name),np.full((2048,2048),value,np.uint8))
            candidates=[dict(capture_day=(date(2025,1,1)+timedelta(days=40*i)).isoformat(),pixel_sha256=str(i)) for i in range(20)]
            assigned=assign_external_splits(candidates)
            dev=[r for r in assigned if r['split']=='external_dev'][:2]
            confirm=[r for r in assigned if r['split']=='external_confirm'][:1]
            self.assertEqual((len(dev),len(confirm)),(2,1))
            images=[];annotations=[];source_hashes={}
            for i,row in enumerate(dev+confirm):
                name=row['capture_day'].replace('-','')+'000000Ch.png'
                path=root/'external'/name
                cv2.imwrite(str(path),np.full((2048,2048),30+i,np.uint8));source_hashes[path.stem.lower()]=runner.digest(path)
                images.append(dict(id=str(i),file_name=name,date_captured=row['capture_day'],width=2048,height=2048))
                annotations.append(dict(image_id=str(i),segmentation=[[10,10,30,10,30,30,10,30]] if i<2 else 'RESERVED_CONFIRM_MUST_NOT_BE_RASTERIZED'))
            ann=root/'annotations.json';ann.write_text(json.dumps(dict(images=images,annotations=annotations)))
            config=dict(external_annotations=str(ann),external_images=str(root/'external'),official_train_images=str(root/'train'),
                official_test_images=str(root/'test'),output=str(root/'results'),yolo_checkpoint=str(root/'yolo.pt'),native_checkpoint=str(root/'native.pt'))
            (root/'yolo.pt').write_bytes(b'fixture-yolo');(root/'native.pt').write_bytes(b'fixture-native')
            pins={name:runner.digest(config[name+'_checkpoint']) for name in ('yolo','native')}
            calls=[]
            def fake_process(path,image_sha,*args,**kwargs):
                self.assertEqual(image_sha,runner.digest(path));self.assertEqual(image_sha,source_hashes[path.stem])
                pixels=cv2.imread(str(path),0);self.assertEqual(pixels.shape,(2048,2048))
                mask=np.zeros((2048,2048),np.uint8);mask[10:30,10:30]=1
                rle=mu.encode(np.asfortranarray(mask));calls.append(path.stem)
                return dict(file_name=path.name,rles=[rle],instances=1,capped_yolo_views=0),False
            with patch.object(torch.cuda,'is_available',return_value=True),patch.object(torch.cuda,'get_device_name',return_value='CPU fixture'),\
                patch.object(torch.cuda,'is_bf16_supported',return_value=False),patch.object(runner,'CHECKPOINT_SHA256',pins),\
                patch.object(runner,'NativeProcessClient',FakeNative),patch.object(runner,'InferenceProvider'),patch.object(runner,'process_frame',side_effect=fake_process):
                result=runner.run_external(config)
            self.assertEqual(len(calls),2);self.assertEqual(result['split'],'external_dev');self.assertFalse(result['training_performed'])
            self.assertEqual(result['pq'],1.);self.assertTrue(FakeNative.closed)
            with zipfile.ZipFile(result['archive']) as z:
                self.assertNotIn('external_confirm_ground_truth.csv',z.namelist())
                metric=json.loads(z.read('metrics.json'));self.assertEqual(metric['physical_images'],2)
                truth=z.read('external_dev_ground_truth.csv').decode();self.assertNotIn('RESERVED_CONFIRM',truth)
            self.assertEqual(sorted(source_hashes.values()),sorted(runner.digest(p) for p in (root/'external').iterdir()))

    def test_bootstrap_clusters_raters_and_physical_images_by_capture_block(self):
        groups=[dict(gt_instances=1,time_block_27days=block,matched_iou_sum=.7,tp_pairs=1,fp=0,fn=0)
            for block in ('a','a','a','b')]
        result=runner.external_bootstrap(dict(per_group=groups),draws=100)
        self.assertEqual(result['resampled_temporal_blocks'],2)
        np.testing.assert_allclose(result['percentile95'],[.7,.7])
        self.assertIsNone(runner.external_bootstrap(dict(per_group=groups[:1]))['percentile95'])


if __name__=='__main__':unittest.main()
