"""Fixed-rule geometry and prediction-before-label ordering on synthetic fixtures."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from pycocotools import mask as mu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
sys.path.insert(0,str(ROOT/'tests'))
import experiment_grouped_oof_postprocess as experiment
from test_grouped_oof_evaluation import fixture,refresh_record_checksum


def component(x,y,width,height,mean=.9):
    mask=np.zeros((2048,2048),np.uint8)
    mask[y:y+height,x:x+width]=1
    rle=mu.encode(np.asfortranarray(mask))
    return dict(area=width*height,fg_mean=mean,
                rle=dict(size=[2048,2048],counts=rle['counts'].decode()))


class OOFPostprocessTests(unittest.TestCase):
    def test_small_object_admission_requires_high_mean_and_preserves_baseline(self):
        large=component(20,20,20,20,.6)
        small=component(100,100,10,20,.8)
        weak=component(200,200,10,20,.79)
        tiny=component(300,300,9,10,.99)
        record=dict(components={'0.5':[large,small,weak,tiny],'0.35':[],'0.65':[]})
        masks=experiment.preset_masks(record)
        self.assertEqual(masks['baseline'],[large['rle']])
        self.assertEqual(masks['small_object_recall'],[large['rle'],small['rle']])
        self.assertEqual(masks['precision'],[])

    def test_hysteresis_requires_one_hundred_seed_pixels_and_keeps_instances_disjoint(self):
        supported=component(20,20,20,20)
        unsupported=component(100,100,20,20)
        too_small=component(200,200,10,20)
        seed=component(20,20,10,10)
        record=dict(components={'0.5':[],'0.35':[supported,unsupported,too_small],'0.65':[seed]})
        masks=experiment.preset_masks(record)
        self.assertEqual(masks['hysteresis'],[supported['rle']])
        self.assertEqual(masks['precision'],[])
        rles=[experiment.checked_rle(r) for r in masks['hysteresis']]
        self.assertEqual(int(mu.area(mu.merge(rles))),sum(int(mu.area(r)) for r in rles))

    def test_partial_pipeline_writes_every_candidate_before_gt_and_keeps_baseline_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);annotations,plan=fixture(root)
            for fold in range(5):
                if fold:
                    (root/f'fold{fold}/oof_summary.json').unlink()
                    continue
                path=root/'fold0/oof_records.jsonl'
                records=[json.loads(line) for line in path.read_text().splitlines()]
                for record in records:
                    record['components']['0.35']=record['components']['0.5']
                    record['components']['0.65']=record['components']['0.5']
                path.write_text(''.join(json.dumps(r)+'\n' for r in records))
                refresh_record_checksum(root/'fold0')
            output=root/'experiments'
            original=Path.read_text
            def guarded_read(path,*args,**kwargs):
                if path==annotations:
                    self.assertTrue(all((output/f'fold0/{name}.csv').is_file() for name in experiment.PRESETS))
                return original(path,*args,**kwargs)
            with patch.object(Path,'read_text',guarded_read):
                report=experiment.run(annotations,plan,root,output,output/'report.json')
            self.assertEqual(report['status'],'partial_folds')
            self.assertEqual(report['completed_folds'],[0])
            self.assertFalse(report['complete_oof'])
            self.assertIsNone(report['results']['baseline']['pooled_oof'])
            self.assertFalse(report['selection_performed'])
            self.assertEqual((output/'fold0/baseline.csv').read_bytes(),(root/'fold0/oof_predictions.csv').read_bytes())
            self.assertGreater(report['deltas_from_baseline']['small_object_recall']['fp'],0)


if __name__=='__main__':
    unittest.main()
