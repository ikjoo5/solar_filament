"""YOLO support counts independent families and excludes clipped native tiles."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from pycocotools import mask as mu

TOOLS=Path(__file__).resolve().parents[1]/'tools';sys.path.insert(0,str(TOOLS))
import experiment_context_yolo_rescue as experiment


class ContextYOLOTests(unittest.TestCase):
    def rle(self):
        mask=np.zeros((64,64),np.uint8,order='F');mask[10:30,10:30]=1
        return mu.encode(mask)

    def test_multiple_views_of_one_family_do_not_count_as_two(self):
        rle=self.rle()
        raw={'geometric':[dict(rle=rle,confidence=.4),dict(rle=rle,confidence=.6)]}
        support=experiment.family_support(rle,raw)
        self.assertEqual(support,{'geometric':.6})
        self.assertTrue(experiment.admitted(support,experiment.plan()[1]))
        self.assertFalse(experiment.admitted(support,experiment.plan()[2]))

    def test_confidence_and_family_gates_follow_all_four_fixed_rules(self):
        support={'geometric':.35,'preprocessing':.25}
        self.assertEqual([experiment.admitted(support,rule) for rule in experiment.plan()],[True,False,True,False])

    def test_edge_clipped_tile_and_low_confidence_are_excluded(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'raw.jsonl';counts=self.rle()['counts'].decode('ascii')
            row=dict(image='image',shape=[2048,2048],coordinate_space='original',instances=[
                dict(confidence=.9,edge_clipped=True,rle=counts),
                dict(confidence=.4,edge_clipped=False,rle=counts),
                dict(confidence=.2,edge_clipped=False,rle=counts)])
            path.write_text(json.dumps(row)+'\n')
            with patch.object(experiment,'SOURCES',{'native_tiles':path}):
                raw,records=experiment.load_raw()
            self.assertEqual(records,{'native_tiles':1})
            self.assertEqual([p['confidence'] for p in raw['image']['native_tiles']],[.4])


if __name__=='__main__':unittest.main()
