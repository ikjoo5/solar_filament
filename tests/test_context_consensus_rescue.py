"""Consensus admission must preserve every reference and reject single-model hits."""
from pathlib import Path
import sys
import unittest

import numpy as np
from pycocotools import mask as mu

TOOLS=Path(__file__).resolve().parents[1]/'tools'
sys.path.insert(0,str(TOOLS))
import experiment_context_consensus_rescue as experiment
from filament_eval import Filament


def encoded(y,x,h=20,w=20):
    mask=np.zeros((2048,2048),np.uint8,order='F');mask[y:y+h,x:x+w]=1
    value=mu.encode(mask)
    return dict(size=[2048,2048],counts=value['counts'].decode('ascii'))


def record(rle,mean=.95,views=4):
    area=int(mu.area(rle));empty=encoded(0,0,0,0)
    component=dict(rle=rle,area=area,stats=dict(foreground=dict(mean=mean)))
    return dict(tta=dict(components={'0.35':[component],'0.5':[component]}),
        view_unions={name:rle if index<views else empty
                     for index,name in enumerate(('base','hflip','vflip','hvflip'))})


class ConsensusRescueTests(unittest.TestCase):
    def test_bounded_six_predeclared_rules(self):
        rules=experiment.plan();self.assertEqual(len(rules),6)
        self.assertEqual(len({r['name'] for r in rules}),6)
        self.assertTrue(all(r['minimum_area']==300 for r in rules))

    def test_original_territory_is_preserved_and_addition_is_trimmed(self):
        baseline=encoded(40,58)
        original=Filament('image_1','image','image',baseline['counts'],400)
        candidate=encoded(40,40)
        rows,counts=experiment.make_rescue('image',[original],record(candidate),record(candidate),experiment.plan()[0])
        self.assertEqual(rows[0],original);self.assertEqual(len(rows),2)
        self.assertEqual(rows[1].area,360)
        self.assertEqual(experiment._intersection(rows[0].rle(2048,2048),rows[1].rle(2048,2048)),0)
        self.assertEqual(counts['physical_additions'],1)

    def test_one_model_false_positive_or_unstable_peer_is_not_admitted(self):
        old=record(encoded(40,40));new=record(encoded(100,100))
        rows,_=experiment.make_rescue('image',[],old,new,experiment.plan()[0])
        self.assertEqual(rows,[])
        candidate=encoded(40,40)
        for peer in (record(candidate,mean=.7),record(candidate,views=1)):
            rows,_=experiment.make_rescue('image',[],peer,record(candidate),experiment.plan()[0])
            self.assertEqual(rows,[])


if __name__=='__main__':unittest.main()
