"""The actual Kaggle validator forbids even tiny same-image intersections."""
from pathlib import Path
import sys
import unittest

import numpy as np
from pycocotools import mask as mu

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from filament_eval import Filament
from build_public_addition_submission import require_disjoint


def instance(name, rect):
    pixels=np.zeros((2048,2048),np.uint8)
    top,left,height,width=rect
    pixels[top:top+height,left:left+width]=1
    mask=mu.encode(np.asfortranarray(pixels))
    stem=name.split('_')[0]
    return Filament(name,stem,stem,mask['counts'].decode(),int(mu.area(mask)))


class SubmissionDisjointnessTests(unittest.TestCase):
    def test_nonintersecting_masks_and_separate_images_pass(self):
        require_disjoint([instance('a_1',(0,0,20,20)),instance('a_2',(20,0,20,20)),
                          instance('b_1',(0,0,20,20))])

    def test_single_pixel_overlap_rejected_despite_tiny_iou(self):
        with self.assertRaisesRegex(ValueError,'Overlapping submission masks'):
            require_disjoint([instance('a_1',(0,0,100,100)),instance('a_2',(99,99,1,1))])

    def test_duplicate_geometry_with_distinct_ids_rejected(self):
        with self.assertRaisesRegex(ValueError,'Overlapping submission masks'):
            require_disjoint([instance('a_1',(5,5,5,5)),instance('a_2',(5,5,5,5))])


if __name__=='__main__':unittest.main()
