"""Critical native-coordinate/ownership/empty-image submission contracts."""
import sys
from pathlib import Path
import unittest

import numpy as np
from pycocotools import mask as mu

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import submission_yolo_recipe as recipe


class SubmissionYoloRecipeTests(unittest.TestCase):
    def test_roi_encoder_preserves_native_position_and_holes(self):
        local=np.ones((17,21),bool);local[3:8,4:11]=False
        encoded=recipe.encode_roi(local,2000,2010)
        native=mu.decode(encoded)
        self.assertEqual(native.shape,(2048,2048))
        np.testing.assert_array_equal(native[2000:2017,2010:2031],local)
        self.assertEqual(int(native.sum()),int(local.sum()))
        self.assertEqual(recipe.native_rle({'rle':encoded['counts'].decode('ascii')}),encoded)

    def test_original_ownership_blocks_new_mask_stealing_existing_pixels(self):
        first=np.zeros((30,40),bool);first[8:20,4:15]=True
        second=np.zeros_like(first);second[8:20,20:31]=True
        a=first.copy();a[8:20,10:25]=True
        b=second.copy();b[8:20,10:25]=True
        region=np.s_[:30,:40]
        result=recipe.resolve_rois([a,b],[first,second],[region,region],first.shape)
        self.assertFalse((result[0]&result[1]).any())
        self.assertTrue(result[0][first].all())
        self.assertTrue(result[1][second].all())
        self.assertFalse(result[0][second].any())
        self.assertFalse(result[1][first].any())

    def test_no_detections_remains_a_valid_empty_image(self):
        def record(view,**extra):
            return dict(image='empty',view=view,shape=[2048,2048],
                        coordinate_space='original',instances=[],**extra)
        old=[record(v) for v in recipe.GEOMETRIC_VIEWS]
        prep=[record(v) for v in recipe.PREPROCESSING_VIEWS]
        tiles=[record(f'tile{s}_y{y}_x{x}',input_size=s,crop_bbox=[x,y,1024,1024])
               for y in (0,512,1024) for x in (0,512,1024) for s in (1536,2048)]
        self.assertEqual(recipe.build_yolo_reference('empty',old,prep,tiles),[])
        tiles[-1]['coordinate_space']='crop'
        with self.assertRaisesRegex(ValueError,'coordinate'):
            recipe.build_yolo_reference('empty',old,prep,tiles)

    def test_rescue_strict_novelty_boundary_keeps_existing_instance(self):
        full=np.ones((20,25),bool)
        proposal=recipe.encode_roi(full,500,500)
        raw=dict(confidence=.3,rle=proposal['counts'].decode('ascii'))
        old={v:dict(instances=[raw] if v=='base' else []) for v in recipe.GEOMETRIC_VIEWS}
        prep={v:dict(instances=[raw] if v=='clahe16' else []) for v in recipe.PREPROCESSING_VIEWS}
        for overlap,expected_count in [(50,1),(49,2)]:
            mask=np.zeros_like(full);mask.ravel()[:overlap]=True
            original=recipe.encode_roi(mask,500,500)
            result=recipe._preprocessing_rescue([original],old,prep)
            self.assertEqual(len(result),expected_count)
            self.assertEqual(result[0],original)
            if len(result)>1:
                self.assertEqual(int(mu.area(mu.merge(result,intersect=True))),0)


if __name__=='__main__':
    unittest.main()
