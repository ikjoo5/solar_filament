"""Real DenseCRF calls plus native-coordinate and missing-pixel invariants."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
from pycocotools import mask as mu

TOOLS=Path(__file__).resolve().parents[1]/'tools'
if str(TOOLS) not in sys.path:sys.path.insert(0,str(TOOLS))
from experiment_paper_densecrf import densecrf_refine,geometry_guard,plan
from experiment_view_combinations import encode_roi


class PaperDenseCRFTests(unittest.TestCase):
    def example(self):
        original=np.zeros((48,64),bool);original[10:38,16:48]=True;original[21:28,28:36]=False
        gray=np.full(original.shape,170,np.uint8);gray[original]=70
        return gray,original

    def test_missing_pixel_roi_never_runs_zero_imputed_appearance(self):
        gray,original=self.example();coverage=np.ones_like(original);coverage[0,0]=False
        mask,status=densecrf_refine(gray,original,coverage=coverage)
        np.testing.assert_array_equal(mask,original);self.assertEqual(status,'fallback_missing_pixels')

    @unittest.skipUnless(importlib.util.find_spec('pydensecrf'),'actual C++ DenseCRF dependency missing')
    def test_actual_solver_preserves_core_hole_and_other_instance_territory(self):
        gray,original=self.example();other=np.zeros_like(original);other[:,50:54]=True
        mask,status=densecrf_refine(gray,original,other,coverage=np.ones_like(original))
        self.assertTrue(geometry_guard(mask,original));self.assertFalse((mask&other).any())
        self.assertTrue(mask[17,23]);self.assertFalse(mask[24,32]);self.assertIn(status,('changed','unchanged','fallback_geometry_guard'))
        encoded=encode_roi(mask,800,900)
        decoded=mu.decode(encoded)
        np.testing.assert_array_equal(decoded[800:848,900:964],mask)
        self.assertEqual(int(decoded.sum()),int(mask.sum()))

    def test_no_implicit_coverage_and_four_fixed_rules(self):
        gray,original=self.example()
        with self.assertRaises(ValueError):densecrf_refine(gray,original)
        with self.assertRaises(ValueError):densecrf_refine(gray.astype(float),original,coverage=np.ones_like(original))
        self.assertEqual(len(plan()),4);self.assertEqual(len({r['name'] for r in plan()}),4)


if __name__=='__main__':unittest.main()
