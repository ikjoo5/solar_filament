"""Portable submission behavior for blank frames, native placement and guards."""
import sys
from pathlib import Path
import unittest

import numpy as np
from pycocotools import mask as mu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from submission_native_recipe import build_native_submission


def encoded(mask):
    value = mu.encode(np.asfortranarray(mask.astype(np.uint8)))
    return dict(size=list(value['size']),counts=value['counts'].decode('ascii'))


def component(mask, confidence=.99):
    return dict(rle=encoded(mask),area=int(mask.sum()),
                stats=dict(foreground=dict(mean=confidence)))


def rich(views=None, support=None, base_components=None, tta_components=None, seed_components=None):
    blank = np.zeros((2048,2048),bool)
    if views is None:
        views = blank
    if support is None:
        support = blank
    return dict(file_name='sample.jpeg',size=[2048,2048],
        components={'0.5':seed_components or [],'0.65':base_components or []},
        tta=dict(components={'0.35':[],'0.65':tta_components or []},
                 unions=dict(foreground={'0.2':encoded(support)})),
        view_unions={name:encoded(views) for name in ('base','hflip','vflip','hvflip')})


class SubmissionNativeRecipeTests(unittest.TestCase):
    def test_blank_test_frame_has_no_fake_instance(self):
        self.assertEqual(build_native_submission('sample',[],[],[],[],rich()),[])

    def test_refinement_preserves_native_hole_and_moves_only_accepted_boundary(self):
        original = np.zeros((2048,2048),bool)
        original[100:120,150:170] = True
        original[108:111,158:161] = False
        peer = original.copy()
        peer[100:120,149] = True
        c = component(peer)
        result = build_native_submission('sample',[encoded(original)],[],[],[],
            rich(views=peer,support=peer,base_components=[c],tta_components=[c]),True)
        self.assertEqual(len(result['final']),1)
        np.testing.assert_array_equal(mu.decode(result['final'][0]),peer)
        self.assertFalse(mu.decode(result['final'][0])[109,159])
        self.assertEqual(result['decisions']['boundary_changed'],1)

    def test_clipped_tile_cannot_corroborate_native_rescue(self):
        candidate = np.zeros((2048,2048),bool)
        candidate[100:120,150:170] = True
        c = component(candidate)
        data = rich(views=candidate,support=candidate,seed_components=[c])
        raw = dict(image='sample',shape=[2048,2048],coordinate_space='original',
            instances=[dict(confidence=.6,rle=c['rle']['counts'],edge_clipped=True)])
        rejected = build_native_submission('sample',[],[],[],[raw],data,True)
        self.assertEqual(rejected['native_reference'],[])
        raw['instances'][0]['edge_clipped'] = False
        accepted = build_native_submission('sample',[],[],[],[raw],data,True)
        self.assertEqual(len(accepted['native_reference']),1)
        self.assertEqual(len(accepted['final']),1)

    def test_pruning_requires_both_supports_to_be_low(self):
        original = np.zeros((2048,2048),bool)
        original[100:120,150:170] = True
        unsupported = build_native_submission('sample',[encoded(original)],[],[],[],rich(),True)
        self.assertEqual(unsupported['decisions']['final_removed'],1)
        self.assertEqual(unsupported['final'],[])
        support = np.zeros_like(original)
        support[100:110,150:170] = True  # 50% TTA support, zero view agreement.
        retained = build_native_submission('sample',[encoded(original)],[],[],[],
            rich(support=support),True)
        self.assertEqual(retained['decisions']['final_removed'],0)
        np.testing.assert_array_equal(mu.decode(retained['final'][0]),original)


if __name__ == '__main__':
    unittest.main()
