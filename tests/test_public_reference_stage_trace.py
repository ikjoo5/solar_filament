"""CPU stage tracing must reproduce the frozen baseline and reject changed code."""
import copy
import hashlib
from pathlib import Path
import sys
import unittest

import numpy as np
from pycocotools import mask as mu

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
from public_reference_stage_trace import RECIPE_FILES, STAGE_ORDER, replay_stages
from submission_driver import finalize_frame
from submission_yolo_recipe import GEOMETRIC_VIEWS, PREPROCESSING_VIEWS


def fixture(empty=False):
    mask = np.zeros((2048, 2048), np.uint8)
    if not empty:
        mask[600:620, 600:650] = 1
    encoded = mu.encode(np.asfortranarray(mask))
    rle = dict(size=[2048, 2048], counts=encoded['counts'].decode())
    instances = [] if empty else [dict(confidence=.9, rle=rle['counts'], edge_clipped=False)]
    def views(names):
        return [dict(image='sample', view=name, shape=[2048, 2048],
                     coordinate_space='original', instances=copy.deepcopy(instances)) for name in names]
    components = [] if empty else [dict(rle=rle, area=1000, bbox=[600, 600, 50, 20],
        stats=dict(foreground=dict(mean=.99, max=.99, p90=.99), spine=dict(mean=.8, max=.8, p90=.8)))]
    branch = dict(components={threshold:copy.deepcopy(components) for threshold in ('0.2','0.35','0.5','0.65','0.8')},
                  unions=dict(foreground={threshold:rle for threshold in ('0.2','0.35','0.5','0.65','0.8')}))
    native = dict(branch, file_name='sample.jpeg', size=[2048,2048], tta=copy.deepcopy(branch),
                  view_unions={view:rle for view in ('base','hflip','vflip','hvflip')})
    tiles = []
    for y in (0, 512, 1024):
        for x in (0, 512, 1024):
            for size in (1536, 2048):
                record = views([f'tile_{y}_{x}_{size}'])[0]
                record.update(crop_bbox=[x, y, 1024, 1024], input_size=size)
                tiles.append(record)
    stages = dict(geometric=views(GEOMETRIC_VIEWS), preprocessing=views(PREPROCESSING_VIEWS),
                  tiled=tiles, native=native)
    hashes = {name:hashlib.sha256((TOOLS/name).read_bytes()).hexdigest() for name in RECIPE_FILES}
    return stages, hashes


class StageTraceTests(unittest.TestCase):
    def test_real_frozen_recipe_nonempty_and_blank_parity(self):
        for empty in (False, True):
            with self.subTest(empty=empty):
                stages, hashes = fixture(empty)
                baseline = finalize_frame('sample', stages)
                self.assertEqual(bool(baseline), not empty)
                result = replay_stages('sample', stages, baseline, hashes)
                self.assertEqual(tuple(result), STAGE_ORDER)
                self.assertEqual(result['baseline_final'], baseline)

    def test_reference_to_different_baseline_is_rejected(self):
        stages, hashes = fixture()
        with self.assertRaisesRegex(ValueError, 'cached baseline RLE sequence'):
            replay_stages('sample', stages, [], hashes)

    def test_missing_and_mismatched_recipe_hash_are_rejected(self):
        stages, hashes = fixture(True)
        with self.assertRaisesRegex(ValueError, 'protocol.sources'):
            replay_stages('sample', stages, [])
        changed = dict(hashes, **{'submission_yolo_recipe.py':'0'*64})
        with self.assertRaisesRegex(ValueError, 'source differs'):
            replay_stages('sample', stages, [], changed)

    def test_partial_raw_cache_is_not_treated_as_empty(self):
        stages, hashes = fixture(True)
        stages.pop('native')
        with self.assertRaisesRegex(ValueError, 'all four raw inference stages'):
            replay_stages('sample', stages, [], hashes)


if __name__ == '__main__':
    unittest.main()
