"""Pixel-refinement safeguards and native tile loading, without model inference."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np
from pycocotools import mask as mu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from refine_masks_with_pixels import TilePixels, assign_nearest, refine_mask, resolve_rois, settings
from experiment_view_combinations import encode_roi


class PixelRefinementTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(2026)
        truth = np.zeros((48, 72), bool)
        truth[12:36, 30:38] = True
        gray = np.clip(rng.normal(150, 2, truth.shape), 0, 255).astype(np.uint8)
        gray[truth] = np.clip(rng.normal(85, 2, truth.sum()), 0, 255)
        coarse = np.zeros_like(truth)
        coarse[11:37, 29:39] = True
        return gray, truth, coarse

    def test_fixed_small_unique_grid(self):
        self.assertEqual(len(settings()), 13)
        self.assertEqual(len({s['name'] for s in settings()}), 13)

    def test_dark_stripe_refines_and_preserves_core(self):
        gray, truth, coarse = self.fixture()
        before = (truth & coarse).sum()/(truth | coarse).sum()
        for method in ('grabcut', 'bayes'):
            refined, status = refine_mask(gray, coarse, method=method, band=5)
            self.assertEqual(status, 'changed')
            self.assertGreater((truth & refined).sum()/(truth | refined).sum(), before)
            self.assertTrue(refined.any())
            self.assertEqual(refined.dtype, bool)

    def test_constant_or_missing_pixels_keep_original(self):
        gray, _, original = self.fixture()
        for method in ('grabcut', 'bayes'):
            for pixels, coverage, reason in [
                (np.full_like(gray, 120), None, 'fallback_low_dark_contrast'),
                (gray, np.zeros_like(gray, bool), 'fallback_missing_pixels')]:
                refined, status = refine_mask(pixels, original, method=method, coverage=coverage)
                np.testing.assert_array_equal(refined, original)
                self.assertEqual(status, reason)

    def test_nearby_original_ownership_and_new_collision(self):
        a = np.zeros((16, 20), bool); a[5:10, 4:7] = True
        b = np.zeros_like(a); b[5:10, 11:14] = True
        ca = cv2.dilate(a.astype(np.uint8), np.ones((7, 9), np.uint8)).astype(bool)
        cb = cv2.dilate(b.astype(np.uint8), np.ones((7, 9), np.uint8)).astype(bool)
        ra, rb = assign_nearest([ca, cb], [a, b])
        self.assertFalse(np.any(ra & rb))
        self.assertTrue(np.all(ra[a]) and np.all(rb[b]))
        self.assertTrue(ra[7, 8] and rb[7, 10])
        with self.assertRaises(ValueError):
            assign_nearest([ca, cb], [a, a])

    def test_roi_ownership_matches_full_and_ties_keep_first(self):
        shape = (32, 40)
        originals = []
        for y, x in [(12, 12), (16, 12), (12, 20)]:
            mask = np.zeros(shape, bool); mask[y, x] = True; originals.append(mask)
        candidates = [cv2.dilate(m.astype(np.uint8), np.ones((9, 13), np.uint8)).astype(bool)
                      for m in originals]
        expected = assign_nearest(candidates, originals)
        regions, local_originals, local_candidates = [], [], []
        for original, candidate in zip(originals, candidates):
            ys, xs = np.nonzero(candidate | original)
            region = np.s_[ys.min():ys.max()+1, xs.min():xs.max()+1]
            regions.append(region); local_originals.append(original[region])
            local_candidates.append(candidate[region])
        local = resolve_rois(local_candidates, local_originals, regions, shape)
        for region, actual, full in zip(regions, local, expected):
            image = np.zeros(shape, bool); image[region] = actual
            np.testing.assert_array_equal(image, full)
        # sqrt(5) from each of the first two seeds: exact tie keeps first.
        self.assertTrue(expected[0][14, 13])
        self.assertFalse(expected[1][14, 13])

    def test_thin_mask_has_explicit_fallback(self):
        gray = np.full((30, 40), 150, np.uint8)
        original = np.zeros_like(gray, bool); original[8:23, 20] = True
        gray[original] = 90
        refined, status = refine_mask(gray, original, core_mode='erode3')
        np.testing.assert_array_equal(refined, original)
        self.assertEqual(status, 'fallback_insufficient_samples')

    def test_background_ring_missing_retains_original(self):
        gray, _, original = self.fixture()
        coverage = np.ones_like(original)
        coverage[8, 33] = False  # Background/padding, all original FG still present.
        refined, status = refine_mask(gray, original, coverage=coverage)
        self.assertEqual(status, 'fallback_missing_pixels')
        np.testing.assert_array_equal(refined, original)

    def test_corner_clipped_roi_native_rle(self):
        image = np.full((2048, 2048), 150, np.uint8)
        original = np.zeros_like(image, bool); original[:20, :8] = True
        image[:19, :7] = 85
        local, _ = refine_mask(image[:30, :18], original[:30, :18], method='bayes', band=7)
        encoded = encode_roi(local, 0, 0)
        expected = np.zeros_like(image); expected[:30, :18] = local
        np.testing.assert_array_equal(mu.decode(encoded), expected)
        self.assertGreater(int(mu.area(encoded)), 0)

    def test_tile_roundtrip_hash_geometry_missing_and_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pixels = np.arange(120, dtype=np.uint8).reshape(10, 12)
            raw = cv2.imencode('.png', pixels)[1].tobytes()
            (root/'tile.png').write_bytes(raw)
            tile = dict(path='tile.png', x=3, y=4, width=12, height=10,
                        sha256=hashlib.sha256(raw).hexdigest())
            manifest = dict(schema_version=1, image_size=[24, 30], images={'x': dict(tiles=[tile])})
            (root/'manifest.json').write_text(json.dumps(manifest))
            image, covered = TilePixels(root).load('x')
            np.testing.assert_array_equal(image[4:14, 3:15], pixels)
            self.assertEqual(int(covered.sum()), 120)
            (root/'tile.png').unlink()
            store = TilePixels(root)
            _, covered = store.load('x')
            self.assertFalse(covered.any()); self.assertEqual(store.missing['x'], 1)
            tile['path'] = '../outside.png'
            (root/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError): TilePixels(root).load('x')

    def test_tile_checksum_and_overlapping_pixel_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tiles = []
            for index, value in enumerate((100, 101)):
                raw = cv2.imencode('.png', np.full((8, 8), value, np.uint8))[1].tobytes()
                name = f'{index}.png'; (root/name).write_bytes(raw)
                tiles.append(dict(path=name, x=0, y=0, width=8, height=8,
                                  sha256=hashlib.sha256(raw).hexdigest()))
            manifest = dict(schema_version=1, image_size=[8, 8], images={'x': dict(tiles=tiles)})
            (root/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'different pixel'):
                TilePixels(root).load('x')
            tiles.pop(); tiles[0]['sha256'] = 'incorrect'
            (root/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                TilePixels(root).load('x')


if __name__ == '__main__':
    unittest.main()
