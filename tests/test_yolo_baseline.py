"""Pure split, mask, and preprocessing checks; no GPU/YOLO installation needed."""

import csv
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.interpolate import UnivariateSpline


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location("evaluate_yolo_baseline", TOOLS / "evaluate_yolo_baseline.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class BaselineHelperTests(unittest.TestCase):
    def images(self):
        return [{"id": "a-one", "file_name": "one.jpeg"},
                {"id": "b-one", "file_name": "one.jpeg"},
                {"id": "a-two", "file_name": "two.jpeg"}]

    def test_complement_is_explicit_and_keeps_all_annotators(self):
        with self.assertRaisesRegex(runner.RunnerError, "explicitly"):
            runner.select_validation(self.images(), ["a-two"])
        selected, report = runner.select_validation(self.images(), ["a-two"], use_train_complement=True)
        self.assertEqual({row["id"] for row in selected}, {"a-one", "b-one"})
        self.assertEqual(report["selected_physical_images"], 1)
        self.assertIn("not verified", report["provenance"])

    def test_physical_annotator_leakage_is_rejected(self):
        with self.assertRaisesRegex(runner.RunnerError, "physical"):
            runner.select_validation(self.images(), ["a-one"], use_train_complement=True)

    def test_explicit_validation_cannot_omit_annotator(self):
        with self.assertRaisesRegex(runner.RunnerError, "omits annotator"):
            runner.select_validation(self.images(), ["a-two"], ["a-one"])

    def test_empty_or_unknown_validation_is_rejected(self):
        with self.assertRaisesRegex(runner.RunnerError, "empty"):
            runner.select_validation(self.images(), ["a-one", "b-one", "a-two"], use_train_complement=True)
        with self.assertRaisesRegex(runner.RunnerError, "unknown"):
            runner.select_validation(self.images(), ["unknown"], use_train_complement=True)

    def test_split_file_matching_and_duplicates(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "split.csv"
            for rows, message in [([("a-one", "wrong.jpeg")], "does not match"),
                                  ([("a-one", "one.jpeg"), ("a-one", "one.jpeg")], "duplicate")]:
                with path.open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["image_id", "file_name"])
                    writer.writerows(rows)
                with self.assertRaisesRegex(runner.RunnerError, message):
                    runner.read_split(path, self.images())

    def instance(self, coordinates, confidence=.8):
        mask = np.zeros((8, 8), dtype=bool)
        for y, x in coordinates:
            mask[y, x] = True
        return {"mask": mask, "confidence": confidence}

    def test_preserve_keeps_disconnected_identity(self):
        raw = [self.instance([(0, 0), (7, 7)])]
        self.assertEqual(len(runner.postprocess_instances(raw, .5, "preserve", 1)), 1)
        self.assertEqual(len(runner.postprocess_instances(raw, .5, "union_cc", 1)), 2)

    def test_union_merges_touching_instances(self):
        raw = [self.instance([(1, 1)]), self.instance([(1, 2)])]
        self.assertEqual(len(runner.postprocess_instances(raw, .5, "preserve", 1)), 2)
        self.assertEqual(len(runner.postprocess_instances(raw, .5, "union_cc", 1)), 1)

    def test_threshold_area_and_invalid_outputs(self):
        raw = [self.instance([(1, 1)], confidence=.25)]
        self.assertEqual(len(runner.postprocess_instances(raw, .25, "preserve", 1)), 1)
        self.assertEqual(len(runner.postprocess_instances(raw, .3, "union_cc", 1)), 0)
        self.assertEqual(len(runner.postprocess_instances(raw, .2, "preserve", 2)), 0)
        with self.assertRaises(runner.RunnerError):
            runner.postprocess_instances(raw, float("nan"), "preserve", 1)

    def test_preserved_overlap_is_not_silently_rewritten(self):
        raw = [self.instance([(1, 1)]), self.instance([(1, 1)])]
        masks = runner.postprocess_instances(raw, .5, "preserve", 1)
        rows = runner.masks_to_filaments("img", masks)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].counts, rows[1].counts)

    def test_limb_preprocessing_matches_provided_formula(self):
        # Real native geometry and a nonconstant deterministic radial/intensity
        # pattern exercise spline and quantization, not only zero/background.
        yy, xx = np.ogrid[:2048, :2048]
        image = ((xx // 19 + yy // 13) % 170 + 35).astype(np.uint8)
        radius = (np.sqrt((xx - 1024)**2 + (yy - 1024)**2) / 900).astype(np.float32)
        disk = radius <= 1
        rs = radius[disk]
        bins = np.minimum((rs * 100).astype(np.int32), 99)
        radii = (np.arange(100) + .5) / 100
        pixels = image[disk].astype(np.uint8)
        hist = np.bincount(bins * 256 + pixels.astype(np.int32), minlength=100 * 256).reshape(100, 256)
        counts = hist.sum(axis=1)
        medians = np.argmax(hist.cumsum(axis=1) >= ((counts + 1) // 2)[:, None], axis=1).astype(np.float64)
        spline = UnivariateSpline(radii, medians, k=3, s=400)
        ref = float(np.median(spline(radii)[radii <= .15]))
        gain = np.clip(ref / np.maximum(spline(rs), 1.0), .25, 4)
        expected = np.zeros(image.shape, dtype=np.uint8)
        expected[disk] = np.clip(image[disk].astype(np.float32) * gain, 0, 255).astype(np.uint8)
        np.testing.assert_array_equal(runner.preprocess_limb_only(image), expected)
        with self.assertRaises(runner.RunnerError):
            runner.preprocess_limb_only(image[:100, :100])


if __name__ == "__main__":
    unittest.main()
