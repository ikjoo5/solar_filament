"""Contract checks for the supplied evaluation notebook and compressed RLE I/O."""

import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from pycocotools import mask as mask_util


SOURCE = Path(__file__).resolve().parents[1] / "tools" / "filament_eval.py"
SPEC = importlib.util.spec_from_file_location("filament_eval", SOURCE)
evaluation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluation
SPEC.loader.exec_module(evaluation)


class EvaluationContractTests(unittest.TestCase):
    height, width = 4, 5

    def mask(self, pixels):
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        for y, x in pixels:
            mask[y, x] = 1
        return mask

    def instance(self, filament_id, pixels, gt=False):
        mask = self.mask(pixels)
        counts = mask_util.encode(np.asfortranarray(mask))["counts"].decode("ascii")
        image_id, group_id = evaluation.parse_filament_id(filament_id, gt)
        return evaluation.Filament(filament_id, image_id, group_id, counts, int(mask.sum()))

    def score(self, gt, pred, **kwargs):
        return evaluation.evaluate(gt, pred, self.height, self.width, **kwargs)

    def write_csv(self, path, rows):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["filament_id", "segmentation_rle"])
            writer.writerows([(r.filament_id, r.counts) for r in rows])

    def test_perfect_match(self):
        gt = [self.instance("a-img_1", [(0, 0), (1, 0)], gt=True)]
        pred = [self.instance("img_1", [(0, 0), (1, 0)])]
        result = self.score(gt, pred)
        self.assertEqual((result["pq"], result["sq"], result["rq"]), (1, 1, 1))

    def test_false_positive_and_false_negative(self):
        gt = [self.instance("a-img_1", [(0, 0)], gt=True),
              self.instance("a-img_2", [(1, 0)], gt=True)]
        pred = [self.instance("img_1", [(0, 0)]), self.instance("img_2", [(3, 4)])]
        result = self.score(gt, pred)
        self.assertEqual((result["tp_pairs"], result["fp"], result["fn"]), (1, 1, 1))
        self.assertEqual(result["pq"], 0.5)

    def test_strict_half_iou_is_not_match(self):
        gt = [self.instance("a-img_1", [(0, 0), (0, 1)], gt=True)]
        pred = [self.instance("img_1", [(0, 0)])]
        result = self.score(gt, pred)
        self.assertEqual((result["tp_pairs"], result["fp"], result["fn"]), (0, 1, 1))
        self.assertEqual(result["pq"], 0)

    def test_global_aggregation_differs_from_macro_average(self):
        gt = [self.instance("a-one_1", [(0, 0)], gt=True)]
        gt += [self.instance(f"a-many_{i}", [(i, 0)], gt=True) for i in range(1, 4)]
        pred = [self.instance("one_1", [(0, 0)])]
        result = self.score(gt, pred)
        self.assertAlmostEqual(result["pq"], 1 / 2.5)
        macro = np.mean([g["pq_diagnostic"] for g in result["per_group"]])
        self.assertEqual(macro, 0.5)
        self.assertNotEqual(result["pq"], macro)

    def test_multiple_annotators_repeat_prediction_evaluation(self):
        gt = [self.instance("a-img_1", [(0, 0)], gt=True),
              self.instance("b-img_1", [(0, 0)], gt=True),
              self.instance("a-other_1", [(1, 1)], gt=True)]
        pred = [self.instance("img_1", [(0, 0)])]
        result = self.score(gt, pred)
        self.assertEqual(result["annotator_image_groups"], 3)
        self.assertEqual(result["tp_pairs"], 2)
        self.assertEqual(result["pq"], 0.8)

    def test_duplicate_masks_with_distinct_ids_use_all_pairs(self):
        gt = [self.instance("a-img_1", [(0, 0)], gt=True)]
        pred = [self.instance("img_1", [(0, 0)]), self.instance("img_2", [(0, 0)])]
        result = self.score(gt, pred)
        self.assertEqual((result["tp_pairs"], result["fp"], result["fn"]), (2, 0, 0))
        self.assertEqual(result["pq"], 1)
        audit = evaluation.audit(pred, self.height, self.width)
        self.assertEqual(audit["groups_with_overlap"], 1)
        self.assertEqual(audit["overlap_pixels_sum_across_groups"], 1)

    def test_prediction_outside_gt_requires_explicit_filter(self):
        gt = [self.instance("a-img_1", [(0, 0)], gt=True)]
        pred = [self.instance("unknown_1", [(1, 1)])]
        with self.assertRaisesRegex(evaluation.ValidationError, "silently ignore"):
            self.score(gt, pred)
        result = self.score(gt, pred, filter_pred_to_gt=True)
        self.assertEqual(result["filtered_prediction_rows"], 1)
        self.assertEqual(result["pq"], 0)

    def test_fortran_rle_preserves_asymmetric_pixel_coordinates(self):
        original = self.mask([(0, 4), (1, 0), (3, 2)])
        counts = mask_util.encode(np.asfortranarray(original))["counts"].decode("ascii")
        decoded, area = evaluation.decode_rle(counts, self.height, self.width)
        np.testing.assert_array_equal(decoded, original)
        self.assertEqual(area, 3)

    def test_random_rle_counts_validation(self):
        rng = np.random.default_rng(123)
        for _ in range(30):
            original = (rng.random((self.height, self.width)) > 0.6).astype(np.uint8)
            original[0, 0] = 1
            counts = mask_util.encode(np.asfortranarray(original))["counts"].decode("ascii")
            decoded, _ = evaluation.decode_rle(counts, self.height, self.width)
            np.testing.assert_array_equal(decoded, original)

    def test_malformed_and_empty_rle_rejected(self):
        for counts in ("", "!", "o", "1", "00"):
            with self.subTest(counts=counts), self.assertRaises(evaluation.ValidationError):
                evaluation.decode_rle(counts, self.height, self.width)
        zero = np.zeros((self.height, self.width), dtype=np.uint8)
        counts = mask_util.encode(np.asfortranarray(zero))["counts"].decode("ascii")
        with self.assertRaisesRegex(evaluation.ValidationError, "empty instance"):
            evaluation.decode_rle(counts, self.height, self.width)

    def test_invalid_ids(self):
        for value in ("", "img", "_1", "img_", "img _1", "img_1\n"):
            with self.subTest(value=value), self.assertRaises(evaluation.ValidationError):
                evaluation.parse_filament_id(value)
        for value in ("img_1", "-img_1", "a-_1"):
            with self.subTest(value=value), self.assertRaises(evaluation.ValidationError):
                evaluation.parse_filament_id(value, ground_truth=True)
        self.assertEqual(evaluation.parse_filament_id("a-img-extra_abc", True),
                         ("img-extra", "a-img-extra"))

    def test_duplicate_ids_rejected_when_reading_csv(self):
        row = self.instance("img_1", [(0, 0)])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "pred.csv"
            self.write_csv(path, [row, row])
            with self.assertRaisesRegex(evaluation.ValidationError, "duplicate filament_id"):
                evaluation.read_csv(path, self.height, self.width)

    def test_cli_lint_reports_no_pq_and_missing_expected_images(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "pred.csv"
            expected = Path(folder) / "images.txt"
            self.write_csv(path, [self.instance("img_1", [(0, 0)])])
            expected.write_text("img\nblank\n", encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = evaluation.main(["--pred", str(path), "--height", "4", "--width", "5",
                                          "--expected-images", str(expected)])
            report = json.loads(output.getvalue())
            self.assertEqual(status, 0)
            self.assertNotIn("evaluation", report)
            self.assertEqual(report["coverage"]["images_without_predictions"], ["blank"])
            self.assertTrue(any("PQ has not" in w for w in report["warnings"]))


if __name__ == "__main__":
    unittest.main()
