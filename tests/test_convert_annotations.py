"""Instance-preserving polygon conversion and malformed metadata checks."""

import copy
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from pycocotools import mask as mask_util


SOURCE = Path(__file__).resolve().parents[1] / "tools" / "convert_annotations.py"
SPEC = importlib.util.spec_from_file_location("convert_annotations", SOURCE)
conversion = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = conversion
SPEC.loader.exec_module(conversion)


class AnnotationConversionTests(unittest.TestCase):
    def fixture(self):
        return {"info": {"version": "test"}, "licenses": {},
                "categories": [{"id": i, "name": name} for i, name in enumerate(
                    ["Left", "Right", "Unidentifiable", "Ambiguous"], start=1)],
                "images": [{"id": "a-img", "file_name": "img.jpeg", "width": 2048,
                            "height": 2048, "date_captured": "2011-01-01 00:00:00"}],
                "annotations": [{"id": "annotation-uuid", "image_id": "a-img", "category_id": 1,
                                  "iscrowd": 0, "area": 4,
                                  "segmentation": [[0, 0, 2, 0, 2, 2, 0, 2]],
                                  "bbox": [0, 0, 2, 2], "spine": [0, 0, 1, 1]}]}

    def test_single_instance_csv_and_mapping(self):
        rows, manifest, summary = conversion.convert_data(self.fixture())
        self.assertEqual(rows[0][0], "a-img_1")
        self.assertEqual(manifest["annotations"][0]["annotation_id"], "annotation-uuid")
        mask = mask_util.decode({"size": [2048, 2048], "counts": rows[0][1].encode("ascii")})
        self.assertEqual(int(mask.sum()), 4)
        self.assertEqual(summary["annotations"], 1)
        self.assertEqual(summary["excluded_annotations"], 0)

    def test_multiple_polygons_stay_one_instance(self):
        data = self.fixture()
        data["annotations"][0]["segmentation"].append([10, 10, 12, 10, 12, 12, 10, 12])
        data["annotations"][0]["bbox"] = [0, 0, 12, 12]
        data["annotations"][0]["area"] = 8
        rows, manifest, summary = conversion.convert_data(data)
        self.assertEqual(len(rows), 1)
        self.assertEqual(manifest["annotations"][0]["rasterized_area"], 8)
        self.assertEqual(summary["multipart_annotations"], 1)
        mask = mask_util.decode({"size": [2048, 2048], "counts": rows[0][1].encode("ascii")})
        self.assertTrue(mask[0, 0])
        self.assertTrue(mask[10, 10])
        self.assertFalse(mask[5, 5])

    def test_all_categories_including_ambiguous_are_retained(self):
        data = self.fixture()
        source = data["annotations"][0]
        data["annotations"] = [dict(copy.deepcopy(source), id=f"uuid-{i}", category_id=i)
                               for i in range(1, 5)]
        rows, _, summary = conversion.convert_data(data)
        self.assertEqual(len(rows), 4)
        self.assertEqual([c["annotations"] for c in summary["categories"]], [1, 1, 1, 1])

    def test_duplicate_and_orphan_ids_rejected(self):
        for change in ("image_duplicate", "annotation_duplicate", "orphan", "unknown_category"):
            data = self.fixture()
            if change == "image_duplicate":
                data["images"].append(copy.deepcopy(data["images"][0]))
            elif change == "annotation_duplicate":
                data["annotations"].append(copy.deepcopy(data["annotations"][0]))
            elif change == "orphan":
                data["annotations"][0]["image_id"] = "a-missing"
            else:
                data["annotations"][0]["category_id"] = 99
            with self.subTest(change=change), self.assertRaises(conversion.AnnotationError):
                conversion.convert_data(data)

    def test_image_dimensions_and_filename_must_match(self):
        for field, value in (("width", 1024), ("height", 2047), ("id", "bad_img"),
                             ("file_name", "other.jpeg"), ("file_name", "../img.jpeg")):
            data = self.fixture()
            data["images"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(conversion.AnnotationError):
                conversion.convert_data(data)

    def test_invalid_and_empty_polygons_are_errors(self):
        for polygon in ([], [[]], [[0, 0, 1]], [[0, 0, 1, 1]],
                        [[0, 0, float("nan"), 1, 2, 2]], [[0, 0, 2050, 0, 2, 2]],
                        [[0, 0, 0, 0, 0, 0]]):
            data = self.fixture()
            data["annotations"][0]["segmentation"] = polygon
            with self.subTest(polygon=polygon), self.assertRaises(conversion.AnnotationError):
                conversion.convert_data(data)

    def test_empty_image_group_is_explicitly_reported(self):
        data = self.fixture()
        data["images"].append(dict(data["images"][0], id="b-img"))
        rows, manifest, summary = conversion.convert_data(data)
        self.assertEqual(len(rows), 1)
        self.assertEqual(summary["empty_annotator_image_groups"], ["b-img"])
        self.assertEqual(manifest["images"][1]["instances"], 0)
        self.assertTrue(summary["warnings"])

    def test_source_area_disagreement_is_reported_not_modified(self):
        data = self.fixture()
        data["annotations"][0]["area"] = 999
        _, manifest, summary = conversion.convert_data(data)
        self.assertEqual(summary["source_area_mismatch_count"], 1)
        self.assertEqual(manifest["annotations"][0]["rasterized_area"], 4)
        self.assertEqual(manifest["annotations"][0]["source_area"], 999)

    def test_file_outputs_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "annotations.json"
            target = Path(folder) / "converted"
            source.write_text(json.dumps(self.fixture()), encoding="utf-8")
            summary = conversion.convert_file(source, target)
            self.assertEqual(summary["status"], "complete")
            with (target / "gt.csv").open(newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, ["filament_id", "segmentation_rle"])
                self.assertEqual(len(list(reader)), 1)
            self.assertTrue((target / "manifest.json").exists())
            self.assertTrue((target / "summary.json").exists())
            original_csv = (target / "gt.csv").read_bytes()
            with self.assertRaisesRegex(conversion.AnnotationError, "refusing to overwrite"):
                conversion.convert_file(source, target)
            self.assertEqual((target / "gt.csv").read_bytes(), original_csv)

    def test_invalid_input_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "annotations.json"
            target = Path(folder) / "converted"
            data = self.fixture()
            data["annotations"][0]["segmentation"] = []
            source.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(conversion.AnnotationError):
                conversion.convert_file(source, target)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
