#!/usr/bin/env python3
"""Convert MAGFiLO COCO-like polygons into notebook-compatible GT CSV.

Each source annotation remains one instance, including multipart polygons.
No chirality category is excluded. The original JSON is never modified.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

from pycocotools import mask as mask_util


class AnnotationError(ValueError):
    """The source JSON does not satisfy this dataset's annotation contract."""


def _identifier(value, context: str) -> str:
    if (not isinstance(value, str) or not value
            or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in value)):
        raise AnnotationError(f"{context}: expected a nonempty string without whitespace/control characters")
    return value


def _number(value, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise AnnotationError(f"{context}: expected a finite number")
    return float(value)


def _points(values, context: str, width: int, height: int, minimum: int) -> None:
    if not isinstance(values, list) or len(values) < minimum or len(values) % 2:
        raise AnnotationError(f"{context}: expected an even coordinate list with at least {minimum} values")
    for i, value in enumerate(values):
        number = _number(value, f"{context}[{i}]")
        limit = width if i % 2 == 0 else height
        if not 0 <= number <= limit:
            raise AnnotationError(f"{context}[{i}]: coordinate {number} outside [0,{limit}]")


def convert_data(data: dict, source_metadata: dict | None = None) -> tuple[list[tuple], dict, dict]:
    """Validate and rasterize actual MAGFiLO fields, returning CSV rows/metadata.

    This is not a generic conversion of arbitrary COCO JSON: images must have
    annotator-image IDs, 2048-square dimensions, and matching filename stems.
    """
    if not isinstance(data, dict):
        raise AnnotationError("annotation root must be a JSON object")
    for field in ("images", "annotations", "categories"):
        if not isinstance(data.get(field), list) or not data[field]:
            raise AnnotationError(f"{field} must be a nonempty list")

    categories = {}
    for category in data["categories"]:
        if not isinstance(category, dict):
            raise AnnotationError("category entries must be objects")
        category_id = category.get("id")
        if type(category_id) is not int or category_id <= 0:
            raise AnnotationError("category IDs must be positive integers")
        if category_id in categories:
            raise AnnotationError(f"duplicate category ID {category_id}")
        if not isinstance(category.get("name"), str) or not category["name"].strip():
            raise AnnotationError(f"category {category_id}: name must be nonempty")
        categories[category_id] = category

    images = {}
    files_by_stem = {}
    image_manifest = []
    for image in data["images"]:
        if not isinstance(image, dict):
            raise AnnotationError("image entries must be objects")
        image_id = _identifier(image.get("id"), "image.id")
        if image_id in images:
            raise AnnotationError(f"duplicate image ID {image_id}")
        if "_" in image_id:
            raise AnnotationError(f"image {image_id}: '_' conflicts with notebook ID parsing")
        annotator, separator, stem = image_id.partition("-")
        if not separator or not annotator or not stem:
            raise AnnotationError(f"image {image_id}: expected '<annotator>-<image-stem>'")
        filename = image.get("file_name")
        if (not isinstance(filename, str) or not filename or "/" in filename or "\\" in filename
                or Path(filename).name != filename or not Path(filename).suffix):
            raise AnnotationError(f"image {image_id}: file_name must be a plain filename with extension")
        if Path(filename).stem != stem:
            raise AnnotationError(f"image {image_id}: file_name stem does not match ID image suffix")
        if (type(image.get("height")) is not int or type(image.get("width")) is not int
                or image["height"] != 2048 or image["width"] != 2048):
            raise AnnotationError(f"image {image_id}: expected width=height=2048")
        if stem in files_by_stem and files_by_stem[stem] != filename:
            raise AnnotationError(f"physical image {stem}: inconsistent file_name across annotators")
        files_by_stem[stem] = filename
        images[image_id] = image
        image_manifest.append({"annotator_image_id": image_id, "annotator_id": annotator,
                               "image_id": stem, "file_name": filename,
                               "height": image["height"], "width": image["width"],
                               "date_captured": image.get("date_captured"),
                               "url": image.get("url"), "license": image.get("license")})

    rows = []
    annotation_manifest = []
    seen_annotations = set()
    group_counts = Counter()
    category_counts = Counter({category_id: 0 for category_id in categories})
    area_mismatches = []
    multipolygon_count = 0
    for annotation in data["annotations"]:
        if not isinstance(annotation, dict):
            raise AnnotationError("annotation entries must be objects")
        annotation_id = _identifier(annotation.get("id"), "annotation.id")
        if annotation_id in seen_annotations:
            raise AnnotationError(f"duplicate annotation ID {annotation_id}")
        seen_annotations.add(annotation_id)
        image_id = annotation.get("image_id")
        if not isinstance(image_id, str) or image_id not in images:
            raise AnnotationError(f"annotation {annotation_id}: orphan/invalid image_id {image_id!r}")
        image = images[image_id]
        category_id = annotation.get("category_id")
        if type(category_id) is not int or category_id not in categories:
            raise AnnotationError(f"annotation {annotation_id}: unknown category_id {category_id!r}")
        if type(annotation.get("iscrowd")) is not int or annotation["iscrowd"] != 0:
            raise AnnotationError(f"annotation {annotation_id}: only non-crowd polygon instances are supported")
        polygons = annotation.get("segmentation")
        if not isinstance(polygons, list) or not polygons:
            raise AnnotationError(f"annotation {annotation_id}: segmentation must be a nonempty polygon list")
        for index, polygon in enumerate(polygons):
            _points(polygon, f"annotation {annotation_id} polygon {index}",
                    image["width"], image["height"], minimum=6)
        spine = annotation.get("spine")
        _points(spine, f"annotation {annotation_id} spine", image["width"], image["height"], minimum=4)
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise AnnotationError(f"annotation {annotation_id}: bbox must contain x,y,width,height")
        x, y, w, h = [_number(v, f"annotation {annotation_id} bbox") for v in bbox]
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > image["width"] + 1e-7 or y + h > image["height"] + 1e-7:
            raise AnnotationError(f"annotation {annotation_id}: bbox is empty or outside the image")
        source_area = _number(annotation.get("area"), f"annotation {annotation_id} area")
        if source_area <= 0:
            raise AnnotationError(f"annotation {annotation_id}: source area must be positive")
        try:
            # merge(union) preserves the source annotation's single identity.
            # pycocotools uses the standard COCO polygon rasterization and
            # returns compressed, column-major/F-order RLE counts.
            parts = mask_util.frPyObjects(polygons, image["height"], image["width"])
            rle = mask_util.merge(parts, intersect=False)
            area = int(mask_util.area(rle))
            counts = rle["counts"].decode("ascii")
        except (TypeError, ValueError, OverflowError, UnicodeError) as exc:
            raise AnnotationError(f"annotation {annotation_id}: polygon rasterization failed: {exc}") from exc
        if area <= 0:
            raise AnnotationError(f"annotation {annotation_id}: polygon rasterizes to an empty mask")
        if list(rle["size"]) != [image["height"], image["width"]]:
            raise AnnotationError(f"annotation {annotation_id}: rasterized RLE shape mismatch")
        group_counts[image_id] += 1
        filament_id = f"{image_id}_{group_counts[image_id]}"
        rows.append((filament_id, counts))
        category_counts[category_id] += 1
        multipolygon_count += len(polygons) > 1
        if not math.isclose(source_area, area, rel_tol=0, abs_tol=1e-6):
            area_mismatches.append({"annotation_id": annotation_id, "source_area": source_area,
                                    "rasterized_area": area})
        annotation_manifest.append({"filament_id": filament_id, "annotation_id": annotation_id,
                                    "annotator_image_id": image_id,
                                    "image_id": image_id.partition("-")[2],
                                    "category_id": category_id,
                                    "category_name": categories[category_id]["name"],
                                    "polygon_count": len(polygons), "source_area": source_area,
                                    "rasterized_area": area, "source_bbox": bbox,
                                    "spine": spine, "iscrowd": annotation["iscrowd"]})

    empty_groups = sorted(set(images) - set(group_counts))
    for entry in image_manifest:
        entry["instances"] = group_counts[entry["annotator_image_id"]]
    warnings = []
    if empty_groups:
        warnings.append("Some image/annotator groups have zero annotations and cannot be represented "
                        "as instance rows in the supplied notebook; see empty_annotator_image_groups.")
    if area_mismatches:
        warnings.append("Some source area fields differ from COCO-rasterized mask areas; "
                        "masks were not altered to force agreement. See manifest area_mismatches.")
    summary = {"status": "complete", "source": source_metadata or {},
               "annotator_image_groups": len(images), "physical_images": len(files_by_stem),
               "annotations": len(rows), "csv_rows": len(rows),
               "annotators": len({i["annotator_id"] for i in image_manifest}),
               "dimensions": [2048, 2048],
               "categories": [{"id": c, "name": categories[c]["name"],
                               "annotations": category_counts[c]} for c in sorted(categories)],
               "excluded_annotations": 0, "multipart_annotations": multipolygon_count,
               "empty_annotator_image_groups": empty_groups,
               "source_area_mismatch_count": len(area_mismatches),
               "warnings": warnings,
               "identity_policy": "one source annotation -> one GT instance; multipart polygon union",
               "id_policy": "<source image.id>_<1-based annotation order within group>",
               "rasterization": "pycocotools.frPyObjects + merge(intersect=False), compressed COCO RLE"}
    manifest = {"schema_version": 1, "source": source_metadata or {},
                "source_info": data.get("info"), "source_licenses": data.get("licenses"),
                "categories": data["categories"], "images": image_manifest,
                "annotations": annotation_manifest,
                "area_mismatches": area_mismatches,
                "empty_annotator_image_groups": empty_groups}
    return rows, manifest, summary


def convert_file(annotations: str | Path, output_dir: str | Path) -> dict:
    source = Path(annotations)
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise AnnotationError(f"output directory already exists; refusing to overwrite: {destination}")
    try:
        with source.open("rb") as handle:
            checksum = hashlib.file_digest(handle, "sha256").hexdigest()
        with source.open(encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AnnotationError(f"cannot read annotation JSON {source}: {exc}") from exc
    metadata = {"file_name": source.name, "sha256": checksum,
                "size_bytes": source.stat().st_size}
    rows, manifest, summary = convert_data(data, metadata)
    # Finish input validation before creating outputs. Existing directories,
    # including empty ones, are refused; each output file is also exclusive.
    try:
        destination.mkdir(parents=True, exist_ok=False)
        with (destination / "gt.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["filament_id", "segmentation_rle"])
            writer.writerows(rows)
        with (destination / "manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        # Summary is written last: it records completion, not merely startup.
        with (destination / "summary.json").open("x", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    except OSError as exc:
        raise AnnotationError(f"output write failed; any partial directory was retained at {destination}: {exc}") from exc
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, help="original MAGFiLO annotation JSON")
    parser.add_argument("--output-dir", required=True, help="new directory for gt.csv, manifest.json, summary.json")
    args = parser.parse_args(argv)
    try:
        summary = convert_file(args.annotations, args.output_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (AnnotationError, ValueError, OverflowError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    sys.exit(main())
