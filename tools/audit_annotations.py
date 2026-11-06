#!/usr/bin/env python3
"""Read-only COCO polygon audit; rasterize at native coordinates before ROI analysis.

Writes summary statistics only. Does not copy the source dataset, modify labels,
infer image pixels, or estimate model PQ. Requires numpy, scipy, pycocotools.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from pycocotools import mask as mask_utils
from scipy import ndimage


def histogram(values):
    return {str(k): int(v) for k, v in sorted(collections.Counter(values).items())}


def quantiles(values):
    if not values:
        return None
    q = [0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1]
    return {str(x): float(y) for x, y in zip(q, np.quantile(values, q))}


def rasterize(annotation, height, width):
    seg = annotation["segmentation"]
    if isinstance(seg, list):
        if not seg:
            return np.zeros((height, width), np.uint8), (0, 0, 0, 0)
        rle = mask_utils.merge(mask_utils.frPyObjects(seg, height, width))
    elif isinstance(seg["counts"], list):
        rle = mask_utils.frPyObjects(seg, height, width)
    else:
        rle = dict(seg)
        if isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode("ascii")
    mask = mask_utils.decode(rle)
    if mask.ndim != 2 or mask.shape != (height, width):
        raise ValueError(f"Unexpected mask shape: {annotation['id']}, {mask.shape}")
    bbox = mask_utils.toBbox(mask_utils.encode(np.asfortranarray(mask)))
    return mask, bbox


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--submission", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cx", type=float, default=1024)
    parser.add_argument("--cy", type=float, default=1024)
    parser.add_argument("--radius", type=float, default=900)
    args = parser.parse_args()
    started = time.monotonic()
    source_bytes = args.annotations.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    data = json.loads(source_bytes)
    del source_bytes
    images, annotations = data["images"], data["annotations"]
    image_by_id = {i["id"]: i for i in images}
    ann_by_image = collections.defaultdict(list)
    for ann in annotations:
        if ann["image_id"] not in image_by_id:
            raise ValueError(f"Unknown annotation image ID: {ann['image_id']}")
        ann_by_image[ann["image_id"]].append(ann)
    by_file = collections.defaultdict(list)
    for image in images:
        by_file[image["file_name"]].append(image)
    category_names = {c["id"]: c["name"] for c in data["categories"]}
    areas, declared_areas, lengths, component_counts, hole_counts = [], [], [], [], []
    category_areas = collections.defaultdict(list)
    component_areas = []
    minor_pixels = 0
    ann_minor_components = 0
    ann_area_lt20 = 0
    ann_outside = 0
    ann_mostly_outside = 0
    outside_pixels = 0
    blank_annotations = []
    overlap_images = []
    overlap_pixels = 0
    overlap_annotations = 0
    spine_empty, spine_malformed = 0, 0
    spine_point_counts = []
    structure8 = np.ones((3, 3), dtype=np.uint8)
    circle_cache = {}
    per_image_counts = []
    multi_component_examples = []
    outside_examples = []
    all_removed_by_lt20 = 0
    roundtrip = {
        "diagnostic_type": "annotation round-trip diagnostic, not model/system PQ",
        "prediction_source": "Each annotation-image group's own GT union, then 8CC and component area >=20",
        "warning": "Prediction is regenerated separately from each annotator's GT; it is not a deployable shared prediction.",
        "matching": "all GT/pred pairs with IoU >0.5; unmatched columns FP and unmatched rows FN",
        "tp_pairs": 0, "fp": 0, "fn": 0, "matched_iou_sum": 0.0,
        "predicted_instances_total": 0, "gt_nonempty_instances_total": 0,
        "groups_with_object_count_change": 0, "groups_with_more_predicted_instances": 0,
        "groups_with_fewer_predicted_instances": 0, "groups_with_pq_below_one": 0,
        "changed_group_examples": [],
    }
    for image_number, image in enumerate(images, start=1):
        height, width = image["height"], image["width"]
        shape = (height, width)
        if shape not in circle_cache:
            yy, xx = np.ogrid[:height, :width]
            circle_cache[shape] = (xx-args.cx)**2 + (yy-args.cy)**2 > args.radius**2
        outside_circle = circle_cache[shape]
        # At most one native-size coverage map and one decoded annotation at a time.
        coverage = np.zeros(shape, dtype=np.uint16)
        rois = []
        image_anns = ann_by_image[image["id"]]
        per_image_counts.append(len(image_anns))
        for ann in image_anns:
            mask, bbox = rasterize(ann, height, width)
            area = int(mask.sum())
            areas.append(area)
            declared_areas.append(float(ann["area"]))
            category_areas[category_names[ann["category_id"]]].append(area)
            if area == 0:
                blank_annotations.append(ann["id"])
                roi = np.zeros((0, 0), np.uint8)
                x = y = 0
                n_components = 0
                sizes = np.array([], dtype=int)
            else:
                x, y, bw, bh = (int(v) for v in bbox)
                roi = np.ascontiguousarray(mask[y:y+bh, x:x+bw])
                components, n_components = ndimage.label(roi, structure=structure8)
                sizes = np.bincount(components.ravel())[1:]
                # Count geometric holes with 4-connected background, complementary
                # to the 8-connected foreground convention of the baseline.
                hole_mask = ndimage.binary_fill_holes(roi) & ~roi.astype(bool)
                _, n_holes = ndimage.label(hole_mask)
                hole_counts.append(int(n_holes))
                if n_components > 1 and len(multi_component_examples) < 8:
                    multi_component_examples.append({"annotation_id": ann["id"],
                        "image_id": ann["image_id"], "component_areas": sizes.tolist()})
                outside = int(np.logical_and(roi, outside_circle[y:y+bh, x:x+bw]).sum())
                outside_pixels += outside
                ann_outside += int(outside > 0)
                ann_mostly_outside += int(outside > area/2)
                if outside and len(outside_examples) < 8:
                    outside_examples.append({"annotation_id": ann["id"],
                        "image_id": ann["image_id"], "area": area, "outside_pixels": outside})
                coverage[y:y+bh, x:x+bw] += roi
                rois.append((y, x, roi))
            component_counts.append(int(n_components))
            component_areas.extend(int(v) for v in sizes)
            small = sizes < 20
            minor_pixels += int(sizes[small].sum())
            ann_minor_components += int(small.any())
            all_removed_by_lt20 += int(bool(area) and not bool((sizes >= 20).any()))
            ann_area_lt20 += int(area < 20)
            spine = ann.get("spine", [])
            if not spine:
                spine_empty += 1
                spine_point_counts.append(0)
            elif len(spine) % 2:
                spine_malformed += 1
            else:
                pts = np.asarray(spine, dtype=float).reshape(-1, 2)
                spine_point_counts.append(len(pts))
                lengths.append(float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()))
        overlapping = coverage > 1
        pixels = int(overlapping.sum())
        if pixels:
            overlap_images.append(image["id"])
            overlap_pixels += pixels
            for y, x, roi in rois:
                bh, bw = roi.shape
                overlap_annotations += int(bool(np.logical_and(roi, overlapping[y:y+bh, x:x+bw]).any()))
        # Annotation conversion diagnostic only: each group supplies its own
        # "predictions", so these figures must not be reported as system PQ.
        union_components, n_union_components = ndimage.label(coverage > 0, structure=structure8)
        union_areas = np.bincount(union_components.ravel(), minlength=n_union_components+1)
        keep = np.flatnonzero(union_areas >= 20)
        keep = keep[keep != 0]
        n_gt, n_pred = len(rois), len(keep)
        ious = np.zeros((n_gt, n_pred), dtype=np.float64)
        for k, (y, x, roi) in enumerate(rois):
            bh, bw = roi.shape
            local_labels = union_components[y:y+bh, x:x+bw]
            intersections = np.bincount(local_labels[roi != 0], minlength=n_union_components+1)[keep]
            gt_area = int(roi.sum())
            ious[k] = intersections/(gt_area+union_areas[keep]-intersections)
        hits = ious > 0.5
        tp = int(hits.sum())
        fp = int((~hits.any(axis=0)).sum())
        fn = int((~hits.any(axis=1)).sum())
        iou_sum = float(ious[hits].sum())
        denominator = tp+0.5*fp+0.5*fn
        group_pq = iou_sum/denominator if denominator else 0.0
        roundtrip["tp_pairs"] += tp
        roundtrip["fp"] += fp
        roundtrip["fn"] += fn
        roundtrip["matched_iou_sum"] += iou_sum
        roundtrip["predicted_instances_total"] += n_pred
        roundtrip["gt_nonempty_instances_total"] += n_gt
        roundtrip["groups_with_object_count_change"] += int(n_pred != n_gt)
        roundtrip["groups_with_more_predicted_instances"] += int(n_pred > n_gt)
        roundtrip["groups_with_fewer_predicted_instances"] += int(n_pred < n_gt)
        roundtrip["groups_with_pq_below_one"] += int(group_pq < 1-1e-12)
        if group_pq < 1-1e-12 and len(roundtrip["changed_group_examples"]) < 20:
            roundtrip["changed_group_examples"].append({"image_id": image["id"],
                "n_gt": n_gt, "n_pred": n_pred, "tp_pairs": tp,
                "fp": fp, "fn": fn, "diagnostic_pq": group_pq})
        if image_number % 100 == 0:
            print(f"Audited {image_number}/{len(images)} records; {time.monotonic()-started:.1f}s", flush=True)
    stem_set = {Path(name).stem for name in by_file}
    date_set = {str(i["date_captured"])[:10] for i in images}
    image_prefixes = collections.Counter(str(i["id"]).split("-", 1)[0] for i in images)
    category_counts = collections.Counter(category_names[a["category_id"]] for a in annotations)
    roundtrip_denominator = roundtrip["tp_pairs"] + .5*roundtrip["fp"] + .5*roundtrip["fn"]
    roundtrip["global_diagnostic_pq"] = roundtrip["matched_iou_sum"]/roundtrip_denominator if roundtrip_denominator else 0.0
    roundtrip["global_diagnostic_sq"] = roundtrip["matched_iou_sum"]/roundtrip["tp_pairs"] if roundtrip["tp_pairs"] else 0.0
    roundtrip["global_diagnostic_rq"] = roundtrip["tp_pairs"]/roundtrip_denominator if roundtrip_denominator else 0.0
    result = {
        "source": {"file_name": args.annotations.name, "sha256": source_sha},
        "method": {"rasterization": "pycocotools frPyObjects + merge + decode at native H,W",
            "connectivity": 8, "holes_background_connectivity": 4,
            "area_units": "pixels", "spine_length_units": "pixels",
            "fixed_circle": {"cx": args.cx, "cy": args.cy, "radius": args.radius},
            "notes": ["Image records are annotation variants, not all distinct physical images.",
                "Prefix codes are reported without inferring annotator identity semantics.",
                "Overlap is within one annotation image record, not across annotators.",
                "Full-size masks are rasterized before bbox cropping; polygon rounding is unchanged.",
                "Counts and summed areas include annotation variants; not unique physical objects."]},
        "metadata": {"image_records": len(images), "unique_filenames": len(by_file),
            "annotation_count": len(annotations),
            "duplicate_image_ids": len(images)-len(image_by_id),
            "duplicate_annotation_ids": len(annotations)-len({a['id'] for a in annotations}),
            "dimensions": histogram(f"{i['width']}x{i['height']}" for i in images),
            "variants_per_physical_image": histogram(len(items) for items in by_file.values()),
            "annotation_group_prefix_counts": dict(sorted(image_prefixes.items())),
            "unique_annotation_group_prefixes": len(image_prefixes),
            "image_id_suffix_filename_mismatches": sum(str(i['id']).split('-', 1)[-1] != Path(i['file_name']).stem for i in images),
            "unique_capture_dates": len(date_set),
            "capture_min": min(i['date_captured'] for i in images),
            "capture_max": max(i['date_captured'] for i in images),
            "observation_site_suffix_counts_unique_files": dict(sorted(collections.Counter(Path(n).stem[14:] for n in by_file).items())),
            "annotation_categories": dict(sorted(category_counts.items())),
            "iscrowd": histogram(a.get('iscrowd') for a in annotations),
            "polygons_per_annotation": histogram(len(a['segmentation']) for a in annotations),
            "annotations_per_image_record": quantiles(per_image_counts),
            "empty_image_records": sum(n == 0 for n in per_image_counts)},
        "raster": {"area_quantiles": quantiles(areas),
            "declared_area_quantiles": quantiles(declared_areas),
            "declared_area_unequal_raster_count": int(np.count_nonzero(np.asarray(areas) != np.asarray(declared_areas))),
            "declared_minus_raster_quantiles": quantiles((np.asarray(declared_areas)-np.asarray(areas)).tolist()),
            "summed_annotation_area": int(sum(areas)),
            "area_by_category": {k: quantiles(v) for k, v in sorted(category_areas.items())},
            "blank_annotation_count": len(blank_annotations), "blank_annotation_examples": blank_annotations[:8],
            "components_per_annotation": histogram(component_counts),
            "component_area_quantiles": quantiles(component_areas),
            "multi_component_examples": multi_component_examples,
            "annotations_containing_components_lt20": ann_minor_components,
            "components_lt20": sum(v < 20 for v in component_areas),
            "pixels_in_components_lt20": minor_pixels,
            "annotations_entirely_removed_by_component_lt20_filter": all_removed_by_lt20,
            "annotations_total_area_lt20": ann_area_lt20,
            "hole_count_distribution": histogram(hole_counts),
            "annotations_with_holes": sum(v > 0 for v in hole_counts),
            "fixed_circle_annotations_losing_pixels": ann_outside,
            "fixed_circle_annotations_losing_majority": ann_mostly_outside,
            "fixed_circle_pixels_lost": outside_pixels,
            "fixed_circle_fraction_summed_gt_area_lost": outside_pixels/sum(areas) if sum(areas) else None,
            "fixed_circle_affected_examples": outside_examples,
            "overlap_image_records": len(overlap_images),
            "overlap_annotation_count": overlap_annotations,
            "overlap_pixels_summed_over_records": overlap_pixels,
            "overlap_image_examples": overlap_images[:8]},
        "spines": {"empty_count": spine_empty, "malformed_odd_coordinate_count": spine_malformed,
            "point_count_quantiles": quantiles(spine_point_counts), "polyline_length_quantiles": quantiles(lengths)},
        "annotation_union_cc_roundtrip": roundtrip,
        "requested_sample_20110109104734Ch": {
            "images": [{"image_id": i['id'], "file_name": i['file_name'],
                "date_captured": i['date_captured'], "annotation_count": len(ann_by_image[i['id']]),
                "category_counts": dict(collections.Counter(category_names[a['category_id']] for a in ann_by_image[i['id']]))}
                for i in images if Path(i['file_name']).stem == '20110109104734Ch']},
    }
    if args.submission:
        with args.submission.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        submission_stems = {r["filament_id"].rsplit("_", 1)[0] for r in rows}
        overlap_stems = sorted(stem_set & submission_stems)
        result["submitted_csv"] = {"file_name": args.submission.name, "rows": len(rows),
            "unique_image_stems": len(submission_stems),
            "duplicate_filament_ids": len(rows)-len({r['filament_id'] for r in rows}),
            "training_image_stem_overlap_count": len(overlap_stems),
            "training_image_stem_overlap_examples": overlap_stems[:20],
            "note": "A zero-row image is absent from this CSV; total test-image coverage is unknown."}
    result["elapsed_seconds"] = round(time.monotonic()-started, 3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "elapsed_seconds": result['elapsed_seconds'],
        "annotations": len(annotations), "raster": result['raster']}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
