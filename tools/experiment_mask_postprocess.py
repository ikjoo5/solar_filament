#!/usr/bin/env python3
"""Small, annotation-independent postprocessing ablations on fixed dev predictions.

All candidates use the same 91-image development set. These are tuning results,
not independent CV estimates or Kaggle leaderboard improvements. CSVs have no
confidence field: input order is only input order, never asserted confidence.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import time

import numpy as np
from pycocotools import mask as mu
from scipy import ndimage

from filament_eval import Filament, _check_compressed_counts, evaluate

SIZE = 2048


def read_rows(path: Path, gt: bool = False, selected: set[str] | None = None) -> list[Filament]:
    result = []
    seen = set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            fid = row["filament_id"]
            group = fid.split("_", 1)[0]
            if selected is not None and group not in selected:
                continue
            assert fid not in seen, f"duplicate ID {fid}"
            seen.add(fid)
            image = group.split("-", 1)[1] if gt else group
            counts = row["segmentation_rle"]
            area = _check_compressed_counts(counts, SIZE * SIZE)
            assert area > 0
            result.append(Filament(fid, image, group, counts, area))
    return result


def group_rows(rows: list[Filament]) -> dict[str, list[Filament]]:
    result = defaultdict(list)
    for row in rows:
        result[row.image_id].append(row)
    return dict(result)


def encode(image: str, index: int, mask: np.ndarray) -> Filament:
    rle = mu.encode(np.asarray(mask, dtype=np.uint8, order="F"))
    return Filament(f"{image}_{index}", image, image, rle["counts"].decode("ascii"), int(mu.area(rle)))


def overlap_stats(rows: list[Filament]) -> dict:
    # Pairwise RLE intersections, unioned, count exactly pixels covered >=2.
    # This avoids repeatedly decoding full 2048-square dense masks.
    overlapping_groups = overlapping_pixels = exact_duplicate_pairs = 0
    for members in group_rows(rows).values():
        if len(members) < 2:
            continue
        rles = [r.rle(SIZE, SIZE) for r in members]
        matrix = np.asarray(mu.iou(rles, rles, [0] * len(rles)))
        exact_duplicate_pairs += int(np.triu(matrix == 1, k=1).sum())
        pairs = np.argwhere(np.triu(matrix > 0, k=1))
        if len(pairs):
            intersections = [mu.merge([rles[int(i)], rles[int(j)]], intersect=True) for i, j in pairs]
            pixels = int(mu.area(mu.merge(intersections, intersect=False)))
            overlapping_groups += pixels > 0
            overlapping_pixels += pixels
    return dict(overlap_images=overlapping_groups, overlap_pixels=overlapping_pixels,
                exact_duplicate_mask_pairs=exact_duplicate_pairs)


def disjoint(rows: list[Filament], order: str) -> list[Filament]:
    output = []
    for image, members in group_rows(rows).items():
        if order == "area_desc":
            members = sorted(members, key=lambda r: -r.area)
        covered_rle = None
        index = 0
        for row in members:
            original = row.rle(SIZE, SIZE)
            intersection = (int(mu.area(mu.merge([original, covered_rle], intersect=True)))
                            if covered_rle is not None else 0)
            if intersection == 0:
                # Most masks are unchanged; reuse their original RLE exactly.
                kept = original
                area = row.area
            else:
                x, y, w, h = mu.toBbox(original).astype(int)
                a = mu.decode(original)[y:y+h, x:x+w].astype(bool)
                b = mu.decode(covered_rle)[y:y+h, x:x+w].astype(bool)
                roi = a & ~b
                area = int(roi.sum())
                if area < 20:
                    continue
                mask = np.zeros((SIZE, SIZE), dtype=np.uint8, order="F")
                mask[y:y+h, x:x+w] = roi
                kept = mu.encode(mask)
            # Sub-threshold fragments do not claim pixels.
            if area >= 20:
                index += 1
                output.append(Filament(f"{image}_{index}", image, image,
                                       kept["counts"].decode("ascii"), area))
                covered_rle = (mu.merge([covered_rle, kept], intersect=False)
                               if covered_rle is not None else kept)
    return output


def mask_nms(rows: list[Filament], threshold: float) -> list[Filament]:
    output = []
    for members in group_rows(rows).values():
        rles = [r.rle(SIZE, SIZE) for r in members]
        iou = np.asarray(mu.iou(rles, rles, [0] * len(members)))
        keep = []
        for i in range(len(members)):
            if not any(iou[i, j] > threshold for j in keep):
                keep.append(i)
        output.extend(members[i] for i in keep)
    return output


def morphology(rows: list[Filament], operation: str) -> list[Filament]:
    output = []
    structure = ndimage.generate_binary_structure(2, 1)  # 3x3 cross, one iteration.
    for image, members in group_rows(rows).items():
        index = 0
        for row in members:
            rle = row.rle(SIZE, SIZE)
            x, y, w, h = mu.toBbox(rle).astype(int)
            x1, y1 = max(0, x - 2), max(0, y - 2)
            x2, y2 = min(SIZE, x + w + 2), min(SIZE, y + h + 2)
            mask = mu.decode(rle).astype(bool)
            roi = mask[y1:y2, x1:x2]
            transform = ndimage.binary_dilation if operation == "dilate" else ndimage.binary_erosion
            changed = transform(roi, structure=structure, iterations=1, border_value=0)
            if int(changed.sum()) >= 20:
                result = np.zeros((SIZE, SIZE), dtype=np.uint8, order="F")
                result[y1:y2, x1:x2] = changed
                index += 1
                output.append(encode(image, index, result))
    # A dilation can create overlaps; resolve those without looking at GT.
    return disjoint(output, "input") if operation == "dilate" else output


def report(gt: list[Filament], pred: list[Filament], name: str, rule: str) -> dict:
    value = evaluate(gt, pred, SIZE, SIZE)
    # Multiple predictions can match one GT under the literal all-pairs metric.
    duplicate_matches = 0
    unique_gt = unique_pred = 0
    by_pred = group_rows(pred)
    gt_groups = defaultdict(list)
    for row in gt:
        gt_groups[row.group_id].append(row)
    for members in gt_groups.values():
        candidates = by_pred.get(members[0].image_id, [])
        if candidates:
            iou = np.asarray(mu.iou([r.rle(SIZE, SIZE) for r in members],
                                    [r.rle(SIZE, SIZE) for r in candidates], [0] * len(candidates)))
            hit = iou > .5
            unique_gt += int(hit.any(axis=1).sum())
            unique_pred += int(hit.any(axis=0).sum())
            duplicate_matches += int(np.maximum(hit.sum(axis=1) - 1, 0).sum())
    return dict(name=name, rule=rule, prediction_rows=len(pred), pq=value["pq"], sq=value["sq"], rq=value["rq"],
                tp_pairs=value["tp_pairs"], fp=value["fp"], fn=value["fn"],
                unique_matched_gt=unique_gt, unique_matched_predictions_per_annotator=unique_pred,
                extra_pairs_beyond_one_per_gt=duplicate_matches,
                **overlap_stats(pred))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--val-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    with args.val_split.open(newline="") as handle:
        groups = {row["image_id"] for row in csv.DictReader(handle)}
    gt = read_rows(args.gt, gt=True, selected=groups)
    assert len(gt) == 708 and len(groups) == 101
    baseline = read_rows(args.results / "union_cc_conf0.25.csv")
    experiments = [("union_cc_conf0.25", "Unmodified uploaded baseline", baseline)]
    for confidence in ("0.15", "0.25"):
        rows = read_rows(args.results / f"preserve_conf{confidence}.csv")
        prefix = f"preserve_conf{confidence}"
        experiments.append((prefix, "Unmodified uploaded diagnostic masks", rows))
        for order in ("input", "area_desc"):
            experiments.append((f"{prefix}_disjoint_{order}",
                f"Assign overlap pixels to earlier mask in {order} order; area>=20; preserve disconnected identity", disjoint(rows, order)))
        for threshold in (.7, .9):
            experiments.append((f"{prefix}_nms{threshold}",
                f"Input-order greedy mask IoU NMS >{threshold}; no confidence assumed; overlaps can remain", mask_nms(rows, threshold)))
    for operation in ("dilate", "erode"):
        experiments.append((f"union_cc_conf0.25_{operation}1_cross",
            f"Per-object {operation} one 3x3-cross iteration, keep identity and area>=20; dilation overlap to earlier input mask", morphology(baseline, operation)))
    reports = []
    candidates = []
    for name, rule, rows in experiments:
        item = report(gt, rows, name, rule)
        reports.append(item)
        print(name, "PQ", item["pq"], "overlap_images", item["overlap_images"], flush=True)
        if len(reports) > 1 and item["pq"] > reports[0]["pq"] and item["overlap_images"] == 0:
            candidates.append((item["pq"], name, rows))
    saved = None
    if candidates:
        assert not args.candidate_dir.exists(), "candidate output already exists"
        args.candidate_dir.mkdir(parents=True)
        _, name, rows = max(candidates, key=lambda x: x[0])
        path = args.candidate_dir / f"{name}.csv"
        with path.open("x", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["filament_id", "segmentation_rle"])
            writer.writerows((r.filament_id, r.counts) for r in rows)
        saved = {"name": name, "csv": str(path), "scope": "validation predictions, not test submission"}
    output = dict(scope="Limited development-set postprocessing tuning; not independent CV or Kaggle leaderboard score",
        gt_instances=len(gt), annotation_groups=len(groups), images=len({r.image_id for r in gt}),
        rules="No GT-dependent per-mask selection; input order is not asserted confidence; thresholds fixed before evaluation",
        caveat="Literal all-pairs PQ can count several overlapping predictions matching the same GT. Compare disjoint candidates separately.",
        experiments=reports, best_nonoverlap_candidate_csv=saved, elapsed_seconds=time.monotonic() - start)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
