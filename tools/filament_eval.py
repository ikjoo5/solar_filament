#!/usr/bin/env python3
"""Validate filament CSVs and reproduce the supplied notebook's global PQ.

Dependencies: numpy, pycocotools. Ground truth must already be a two-column
CSV; conversion of an unknown annotation JSON schema is deliberately excluded.
The notebook uses all pairs with IoU > .5, without Hungarian assignment.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
from pycocotools import mask as mask_util


class ValidationError(ValueError):
    """The input does not satisfy the evaluator's CSV/mask contract."""


@dataclass(frozen=True)
class Filament:
    filament_id: str
    image_id: str
    group_id: str
    counts: str
    area: int

    def rle(self, height: int, width: int) -> dict:
        return {"size": [height, width], "counts": self.counts.encode("ascii")}


def parse_filament_id(value: str, ground_truth: bool = False) -> tuple[str, str]:
    if not isinstance(value, str) or not value or any(c.isspace() for c in value):
        raise ValidationError("filament_id must be nonempty and contain no whitespace")
    if any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValidationError("filament_id contains a control character")
    prefix, separator, suffix = value.partition("_")
    if not separator or not prefix or not suffix:
        raise ValidationError("filament_id must have a nonempty prefix and instance suffix")
    if ground_truth:
        annotator, separator, image_id = prefix.partition("-")
        if not separator or not annotator or not image_id:
            raise ValidationError("GT prefix must be '<annotator>-<image>'")
        return image_id, prefix
    return prefix, prefix


def _check_compressed_counts(counts: str, pixels: int) -> int:
    """Check COCO's signed/delta variable-length counts before its C decoder.

    In particular, require exact image coverage: some truncated/short RLEs
    otherwise leave unspecified bytes in the decoder's allocated output.
    """
    if not isinstance(counts, str) or not counts:
        raise ValidationError("segmentation_rle must be a nonempty compressed COCO string")
    previous: list[int] = []
    total = foreground = index = position = 0
    while position < len(counts):
        value = shift = 0
        while True:
            if position >= len(counts):
                raise ValidationError("unterminated compressed RLE count")
            digit = ord(counts[position]) - 48
            position += 1
            if not 0 <= digit <= 63:
                raise ValidationError("compressed RLE contains an invalid character")
            value |= (digit & 31) << shift
            shift += 5
            if shift > 65:
                raise ValidationError("compressed RLE count exceeds supported integer size")
            if not digit & 32:
                if digit & 16:
                    value |= -1 << shift
                break
        if index > 2:
            value += previous[-2]
        if value < 0:
            raise ValidationError("compressed RLE decodes to a negative run length")
        total += value
        if total > pixels:
            raise ValidationError("compressed RLE exceeds the requested image size")
        if index % 2:
            foreground += value
        previous.append(value)
        if len(previous) > 2:
            previous.pop(0)
        index += 1
    if total != pixels:
        raise ValidationError(f"compressed RLE covers {total} pixels; expected {pixels}")
    return foreground


def decode_rle(counts: str, height: int, width: int) -> tuple[np.ndarray, int]:
    if height <= 0 or width <= 0:
        raise ValidationError("height and width must be positive")
    expected_area = _check_compressed_counts(counts, height * width)
    if expected_area == 0:
        raise ValidationError("empty instance masks are not permitted; omit their rows")
    try:
        mask = mask_util.decode({"size": [height, width], "counts": counts.encode("ascii")})
    except (ValueError, TypeError, OverflowError, UnicodeError) as exc:
        raise ValidationError(f"cannot decode compressed COCO RLE: {exc}") from exc
    if mask.shape != (height, width) or not np.all((mask == 0) | (mask == 1)):
        raise ValidationError("decoded mask has invalid shape or nonbinary pixels")
    area = int(mask.sum(dtype=np.int64))
    if area != expected_area:
        raise ValidationError("decoded mask area differs from the validated RLE runs")
    return mask, area


def read_csv(path: str | Path, height: int = 2048, width: int = 2048,
             ground_truth: bool = False) -> list[Filament]:
    # Long filament encodings can exceed csv's platform-default field limit.
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    result: list[Filament] = []
    seen: set[str] = set()
    try:
        with Path(path).open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if (reader.fieldnames is None or len(reader.fieldnames) != 2
                    or set(reader.fieldnames) != {"filament_id", "segmentation_rle"}):
                raise ValidationError("CSV must contain exactly filament_id,segmentation_rle columns")
            for line, row in enumerate(reader, start=2):
                try:
                    if None in row or any(value is None for value in row.values()):
                        raise ValidationError("CSV row has missing or extra fields")
                    filament_id = row["filament_id"]
                    image_id, group_id = parse_filament_id(filament_id, ground_truth)
                    if filament_id in seen:
                        raise ValidationError(f"duplicate filament_id: {filament_id}")
                    _, area = decode_rle(row["segmentation_rle"], height, width)
                    seen.add(filament_id)
                    result.append(Filament(filament_id, image_id, group_id,
                                           row["segmentation_rle"], area))
                except ValidationError as exc:
                    raise ValidationError(f"{path}: row {line}: {exc}") from exc
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValidationError(f"cannot read CSV {path}: {exc}") from exc
    return result


def _group(rows: Iterable[Filament]) -> dict[str, list[Filament]]:
    groups: dict[str, list[Filament]] = {}
    for row in rows:
        groups.setdefault(row.group_id, []).append(row)
    return groups


def audit(rows: list[Filament], height: int, width: int) -> dict:
    """Audit overlap without changing or making masks disjoint."""
    groups = []
    for group_id, members in sorted(_group(rows).items()):
        covered = np.zeros((height, width), dtype=bool)
        overlapping = np.zeros((height, width), dtype=bool)
        for row in members:
            # Decode one instance at a time; never allocate n_masks*H*W.
            mask = mask_util.decode(row.rle(height, width)).astype(bool)
            overlapping |= covered & mask
            covered |= mask
        groups.append({"group_id": group_id, "image_id": members[0].image_id,
                       "instances": len(members), "union_pixels": int(covered.sum()),
                       "overlap_pixels": int(overlapping.sum())})
    areas = [row.area for row in rows]
    return {"rows": len(rows), "images": len({row.image_id for row in rows}),
            "groups": len(groups), "minimum_instance_area": min(areas, default=None),
            "maximum_instance_area": max(areas, default=None),
            "groups_with_overlap": sum(g["overlap_pixels"] > 0 for g in groups),
            "overlap_pixels_sum_across_groups": sum(g["overlap_pixels"] for g in groups),
            "per_group": groups}


def evaluate(gt: list[Filament], pred: list[Filament], height: int, width: int,
             filter_pred_to_gt: bool = False) -> dict:
    if not gt:
        raise ValidationError("GT CSV has no instances; notebook entry discovery cannot evaluate it")
    gt_images = {row.image_id for row in gt}
    unknown = sorted({row.image_id for row in pred} - gt_images)
    if unknown and not filter_pred_to_gt:
        raise ValidationError(
            f"prediction images absent from GT: {unknown}; the notebook would silently ignore "
            "them. For an intentional subset use --filter-pred-to-gt")
    selected = [row for row in pred if row.image_id in gt_images]
    predictions = _group(selected)
    total_tp = total_fp = total_fn = 0
    total_iou = 0.0
    per_group = []
    for group_id, truth in sorted(_group(gt).items()):
        image_id = truth[0].image_id
        candidates = predictions.get(image_id, [])
        if candidates:
            # COCO API's rows correspond to dt=GT here, columns to gt=pred.
            # iscrowd=0 gives ordinary IoU, not intersection/detection-area.
            iou = np.asarray(mask_util.iou(
                [r.rle(height, width) for r in truth],
                [r.rle(height, width) for r in candidates], [0] * len(candidates)),
                dtype=np.float64)
            hits = iou > 0.5
            tp = int(hits.sum())
            fp = int((~hits.any(axis=0)).sum())
            fn = int((~hits.any(axis=1)).sum())
            iou_sum = float(iou[hits].sum())
        else:
            tp, fp, fn, iou_sum = 0, 0, len(truth), 0.0
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_iou += iou_sum
        denominator = tp + 0.5 * fp + 0.5 * fn
        per_group.append({"group_id": group_id, "image_id": image_id,
                          "n_gt": len(truth), "n_pred": len(candidates),
                          "tp_pairs": tp, "fp": fp, "fn": fn,
                          "matched_iou_sum": iou_sum,
                          "pq_diagnostic": iou_sum / denominator if denominator else 0.0})
    denominator = total_tp + 0.5 * total_fp + 0.5 * total_fn
    return {"aggregation": "global TP-IoU / (TP + 0.5 FP + 0.5 FN), not macro average",
            "matching": "all pairs with IoU > 0.5; no one-to-one assignment",
            "pq": total_iou / denominator if denominator else 0.0,
            "sq": total_iou / total_tp if total_tp else 0.0,
            "rq": total_tp / denominator if denominator else 0.0,
            "tp_pairs": total_tp, "fp": total_fp, "fn": total_fn,
            "matched_iou_sum": total_iou, "denominator": denominator,
            "annotator_image_groups": len(per_group), "gt_images": len(gt_images),
            "images_without_predictions": sorted(gt_images - {r.image_id for r in selected}),
            "selected_prediction_rows": len(selected),
            "filtered_prediction_rows": len(pred) - len(selected),
            "filtered_prediction_images": unknown,
            "per_group": per_group,
            "limitations": [
                "Entry discovery uses GT instance rows only; blank-GT images/annotators cannot "
                "be represented or scored by this notebook contract.",
                "Predictions are evaluated once for every annotator-image GT group.",
                "COCO RLE integer intersections/float64 IoUs replace notebook torch float32 "
                "matmul; tiny rounding differences are possible. Matching remains strict >0.5.",
                "Overlaps are reported without automatic correction; all qualifying pairs "
                "are counted exactly as in the supplied notebook."]}


def read_expected_images(path: str | Path) -> set[str]:
    try:
        lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValidationError(f"cannot read expected images: {exc}") from exc
    result = set()
    for line, value in enumerate(lines, start=1):
        if not value:
            continue
        if value != value.strip() or any(c.isspace() for c in value) or "_" in value:
            raise ValidationError(f"expected-images line {line}: use a plain image stem without whitespace/_")
        if value in result:
            raise ValidationError(f"expected-images line {line}: duplicate image stem {value}")
        result.add(value)
    if not result:
        raise ValidationError("expected-images contains no image stems")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", required=True, help="prediction CSV")
    parser.add_argument("--gt", help="GT CSV, not annotation JSON")
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--expected-images", help="UTF-8 text file, one image stem per line")
    parser.add_argument("--filter-pred-to-gt", action="store_true",
                        help="explicitly discard prediction images outside an intentional GT subset")
    args = parser.parse_args(argv)
    try:
        if args.height <= 0 or args.width <= 0:
            raise ValidationError("height and width must be positive")
        if args.filter_pred_to_gt and not args.gt:
            raise ValidationError("--filter-pred-to-gt requires --gt")
        pred = read_csv(args.pred, args.height, args.width)
        warnings = []
        report = {"status": "ok", "mask_size": [args.height, args.width],
                  "prediction_file": str(args.pred),
                  "predictions": audit(pred, args.height, args.width)}
        pred_images = {r.image_id for r in pred}
        if not pred:
            warnings.append("Prediction CSV has zero instance rows; it does not demonstrate image coverage.")
        if report["predictions"]["groups_with_overlap"]:
            warnings.append("Prediction instances overlap; masks were audited but not modified.")
        if args.expected_images:
            expected = read_expected_images(args.expected_images)
            unknown = sorted(pred_images - expected)
            if unknown:
                raise ValidationError(f"prediction images outside expected-images: {unknown}")
            missing = sorted(expected - pred_images)
            report["coverage"] = {"expected_images": len(expected),
                                  "images_with_predictions": len(pred_images),
                                  "images_without_predictions": missing}
            if missing:
                warnings.append("Some expected images have no prediction rows; these may be true "
                                "no-detection cases or missing inference outputs and need review.")
        if args.gt:
            gt = read_csv(args.gt, args.height, args.width, ground_truth=True)
            report["ground_truth_file"] = str(args.gt)
            report["ground_truth"] = audit(gt, args.height, args.width)
            report["evaluation"] = evaluate(gt, pred, args.height, args.width, args.filter_pred_to_gt)
            warnings.append("Blank-GT images/annotators are absent from notebook entry discovery; "
                            "the score is restricted to GT groups represented by instance rows.")
            if report["evaluation"]["filtered_prediction_rows"]:
                warnings.append("Prediction rows outside GT were explicitly filtered; see evaluation report.")
            if args.expected_images:
                report["coverage"]["expected_images_without_gt_rows"] = sorted(
                    expected - {r.image_id for r in gt})
        else:
            warnings.append("No GT supplied: CSV/mask audit only; PQ has not been computed.")
        report["warnings"] = warnings
        if warnings:
            report["status"] = "ok_with_warnings"
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValidationError, ValueError, OverflowError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
