"""Frozen Q50 append-only decoder: prediction fields only, no files or labels.

COCO RLEs use compressed counts and original 2048x2048 geometry. All baseline
masks are preserved. Two distinct RLEs and two views are geometric agreement,
not independent evidence of correctness. Thresholds reproduce the prior Q50.
"""
from collections import Counter, defaultdict
import hashlib
import math

import numpy as np
from pycocotools import mask as mu


def _probability(value):
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError("Expected finite probability in [0,1]")
    return result


def _counts(value):
    if isinstance(value, str):
        return value.encode("ascii")
    if isinstance(value, bytes):
        return value
    raise ValueError("Expected compressed COCO RLE counts")


def _mask(value):
    if isinstance(value, dict):
        if list(value.get("size", [])) != [2048, 2048]:
            raise ValueError("Q50 requires native 2048x2048 RLE masks")
        return dict(size=[2048, 2048], counts=_counts(value["counts"]))
    return dict(size=[2048, 2048], counts=_counts(value))


def area(mask):
    return int(mu.area(mask))


def intersection(first, second):
    return area(mu.merge([first, second], intersect=True))


def _values(instances, scores, geometric_records):
    if len(instances) != len(scores):
        raise ValueError("Quality score count must equal raw candidate count")
    if len(geometric_records) != 4 or {r["view"] for r in geometric_records} != {"base", "scale", "hflip", "vflip"}:
        raise ValueError("Expected exactly four geometric views")
    by_rle = defaultdict(dict)
    for record in geometric_records:
        for row in record["instances"]:
            rle = _counts(row["rle"])
            confidence = _probability(row["confidence"])
            by_rle[rle][record["view"]] = max(by_rle[rle].get(record["view"], 0), confidence)
    values = []
    seen = set()
    for i, (row, quality) in enumerate(zip(instances, scores)):
        rle = _counts(row["rle"])
        confidence = _probability(row["confidence"])
        quality = _probability(quality)
        if rle in seen:
            raise ValueError("Raw candidate RLEs must be unique")
        seen.add(rle)
        if rle not in by_rle or abs(confidence - max(by_rle[rle].values())) >= 1e-7:
            raise ValueError("Raw RLE/confidence does not match geometric cache at index " + str(i))
        values.append(dict(index=i, mask=_mask(rle), confidence=confidence, quality=quality,
            rle_sha256=hashlib.sha256(rle).hexdigest(),
            views=[view for view, c in by_rle[rle].items() if c >= .10]))
    return values


def support_counts(values):
    support = [r for r in values if r["confidence"] >= .10 and area(r["mask"]) >= 300 and r["views"]]
    if not support:
        return {r["index"]: dict(passed=False, unique_masks=0, unique_views=0, support_sources=[]) for r in values}
    matrix = np.asarray(mu.iou([r["mask"] for r in values], [r["mask"] for r in support], [0] * len(support)))
    result = {}
    for i, row in enumerate(values):
        neighbors = [support[int(j)] for j in np.flatnonzero(matrix[i] >= .65)]
        rles = {r["rle_sha256"] for r in neighbors}
        views = {v for r in neighbors for v in r["views"]}
        result[row["index"]] = dict(passed=len(rles) >= 2 and len(views) >= 2,
            unique_masks=len(rles), unique_views=len(views), support_sources=[r["index"] for r in neighbors])
    return result


def append_q50(originals, instances, scores, geometric_records):
    """Append accepted masks; return (RLE list, gate counters, accepted trace).

    Counts strings or bytes are accepted. Returned original entries are the
    original dict objects, without mutations; added entries use bytes counts.
    Instances' targets are never accessed. Scores must preserve raw ordering.
    """
    original = [_mask(r) for r in originals]
    original_union = mu.merge(original) if original else None
    if original and area(original_union) != sum(map(area, original)):
        raise ValueError("Baseline masks overlap")
    values = _values(instances, scores, geometric_records)
    support = support_counts(values)
    stats = Counter(raw=len(values))
    decisions = []
    covered = original_union
    output = list(original)
    overlaps = np.asarray(mu.iou([r["mask"] for r in values], original, [0] * len(original))) if values and original else None
    for row in sorted(values, key=lambda r: (-r["quality"], -r["confidence"], r["index"])):
        i, raw = row["index"], row["mask"]
        a = area(raw)
        if row["confidence"] < .10:
            stats["reject_confidence"] += 1; continue
        if a < 300:
            stats["reject_area"] += 1; continue
        if row["quality"] < .50:
            stats["reject_quality"] += 1; continue
        if not support[i]["passed"]:
            stats["reject_geometry_quorum"] += 1; continue
        if overlaps is not None and overlaps[i].max() > .10:
            stats["reject_baseline_iou"] += 1; continue
        old_overlap = intersection(raw, original_union) if original_union else 0
        if old_overlap / a > .05:
            stats["reject_baseline_clipping"] += 1; continue
        removed = intersection(raw, covered) if covered else 0
        if removed / a > .05:
            stats["reject_total_clipping"] += 1; continue
        if a - removed < 300:
            stats["reject_retained_area"] += 1; continue
        final = raw
        if removed:
            pixels = mu.decode(raw).astype(bool) & ~mu.decode(covered).astype(bool)
            final = mu.encode(np.asfortranarray(pixels.astype(np.uint8)))
        output.append(final)
        covered = mu.merge(output)
        stats["accepted"] += 1
        decisions.append(dict(source=i, raw_rle_sha256=row["rle_sha256"],
            final_rle_sha256=hashlib.sha256(final["counts"]).hexdigest(),
            confidence=row["confidence"], quality=row["quality"], clipped_fraction=removed / a, support=support[i]))
    if output and area(mu.merge(output)) != sum(map(area, output)):
        raise RuntimeError("Append decoder produced overlapping masks")
    output[:len(originals)] = originals
    return output, dict(stats), decisions
