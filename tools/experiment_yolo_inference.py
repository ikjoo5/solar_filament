#!/usr/bin/env python3
"""Bounded low-confidence, resolution and flip experiments on existing images.

All masks are returned to the original image coordinates. Cross-view foreground
agreement is a binary vote, not a model probability/logit. The raw mask and
confidence cache supports later image-free analysis without reconstructing
confidence from a thresholded submission CSV.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from pycocotools import mask as mask_util
from scipy import ndimage

from convert_annotations import convert_data
from evaluate_yolo_baseline import (
    RunnerError, infer_instances, masks_to_filaments, preprocess_limb_only,
    read_split, select_validation, sha256, write_csv, write_json,
)
from filament_eval import Filament, evaluate


VIEW_SIZES = {"base2048": 2048, "scale2560": 2560,
              "hflip2048": 2048, "vflip2048": 2048}
DEFAULT_VIEWS = tuple(VIEW_SIZES)
BASE_THRESHOLDS = (.05, .10, .15, .20, .25)
VOTE_THRESHOLDS = (.10, .15, .20)
RAW_MIN_CONFIDENCE = .05


def apply_view(array: np.ndarray, view: str) -> np.ndarray:
    if view not in VIEW_SIZES:
        raise RunnerError(f"unknown view: {view}")
    if array.ndim != 2:
        raise RunnerError("views require a two-dimensional image or mask")
    if view == "hflip2048":
        return np.flip(array, axis=1).copy()
    if view == "vflip2048":
        return np.flip(array, axis=0).copy()
    # imgsz changes the model's internal letterboxing, not original coordinates.
    return np.ascontiguousarray(array)


def invert_view(mask: np.ndarray, view: str) -> np.ndarray:
    # Horizontal/vertical reflection is its own inverse. Scale inference uses
    # retina masks in the original input shape before this function is called.
    return apply_view(mask, view)


def infer_view(model, processed: np.ndarray, view: str,
               device: str = "0", max_det: int = 100) -> list[dict]:
    transformed = apply_view(processed, view)
    instances = infer_instances(model, transformed, RAW_MIN_CONFIDENCE,
                                device, VIEW_SIZES[view], max_det)
    return [{"mask": invert_view(instance["mask"], view),
             "confidence": float(instance["confidence"])} for instance in instances]


def view_foreground(instances: list[dict], confidence: float,
                    shape: tuple[int, int]) -> np.ndarray:
    if not np.isfinite(confidence) or not 0 <= confidence <= 1:
        raise RunnerError("confidence must be finite in [0,1]")
    foreground = np.zeros(shape, dtype=bool)
    for instance in instances:
        mask = np.asarray(instance["mask"], dtype=bool)
        if mask.shape != shape:
            raise RunnerError("original-coordinate mask shape mismatch")
        value = float(instance["confidence"])
        if not np.isfinite(value):
            raise RunnerError("nonfinite confidence")
        if value >= confidence:
            foreground |= mask
    return foreground


def vote_foreground(view_foregrounds: list[np.ndarray], fraction: float) -> np.ndarray:
    if not view_foregrounds or not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise RunnerError("binary voting requires views and a fraction in (0,1]")
    shape = view_foregrounds[0].shape
    counts = np.zeros(shape, dtype=np.uint16)
    for foreground in view_foregrounds:
        if foreground.shape != shape:
            raise RunnerError("view foreground shapes differ")
        counts += np.asarray(foreground, dtype=bool)
    return counts >= math.ceil(fraction * len(view_foregrounds))


def component_masks(foreground: np.ndarray, min_area: int) -> list[np.ndarray]:
    if foreground.ndim != 2 or min_area < 1:
        raise RunnerError("component decoding requires a 2-D mask and positive minimum area")
    labels, count = ndimage.label(foreground, structure=np.ones((3, 3), dtype=np.uint8))
    areas = np.bincount(labels.ravel())
    kept = sorted((i for i in range(1, count + 1) if areas[i] >= min_area),
                  key=lambda i: -int(areas[i]))
    return [labels == i for i in kept]


def experiment_plan() -> list[dict]:
    """Twenty-two settings fixed before the new inference run."""
    plan = []
    for confidence in BASE_THRESHOLDS:
        for area in (20, 100):
            plan.append({"name": f"base_c{confidence:.2f}_a{area}", "kind": "base",
                         "confidence": confidence, "vote_fraction": None, "min_area": area})
    for confidence in VOTE_THRESHOLDS:
        for fraction in (.5, .75):
            for area in (100, 200):
                plan.append({"name": f"vote_c{confidence:.2f}_v{fraction:.2f}_a{area}",
                             "kind": "binary_vote", "confidence": confidence,
                             "vote_fraction": fraction, "min_area": area})
    return plan


def cache_record(filename: str, view: str, instances: list[dict]) -> dict:
    records = []
    for index, instance in enumerate(instances, 1):
        mask = np.asarray(instance["mask"], dtype=np.uint8)
        if mask.ndim != 2:
            raise RunnerError("cache masks must be two dimensional")
        rle = mask_util.encode(np.asfortranarray(mask))
        records.append({"instance": index, "confidence": float(instance["confidence"]),
                        "area": int(mask_util.area(rle)),
                        "segmentation_rle": rle["counts"].decode("ascii")})
    return {"file_name": filename, "image_id": Path(filename).stem, "view": view,
            "model_imgsz": VIEW_SIZES[view], "coordinate_space": "original image",
            "shape": list(instances[0]["mask"].shape) if instances else [2048, 2048],
            "minimum_saved_confidence": RAW_MIN_CONFIDENCE, "instances": records}


def prepare(args):
    paths = {"annotations": Path(args.annotations), "checkpoint": Path(args.checkpoint),
             "train_split": Path(args.train_split), "val_split": Path(args.val_split)}
    for key, path in paths.items():
        if not path.is_file():
            raise RunnerError(f"missing {key}: {path}")
    destination = Path(args.output_dir)
    if destination.exists() or destination.is_symlink():
        raise RunnerError(f"choose a new output directory: {destination}")
    views = list(args.views)
    if len(set(views)) != len(views) or "base2048" not in views:
        raise RunnerError("views must be unique and include base2048")
    if not set(views) <= set(VIEW_SIZES):
        raise RunnerError("unrecognized inference view")
    if args.max_images is not None and args.max_images < 1:
        raise RunnerError("--max-images must be positive")
    if args.max_det < 1:
        raise RunnerError("--max-det must be positive")
    data = json.loads(paths["annotations"].read_text(encoding="utf-8-sig"))
    train = read_split(paths["train_split"], data["images"])
    val = read_split(paths["val_split"], data["images"])
    selected, selection = select_validation(data["images"], train, val)
    filenames = sorted({i["file_name"] for i in selected})
    full_count = len(filenames)
    if args.max_images is not None:
        filenames = filenames[:args.max_images]
    names = set(filenames)
    selected = [i for i in selected if i["file_name"] in names]
    ids = {i["id"] for i in selected}
    missing = [name for name in filenames if not (Path(args.images)/name).is_file()]
    if missing:
        raise RunnerError(f"missing {len(missing)} original JPEGs: {missing[:20]}")
    subset = dict(data, images=selected,
                  annotations=[a for a in data["annotations"] if a["image_id"] in ids])
    converted, _, conversion = convert_data(subset)
    if conversion["empty_annotator_image_groups"]:
        raise RunnerError("selected dataset has blank GT groups that the notebook cannot score")
    truth = []
    for filament_id, counts in converted:
        group = filament_id.split("_", 1)[0]
        image_id = group.split("-", 1)[1]
        area = int(mask_util.area({"size": [2048, 2048], "counts": counts.encode("ascii")}))
        truth.append(Filament(filament_id, image_id, group, counts, area))
    selection.update({"full_validation_images": full_count, "executed_images": len(filenames),
                      "executed_annotation_groups": len(ids),
                      "run_scope": "smoke subset, not full validation" if len(filenames) < full_count
                      else "fixed development validation", "filenames": filenames})
    return paths, destination, views, filenames, selection, truth


def run(args) -> dict:
    started = time.monotonic()
    paths, destination, views, filenames, selection, truth = prepare(args)
    plan = experiment_plan()
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination/"experiment_plan.json", {"settings": plan, "views": views,
               "raw_min_confidence": RAW_MIN_CONFIDENCE,
               "vote_definition": "Equal-weight binary foreground union per view; NOT calibrated probabilities or logits"})
    write_json(destination/"validation_selection.json", selection)
    write_csv(destination/"validation_gt.csv", truth)
    from checkpoint_utils import load_yolo_checkpoint
    import cv2
    model = load_yolo_checkpoint(paths["checkpoint"])
    predictions = {item["name"]: [] for item in plan}
    view_counts = defaultdict(int)
    capped = []
    with (destination/"raw_view_predictions.jsonl").open("x", encoding="utf-8") as cache:
        for index, filename in enumerate(filenames, 1):
            raw_image = cv2.imread(str(Path(args.images)/filename), cv2.IMREAD_GRAYSCALE)
            if raw_image is None:
                raise RunnerError(f"cannot read {filename}")
            processed = preprocess_limb_only(raw_image)
            base_foregrounds = {}
            vote_counts = {c: np.zeros(processed.shape, dtype=np.uint16) for c in VOTE_THRESHOLDS}
            for view in views:
                instances = infer_view(model, processed, view, args.device, args.max_det)
                view_counts[view] += len(instances)
                if len(instances) >= args.max_det:
                    capped.append({"file_name": filename, "view": view, "instances": len(instances)})
                cache.write(json.dumps(cache_record(filename, view, instances), allow_nan=False)+"\n")
                cache.flush()
                for confidence in VOTE_THRESHOLDS:
                    vote_counts[confidence] += view_foreground(instances, confidence, processed.shape)
                if view == "base2048":
                    base_foregrounds = {c: view_foreground(instances, c, processed.shape)
                                        for c in BASE_THRESHOLDS}
                print(f"[{index}/{len(filenames)}] {filename} {view}: {len(instances)} masks",
                      file=sys.stderr, flush=True)
            for item in plan:
                if item["kind"] == "base":
                    foreground = base_foregrounds[item["confidence"]]
                else:
                    count = math.ceil(item["vote_fraction"]*len(views))
                    foreground = vote_counts[item["confidence"]] >= count
                masks = component_masks(foreground, item["min_area"])
                predictions[item["name"]].extend(masks_to_filaments(Path(filename).stem, masks))
    results = []
    for item in plan:
        predicted = predictions[item["name"]]
        metrics = evaluate(truth, predicted, 2048, 2048)
        csv_name = item["name"]+".csv"
        write_csv(destination/csv_name, predicted)
        report = dict(item, metrics=metrics, prediction_rows=len(predicted),
                      overlap_policy="all candidate instances are disjoint 8-connected components")
        write_json(destination/(item["name"]+".metrics.json"), report)
        results.append(dict(item, pq=metrics["pq"], sq=metrics["sq"], rq=metrics["rq"],
                            tp_pairs=metrics["tp_pairs"], fp=metrics["fp"], fn=metrics["fn"],
                            unique_matched_gt=len(truth)-metrics["fn"],
                            extra_pairs_beyond_unique_gt=metrics["tp_pairs"]-(len(truth)-metrics["fn"]),
                            prediction_csv=csv_name))
        print(f"{item['name']}: PQ={metrics['pq']:.6f}", file=sys.stderr, flush=True)
    versions = {}
    for package in ["torch", "torchvision", "numpy", "scipy", "ultralytics", "opencv-python", "pycocotools"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    summary = {"status": "complete", "selection": selection,
               "sources": {key: {"filename": path.name, "sha256": sha256(path)} for key, path in paths.items()},
               "views": views, "forward_passes": len(filenames)*len(views),
               "raw_min_confidence": RAW_MIN_CONFIDENCE, "max_det": args.max_det,
               "view_instance_counts": dict(view_counts), "max_det_reached": capped,
               "device_requested": args.device, "versions": versions,
               "comparisons": sorted(results, key=lambda r: -r["pq"]),
               "best_development_setting": max(results, key=lambda r: r["pq"]),
               "elapsed_seconds": time.monotonic()-started,
               "limitations": ["Same-set development tuning; not an independent score or promised PQ>0.5.",
                               "View voting averages binary foreground unions, not mask logits or calibrated probabilities.",
                               "Identity and scale2560 are distinct views; this is a bounded four-view experiment, not full D4.",
                               "Connected components preserve no cross-component object identity.",
                               "No new training, external image download, or leaderboard submission is performed."]}
    write_json(destination/"summary.json", summary)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--train-split", required=True)
    parser.add_argument("--val-split", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--views", nargs="+", choices=tuple(VIEW_SIZES), default=list(DEFAULT_VIEWS))
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--max-images", type=int)
    args = parser.parse_args(argv)
    try:
        summary = run(args)
        print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError, ImportError, RuntimeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc),
                          "note": "Partial outputs are retained; choose a new output directory when retrying."},
                         ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
