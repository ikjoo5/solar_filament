#!/usr/bin/env python3
"""Evaluate the uploaded YOLO baseline where the original JPEGs already live.

No images are uploaded or downloaded by this script. A train complement is an
explicit candidate validation set, not proof of the checkpoint's original split.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from functools import lru_cache
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np
from pycocotools import mask as mask_util
from scipy import ndimage
from scipy.interpolate import UnivariateSpline

from convert_annotations import AnnotationError, convert_data
from filament_eval import Filament, ValidationError, audit, evaluate


class RunnerError(ValueError):
    """The requested local validation run cannot be performed faithfully."""


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_split(path: str | Path, images: list[dict]) -> list[str]:
    """Validate both annotation IDs and their physical filenames."""
    by_id = {image["id"]: image for image in images}
    if len(by_id) != len(images):
        raise RunnerError("annotation JSON has duplicate image IDs")
    selected = []
    seen = set()
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"image_id", "file_name"}.issubset(reader.fieldnames):
            raise RunnerError("split CSV needs image_id,file_name columns")
        for line, row in enumerate(reader, 2):
            if None in row or row.get("image_id") is None or row.get("file_name") is None:
                raise RunnerError(f"split row {line}: missing/extra CSV fields")
            image_id, filename = row["image_id"], row["file_name"]
            if image_id in seen:
                raise RunnerError(f"split row {line}: duplicate image_id {image_id}")
            if image_id not in by_id:
                raise RunnerError(f"split row {line}: unknown image_id {image_id}")
            if filename != by_id[image_id]["file_name"]:
                raise RunnerError(f"split row {line}: file_name does not match JSON for {image_id}")
            seen.add(image_id)
            selected.append(image_id)
    return selected


def select_validation(images: list[dict], train_ids: list[str],
                      val_ids: list[str] | None = None,
                      use_train_complement: bool = False) -> tuple[list[dict], dict]:
    """Never silently assume that a train complement is the original validation."""
    by_id = {image["id"]: image for image in images}
    if len(by_id) != len(images):
        raise RunnerError("duplicate annotation image IDs")
    if len(set(train_ids)) != len(train_ids) or not set(train_ids) <= set(by_id):
        raise RunnerError("train split contains duplicate or unknown IDs")
    if val_ids is None:
        if not use_train_complement:
            raise RunnerError("supply --val-split or explicitly select --use-train-complement")
        val_ids = [image["id"] for image in images if image["id"] not in set(train_ids)]
        provenance = "explicit train-complement candidate; checkpoint original validation not verified"
    else:
        if use_train_complement:
            raise RunnerError("--val-split and --use-train-complement are mutually exclusive")
        provenance = "user-supplied validation split; checkpoint original split not independently verified"
    if not val_ids:
        raise RunnerError("validation selection is empty")
    if len(set(val_ids)) != len(val_ids) or not set(val_ids) <= set(by_id):
        raise RunnerError("validation split contains duplicate or unknown IDs")
    if set(train_ids) & set(val_ids):
        raise RunnerError("train and validation share annotation image IDs")
    train_files = {by_id[i]["file_name"] for i in train_ids}
    val_files = {by_id[i]["file_name"] for i in val_ids}
    if train_files & val_files:
        raise RunnerError("train and validation share physical image filenames (annotator leakage)")
    selected = [by_id[i] for i in val_ids]
    # Every available annotator for a selected physical image is required.
    expected_ids = {image["id"] for image in images if image["file_name"] in val_files}
    if expected_ids != set(val_ids):
        raise RunnerError("validation omits annotator variants for selected physical images")
    metadata = {"provenance": provenance, "train_annotation_groups": len(train_ids),
                "train_physical_images": len(train_files),
                "selected_annotation_groups": len(selected),
                "selected_physical_images": len(val_files),
                "unselected_annotation_groups": len(images) - len(set(train_ids) | set(val_ids)),
                "physical_train_validation_overlap": 0,
                "warning": "This is development-set PQ, not an independent reproduction of Kaggle LB 0.37."}
    return selected, metadata


@lru_cache(maxsize=1)
def _limb_geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.ogrid[:2048, :2048]
    r_norm = (np.sqrt((xx - 1024)**2 + (yy - 1024)**2) / 900).astype(np.float32)
    disk = r_norm <= 1.0
    disk_radii = r_norm[disk]
    bins = np.minimum((disk_radii * 100).astype(np.int32), 99)
    radii = (np.arange(100) + 0.5) / 100
    return disk, disk_radii, bins, radii


def preprocess_limb_only(image: np.ndarray) -> np.ndarray:
    """Exact provided baseline settings: center1024/r900/100 bins/s400/gain4."""
    if image.dtype != np.uint8 or image.shape != (2048, 2048):
        raise RunnerError(f"expected original uint8 (2048,2048), got {image.shape}/{image.dtype}")
    disk, disk_radii, bins, radii = _limb_geometry()
    pixels = image[disk].astype(np.uint8)
    hist = np.bincount(bins * 256 + pixels.astype(np.int32), minlength=100 * 256).reshape(100, 256)
    counts = hist.sum(axis=1)
    if np.any(counts == 0):
        raise RunnerError("empty radial bin; image geometry differs from baseline")
    cum = hist.cumsum(axis=1)
    medians = np.argmax(cum >= ((counts + 1) // 2)[:, None], axis=1).astype(np.float64)
    spline = UnivariateSpline(radii, medians, k=3, s=400)
    fitted_profile = spline(radii)
    reference = float(np.median(fitted_profile[radii <= 0.15]))
    gain = np.clip(reference / np.maximum(spline(disk_radii), 1.0), 0.25, 4.0)
    corrected = np.zeros(image.shape, dtype=np.uint8)
    corrected[disk] = np.clip(image[disk].astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return corrected


def postprocess_instances(instances: list[dict], threshold: float, mode: str,
                          min_area: int = 20) -> list[np.ndarray]:
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise RunnerError("confidence threshold must be finite in [0,1]")
    if min_area < 0 or mode not in {"union_cc", "preserve"}:
        raise RunnerError("invalid area threshold or postprocessing mode")
    selected = []
    shape = None
    for instance in instances:
        conf = float(instance["confidence"])
        mask = np.asarray(instance["mask"], dtype=bool)
        if mask.ndim != 2 or not np.isfinite(conf):
            raise RunnerError("predicted mask must be two-dimensional with finite confidence")
        if shape is not None and shape != mask.shape:
            raise RunnerError("predicted mask shapes differ")
        shape = mask.shape
        if conf >= threshold:
            selected.append(mask)
    if not selected:
        return []
    if mode == "preserve":
        # Keep disconnected pieces in one model instance. Do not silently
        # change overlapping masks; audit those before any final submission.
        return [mask for mask in selected if mask.any() and int(mask.sum()) >= min_area]
    union = np.zeros_like(selected[0], dtype=bool)
    for mask in selected:
        union |= mask
    labels, count = ndimage.label(union, structure=np.ones((3, 3), dtype=np.uint8))
    areas = np.bincount(labels.ravel())
    order = sorted((i for i in range(1, count + 1) if areas[i] >= min_area),
                   key=lambda i: -int(areas[i]))
    return [labels == i for i in order]


def infer_instances(model, image: np.ndarray, min_confidence: float = .15,
                    device: str = "cpu", imgsz: int = 2048, max_det: int = 100) -> list[dict]:
    import cv2

    if image.ndim != 2 or image.dtype != np.uint8:
        raise RunnerError("inference input must be a grayscale uint8 image")
    result = model.predict(source=cv2.cvtColor(image, cv2.COLOR_GRAY2BGR),
                           imgsz=imgsz, conf=min_confidence, iou=.70,
                           retina_masks=True, device=device, max_det=max_det, verbose=False)[0]
    if result.masks is None or result.boxes is None:
        return []
    masks = result.masks.data.detach().float().cpu().numpy()
    confidences = result.boxes.conf.detach().float().cpu().numpy()
    if len(masks) != len(confidences):
        raise RunnerError("YOLO returned different mask/confidence counts")
    instances = []
    for mask, confidence in zip(masks, confidences):
        if mask.shape != image.shape:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        instances.append({"mask": mask >= .5, "confidence": float(confidence)})
    return instances


def masks_to_filaments(image_id: str, masks: list[np.ndarray]) -> list[Filament]:
    result = []
    for index, mask in enumerate(masks, 1):
        counts = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("ascii")
        result.append(Filament(f"{image_id}_{index}", image_id, image_id, counts, int(mask.sum())))
    return result


def write_csv(path: Path, rows: list[Filament]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filament_id", "segmentation_rle"])
        writer.writerows((row.filament_id, row.counts) for row in rows)


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def run(args) -> dict:
    started = time.monotonic()
    paths = {"annotations": Path(args.annotations), "train_split": Path(args.train_split),
             "checkpoint": Path(args.checkpoint)}
    if args.val_split:
        paths["val_split"] = Path(args.val_split)
    for kind, path in paths.items():
        if not path.is_file():
            raise RunnerError(f"missing {kind}: {path}")
    destination = Path(args.output_dir)
    if destination.exists() or destination.is_symlink():
        raise RunnerError(f"output directory already exists; choose a new one: {destination}")
    if not args.confidences or any(not np.isfinite(v) or not 0 <= v <= 1 for v in args.confidences):
        raise RunnerError("confidences must be finite values in [0,1]")
    confidences = sorted(set(float(c) for c in args.confidences))
    if len({f"{c:.8f}" for c in confidences}) != len(confidences):
        raise RunnerError("confidence values are too close to distinguish in output filenames")
    if args.max_images is not None and args.max_images <= 0:
        raise RunnerError("--max-images must be positive")
    if args.min_area < 0 or args.max_det <= 0:
        raise RunnerError("invalid minimum area or max_det")
    data = json.loads(paths["annotations"].read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or not isinstance(data.get("images"), list):
        raise RunnerError("annotation JSON needs images records")
    train_ids = read_split(paths["train_split"], data["images"])
    val_ids = read_split(paths["val_split"], data["images"]) if "val_split" in paths else None
    selected, selection = select_validation(data["images"], train_ids, val_ids, args.use_train_complement)
    files = sorted({image["file_name"] for image in selected})
    original_selection_count = len(files)
    if args.max_images is not None:
        files = files[:args.max_images]
    files_set = set(files)
    selected = [image for image in selected if image["file_name"] in files_set]
    selected_ids = {image["id"] for image in selected}
    images_root = Path(args.images)
    missing = [name for name in files if not (images_root / name).is_file()]
    if missing:
        raise RunnerError(f"{len(missing)} required original JPEGs are missing from {images_root}: {missing[:20]}")
    subset = dict(data)
    subset["images"] = selected
    subset["annotations"] = [a for a in data["annotations"] if a["image_id"] in selected_ids]
    gt_rows, manifest, conversion = convert_data(subset)
    gt = []
    for filament_id, counts in gt_rows:
        group = filament_id.split("_", 1)[0]
        image_id = group.split("-", 1)[1]
        area = int(mask_util.area({"size": [2048, 2048], "counts": counts.encode("ascii")}))
        gt.append(Filament(filament_id, image_id, group, counts, area))
    if conversion["empty_annotator_image_groups"]:
        raise RunnerError("selected GT has blank annotation groups the notebook cannot score")
    run_scope = "smoke subset, not full validation" if len(files) < original_selection_count else "selected development validation"
    selection.update({"run_scope": run_scope, "available_validation_images": original_selection_count,
                      "executed_physical_images": len(files), "executed_annotation_groups": len(selected),
                      "filenames": files, "annotation_image_ids": sorted(selected_ids)})
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "validation_selection.json", selection)
    write_csv(destination / "validation_gt.csv", gt)
    write_json(destination / "gt_conversion_summary.json", conversion)
    # Heavy imports happen after all file/split checks. The loader uses the
    # reviewed architecture allowlist and never generic untrusted pickle load.
    import cv2
    from checkpoint_utils import load_yolo_checkpoint

    model = load_yolo_checkpoint(paths["checkpoint"])
    modes = ["union_cc", "preserve"] if args.postprocess == "both" else [args.postprocess]
    predictions = {(mode, conf): [] for mode in modes for conf in confidences}
    for index, filename in enumerate(files, 1):
        image = cv2.imread(str(images_root / filename), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RunnerError(f"OpenCV could not read {filename}")
        if image.shape != (2048, 2048) or image.dtype != np.uint8:
            raise RunnerError(f"{filename}: expected original uint8 2048x2048, got {image.shape}/{image.dtype}")
        processed = preprocess_limb_only(image) if args.preprocessing == "limb_only" else image
        raw = infer_instances(model, processed, min(confidences), args.device, 2048, args.max_det)
        for mode in modes:
            for conf in confidences:
                masks = postprocess_instances(raw, conf, mode, args.min_area)
                predictions[(mode, conf)].extend(masks_to_filaments(Path(filename).stem, masks))
        print(f"[{index}/{len(files)}] {filename}: raw_instances={len(raw)}", file=sys.stderr, flush=True)
    records = []
    for (mode, confidence), predicted in predictions.items():
        stem = f"{mode}_conf{confidence:.8f}".rstrip("0").rstrip(".")
        write_csv(destination / f"{stem}.csv", predicted)
        metrics = evaluate(gt, predicted, 2048, 2048)
        mask_audit = audit(predicted, 2048, 2048)
        report = {"postprocess": mode, "confidence": confidence,
                  "run_scope": run_scope, "metrics": metrics, "prediction_audit": mask_audit}
        if mode == "preserve":
            report["warning"] = "Preserved model instances can overlap; compare diagnostically and verify competition policy before submission."
        write_json(destination / f"{stem}.metrics.json", report)
        records.append({"postprocess": mode, "confidence": confidence,
                        "pq": metrics["pq"], "sq": metrics["sq"], "rq": metrics["rq"],
                        "tp_pairs": metrics["tp_pairs"], "fp": metrics["fp"], "fn": metrics["fn"],
                        "overlap_groups": mask_audit["groups_with_overlap"],
                        "prediction_csv": f"{stem}.csv"})
    versions = {}
    for package in ("numpy", "scipy", "pycocotools", "torch", "ultralytics", "opencv-python", "opencv-python-headless"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    summary = {"status": "complete", "run_scope": run_scope, "selection": selection,
               "sources": {kind: {"filename": path.name, "sha256": sha256(path)} for kind, path in paths.items()},
               "versions": versions, "device_requested": args.device,
               "settings": {"image_size": 2048, "max_det": args.max_det,
                            "min_area": args.min_area, "preprocessing": args.preprocessing,
                            "limb_geometry": {"cx": 1024, "cy": 1024, "radius": 900,
                                              "bins": 100, "spline_s": 400, "max_gain": 4.0},
                            "confidences": confidences, "postprocess_modes": modes},
               "comparisons": sorted(records, key=lambda row: -row["pq"]),
               "best_development_setting": max(records, key=lambda row: row["pq"]),
               "elapsed_seconds": time.monotonic() - started,
               "limitations": ["This is checkpoint inference on a declared development split, not new model training or Kaggle submission.",
                               "Confidence/postprocessing selection on this same set is tuning, not an unbiased final score.",
                               "Train-complement selection alone does not establish the original validation or synthetic-data provenance.",
                               "The supplied evaluator uses global all-pairs strict IoU>0.5; it does not enforce one-to-one matching."]}
    write_json(destination / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--train-split", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--val-split")
    selection.add_argument("--use-train-complement", action="store_true")
    parser.add_argument("--images", required=True, help="existing directory containing original JPEG filenames")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True, help="new directory; existing outputs are never overwritten")
    parser.add_argument("--confidences", type=float, nargs="+", default=[.15, .25, .35, .45, .55])
    parser.add_argument("--postprocess", choices=["union_cc", "preserve", "both"], default="both")
    parser.add_argument("--preprocessing", choices=["limb_only", "raw"], default="limb_only")
    parser.add_argument("--min-area", type=int, default=20)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--max-images", type=int, help="deterministic limited smoke test, not a full validation claim")
    parser.add_argument("--device", default="0", help="Ultralytics device: 0 for first GPU, cpu for CPU smoke test")
    args = parser.parse_args(argv)
    try:
        summary = run(args)
        print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (RunnerError, AnnotationError, ValidationError, OSError, ImportError,
            ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc),
                          "note": "Any partial output directory is retained; choose a new directory after fixing the error."},
                         ensure_ascii=False, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
