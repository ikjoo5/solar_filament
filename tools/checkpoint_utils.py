"""Restricted checkpoint loading for the supplied YOLO26 segmentation baseline.

The checkpoint contains a serialized model, not just a tensor state_dict.
Only known PyTorch/Ultralytics layer classes are allowlisted and loading stays
weights_only=True. Unknown globals fail instead of falling back to pickle.
"""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import tempfile


ALLOWED_GLOBALS = frozenset({
    "torch.nn.modules.container.ModuleList",
    "torch.nn.modules.container.Sequential",
    "torch.nn.modules.activation.SiLU",
    "torch.nn.modules.upsampling.Upsample",
    "torch.nn.modules.linear.Identity",
    "torch.nn.modules.pooling.MaxPool2d",
    "torch.nn.modules.conv.Conv2d",
    "torch.nn.modules.conv.ConvTranspose2d",
    "torch.nn.modules.batchnorm.BatchNorm2d",
    "ultralytics.nn.tasks.SegmentationModel",
    "ultralytics.nn.modules.conv.Conv",
    "ultralytics.nn.modules.conv.Concat",
    "ultralytics.nn.modules.conv.DWConv",
    "ultralytics.nn.modules.block.C3k",
    "ultralytics.nn.modules.block.C3k2",
    "ultralytics.nn.modules.block.Bottleneck",
    "ultralytics.nn.modules.block.Attention",
    "ultralytics.nn.modules.block.C2PSA",
    "ultralytics.nn.modules.block.PSABlock",
    "ultralytics.nn.modules.block.SPPF",
    "ultralytics.nn.modules.block.Proto26",
    "ultralytics.nn.modules.head.Segment26",
})


def validate_checkpoint_globals(names) -> list[str]:
    names = sorted(set(names))
    unknown = sorted(set(names) - ALLOWED_GLOBALS)
    if unknown:
        raise ValueError(f"Checkpoint has unreviewed globals: {unknown}")
    return names


def _load_restricted(path: str | Path) -> dict:
    import torch

    names = validate_checkpoint_globals(torch.serialization.get_unsafe_globals_in_checkpoint(path))
    classes = []
    for name in names:
        module, _, attribute = name.rpartition(".")
        classes.append(getattr(importlib.import_module(module), attribute))
    with torch.serialization.safe_globals(classes):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a checkpoint dictionary")
    return checkpoint


def checkpoint_metadata(path: str | Path, checkpoint: dict | None = None) -> dict:
    if checkpoint is None:
        checkpoint = _load_restricted(path)
    with Path(path).open("rb") as handle:
        checksum = hashlib.file_digest(handle, "sha256").hexdigest()
    args = checkpoint.get("train_args", {}) or {}
    selected = {k: args[k] for k in (
        "model", "task", "epochs", "patience", "batch", "imgsz", "mask_ratio",
        "overlap_mask", "optimizer", "lr0", "seed", "max_det", "mosaic", "mixup",
        "copy_paste", "degrees", "fliplr", "flipud", "amp", "data", "name",
    ) if k in args}
    return {"checkpoint_sha256": checksum, "checkpoint_bytes": Path(path).stat().st_size,
            "ultralytics_saved_version": checkpoint.get("version"),
            "saved_date": checkpoint.get("date"), "epoch_field": checkpoint.get("epoch"),
            "train_args": selected,
            "loading": "torch weights_only=True with reviewed layer-class allowlist"}


def load_yolo_checkpoint(path: str | Path):
    """Return an inference wrapper around the exact stored model parameters.

    A YAML-only model initializes the official wrapper; it is then replaced by
    the restricted-loaded model. No checkpoint is passed to Ultralytics' pickle
    loader, and no pretrained weights are downloaded.
    """
    import torch
    import yaml
    from ultralytics import YOLO
    from ultralytics.nn.tasks import SegmentationModel

    checkpoint = _load_restricted(path)
    stored = checkpoint.get("ema")
    if stored is None:
        stored = checkpoint.get("model")
    if not isinstance(stored, SegmentationModel):
        raise ValueError("Expected an Ultralytics SegmentationModel")
    if not isinstance(stored.yaml, dict):
        raise ValueError("Checkpoint model is missing its architecture YAML")
    with tempfile.TemporaryDirectory(prefix="filament_model_") as folder:
        cfg = Path(folder) / "yolo26s-seg.yaml"
        cfg.write_text(yaml.safe_dump(stored.yaml), encoding="utf-8")
        wrapper = YOLO(str(cfg), task="segment", verbose=False)
    wrapper.model = stored.float().eval()
    wrapper.task = "segment"
    wrapper.ckpt_path = str(Path(path).resolve())
    wrapper.overrides.update({"model": wrapper.ckpt_path, "task": "segment"})
    wrapper.model.pt_path = wrapper.ckpt_path
    wrapper.checkpoint_report = checkpoint_metadata(path, checkpoint)
    wrapper.checkpoint_report["parameters"] = sum(p.numel() for p in wrapper.model.parameters())
    return wrapper
