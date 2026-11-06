"""Isolated YOLO-only validation export after explicitly authorized training.

The inference recipe is the deployed 4+8+18-view provider. This worker never
constructs a native/SMP model and validates the selected checkpoint SHA before
Ultralytics opens it. It writes a cache compatible with chase_best84_evaluation.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re

os.environ.setdefault('TORCH_CUDNN_V8_API_DISABLED', '1')

import cv2
import numpy as np
import torch

from submission_driver import (
    SPLIT_SHA256, atomic_json, compact, digest, read_envelope, valid_stage,
    write_envelope,
)
from submission_inference import InferenceProvider


def _read_json(path):
    raw = Path(path).read_bytes()
    return json.loads(gzip.decompress(raw) if raw[:2] == b'\x1f\x8b' else raw)


def _atomic_gzip(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.pending')
    try:
        with temporary.open('wb') as stream:
            stream.write(gzip.compress(compact(value).encode(), compresslevel=6, mtime=0))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def generate_yolo_cache(checkpoint, checkpoint_sha256, cfg, output, *, provider_factory=InferenceProvider):
    """Freeze one explicitly selected trained YOLO checkpoint on fixed val91."""
    checkpoint, output = Path(checkpoint), Path(output)
    if not isinstance(checkpoint_sha256, str) or not re.fullmatch('[0-9a-f]{64}', checkpoint_sha256):
        raise ValueError('An explicit trained checkpoint SHA256 is required')
    if digest(checkpoint) != checkpoint_sha256:
        raise ValueError('Selected trained YOLO checkpoint SHA256 differs')
    if digest(cfg['val_split']) != SPLIT_SHA256['val_split']:
        raise ValueError('Frozen validation split changed')
    with Path(cfg['val_split']).open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    names = sorted({row['file_name'] for row in rows})
    if len(names) != 91 or len({row['image_id'] for row in rows}) != 101:
        raise ValueError('Expected frozen validation91/annotator101')
    old_value = cfg.get('baseline_yolo_cache') or cfg.get('yolo_cache')
    if old_value is None:
        raise ValueError('The pinned baseline YOLO cache is required')
    old_path = Path(old_value)
    if output.resolve() == old_path.resolve():
        raise ValueError('Preserve the baseline YOLO cache; select a new output path')
    old = _read_json(old_path)
    if old.get('schema_version') != 1 or set(old.get('frames', {})) != {Path(name).stem for name in names}:
        raise ValueError('Baseline cache validation image set differs')
    hashes = {}
    for name in names:
        if Path(name).name != name:
            raise ValueError('Validation names must be plain filenames')
        expected = old.get('image_sha256', {}).get(name, old.get('image_sha256', {}).get(Path(name).stem))
        if not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected):
            raise ValueError(f'Baseline original JPEG hash missing: {name}')
        if digest(Path(cfg['images']) / name) != expected:
            raise ValueError(f'Original validation JPEG changed: {name}')
        hashes[name] = expected
    device = torch.device(cfg.get('device', 'cuda'))
    runtime = dict(torch=torch.__version__, torch_cuda=torch.version.cuda,
                   cudnn=torch.backends.cudnn.version(), device=str(device),
                   numpy=np.__version__, opencv=cv2.__version__)
    try:
        runtime['ultralytics'] = importlib.metadata.version('ultralytics')
    except importlib.metadata.PackageNotFoundError:
        if provider_factory is InferenceProvider:
            raise
        runtime['ultralytics'] = 'test provider'
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('Select a GPU runtime for YOLO validation')
        runtime['gpu'] = torch.cuda.get_device_name(device)
    code = {name: digest(Path(__file__).with_name(name)) for name in
            ('chase_yolo_predict.py', 'submission_inference.py', 'submission_driver.py')}
    protocol = dict(schema_version=1, checkpoint_sha256=checkpoint_sha256,
                    source_images=hashes, baseline_cache_sha256=digest(old_path),
                    val_split_sha256=SPLIT_SHA256['val_split'], runtime=runtime,
                    sources=code, recipe='unchanged original YOLO geometric4/preprocessing8/tiled18',
                    role='explicitly selected new YOLO checkpoint; native not loaded')
    identity = hashlib.sha256(compact(protocol).encode()).hexdigest()
    stages_root = output.parent / (output.name + '.stages')
    stages_root.mkdir(parents=True, exist_ok=True)
    identity_path = stages_root / 'identity.json'
    if identity_path.is_file() and _read_json(identity_path).get('identity') != identity:
        raise ValueError('YOLO checkpoint/inputs/code/runtime changed; preserve this cache and choose a new output')
    if not identity_path.is_file() and any(stages_root.glob('*.gz')):
        raise ValueError('Unidentified YOLO stage files preserved; choose a new output')
    if output.is_file():
        try:
            existing = _read_json(output)
        except (OSError, ValueError, EOFError):
            if not identity_path.is_file():
                raise ValueError('Unidentified damaged output preserved; choose a new output')
        else:
            if existing.get('identity') != identity:
                raise ValueError('Existing YOLO output belongs to different inputs; preserve it and choose a new output')
    atomic_json(identity_path, dict(identity=identity, protocol=protocol))
    provider = provider_factory(checkpoint, None, device=str(device), verify_weights=False)
    frames = {}
    for number, name in enumerate(names, 1):
        stem = Path(name).stem
        frame = {}
        pixels = None
        for family in ('geometric', 'preprocessing', 'tiled'):
            cache = stages_root / f'{stem}.{family}.json.gz'
            stage = read_envelope(cache, identity, hashes[name])
            if not valid_stage(family, stage, stem):
                if pixels is None:
                    raw = (Path(cfg['images']) / name).read_bytes()
                    if hashlib.sha256(raw).hexdigest() != hashes[name]:
                        raise ValueError(f'Original JPEG changed during prediction: {name}')
                    pixels = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
                    if pixels is None or pixels.shape != (2048, 2048) or pixels.dtype != np.uint8:
                        raise ValueError('Expected uint8 2048-square validation JPEG')
                stage = getattr(provider, family)(stem, pixels)
                if not valid_stage(family, stage, stem):
                    raise ValueError(f'Invalid YOLO output: {stem}/{family}')
                write_envelope(cache, stage, identity, hashes[name])
            frame[family] = stage
        frames[stem] = frame
        print(f'New YOLO full validation views [{number}/{len(names)}] {name}', flush=True)
    result = dict(schema_version=1, frames=frames, image_sha256=hashes,
                  yolo_checkpoint_sha256=checkpoint_sha256, identity=identity,
                  sources=dict(checkpoint=dict(path=str(checkpoint), sha256=checkpoint_sha256),
                               code=code, baseline_cache=dict(path=str(old_path), sha256=digest(old_path))),
                  protocol=protocol)
    _atomic_gzip(output, result)
    summary = dict(path=str(output), sha256=digest(output), identity=identity,
                   yolo_checkpoint_sha256=checkpoint_sha256, images=len(names),
                   views_per_image=dict(geometric=4, preprocessing=8, tiled=18))
    atomic_json(output.with_name(output.name + '.summary.json'), summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    cfg = json.loads(Path(args.config).read_text())
    result = generate_yolo_cache(args.checkpoint, args.checkpoint_sha256, cfg, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
