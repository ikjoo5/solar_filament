"""Same-run full YOLO+native validation for CHASE transfer experiments.

YOLO predictions and the deployed finalizer are frozen. Only the injected native
model changes. Validation annotations are used only by the final score function.
Per-image caches bind model tensors, inputs, code and runtime, and survive an
interrupted evaluation without changing the completed training checkpoint.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re

import cv2
import numpy as np
from pycocotools import mask as mu
import torch

from submission_driver import (
    SPLIT_SHA256, atomic_json, checked_rle, compact, digest, export_csv,
    finalize_frame, json_ready, read_envelope, valid_stage,
    validate_predictions, write_envelope,
)
from submission_inference import InferenceProvider

ANNOTATIONS_SHA256 = '5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1'
EVALUATION_SOURCES = (
    'chase_best84_evaluation.py', 'submission_inference.py',
    'submission_driver.py', 'submission_yolo_recipe.py',
    'submission_native_recipe.py',
)


def fingerprint(value):
    return hashlib.sha256(compact(value).encode()).hexdigest()


def model_digest(model):
    """Hash tensor content, including dtype/shape, without changing parameters."""
    value = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError('Only tensor model states are supported')
        cpu = tensor.detach().cpu().contiguous()
        value.update(compact([name, str(cpu.dtype), list(cpu.shape)]).encode())
        value.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
    return value.hexdigest()


def runtime_identity(device):
    target = torch.device(device)
    result = dict(device=str(target), torch=torch.__version__,
                  torch_cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                  numpy=np.__version__, opencv=cv2.__version__)
    for name in ('scipy', 'pycocotools', 'segmentation-models-pytorch', 'timm'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = 'not installed'
    if target.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('Select a Colab GPU before evaluation')
        index = target.index if target.index is not None else torch.cuda.current_device()
        result.update(gpu=torch.cuda.get_device_name(index),
                      capability=list(torch.cuda.get_device_capability(index)),
                      bf16=bool(torch.cuda.is_bf16_supported()))
    return result


def load_cohort(cfg):
    """Read the frozen physical split and retain all annotator GT groups."""
    paths = {name: Path(cfg[name]) for name in ('annotations', 'train_split', 'val_split')}
    expected = dict(SPLIT_SHA256, annotations=ANNOTATIONS_SHA256)
    for name, path in paths.items():
        if digest(path) != expected[name]:
            raise ValueError(f'Frozen {name} differs: {path}')
    annotation = json.loads(paths['annotations'].read_text())
    images = {str(row['id']): row for row in annotation['images']}
    if len(images) != len(annotation['images']):
        raise ValueError('Duplicate annotation image IDs')

    def read(name):
        with paths[name].open(newline='', encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        ids = {row['image_id'] for row in rows}
        if len(ids) != len(rows) or not ids <= images.keys():
            raise ValueError('Unknown or duplicate split image IDs')
        for row in rows:
            filename = row['file_name']
            if (Path(filename).name != filename or
                    images[row['image_id']]['file_name'] != filename):
                raise ValueError('Split filename differs from annotations')
        return ids, {row['file_name'] for row in rows}

    train_ids, train_names = read('train_split')
    val_ids, val_names = read('val_split')
    if train_ids & val_ids or train_names & val_names:
        raise ValueError('Train and validation overlap')
    if (len(train_names), len(val_names), len(val_ids)) != (616, 91, 101):
        raise ValueError('Expected frozen train616/validation91/annotator101 split')
    if val_ids != {key for key, row in images.items() if row['file_name'] in val_names}:
        raise ValueError('Validation annotator variants were omitted')
    groups = {key: dict(image=Path(images[key]['file_name']).stem, masks=[])
              for key in sorted(val_ids)}
    for ann in annotation['annotations']:
        key = str(ann['image_id'])
        if key not in groups:
            continue
        encoded = mu.merge(mu.frPyObjects(ann['segmentation'], 2048, 2048))
        groups[key]['masks'].append(checked_rle(encoded))
    if sum(len(row['masks']) for row in groups.values()) != 708:
        raise ValueError('Expected 708 frozen validation GT instances')
    if any(not row['masks'] for row in groups.values()):
        raise ValueError('Missing frozen validation annotations')
    return dict(names=sorted(val_names), groups=groups, source_sha256=expected)


def load_yolo_cache(path, names):
    """Require 4+8+18 views and original JPEG hashes for every val image."""
    path = Path(path)
    raw = path.read_bytes()
    if raw[:2] == b'\x1f\x8b':
        raw = gzip.decompress(raw)
    data = json.loads(raw)
    if data.get('schema_version') != 1:
        raise ValueError('Unsupported frozen YOLO cache schema')
    frames, hashes = data.get('frames'), data.get('image_sha256')
    stems = {Path(name).stem for name in names}
    if not isinstance(frames, dict) or set(frames) != stems:
        raise ValueError('Frozen YOLO cache must exactly cover validation images')
    if not isinstance(hashes, dict):
        raise ValueError('Frozen YOLO cache requires original JPEG SHA256 hashes')
    canonical_hashes = {}
    for filename in names:
        stem = Path(filename).stem
        file_hash, stem_hash = hashes.get(filename), hashes.get(stem)
        if file_hash is not None and stem_hash is not None and file_hash != stem_hash:
            raise ValueError('Conflicting original image hashes')
        image_sha = file_hash if file_hash is not None else stem_hash
        if not isinstance(image_sha, str) or not re.fullmatch('[0-9a-f]{64}', image_sha):
            raise ValueError(f'Original JPEG hash missing: {filename}')
        canonical_hashes[filename] = image_sha
        frame = frames[stem]
        if not isinstance(frame, dict) or set(frame) != {'geometric', 'preprocessing', 'tiled'}:
            raise ValueError('Frozen cache contains unexpected stages')
        for family in ('geometric', 'preprocessing', 'tiled'):
            if not valid_stage(family, frame[family], stem):
                raise ValueError(f'Invalid frozen YOLO views: {stem}/{family}')
            for record in frame[family]:
                if record.get('shape') != [2048, 2048] or record.get('coordinate_space') != 'original':
                    raise ValueError('Frozen YOLO masks must be in original coordinates')
                for proposal in record['instances']:
                    confidence = proposal['confidence']
                    if not isinstance(confidence, (int, float)) or not np.isfinite(confidence) or not 0 <= confidence <= 1:
                        raise ValueError('Invalid frozen confidence')
                    checked_rle(dict(size=[2048, 2048], counts=proposal['rle']))
    if not isinstance(data.get('sources'), dict) or not data['sources']:
        raise ValueError('Frozen cache source provenance is required')
    return dict(frames=frames, image_sha256=canonical_hashes,
                sources=data['sources'], sha256=digest(path))


def score_predictions(groups, predicted):
    """Official notebook's global all-pairs IoU>.5, per annotator group."""
    known = {row['image'] for row in groups.values()}
    if set(predicted) != known:
        raise ValueError('Predictions must explicitly include every validation image, including blanks')
    tp = fp = fn = 0
    total_iou = 0.0
    for group in groups.values():
        truth, prediction = group['masks'], predicted[group['image']]
        if not truth:
            fp += len(prediction)
            continue
        if not prediction:
            fn += len(truth)
            continue
        iou = np.asarray(mu.iou(truth, prediction, [0] * len(prediction)))
        hits = iou > .5
        tp += int(hits.sum())
        fp += int((~hits.any(axis=0)).sum())
        fn += int((~hits.any(axis=1)).sum())
        total_iou += float(iou[hits].sum())
    denominator = tp + .5 * fp + .5 * fn
    return dict(pq=total_iou / denominator if denominator else 0.,
                sq=total_iou / tp if tp else 0.,
                rq=tp / denominator if denominator else 0.,
                tp_pairs=tp, fp=fp, fn=fn, matched_iou_sum=total_iou)


def _native_valid(value, stem, filename, image_sha, identity):
    return (valid_stage('native', value, stem) and value.get('file_name') == filename
            and value.get('image_sha256') == image_sha
            and value.get('identity') == identity and value.get('size') == [2048, 2048])


def evaluate_candidate(model, cfg, label, *, provider_factory=InferenceProvider):
    """Evaluate only on fixed validation; epoch000 is the same-runtime control."""
    if not isinstance(label, str) or not re.fullmatch('[A-Za-z0-9_-]+', label):
        raise ValueError('Evaluation label must be a safe single path component')
    cohort = load_cohort(cfg)
    yolo = load_yolo_cache(cfg['yolo_cache'], cohort['names'])
    root = Path(cfg['output_root']) / 'validation' / label
    root.mkdir(parents=True, exist_ok=True)
    (root / 'completed').mkdir(exist_ok=True)
    device = cfg.get('device', 'cuda')
    native_batch = int(cfg.get('native_batch', 1))
    if native_batch < 1:
        raise ValueError('native_batch must be positive')
    model_sha = model_digest(model)
    for filename, expected in yolo['image_sha256'].items():
        if digest(Path(cfg['images']) / filename) != expected:
            raise ValueError(f'Validation image differs from frozen YOLO input: {filename}')
    protocol = dict(schema_version=1, model_sha256=model_sha,
        data_sha256=cohort['source_sha256'], yolo_cache_sha256=yolo['sha256'],
        source_images=yolo['image_sha256'],
        sources={name: digest(Path(__file__).with_name(name)) for name in EVALUATION_SOURCES},
        runtime=runtime_identity(device), native_batch=native_batch,
        recipe='frozen YOLO4+8+18 + injected best84 architecture; unchanged finalize_frame; no R1',
        metric='global all-pairs IoU strictly >0.5; 91 physical images/101 annotator groups/708 GT',
        comparison_role=cfg.get('comparison_role', 'old_yolo_candidate_native'))
    identity = fingerprint(protocol)
    # The same native outputs serve both old/new YOLO combinations. Binding to
    # a YOLO cache would unnecessarily repeat the expensive four-view inference.
    native_protocol = {key: protocol[key] for key in
        ('model_sha256', 'source_images', 'sources', 'runtime', 'native_batch')}
    native_identity = fingerprint(native_protocol)
    native_cache = Path(cfg.get('native_validation_cache_root', Path(cfg['output_root']) / 'native_validation_cache'))
    native_root = native_cache / native_identity
    native_root.mkdir(parents=True, exist_ok=True)
    atomic_json(native_root / 'identity.json', dict(identity=native_identity, protocol=native_protocol))
    identity_path = root / 'identity.json'
    if identity_path.is_file():
        saved = json.loads(identity_path.read_text())
        if saved.get('identity') != identity:
            raise ValueError('Validation model/data/code/runtime changed; preserve this output and use a new run')
    elif any((root / 'completed').glob('*.gz')):
        raise ValueError('Unidentified validation caches preserved; use a new output folder')
    atomic_json(identity_path, dict(identity=identity, protocol=protocol))
    factory = provider_factory(None, None, device=device, native_batch=native_batch,
                               native_model=model, verify_weights=False)
    was_training = bool(model.training)
    predictions, records = {}, []
    reused = 0
    try:
        model.eval()
        for number, filename in enumerate(cohort['names'], 1):
            stem = Path(filename).stem
            image_sha = yolo['image_sha256'][filename]
            complete_path = root / 'completed' / (stem + '.json.gz')
            complete = read_envelope(complete_path, identity, image_sha)
            masks = None
            if isinstance(complete, dict) and complete.get('file_name') == filename:
                try:
                    masks = validate_predictions(complete['rles'])
                    if complete.get('instances') != len(masks):
                        masks = None
                except (ValueError, KeyError, TypeError):
                    masks = None
            if masks is None:
                native_path = native_root / (stem + '.json.gz')
                native = read_envelope(native_path, native_identity, image_sha)
                if not _native_valid(native, stem, filename, image_sha, native_identity):
                    raw = (Path(cfg['images']) / filename).read_bytes()
                    if hashlib.sha256(raw).hexdigest() != image_sha:
                        raise ValueError(f'Validation image changed during evaluation: {filename}')
                    pixels = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
                    if pixels is None or pixels.shape != (2048, 2048) or pixels.dtype != np.uint8:
                        raise ValueError('Expected original uint8 2048-square GONG JPEG')
                    native = factory.native(stem, pixels, file_name=filename,
                                            image_sha256=image_sha, identity=native_identity)
                    if not _native_valid(native, stem, filename, image_sha, native_identity):
                        raise ValueError('Invalid native prediction record')
                    write_envelope(native_path, native, native_identity, image_sha)
                masks = validate_predictions(finalize_frame(stem, dict(yolo['frames'][stem], native=native)))
                complete = dict(file_name=filename, rles=json_ready(masks), instances=len(masks))
                write_envelope(complete_path, complete, identity, image_sha)
            else:
                reused += 1
            predictions[stem] = masks
            records.append(dict(file_name=filename, rles=masks))
            print(f'{label} full pipeline validation [{number}/{len(cohort["names"])}] {filename}', flush=True)
        csv_path = root / 'predictions.csv'
        rows = export_csv(csv_path, records)
        result = dict(**score_predictions(cohort['groups'], predictions),
            schema_version=1, label=label, identity=identity, model_sha256=model_sha,
            native_identity=native_identity, comparison_role=protocol['comparison_role'],
            physical_images=len(cohort['names']), annotation_groups=len(cohort['groups']),
            gt_instances=sum(len(row['masks']) for row in cohort['groups'].values()),
            prediction_rows=rows, overlapping_pixels=0, reused_images=reused,
            csv_path=str(csv_path), csv_sha256=digest(csv_path),
            report_path=str(root / 'metrics.json'),
            scope='Fixed development validation; full frozen YOLO+native pipeline; not public LB',
            reference='Compare to epoch000 from this run, not historical PQ constants')
        atomic_json(root / 'metrics.json', result)
        return result
    finally:
        model.train(was_training)
