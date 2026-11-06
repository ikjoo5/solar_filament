"""Fixed full-cohort PQ evaluation for the DETR60 -> 150 continuation.

The ROI30 segmenter and decoding are frozen. Evaluation runs one detector view
on every validation photo, retaining separate annotator groups in the official
PQ calculation. The same predeclared three confidence thresholds are scored at every epoch;
no ground-truth boxes enter inference.
"""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import cv2
import numpy as np
from pycocotools import mask as mu
import torch

import bbox_detr_continuation as continuation
import bbox_probe_pipeline as probe
import filament_detr as detr
import roi_filament_pipeline as roi
from roi_filament_core import RoiFilamentNet, crop_roi, prepare_image, restore_mask
from native_instance_data import load_cohort
from native_instance_training import (atomic_json, capture_rng, configure_cuda,
    device_for, digest, identity, input_image_hashes, restore_rng)


CONFIDENCE_GRID = [.6, .8, .9]
FIXED_DECODING = dict(eval_mask_threshold=.5,
    eval_min_area=300, eval_min_quality=.6, eval_nms_iou=.5)


def evaluation_settings(config, base_config):
    """Only batch size, hardware and detector resolution are configurable."""
    for key, expected in FIXED_DECODING.items():
        if config.get(key, expected) != expected:
            raise ValueError('Long-run checkpoint selection requires fixed '+key)
    if config.get("eval_box_thresholds", CONFIDENCE_GRID) != CONFIDENCE_GRID:
        raise ValueError("Use the same predeclared confidence grid [.6,.8,.9] at every epoch")
    batch = config.get('roi_eval_batch', 8)
    size = config.get('detector_image_size', 1280)
    precision = config.get('detector_precision', 'bf16')
    if not isinstance(batch, int) or isinstance(batch, bool) or batch < 1:
        raise ValueError('roi_eval_batch must be a positive integer')
    if not isinstance(size, int) or size < 32 or size % 32:
        raise ValueError('Detector image size must be a positive multiple of32')
    if precision not in ('bf16', 'fp16', 'fp32'):
        raise ValueError('Invalid detector precision')
    return dict(FIXED_DECODING, eval_box_thresholds=CONFIDENCE_GRID, roi_eval_batch=batch, detector_image_size=size,
        detector_precision=precision, bbox_head_precision='fp32',
        roi_precision='fp16' if base_config['amp'] and str(config['device']).startswith('cuda') else 'fp32',
        roi_size=base_config['roi_size'], roi_context=base_config['roi_context'],
        preprocessing=base_config['preprocessing'], raw_box_threshold=.05,
        detr_max_det=base_config.get('detr_max_det', 100),
        roi_candidate_floor=.6, threshold_search='fixed predeclared confidence grid only')


def prepare_context(config):
    """Read and verify original data/ROI provenance once per launcher process.

    This does not write to or execute code from the completed experiments.
    The training engine separately verifies the source DETR60 checkpoint.
    """
    base = Path(config['original_base_root']).resolve()
    out = Path(config['output_root']).resolve()
    if base == out or base in out.parents or out in base.parents:
        raise ValueError('Original experiment and long-run output must be disjoint')
    original = json.loads((base/'experiment_config.json').read_text())
    old = roi.config_defaults(original)
    if Path(old['output_root']).resolve() != base:
        raise ValueError('Original ROI output path changed')
    pointer = json.loads((base/'current_code.json').read_text())
    if digest(base/'experiment_config.json') != pointer['config_sha256']:
        raise ValueError('Original ROI config changed')
    manifest = base/'code'/pointer['version']/'source_manifest.json'
    if digest(manifest) != pointer['manifest_sha256']:
        raise ValueError('Original ROI source manifest changed')
    if old['strict'] and pointer['manifest_sha256'] != probe.BASE_MANIFEST:
        raise ValueError('Expected the reviewed ROI30 original experiment')
    for name, expected in json.loads(manifest.read_text()).items():
        if Path(name).name != name or digest(manifest.parent/name) != expected:
            raise ValueError('Original source manifest entry changed: '+name)
        if name in probe.FROZEN and digest(Path(__file__).with_name(name)) != expected:
            raise ValueError('Bundled frozen helper changed: '+name)
    cohort = load_cohort(old)
    counts = cohort['counts']
    if old['strict'] and (counts['validation_physical_images'], counts['validation_annotation_groups'],
                         counts['validation_instances']) != (91, 101, 708):
        raise ValueError('Expected the complete fixed91/101-rater/708-GT validation cohort')
    stamps = input_image_hashes(old, cohort)
    source = json.loads((base/'roi/training_identity.json').read_text())
    if identity(source['protocol']) != source['identity']:
        raise ValueError('Original ROI identity checksum mismatch')
    current = roi.roi_protocol(old, cohort, stamps)
    previous = source['protocol']['static']
    if {k:v for k,v in current.items() if k!='runtime'} != {k:v for k,v in previous.items() if k!='runtime'}:
        raise ValueError('Original ROI data, source or settings changed')
    checkpoint = base/'roi'/f"epoch{old['roi_epochs']:03d}.pt"
    roi_sha = digest(checkpoint)
    if old['strict'] and (old['roi_epochs'] != 30 or roi_sha != probe.BASE_ROI_SHA):
        raise ValueError('Expected the frozen reviewed ROI30 weights')
    saved = continuation.load_checkpoint(checkpoint)
    if (saved.get('identity'), saved.get('completed_epoch')) != (source['identity'], old['roi_epochs']):
        raise ValueError('Frozen ROI checkpoint identity mismatch')
    del saved
    settings = evaluation_settings(config, old)
    device = device_for(config['device'])
    if settings['detector_precision'] != 'fp32' and device.type != 'cuda':
        raise ValueError('CPU evaluation fixtures must explicitly request fp32')
    if settings['detector_precision'] == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('A BF16-capable GPU is required; choose A100')
    protocol = dict(version=1, roi_checkpoint_sha256=roi_sha,
        roi_training_identity=source['identity'], metadata=cohort['input_sha256'],
        images=stamps, settings=settings, source_hashes=dict(probe.source_hashes(),
            detr_longrun_eval=digest(Path(__file__))), runtime=detr.runtime_versions(),
        roi_runtime=roi.versions(), device=str(device),
        gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None)
    run_id = identity(protocol)
    guard = out/'evaluation_protocol.json'
    if guard.exists():
        previous = json.loads(guard.read_text())
        if previous.get('identity') != run_id or identity(previous.get('protocol')) != run_id:
            raise ValueError('Evaluation source, data, precision, batch or runtime changed; preserve this run')
    else:
        out.mkdir(parents=True, exist_ok=True)
        atomic_json(guard, dict(identity=run_id, protocol=protocol))
    return dict(base_config=old, cohort=cohort, stamps=stamps, roi_checkpoint=checkpoint,
        roi_sha=roi_sha, identity=run_id, protocol=protocol)


def infer_rois_batched(model, image, proposals, config, device, batch_size=8):
    """Serial-reference geometry and quality, with only the network call batched.

    Inputs are corrected once on the whole native image. Inference preserves
    the original proposal ordering, crop geometry, interpolation, threshold
    and FP16 behavior. Every mask carries its exact source box and box hash.
    """
    if batch_size < 1:
        raise ValueError('Positive ROI batch size required')
    rows = [p for p in roi.checked_boxes(proposals) if p['score'] >= config['proposal_confidence']]
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start+batch_size]
            crops = [crop_roi(image, p['box'], context=config['roi_context'], size=config['roi_size']) for p in chunk]
            inputs = torch.from_numpy(np.ascontiguousarray(np.stack([c[0] for c in crops]))).float().to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=bool(config['amp'] and device.type=='cuda')):
                probabilities = model(inputs).sigmoid()[:,0].float().cpu().numpy()
            if len(probabilities) != len(chunk) or not np.isfinite(probabilities).all():
                raise FloatingPointError('Invalid ROI probabilities')
            for proposal, crop, probability in zip(chunk, crops, probabilities):
                foreground = probability >= config['mask_threshold']
                quality = float(probability[foreground].mean()) if foreground.any() else 0.
                rle = restore_mask(probability, crop[2], threshold=config['mask_threshold'])
                if rle is not None and int(mu.area(rle)):
                    result.append(dict(rle=rle, box=proposal['box'], box_id=identity(proposal['box']),
                        box_score=proposal['score'], roi_quality=quality,
                        selection_score=proposal['score']*quality))
    return result


def _load_models(context, config, checkpoint_path, epoch, device):
    saved = continuation.load_checkpoint(checkpoint_path)
    if saved.get('completed_epoch') != epoch:
        raise ValueError('Detector checkpoint does not match the requested cumulative epoch')
    model = detr.build_detector(context['base_config'], pretrained=False, model_config=saved['model_config'])
    model.load_state_dict(saved['model'])
    continuation.install_fp32_bbox_head(model)
    model = model.to(device).eval()
    del saved
    saved_roi = continuation.load_checkpoint(context['roi_checkpoint'])
    segmenter = RoiFilamentNet(encoder=context['base_config']['roi_encoder'], encoder_weights=None)
    segmenter.load_state_dict(saved_roi['model'])
    segmenter = segmenter.to(device).eval()
    del saved_roi
    return model, segmenter


def evaluate_checkpoint(config, checkpoint_path, epoch, *, context=None):
    """Return global PQ and diagnostics after every photo has been processed.

    A persistent per-image cache supports interrupted validation. Its identity
    binds model bytes, ROI bytes, data pixels, preprocessing, precision, batch
    size and helper source. Empty predictions are explicit records. Scores are
    always recomputed from the complete cache, never trusted as cached numbers.
    """
    context = prepare_context(config) if context is None else context
    old, cohort = context['base_config'], context['cohort']
    settings = evaluation_settings(config, old)
    if settings != context['protocol']['settings']:
        raise ValueError('Evaluation settings changed since context preparation')
    if digest(context['roi_checkpoint']) != context['roi_sha']:
        raise ValueError('Frozen ROI checkpoint changed')
    checkpoint_path = Path(checkpoint_path)
    checkpoint_sha = digest(checkpoint_path)
    eval_id = identity(dict(context=context['identity'], checkpoint_sha256=checkpoint_sha, epoch=epoch))
    folder = Path(config['output_root'])/'evaluations'/f'epoch{epoch:03d}_{eval_id[:16]}'
    folder.mkdir(parents=True, exist_ok=True)
    atomic_json(folder/'owner.json', dict(owner='detr_longrun_eval', identity=eval_id,
        context_identity=context['identity'], epoch=epoch, checkpoint_sha256=checkpoint_sha))
    names = cohort['validation_names']
    if len(names) != len(set(names)) or len({Path(n).stem for n in names}) != len(names):
        raise ValueError('Duplicate validation photo keys')
    if {Path(n).stem for n in names} != set(cohort['mapping'].values()):
        raise ValueError('Incomplete validation photo/annotator mapping')
    predictions = {str(threshold):{} for threshold in CONFIDENCE_GRID}
    boxes, record_hashes = {}, {}
    detector = segmenter = None
    device = device_for(config['device'])
    configure_cuda()
    detector_config = dict(old, detr_image_size=settings['detector_image_size'],
        detector_precision=settings['detector_precision'], detr_confidence=.05)
    roi_config = dict(old, proposal_confidence=.6, mask_threshold=.5)
    rng = capture_rng()
    try:
        for number, name in enumerate(names, 1):
            stem = Path(name).stem
            image_path = Path(old['images'])/name
            if digest(image_path) != context['stamps'][name]:
                raise ValueError('Validation image changed: '+name)
            path = folder/'records'/(stem+'.json.gz')
            if path.exists():
                record = roi.read_record(path)
                if (record.get('identity'), record.get('image_sha256'), record.get('image_name')) != (eval_id, context['stamps'][name], name):
                    raise ValueError('Validation record identity changed: '+name)
            else:
                if detector is None:
                    detector, segmenter = _load_models(context, config, checkpoint_path, epoch, device)
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if image is None or image.shape != (2048,2048):
                    raise ValueError('Original2048 validation image required: '+name)
                corrected = prepare_image(image, old['preprocessing'])
                proposals = probe.infer_detr_preprocessed(detector, corrected, detector_config)
                candidates = infer_rois_batched(segmenter, corrected, proposals, roi_config,
                    device, batch_size=settings['roi_eval_batch'])
                record = dict(identity=eval_id, image_name=name, image_sha256=context['stamps'][name],
                    actual_boxes=proposals, actual=candidates)
                roi.write_record(path, record)
            boxes[stem] = roi.checked_boxes(record['actual_boxes'])
            candidates = roi.decode_serialized_candidates(record['actual'])
            for row in candidates:
                if row['box_id'] != identity(row['box']) or row['box_score'] < .6:
                    raise ValueError('ROI candidate box association changed')
            selector = probe.MaskSelector(candidates, nms=.5)
            for threshold in CONFIDENCE_GRID:
                predictions[str(threshold)][stem] = selector(threshold, 300, .6)
            record_hashes[str(path.relative_to(folder))] = digest(path)
            print(f'DETR{epoch} fixed PQ validation {number}/{len(names)} {name}', flush=True)
    finally:
        del detector, segmenter
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        restore_rng(rng)
    if digest(checkpoint_path) != checkpoint_sha:
        raise ValueError('Detector checkpoint changed during validation')
    for prediction in predictions.values():
        if set(prediction) != {Path(n).stem for n in names}:
            raise ValueError('Missing validation predictions, including empty photos')
    grid_metrics = {key:roi.score_groups(cohort['truth'], cohort['mapping'], prediction)
                    for key,prediction in predictions.items()}
    # Deterministic tie: the predeclared order .6,.8,.9, never an adaptive grid.
    selected_key = max(grid_metrics, key=lambda key:grid_metrics[key]['score']['pq'])
    metrics = grid_metrics[selected_key]
    diagnostics = probe.bbox_metrics(cohort['truth'], cohort['mapping'], boxes, [.05]+CONFIDENCE_GRID)
    csv_sha256 = {}
    for key, prediction in predictions.items():
        csv_path = folder/('predictions_conf'+key+'.csv')
        roi.write_predictions(csv_path, prediction)
        csv_sha256[csv_path.name] = digest(csv_path)
    report = dict(status='complete', identity=eval_id, epoch=epoch,
        scope='Repeated fixed-split development PQ; not an independent test or leaderboard score',
        evaluation_dir=str(folder), report_path=str(folder/'report.json'),
        checkpoint_sha256=checkpoint_sha, roi_checkpoint_sha256=context['roi_sha'],
        context_identity=context['identity'], cohort=cohort['counts'],
        score=metrics['score'], per_group=metrics['per_group'], bbox_diagnostics=diagnostics,
        selected_confidence=float(selected_key), confidence_grid_results=grid_metrics,
        fixed_08_score=grid_metrics['0.8']['score'],
        prediction_protocol=settings, physical_images_processed=len(names),
        physical_images_with_empty_predictions=sum(not rows for rows in predictions[selected_key].values()),
        raw_proposal_count=sum(map(len, boxes.values())), roi_retrained=False,
        roi_epochs=old['roi_epochs'], threshold_search='fixed predeclared confidence grid only',
        candidate_export_scope='Raw boxes >=0.05; ROI masks only for boxes >=0.6, before quality/area/NMS',
        record_sha256=record_hashes, csv_sha256=csv_sha256)
    atomic_json(folder/'report.json', report)
    print(f'DETR{epoch} fixed validation PQ={report["score"]["pq"]:.6f}, '
          f'FP={report["score"]["fp"]}, FN={report["score"]["fn"]}', flush=True)
    return report


def retain_evaluations(config, retained_paths):
    """Prune owned completed raw evaluations only AFTER controller commit.

    Best/latest paths must come from the durable controller. Unknown folders,
    interrupted evaluations and numeric report copies are never removed here.
    """
    root = (Path(config['output_root'])/'evaluations').resolve()
    keep = {Path(p).resolve() for p in retained_paths if p}
    for path in keep:
        if path.parent != root or not (path/'report.json').is_file():
            raise ValueError('Retained evaluation is missing or outside this run')
    if not keep:
        raise ValueError('At least the committed current evaluation must be retained')
    removed = []
    if not root.exists():
        return removed
    summary_root = Path(config['output_root'])/'epoch_reports'
    for folder in root.iterdir():
        if folder.resolve() in keep or folder.is_symlink() or not folder.is_dir():
            continue
        owner_path, report_path = folder/'owner.json', folder/'report.json'
        if not owner_path.is_file() or not report_path.is_file():
            continue
        owner, report = json.loads(owner_path.read_text()), json.loads(report_path.read_text())
        if owner.get('owner') != 'detr_longrun_eval' or owner.get('identity') != report.get('identity'):
            continue
        summary_root.mkdir(parents=True, exist_ok=True)
        atomic_json(summary_root/(folder.name+'.json'), report)
        shutil.rmtree(folder)
        removed.append(str(folder))
    return removed


def export_results(config, selected_evaluations):
    """Export reviewable best/latest predictions and small run metadata.

    Checkpoints, images and all unselected raw caches are excluded. Multipart
    archives remain below28MiB and carry a checksum manifest. Return paths.
    """
    root = Path(config['output_root']).resolve()
    destination = root/'exports'
    destination.mkdir(parents=True, exist_ok=True)
    selected = []
    for item in selected_evaluations:
        path = Path(item['evaluation_dir'] if isinstance(item, dict) else item).resolve()
        if path.parent != root/'evaluations' or not (path/'report.json').is_file():
            raise ValueError('Export requires completed evaluations belonging to this run')
        if path not in selected:
            selected.append(path)
    if not selected:
        raise ValueError('No completed evaluation selected for export')
    common, records = [], []
    for path in root.iterdir():
        if path.is_file() and path.suffix in ('.json','.csv','.md','.txt') and path.name != 'latest_results.json':
            common.append((path, str(path.relative_to(root))))
    for folder in (root/'code', root/'epoch_reports'):
        for path in folder.rglob('*') if folder.exists() else []:
            if path.is_file() and path.suffix in ('.py','.json','.csv','.md','.txt'):
                records.append((path, str(path.relative_to(root))))
    for folder in selected:
        for path in folder.rglob('*'):
            if path.is_file() and path.suffix in ('.json','.csv','.gz') and not '.pending' in path.name:
                records.append((path, str(path.relative_to(root))))
    entries = {name:dict(bytes=path.stat().st_size,sha256=digest(path)) for path,name in common+records}
    base = sum(path.stat().st_size+2*len(name)+256 for path,name in common)+2*len(json.dumps(entries))+65536
    budget = 26*1024*1024
    if base > budget//2:
        raise ValueError('Long-run summary exceeds export budget')
    parts, part, size = [], [], base
    for path, name in sorted(records):
        amount = path.stat().st_size+2*len(name)+512
        if base+amount > budget:
            raise ValueError('One result is too large for the upload budget')
        if part and size+amount > budget:
            parts.append(part)
            part, size = [], base
        part.append((path,name))
        size += amount
    parts.append(part)
    export_id, archives, pending_files = identity(entries), [], []
    try:
        for index, part in enumerate(parts, 1):
            path = destination/f'detr_longrun_{export_id[:12]}_part{index:02d}.zip'
            pending = path.with_suffix('.pending')
            pending_files.append((pending,path))
            manifest = dict(owner='detr_longrun_eval', run_root=str(root), export_id=export_id,
                part=index, parts_total=len(parts), selected_evaluations=[str(p.relative_to(root)) for p in selected],
                entries=entries)
            with zipfile.ZipFile(pending, 'w', zipfile.ZIP_DEFLATED) as archive:
                for source, name in sorted(common+part):
                    archive.write(source, name)
                archive.writestr('export_manifest.json', json.dumps(manifest, indent=2))
            if pending.stat().st_size > 28*1024*1024:
                raise ValueError('Export exceeds28MiB')
            # Verify actual archive content before replacing any prior export.
            with zipfile.ZipFile(pending) as archive:
                if json.loads(archive.read('export_manifest.json')) != manifest:
                    raise ValueError('Export manifest changed during creation')
                for member in archive.namelist():
                    if member == 'export_manifest.json':
                        continue
                    raw = archive.read(member)
                    if len(raw) != entries[member]['bytes'] or hashlib.sha256(raw).hexdigest() != entries[member]['sha256']:
                        raise ValueError('Export content changed during creation: '+member)
        # A write/verification failure above keeps the entire previous set.
        for pending,path in pending_files:
            pending.replace(path)
            archives.append(str(path))
        atomic_json(destination/'current_exports.json', dict(owner='detr_longrun_eval',
            run_root=str(root), export_id=export_id, archives=archives))
    finally:
        for pending,_ in pending_files:
            pending.unlink(missing_ok=True)
    # Only archives self-identifying as this run's exports may be removed.
    # Unknown, corrupted, externally copied and user-created ZIPs are retained.
    keep = set(archives)
    for prior in destination.glob('detr_longrun_*_part*.zip'):
        if str(prior) in keep or prior.is_symlink():
            continue
        try:
            with zipfile.ZipFile(prior) as archive:
                manifest = json.loads(archive.read('export_manifest.json'))
            own = (manifest.get('owner') == 'detr_longrun_eval'
                and manifest.get('run_root') == str(root)
                and identity(manifest['entries']) == manifest.get('export_id')
                and prior.name == f'detr_longrun_{manifest["export_id"][:12]}_part{manifest["part"]:02d}.zip')
        except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile):
            continue
        if own:
            prior.unlink()
    return archives
