"""Train fresh YOLO and ConvNeXt U-Net on every GONG/SWEFil/CHASE tile.

The two models run sequentially and have separate verified checkpoint journals.
Existing solar checkpoints are never initialization inputs. No test images are
used in training, validation, or checkpoint selection.
"""
from __future__ import annotations

import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import zipfile

os.environ.setdefault('TORCH_CUDNN_V8_API_DISABLED', '1')
import cv2
import numpy as np
from pycocotools import mask as mu
import torch

import chase_shape_training as original
import swefil_mixed_training as mixed
from chase_shape_training import atomic_json, digest, fingerprint, serial

base = original.base
VERSION = 'fullpass_both_t4_v1'
COUNTS = dict(real=616, swefil=250, synthetic=616)
DEFAULTS = dict(original.DEFAULTS, steps_per_epoch=5928, synthetic_ratio=616/1482,
    swefil_ratio=250/1482, warmup_epochs=0, real_finish_epochs=0,
    checkpoint_interval_updates=100)
SOURCE_FILES = tuple(dict.fromkeys((*mixed.SOURCE_FILES,
    'fullpass_both_training.py', 'fullpass_training_data.py',
    'fullpass_native_training.py', 'fullpass_yolo_training.py',
    'fullpass_checkpoint_store.py', 'chase_yolo_training.py',
    'targeted_synthetic_training.py')))


def normalize_config(value):
    for key, expected in DEFAULTS.items():
        if key in value and value[key] != expected:
            raise ValueError('The full-pass recipe is fixed: '+key)
    if any(value.get(key) for key in ('initial_checkpoint', 'resume_dir', 'yolo_checkpoint')):
        raise ValueError('Fresh general pretrained initialization only; this run resumes automatically.')
    project = Path(value['base']).expanduser().resolve()
    previous = Path(value.get('chase_training_root', project/'fixed_split_experiments/chase_shape_best84_recipe_t4_v1')).resolve()
    old = original.normalize_config(json.loads((previous/'experiment_config.json').read_text()))
    root = Path(value['output_root']).resolve()
    cache = Path(value.get('cache_root', '/content/fullpass_both_t4_cache')).resolve()
    if not root.is_relative_to(project) or root == project:
        raise ValueError('Place the new output inside the project, in its own experiment directory.')
    for protected in (previous, project/'fixed_split_experiments/swefil_all_mixed_native_t4_v1',
                      project/'MAGFiLO_1.0_Kaggle_2026', project/'external_CHASE',
                      project/'external_SWEFil', project/'clean_solar.jpg'):
        if root.is_relative_to(protected) or protected.is_relative_to(root):
            raise ValueError('The new run must preserve previous runs and source data.')
    if cache.is_relative_to('/content/drive') or cache.is_relative_to(root) or root.is_relative_to(cache):
        raise ValueError('Use a separate local Colab cache directory.')
    # Inherit source paths, never the old solar initialization checkpoint.
    cfg = {k:v for k,v in old.items() if k not in ('baseline_checkpoint', 'initial_checkpoint', 'resume_dir')}
    cfg.update(DEFAULTS)
    cfg.update(base=str(project), chase_training_root=str(previous), output_root=str(root),
        cache_root=str(cache), samples_per_epoch=5928, effective_batch=4, updates_per_epoch=1482,
        data_counts=COUNTS.copy(), initialization='fresh_general_pretraining_no_solar_checkpoint',
        sampling='Every physical image once via all four nonoverlapping1024 tiles; shuffled order only.',
        comparison='Fresh COCO YOLO26s-seg and ImageNet ConvNeXt-Tiny U-Net; all domains every epoch.',
        loss='U-Net foreground BCE+Dice and 0.1 genuine GONG spine BCE; no SWEFil/CHASE spine loss.',
        external_pool='All250 annotated SWEFil observations, processed version once, including2sunspot-only images.',
        validation_scope='Fixed91 GONG development images, excluded from official training. External temporal proximity is diagnostic.',
        drive_checkpoint_path_parts=value.get('drive_checkpoint_path_parts',
            [project.name, *root.relative_to(project).parts, 'verified_checkpoints']))
    return cfg


def parse_models(value):
    models = value.split(',') if isinstance(value, str) else list(value)
    if not models or len(set(models)) != len(models) or not set(models) <= {'yolo', 'unet'}:
        raise ValueError('Models must be a nonempty unique list of yolo and/or unet.')
    return models


def disjoint_yolo_masks(masks, scores, threshold, minimum_area):
    """Resolve overlaps by descending confidence before the final area filter."""
    occupied = np.zeros(masks.shape[-2:], dtype=bool)
    instances = []
    for index in np.argsort(-np.asarray(scores), kind='stable'):
        if float(scores[index]) < threshold:
            continue
        mask = (masks[index] > .5) & ~occupied
        if int(mask.sum()) < minimum_area:
            continue
        occupied |= mask
        instances.append(mu.encode(np.asfortranarray(mask.astype(np.uint8))))
    return instances


def make_yolo_validator(store, names):
    """Full2048 validation preserves native object scale after1024 training."""
    def validator(model, cfg):
        thresholds, areas = (.1, .2, .3, .4), (100, 200, 300)
        predictions = {(t,a): {} for t in thresholds for a in areas}
        for index, name in enumerate(names, 1):
            image = store.load(name)
            result = model.predict(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR),
                imgsz=2048, conf=.05, iou=.7, max_det=100, retina_masks=True,
                device=0, half=True, verbose=False)[0]
            if result.masks is None:
                masks, scores = np.empty((0,2048,2048), np.uint8), np.empty(0)
            else:
                masks = (result.masks.data > .5).cpu().numpy()
                scores = result.boxes.conf.float().cpu().numpy()
                if masks.shape[1:] != (2048,2048) or len(masks) != len(scores):
                    raise ValueError('Expected original-grid YOLO masks and matching confidences.')
            for (threshold, area), output in predictions.items():
                output[Path(name).stem] = disjoint_yolo_masks(masks, scores, threshold, area)
            print(f'YOLO validation [{index}/{len(names)}] {name}', flush=True)
        grid = []
        for (threshold, area), predicted in predictions.items():
            score = base.score(predicted)
            grid.append(dict(threshold=threshold, min_area=area, pq=score['PQ'], sq=score['SQ'],
                rq=score['RQ'], tp_pairs=score['TP'], fp=score['FP'], fn=score['FN']))
        return max(grid, key=lambda row:row['pq']), grid, predictions
    return validator


def publish_results(root, results, requested, cfg, prepared):
    root = Path(root)
    # Include an already completed other branch without misreporting its stage.
    for model in ('yolo', 'unet'):
        if model not in results:
            path = root/model/'latest_result.json'
            if path.is_file():
                results[model] = json.loads(path.read_text())
    status = 'complete' if all(results.get(m,{}).get('epochs_completed') == cfg['epochs']
                             for m in ('yolo','unet')) else 'paused'
    result = dict(status=status, requested_models=requested, models=results,
        epochs_completed={m:r.get('epochs_completed',0) for m,r in results.items()},
        epochs_target_per_model=cfg['epochs'], tiles_per_model_per_epoch=5928,
        weights_in_zip=False, scope='Separate native and YOLO training; no ensemble or public score claimed.')
    atomic_json(root/'summary.json', result)
    archive = root/'fullpass_both_results.zip'
    pending = archive.with_suffix('.pending.zip')
    with zipfile.ZipFile(pending, 'w', zipfile.ZIP_DEFLATED) as output:
        for name in ('summary.json','prepared_summary.json','coverage_manifest.json','experiment_config.json'):
            path = root/name
            if path.is_file(): output.write(path, name)
        for model in ('yolo', 'unet'):
            folder = root/model
            if folder.is_dir():
                for path in sorted(folder.rglob('*')):
                    if path.is_file() and path.suffix in ('.json','.csv') and 'verified_checkpoints' not in path.parts:
                        output.write(path, str(path.relative_to(root)))
        for name in SOURCE_FILES:
            output.write(Path(__file__).with_name(name), 'code/'+name)
    pending.replace(archive)
    result.update(archive=str(archive), archive_sha256=digest(archive))
    atomic_json(root/'latest_result.json', result)
    return result


def run(config, phase='all', run_until=None, models=('yolo','unet')):
    if phase not in ('prepare','train','all'):
        raise ValueError('Unsupported phase')
    models = parse_models(models)
    cfg = normalize_config(config)
    if run_until is not None and not 1 <= run_until <= cfg['epochs']:
        raise ValueError('run-until must be between1 and the fixed100-epoch schedule.')
    root = Path(cfg['output_root']); root.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(2)
    with (root/'.training.lock').open('a+') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Another process is using this full-pass run.')
        import chase_shape_cache
        restored = chase_shape_cache.restore(cfg['chase_training_root'])
        if restored['status'] not in ('local_complete', 'restored'):
            raise FileNotFoundError('Saved CHASE synthetic PNG archive is required; no resynthesis: '+str(restored))
        prepared = json.loads((Path(cfg['chase_training_root'])/'prepared_summary.json').read_text())
        data = json.loads(Path(cfg['annotations']).read_text())
        base.TRAIN, base.VAL, base.IMAGES = (Path(cfg[key]) for key in ('train_split','val_split','images'))
        _, names, base.GT = base.prepare_split(data)
        from swefil_training_data import prepare_swefil
        swefil = prepare_swefil(Path(cfg['base']), Path(cfg['cache_root'])/'swefil_source', val_names=names)
        if len(swefil['records']) != COUNTS['swefil']:
            raise ValueError('The complete pinned annotated SWEFil pool must contain250 observations.')
        device = torch.device('cpu' if phase == 'prepare' else cfg['device'])
        if phase != 'prepare' and not torch.cuda.is_available():
            raise RuntimeError('Select a Colab T4 GPU runtime for training.')
        source, hashes = original.source_identity(cfg, prepared, device)
        from fullpass_training_data import prepare_dataset
        dataset, store, names = prepare_dataset(cfg, prepared, swefil, hashes)
        counts = {domain:len(record['files']) for domain,record in dataset.domains.items()}
        if counts != COUNTS or len(dataset) != 5928:
            raise ValueError('Full data coverage differs: '+str(counts))
        coverage = dataset.coverage_manifest()
        atomic_json(root/'coverage_manifest.json', coverage)
        summary = dict(chase_synthesis_identity=prepared['identity'], chase_cache=restored,
            swefil={k:v for k,v in swefil.items() if k != 'records'},
            counts=counts, physical_images=1482, tiles_per_epoch=5928, official_validation_images=91,
            no_probability_sampling=True, no_domain_exclusion_epochs=True)
        atomic_json(root/'prepared_summary.json', summary)
        if phase == 'prepare':
            result = dict(status='prepared', output_root=str(root), **summary)
            atomic_json(root/'latest_result.json', result)
            return result
        original.configure_cuda()
        source.update(version=VERSION, settings={k:cfg[k] for k in DEFAULTS},
            code={name:digest(Path(__file__).with_name(name)) for name in SOURCE_FILES},
            swefil_manifest_sha256=swefil['manifest_sha256'], coverage=coverage,
            external_filter='All250 annotated observations; temporal proximity reported only.')
        from fullpass_checkpoint_store import BoundedDriveCheckpointStore
        import verified_drive_checkpoints as durable
        service = durable.make_drive_service()
        results = {}
        for model in models:
            branch = root/model; branch.mkdir(exist_ok=True)
            cfg_model = dict(cfg, output_root=str(branch), cache_root=str(Path(cfg['cache_root'])/model))
            branch_source = dict(source, model=model)
            if model == 'yolo':
                import fullpass_yolo_training as yolo
                branch_source['yolo_recipe'] = yolo.DEFAULTS
            identity = fingerprint(branch_source)
            identity_path = branch/'identity.json'
            if identity_path.exists() and original.read_envelope(identity_path, identity) != branch_source:
                raise ValueError('The training identity changed; preserve this experiment and use a new directory.')
            original.write_envelope(identity_path, identity, branch_source)
            folder_id = durable.resolve_experiment_folder(service,
                [*cfg['drive_checkpoint_path_parts'], model], identity, create_from=3,
                visibility_attempts=6, visibility_wait_seconds=2)
            checkpoints = BoundedDriveCheckpointStore(service, folder_id, identity,
                Path(cfg_model['cache_root'])/'remote_checkpoints', minimum_free_bytes=2_000_000_000)
            if model == 'unet':
                from fullpass_native_training import run_native
                def validator(network, device, amp, settings):
                    return base.validate_native(network, store, names, device, amp, settings)
                results[model] = run_native(cfg_model, dataset, validator, branch, identity,
                    checkpoints, run_until=run_until, prepared=summary)
            else:
                from fullpass_training_data import export_yolo_dataset
                yaml, metadata = export_yolo_dataset(dataset, Path(cfg_model['cache_root'])/'tiles')
                results[model] = yolo.run_yolo(cfg_model, yaml, metadata, branch, identity,
                    checkpoints, run_until=run_until, validator=make_yolo_validator(store, names))
            atomic_json(branch/'latest_result.json', results[model])
            gc.collect(); torch.cuda.empty_cache()
            publish_results(root, results, models, cfg, summary)
        return publish_results(root, results, models, cfg, summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--phase', choices=('prepare','train','all'), default='all')
    parser.add_argument('--run-until', type=int)
    parser.add_argument('--models', default='yolo,unet')
    args = parser.parse_args()
    print(json.dumps(run(json.loads(args.config.read_text()), args.phase,
        args.run_until, args.models), indent=2), flush=True)


if __name__ == '__main__': main()
