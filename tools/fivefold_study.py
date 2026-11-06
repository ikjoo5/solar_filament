"""A fresh fivefold study evaluated at the same declared budget in every fold.

Stage30 trains each fold through epoch30 of an unchanged100epoch schedule.
Stage60 resumes the same optimizers; stage exports and compact weights remain
separate. The v1 fold0-only experiment and its checkpoints are never imported.
"""
import argparse
from contextlib import redirect_stdout
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

import torch

import grouped_cv_training as training

SCHEMA_VERSION = 2
TRAINING_DEFAULTS = dict(encoder='tu-convnext_tiny', pretrained=True, epochs=100,
    crop=1024, batch=1, accumulate=4, steps_per_epoch=400, lr=1e-4,
    spine_weight=.1, workers=0, infer_batch=1, overlap=.25, validate_every=100,
    seed=42, device='cuda', threshold=.5, min_area=300, synthetic_ratio=0.,
    warmup_epochs=0, real_finish_epochs=0, include_empty_synthetic=False,
    preprocessing='limb_only', preflight_backward=True)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def preserve_json(path, value):
    path = Path(path)
    raw = json.dumps(value, ensure_ascii=False, indent=2)+'\n'
    if path.is_file():
        if path.read_text() != raw:
            raise ValueError('Existing study metadata differs; preserved: '+str(path))
    else:
        training.atomic_json(path, value)


def source_fingerprints(config, injected_export=False):
    rich = Path(config.get('rich_export_source', Path(__file__).with_name('fivefold_rich_inference.py')))
    files = dict(study=Path(__file__), training=Path(training.__file__),
                 runtime=Path(config['base_runtime']))
    if rich.is_file():
        files['rich_inference'] = rich
    elif not injected_export:
        raise FileNotFoundError(rich)
    result = {key: training.digest(path) for key, path in files.items()}
    if injected_export and 'rich_inference' not in result:
        result['rich_inference'] = 'injected_test_exporter'
    return result


def export_folder(canonical, profile, source, settings, explicit_variant=None):
    recipe = dict(inference_profile=profile, rich_source_sha256=source['rich_inference'],
        crop=settings['crop'], overlap=settings['overlap'], infer_batch=settings['infer_batch'])
    identity = fingerprint(recipe)
    marker = canonical/'export_recipe.json'
    if explicit_variant is not None:
        folder = canonical/'variants'/explicit_variant
    elif profile != 'tta_clahe':
        folder = canonical/'variants'/(profile+'_'+identity[:12])
    elif marker.is_file() and read(marker) != dict(identity=identity, payload=recipe):
        folder = canonical/'variants'/(profile+'_'+identity[:12])
    else:
        folder = canonical
    return folder, dict(identity=identity, payload=recipe)


def prepare_study(config, injected_export=False):
    """No GPU or image loading: freeze data/model settings and execution budget."""
    root = Path(config['study_root'])
    cfg = dict(TRAINING_DEFAULTS, **config.get('training_settings', {}))
    if cfg['epochs'] != 100 or cfg['validate_every'] != 100:
        raise ValueError('The study keeps the100epoch schedule and evaluates declared stages separately')
    if cfg['synthetic_ratio'] != 0 or not cfg['pretrained']:
        raise ValueError('This reviewed study uses generic ImageNet initialization and real training data only')
    if any(config.get(key) for key in ('initial_checkpoint', 'warm_start', 'resume_dir', 'checkpoint')):
        raise ValueError('Start a fresh study; old solar checkpoint imports are not allowed')
    stage = int(config.get('stage_epoch', 30))
    folds = [int(fold) for fold in config.get('folds', [0,1,2,3,4])]
    if not 1 <= stage <= cfg['epochs'] or not folds or len(set(folds)) != len(folds) or any(fold not in range(5) for fold in folds):
        raise ValueError('Choose a declared stage1..100 and unique folds0..4')
    profile = config.get('inference_profile', 'tta_clahe')
    if profile not in ('tta', 'tta_clahe'):
        raise ValueError('Expected tta or tta_clahe inference profile')
    variant = config.get('export_variant')
    if variant is not None and not re.fullmatch(r'[A-Za-z0-9_-]+', variant):
        raise ValueError('A simple export variant name is required')
    plan = read(config['fold_plan'])
    annotation_sha = training.digest(config['annotations'])
    if plan['folds'] != 5 or plan['sources_sha256']['annotations'] != annotation_sha:
        raise ValueError('A matching official fivefold plan is required')
    source = source_fingerprints(config, injected_export)
    training_source = {key: source[key] for key in ('study', 'training', 'runtime')}
    identity_payload = dict(schema_version=SCHEMA_VERSION,
        training_settings=cfg, annotations_sha256=annotation_sha,
        fold_plan_sha256=training.digest(config['fold_plan']),
        training_source_sha256=training_source,
        selection='Shared declared epoch; no per-fold best checkpoint selection')
    study_identity = fingerprint(identity_payload)
    frozen = dict(schema_version=SCHEMA_VERSION, study_identity=study_identity,
        identity_sha_algorithm='json_sort_keys_default', identity_payload=identity_payload,
        description='Fivefold native model study, fresh generic initialization; no best84 weights',
        expected_folds=[0,1,2,3,4])
    root.mkdir(parents=True, exist_ok=True)
    preserve_json(root/'study_config.json', frozen)
    release = dict(study_identity=study_identity, source_sha256=source)
    preserve_json(root/'releases'/(fingerprint(release)+'.json'), release)
    preserve_json(root/f'stage{stage:03d}_plan.json', dict(schema_version=SCHEMA_VERSION,
        study_identity=study_identity, stage_epoch=stage, training_epochs=cfg['epochs'],
        expected_folds=[0,1,2,3,4],
        evaluation_kind='fixed_budget_exploration' if stage < cfg['epochs'] else 'fixed_schedule_final',
        stage_budget_epochs=stage*5,
        interpretation='All folds must reach this same epoch before a complete OOF comparison. Exploratory30epoch results do not establish convergence.'))
    split_root = Path(config.get('splits_root', Path(config['fold_plan']).parent))
    jobs = []
    for fold in folds:
        fold_cfg = dict(cfg, fold=fold, seed=cfg['seed']+fold,
            images=str(config['images']), annotations=str(config['annotations']),
            base_runtime=str(config['base_runtime']), fold_plan=str(config['fold_plan']),
            train_split=str(split_root/f'fold{fold}/train_split.csv'),
            val_split=str(split_root/f'fold{fold}/val_split.csv'),
            output_root=str(root), cache_root=str(config['cache_root']),
            run_until_epoch=stage)
        for key in ('train_split', 'val_split'):
            if not Path(fold_cfg[key]).is_file():
                raise FileNotFoundError(fold_cfg[key])
        canonical = root/f'fold{fold}'/f'stage{stage:03d}'
        stage_dir, recipe = export_folder(canonical, profile, source, cfg, variant)
        jobs.append(dict(fold=fold, stage_epoch=stage, training_config=fold_cfg,
                         model_path=str(canonical/'model.pt'), stage_dir=str(stage_dir), export_recipe=recipe))
    return dict(root=str(root), study=frozen, stage_epoch=stage, stage_root=str(root),
        jobs=jobs, expected_folds=[0,1,2,3,4], requested_folds=folds,
        training_schedule_epochs=cfg['epochs'], total_initial_stage_epochs=stage*5,
        source_sha256=source, training_source_sha256=training_source, inference_profile=profile,
        export_variant=variant)


def check_stage_payload(payload, job, prepared):
    if (payload.get('study_identity') != prepared['study']['study_identity']
            or payload.get('epoch') != job['stage_epoch'] or payload.get('fold') != job['fold']
            or payload.get('training_source_sha256') != prepared['training_source_sha256']):
        raise ValueError('Stage model belongs to another study, fold, source or budget')
    cfg = payload['config']
    if cfg.get('fold') != job['fold'] or cfg.get('epochs') != 100:
        raise ValueError('Stage training configuration differs')
    if fingerprint(cfg['cv_identity_payload']) != cfg['cv_identity']:
        raise ValueError('Unverifiable training identity in stage checkpoint')
    for key in training.IDENTITY_KEYS:
        expected = job['training_config'].get(key, training.DEFAULTS.get(key))
        if cfg.get(key) != expected:
            raise ValueError('Stage training setting differs: '+key)


def snapshot_stage(runtime, full, model_path, job, prepared, checkpoint_path):
    cfg = full['config']
    payload = dict(schema_version=SCHEMA_VERSION, study_identity=prepared['study']['study_identity'],
        source_sha256=prepared['source_sha256'], training_source_sha256=prepared['training_source_sha256'], fold=job['fold'],
        epoch=int(full['epoch']), config=cfg, training_cv_identity=cfg['cv_identity'],
        training_checkpoint_sha256=training.digest(checkpoint_path),
        model_state_dict=full['model_state_dict'])
    check_stage_payload(payload, job, prepared)
    expected = dict(annotations=training.digest(job['training_config']['annotations']),
        fold_plan=training.digest(job['training_config']['fold_plan']),
        train_split=training.digest(job['training_config']['train_split']),
        val_split=training.digest(job['training_config']['val_split']),
        base_runtime=training.digest(job['training_config']['base_runtime']),
        cv_runner=training.digest(training.__file__))
    if cfg.get('source_sha256') != expected:
        raise ValueError('Fold training inputs changed before stage snapshot')
    runtime.atomic_torch_save(model_path, payload)
    return payload


def default_export(runtime, model, cfg, stage_dir, provenance):
    from fivefold_rich_inference import export_stage
    return export_stage(runtime, model, cfg, stage_dir, provenance)


def finished_stage(stage_dir, prepared, fold):
    path = stage_dir/'stage_summary.json'
    if not path.is_file():
        return None
    summary = read(path)
    if (summary.get('status') != 'complete' or summary.get('study_identity') != prepared['study']['study_identity']
            or summary.get('fold') != fold or summary.get('stage_epoch') != prepared['stage_epoch']
            or summary.get('inference_profile') != prepared['inference_profile']
            or summary.get('source_sha256') != prepared['source_sha256']):
        raise ValueError('Existing stage summary differs; preserved')
    for path, key in [(Path(summary['model_path']), 'model_sha256'),
                      (Path(summary['export']['rich_predictions']), 'rich_predictions_sha256'),
                      (Path(summary['export']['summary']), 'export_summary_sha256')]:
        if not path.is_file() or training.digest(path) != summary[key]:
            raise ValueError('Completed stage artifact changed or missing: '+str(path))
    return summary


def study_status(prepared):
    completed, summaries = [], []
    root = Path(prepared['root'])
    for fold in range(5):
        canonical = root/f'fold{fold}'/f"stage{prepared['stage_epoch']:03d}"
        folder, _ = export_folder(canonical, prepared['inference_profile'], prepared['source_sha256'],
            prepared['study']['identity_payload']['training_settings'], prepared['export_variant'])
        item = finished_stage(folder, prepared, fold)
        if item:
            completed.append(fold); summaries.append(item)
    return dict(schema_version=SCHEMA_VERSION,
        status='complete_stage' if len(completed) == 5 else 'partial_stage',
        study_identity=prepared['study']['study_identity'], stage_epoch=prepared['stage_epoch'],
        training_schedule_epochs=100, expected_folds=[0,1,2,3,4], completed_folds=completed,
        missing_folds=sorted(set(range(5))-set(completed)), complete_fivefold=len(completed) == 5,
        stage_root=str(root), inference_profile=prepared['inference_profile'], summaries=summaries,
        scores_computed_here=False, checkpoint_selection='Same declared stage in all folds')


class StageOutput:
    """Clarify the legacy schedule denominator without hiding any log/error."""
    def __init__(self, stream, stage):
        self.stream, self.stage = stream, stage
    def write(self, text):
        text = text.replace('fixed-final100', f'stage-evaluate{self.stage} (schedule-max100)')
        text = re.sub(r'epoch(\d+)/100\b', lambda m: f'epoch{m.group(1)}/{self.stage} (schedule100)', text)
        return self.stream.write(text)
    def flush(self):
        self.stream.flush()
    def __getattr__(self, name):
        return getattr(self.stream, name)


def run(config, _train=None, _export=None, _runtime=None):
    prepared = prepare_study(config, injected_export=_export is not None)
    if config.get('plan_only', False):
        return dict(prepared, status='plan_only', training_performed=False,
                    new_network_inference=False)
    runtime = _runtime or training.load_runtime(config['base_runtime'])
    train = _train or training.run
    export = _export or default_export
    for job in prepared['jobs']:
        stage_dir = Path(job['stage_dir']); stage_dir.mkdir(parents=True, exist_ok=True)
        done = finished_stage(stage_dir, prepared, job['fold'])
        if done:
            print(f"stage{prepared['stage_epoch']} fold{job['fold']}: completed artifact verified, skipping", flush=True)
            continue
        model_path = Path(job['model_path'])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        if model_path.is_file():
            payload = torch.load(model_path, map_location='cpu', weights_only=True)
            check_stage_payload(payload, job, prepared)
        else:
            last = Path(job['training_config']['output_root'])/f"fold{job['fold']}"/'last.pt'
            if last.is_file():
                saved = torch.load(last, map_location='cpu', weights_only=True)
                if int(saved['epoch']) > job['stage_epoch']:
                    raise ValueError('Training has advanced past this missing stage; earlier weights cannot be reconstructed')
                del saved
            print(f"Fold{job['fold']}: 이번 비교는{job['stage_epoch']}epoch에서 종료합니다 "
                  f"(5개fold 총{job['stage_epoch']*5}epoch). "
                  '100은 재개를 위한 학습률스케줄 상한이며 자동으로 진행하지 않습니다.', flush=True)
            with redirect_stdout(StageOutput(sys.stdout, job['stage_epoch'])):
                train(job['training_config'])
            if not last.is_file():
                raise FileNotFoundError('Training returned without a complete checkpoint: '+str(last))
            full = torch.load(last, map_location='cpu', weights_only=True)
            if int(full['epoch']) != job['stage_epoch']:
                raise ValueError('Training did not stop at the declared common stage')
            payload = snapshot_stage(runtime, full, model_path, job, prepared, last)
            del full
        model_sha = training.digest(model_path)
        cfg = dict(payload['config'], stage_epoch=job['stage_epoch'],
                   inference_profile=prepared['inference_profile'])
        preserve_json(stage_dir/'export_recipe.json', job['export_recipe'])
        device = torch.device(cfg['device'])
        if device.type == 'cuda':
            training.configure_cuda()
            torch.cuda.empty_cache()
        model = runtime.build_model(cfg['encoder'], pretrained=False).to(device)
        model.load_state_dict(payload['model_state_dict']); model.eval()
        provenance = dict(study_identity=prepared['study']['study_identity'],
            fold=job['fold'], stage_epoch=job['stage_epoch'], model_sha256=model_sha,
            training_checkpoint_sha256=payload['training_checkpoint_sha256'],
            training_cv_identity=payload['training_cv_identity'],
            source_sha256=prepared['source_sha256'])
        exported = export(runtime, model, cfg, stage_dir, provenance)
        fold_plan = read(config['fold_plan'])
        fold_info = next(row for row in fold_plan['fold_summaries'] if row['fold'] == job['fold'])
        images = [row['file_name'] for row in fold_plan['observations']
                  if fold_info['roles'][row['physical_identifier']] == 'validation']
        if exported['image_coverage'] != len(images):
            raise ValueError('Stage export must cover every validation image, including empty predictions')
        summary = dict(schema_version=SCHEMA_VERSION, status='complete', **provenance,
            v1_cv_identity=payload['training_cv_identity'], training_config=payload['config'], model_path=str(model_path),
            train_physical_identifiers=sorted(key for key, role in fold_info['roles'].items() if role == 'train'),
            image_names=sorted(images), image_coverage=len(images),
            inference_profile=prepared['inference_profile'], export=exported,
            rich_predictions_sha256=training.digest(exported['rich_predictions']),
            export_summary_sha256=training.digest(exported['summary']),
            evaluation_kind='fixed_budget_exploration' if job['stage_epoch'] < 100 else 'fixed_schedule_final')
        training.atomic_json(stage_dir/'stage_summary.json', summary)
        del payload, model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        training.atomic_json(Path(prepared['root'])/f"stage{prepared['stage_epoch']:03d}_progress.json", study_status(prepared))
    result = study_status(prepared)
    training.atomic_json(Path(prepared['root'])/f"stage{prepared['stage_epoch']:03d}_progress.json", result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--result', required=True)
    args = parser.parse_args()
    result = run(read(args.config))
    training.atomic_json(args.result, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
