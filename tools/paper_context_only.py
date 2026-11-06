"""Train only the context model; compare to the unchanged original best84.

The three learning sources remain byte-identical to the previous paired cell,
so an already-started context run retains its strict optimizer/RNG resume.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import torch

import paper_method_training as legacy

METHOD = 'multiscale_context'
LEARNING_SOURCE_SHA = {
    'paper_method_training.py': '70a2c5a4293bcd2770f7e657f16d71a7eaaa08216862357cfa22734ab4715cf1',
    'paper_multiscale.py': '15cf2002b3dabcc53736c46815d0b8a868630b14c78cf85442f2f96d6041ee5c',
    'base_runtime.py': legacy.BASE_RUNTIME_SHA,
}


def verified_assets(config):
    root = Path(legacy.__file__).parent
    assets = {name: str(root/name) for name in LEARNING_SOURCE_SHA}
    assets['base_runtime.py'] = config['base_runtime']
    hashes = {name: legacy.digest(path) for name, path in assets.items()}
    if hashes != LEARNING_SOURCE_SHA:
        raise ValueError('Learning source differs from the reviewed previous cell')
    return assets, hashes


def baseline_reference(config):
    """Read the original checkpoint, never a continued-training control run."""
    if legacy.digest(config['initial_checkpoint']) != legacy.INITIAL_CHECKPOINT_SHA:
        raise ValueError('Fixed baseline must be the original best84 checkpoint')
    checkpoint = torch.load(config['initial_checkpoint'], map_location='cpu', weights_only=True)
    validation = checkpoint.get('validation')
    if (checkpoint.get('epoch') != 84 or not isinstance(validation, dict)
            or validation.get('threshold') != .5 or validation.get('min_area') != 300
            or not isinstance(validation.get('pq'), (int, float))):
        raise ValueError('Original best84 has no comparable fixed-threshold validation record')
    saved = checkpoint.get('config', {})
    for key in ('crop', 'overlap', 'preprocessing', 'encoder'):
        if key in saved and saved[key] != config[key]:
            raise ValueError(f'Baseline inference setting differs: {key}')
    if saved.get('source_sha256') and saved['source_sha256'] != config['expected_source_sha256']:
        raise ValueError('Original best84 and requested split/data metadata differ')
    return dict(kind='fixed_original_best84', checkpoint_sha256=legacy.INITIAL_CHECKPOINT_SHA,
                epoch=84, validation=validation, trained_again=False,
                measurement='Historical native single-model validation from the original checkpoint')


def verify_metadata_hashes(config):
    """Do not bypass input identity checks when training/export is resumed."""
    expected = config['expected_source_sha256']
    if not isinstance(expected, dict):
        raise ValueError('Expected metadata hashes must be a mapping')
    for key, sha in expected.items():
        if legacy.digest(config[key]) != sha:
            raise ValueError(f'Input metadata differs from the verified file: {key}')


def context_identity(legacy_identity, driver_sha):
    value = dict(legacy_identity=legacy_identity, driver_sha256=driver_sha,
                 execution_methods=[METHOD], baseline_sha256=legacy.INITIAL_CHECKPOINT_SHA)
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def current_device_reference(config, reference, output, identity):
    """Evaluate fixed best84 once under this device's precision and batch size."""
    if not config.get('evaluate_baseline_current_device', False):
        return reference
    device = torch.device(config['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('Select a GPU runtime before baseline evaluation')
    dtype = ((torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
             if device.type == 'cuda' else None)
    hardware = dict(device_type=device.type, gpu=(torch.cuda.get_device_name(device)
                    if device.type == 'cuda' else 'cpu'), amp_dtype=str(dtype),
                    infer_batch=config.get('infer_batch', config['batch']))
    cache_identity = dict(experiment_identity=identity, **hardware)
    path = output/'best84_current_device_reference.json'
    if path.is_file():
        cached = json.loads(path.read_text())
        if cached.get('cache_identity') == cache_identity:
            return cached['reference']
        raise ValueError('Baseline device/precision/batch changed; use a separate output folder')
    runtime = legacy.load_runtime(Path(config['base_runtime']))
    model = runtime.build_model(config['encoder'], pretrained=False)
    checkpoint = torch.load(config['initial_checkpoint'], map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict']); del checkpoint
    model.to(device).eval()
    runtime.IMAGES=Path(config['images']); runtime.ANN=Path(config['annotations'])
    runtime.TRAIN=Path(config['train_split']); runtime.VAL=Path(config['val_split'])
    data=json.loads(Path(config['annotations']).read_text())
    _, names, runtime.GT = runtime.prepare_split(data)
    store = runtime.ImageStore(Path(config['images']), Path(config['cache_root'])/'images_real', 'limb_only')
    try:
        validation, _, _ = runtime.validate_native(model, store, names, device, dtype, config)
    finally:
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    result = dict(reference, historical_validation=reference['validation'], validation=validation,
                  hardware=hardware, measurement='Frozen best84 re-evaluated on current device/precision; no training')
    legacy.atomic_json(path, dict(cache_identity=cache_identity, reference=result))
    return result


class _PilotPause(Exception):
    def __init__(self, run, epoch):
        self.run, self.epoch = Path(run), int(epoch)


def completed_epochs(run):
    path = Path(run)/'last.pt'
    if not path.is_file():
        return 0
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    return int(checkpoint['epoch'])


def open_progress(output, identity, legacy_identity):
    path = output/'context_only_progress.json'
    if path.is_file():
        journal = json.loads(path.read_text())
        if journal.get('identity') != identity:
            raise ValueError('Context-only settings/source changed; keep the same cell to resume')
        if set(journal.get('runs', {})) - {METHOD}:
            raise ValueError('Context-only journal must not include a control training run')
        return path, journal
    journal = dict(identity=identity, legacy_identity=legacy_identity,
                   execution_methods=[METHOD], runs={})
    previous_path = output/'experiment_progress.json'
    if previous_path.is_file():
        previous = json.loads(previous_path.read_text())
        if previous.get('identity') != legacy_identity:
            raise ValueError('Previous experiment settings/source differ; refusing ambiguous resume')
        context = previous.get('runs', {}).get(METHOD)
        if context:
            journal['runs'][METHOD] = dict(context)
            journal['adopted_context_from'] = str(previous_path)
    return path, journal


def configure_runtime(config, journal, journal_path, assets, driver_sha):
    runtime = legacy.make_runtime(config, METHOD, journal, journal_path, assets)
    save_legacy = runtime.atomic_json_save
    def save(path, value):
        save_legacy(path, value)
        path = Path(path)
        if path.name == 'config.json':
            value.update(execution_methods=[METHOD], fixed_baseline_trained_again=False,
                initialization=('Completed-epoch context model/optimizer/RNG resume' if config.get('resume_dir')
                                else 'Original best84 warm start with a new context block and fresh optimizer'),
                experiment_scope='Fixed original best84 versus context-only fine-tuning; native single-model PQ',
                context_only_driver_sha256=driver_sha)
            legacy.atomic_json(path, value)
            destination = path.parent/'paper_context_only.py'
            raw = Path(__file__).read_bytes()
            if destination.is_file() and destination.read_bytes() != raw:
                raise ValueError('Saved context-only orchestration source differs')
            if not destination.is_file():
                destination.write_bytes(raw)
    runtime.atomic_json_save = save
    return runtime


def refresh_best_metadata(runtime, run):
    checkpoint = torch.load(Path(run)/'best.pt', map_location='cpu', weights_only=True)
    runtime.export_best_predictions(Path(run), checkpoint)
    return dict(epoch=checkpoint['epoch'], best=checkpoint['validation'])


def run_context(config):
    config = dict(config)
    legacy.assert_training_sources_are_official(config)
    verify_metadata_hashes(config)
    # These learning settings preserve the previous cell's identity exactly.
    # Execution below explicitly selects only METHOD and never calls run_suite.
    if config['methods'] != ['control', METHOD]:
        raise ValueError('Keep legacy method metadata for strict source/config compatibility')
    if config['thresholds'] != [.5] or config['min_areas'] != [300]:
        raise ValueError('The baseline comparison uses threshold .5 and minimum area 300')
    if config['epochs'] != 100:
        raise ValueError('Context training is capped at 100 total epochs, including resumed epochs')
    target = int(config.get('run_until_epoch', 100))
    if not 1 <= target <= 100 or (target not in (1,100) and target % int(config['validate_every'])):
        raise ValueError('run_until_epoch must be 1, 100 or a validation-interval boundary within 100')
    config['initial_checkpoint_sha256'] = legacy.INITIAL_CHECKPOINT_SHA
    assets, hashes = verified_assets(config)
    config['research_source_sha256'] = hashes
    reference = baseline_reference(config)
    driver_sha = legacy.digest(__file__)
    legacy_identity = legacy.suite_fingerprint(config, hashes)
    identity = context_identity(legacy_identity, driver_sha)
    output = Path(config['output_root']); output.mkdir(parents=True, exist_ok=True)
    reference = current_device_reference(config, reference, output, identity)
    journal_path, journal = open_progress(output, identity, legacy_identity)
    journal.update(warm_start_validation=reference['validation'], baseline=reference,
                   context_only_driver_sha256=driver_sha)
    legacy.atomic_json(journal_path, journal)
    legacy.atomic_json(output/'best84_fixed_reference.json', reference)
    previous = journal['runs'].get(METHOD, {})
    run = Path(previous['path']) if previous.get('path') else None
    cfg = dict(config, execution_methods=[METHOD], paper_method=METHOD, pretrained=False,
               output_root=str(output/METHOD), resume_dir=None)
    if run and (run/'last.pt').is_file():
        cfg['resume_dir'] = str(run)
    runtime = configure_runtime(cfg, journal, journal_path, assets, driver_sha)
    complete = previous.get('status') in ('train_complete', 'complete')
    epochs_done = completed_epochs(run) if run else 0
    if (complete or epochs_done >= target) and run and (run/'best.pt').is_file():
        print('Context training already complete; resuming result export only.', flush=True)
    else:
        print(f'Training only multiscale_context through epoch {target}; scheduler horizon remains 100. Fixed best84 is not trained.', flush=True)
        original_package = runtime.package_results if target < 100 else None
        if original_package:
            def package_and_pause(output_path, summary):
                result = original_package(output_path, summary)
                epoch = int(summary['epochs_completed'])
                if epoch >= target and epoch < 100:
                    raise _PilotPause(output_path, epoch)
                return result
            runtime.package_results = package_and_pause
        try:
            result = runtime.run_training(cfg)
            run = Path(result['output'])
            epochs_done = completed_epochs(run)
        except _PilotPause as pause:
            run, epochs_done = pause.run, pause.epoch
        finally:
            if original_package:
                runtime.package_results = original_package
        journal['runs'][METHOD].update(status='train_complete' if epochs_done >= 100 else 'paused',
                                      path=str(run), epochs_completed=epochs_done)
        legacy.atomic_json(journal_path, journal)
    metadata = refresh_best_metadata(runtime, run)
    rich = legacy.export_rich(runtime, cfg, run)
    status = 'complete' if epochs_done >= 100 else 'paused'
    journal['runs'][METHOD].update(status=status, path=str(run), best=metadata['best'],
                                  best_epoch=metadata['epoch'], rich_export=str(rich), epochs_completed=epochs_done)
    legacy.atomic_json(journal_path, journal)
    comparison = dict(scope='Fixed original best84 versus context-only continued fine-tuning; native single-model PQ',
        baseline=reference, multiscale_context=metadata['best'], best_context_epoch=metadata['epoch'],
        delta_pq=metadata['best']['pq']-reference['validation']['pq'],
        baseline_trained_again=False, execution_methods=[METHOD], epochs_maximum=100,
        status=status, epochs_completed=epochs_done, run_until_epoch=target, scheduler_total_epochs=100,
        external_data_used_for_training=False, full_ensemble_pq_measured=False,
        limitation='This comparison includes both the added block and additional training; it does not isolate architecture causality.')
    legacy.atomic_json(output/'context_only_comparison.json', comparison)
    archive = output/'paper_context_only_comparison.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in (journal_path, output/'best84_fixed_reference.json', output/'context_only_comparison.json'):
            z.write(path, path.name)
        for name in ('config.json','history.json','best_epoch.json','best_validation.csv','raw_paper_views.jsonl'):
            if (run/name).is_file():
                z.write(run/name, METHOD+'/'+name)
    return dict(archive=str(archive), comparison=comparison, output=str(output), context_run=str(run))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True); parser.add_argument('--result', required=True)
    args = parser.parse_args()
    result = run_context(json.loads(Path(args.config).read_text()))
    legacy.atomic_json(args.result, result)


if __name__ == '__main__':
    main()
