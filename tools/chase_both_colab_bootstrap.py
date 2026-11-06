"""Durable Colab orchestration for CHASE-only YOLO and best84 updates.

Each trainer has its own process, optimizer and completed-epoch checkpoints.
The final comparison combines two independently trained models in four ways.
This module does not modify the original checkpoint or production recipe.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

NATIVE_SHA = 'ae6b209f72355fd52d2cb07182e69028e97c703a1977ad1e7b38b27cf8cebe69'
YOLO_SHA = '81c2a27d95821b5ede283169c0da8c08d360ecc4c2a60a1b9c6baf96d76bcccc'
ANN_SHA = '5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1'
TOTAL_EPOCHS = 60


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path, value, *, unchanged=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n'
    if unchanged and path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Stored experiment configuration changed: '+str(path))
        return
    pending = path.with_name(path.name+'.pending')
    try:
        pending.write_text(encoded, encoding='utf-8')
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def configurations(settings):
    root = Path(settings['root'])
    assets = Path(settings['assets'])
    common = dict(images=settings['images'], annotations=settings['annotations'],
        train_split=str(assets/'train_split.csv'), val_split=str(assets/'val_split.csv'),
        chase_root=settings['chase_root'], chase_manifest=str(assets/'chase_ahas2024_2025_manifest.json'),
        cache_root=settings['cache_root'], yolo_cache=str(assets/'frozen_yolo_validation.json.gz'),
        native_validation_cache_root=str(root/'native_validation_cache'),
        epochs=TOTAL_EPOCHS, epoch_sampling='full_pass', steps_per_epoch=695, batch=1, workers=0,
        training_domain='chase_only', chase_ratio=1.0, real_finish_epochs=0,
        lr=2e-5, weight_decay=.01, amp=True, seed=42,
        initial_checkpoint=settings['native_checkpoint'], yolo_checkpoint=settings['yolo_checkpoint'])
    native = dict(common, output_root=str(root/'native'), device='cuda', crop=1024,
                  accumulate=4, validate_every=5, native_batch=1, amp=True,
                  comparison_role='old_yolo_candidate_native')
    yolo = dict(common, output_root=str(root/'yolo'), device='0', imgsz=2048, nbs=1)
    compare = dict(native, output_root=str(root/'comparison'))
    return dict(native=native, yolo=yolo, comparison=compare)


def validate_paths(settings):
    for key, expected in [('native_checkpoint', NATIVE_SHA), ('yolo_checkpoint', YOLO_SHA), ('annotations', ANN_SHA)]:
        path = Path(settings[key])
        if not path.is_file():
            raise FileNotFoundError(key+' not found: '+str(path))
        if digest(path) != expected:
            raise ValueError(key+' does not match the existing LB0.39 experiment: '+str(path))
    if not Path(settings['images']).is_dir():
        raise FileNotFoundError('GONG train_images not found: '+settings['images'])


def child(command, title, *, runner=subprocess.run):
    print('\n'+title, flush=True)
    environment = dict(os.environ, PYTHONUNBUFFERED='1', TORCH_CUDNN_V8_API_DISABLED='1',
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    # Same process group as this orchestrator: notebook interruption stops both.
    runner(command, check=True, cwd=Path(__file__).resolve().parent, env=environment)


def prepare_worker(config):
    from chase_transfer_data import prepare_chase, ChaseDataset
    records, report_path = prepare_chase(config['chase_root'],
        Path(config['cache_root'])/'download', config['chase_manifest'])
    data = ChaseDataset(records, crop=1024, length=695, seed=42,
                        cache_dir=Path(config['cache_root'])/'chase_aligned')
    report = data.preflight()
    root = Path(config['output_root']).parent
    atomic_json(root/'chase_preparation.json', dict(status='prepared', observations=len(records),
        acquisition_report=str(report_path), alignment=report,
        explicit_overlap_exclusions_only=True, morphology_exclusion=False))
    print('CHASE 원반 정렬 준비 완료:', len(records), '장', flush=True)


def compare_worker(config, cap):
    import gc
    import torch
    import solar_a100_train_standalone as legacy
    from chase_best84_training import configure_cuda
    from chase_best84_evaluation import evaluate_candidate
    configure_cuda()
    if not torch.cuda.is_available():
        raise RuntimeError('Select a Colab T4/A100 GPU runtime for comparison')
    root = Path(config['output_root']).parent
    native_result = json.loads((root/'native/result.json').read_text())
    yolo_result = json.loads((root/'yolo/result.json').read_text())
    if native_result['epochs_completed'] != cap or yolo_result['epochs_completed'] != cap:
        raise ValueError('Both models must have completed the requested comparison epoch')
    for name, result in [('native', native_result), ('yolo', yolo_result)]:
        if result.get('training_domain') != 'chase_only' or result.get('epoch_sampling') != 'full_pass':
            raise ValueError(name+' result is not from the CHASE-only full-pass experiment')
    new_native = Path(native_result['last'])
    new_yolo = Path(yolo_result['checkpoint'])
    if digest(new_yolo) != yolo_result['checkpoint_sha256']:
        raise ValueError('Selected new YOLO checkpoint changed')
    cache = root/'comparison'/f'yolo_epoch{cap:03d}.json.gz'
    old_cache = config['yolo_cache']
    results = []
    for native_name, checkpoint in [('old_native', Path(config['initial_checkpoint'])), ('new_native', new_native)]:
        if native_name == 'old_native' and digest(checkpoint) != NATIVE_SHA:
            raise ValueError('Original best84 checkpoint changed')
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        expected_epoch = 84 if native_name == 'old_native' else cap
        if payload.get('epoch') != expected_epoch:
            raise ValueError('Native checkpoint epoch differs from requested stage')
        if native_name == 'new_native' and payload.get('identity') != native_result.get('identity'):
            raise ValueError('Native checkpoint belongs to a different completed training run')
        model = legacy.build_model(encoder='tu-convnext_tiny', pretrained=False)
        model.load_state_dict(payload['model_state_dict'], strict=True)
        del payload
        model = model.to('cuda').eval()
        for yolo_name, yolo_cache in [('old_yolo', old_cache), ('new_yolo', str(cache))]:
            role = yolo_name+'_'+native_name
            cfg = dict(config, yolo_cache=yolo_cache, comparison_role=role)
            print('비교:', role, flush=True)
            metric = evaluate_candidate(model, cfg, f'epoch{cap:03d}_{role}')
            results.append(dict(role=role, native_checkpoint=str(checkpoint),
                native_checkpoint_sha256=digest(checkpoint),
                yolo_checkpoint=str(new_yolo) if yolo_name == 'new_yolo' else config['yolo_checkpoint'],
                **metric))
        del model
        gc.collect(); torch.cuda.empty_cache()
    baseline = next(r for r in results if r['role'] == 'old_yolo_old_native')
    for row in results:
        row['delta_pq_vs_same_run_baseline'] = row['pq']-baseline['pq']
        row['delta_fp_vs_same_run_baseline'] = row['fp']-baseline['fp']
        row['delta_fn_vs_same_run_baseline'] = row['fn']-baseline['fn']
    report = dict(schema_version=1, epoch_per_model=cap, total_schedule_per_model=TOTAL_EPOCHS,
        results=results, selection='fixed requested epoch; native best.pt reported separately',
        scope='Fixed GONG validation91; same inference/finalizer; not leaderboard or causal GONG-only continuation ablation',
        external_observations=695, gong_training_observations=0, validation_observations=91,
        additional_training_domain='CHASE only; no GONG replay or final GONG fine-tuning',
        epoch_sampling='full_pass', configured_samples_per_epoch=695,
        configured_sample_exposures_per_model=cap*695,
        effective_training_samples_per_epoch={
            'yolo': yolo_result.get('effective_steps_per_epoch'),
            'native': native_result.get('effective_steps_per_epoch')},
        original_weights_preserved=True, rule_refiner_trained=False,
        public_reference318='not measured: this run uses validation only')
    atomic_json(root/'comparison'/f'epoch{cap:03d}_comparison.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def package_report(root, result):
    root = Path(root)
    atomic_json(root/'latest_result.json', result)
    archive = root/f'chase_only_both_fullpass_epoch{result["run_until"]:03d}_results.zip'
    pending = archive.with_suffix('.pending.zip')
    excluded = {'native_validation_cache', 'completed', 'code', 'assets', 'training_data',
                'dataset', 'datasets', 'images', 'labels', 'external_CHASE', 'yolo_data'}
    with zipfile.ZipFile(pending, 'w', zipfile.ZIP_DEFLATED) as output:
        for path in sorted(root.rglob('*')):
            if not path.is_file() or any(part in excluded or part.endswith('.stages') for part in path.relative_to(root).parts):
                continue
            if path.suffix not in ('.json', '.csv', '.yaml', '.txt', '.log', '.md'):
                continue
            # Logs can grow indefinitely across runtime restarts; keep last2MiB.
            if path.suffix == '.log' and path.stat().st_size > 2*1024**2:
                with path.open('rb') as stream:
                    stream.seek(-2*1024**2, 2)
                    output.writestr(str(path.relative_to(root)), stream.read())
            elif path.stat().st_size < 8*1024**2:
                output.write(path, str(path.relative_to(root)))
    pending.replace(archive)
    result['archive'] = str(archive)
    result['archive_sha256'] = digest(archive)
    atomic_json(root/'latest_result.json', result)
    return result


def run(settings, cap, arms, phase, *, runner=subprocess.run):
    if not 1 <= cap <= TOTAL_EPOCHS:
        raise ValueError('run-until must be1–60; both models use a fixed60-epoch schedule')
    if not arms or len(set(arms)) != len(arms) or not set(arms) <= {'yolo', 'native'}:
        raise ValueError('Choose unique yolo/native arms')
    root = Path(settings['root']); root.mkdir(parents=True, exist_ok=True)
    code = Path(__file__).resolve().parent
    validate_paths(settings)
    configs = configurations(settings)
    paths = {}
    for name, cfg in configs.items():
        paths[name] = root/'configs'/f'{name}.json'
        atomic_json(paths[name], cfg, unchanged=True)
    atomic_json(root/'experiment.json', dict(schema_version=1, settings=settings,
        epochs_per_model=TOTAL_EPOCHS, source_files={p.name:digest(p) for p in sorted(code.glob('*.py'))},
        baseline_yolo_sha256=YOLO_SHA, baseline_native_sha256=NATIVE_SHA,
        chase_images=695, gong_train_images=0, gong_validation_images=91,
        training_domain='chase_only', epoch_sampling='full_pass', configured_samples_per_epoch=695,
        configured_sample_exposures_per_model=TOTAL_EPOCHS*695, final_gong_finetune=False,
        overlap_policy='exclude explicitly marked cases only; do not infer overlaps from branches',
        preprocessing='CHASE alignment only; GONG existing limb for validation only; fixed tensor normalization',
        model_scope='separate YOLO26s-seg and best84; fixed rule-based finalizer'), unchanged=True)
    if phase != 'prepare':
        child([sys.executable, '-c', 'import torch; assert torch.cuda.is_available(), "Colab GPU runtime required"; print("GPU:",torch.cuda.get_device_name(0),flush=True)'],
              '[1] GPU 확인 — 자료 준비 전에 확인합니다', runner=runner)
    if phase in ('all', 'prepare'):
        child([sys.executable, '-u', str(code/Path(__file__).name), '--worker', 'prepare',
               '--config', str(paths['native'])], '[2] CHASE695장 다운로드/원반 정렬 검증', runner=runner)
    if phase == 'prepare':
        return package_report(root, dict(status='prepared', run_until=cap, arms=arms,
            fixed_epochs_per_model=TOTAL_EPOCHS, comparison_available=False))
    if phase in ('all', 'train'):
        for arm in arms:
            script = 'chase_yolo_training.py' if arm == 'yolo' else 'chase_best84_training.py'
            child([sys.executable, '-u', str(code/script), '--config', str(paths[arm]), '--run-until', str(cap)],
                  f'[3] {arm} 추가 학습: 전체{TOTAL_EPOCHS}, 이번{cap}epoch까지 재개', runner=runner)
    complete_pair = all((root/arm/'result.json').is_file() and
        json.loads((root/arm/'result.json').read_text()).get('epochs_completed') == cap for arm in ('yolo', 'native'))
    comparison_available = False
    if phase in ('all', 'compare') and complete_pair:
        yolo_result = json.loads((root/'yolo/result.json').read_text())
        child([sys.executable, '-u', str(code/'chase_yolo_predict.py'), '--config', str(paths['comparison']),
               '--checkpoint', yolo_result['checkpoint'], '--checkpoint-sha256', yolo_result['checkpoint_sha256'],
               '--output', str(root/'comparison'/f'yolo_epoch{cap:03d}.json.gz')],
              '[4] 추가 학습한 YOLO의 validation91 전체30뷰 추론', runner=runner)
        child([sys.executable, '-u', str(code/Path(__file__).name), '--worker', 'compare',
               '--config', str(paths['comparison']), '--run-until', str(cap)],
              '[5] 기존/새 YOLO × 기존/새 best84 네 조합 평가', runner=runner)
        comparison_available = True
    elif phase == 'compare':
        raise ValueError('Both model results must exist at the requested completed epoch before comparison')
    result = dict(status='complete' if cap == TOTAL_EPOCHS and comparison_available else 'paused',
        run_until=cap, arms=arms, fixed_epochs_per_model=TOTAL_EPOCHS,
        comparison_available=comparison_available,
        comparison_report=str(root/'comparison'/f'epoch{cap:03d}_comparison.json') if comparison_available else None,
        note='Training at60 ends; remaining work is validation inference and four-combination evaluation.')
    return package_report(root, result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings'); parser.add_argument('--config')
    parser.add_argument('--worker', choices=('prepare', 'compare'))
    parser.add_argument('--run-until', type=int, default=TOTAL_EPOCHS)
    parser.add_argument('--arms', default='yolo,native')
    parser.add_argument('--phase', choices=('all', 'prepare', 'train', 'compare'), default='all')
    args = parser.parse_args(argv)
    if args.worker:
        if not args.config: parser.error('--worker requires --config')
        config = json.loads(Path(args.config).read_text())
        if args.worker == 'prepare': prepare_worker(config)
        else: compare_worker(config, args.run_until)
        return
    if not args.settings: parser.error('--settings is required')
    settings = json.loads(Path(args.settings).read_text())
    root = Path(settings['root']); root.mkdir(parents=True, exist_ok=True)
    with (root/'orchestrator.lock').open('a+') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('This two-model experiment is already running')
        result = run(settings, args.run_until, args.arms.split(','), args.phase)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__': main()
