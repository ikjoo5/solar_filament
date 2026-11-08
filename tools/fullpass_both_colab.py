"""Single-cell, durable Colab entry for full-coverage YOLO and native training.

Every epoch visits all four 1024 tiles of every admitted image. The two models
train sequentially on one T4. Saved source is reused after a runtime reset.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

import chase_shape_colab as common
import swefil_mixed_colab as previous

SCOPE = 'fullpass_both_t4_v1'
STATE_NAME = 'bootstrap_state.json'
DEFAULT_CACHE = '/content/fullpass_both_t4_cache'
ULTRALYTICS_VERSION = '8.4.182'
SOURCE_FILES = tuple(dict.fromkeys((
    'tools/fullpass_both_colab.py', 'tools/fullpass_both_training.py',
    'tools/fullpass_training_data.py', 'tools/fullpass_native_training.py',
    'tools/fullpass_yolo_training.py', 'tools/fullpass_checkpoint_store.py',
    'tools/chase_yolo_training.py', 'tools/targeted_synthetic_training.py',
    *previous.SOURCE_FILES,
)))
sha = common.sha
canonical = common.canonical
immutable_write = common.immutable_write
stop_group = common.stop_group
checked_result = common.checked_result


def settings(base, output, cache_root=DEFAULT_CACHE, chase_training_root=None):
    base, output, cache = (Path(p).resolve() for p in (base, output, cache_root))
    old = Path(chase_training_root or base/'fixed_split_experiments/chase_shape_best84_recipe_t4_v1').resolve()
    if not output.is_relative_to(base) or output == base:
        raise ValueError('The new experiment output must be inside the project BASE.')
    for protected in (old, base/'MAGFiLO_1.0_Kaggle_2026', base/'external_CHASE',
                      base/'external_SWEFil', base/'clean_solar.jpg',
                      base/'fixed_split_experiments/swefil_all_mixed_native_t4_v1'):
        common.separate(output, protected)
        common.separate(cache, protected)
    common.separate(output, cache)
    if cache.is_relative_to('/content/drive'):
        raise ValueError('Use local Colab disk for cache_root, outside Drive.')
    return dict(base=str(base), output_root=str(output), cache_root=str(cache),
                chase_training_root=str(old),
                drive_checkpoint_path_parts=[base.name, *output.relative_to(base).parts, 'verified_checkpoints'])


def restore(output, base=None, cache_root=None, chase_training_root=None):
    output = Path(output).resolve()
    state_path = output/STATE_NAME
    if state_path.is_symlink():
        raise ValueError('Unsafe saved bootstrap state.')
    state = json.loads(state_path.read_text())
    if state.get('scope') != SCOPE or state.get('output_root') != str(output):
        raise ValueError('Saved bootstrap belongs to a different experiment.')
    if base is not None and state.get('base') != str(Path(base).resolve()):
        raise ValueError('Saved BASE differs; preserve this run and use a new directory.')
    version = state.get('version', '')
    if not re.fullmatch('[0-9a-f]{16}', version):
        raise ValueError('Invalid source version.')
    code = output/'code'/version
    manifest_path = code/'source_manifest.json'
    if code.is_symlink() or not code.resolve().is_relative_to(output):
        raise ValueError('Saved source directory escapes the experiment.')
    if manifest_path.is_symlink() or sha(manifest_path) != state['manifest_sha256']:
        raise ValueError('Saved source manifest checksum changed.')
    manifest = json.loads(manifest_path.read_text())
    if set(manifest) != set(SOURCE_FILES):
        raise ValueError('Missing or unexpected source snapshot members.')
    for name, expected in manifest.items():
        path = code/name
        if path.is_symlink() or not path.resolve().is_relative_to(code.resolve()):
            raise ValueError('Saved source path escapes the snapshot.')
        if not path.is_file() or sha(path) != expected:
            raise ValueError('Saved source checksum changed: '+name)
    if hashlib.sha256(canonical(manifest)).hexdigest()[:16] != version:
        raise ValueError('Source version differs from its manifest.')
    config_path = output/'experiment_config.json'
    if config_path.is_symlink() or sha(config_path) != state['config_sha256']:
        raise ValueError('Saved configuration checksum changed.')
    config = json.loads(config_path.read_text())
    expected = settings(state['base'], output, cache_root or config['cache_root'],
                        chase_training_root or config['chase_training_root'])
    if config != expected:
        raise ValueError('Saved configuration differs from requested paths/settings.')
    return config, config_path, code, state


def install(repository, base, output, cache_root=DEFAULT_CACHE, chase_training_root=None):
    repository, output = Path(repository).resolve(), Path(output).resolve()
    if (output/STATE_NAME).exists():
        return restore(output, base, cache_root, chase_training_root)
    if any((output/name).exists() for name in ('train', 'yolo', 'unet', 'verified_checkpoints', 'latest_result.json')):
        raise ValueError('Existing training output has no bootstrap identity; choose a new output.')
    config = settings(base, output, cache_root, chase_training_root)
    payloads = {}
    for name in SOURCE_FILES:
        path = repository/name
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(repository):
            raise FileNotFoundError('Missing or unsafe source: '+str(path))
        payload = path.read_bytes()
        if name.endswith('.py'):
            compile(payload, name, 'exec')
        else:
            json.loads(payload)
        payloads[name] = payload
    manifest = {name: hashlib.sha256(raw).hexdigest() for name, raw in payloads.items()}
    version = hashlib.sha256(canonical(manifest)).hexdigest()[:16]
    code = output/'code'/version
    config_bytes, manifest_bytes = canonical(config)+b'\n', canonical(manifest)+b'\n'
    try:
        commit = subprocess.check_output(['git', '-C', str(repository), 'rev-parse', 'HEAD'], text=True).strip()
        if not re.fullmatch('[0-9a-f]{40,64}', commit):
            raise ValueError('Invalid source commit.')
        dirty = bool(subprocess.check_output(['git', '-C', str(repository), 'status', '--porcelain'], text=True).strip())
    except subprocess.CalledProcessError:
        commit, dirty = None, None
    state = dict(schema_version=1, scope=SCOPE, base=config['base'], output_root=str(output),
                 version=version, source_commit=commit, source_worktree_dirty=dirty,
                 manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                 config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                 bootstrap='tools/fullpass_both_colab.py')
    writes = {code/name: payload for name, payload in payloads.items()}
    writes.update({code/'source_manifest.json': manifest_bytes,
                   output/'experiment_config.json': config_bytes,
                   output/STATE_NAME: canonical(state)+b'\n'})
    for path, payload in writes.items():
        if path.is_symlink() or not path.resolve().is_relative_to(output):
            raise ValueError('Unsafe snapshot destination: '+str(path))
        if path.exists() and (not path.is_file() or path.read_bytes() != payload):
            raise ValueError('Preserving changed existing file: '+str(path))
    for path, payload in writes.items():
        immutable_write(path, payload)
    return restore(output, base, cache_root, chase_training_root)


def ensure_dependencies(output):
    """Install pinned YOLO/native dependencies without replacing Colab's CUDA ABI."""
    previous.ensure_dependencies(output)
    try:
        actual = importlib.metadata.version('ultralytics')
    except importlib.metadata.PackageNotFoundError:
        actual = None
    if actual == ULTRALYTICS_VERSION:
        return
    preserved = {name: importlib.metadata.version(name)
                 for name in ('torch', 'torchvision', 'numpy')}
    constraint = Path(output)/'keep_colab_runtime.txt'
    constraint.write_text(''.join(name+'=='+version+'\n' for name,version in preserved.items()))
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade-strategy',
                    'only-if-needed', '-c', str(constraint),
                    'ultralytics=='+ULTRALYTICS_VERSION], check=True)
    if any(importlib.metadata.version(name) != version for name,version in preserved.items()):
        raise RuntimeError('A preserved Colab runtime package changed during installation.')


def selected_models(models):
    if isinstance(models, str):
        models = tuple(models.split(','))
    else:
        models = tuple(models)
    if not models or len(set(models)) != len(models) or any(name not in ('yolo', 'unet') for name in models):
        raise ValueError('Select yolo,unet once each, or one of these models.')
    return models

def monitored(command, log_path, cwd, env):
    """Relay each worker line to Colab and append the same output to Drive."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print('Full log:', log_path, flush=True)
    with log_path.open('a', encoding='utf-8', buffering=1) as log:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                                   errors='replace', bufsize=1, start_new_session=True)
        try:
            for line in process.stdout:
                log.write(line)
                print(line, end='', flush=True)
            status = process.wait()
            if status:
                raise subprocess.CalledProcessError(status, command)
        finally:
            stop_group(process)
            process.stdout.close()


def run(base, output, phase='all', run_until=100, cache_root=DEFAULT_CACHE,
        repository=None, chase_training_root=None, models=('yolo', 'unet')):
    models = selected_models(models)
    if phase not in ('prepare', 'train', 'all') or type(run_until) is not int or not 1 <= run_until <= 100:
        raise ValueError('Use prepare/train/all and run-until 1..100; the full schedule is 100.')
    output = Path(output).resolve()
    if (output/STATE_NAME).is_file():
        config, config_path, code, state = restore(output, base, cache_root, chase_training_root)
    else:
        repository = Path(repository or Path(__file__).resolve().parents[1])
        config, config_path, code, state = install(repository, base, output, cache_root, chase_training_root)
    old = Path(config['chase_training_root'])
    for name in ('experiment_config.json', 'prepared_summary.json'):
        if not (old/name).is_file():
            raise FileNotFoundError('Existing CHASE synthesis metadata is required: '+str(old/name))
    print('Fresh solar training: COCO YOLO26s-seg and ImageNet ConvNeXt-Tiny + new U-Net decoder.', flush=True)
    print('All 616 GONG + 250 SWEFil + 616 CHASE synthetic images every epoch.', flush=True)
    print('Four nonoverlapping 1024 tiles per image: 5,928 tiles per model per epoch.', flush=True)
    print('Model order:', ', '.join(models), '| 100 epochs each; current stage cap:', run_until, flush=True)
    print('No old solar model weights are loaded. Existing verified image caches are reused.', flush=True)
    print('Source bundle:', state['version'], '| checkpoints use Drive API read-back verification.', flush=True)
    ensure_dependencies(output)
    env = dict(os.environ, PYTHONUNBUFFERED='1', TORCH_CUDNN_V8_API_DISABLED='1',
               CUBLAS_WORKSPACE_CONFIG=':4096:8')
    command = [sys.executable, '-u', str(code/'tools/fullpass_both_training.py'),
               '--config', str(config_path), '--phase', phase, '--run-until', str(run_until),
               '--models', ','.join(models)]
    monitored(command, output/'training.log', code/'tools', env)
    result = checked_result(output)
    print('Status:', result.get('status'), 'Completed epochs by model:', result.get('epochs_completed', {}), flush=True)
    if result.get('archive'):
        print('Review ZIP:', result['archive'], flush=True)
    return result


def main():
    # The notebook terminates this bootstrap's group on Stop. The training worker
    # owns another group, so unwind monitored() rather than orphaning that worker.
    def interrupted(signum, frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path('/content/drive/MyDrive/filament-segmentation-2026'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--phase', choices=('prepare', 'train', 'all'), default='all')
    parser.add_argument('--run-until', type=int, default=100)
    parser.add_argument('--cache-root', default=DEFAULT_CACHE)
    parser.add_argument('--chase-training-root', type=Path)
    parser.add_argument('--models', default='yolo,unet', help='Sequential model order, or yolo/unet alone.')
    args = parser.parse_args()
    run(args.base, args.output or args.base/'fixed_split_experiments'/SCOPE,
        args.phase, args.run_until, args.cache_root, chase_training_root=args.chase_training_root,
        models=args.models)


if __name__ == '__main__':
    main()
