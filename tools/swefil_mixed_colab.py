"""Durable Colab entry for fresh GONG + all annotated SWEFil + CHASE training.

Only source and small configuration are snapshotted here. Training state is
published and read back with the Drive API by the worker checkpoint store.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

import chase_shape_colab as previous

SCOPE = 'swefil_all_mixed_native_t4_v1'
STATE_NAME = 'bootstrap_state.json'
DEFAULT_CACHE = '/content/swefil_all_mixed_native_t4_cache'
SOURCE_FILES = tuple(dict.fromkeys((
    'tools/swefil_mixed_colab.py', 'tools/swefil_mixed_training.py',
    'tools/swefil_training_data.py', 'tools/verified_drive_checkpoints.py',
    'tools/chase_shape_cache.py', *previous.SOURCE_FILES,
)))
sha = previous.sha
canonical = previous.canonical
immutable_write = previous.immutable_write
stop_group = previous.stop_group
checked_result = previous.checked_result


def settings(base, output, cache_root=DEFAULT_CACHE, chase_training_root=None):
    base, output, cache = (Path(p).resolve() for p in (base, output, cache_root))
    old = Path(chase_training_root or base/'fixed_split_experiments/chase_shape_best84_recipe_t4_v1').resolve()
    if not output.is_relative_to(base) or output == base:
        raise ValueError('The new experiment output must be inside the project BASE.')
    for protected in (old, base/'MAGFiLO_1.0_Kaggle_2026', base/'external_CHASE',
                      base/'external_SWEFil', base/'clean_solar.jpg'):
        previous.separate(output, protected)
        previous.separate(cache, protected)
    previous.separate(output, cache)
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
    if any((output/name).exists() for name in ('train', 'verified_checkpoints', 'latest_result.json')):
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
                 bootstrap='tools/swefil_mixed_colab.py')
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
    previous.ensure_dependencies(output)
    required = []
    for module, package in (('googleapiclient', 'google-api-python-client'),
                            ('google.auth', 'google-auth'),
                            ('google_auth_httplib2', 'google-auth-httplib2')):
        try:
            present = importlib.util.find_spec(module) is not None
        except (ModuleNotFoundError, ValueError):
            present = False
        if not present:
            required.append(package)
    if required:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade-strategy',
                        'only-if-needed', *required], check=True)


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
        repository=None, chase_training_root=None):
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
    print('Fresh ImageNet ConvNeXt-Tiny + new U-Net decoder; fixed 100-epoch schedule.', flush=True)
    print('All 250 annotated SWEFil observations are admitted. Source bundle:', state['version'], flush=True)
    print('Epochs 4–94: GONG 30%, SWEFil 60%, CHASE synthetic 10% '
          '(expected 120/240/40 of 400 random crops). Epochs 1–3 and 95–100: GONG only.', flush=True)
    print('Old CHASE synthesis is reused; old model weights are not loaded.', flush=True)
    print('Checkpoints: immutable generations, Drive API upload + server read-back verification.', flush=True)
    ensure_dependencies(output)
    env = dict(os.environ, PYTHONUNBUFFERED='1', TORCH_CUDNN_V8_API_DISABLED='1',
               CUBLAS_WORKSPACE_CONFIG=':4096:8')
    command = [sys.executable, '-u', str(code/'tools/swefil_mixed_training.py'),
               '--config', str(config_path), '--phase', phase, '--run-until', str(run_until)]
    monitored(command, output/'training.log', code/'tools', env)
    result = checked_result(output)
    print('Status:', result.get('status'), 'Completed epoch:', result.get('epochs_completed', 0), flush=True)
    if result.get('archive'):
        print('Review ZIP:', result['archive'], flush=True)
    if result.get('best_checkpoint_url'):
        print('Verified best weights:', result['best_checkpoint_url'], flush=True)
    if result.get('last_checkpoint_url'):
        print('Verified resumable last weights:', result['last_checkpoint_url'], flush=True)
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
    args = parser.parse_args()
    run(args.base, args.output or args.base/'fixed_split_experiments'/SCOPE,
        args.phase, args.run_until, args.cache_root, chase_training_root=args.chase_training_root)


if __name__ == '__main__':
    main()
