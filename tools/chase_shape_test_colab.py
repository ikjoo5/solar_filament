"""Durable Colab launcher for epoch82 native-only test inference.

Only source code and launcher settings are copied. Training, synthesis, weights,
test pixels and prediction inputs are never bundled or modified by this launcher.
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
import uuid


SCOPE = 'chase_shape_epoch082_native_t4_v1'
STATE_NAME = 'test_bootstrap_state.json'
SOURCE_FILES = (
    'tools/chase_shape_test_colab.py', 'tools/chase_shape_test_inference.py',
    'tools/solar_a100_train_standalone.py', 'tools/chase_best84_training.py',
    'tools/chase_independent_decode.py',
)
BASE = Path('/content/drive/MyDrive/filament-segmentation-2026')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def atomic_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name+'.'+uuid.uuid4().hex+'.pending')
    try:
        pending.write_bytes(payload)
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def separate(output, protected):
    output, protected = Path(output).resolve(), Path(protected).resolve()
    if output == protected or output in protected.parents or protected in output.parents:
        raise ValueError('Inference output must be separate from training/test inputs: '+str(protected))


def settings(training_root, test_images, output):
    training_root, test_images, output = map(lambda p: Path(p).resolve(),
                                            (training_root, test_images, output))
    separate(output, training_root)
    separate(output, test_images)
    return dict(scope=SCOPE, training_root=str(training_root), test_images=str(test_images),
                output_root=str(output), expected_epoch=82, expected_images=180,
                inference_only=True, native_only=True)


def restore(output, training_root=None, test_images=None):
    """Restore an existing inference snapshot without touching training state."""
    output = Path(output).resolve()
    pointer = output/STATE_NAME
    if pointer.is_symlink():
        raise ValueError('Unsafe inference bootstrap pointer')
    state = json.loads(pointer.read_text())
    if state.get('scope') != SCOPE or state.get('output_root') != str(output):
        raise ValueError('Saved bootstrap belongs to another inference experiment')
    version = state.get('version', '')
    if not re.fullmatch('[0-9a-f]{16}', version):
        raise ValueError('Invalid source version')
    code = output/'code'/version
    manifest_path = code/'source_manifest.json'
    if code.is_symlink() or not code.resolve().is_relative_to(output):
        raise ValueError('Unsafe source snapshot directory')
    if manifest_path.is_symlink() or sha(manifest_path) != state.get('manifest_sha256'):
        raise ValueError('Saved source manifest checksum differs')
    manifest = json.loads(manifest_path.read_text())
    if set(manifest) != set(SOURCE_FILES) or hashlib.sha256(canonical(manifest)).hexdigest()[:16] != version:
        raise ValueError('Saved source inventory/version differs')
    for name, expected in manifest.items():
        path = code/name
        if path.is_symlink() or not path.resolve().is_relative_to(code.resolve()):
            raise ValueError('Unsafe saved source path: '+name)
        if not path.is_file() or sha(path) != expected:
            raise ValueError('Saved source checksum differs: '+name)
    config_path = output/'inference_launcher_config.json'
    if config_path.is_symlink() or sha(config_path) != state.get('config_sha256'):
        raise ValueError('Saved inference settings checksum differs')
    config = json.loads(config_path.read_text())
    wanted = settings(training_root or config['training_root'],
                      test_images or config['test_images'], output)
    if config != wanted:
        raise ValueError('Inference paths/settings changed; use the saved paths or a new output')
    return config, code, state


def install(repository, training_root, test_images, output):
    """Pin a small complete source bundle on Drive on the first invocation."""
    repository, output = Path(repository).resolve(), Path(output).resolve()
    if (output/STATE_NAME).exists():
        return restore(output, training_root, test_images)
    if any((output/name).exists() for name in ('identity.json', 'latest_result.json', 'cache', 'train')):
        raise ValueError('Existing results lack an inference bootstrap identity; preserve them and use a new output')
    config = settings(training_root, test_images, output)
    payloads = {}
    for name in SOURCE_FILES:
        path = repository/name
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(repository):
            raise FileNotFoundError('Missing inference source: '+str(path))
        payload = path.read_bytes()
        compile(payload, name, 'exec')
        payloads[name] = payload
    manifest = {name:hashlib.sha256(raw).hexdigest() for name,raw in payloads.items()}
    version = hashlib.sha256(canonical(manifest)).hexdigest()[:16]
    code = output/'code'/version
    config_bytes, manifest_bytes = canonical(config)+b'\n', canonical(manifest)+b'\n'
    try:
        commit = subprocess.check_output(['git','-C',str(repository),'rev-parse','HEAD'],text=True).strip()
        if not re.fullmatch('[0-9a-f]{40,64}',commit):
            raise ValueError('Invalid repository commit')
        dirty = bool(subprocess.check_output(['git','-C',str(repository),'status','--porcelain'],text=True).strip())
    except subprocess.CalledProcessError:
        commit, dirty = None, None
    state = dict(scope=SCOPE, version=version, output_root=str(output),
        source_commit=commit, source_worktree_dirty=dirty,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        config_sha256=hashlib.sha256(config_bytes).hexdigest())
    writes = {code/name:raw for name,raw in payloads.items()}
    writes.update({code/'source_manifest.json':manifest_bytes,
                   output/'inference_launcher_config.json':config_bytes,
                   output/STATE_NAME:canonical(state)+b'\n'})
    for path,raw in writes.items():
        if path.is_symlink() or not path.resolve().is_relative_to(output):
            raise ValueError('Unsafe snapshot destination: '+str(path))
        if path.exists() and (not path.is_file() or path.read_bytes()!=raw):
            raise ValueError('Preserving changed existing file: '+str(path))
    for path,raw in writes.items():
        if not path.exists():
            atomic_write(path,raw)
    return restore(output, training_root, test_images)


def ensure_dependencies(output):
    """Keep Colab's installed CUDA torch/torchvision and NumPy ABI unchanged."""
    preserved = {}
    for package in ('torch','torchvision'):
        try:
            preserved[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError('Select a Colab GPU runtime with its provided torch/torchvision') from error
    try:
        preserved['numpy'] = importlib.metadata.version('numpy')
    except importlib.metadata.PackageNotFoundError:
        pass
    needed = []
    for package,version in [('segmentation-models-pytorch','0.5.0'),('timm','1.0.20'),('pycocotools','2.0.11')]:
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        if actual != version:
            needed.append(package+'=='+version)
    for module,package in [('numpy','numpy'),('cv2','opencv-python-headless'),('scipy','scipy')]:
        if importlib.util.find_spec(module) is None:
            needed.append(package)
    if needed:
        constraint = Path(output)/'keep_colab_runtime.txt'
        constraint.write_text(''.join(k+'=='+v+'\n' for k,v in preserved.items()))
        print('Installing inference dependencies:', ', '.join(needed),flush=True)
        subprocess.run([sys.executable,'-m','pip','install','--upgrade-strategy','only-if-needed',
                        '-c',str(constraint),*needed],check=True)
    if any(importlib.metadata.version(k)!=v for k,v in preserved.items()):
        raise RuntimeError('A preserved Colab runtime package changed during installation')


def tail(path, lines=12):
    path = Path(path)
    if not path.is_file():
        return ''
    with path.open('rb') as stream:
        stream.seek(max(0,path.stat().st_size-20000))
        return '\n'.join(stream.read().decode('utf-8','replace').splitlines()[-lines:])


def stop_group(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def monitored(command, log_path, cwd, env):
    print('Full inference log:',log_path,flush=True)
    with Path(log_path).open('ab',buffering=0) as stream:
        process = subprocess.Popen(command,cwd=cwd,env=env,stdout=stream,
                                   stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while True:
                try:
                    status = process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    print(tail(log_path,8),flush=True)
            if status:
                print(tail(log_path,40),flush=True)
                raise subprocess.CalledProcessError(status,command)
        finally:
            stop_group(process)
    print(tail(log_path),flush=True)


def checked_result(output):
    output = Path(output).resolve()
    result = json.loads((output/'latest_result.json').read_text())
    if Path(result.get('output_root','')).resolve()!=output:
        raise ValueError('Inference result belongs to another output directory')
    if result.get('status') != 'complete' or result.get('images_completed') != 180:
        raise RuntimeError('All 180 images must complete before a submission is downloaded')
    if result.get('checkpoint_epoch') != 82:
        raise ValueError('Result is not from the selected epoch82 checkpoint')
    for key in ('csv','archive'):
        path = Path(result[key]).resolve()
        if not path.is_relative_to(output) or not path.is_file() or Path(result[key]).is_symlink():
            raise ValueError('Missing or unsafe inference result: '+key)
        if sha(path)!=result.get(key+'_sha256'):
            raise ValueError('Inference result checksum differs: '+key)
    return result


def run(training_root, test_images, output, repository=None):
    output = Path(output).resolve()
    if (output/STATE_NAME).is_file():
        config,code,state = restore(output,training_root,test_images)
    else:
        repository = Path(repository).resolve() if repository else Path(__file__).resolve().parents[1]
        config,code,state = install(repository,training_root,test_images,output)
    training = Path(config['training_root'])
    for name in ('train/state.json','train/config.json','train/identity.json'):
        if not (training/name).is_file():
            raise FileNotFoundError('Training metadata needed to identify best82: '+str(training/name))
    if not Path(config['test_images']).is_dir():
        raise FileNotFoundError('Original test_images directory: '+config['test_images'])
    print('Inference only: native epoch82, limb-only, threshold0.35, area300, official180images.',flush=True)
    print('Saved source version:',state['version'],flush=True)
    ensure_dependencies(output)
    env = dict(os.environ,PYTHONUNBUFFERED='1',TORCH_CUDNN_V8_API_DISABLED='1',
               CUBLAS_WORKSPACE_CONFIG=':4096:8')
    command = [sys.executable,'-u',str(code/'tools/chase_shape_test_inference.py'),
        '--training-root',config['training_root'],'--test-images',config['test_images'],
        '--output',str(output),'--result',str(output/'latest_result.json')]
    monitored(command,output/'inference.log',code/'tools',env)
    result = checked_result(output)
    print('Submission CSV:',result['csv'],flush=True)
    print('Review ZIP:',result['archive'],flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-root',type=Path,default=BASE/'fixed_split_experiments/chase_shape_best84_recipe_t4_v1')
    parser.add_argument('--test-images',type=Path,default=BASE/'MAGFiLO_1.0_Kaggle_2026/test/test_images')
    parser.add_argument('--output',type=Path,default=BASE/'submissions'/SCOPE)
    args = parser.parse_args()
    def interrupt(number,frame):
        raise KeyboardInterrupt('Interrupted by signal '+str(number))
    for number in (signal.SIGINT,signal.SIGTERM):
        signal.signal(number,interrupt)
    run(args.training_root,args.test_images,args.output)


if __name__ == '__main__':
    main()
