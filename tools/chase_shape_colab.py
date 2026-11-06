"""Small, durable Colab launcher for the best84-recipe CHASE-shape experiment.

This module uses the standard library until dependencies have been checked.
Its saved source bundle contains code and public metadata, never model weights,
images, annotations, split CSVs, or prediction payloads.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import sys
import uuid


SCOPE = 'chase_shape_best84_recipe_t4_v1'
STATE_NAME = 'bootstrap_state.json'
SOURCE_FILES = (
    'tools/chase_shape_colab.py', 'tools/chase_shape_training.py',
    'tools/chase_shape_synthesis.py', 'tools/chase_transfer_data.py',
    'tools/targeted_synthetic_data.py', 'tools/solar_a100_train_standalone.py',
    'tools/chase_best84_training.py', 'tools/chase_independent_decode.py',
    'configs/chase_ahas2024_2025_manifest.json',
    'configs/best84_training_reference.json',
)
DEFAULT_CACHE = '/content/chase_shape_best84_recipe_t4_cache'
BACKGROUND_PROVENANCE = 'external_user_supplied'
BACKGROUND_SOURCE_NOTE = ('User reports an external website image observed to contain no filaments; '
                          'URL not provided.')


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


def immutable_write(path, payload):
    path = Path(path)
    if path.exists():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != payload:
            raise ValueError('Preserving a changed existing file; use a separate output directory: '+str(path))
    else:
        atomic_write(path, payload)


def separate(output, protected):
    output, protected = Path(output).resolve(), Path(protected).resolve()
    if output == protected or output in protected.parents or protected in output.parents:
        raise ValueError('The experiment output must be separate from original data and weights: '+str(protected))


def settings(base, output, code, cache_root=DEFAULT_CACHE, clean_background=None,
             clean_background_provenance=BACKGROUND_PROVENANCE):
    base, output, code, cache_root = map(lambda p: Path(p).resolve(), (base, output, code, cache_root))
    train_root = base/'MAGFiLO_1.0_Kaggle_2026/train'
    splits = base.parent/'SolarFilament_LimbOnly_YOLO26_mixed_FullRes'
    baseline = base.parent/'SolarFilament_NativeUNet_A100/20261028T030808Z_bedf3e21/best.pt'
    chase = base/'external_CHASE/ahas_2024_2025'
    clean = Path(clean_background or base/'clean_solar.jpg').resolve()
    for protected in (train_root, splits, baseline.parent, chase):
        separate(output, protected)
        separate(cache_root, protected)
    separate(output, cache_root)
    separate(output, clean)
    separate(cache_root, clean)
    if cache_root.is_relative_to(Path('/content/drive')):
        raise ValueError('Temporary synthesis caches must be on local Colab disk, outside Drive.')
    return dict(images=str(train_root/'train_images'),
        test_images=str(base/'MAGFiLO_1.0_Kaggle_2026/test/test_images'),
        annotations=str(train_root/'MAGFiLO_1.0_Annotations_kaggle2026_train.json'),
        train_split=str(splits/'train_split.csv'), val_split=str(splits/'val_split.csv'),
        chase_root=str(chase), chase_manifest=str(code/'configs/chase_ahas2024_2025_manifest.json'),
        baseline_checkpoint=str(baseline), cache_root=str(cache_root), output_root=str(output), seed=42,
        clean_background=str(clean), clean_background_provenance=clean_background_provenance,
        clean_background_source_note=BACKGROUND_SOURCE_NOTE)


def _safe_member(name):
    if not isinstance(name, str) or name not in SOURCE_FILES:
        raise ValueError('Unexpected source snapshot member: '+str(name))
    parts = PurePosixPath(name).parts
    if len(parts) != 2 or '\\' in name or parts[0] not in ('tools', 'configs'):
        raise ValueError('Unsafe source snapshot member: '+name)


def restore(output, base=None, cache_root=None, clean_background=None,
            clean_background_provenance=None):
    """Verify every saved source/config before using the durable bundle."""
    output = Path(output).resolve()
    state = json.loads((output/STATE_NAME).read_text())
    if state.get('scope') != SCOPE or state.get('output_root') != str(output):
        raise ValueError('Saved bootstrap belongs to a different experiment directory.')
    if base is not None and state.get('base') != str(Path(base).resolve()):
        raise ValueError('Saved BASE differs. Preserve the run or use a new output directory.')
    version = state.get('version', '')
    if not re.fullmatch('[0-9a-f]{16}', version):
        raise ValueError('Invalid saved source version.')
    code = output/'code'/version
    manifest_path = code/'source_manifest.json'
    if code.is_symlink() or not code.resolve().is_relative_to(output):
        raise ValueError('Saved source directory escapes the experiment.')
    if manifest_path.is_symlink() or sha(manifest_path) != state['manifest_sha256']:
        raise ValueError('Saved source manifest checksum changed.')
    manifest = json.loads(manifest_path.read_text())
    if set(manifest) != set(SOURCE_FILES):
        raise ValueError('Saved source manifest has missing or unexpected files.')
    for name, expected in manifest.items():
        _safe_member(name)
        path = code/name
        if path.is_symlink() or not path.resolve().is_relative_to(code.resolve()):
            raise ValueError('Saved source path escapes the source snapshot.')
        if not path.is_file() or sha(path) != expected:
            raise ValueError('Saved source checksum changed: '+name)
    if hashlib.sha256(canonical(manifest)).hexdigest()[:16] != version:
        raise ValueError('Saved source version does not match its manifest.')
    config_path = output/'experiment_config.json'
    if config_path.is_symlink() or sha(config_path) != state['config_sha256']:
        raise ValueError('Saved experiment configuration checksum changed.')
    config = json.loads(config_path.read_text())
    expected = settings(state['base'], output, code, cache_root or config['cache_root'],
                        clean_background or config['clean_background'],
                        clean_background_provenance or config['clean_background_provenance'])
    if config != expected:
        raise ValueError('Saved configuration differs from the requested paths/settings.')
    return config, config_path, code, state


def install(repository, base, output, cache_root=DEFAULT_CACHE, clean_background=None,
            clean_background_provenance=BACKGROUND_PROVENANCE):
    """Pin and copy a complete small source bundle; preserve existing runs."""
    repository, output = Path(repository).resolve(), Path(output).resolve()
    if (output/STATE_NAME).exists():
        return restore(output, base, cache_root, clean_background, clean_background_provenance)
    # Refuse to attach a fresh bootstrap to an unidentified training directory.
    if any((output/name).exists() for name in ('train', 'training_state.json', 'last.pt', 'best.pt', 'latest_result.json')):
        raise ValueError('Existing training outputs have no bootstrap identity; use a new output directory.')
    payloads = {}
    for name in SOURCE_FILES:
        _safe_member(name)
        path = repository/name
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(repository):
            raise FileNotFoundError('Required source/config is missing: '+str(path))
        payload = path.read_bytes()
        if name.endswith('.py'):
            compile(payload, name, 'exec')
        else:
            json.loads(payload)
        payloads[name] = payload
    manifest = {name:hashlib.sha256(value).hexdigest() for name,value in payloads.items()}
    version = hashlib.sha256(canonical(manifest)).hexdigest()[:16]
    code = output/'code'/version
    config = settings(base, output, code, cache_root, clean_background, clean_background_provenance)
    config_bytes = canonical(config)+b'\n'
    manifest_bytes = canonical(manifest)+b'\n'
    try:
        commit = subprocess.check_output(['git', '-C', str(repository), 'rev-parse', 'HEAD'], text=True).strip()
        if not re.fullmatch('[0-9a-f]{40,64}', commit):
            raise ValueError('Unexpected source commit.')
        dirty = bool(subprocess.check_output(['git', '-C', str(repository), 'status', '--porcelain'], text=True).strip())
    except subprocess.CalledProcessError:
        commit, dirty = None, None
    state = dict(schema_version=1, scope=SCOPE, base=str(Path(base).resolve()),
        output_root=str(output), version=version, source_commit=commit,
        source_worktree_dirty=dirty, manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        bootstrap='tools/chase_shape_colab.py')
    writes = {code/name:payload for name,payload in payloads.items()}
    writes.update({code/'source_manifest.json':manifest_bytes,
                   output/'experiment_config.json':config_bytes,
                   output/STATE_NAME:canonical(state)+b'\n'})
    # Inspect all existing paths before writing any payload.
    for path,payload in writes.items():
        if path.is_symlink() or not path.resolve().is_relative_to(output):
            raise ValueError('Refusing an unsafe snapshot destination: '+str(path))
        if path.exists() and (not path.is_file() or path.read_bytes()!=payload):
            raise ValueError('Preserving a changed existing file: '+str(path))
    for path,payload in writes.items():
        immutable_write(path,payload)
    return restore(output, base, cache_root, clean_background, clean_background_provenance)


def ensure_dependencies(output):
    """Retain Colab's GPU torch/torchvision and installed NumPy ABI."""
    preserved = {}
    for package in ('torch', 'torchvision'):
        try:
            preserved[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError('Use a Colab runtime with its provided torch and torchvision.') from exc
    try:
        preserved['numpy'] = importlib.metadata.version('numpy')
    except importlib.metadata.PackageNotFoundError:
        pass
    required = []
    for package,version in [('segmentation-models-pytorch','0.5.0'),('timm','1.0.20'),('pycocotools','2.0.11')]:
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        if actual != version:
            required.append(package+'=='+version)
    for module,package in [('numpy','numpy'),('cv2','opencv-python-headless'),('scipy','scipy'),('requests','requests')]:
        if importlib.util.find_spec(module) is None:
            required.append(package)
    if required:
        constraint = Path(output)/'keep_colab_runtime.txt'
        constraint.write_text(''.join(name+'=='+version+'\n' for name,version in preserved.items()))
        print('Installing additional packages while preserving Colab torch/torchvision:', ', '.join(required), flush=True)
        subprocess.run([sys.executable,'-m','pip','install','--upgrade-strategy','only-if-needed',
                        '-c',str(constraint),*required],check=True)
    if any(importlib.metadata.version(name)!=version for name,version in preserved.items()):
        raise RuntimeError('A preserved Colab package changed during installation.')


def tail(path, lines=20):
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
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def monitored(command, log_path, cwd, env):
    """Bounded notebook output; the full child traceback remains on Drive."""
    log_path = Path(log_path)
    print('Full log:', log_path, flush=True)
    with log_path.open('ab',buffering=0) as stream:
        process = subprocess.Popen(command,cwd=cwd,env=env,stdout=stream,
                                   stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while True:
                try:
                    status=process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    print(tail(log_path,8),flush=True)
            if status:
                print(tail(log_path,40),flush=True)
                raise subprocess.CalledProcessError(status,command)
        finally:
            stop_group(process)
    print(tail(log_path,12),flush=True)


def checked_result(output):
    output = Path(output).resolve()
    path = output/'latest_result.json'
    if not path.is_file():
        raise FileNotFoundError('The worker did not save its result: '+str(path))
    result = json.loads(path.read_text())
    if result.get('output_root') is not None and Path(result['output_root']).resolve()!=output:
        raise ValueError('Result belongs to another output directory.')
    if result.get('archive'):
        archive=Path(result['archive']).resolve()
        if not archive.is_relative_to(output) or not archive.is_file():
            raise ValueError('Result archive is missing or outside this experiment.')
        if sha(archive)!=result.get('archive_sha256'):
            raise ValueError('Result archive checksum changed.')
    return result


def run(base, output, phase='all', run_until=100, cache_root=DEFAULT_CACHE, repository=None,
        clean_background=None, clean_background_provenance=BACKGROUND_PROVENANCE):
    if phase not in ('prepare','train','all') or type(run_until) is not int or not 1<=run_until<=100:
        raise ValueError('Use phase prepare/train/all and run-until between 1 and 100; schedule remains 100.')
    output = Path(output).resolve()
    if (output/STATE_NAME).is_file():
        config,config_path,code,state=restore(output,base,cache_root,clean_background,clean_background_provenance)
    else:
        repository=Path(repository).resolve() if repository else Path(__file__).resolve().parents[1]
        config,config_path,code,state=install(repository,base,output,cache_root,clean_background,clean_background_provenance)
    for key in ('annotations','train_split','val_split','clean_background'):
        if not Path(config[key]).is_file():
            raise FileNotFoundError(key+': '+config[key])
    if not Path(config['images']).is_dir():
        raise FileNotFoundError('GONG train_images: '+config['images'])
    if phase!='prepare' and not Path(config['baseline_checkpoint']).is_file():
        raise FileNotFoundError('Original best84 checkpoint for evaluation only: '+config['baseline_checkpoint'])
    print('Fresh ImageNet ConvNeXt-Tiny + new U-Net decoder; no best84 weight initialization.',flush=True)
    print('Fixed schedule: 100 epochs; current stage:',run_until,'; source bundle:',state['version'],flush=True)
    ensure_dependencies(output)
    env=dict(os.environ,PYTHONUNBUFFERED='1',TORCH_CUDNN_V8_API_DISABLED='1',
             CUBLAS_WORKSPACE_CONFIG=':4096:8')
    log_path=output/'training.log'
    if phase!='prepare':
        probe="""import torch, torchvision
from chase_best84_training import configure_cuda
assert torch.cuda.is_available(), 'Select a Colab T4 GPU runtime.'
configure_cuda()
print('PyTorch:',torch.__version__,'CUDA:',torch.version.cuda,'GPU:',torch.cuda.get_device_name(0),flush=True)
layer=torch.nn.Conv2d(3,3,3,padding=1,groups=3).cuda()
with torch.autocast('cuda',dtype=torch.float16):
    out=layer(torch.randn(1,3,64,64,device='cuda'))
    loss=out.float().square().mean()
loss.backward()
assert torch.isfinite(out).all() and out.dtype==torch.float16
torch.cuda.synchronize()
print('CUDA FP16 depthwise forward/backward check passed.',flush=True)
"""
        monitored([sys.executable,'-u','-c',probe],log_path,code/'tools',env)
    command=[sys.executable,'-u',str(code/'tools/chase_shape_training.py'),
             '--config',str(config_path),'--phase',phase,'--run-until',str(run_until)]
    monitored(command,log_path,code/'tools',env)
    result=checked_result(output)
    print('Status:',result.get('status'),'Completed epoch:',result.get('epochs_completed',0),flush=True)
    if result.get('archive'):
        print('Review ZIP:',result['archive'],flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=Path('/content/drive/MyDrive/filament-segmentation-2026'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--phase',choices=('prepare','train','all'),default='all')
    parser.add_argument('--run-until',type=int,default=100)
    parser.add_argument('--cache-root',default=DEFAULT_CACHE)
    parser.add_argument('--clean-background',type=Path)
    parser.add_argument('--clean-background-provenance',default=BACKGROUND_PROVENANCE)
    args=parser.parse_args()
    output=args.output or args.base/'fixed_split_experiments'/SCOPE
    def interrupt(signum,frame):
        raise KeyboardInterrupt('Interrupted by signal '+str(signum))
    for number in (signal.SIGTERM,signal.SIGINT):
        signal.signal(number,interrupt)
    run(args.base,output,args.phase,args.run_until,args.cache_root,
        clean_background=args.clean_background,clean_background_provenance=args.clean_background_provenance)


if __name__=='__main__':
    main()
