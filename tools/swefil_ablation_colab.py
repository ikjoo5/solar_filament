"""Resumable Colab launcher for frozen SWEFil epoch100 validation ablations.

Downloads the reviewed weight through the read-only Drive API. It does not train,
create synthetic images, inspect test predictions, or submit to Kaggle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

import swefil_test_colab as shared

SCOPE = 'swefil_epoch100_inference_ablation_t4_v1'
STATE_NAME = 'ablation_bootstrap_state.json'
CONFIG_NAME = 'ablation_launcher_config.json'
DEFAULT_RECIPES = ('baseline', 'tta4', 'dense', 'raw')
SOURCE_FILES = (
    'tools/swefil_ablation_colab.py', 'tools/swefil_inference_ablation.py',
    'tools/swefil_test_colab.py', 'tools/swefil_test_inference.py',
    'tools/solar_a100_train_standalone.py', 'tools/chase_best84_training.py',
    'tools/chase_independent_decode.py', 'tools/verified_drive_checkpoints.py',
)
BASE = shared.BASE
CHECKPOINT_SHA = shared.CHECKPOINT_SHA


def normalize_recipes(recipes):
    values = recipes.split(',') if isinstance(recipes, str) else list(recipes)
    values = [value.strip() for value in values]
    if not values or any(value not in DEFAULT_RECIPES for value in values):
        raise ValueError('Recipes must be chosen from: '+','.join(DEFAULT_RECIPES))
    return ['baseline', *[name for name in DEFAULT_RECIPES[1:] if name in values]]


def settings(training_root, output, recipes=DEFAULT_RECIPES, *, images=None,
             annotations=None, val_split=None, checkpoint=None, checkpoint_file_id=None):
    training_root, output = Path(training_root).resolve(), Path(output).resolve()
    shared.separate(output, training_root)
    if images is not None:
        shared.separate(output, images)
    extra = dict(images=images, annotations=annotations, val_split=val_split)
    return dict(scope=SCOPE, training_root=str(training_root), output_root=str(output),
        recipes=normalize_recipes(recipes), expected_epoch=100, validation_images=91,
        annotation_groups=101, gt_instances=708, checkpoint_sha256=CHECKPOINT_SHA,
        training_identity=shared.TRAINING_IDENTITY, inference_only=True,
        checkpoint_override=str(Path(checkpoint).absolute()) if checkpoint else None,
        checkpoint_file_id=checkpoint_file_id,
        **{key:str(Path(value).resolve()) if value else None for key,value in extra.items()})


def restore(output, training_root=None, recipes=DEFAULT_RECIPES, **overrides):
    """Verify the complete Drive snapshot before executing its worker."""
    output = Path(output).resolve()
    pointer = output/STATE_NAME
    if pointer.is_symlink():
        raise ValueError('Unsafe ablation bootstrap pointer')
    state = json.loads(pointer.read_text())
    if state.get('scope') != SCOPE or state.get('output_root') != str(output):
        raise ValueError('Saved bootstrap belongs to another experiment')
    version = state.get('version', '')
    if not re.fullmatch('[0-9a-f]{16}', version):
        raise ValueError('Invalid source version')
    code = output/'code'/version
    manifest_path = code/'source_manifest.json'
    if code.is_symlink() or not code.resolve().is_relative_to(output):
        raise ValueError('Unsafe source snapshot directory')
    if manifest_path.is_symlink() or shared.sha(manifest_path) != state.get('manifest_sha256'):
        raise ValueError('Saved source manifest checksum differs')
    manifest = json.loads(manifest_path.read_text())
    if (set(manifest) != set(SOURCE_FILES)
            or hashlib.sha256(shared.canonical(manifest)).hexdigest()[:16] != version):
        raise ValueError('Saved source inventory/version differs')
    for name,expected in manifest.items():
        path = code/name
        if path.is_symlink() or not path.resolve().is_relative_to(code.resolve()):
            raise ValueError('Unsafe saved source path: '+name)
        if not path.is_file() or shared.sha(path) != expected:
            raise ValueError('Saved source checksum differs: '+name)
    config_path = output/CONFIG_NAME
    if config_path.is_symlink() or shared.sha(config_path) != state.get('config_sha256'):
        raise ValueError('Saved ablation settings checksum differs')
    config = json.loads(config_path.read_text())
    extra = {key:overrides.get(key) or config.get(key) for key in ('images','annotations','val_split')}
    wanted = settings(training_root or config['training_root'], output, recipes,
        checkpoint=config.get('checkpoint_override'), checkpoint_file_id=config.get('checkpoint_file_id'), **extra)
    if config != wanted:
        raise ValueError('Ablation inputs/recipes changed; retain saved settings or use a new output folder')
    return config, code, state


def install(repository, training_root, output, recipes=DEFAULT_RECIPES, **options):
    """Publish one immutable, complete, small source bundle to this run's folder."""
    repository, output = Path(repository).resolve(), Path(output).resolve()
    if (output/STATE_NAME).exists() or (output/STATE_NAME).is_symlink():
        return restore(output, training_root, recipes, **options)
    if any((output/name).exists() for name in ('identity.json','latest_result.json','cache','train')):
        raise ValueError('Existing experiment has no bootstrap identity; preserve it and use a new output')
    config = settings(training_root, output, recipes, **options)
    payloads = {}
    for name in SOURCE_FILES:
        path = repository/name
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(repository):
            raise FileNotFoundError('Missing ablation source: '+str(path))
        raw = path.read_bytes()
        compile(raw, name, 'exec')
        payloads[name] = raw
    manifest = {name:hashlib.sha256(raw).hexdigest() for name,raw in payloads.items()}
    version = hashlib.sha256(shared.canonical(manifest)).hexdigest()[:16]
    code = output/'code'/version
    manifest_raw, config_raw = shared.canonical(manifest)+b'\n', shared.canonical(config)+b'\n'
    try:
        commit = subprocess.check_output(['git','-C',str(repository),'rev-parse','HEAD'],
                                         text=True,stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(['git','-C',str(repository),'status','--porcelain'],text=True).strip())
    except subprocess.CalledProcessError:
        commit, dirty = None, None
    state = dict(scope=SCOPE, version=version, output_root=str(output), source_commit=commit,
                 source_worktree_dirty=dirty, manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
                 config_sha256=hashlib.sha256(config_raw).hexdigest())
    writes = {code/name:raw for name,raw in payloads.items()}
    writes.update({code/'source_manifest.json':manifest_raw, output/CONFIG_NAME:config_raw,
                   output/STATE_NAME:shared.canonical(state)+b'\n'})
    for path,raw in writes.items():
        if path.is_symlink() or not path.resolve().is_relative_to(output):
            raise ValueError('Unsafe snapshot destination: '+str(path))
        if path.exists() and (not path.is_file() or path.read_bytes()!=raw):
            raise ValueError('Preserving changed existing file: '+str(path))
    # The state pointer is written last: interrupted installation can be repeated.
    for path,raw in writes.items():
        if not path.exists():
            shared.atomic_write(path,raw)
    return restore(output,training_root,recipes,**options)


def checked_result(output, recipes=DEFAULT_RECIPES):
    output = Path(output).resolve()
    path = output/'latest_result.json'
    if path.is_symlink():
        raise ValueError('Unsafe ablation result pointer')
    result = json.loads(path.read_text())
    if Path(result.get('output_root','')).resolve() != output:
        raise ValueError('Result belongs to another output directory')
    if result.get('status') != 'complete':
        raise RuntimeError('Validation ablation did not complete: '+str(result.get('status'))+
                           '. Inspect inference.log and latest_result.json; no test CSV was produced.')
    expected = dict(validation_images=91,annotation_groups=101,gt_instances=708,
                    checkpoint_epoch=100,checkpoint_sha256=CHECKPOINT_SHA,
                    recipes=normalize_recipes(recipes))
    if any(result.get(key)!=value for key,value in expected.items()):
        raise ValueError('Result checkpoint, recipes or validation coverage differs')
    archive = Path(result['archive'])
    if archive.is_symlink() or not archive.resolve().is_relative_to(output) or not archive.is_file():
        raise ValueError('Missing or unsafe review archive')
    if shared.sha(archive) != result.get('archive_sha256'):
        raise ValueError('Review archive checksum differs')
    return result


def run(training_root, output, recipes=DEFAULT_RECIPES, *, repository=None,
        images=None, annotations=None, val_split=None, checkpoint=None, checkpoint_file_id=None):
    output = Path(output).resolve()
    options = dict(images=images,annotations=annotations,val_split=val_split,
                   checkpoint=checkpoint,checkpoint_file_id=checkpoint_file_id)
    repository = Path(repository).resolve() if repository else Path(__file__).resolve().parents[1]
    config,code,state = install(repository,training_root,output,recipes,**options)
    training = Path(config['training_root'])
    for name in ('train/config.json','train/identity.json'):
        if not (training/name).is_file():
            raise FileNotFoundError('Training metadata is missing: '+str(training/name))
    print('Validation inference only; weights fixed at epoch100. Recipes:',','.join(config['recipes']),flush=True)
    print('Saved source version:',state['version'],'; completed image caches are reused.',flush=True)
    shared.ensure_dependencies(output)
    active_checkpoint, active_file_id = checkpoint, checkpoint_file_id
    if checkpoint is None and checkpoint_file_id is None:
        active_checkpoint, active_file_id = config['checkpoint_override'], config['checkpoint_file_id']
    selected = shared.download_checkpoint(training,active_checkpoint,active_file_id)
    command = [sys.executable,'-u',str(code/'tools/swefil_inference_ablation.py'),
               '--training-root',config['training_root'],'--checkpoint',str(selected),
               '--output',str(output),'--recipes',','.join(config['recipes'])]
    for key in ('images','annotations','val_split'):
        if config[key]:
            command.extend(['--'+key.replace('_','-'),config[key]])
    env = dict(os.environ,PYTHONUNBUFFERED='1',TORCH_CUDNN_V8_API_DISABLED='1',
               CUBLAS_WORKSPACE_CONFIG=':4096:8')
    shared.monitored(command,output/'inference.log',code/'tools',env)
    result = checked_result(output,config['recipes'])
    print('Completed validation review ZIP:',result['archive'],flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-root',type=Path,default=BASE/'fixed_split_experiments/swefil_all_mixed_native_t4_v1')
    parser.add_argument('--output',type=Path,default=BASE/'fixed_split_experiments'/SCOPE)
    parser.add_argument('--recipes',default=','.join(DEFAULT_RECIPES))
    parser.add_argument('--images',type=Path)
    parser.add_argument('--annotations',type=Path)
    parser.add_argument('--val-split',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--checkpoint-file-id')
    args = parser.parse_args()
    def interrupt(number,frame):
        raise KeyboardInterrupt('Interrupted by signal '+str(number))
    for number in (signal.SIGINT,signal.SIGTERM):
        signal.signal(number,interrupt)
    run(args.training_root,args.output,args.recipes,images=args.images,annotations=args.annotations,
        val_split=args.val_split,checkpoint=args.checkpoint,checkpoint_file_id=args.checkpoint_file_id)


if __name__ == '__main__':
    main()
