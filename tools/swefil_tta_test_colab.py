"""Resumable Colab launcher for reviewed SWEFil epoch100 four-view test inference.

Downloads the reviewed weight through the read-only Drive API. It does not train,
create synthetic images, or submit to Kaggle.
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

SCOPE = 'swefil_epoch100_tta4_t4_v1'
STATE_NAME = 'tta_test_bootstrap_state.json'
CONFIG_NAME = 'tta_test_launcher_config.json'
CSV_NAME = 'submission_swefil_epoch100_tta4.csv'
ARCHIVE_NAME = 'swefil_epoch100_tta4_test_results.zip'
VALIDATION_PQ = 0.4207475911810596
RECIPE = dict(encoder='tu-convnext_tiny', preprocessing='limb_only', tile=1024,
              overlap=.25, infer_batch=1, precision='float16', threshold=.65,
              min_area=300, connectivity=8, tta=True, yolo=False,
              public_predictions=False, checkpoint_epoch=100,
              views=['base','h','v','hv'], channel=0,
              averaging='probability_mean_after_inverse_flip')
SOURCE_FILES = (
    'tools/swefil_tta_test_colab.py', 'tools/swefil_tta_test_inference.py',
    'tools/swefil_inference_ablation.py',
    'tools/swefil_test_colab.py', 'tools/swefil_test_inference.py',
    'tools/solar_a100_train_standalone.py', 'tools/chase_best84_training.py',
    'tools/chase_independent_decode.py', 'tools/verified_drive_checkpoints.py',
)
BASE = shared.BASE
CHECKPOINT_SHA = shared.CHECKPOINT_SHA


def settings(training_root, test_images, output, *, checkpoint=None, checkpoint_file_id=None):
    training_root, test_images, output = map(lambda p: Path(p).resolve(),
                                             (training_root, test_images, output))
    shared.separate(output, training_root)
    shared.separate(output, test_images)
    return dict(scope=SCOPE, training_root=str(training_root), test_images=str(test_images),
        output_root=str(output), recipe=RECIPE, expected_epoch=100, expected_images=180,
        checkpoint_sha256=CHECKPOINT_SHA, training_identity=shared.TRAINING_IDENTITY,
        inference_only=True, validation_pq=VALIDATION_PQ,
        checkpoint_override=str(Path(checkpoint).absolute()) if checkpoint else None,
        checkpoint_file_id=checkpoint_file_id)


def restore(output, training_root=None, test_images=None, **overrides):
    """Verify the complete Drive snapshot before executing its worker."""
    output = Path(output).resolve()
    pointer = output/STATE_NAME
    if pointer.is_symlink():
        raise ValueError('Unsafe TTA bootstrap pointer')
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
        raise ValueError('Saved TTA settings checksum differs')
    config = json.loads(config_path.read_text())
    # A checkpoint locator may change, while its pinned weight hash stays fixed.
    wanted = settings(training_root or config['training_root'],
        test_images or config['test_images'], output,
        checkpoint=config.get('checkpoint_override'), checkpoint_file_id=config.get('checkpoint_file_id'))
    if config != wanted:
        raise ValueError('TTA inputs/settings changed; retain saved paths or use a new output folder')
    return config, code, state


def install(repository, training_root, test_images, output, **options):
    """Publish one immutable, complete, small source bundle to this run's folder."""
    repository, output = Path(repository).resolve(), Path(output).resolve()
    if (output/STATE_NAME).exists() or (output/STATE_NAME).is_symlink():
        return restore(output, training_root, test_images, **options)
    if any((output/name).exists() for name in ('identity.json','latest_result.json','predictions','train',
            'test_bootstrap_state.json','ablation_bootstrap_state.json',
            'inference_launcher_config.json','ablation_launcher_config.json',
            'submission_swefil_epoch100_native.csv')):
        raise ValueError('Existing experiment has no bootstrap identity; preserve it and use a new output')
    config = settings(training_root, test_images, output, **options)
    payloads = {}
    for name in SOURCE_FILES:
        path = repository/name
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(repository):
            raise FileNotFoundError('Missing TTA inference source: '+str(path))
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
    return restore(output,training_root,test_images,**options)


def checked_result(output):
    output = Path(output).resolve()
    path = output/'latest_result.json'
    if path.is_symlink():
        raise ValueError('Unsafe TTA result pointer')
    result = json.loads(path.read_text())
    if Path(result.get('output_root','')).resolve() != output:
        raise ValueError('Result belongs to another output directory')
    if result.get('status') != 'complete' or result.get('images_completed') != 180:
        raise RuntimeError('All 180 test images must complete before a TTA submission is downloaded')
    expected = dict(images_total=180, checkpoint_epoch=100,
                    checkpoint_sha256=CHECKPOINT_SHA, recipe=RECIPE, overlap_pixels=0)
    if any(result.get(key)!=value for key,value in expected.items()):
        raise ValueError('Result checkpoint, TTA recipe, coverage or overlap audit differs')
    reference = result.get('validation_reference')
    if not isinstance(reference,dict) or reference.get('pq') != VALIDATION_PQ:
        raise ValueError('Result lacks the reviewed TTA validation reference')
    for key,name in (('csv',CSV_NAME),('archive',ARCHIVE_NAME)):
        path = Path(result[key])
        if (path.name != name or path.is_symlink() or not path.resolve().is_relative_to(output)
                or not path.is_file()):
            raise ValueError('Missing or unsafe TTA result: '+key)
        if shared.sha(path) != result.get(key+'_sha256'):
            raise ValueError('TTA result checksum differs: '+key)
    return result


def run(training_root, test_images, output, *, repository=None,
        checkpoint=None, checkpoint_file_id=None):
    output = Path(output).resolve()
    options = dict(checkpoint=checkpoint,checkpoint_file_id=checkpoint_file_id)
    repository = Path(repository).resolve() if repository else Path(__file__).resolve().parents[1]
    config,code,state = install(repository,training_root,test_images,output,**options)
    training = Path(config['training_root'])
    for name in ('train/config.json','train/identity.json'):
        if not (training/name).is_file():
            raise FileNotFoundError('Training metadata is missing: '+str(training/name))
    if not Path(config['test_images']).is_dir():
        raise FileNotFoundError('Original test image directory: '+config['test_images'])
    print('Inference only: frozen epoch100, four flip views, threshold .65, area 300.',flush=True)
    print('Saved source version:',state['version'],'; completed image predictions are reused.',flush=True)
    shared.ensure_dependencies(output)
    active_checkpoint, active_file_id = checkpoint, checkpoint_file_id
    if checkpoint is None and checkpoint_file_id is None:
        active_checkpoint, active_file_id = config['checkpoint_override'], config['checkpoint_file_id']
    selected = shared.download_checkpoint(training,active_checkpoint,active_file_id)
    command = [sys.executable,'-u',str(code/'tools/swefil_tta_test_inference.py'),
               '--training-root',config['training_root'],'--test-images',config['test_images'],
               '--checkpoint',str(selected),'--output',str(output),
               '--result',str(output/'latest_result.json')]
    env = dict(os.environ,PYTHONUNBUFFERED='1',TORCH_CUDNN_V8_API_DISABLED='1',
               CUBLAS_WORKSPACE_CONFIG=':4096:8')
    shared.monitored(command,output/'inference.log',code/'tools',env)
    result = checked_result(output)
    print('Submission CSV:',result['csv'],flush=True)
    print('Review ZIP:',result['archive'],flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-root',type=Path,default=BASE/'fixed_split_experiments/swefil_all_mixed_native_t4_v1')
    parser.add_argument('--output',type=Path,default=BASE/'submissions'/SCOPE)
    parser.add_argument('--test-images',type=Path,default=BASE/'MAGFiLO_1.0_Kaggle_2026/test/test_images')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--checkpoint-file-id')
    args = parser.parse_args()
    def interrupt(number,frame):
        raise KeyboardInterrupt('Interrupted by signal '+str(number))
    for number in (signal.SIGINT,signal.SIGTERM):
        signal.signal(number,interrupt)
    run(args.training_root,args.test_images,args.output,checkpoint=args.checkpoint,
        checkpoint_file_id=args.checkpoint_file_id)


if __name__ == '__main__':
    main()
