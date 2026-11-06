# Colab T4의 새 셀 하나에 전체 복사하세요. 추가 학습 없이 last.pt만 추론합니다.
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

BASE = Path('/content/drive/MyDrive/filament-segmentation-2026')
TRAINING_ROOT = BASE/'paper_method_experiments/context_only_t4_v1_fp16_v2'
WORK = Path('/content/context_last_export_sources')
SOURCE_SHA = {
    'paper_method_training.py':'70a2c5a4293bcd2770f7e657f16d71a7eaaa08216862357cfa22734ab4715cf1',
    'paper_multiscale.py':'15cf2002b3dabcc53736c46815d0b8a868630b14c78cf85442f2f96d6041ee5c',
    'base_runtime.py':'6d60b5705fb0fbde03a6fd2922c8b83839fa49b5a78cc3434e7dcf4a1b4a08b4',
}


def snapshot_checkpoint(source, destination):
    """One open source handle remains stable if atomic training last.pt advances."""
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name+'.pending')
    digest = hashlib.sha256()
    try:
        with source.open('rb') as incoming, temporary.open('wb') as outgoing:
            while chunk := incoming.read(8*1024**2):
                outgoing.write(chunk)
                digest.update(chunk)
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return digest.hexdigest()


def restore_learning_sources(run, work):
    run, work = Path(run), Path(work)
    work.mkdir(parents=True, exist_ok=True)
    for name, expected in SOURCE_SHA.items():
        raw = (run/name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('저장된 학습 소스 SHA가 다릅니다: '+name)
        (work/name).write_bytes(raw)


WORKER = r'''
import argparse, csv, hashlib, json, shutil, sys, zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from torch import nn
import paper_method_training as legacy

torch.cuda.is_bf16_supported = lambda *args, **kwargs: False
torch.backends.cudnn.benchmark = False
original_conv_forward = nn.Conv2d._conv_forward
def safe_depthwise(self, inputs, weight, bias):
    if inputs.is_cuda and self.groups > 1 and self.groups == self.in_channels:
        with torch.backends.cudnn.flags(enabled=False):
            return original_conv_forward(self, inputs, weight, bias)
    return original_conv_forward(self, inputs, weight, bias)
nn.Conv2d._conv_forward = safe_depthwise

def atomic_copy(source, target):
    temporary = target.with_name(target.name+'.pending')
    with Path(source).open('rb') as incoming, temporary.open('wb') as outgoing:
        shutil.copyfileobj(incoming, outgoing, 1024**2)
    temporary.replace(target)

parser = argparse.ArgumentParser()
parser.add_argument('--local-run', required=True)
parser.add_argument('--checkpoint-sha', required=True)
parser.add_argument('--base', required=True)
parser.add_argument('--training-run', required=True)
parser.add_argument('--result', required=True)
args = parser.parse_args()
local_run = Path(args.local_run)
snapshot = local_run/'best.pt'  # local byte-identical copy of last.pt; training best.pt is untouched
if legacy.digest(snapshot) != args.checkpoint_sha:
    raise ValueError('Local last.pt snapshot SHA differs')
checkpoint = torch.load(snapshot, map_location='cpu', weights_only=True)
epoch = int(checkpoint['epoch'])
cfg = dict(checkpoint['config'])
if cfg.get('paper_method') != 'multiscale_context' or cfg.get('amp_dtype') != 'torch.float16':
    raise ValueError('Expected the trained context model with the actual FP16 configuration')
if cfg.get('initial_checkpoint_sha256') != legacy.INITIAL_CHECKPOINT_SHA:
    raise ValueError('Original best84 initialization SHA differs')
if legacy.digest(cfg['initial_checkpoint']) != legacy.INITIAL_CHECKPOINT_SHA:
    raise ValueError('Original best84 checkpoint bytes differ')
for key, expected in cfg['expected_source_sha256'].items():
    if cfg['source_sha256'].get(key) != expected or legacy.digest(cfg[key]) != expected:
        raise ValueError('Input source differs: '+key)
work = Path(__file__).resolve().parent
actual_sources = {name:legacy.digest(work/name) for name in cfg['research_source_sha256']}
if actual_sources != cfg['research_source_sha256']:
    raise ValueError('Restored learning source differs from checkpoint provenance')
if int(cfg.get('infer_batch', 1)) != 1:
    raise ValueError('Keep T4 inference batch1 for comparison to the exported best45')
if not torch.cuda.is_available():
    raise ValueError('Colab T4 GPU runtime is required')
history = checkpoint['history']
if not history or int(history[-1]['epoch']) != epoch:
    raise ValueError('Checkpoint epoch/history disagree')
last_validation = history[-1].get('validation')
latest_validation = next((dict(epoch=row['epoch'], validation=row['validation'])
                          for row in reversed(history) if 'validation' in row), None)
selected_best_epoch = checkpoint.get('best_epoch')
del checkpoint

output = Path(args.base)/'paper_method_experiments/context_last_exports'/f'epoch{epoch}_{args.checkpoint_sha[:12]}'
output.mkdir(parents=True, exist_ok=True)
export_identity = dict(checkpoint_sha256=args.checkpoint_sha, precision='torch.float16',
    gpu=torch.cuda.get_device_name(0), torch_version=str(torch.__version__),
    cuda_version=torch.version.cuda, cudnn_version=torch.backends.cudnn.version(),
    worker_sha256=legacy.digest(__file__), crop=cfg['crop'], overlap=cfg['overlap'], infer_batch=1)
identity_path = output/'inference_identity.json'
if identity_path.is_file() and json.loads(identity_path.read_text()) != export_identity:
    raise ValueError('Existing export hardware/software/settings differ; refusing mixed cached predictions')
legacy.atomic_json(identity_path, export_identity)
records = output/'records'
records.mkdir(exist_ok=True)
link = local_run/'rich_validation'
if link.is_symlink():
    link.unlink()  # remove only our local link; all Drive records remain
elif link.exists():
    raise ValueError('Local rich_validation must be absent or a symlink')
link.symlink_to(records, target_is_directory=True)

cfg.update(device='cuda', base_runtime=str(work/'base_runtime.py'),
           cache_root='/content/filament_a100_cache', infer_batch=1)
runtime = legacy.load_runtime(Path(cfg['base_runtime']))
runtime.build_original_model = runtime.build_model
print('추가 학습 없이 last.pt 추론:', epoch, 'epoch', flush=True)
print('GPU:', torch.cuda.get_device_name(0), 'precision: FP16', flush=True)
raw = legacy.export_rich(runtime, cfg, local_run)
exported = [json.loads(line) for line in raw.read_text().splitlines()]
with Path(cfg['val_split']).open() as split_file:
    expected_names = {row['file_name'] for row in csv.DictReader(split_file)}
if len(exported) != len(expected_names) or {row['file_name'] for row in exported} != expected_names:
    raise ValueError('Export does not cover the fixed validation filenames exactly once')
if any(row['checkpoint_sha256'] != args.checkpoint_sha for row in exported):
    raise ValueError('Export record checkpoint differs')
combined = output/'raw_paper_views.jsonl'
atomic_copy(raw, combined)
summary = dict(scope='Prediction-only last-checkpoint four-flip export; no training and no new GT scoring',
    training_performed=False, gt_scoring_performed=False, epoch=epoch,
    original_training_run=args.training_run, source_checkpoint='last.pt',
    checkpoint_sha256=args.checkpoint_sha, checkpoint_bytes=snapshot.stat().st_size,
    previously_selected_best_epoch=selected_best_epoch,
    stored_validation_at_this_epoch=last_validation, latest_stored_validation=latest_validation,
    physical_images=len(exported), views=['base','hflip','vflip','hvflip'],
    precision='torch.float16', gpu=torch.cuda.get_device_name(0),
    inference_identity=export_identity,
    depthwise_backend='native CUDA, other convolutions use cuDNN',
    learning_source_sha256=actual_sources, worker_sha256=legacy.digest(__file__),
    source_sha256=cfg['source_sha256'], raw_predictions_sha256=legacy.digest(combined),
    note='Stored validation was measured during training. This export does not compute a new PQ. A last checkpoint after epoch60 is exported at its actual saved epoch.')
legacy.atomic_json(output/'summary.json', summary)
legacy.atomic_json(output/'config.json', cfg)
archive = output/f'context_last_epoch{epoch}_{args.checkpoint_sha[:12]}_rich_predictions.zip'
temporary = archive.with_name(archive.name+'.pending')
with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as z:
    for path in (combined, output/'summary.json', output/'config.json'):
        z.write(path, path.name)
temporary.replace(archive)
legacy.atomic_json(args.result, dict(archive=str(archive), output=str(output), summary=summary))
print('업로드할 결과 ZIP:', archive, flush=True)
'''


def main():
    from google.colab import drive, files
    drive.mount('/content/drive')
    journal = json.loads((TRAINING_ROOT/'context_only_progress.json').read_text())
    run = Path(journal['runs']['multiscale_context']['path'])
    restore_learning_sources(run, WORK)
    local_run = WORK/'local_export_run'
    checkpoint_sha = snapshot_checkpoint(run/'last.pt', local_run/'best.pt')
    (WORK/'export_last_worker.py').write_text(WORKER, encoding='utf-8')
    compile(WORKER, 'export_last_worker.py', 'exec')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
        'numpy==2.1.3', 'segmentation-models-pytorch==0.5.0', 'timm==1.0.20',
        'huggingface-hub==0.34.4', 'scipy==1.17.0', 'pycocotools==2.0.11', 'opencv-python==5.0.0.93'])
    environment = os.environ.copy()
    environment.pop('TORCH_CUDNN_V8_API_DISABLED', None)
    result_path = WORK/'last_export_result.json'
    command = [sys.executable, '-u', str(WORK/'export_last_worker.py'),
        '--local-run', str(local_run), '--checkpoint-sha', checkpoint_sha,
        '--base', str(BASE), '--training-run', str(run), '--result', str(result_path)]
    log_path = BASE/'paper_method_experiments/context_last_exports/export_last.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=str(WORK), env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line); log.flush()
        code = process.wait()
    if code:
        raise RuntimeError(f'Last-checkpoint inference failed (exit{code}); log: {log_path}')
    result = json.loads(result_path.read_text())
    print(json.dumps(result['summary'], ensure_ascii=False, indent=2))
    files.download(result['archive'])


if __name__ == '__main__':
    main()
