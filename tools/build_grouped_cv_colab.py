"""Build a restorable Colab cell from reviewed local Python sources."""
import base64
import hashlib
import json
from pathlib import Path
import zlib

ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_SHA = '5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1'
RUNTIME_SHA = '6d60b5705fb0fbde03a6fd2922c8b83839fa49b5a78cc3434e7dcf4a1b4a08b4'

LAUNCHER = r'''"""Persistent grouped-CV launcher; restore local code before every run."""
import argparse, hashlib, importlib.metadata, json, os, subprocess, sys
from pathlib import Path

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def save(path, value):
    pending = path.with_name(path.name+'.pending')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    pending.replace(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--folds', default=None)
    parser.add_argument('--run-until', type=int, default=None)
    args = parser.parse_args()
    root = Path(args.root)
    saved = json.loads((root/'launch_config.json').read_text())
    folds = saved['folds'] if args.folds is None else [int(x) for x in args.folds.split(',')]
    cap = saved['run_until_epoch'] if args.run_until is None else args.run_until
    if not folds or len(set(folds)) != len(folds) or any(f not in range(5) for f in folds) or cap < 1:
        raise ValueError('Use unique folds0..4 and a positive absolute epoch cap')
    work = Path('/content/solar_grouped_cv')
    work.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root/'code/source_manifest.json').read_text())
    for name, expected in manifest.items():
        source = root/'code'/name
        if digest(source) != expected:
            raise ValueError('Stored CV source changed: '+name)
        (work/name).write_bytes(source.read_bytes())
    # Preserve Colab's CUDA-compatible torch/torchvision and NumPy installation.
    pins = {'segmentation-models-pytorch':'0.5.0', 'timm':'1.0.20', 'pycocotools':'2.0.11'}
    needed = []
    for name, version in pins.items():
        try: installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: installed = None
        if installed != version: needed.append(name+'=='+version)
    for name in ('scipy', 'opencv-python-headless'):
        try:
            if name == 'opencv-python-headless':
                import cv2
            else: import scipy
        except ImportError: needed.append(name)
    if needed:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *needed], check=True)
    subprocess.run([sys.executable, '-c', 'import torch; assert torch.cuda.is_available(), "Select T4 GPU runtime"; print("GPU:",torch.cuda.get_device_name(0),"PyTorch:",torch.__version__)'], check=True)
    sys.path.insert(0, str(work))
    from prepare_grouped_fivefold import build_plan, write_outputs
    base = Path(saved['base'])
    annotation = base/'MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'
    if digest(annotation) != saved['annotations_sha256']:
        raise ValueError('Official training annotations differ from the reviewed source')
    plan = build_plan(json.loads(annotation.read_text()))
    plan['sources_sha256'] = {'annotations': digest(annotation)}
    splits = root/'splits'
    expected_plan = json.dumps(plan, ensure_ascii=False, indent=2)+'\n'
    if splits.exists():
        if (splits/'plan.json').read_text() != expected_plan:
            raise ValueError('Existing fold plan differs; preserved')
    else:
        write_outputs(plan, splits)
    logs = root/'logs'; logs.mkdir(exist_ok=True)
    results = []
    for fold in folds:
        cfg = dict(base_runtime=str(work/'base_runtime.py'), fold_plan=str(splits/'plan.json'),
            fold=fold, images=str(base/'MAGFiLO_1.0_Kaggle_2026/train/train_images'),
            annotations=str(annotation), train_split=str(splits/f'fold{fold}/train_split.csv'),
            val_split=str(splits/f'fold{fold}/val_split.csv'), output_root=str(root/'runs'),
            cache_root='/content/solar_grouped_cv_cache', epochs=saved['epochs'], run_until_epoch=cap,
            encoder='tu-convnext_tiny', pretrained=True, crop=1024, batch=1, accumulate=4,
            steps_per_epoch=400, lr=1e-4, spine_weight=.1, workers=0, infer_batch=1,
            overlap=.25, validate_every=20, seed=42+fold, device='cuda',
            synthetic_ratio=0., warmup_epochs=0, real_finish_epochs=0, threshold=.5, min_area=300)
        config_path = work/f'config_fold{fold}.json'; save(config_path, cfg)
        result_path = work/f'result_fold{fold}.json'
        environment = dict(os.environ)
        environment.pop('TORCH_CUDNN_V8_API_DISABLED', None)
        command = [sys.executable, '-u', str(work/'grouped_cv_training.py'),
                   '--config', str(config_path), '--result', str(result_path)]
        print(f'Fold{fold} only: fixed{saved["epochs"]}-epoch schedule, run through epoch{cap}; automatic resume', flush=True)
        with (logs/f'fold{fold}.log').open('a', buffering=1) as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1, env=environment)
            try:
                for line in process.stdout:
                    print(line, end='', flush=True); log.write(line)
                code = process.wait()
            except BaseException:
                process.terminate()
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
                raise
        if code:
            raise RuntimeError(f'Fold{fold} failed(exit{code}); last complete checkpoint preserved. Log: {logs/f"fold{fold}.log"}')
        results.append(json.loads(result_path.read_text()))
        save(root/'latest_results.json', results)
    print('Finished this run. Results:', root/'latest_results.json', flush=True)

if __name__ == '__main__':
    main()
'''

CELL = r'''# Colab T4: 전체를 한 셀에 복사하세요. 기본값은 fold0 하나만 학습합니다.
FOLDS = [0]
EPOCHS = 100
RUN_UNTIL_EPOCH = 100  # 20으로 바꾸면20epoch완료후일시정지; 나중에100으로재개

from pathlib import Path
from google.colab import drive, files
import base64, hashlib, json, os, signal, subprocess, sys, zlib

drive.mount('/content/drive')
BASE = Path('/content/drive/MyDrive/filament-segmentation-2026')
ROOT = BASE/'grouped_cv_t4_v1'
CODE = ROOT/'code'; CODE.mkdir(parents=True, exist_ok=True)
PAYLOAD = __PAYLOAD__
raw = zlib.decompress(base64.b64decode(PAYLOAD))
if hashlib.sha256(raw).hexdigest() != __BUNDLE_SHA__:
    raise ValueError('Copy-cell source bundle checksum mismatch')
sources = json.loads(raw)

def preserve(path, content):
    if path.exists() and path.read_bytes() != content:
        raise ValueError('기존 다른 파일은 보존했습니다. 저장된 재개 셀을 사용하세요: '+str(path))
    if not path.exists():
        temporary = path.with_name(path.name+'.pending')
        temporary.write_bytes(content); temporary.replace(path)

for name, source in sources.items():
    preserve(CODE/name, source.encode())
runtime = CODE/'base_runtime.py'
if not runtime.exists():
    source_root = Path('/content/drive/MyDrive/SolarFilament_NativeUNet_A100')
    preferred = source_root/'20261028T030808Z_bedf3e21/run_source.py'
    candidates = [preferred]+sorted(source_root.glob('*/run_source.py'))
    selected = next((p for p in candidates if p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==__RUNTIME_SHA__), None)
    if selected is None:
        raise FileNotFoundError('검증된 원래 run_source.py를 찾지 못했습니다: '+str(preferred))
    preserve(runtime, selected.read_bytes())
if hashlib.sha256(runtime.read_bytes()).hexdigest() != __RUNTIME_SHA__:
    raise ValueError('Stored original runtime checksum differs')
manifest = {name:hashlib.sha256((CODE/name).read_bytes()).hexdigest()
            for name in [*sources, 'base_runtime.py']}
preserve(CODE/'source_manifest.json', (json.dumps(manifest, sort_keys=True, indent=2)+'\n').encode())
launch = dict(base=str(BASE), folds=FOLDS, epochs=EPOCHS,
              run_until_epoch=RUN_UNTIL_EPOCH, annotations_sha256=__ANNOTATION_SHA__)
launch_path = ROOT/'launch_config.json'
if launch_path.exists():
    old = json.loads(launch_path.read_text())
    if any(old[k] != launch[k] for k in ('base','epochs','annotations_sha256')):
        raise ValueError('기존 실험의 전체epoch/데이터설정을 변경할 수 없습니다.')
else:
    preserve(launch_path, (json.dumps(launch, indent=2)+'\n').encode())
print('외부데이터/기존solar가중치 없이 새fold학습. 합성데이터사용0. 마지막완료epoch자동재개.', flush=True)
command = [sys.executable, '-u', str(CODE/'cv_launcher.py'), '--root', str(ROOT),
           '--folds', ','.join(map(str,FOLDS)), '--run-until', str(RUN_UNTIL_EPOCH)]
process = subprocess.Popen(command, start_new_session=True)
try:
    code = process.wait()
    if code: raise subprocess.CalledProcessError(code, command)
except BaseException:
    try: os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError: pass
    try: process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL); process.wait()
    raise
for result in json.loads((ROOT/'latest_results.json').read_text()):
    print('결과 ZIP:', result['archive'])
    files.download(result['archive'])
'''

RESUME = r'''# 세션이 종료된 뒤에도 실행할 수 있는 짧은 재개 셀입니다.
FOLDS = [0]                 # fold0 완료후 [1], [2], [3], [4] 순서로 실행
RUN_UNTIL_EPOCH = 100        # 전체스케줄100은 유지; 완료epoch부터 이어갑니다.
from pathlib import Path
from google.colab import drive, files
import hashlib, json, os, signal, subprocess, sys
drive.mount('/content/drive')
ROOT = Path('/content/drive/MyDrive/filament-segmentation-2026/grouped_cv_t4_v1')
manifest_path = ROOT/'code/source_manifest.json'
if not manifest_path.is_file():
    raise FileNotFoundError('최초실행용GroupedCV복사셀을 먼저 실행해야 합니다.')
manifest = json.loads(manifest_path.read_text())
for name, expected in manifest.items():
    path = ROOT/'code'/name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('저장된 학습 코드가 달라졌습니다: '+name)
command = [sys.executable,'-u',str(ROOT/'code/cv_launcher.py'),'--root',str(ROOT),
           '--folds',','.join(map(str,FOLDS)),'--run-until',str(RUN_UNTIL_EPOCH)]
process = subprocess.Popen(command, start_new_session=True)
try:
    code = process.wait()
    if code: raise subprocess.CalledProcessError(code, command)
except BaseException:
    try: os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError: pass
    try: process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL); process.wait()
    raise
for result in json.loads((ROOT/'latest_results.json').read_text()):
    print('결과 ZIP:',result['archive']); files.download(result['archive'])
'''


def build():
    sources = {name: (ROOT/'tools'/name).read_text()
               for name in ('grouped_cv_training.py', 'prepare_grouped_fivefold.py')}
    sources['cv_launcher.py'] = LAUNCHER
    raw = json.dumps(sources, ensure_ascii=False, sort_keys=True).encode()
    cell = CELL.replace('__PAYLOAD__', repr(base64.b64encode(zlib.compress(raw, 9)).decode()))
    for token, value in [('__BUNDLE_SHA__', hashlib.sha256(raw).hexdigest()),
                         ('__RUNTIME_SHA__', RUNTIME_SHA), ('__ANNOTATION_SHA__', ANNOTATION_SHA)]:
        cell = cell.replace(token, repr(value))
    compile(cell, 'grouped-cv-copy-cell', 'exec')
    compile(RESUME, 'grouped-cv-resume-cell', 'exec')
    out = ROOT/'notebooks'
    (out/'SolarFilament_GroupedCV_T4_CopyCell.py').write_text(cell)
    (out/'GroupedCV_T4_Resume_CopyCell.py').write_text(RESUME)
    notebook = dict(nbformat=4, nbformat_minor=5,
        metadata=dict(kernelspec=dict(display_name='Python3', language='python', name='python3'), accelerator='GPU'),
        cells=[dict(id='intro', cell_type='markdown', metadata={}, source=['# Solar filament grouped5-fold CV\n',
            '기본fold0만 실행합니다. 단일native모델의 새기준선이며, 기존앙상블의PQ와 직접 비교하지 않습니다.\n',
            '100epoch고정최종모델을 평가합니다.20epoch마다의진단PQ로 best를선택하지 않습니다.\n']),
            dict(id='training', cell_type='code', execution_count=None, metadata={}, outputs=[], source=cell.splitlines(True))])
    (out/'SolarFilament_GroupedCV_T4.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=2)+'\n')
    return dict(source_sha256={name: hashlib.sha256(value.encode()).hexdigest() for name,value in sources.items()},
                bundle_sha256=hashlib.sha256(raw).hexdigest(), copy_cell_bytes=len(cell.encode()))


if __name__ == '__main__':
    print(json.dumps(build(), indent=2))
