"""Build a self-contained, resumable Colab experiment for mask-quality ranking."""
import base64
import hashlib
import json
from pathlib import Path
import zlib

ROOT = Path(__file__).resolve().parents[1]
MODULES = (
    'fixed_split_mask_quality.py', 'submission_inference.py',
    'evaluate_grouped_oof.py', 'grouped_cv_training.py',
)

CELL = r'''# T4: 고정된 616/91 분할에서만 마스크 품질 모델을 학습합니다.
# 세션이 종료되면 이 셀 전체를 다시 실행하세요. 완료 epoch부터 자동 재개합니다.
BASE = '/content/drive/MyDrive/filament-segmentation-2026'
YOLO_RUN = '/content/drive/MyDrive/SolarFilament_LimbOnly_YOLO26_mixed_FullRes'
TOTAL_EPOCHS = 40  # 전체 학습 일정. 학습을 시작한 뒤에는 유지하세요.
RUN_UNTIL = 40     # 20으로 실행해 중간 점검한 뒤 40으로 바꾸어 재개할 수 있습니다.

from google.colab import drive, files
from pathlib import Path
import base64, hashlib, importlib.metadata, importlib.util, json, os
import signal, subprocess, sys, zlib

drive.mount('/content/drive')
raw = zlib.decompress(base64.b64decode(__PAYLOAD__))
if hashlib.sha256(raw).hexdigest() != __SHA__:
    raise ValueError('코드가 온전히 복사되지 않았습니다. 셀 전체를 다시 복사하세요.')
sources = json.loads(raw)
if set(sources) != set(__MODULES__):
    raise ValueError('학습 코드 묶음이 잘못되었습니다.')
BASE = Path(BASE); YOLO_RUN = Path(YOLO_RUN)
OUTPUT = BASE/'fixed_split_experiments/mask_quality_v1'
WORK = Path('/content/solar_fixed_mask_quality')/__VERSION__
CACHE = Path('/content/solar_fixed_mask_quality_cache')
DATA = BASE/'MAGFiLO_1.0_Kaggle_2026/train'
ANNOTATIONS = DATA/'MAGFiLO_1.0_Annotations_kaggle2026_train.json'
IMAGES = DATA/'train_images'
for path in (ANNOTATIONS, YOLO_RUN/'best.pt', YOLO_RUN/'train_split.csv', YOLO_RUN/'val_split.csv'):
    if not path.is_file(): raise FileNotFoundError(str(path))
if not IMAGES.is_dir(): raise FileNotFoundError(str(IMAGES))
if not isinstance(TOTAL_EPOCHS, int) or TOTAL_EPOCHS < 1:
    raise ValueError('TOTAL_EPOCHS는 양의 정수로 지정하세요.')
if not isinstance(RUN_UNTIL, int) or not 1 <= RUN_UNTIL <= TOTAL_EPOCHS:
    raise ValueError('RUN_UNTIL은 1~TOTAL_EPOCHS로 지정하세요. 재개 시 TOTAL_EPOCHS는 유지하세요.')
WORK.mkdir(parents=True, exist_ok=True)
saved_code = OUTPUT/'code'/__VERSION__
saved_code.mkdir(parents=True, exist_ok=True)
manifest = {}
for name, source in sources.items():
    compile(source, name, 'exec')
    content = source.encode('utf-8')
    manifest[name] = hashlib.sha256(content).hexdigest()
    durable = saved_code/name
    if durable.exists() and durable.read_bytes() != content:
        raise ValueError('Drive의 저장된 코드가 변경되었습니다: '+str(durable))
    if not durable.exists(): durable.write_bytes(content)
    (WORK/name).write_bytes(content)
(saved_code/'source_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')

# Colab에 제공된 torch/CUDA는 유지합니다. 추론 시 사용했던 YOLO 버전만 맞춥니다.
required = []
for name, version in [('ultralytics', '8.4.182'), ('timm', '1.0.20'), ('pycocotools', '2.0.11')]:
    try: installed = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: installed = None
    if installed != version: required.append(name+'=='+version)
for module, package in [('numpy', 'numpy'), ('cv2', 'opencv-python-headless'), ('scipy', 'scipy')]:
    if importlib.util.find_spec(module) is None: required.append(package)
if required:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *required], check=True)
environment = dict(os.environ)
environment.pop('TORCH_CUDNN_V8_API_DISABLED', None)
subprocess.run([sys.executable, '-c',
    'import torch, timm; assert torch.cuda.is_available(), "Colab에서 T4 GPU 런타임을 선택하세요."; '
    'print("PyTorch:", torch.__version__, "GPU:", torch.cuda.get_device_name(0))'],
    env=environment, check=True)
common = [
    '--output', str(OUTPUT), '--cache', str(CACHE), '--images', str(IMAGES),
    '--annotations', str(ANNOTATIONS), '--train-split', str(YOLO_RUN/'train_split.csv'),
    '--val-split', str(YOLO_RUN/'val_split.csv'), '--yolo', str(YOLO_RUN/'best.pt'),
    '--device', 'auto', '--epochs', str(TOTAL_EPOCHS), '--run-until', str(RUN_UNTIL),
    '--batch-size', '16', '--freeze-epochs', '2', '--lr', '0.0001',
    '--train-cap', '48', '--validate-every', '2', '--seed', '42',
]
print(f'고정 YOLO로 4개 뷰 후보를 준비한 뒤, 품질 모델 1개를 총 {TOTAL_EPOCHS} epoch 일정으로 학습합니다.', flush=True)
print(f'이번 실행: 완료한 epoch부터 {RUN_UNTIL} epoch까지. 검증 PQ 최고 체크포인트는 별도 보존합니다.', flush=True)
print('Train 616장 / validation 91장. 5-fold 모델과 기존 제출 가중치는 유지합니다.', flush=True)
print('Drive 저장/재개 폴더:', OUTPUT, flush=True)
print('사진과 ROI 임시 캐시:', CACHE, '(런타임 종료 시 재생성)', flush=True)

def run_phase(phase):
    command = [sys.executable, '-u', str(WORK/'fixed_split_mask_quality.py'), '--phase', phase, *common]
    with (OUTPUT/('colab_'+phase+'.log')).open('a', encoding='utf-8', buffering=1) as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=environment, start_new_session=True)
        try:
            for line in process.stdout:
                print(line, end='', flush=True); log.write(line)
            code = process.wait()
            if code: raise subprocess.CalledProcessError(code, command)
        except BaseException:
            try: os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
            raise

# YOLO側の初期化が品質モデルのCUDA設定に影響しないようプロセスを分けます。
run_phase('prepare')
run_phase('train')
print('完了／再開情報は次のフォルダに保存されています:', OUTPUT, flush=True)
for name in ('result.json', 'summary.json'):
    path = OUTPUT/name
    if path.is_file():
        print(path.read_text(encoding='utf-8'), flush=True)
archives = sorted(OUTPUT.glob('*.zip'), key=lambda path: path.stat().st_mtime)
if archives:
    archive = archives[-1]
    print('評価用に送っていただくZIP:', archive, flush=True)
    try: files.download(str(archive))
    except Exception as exc: print('Driveから直接ダウンロードできます:', str(exc), flush=True)
'''


def build():
    sources = {name: (ROOT/'tools'/name).read_text(encoding='utf-8') for name in MODULES}
    for name, source in sources.items():
        compile(source, name, 'exec')
    raw = json.dumps(sources, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    digest = hashlib.sha256(raw).hexdigest()
    cell = (CELL.replace('__PAYLOAD__', repr(base64.b64encode(zlib.compress(raw, 9)).decode('ascii')))
        .replace('__SHA__', repr(digest)).replace('__VERSION__', repr(digest[:12]))
        .replace('__MODULES__', repr(MODULES)))
    compile(cell, 'fixed-split-quality-colab', 'exec')
    output = ROOT/'notebooks'
    output.mkdir(exist_ok=True)
    cell_path = output/'SolarFilament_FixedSplit_MaskQuality_CopyCell.py'
    cell_path.write_text(cell, encoding='utf-8')
    notebook = dict(nbformat=4, nbformat_minor=5,
        metadata=dict(kernelspec=dict(name='python3', display_name='Python 3', language='python'),
            colab=dict(name='SolarFilament_FixedSplit_MaskQuality.ipynb')),
        cells=[dict(id='intro', cell_type='markdown', metadata={}, source=[
            '# 마스크 품질 모델: 고정 616/91 분할\n',
            'T4 GPU에서 실행하세요. 고정된 YOLO의 후보를 준비하고 ResNet18 품질 모델 1개를 총 40 epoch 학습합니다.\n',
            'Validation 91장은 학습 gradient에 사용하지 않습니다. 5-fold 모델은 수정하지 않습니다.\n',
            '중단 후 같은 셀을 다시 실행하면 마지막 저장 epoch부터 재개합니다. TOTAL_EPOCHS=40을 유지하고 RUN_UNTIL만 20→40으로 바꾸어 중간 점검할 수도 있습니다.\n',
            '검증 PQ 최고 모델을 보존합니다. 새 성능은 아직 측정하지 않았습니다.\n']),
            dict(id='run', cell_type='code', execution_count=None, metadata={}, outputs=[],
                source=cell.splitlines(keepends=True))])
    notebook_path = output/'SolarFilament_FixedSplit_MaskQuality.ipynb'
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return dict(copy_cell=str(cell_path), notebook=str(notebook_path), cell_bytes=len(cell.encode('utf-8')),
        source_bundle_sha256=digest, sources_sha256={name: hashlib.sha256(value.encode()).hexdigest()
            for name, value in sources.items()}, training_split='original616', validation_split='original91',
        maximum_quality_model_epochs=40, default_run_until_epoch=40,
        new_quality_model_training_executed_here=False)


if __name__ == '__main__':
    print(json.dumps(build(), ensure_ascii=False, indent=2))
