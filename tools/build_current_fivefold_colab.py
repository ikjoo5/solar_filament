"""Build a single-cell, Drive-persistent evaluation-only snapshot exporter."""
import base64
import hashlib
import json
from pathlib import Path
import zlib

ROOT = Path(__file__).resolve().parents[1]
REVIEWED_VERSION = '5c1fb752ac88'

CELL = r'''# 현재 저장된 fold 가중치 평가 전용: 추가 학습 없음
from google.colab import drive, files
from pathlib import Path
import base64, hashlib, importlib.metadata, json, os, signal, subprocess, sys, zlib

drive.mount('/content/drive')
ROOT = Path('/content/drive/MyDrive/filament-segmentation-2026/fivefold_study_v2')
OUTPUT = ROOT/'current_epoch_evaluation/native030_yolo_current_v1'
BASE = ROOT.parent
IMAGES = BASE/'MAGFiLO_1.0_Kaggle_2026/train/train_images'
ANNOTATIONS = BASE/'MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'
CODE = ROOT/'code'/__REVIEWED_VERSION__

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

if not CODE.is_dir():
    raise FileNotFoundError('확인한 원본 코드 폴더가 없습니다: '+str(CODE))
for path in (IMAGES, ANNOTATIONS, ROOT/'fold_plan.json'):
    if not path.exists(): raise FileNotFoundError(str(path))

raw = zlib.decompress(base64.b64decode(__PAYLOAD__))
if hashlib.sha256(raw).hexdigest() != __PAYLOAD_SHA__:
    raise ValueError('평가 전용 코드의 복사 내용이 달라졌습니다.')
extras = json.loads(raw)
WORK = Path('/content/solar_fivefold_current_evaluation')/__WORK_VERSION__
WORK.mkdir(parents=True, exist_ok=True)
manifest = json.loads((CODE/'source_manifest.json').read_text())
for name, expected in manifest.items():
    if Path(name).name != name or not name.endswith('.py'):
        raise ValueError('원본 소스 목록의 파일명이 올바르지 않습니다.')
    source = CODE/name
    if sha(source) != expected: raise ValueError('저장된 소스가 달라졌습니다: '+name)
    (WORK/name).write_bytes(source.read_bytes())
for name, source in extras.items():
    if Path(name).name != name or name in manifest:
        raise ValueError('기존 학습 소스와 충돌하는 평가 파일명입니다.')
    (WORK/name).write_text(source)

# Torch/CUDA는 Colab이 제공하는 설치를 유지합니다.
needed = []
for package, version in {'ultralytics':'8.4.182', 'pycocotools':'2.0.11'}.items():
    try: actual = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError: actual = None
    if actual != version: needed.append(package+'=='+version)
for module, package in [('cv2','opencv-python-headless'), ('scipy','scipy')]:
    try: __import__(module)
    except ImportError: needed.append(package)
if needed:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *needed], check=True)

OUTPUT.mkdir(parents=True, exist_ok=True)
environment = dict(os.environ)
environment.pop('TORCH_CUDNN_V8_API_DISABLED', None)
subprocess.run([sys.executable, '-c',
    'import torch; assert torch.cuda.is_available(), "Colab GPU(T4)를 선택하세요."; '
    'print("GPU:", torch.cuda.get_device_name(0), "PyTorch:", torch.__version__)'],
    env=environment, check=True)
command = [sys.executable, '-u', str(WORK/'export_current_fivefold.py'),
    '--root', str(ROOT), '--images', str(IMAGES), '--annotations', str(ANNOTATIONS),
    '--output', str(OUTPUT), '--device', 'cuda', '--native-stage', '30']
print('추가 학습 없이 현재 가중치를 평가합니다. 완료한 추론은 재사용합니다.', flush=True)
print('출력/재개 폴더:', OUTPUT, flush=True)
with (OUTPUT/'colab_execution.log').open('a', buffering=1) as log:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, start_new_session=True, env=environment)
    try:
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
        code = process.wait()
        if code: raise subprocess.CalledProcessError(code, command)
    except BaseException:
        try: os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError: pass
        try: process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise

result = json.loads((OUTPUT/'result.json').read_text())
print('평가 결과:', result.get('report'))
for archive in result['archives']:
    print('보내주실 ZIP:', archive)
    try: files.download(archive)
    except Exception as exc: print('Drive에서 직접 받을 수 있습니다:', exc)
'''


def build():
    sources = {'export_current_fivefold.py': (ROOT/'tools/export_current_fivefold.py').read_text()}
    for name, source in sources.items():
        compile(source, name, 'exec')
    raw = json.dumps(sources, sort_keys=True, ensure_ascii=False).encode()
    sha = hashlib.sha256(raw).hexdigest()
    cell = (CELL.replace('__REVIEWED_VERSION__', repr(REVIEWED_VERSION))
        .replace('__PAYLOAD__', repr(base64.b64encode(zlib.compress(raw, 9)).decode()))
        .replace('__PAYLOAD_SHA__', repr(sha)).replace('__WORK_VERSION__', repr(sha[:12])))
    compile(cell, 'current-epoch-evaluation-cell', 'exec')
    folder = ROOT/'notebooks'
    folder.mkdir(exist_ok=True)
    name = 'SolarFilament_CurrentEpoch_Evaluate_CopyCell.py'
    (folder/name).write_text(cell)
    notebook = dict(nbformat=4, nbformat_minor=5,
        metadata=dict(accelerator='GPU', kernelspec=dict(display_name='Python 3', language='python', name='python3')),
        cells=[dict(id='intro', cell_type='markdown', metadata={}, source=[
            '# Current-epoch evaluation only\n',
            '추가 학습 없이 저장된 모델을 평가합니다. Native30과 각 fold YOLO의 실제 완료 epoch를 구분합니다.\n']),
            dict(id='evaluate', cell_type='code', execution_count=None, metadata={}, outputs=[], source=cell.splitlines(True))])
    (folder/'SolarFilament_CurrentEpoch_Evaluate.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=2)+'\n')
    return dict(bundle_sha256=sha, reviewed_source_version=REVIEWED_VERSION,
                cell_bytes=len(cell.encode()), model_training_performed=False,
                file=str(folder/name))


if __name__ == '__main__':
    print(json.dumps(build(), indent=2))
