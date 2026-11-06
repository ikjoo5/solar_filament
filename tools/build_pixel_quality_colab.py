"""Package the label-free CPU image-descriptor exporter in one Colab cell."""
import base64
import hashlib
import json
from pathlib import Path
import zlib

ROOT=Path(__file__).resolve().parents[1]
CELL=r'''# 원영상의 픽셀 품질 특징 수집 — CPU 전용, 추가 학습 없음
from google.colab import drive, files
from pathlib import Path
import base64, hashlib, importlib.util, json, os, signal, subprocess, sys, zlib

drive.mount('/content/drive')
ROOT=Path('/content/drive/MyDrive/filament-segmentation-2026/fivefold_study_v2')
SNAPSHOT=ROOT/'current_epoch_evaluation/native030_yolo_current_v1'
IMAGES=ROOT.parent/'MAGFiLO_1.0_Kaggle_2026/train/train_images'
OUTPUT=ROOT/'pixel_quality_exports/v1'
MAX_IMAGES=None  # 전체707장. 1로 실행하면 1장만 사전 확인하고, None으로 바꿔 이어갑니다.

for path in (ROOT,SNAPSHOT,IMAGES):
    if not path.is_dir():raise FileNotFoundError(str(path))
payload=zlib.decompress(base64.b64decode(__PAYLOAD__))
if hashlib.sha256(payload).hexdigest()!=__SHA__:
    raise ValueError('복사한 픽셀 특징 코드의 내용이 달라졌습니다.')
WORK=Path('/content/solar_pixel_quality')/__VERSION__
WORK.mkdir(parents=True,exist_ok=True)
script=WORK/'export_solar_pixel_quality.py';script.write_bytes(payload)
needed=[]
for module,package in [('numpy','numpy'),('scipy','scipy'),('cv2','opencv-python-headless'),('pycocotools','pycocotools==2.0.11')]:
    if importlib.util.find_spec(module) is None:needed.append(package)
if needed:subprocess.run([sys.executable,'-m','pip','install','-q',*needed],check=True)
OUTPUT.mkdir(parents=True,exist_ok=True)
command=[sys.executable,'-u',str(script),'--root',str(ROOT),'--snapshot',str(SNAPSHOT),
    '--images',str(IMAGES),'--output',str(OUTPUT)]
if MAX_IMAGES is not None:command+=['--limit-images',str(MAX_IMAGES)]
print('CPU로 사진을 읽어 scalar 특징만 저장합니다. GPU·모델·정답 JSON은 사용하지 않습니다.',flush=True)
print('저장/재개 폴더:',OUTPUT,flush=True)
with (OUTPUT/'colab_execution.log').open('a',buffering=1) as log:
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
        text=True,bufsize=1,start_new_session=True)
    try:
        for line in process.stdout:
            print(line,end='',flush=True);log.write(line)
        code=process.wait()
        if code:raise subprocess.CalledProcessError(code,command)
    except BaseException:
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait()
        raise
result=json.loads((OUTPUT/'result.json').read_text())
print('상태:',result['status'],'완료사진:',result['completed_physical_images'],
    '/',result['expected_physical_images'],'특징행:',result['targets'])
for archive in result['archives']:
    print('보내주실 특징 ZIP:',archive)
    try:files.download(archive)
    except Exception as exc:print('Drive에서 직접 받을 수 있습니다:',exc)
'''


def build():
    source=(ROOT/'tools/export_solar_pixel_quality.py').read_bytes()
    compile(source,'export_solar_pixel_quality.py','exec')
    digest=hashlib.sha256(source).hexdigest()
    cell=(CELL.replace('__PAYLOAD__',repr(base64.b64encode(zlib.compress(source,9)).decode()))
        .replace('__SHA__',repr(digest)).replace('__VERSION__',repr(digest[:12])))
    compile(cell,'pixel-quality-cell','exec')
    folder=ROOT/'notebooks';folder.mkdir(exist_ok=True)
    name=folder/'SolarFilament_PixelQuality_CopyCell.py';name.write_text(cell)
    notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3')),
        cells=[dict(id='intro',cell_type='markdown',metadata={},source=[
            '# CPU 원영상 품질 특징 수집\n',
            '기존 가중치·정답·원본 사진을 수정하지 않고, 사진의 선형 구조와 대비 특징만 수집합니다.\n',
            'CPU 런타임으로 실행할 수 있으며 전체707장의 실제 실행 시간·용량·성능 개선은 아직 측정하지 않았습니다.\n']),
            dict(id='run',cell_type='code',execution_count=None,metadata={},outputs=[],source=cell.splitlines(True))])
    (folder/'SolarFilament_PixelQuality.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=2)+'\n')
    return dict(source_sha256=digest,cell_bytes=len(cell.encode()),file=str(name),gpu_required=False,
        original_image_upload_required=False,model_training=False,ground_truth_read=False)


if __name__=='__main__':print(json.dumps(build(),indent=2))
