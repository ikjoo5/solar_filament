"""Build one-paste and short-resume Colab cells for CPU context descriptors."""
import base64
import hashlib
import json
from pathlib import Path
import zlib

ROOT=Path(__file__).resolve().parents[1]
SETUP=r'''from google.colab import drive, files
from pathlib import Path
import hashlib, importlib.util, json, os, signal, subprocess, sys
drive.mount('/content/drive')
BASE=Path('/content/drive/MyDrive/filament-segmentation-2026')
OUTPUT=BASE/'fixed_split_experiments/mask_quality_v1/candidate_context_features/v1'
needed=[]
for module,package in [('numpy','numpy'),('cv2','opencv-python-headless'),('pycocotools','pycocotools==2.0.11')]:
    if importlib.util.find_spec(module) is None:needed.append(package)
if needed:subprocess.run([sys.executable,'-m','pip','install','-q',*needed],check=True)
'''
RUN=r'''
print('CPU로 기존 train616+validation91의 모든 원시 후보 특징을 수집합니다.',flush=True)
print('학습·모델 추론 없이, 원사진과 가중치를 제외한 특징/RLE/별도 labels만 저장합니다.',flush=True)
print('이미지별 완료 캐시는 Drive에 남으며 중단 후 검증하고 재사용합니다:',OUTPUT,flush=True)
print('사진·코드·특징 설정·라이브러리 버전이 바뀌면 기존 캐시와 섞지 않고 중단합니다.',flush=True)
OUTPUT.mkdir(parents=True,exist_ok=True)
with (OUTPUT/'colab_execution.log').open('a',encoding='utf-8',buffering=1) as log:
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
    try:
        for line in process.stdout:print(line,end='',flush=True);log.write(line)
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
print('상태:',result['status'],'사진:',result['completed_physical_images'],'/',result['expected_physical_images'],
      '역할별 후보:',result['rows_by_role'],flush=True)
for path in result['archives']:
    print('보내주실 ZIP:',path,flush=True)
    try:files.download(path)
    except Exception as exc:print('Drive에서 직접 받을 수 있습니다:',exc,flush=True)
'''
FIRST=r'''# CPU 런타임: 기존 원시 후보의 픽셀/문맥 특징을 수집합니다. 추가 학습 없음.
MAX_IMAGES=None  # 처음 1장만 점검하려면 1. 이후 None으로 바꾸면 전체707장으로 재개합니다.
'''+SETUP+r'''
import base64,zlib
ROOT=BASE/'fixed_split_experiments/mask_quality_v1'
IMAGES=BASE/'MAGFiLO_1.0_Kaggle_2026/train/train_images'
source=zlib.decompress(base64.b64decode(__PAYLOAD__))
if hashlib.sha256(source).hexdigest()!=__SHA__:raise ValueError('복사한 코드가 변경됐습니다. 전체 셀을 다시 복사하세요.')
WORK=Path('/content/solar_candidate_context')/__VERSION__
WORK.mkdir(parents=True,exist_ok=True)
script=WORK/'export_candidate_context_features.py';script.write_bytes(source)
command=[sys.executable,'-u',str(script),'--root',str(ROOT),'--images',str(IMAGES),'--output',str(OUTPUT)]
if MAX_IMAGES is not None:command+=['--limit-images',str(MAX_IMAGES)]
'''+RUN
RESUME='''# 최초 셀을 한번 실행한 뒤, 세션 종료 후에는 이 짧은 셀로 전체 작업을 재개할 수 있습니다.
'''+SETUP+r'''
pointer=OUTPUT/'current_code.json'
if not pointer.is_file():raise FileNotFoundError('최초 실행용 context 특징 복사 셀을 먼저 실행하세요.')
saved=json.loads(pointer.read_text());script=Path(saved['path'])
if not script.is_file() or hashlib.sha256(script.read_bytes()).hexdigest()!=saved['sha256']:
    raise ValueError('Drive에 보관한 특징 수집 코드가 변경됐습니다.')
request=json.loads((OUTPUT/'request.json').read_text())
command=[sys.executable,'-u',str(script),'--root',request['root'],'--images',request['images'],'--output',str(OUTPUT)]
'''+RUN


def build():
    source=(ROOT/'tools/export_candidate_context_features.py').read_bytes()
    compile(source,'export_candidate_context_features.py','exec');digest=hashlib.sha256(source).hexdigest()
    cell=(FIRST.replace('__PAYLOAD__',repr(base64.b64encode(zlib.compress(source,9)).decode()))
        .replace('__SHA__',repr(digest)).replace('__VERSION__',repr(digest[:12])))
    compile(cell,'context-first-cell','exec');compile(RESUME,'context-resume-cell','exec')
    folder=ROOT/'notebooks';folder.mkdir(exist_ok=True)
    first=folder/'SolarFilament_CandidateContextFeatures_CopyCell.py';first.write_text(cell)
    resume=folder/'SolarFilament_CandidateContextFeatures_ResumeCell.py';resume.write_text(RESUME)
    notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python')),
        cells=[dict(id='intro',cell_type='markdown',metadata={},source=[
            '# CPU 원시 후보 문맥 특징 수집\n',
            '원래 train616/validation91의 모든 후보를 사용합니다. 특징과 정답 labels를 분리해 저장하며 학습·추론은 하지 않습니다.\n',
            '원사진과 가중치는 ZIP에 포함하지 않습니다. 실제707장 실행 시간과 출력 크기는 아직 측정하지 않았습니다.\n']),
            dict(id='first',cell_type='code',metadata={},execution_count=None,outputs=[],source=cell.splitlines(True)),
            dict(id='resume-info',cell_type='markdown',metadata={},source=['## 세션 종료 후 재개\n','최초 셀을 실행해 Drive에 코드가 보관된 다음에만 아래 셀을 사용합니다.\n']),
            dict(id='resume',cell_type='code',metadata={},execution_count=None,outputs=[],source=RESUME.splitlines(True))])
    path=folder/'SolarFilament_CandidateContextFeatures.ipynb';path.write_text(json.dumps(notebook,ensure_ascii=False,indent=2)+'\n')
    return dict(copy_cell=str(first),resume_cell=str(resume),notebook=str(path),source_sha256=digest,
        cell_bytes=len(cell.encode()),resume_bytes=len(RESUME.encode()),model_training=False,new_inference=False,CPU_only=True)


if __name__=='__main__':print(json.dumps(build(),ensure_ascii=False,indent=2))
