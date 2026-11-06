"""Build one-paste T4 validation inference, including audited prediction inputs."""
import base64
import hashlib
import json
from pathlib import Path
import shutil
import textwrap
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
MODULES = ('detr_best84_pipeline.py', 'detr_best84_tiles.py',
           'submission_inference.py', 'submission_native_recipe.py',
           'submission_yolo_recipe.py', 'submission_driver.py',
           'roi_filament_core.py', 'roi_filament_pipeline.py',
           'native_instance_data.py', 'native_instance_training.py',
           'native_instance_model.py')
BOOTSTRAP_NAME = 'detr_best84_colab_bootstrap.py'
SEED_PATH = ROOT.parent/'artifacts/detr_best84_seed/prediction_seed.json.gz'

BOOTSTRAP = r'''"""Restore this validation-only T4 experiment from immutable Drive files."""
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

def interrupted(signum, frame):
    raise KeyboardInterrupt('Interrupted by signal '+str(signum))

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def distinct_output(output, source):
    output=Path(output).resolve();source=Path(source).resolve()
    if output==source or output in source.parents or source in output.parents:
        raise ValueError('OUTPUT은 기존 실험/원본 폴더와 겹치지 않는 새 폴더여야 합니다.')

def restore(root, work_root=Path('/content/solar_detr_best84_code')):
    root=Path(root);pointer=json.loads((root/'current_code.json').read_text())
    if not re.fullmatch('[0-9a-f]{16}',pointer['version']):raise ValueError('Invalid saved code version')
    durable=root/'code'/pointer['version'];manifest_path=durable/'source_manifest.json'
    if digest(manifest_path)!=pointer['manifest_sha256']:raise ValueError('Drive 코드 목록이 변경되었습니다.')
    config_path=root/'experiment_config.json'
    if digest(config_path)!=pointer['config_sha256']:raise ValueError('저장된 실험 설정이 변경되었습니다.')
    cfg=json.loads(config_path.read_text())
    if Path(cfg['output_root']).resolve()!=root.resolve():raise ValueError('Saved output differs from this experiment')
    for key in ('original_base_root','source_root','images'):
        if cfg.get(key):distinct_output(root,cfg[key])
    asset=root/'assets'/'prediction_seed.json.gz'
    if Path(cfg['seed_path']).resolve()!=asset.resolve():raise ValueError('Prediction seed belongs to another folder')
    if digest(asset)!=pointer['seed_sha256'] or digest(asset)!=cfg['seed_sha256']:
        raise ValueError('저장된 검증 예측 입력이 변경되었습니다.')
    manifest=json.loads(manifest_path.read_text())
    if not manifest or any(Path(name).name!=name for name in manifest):raise ValueError('Invalid source manifest')
    work=Path(work_root)/pointer['version'];work.mkdir(parents=True,exist_ok=True)
    for name,expected in manifest.items():
        source=durable/name
        if not source.is_file() or digest(source)!=expected:raise ValueError('Drive 코드가 변경되었습니다: '+name)
        content=source.read_bytes();compile(content,name,'exec')
        pending=work/(name+'.pending');pending.write_bytes(content);pending.replace(work/name)
    return cfg,config_path,work

def verify_inputs(cfg):
    for key in ('native_checkpoint','roi_checkpoint','annotations','train_split','val_split','seed_path'):
        if not Path(cfg[key]).is_file():raise FileNotFoundError(key+': '+str(cfg[key]))
    if not Path(cfg['images']).is_dir():raise FileNotFoundError('train_images: '+str(cfg['images']))
    if cfg.get('native_precision')!='fp16' or cfg.get('roi_precision')!='fp16':
        raise ValueError('이번 T4 실험은 FP16으로 고정합니다.')
    if cfg.get('native_batch')!=1 or cfg.get('roi_batch')!=1:
        raise ValueError('이번 T4 실험은 native/ROI batch1로 고정합니다.')

def ensure_dependencies(work):
    versions={}
    for package in ('torch','torchvision'):
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError('Colab 기본 GPU PyTorch 런타임이 필요합니다: '+package) from exc
    required=[]
    for package,expected in [('segmentation-models-pytorch','0.5.0'),('timm','1.0.20'),
                             ('pycocotools','2.0.11')]:
        try:actual=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:actual=None
        if actual!=expected:required.append(package+'=='+expected)
    for module,package in [('numpy','numpy'),('cv2','opencv-python-headless'),('scipy','scipy')]:
        if importlib.util.find_spec(module) is None:required.append(package)
    if required:
        constraints=Path(work)/'keep_colab_torch.txt'
        constraints.write_text(''.join(k+'=='+v+'\n' for k,v in versions.items()))
        subprocess.run([sys.executable,'-m','pip','install','-q','--upgrade-strategy','only-if-needed',
                        '-c',str(constraints),*required],check=True)
    if any(importlib.metadata.version(k)!=v for k,v in versions.items()):
        raise RuntimeError('Colab torch/torchvision versions changed during setup')

def check_gpu(environment,work):
    probe="""import sys,torch,torchvision
sys.path.insert(0,sys.argv[1])
from native_instance_training import configure_cuda
configure_cuda()
assert torch.cuda.is_available(), 'Colab 런타임을 T4 GPU로 선택하세요.'
print('PyTorch:',torch.__version__,'CUDA:',torch.version.cuda,'GPU:',torch.cuda.get_device_name(0),flush=True)
layer=torch.nn.Conv2d(3,2,3,padding=1).cuda().eval()
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
    output=layer(torch.randn(1,3,64,64,device='cuda')).sigmoid()
assert output.dtype==torch.float16 and bool(torch.isfinite(output).all()), 'FP16 검사 실패'
torch.cuda.synchronize()
print('CUDA FP16 추론 사전 검사 통과; BF16을 사용하지 않습니다.',flush=True)
"""
    subprocess.run([sys.executable,'-u','-c',probe,str(work)],env=environment,check=True)

def run(root,max_images=None):
    if max_images is not None and (type(max_images) is not int or not 1<=max_images<=91):
        raise ValueError('MAX_IMAGES는 None 또는1~91 정수입니다.')
    cfg,config_path,work=restore(root);verify_inputs(cfg);ensure_dependencies(work)
    environment=dict(os.environ)
    environment['TORCH_CUDNN_V8_API_DISABLED']='1';environment['PYTHONUNBUFFERED']='1'
    environment['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    check_gpu(environment,work)
    print('T4 검증 추론만 실행: 추가 학습0epoch, 전체validation91장.',flush=True)
    print('DETR70/80 bbox와 기존 YOLO 전후처리 결과는 검증된 고정 캐시를 재사용합니다.',flush=True)
    print('best84 전체타일 기준선과 bbox선택타일3개 후보를 T4 FP16 batch1로 새로 추론합니다.',flush=True)
    print('이미지당36개 전체타일 forward를 후보끼리 공유합니다. 후보마다36회를 반복하지 않습니다.',flush=True)
    print('선택타일이 관측하지 않은 픽셀은 같은뷰 전체타일 예측으로 보완합니다.',flush=True)
    print('고정 ResNet34 ROI도 같은 bbox로 T4에서 비교하며 GPU모델은 순서대로 실행합니다.',flush=True)
    print('현재 실제public LB=0.39; 과거validation≈0.47과 구분합니다. 이번 점수도 고정개발셋 점수입니다.',flush=True)
    print('이번 실행 범위:', '전체91장' if max_images is None else str(max_images)+'장까지(부분점수는 전체PQ가 아님)',flush=True)
    print('Drive 재개 폴더:',root,flush=True)
    command=[sys.executable,'-u',str(work/'detr_best84_pipeline.py'),'--config',str(config_path)]
    if max_images is not None:command+=['--max-images',str(max_images)]
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                             text=True,bufsize=1,env=environment,start_new_session=True)
    try:
        for line in process.stdout:print(line,end='',flush=True)
        code=process.wait()
        if code:raise RuntimeError('추론 실행 실패(종료 코드 '+str(code)+'). 위 내부 traceback과 Drive colab_bootstrap.log를 확인하세요.')
    except BaseException:
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        raise
    finally:process.stdout.close()
    return json.loads((Path(root)/'latest_results.json').read_text())

if __name__=='__main__':
    signal.signal(signal.SIGTERM,interrupted)
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--max-images',type=int,default=None)
    args=parser.parse_args();result=run(Path(args.root),args.max_images)
    print(json.dumps({key:result.get(key) for key in ('status','completed_images','full_cohort_score_available','archives')},ensure_ascii=False,indent=2),flush=True)
'''

CONTROLS = r'''# T4에서 추가 학습 없이 검증91장 추론. 기본값으로 전체를 실행합니다.
MAX_IMAGES = None          # 2로 첫2장만 점검한 뒤 None으로 바꾸면 이어서 전체 진행
AUTO_DOWNLOAD = True       # 분석 ZIP만 받습니다. Kaggle 제출/업로드를 하지 않습니다.
'''

FIRST = CONTROLS+r'''
BASE = '/content/drive/MyDrive/filament-segmentation-2026'
YOLO_RUN = '/content/drive/MyDrive/SolarFilament_LimbOnly_YOLO26_mixed_FullRes'
NATIVE_CHECKPOINT = '/content/drive/MyDrive/SolarFilament_NativeUNet_A100/20261028T030808Z_bedf3e21/best.pt'
OUTPUT = BASE + '/fixed_split_experiments/detr_best84_t4_v1'

from google.colab import drive, files
from pathlib import Path
from datetime import datetime, timezone
import base64,hashlib,json,os,signal,subprocess,sys,zlib
drive.mount('/content/drive')
PAYLOAD = """__PAYLOAD__"""
raw=zlib.decompress(base64.b64decode(PAYLOAD))
if hashlib.sha256(raw).hexdigest()!=__SHA__:raise ValueError('실행 셀 전체를 다시 복사하세요: 코드 checksum 불일치')
bundle=json.loads(raw);sources=bundle['sources'];seed=base64.b64decode(bundle['seed_gzip_b64'],validate=True)
if set(sources)!=set(__FILES__):raise ValueError('Invalid source bundle')
if hashlib.sha256(seed).hexdigest()!=__SEED_SHA__:raise ValueError('검증 예측 입력 checksum 불일치')
root=Path(OUTPUT);original_base=Path(BASE)/'fixed_split_experiments/detection_roi_detr_v1'
source_root=Path(BASE)/'fixed_split_experiments/detr_a100_e150_v1'
original_path=original_base/'experiment_config.json'
original=json.loads(original_path.read_text()) if original_path.is_file() else {}
data_defaults=dict(images=BASE+'/MAGFiLO_1.0_Kaggle_2026/train/train_images',
    annotations=BASE+'/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json',
    train_split=YOLO_RUN+'/train_split.csv',val_split=YOLO_RUN+'/val_split.csv')
data_paths={key:original.get(key,value) for key,value in data_defaults.items()}
for protected in (original_base,source_root,Path(NATIVE_CHECKPOINT).parent,Path(data_paths['images'])):
    a=root.resolve();b=protected.resolve()
    if a==b or a in b.parents or b in a.parents:raise ValueError('OUTPUT은 기존 실험/원본과 겹치지 않는 새 폴더여야 합니다.')
roi_checkpoint=original_base/'roi/epoch030.pt'
for key,value in dict(data_paths,native_checkpoint=NATIVE_CHECKPOINT,roi_checkpoint=str(roi_checkpoint)).items():
    source=Path(value)
    if key=='images':
        if not source.is_dir():raise FileNotFoundError('train_images 폴더를 찾지 못했습니다: '+str(source))
    elif not source.is_file():raise FileNotFoundError(key+' 파일을 찾지 못했습니다: '+str(source))
durable=root/'code'/__VERSION__;durable.mkdir(parents=True,exist_ok=True)
asset=root/'assets'/'prediction_seed.json.gz';asset.parent.mkdir(parents=True,exist_ok=True)

def save_immutable(path, content):
    if path.exists():
        if path.read_bytes()!=content:raise ValueError('저장된 코드/설정/입력과 다릅니다. 새 OUTPUT 폴더를 사용하세요: '+str(path))
    else:
        pending=path.with_name(path.name+'.pending');pending.write_bytes(content);pending.replace(path)

save_immutable(asset,seed)
manifest={}
for name,source in sources.items():
    if Path(name).name!=name:raise ValueError('Invalid source name')
    compile(source,name,'exec');content=source.encode('utf-8')
    manifest[name]=hashlib.sha256(content).hexdigest();save_immutable(durable/name,content)
manifest_bytes=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
save_immutable(durable/'source_manifest.json',manifest_bytes)
cfg=dict(data_paths,output_root=str(root),original_base_root=str(original_base),source_root=str(source_root),
    native_checkpoint=NATIVE_CHECKPOINT,roi_checkpoint=str(roi_checkpoint),
    seed_path=str(asset),seed_sha256=__SEED_SHA__,cache_root='/content/solar_detr_best84_cache',
    device='cuda',native_precision='fp16',roi_precision='fp16',native_batch=1,roi_batch=1,
    strict=True,include_selected_only=False,
    code_dir=str(durable),code_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest())
config_bytes=(json.dumps(cfg,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode()
save_immutable(root/'experiment_config.json',config_bytes)
pointer=dict(version=__VERSION__,manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
             config_sha256=hashlib.sha256(config_bytes).hexdigest(),seed_sha256=__SEED_SHA__)
save_immutable(root/'current_code.json',(json.dumps(pointer,sort_keys=True,indent=2)+'\n').encode())
'''

RESUME = CONTROLS+r'''
# /content가 초기화돼도 Drive의 코드·설정·예측 입력·완료이미지 기록에서 이어갑니다.
OUTPUT = '/content/drive/MyDrive/filament-segmentation-2026/fixed_split_experiments/detr_best84_t4_v1'
from google.colab import drive, files
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,os,re,signal,subprocess,sys
drive.mount('/content/drive')
root=Path(OUTPUT);pointer_path=root/'current_code.json'
if not pointer_path.is_file():raise FileNotFoundError('최초 실행 셀을 먼저 실행하세요: '+str(pointer_path))
pointer=json.loads(pointer_path.read_text())
if not re.fullmatch('[0-9a-f]{16}',pointer['version']):raise ValueError('Invalid code version')
durable=root/'code'/pointer['version'];manifest_path=durable/'source_manifest.json'
if hashlib.sha256(manifest_path.read_bytes()).hexdigest()!=pointer['manifest_sha256']:raise ValueError('Drive 코드 목록이 변경되었습니다.')
for name,expected in json.loads(manifest_path.read_text()).items():
    if Path(name).name!=name or hashlib.sha256((durable/name).read_bytes()).hexdigest()!=expected:
        raise ValueError('Drive 코드가 변경되었습니다: '+name)
if hashlib.sha256((root/'experiment_config.json').read_bytes()).hexdigest()!=pointer['config_sha256']:
    raise ValueError('Drive 실험 설정이 변경되었습니다.')
if hashlib.sha256((root/'assets/prediction_seed.json.gz').read_bytes()).hexdigest()!=pointer['seed_sha256']:
    raise ValueError('Drive 검증 예측 입력이 변경되었습니다.')
'''

EXECUTE = r'''
if MAX_IMAGES is not None and (type(MAX_IMAGES) is not int or not 1<=MAX_IMAGES<=91):
    raise ValueError('MAX_IMAGES는 None 또는1~91 정수입니다. 에폭 수가 아닙니다.')
command=[sys.executable,'-u',str(durable/'detr_best84_colab_bootstrap.py'),'--root',str(root)]
if MAX_IMAGES is not None:command+=['--max-images',str(MAX_IMAGES)]
environment=dict(os.environ);environment['TORCH_CUDNN_V8_API_DISABLED']='1';environment['PYTHONUNBUFFERED']='1'
log_path=root/'colab_bootstrap.log'
print('전체 오류·실행 로그:',log_path)
with log_path.open('a',encoding='utf-8',buffering=1) as log:
    log.write('\n=== '+datetime.now(timezone.utc).isoformat()+' max_images='+str(MAX_IMAGES)+' ===\n')
    process=subprocess.Popen(command,env=environment,start_new_session=True,
                             stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    try:
        for line in process.stdout:print(line,end='',flush=True);log.write(line)
        code=process.wait()
        if code:raise RuntimeError('실행 실패(종료 코드 '+str(code)+'). 위 내부 traceback / 전체 로그: '+str(log_path))
    except BaseException:
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:process.wait(timeout=15)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        raise
    finally:process.stdout.close()
result=json.loads((root/'latest_results.json').read_text())
print('실행 상태:',result.get('status'),'완료이미지:',result.get('completed_images'))
print('현재 실제 public LB는0.39입니다. 이 ZIP은 고정validation 실험 결과이며 제출CSV가 아닙니다.')
for archive in result.get('archives',[]):
    print('분석용 ZIP:',archive)
    if AUTO_DOWNLOAD:
        try:files.download(archive)
        except Exception as exc:print('Drive의 위 경로에서 받을 수 있습니다:',exc)
'''

GUIDE = '''# T4 DETR bbox → best84 검증 추론

- Colab 런타임을 **T4 GPU**로 선택하고 `SolarFilament_DETR_Best84_T4_CopyCell.py` 전체를 한 셀에 복사합니다. 추가 학습은 **0epoch**입니다.
- 검증사진91장에 대한 기존 YOLO 후보와 DETR70/80 bbox를 검증된 예측 캐시에서 재사용합니다. 캐시에는 원본 사진이나 정답 마스크가 없습니다. 이 입력은 고정 validation 전용이며 test 예측으로 사용할 수 없습니다.
- best84는 **T4 FP16·batch1·원본1024타일**로 새 추론합니다. 기존 전체타일 기준선과 DETR70 confidence0.8, DETR80 confidence0.6/0.8의 세 후보를 비교합니다. 임계값은 미리 고정되어 있습니다.
- 전처리는 전체 이미지에서 기존 limb 보정을 먼저 적용합니다. base/hflip/vflip/hvflip 각각에서 bbox를 변환하고 기존 타일그리드를 선택한 뒤 원본으로 복원합니다. 기존 전후처리 규칙을 유지합니다.
- 기준선 전체36타일 forward를 한 번 계산하고 후보들이 결과를 공유합니다. 후보마다 별도36회를 다시 계산하지 않습니다. 선택타일 밖 픽셀은 같은뷰 전체타일 예측으로 보완합니다. 따라서 이번 비교의 총GPU추론이 선택타일 수만큼만 줄어드는 것은 아닙니다.
- best84 단계 완료 후 모델을 해제하고 같은 bbox를 고정 ResNet34 ROI U-Net30에 전달합니다. ROI도 FP16·batch1입니다. 두 모델을 동시에 GPU에 올리지 않습니다.
- 최종 비교에는 **이번 T4 전체타일 기준선**을 사용합니다. 과거 약0.47은 반복한 validation 점수이고, 현재 실제 **public LB는0.39**입니다. A100 BF16 예측을 일부 고정 입력으로 쓰는 이번 실험의 출처와 정밀도는 기록됩니다.
- `MAX_IMAGES=None`이 기본이며91장 전체를 처리합니다. `MAX_IMAGES=2`로 먼저2장을 점검할 수 있고, 이후 None으로 바꾸면 완료이미지를 재사용합니다. 일부 이미지만 처리한 결과를 전체PQ로 보고하지 않습니다.
- 세션이 종료되면 최초 셀을 다시 실행하거나 `_ResumeCell.py`를 한 셀에 붙여 실행합니다. Drive의 코드·설정·입력 캐시·완료이미지 기록을 복원합니다. 두 셀을 동시에 실행하지 마세요.
- 코드·가중치·입력·GPU/라이브러리/정밀도가 바뀌면 다른 실험과 섞이지 않도록 재개를 차단합니다. 같은 조건으로 복원하거나 새OUTPUT을 사용합니다. 중단 당시 완료되지 않은 이미지 단계는 다시 실행할 수 있습니다.
- 내부 오류는 셀 출력과 `colab_bootstrap.log`에 함께 저장합니다. Colab 기본 torch/torchvision CUDA 조합을 유지하고 필요한 라이브러리만 설치합니다.
- 출력 폴더는 `/content/drive/MyDrive/filament-segmentation-2026/fixed_split_experiments/detr_best84_t4_v1`입니다. 원본 사진·기존 실험·가중치는 변경하지 않습니다.
- 완료되면28MiB 이하 분석 ZIP이 생성됩니다. 가중치나 원본 이미지는 ZIP에 넣지 않습니다. ZIP을 검토한 후 개선 후보의 동일한 추론 규칙으로 test 제출CSV를 만들어 public LB에서 확인하는 단계가 필요합니다. 이 셀은 test 추론·Kaggle 업로드를 하지 않습니다.
- 여기서의 코드 검증은 CPU 기능·재개·통합 검사입니다. 실제 T4 소요시간·최대메모리·새PQ는 아직 측정하지 않았습니다.
'''

def build_sources():
    sources={name:(ROOT/'tools'/name).read_text(encoding='utf-8') for name in MODULES}
    sources[BOOTSTRAP_NAME]=BOOTSTRAP
    for name,source in sources.items():compile(source,name,'exec')
    return sources

def build(downloads=None, seed_path=None):
    sources=build_sources();seed=Path(seed_path or SEED_PATH).read_bytes()
    seed_sha=hashlib.sha256(seed).hexdigest()
    raw=json.dumps(dict(sources=sources,seed_gzip_b64=base64.b64encode(seed).decode()),
        ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
    stamp=hashlib.sha256(raw).hexdigest()
    manifest={name:hashlib.sha256(source.encode()).hexdigest() for name,source in sources.items()}
    manifest_bytes=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
    payload='\n'.join(textwrap.wrap(base64.b64encode(zlib.compress(raw,9)).decode(),120))
    cell=FIRST.replace('__PAYLOAD__',payload).replace('__SHA__',repr(stamp)).replace('__FILES__',repr(tuple(sources))).replace('__VERSION__',repr(stamp[:16])).replace('__SEED_SHA__',repr(seed_sha))+EXECUTE
    resume=RESUME+EXECUTE
    compile(cell,'detr-best84-first','exec');compile(resume,'detr-best84-resume','exec')
    output=ROOT/'notebooks';output.mkdir(parents=True,exist_ok=True)
    stem='SolarFilament_DETR_Best84_T4'
    first_path=output/(stem+'_CopyCell.py');resume_path=output/(stem+'_ResumeCell.py')
    notebook_path=output/(stem+'.ipynb');guide_path=output/(stem+'_실행안내.md')
    first_path.write_text(cell,encoding='utf-8');resume_path.write_text(resume,encoding='utf-8')
    guide_path.write_text(GUIDE,encoding='utf-8')
    notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python'),colab=dict(name=notebook_path.name)),
        cells=[dict(id='intro',cell_type='markdown',metadata={},source=GUIDE.splitlines(True)),
               dict(id='first-run',cell_type='code',metadata={},execution_count=None,outputs=[],source=cell.splitlines(True)),
               dict(id='resume-info',cell_type='markdown',metadata={},source=['## 새 세션 재개\n','최초 셀 또는 아래 재개 셀 중 하나만 실행합니다.\n']),
               dict(id='resume',cell_type='code',metadata={},execution_count=None,outputs=[],source=resume.splitlines(True))])
    notebook_path.write_text(json.dumps(notebook,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    metadata=dict(copy_cell=str(first_path),resume_cell=str(resume_path),notebook=str(notebook_path),guide=str(guide_path),
        source_bundle_sha256=stamp,source_count=len(sources),source_sha256=manifest,
        source_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),cell_bytes=len(cell.encode()),resume_bytes=len(resume.encode()),
        seed_sha256=seed_sha,seed_bytes=len(seed),validation_images=91,additional_training_epochs=0,
        actual_public_LB=.39,historical_validation_PQ_approx=.47,new_PQ_measured=False,T4_inference_tested_here=False,
        native_precision='fp16',roi_precision='fp16',native_batch=1,roi_batch=1,
        native_full_tile_forwards_per_image=36,shared_across_arms=True,
        fixed_arms=['full','detr70_c08','detr80_c06','detr80_c08'],seed_is_fixed_validation_only=True,
        auto_Kaggle_submission=False,test_inference=False)
    downloads=Path(downloads) if downloads is not None else ROOT.parent/'artifacts/downloads'
    downloads.mkdir(parents=True,exist_ok=True)
    files=[first_path,resume_path,notebook_path,guide_path]
    for path in files:shutil.copy2(path,downloads/path.name)
    metadata['downloads']={path.name:str(downloads/path.name) for path in files}
    metadata['package']=str(downloads/(stem+'_Package.zip'))
    metadata_path=downloads/(stem+'_metadata.json')
    metadata_path.write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with zipfile.ZipFile(metadata['package'],'w',zipfile.ZIP_DEFLATED) as archive:
        for path in files:archive.write(path,path.name)
        archive.write(metadata_path,metadata_path.name)
        for name,source in sources.items():archive.writestr('source/'+name,source)
        archive.writestr('source/source_manifest.json',manifest_bytes)
        archive.writestr('assets/prediction_seed.json.gz',seed)
    return metadata

if __name__=='__main__':print(json.dumps(build(),ensure_ascii=False,indent=2))
