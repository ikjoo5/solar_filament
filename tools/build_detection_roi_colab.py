"""Self-contained detection→ROI U-Net comparison with durable DETR resume."""
import base64
import hashlib
import json
from pathlib import Path
import textwrap
import zlib

ROOT = Path(__file__).resolve().parents[1]
MODULES = ('roi_filament_core.py', 'roi_filament_pipeline.py', 'filament_detr.py',
           'native_instance_data.py', 'native_instance_training.py', 'native_instance_model.py')
BOOTSTRAP_NAME = 'detection_roi_colab_bootstrap.py'

BOOTSTRAP = r'''"""Restore immutable experiment code/config and stream the actual worker."""
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
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def restore(root, work_root=Path('/content/solar_detection_roi_code')):
    root=Path(root)
    pointer=json.loads((root/'current_code.json').read_text())
    if not re.fullmatch('[0-9a-f]{16}',pointer['version']):raise ValueError('Invalid saved code version')
    durable=root/'code'/pointer['version'];manifest_path=durable/'source_manifest.json'
    if digest(manifest_path)!=pointer['manifest_sha256']:raise ValueError('Drive 코드 목록이 변경되었습니다.')
    config_path=root/'experiment_config.json'
    if digest(config_path)!=pointer['config_sha256']:
        raise ValueError('저장된 설정이 변경되었습니다. 실행 범위 PHASE/ROI_UNTIL/DETR_UNTIL만 바꾸세요.')
    cfg=json.loads(config_path.read_text())
    if Path(cfg['output_root'])!=root:raise ValueError('Saved output folder differs from this experiment')
    manifest=json.loads(manifest_path.read_text())
    if not manifest or any(Path(name).name!=name for name in manifest):raise ValueError('Invalid source manifest')
    work=Path(work_root)/pointer['version'];work.mkdir(parents=True,exist_ok=True)
    for name,expected in manifest.items():
        source=durable/name
        if not source.is_file() or digest(source)!=expected:raise ValueError('Drive 코드가 변경되었습니다: '+name)
        content=source.read_bytes();compile(content,name,'exec')
        pending=work/(name+'.pending');pending.write_bytes(content);pending.replace(work/name)
    return cfg,config_path,work

def check_paths(cfg):
    for key in ('annotations','train_split','val_split','yolo'):
        if not Path(cfg[key]).is_file():raise FileNotFoundError(key+': '+str(cfg[key]))
    if not Path(cfg['images']).is_dir():raise FileNotFoundError('images: '+str(cfg['images']))

def ensure_dependencies(work):
    versions={}
    for package in ('torch','torchvision'):
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError('Colab 기본 GPU PyTorch 런타임이 필요합니다: '+package) from exc
    required=[]
    for package,expected in [('transformers','4.57.1'),('ultralytics','8.4.182'),
                             ('segmentation-models-pytorch','0.5.0'),('timm','1.0.20'),('pycocotools','2.0.11')]:
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

def check_gpu(environment, work, cfg):
    probe="""import sys,torch,torchvision
sys.path.insert(0,sys.argv[1])
from native_instance_training import configure_cuda
configure_cuda()
assert torch.cuda.is_available(), 'Colab 런타임을 T4 GPU로 선택하세요.'
print('PyTorch:',torch.__version__,'CUDA:',torch.version.cuda,'GPU:',torch.cuda.get_device_name(0),flush=True)
layer=torch.nn.Conv2d(3,3,3,padding=1).cuda()
x=torch.randn(1,3,32,32,device='cuda',requires_grad=True)
with torch.autocast('cuda',dtype=torch.float16): loss=layer(x).float().square().mean()
loss.backward();torch.cuda.synchronize()
print('CUDA FP16 forward/backward 사전 검사 통과',flush=True)
"""
    subprocess.run([sys.executable,'-u','-c',probe,str(work)],env=environment,check=True)

def run(root, phase, roi_until, detr_until):
    cfg,config_path,work=restore(root)
    if phase not in ('all','yolo_roi','compare_detr','evaluate'):raise ValueError('Unknown phase')
    for value,key in ((roi_until,'roi_epochs'),(detr_until,'detr_epochs')):
        if not 1<=value<=int(cfg[key]):raise ValueError('실행 목표는 1~저장된 최종 epoch 범위여야 합니다: '+key)
    check_paths(cfg);ensure_dependencies(work)
    environment=dict(os.environ)
    environment['TORCH_CUDNN_V8_API_DISABLED']='1';environment['PYTHONUNBUFFERED']='1'
    environment['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    check_gpu(environment,work,cfg)
    command=[sys.executable,'-u',str(work/'roi_filament_pipeline.py'),'--config',str(config_path),
             '--phase',phase,'--roi-until',str(roi_until),'--detr-until',str(detr_until)]
    print('실행 단계:',phase,'| 이번 DETR:',detr_until,'/ 최종:',cfg['detr_epochs'],
          '| 이번 ROI U-Net:',roi_until,'/ 최종:',cfg['roi_epochs'],flush=True)
    print('기본 all은 DETR30 + ROI U-Net30 = 두 모델 합계60epoch입니다. 기존 YOLO는 추가 학습하지 않습니다.',flush=True)
    print('all에서 DETR이 고정 최종 epoch에 도달하기 전에는 ROI 학습을 시작하지 않습니다. ROI 학습 데이터의 변경을 막기 위한 순서입니다.',flush=True)
    print('ROI epoch는',cfg['roi_steps_per_epoch'],'mini-batches ×',cfg['roi_batch'],'crop 샘플입니다.',flush=True)
    print('yolo_roi는 ROI 학습만, compare_detr는 기존 ROI를 유지하고 DETR 학습·비교, evaluate는 평가만 합니다.',flush=True)
    print('고정 train616/validation91, 실제 대회 train만 학습. Drive 폴더:',root,flush=True)
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                             text=True,bufsize=1,env=environment,start_new_session=True)
    try:
        for line in process.stdout:print(line,end='',flush=True)
        code=process.wait()
        if code:raise subprocess.CalledProcessError(code,command)
    except BaseException:
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        raise
    finally:process.stdout.close()
    return json.loads((Path(root)/'result.json').read_text())

if __name__=='__main__':
    signal.signal(signal.SIGTERM,interrupted)
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--phase',choices=('all','yolo_roi','compare_detr','evaluate'),default='all')
    parser.add_argument('--roi-until',type=int,default=30);parser.add_argument('--detr-until',type=int,default=30)
    args=parser.parse_args();result=run(Path(args.root),args.phase,args.roi_until,args.detr_until)
    print(json.dumps({key:result.get(key) for key in ('status','phase','archives')},ensure_ascii=False,indent=2),flush=True)
'''

CONTROLS = r'''# Default all: DETR30 + ROI U-Net30 = 합계60epoch, 서로 다른 두 모델입니다.
PHASE = 'all'        # all / yolo_roi / compare_detr / evaluate
ROI_UNTIL = 30       # ROI 고정 전체30, 이번 중간정지는10/20 가능
DETR_UNTIL = 30      # all에서30에 도달한 다음에만 ROI를 시작;10/20이면 DETR만 중간정지
AUTO_DOWNLOAD = True
'''

FIRST = CONTROLS + r'''
ROI_ENCODER = 'resnet34'  # resnet18 / resnet34 / resnet50
ROI_SIZE = 384
ROI_CONTEXT = 1.5
DETR_IMAGE_SIZE = 800
ROI_EPOCHS = 30
DETR_EPOCHS = 30
BASE = '/content/drive/MyDrive/filament-segmentation-2026'
YOLO_RUN = '/content/drive/MyDrive/SolarFilament_LimbOnly_YOLO26_mixed_FullRes'
OUTPUT = BASE + '/fixed_split_experiments/detection_roi_detr_v1'

from google.colab import drive, files
from pathlib import Path
from datetime import datetime, timezone
import base64,hashlib,json,os,signal,subprocess,sys,zlib
drive.mount('/content/drive')
if ROI_ENCODER not in ('resnet18','resnet34','resnet50'):raise ValueError('ROI_ENCODER 값을 확인하세요.')
PAYLOAD = """__PAYLOAD__"""
raw=zlib.decompress(base64.b64decode(PAYLOAD))
if hashlib.sha256(raw).hexdigest()!=__SHA__:raise ValueError('실행 셀 전체를 다시 복사하세요: 코드 checksum 불일치')
sources=json.loads(raw)
if set(sources)!=set(__FILES__):raise ValueError('Invalid source bundle')
root=Path(OUTPUT);durable=root/'code'/__VERSION__;durable.mkdir(parents=True,exist_ok=True)

def save_immutable(path, content):
    if path.exists():
        if path.read_bytes()!=content:raise ValueError('저장된 코드/설정과 다릅니다. 새 OUTPUT 폴더를 사용하세요: '+str(path))
    else:
        pending=path.with_name(path.name+'.pending');pending.write_bytes(content);pending.replace(path)

manifest={}
for name,source in sources.items():
    if Path(name).name!=name:raise ValueError('Invalid source name')
    compile(source,name,'exec');content=source.encode('utf-8')
    manifest[name]=hashlib.sha256(content).hexdigest();save_immutable(durable/name,content)
manifest_bytes=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
save_immutable(durable/'source_manifest.json',manifest_bytes)
data=Path(BASE)/'MAGFiLO_1.0_Kaggle_2026/train';old=Path(YOLO_RUN)
cfg=dict(images=str(data/'train_images'),annotations=str(data/'MAGFiLO_1.0_Annotations_kaggle2026_train.json'),
    train_split=str(old/'train_split.csv'),val_split=str(old/'val_split.csv'),yolo=str(old/'best.pt'),
    output_root=str(root),cache_root='/content/solar_detection_roi_cache',
    roi_encoder=ROI_ENCODER,roi_pretrained=True,roi_epochs=ROI_EPOCHS,roi_size=ROI_SIZE,roi_context=ROI_CONTEXT,
    roi_batch=4,roi_accumulate=1,roi_steps_per_epoch=256,roi_lr=1e-4,roi_weight_decay=.01,
    detr_epochs=DETR_EPOCHS,detr_batch=1,detr_accumulate=4,detr_image_size=DETR_IMAGE_SIZE,
    detr_lr=1e-5,detr_backbone_lr=1e-6,detr_pretrained=True,detr_model_name='facebook/detr-resnet-50',
    seed=42,device='cuda',amp=True,preprocessing='limb',strict=True,
    proposal_confidence=.05,eval_box_thresholds=[.05,.30],eval_min_areas=[150,300],
    mask_threshold=.5,roi_min_quality=.6,mask_nms_iou=.5,
    code_dir=str(durable),code_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest())
config_bytes=(json.dumps(cfg,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode()
save_immutable(root/'experiment_config.json',config_bytes)
pointer=dict(version=__VERSION__,manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
             config_sha256=hashlib.sha256(config_bytes).hexdigest())
save_immutable(root/'current_code.json',(json.dumps(pointer,sort_keys=True,indent=2)+'\n').encode())
'''

RESUME = CONTROLS + r'''
# 세션이 초기화돼도 Drive에 저장한 원래 코드·설정·checkpoint에서 재개합니다.
OUTPUT = '/content/drive/MyDrive/filament-segmentation-2026/fixed_split_experiments/detection_roi_detr_v1'
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
    raise ValueError('Drive 학습 설정이 변경되었습니다.')
'''

EXECUTE = r'''
command=[sys.executable,'-u',str(durable/'detection_roi_colab_bootstrap.py'),
         '--root',str(root),'--phase',PHASE,'--roi-until',str(ROI_UNTIL),'--detr-until',str(DETR_UNTIL)]
environment=dict(os.environ);environment['TORCH_CUDNN_V8_API_DISABLED']='1';environment['PYTHONUNBUFFERED']='1'
log_path=root/'colab_bootstrap.log'
print('전체 오류·실행 로그:',log_path)
with log_path.open('a',encoding='utf-8',buffering=1) as log:
    log.write('\n=== '+datetime.now(timezone.utc).isoformat()+' '+PHASE+' ROI'+str(ROI_UNTIL)+' DETR'+str(DETR_UNTIL)+' ===\n')
    process=subprocess.Popen(command,env=environment,start_new_session=True,
                             stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    try:
        for line in process.stdout:print(line,end='',flush=True);log.write(line)
        code=process.wait()
        if code:raise subprocess.CalledProcessError(code,command)
    except BaseException:
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:process.wait(timeout=15)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        raise
    finally:process.stdout.close()
result=json.loads((root/'result.json').read_text())
print('실행 상태:',result.get('status'),'단계:',result.get('phase'))
for archive in result.get('archives',[]):
    print('분석용 ZIP:',archive)
    if AUTO_DOWNLOAD:
        try:files.download(archive)
        except Exception as exc:print('Drive의 위 경로에서 받을 수 있습니다:',exc)
'''


def build_sources():
    sources={name:(ROOT/'tools'/name).read_text(encoding='utf-8') for name in MODULES}
    sources[BOOTSTRAP_NAME]=BOOTSTRAP
    for name,source in sources.items():compile(source,name,'exec')
    return sources


def build():
    sources=build_sources()
    raw=json.dumps(sources,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
    stamp=hashlib.sha256(raw).hexdigest()
    manifest={name:hashlib.sha256(source.encode()).hexdigest() for name,source in sources.items()}
    manifest_bytes=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
    payload='\n'.join(textwrap.wrap(base64.b64encode(zlib.compress(raw,9)).decode(),120))
    cell=FIRST.replace('__PAYLOAD__',payload).replace('__SHA__',repr(stamp)).replace('__FILES__',repr(tuple(sources))).replace('__VERSION__',repr(stamp[:16]))+EXECUTE
    resume=RESUME+EXECUTE
    compile(cell,'detection-roi-first','exec');compile(resume,'detection-roi-resume','exec')
    output=ROOT/'notebooks';output.mkdir(parents=True,exist_ok=True)
    first_path=output/'SolarFilament_DetectionROI_DETR_T4_CopyCell.py'
    resume_path=output/'SolarFilament_DetectionROI_DETR_T4_ResumeCell.py'
    notebook_path=output/'SolarFilament_DetectionROI_DETR_T4.ipynb'
    first_path.write_text(cell,encoding='utf-8');resume_path.write_text(resume,encoding='utf-8')
    intro=['# T4: 검출 bbox → 객체별 ROI U-Net, YOLO / DETR 비교\n',
           '기본 all은 기존 YOLO의 실제 bbox export, DETR30epoch, ROI U-Net30epoch, 전체 validation 비교를 실행합니다. 총60epoch이며 서로 다른 두 모델입니다. 기존 YOLO는 추가 학습하지 않습니다.\n',
           'DETR은 facebook/detr-resnet-50의 일반 사전학습 가중치에서 시작하고, ROI U-Net은 ImageNet ResNet34 encoder에서 시작합니다. 첫 실행에는 사전학습 가중치를 내려받습니다. 논문의 데이터나 가중치를 그대로 사용하는 재현이 아닙니다.\n',
           '원래 train616 / validation91을 유지하고 대회 train만 학습합니다. ROI 입력은 grayscale crop+bbox 위치의2채널이며 출력은 지정 객체 하나의 마스크입니다.\n',
           'DETR epoch는 train 주석 그룹의 통과, ROI epoch는256mini-batch×4=1024개 crop 샘플입니다. 기본 ROI30epoch는30,720crop입니다. 이 두 epoch의 계산량은 같지 않습니다.\n',
           'ROI_UNTIL과DETR_UNTIL을10/20으로 중간 정지하고30으로 재개할 수 있습니다. 전체 스케줄은 각각30으로 고정하며 숨은100epoch 목표가 없습니다. 완료 epoch의 전체optimizer상태를 Drive에서 복원합니다.\n',
           'all에서 DETR_UNTIL이30보다 작으면 DETR만 중간 저장하고 ROI는 아직 학습하지 않습니다. 먼저 DETR30을 완료해야 혼합 ROI 학습 후보가 고정됩니다. 이후 ROI_UNTIL만10→20→30으로 이어갈 수 있습니다.\n',
           'yolo_roi는 ROI만 먼저 학습합니다. compare_detr는 기존 ROI를 다시 학습하지 않고 DETR을 학습해 같은 ROI로 비교합니다. 이때 ROI 음성 학습 분포가 처음 YOLO 기준이었다는 차이를 기록합니다. evaluate는 이미 저장된 가중치 평가만 합니다.\n',
           'yolo_roi로 학습한 폴더를 all로 전환하면 학습 후보 구성이 달라져 거부됩니다. 그 경우 compare_detr로 비교하거나 새 OUTPUT 폴더를 사용합니다.\n',
           '정답 bbox를 제공한 ROI 결과는 oracle 진단이며 실제 검출 bbox를 쓴 PQ와 구분합니다. 테스트셋 정답이나 외부 MAGFiLO/SWEFil을 학습하지 않습니다.\n',
           '처음에는 아래 최초 셀을, 세션 종료 후에는 최초 셀 또는 재개 셀 하나를 실행합니다. 동시에 실행하지 마세요. 실제 T4 시간·메모리·새 PQ는 아직 측정하지 않았습니다.\n']
    notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python'),colab=dict(name=notebook_path.name)),
                  cells=[dict(id='intro',cell_type='markdown',metadata={},source=intro),
                         dict(id='first-run',cell_type='code',metadata={},execution_count=None,outputs=[],source=cell.splitlines(True)),
                         dict(id='resume-info',cell_type='markdown',metadata={},source=['## 새 세션 재개\n','처음 실험한 Drive 폴더를 OUTPUT에 지정합니다. PHASE/ROI_UNTIL/DETR_UNTIL만 실행 범위로 변경합니다.\n']),
                         dict(id='resume',cell_type='code',metadata={},execution_count=None,outputs=[],source=resume.splitlines(True))])
    notebook_path.write_text(json.dumps(notebook,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return dict(copy_cell=str(first_path),resume_cell=str(resume_path),notebook=str(notebook_path),
                source_bundle_sha256=stamp,source_count=len(sources),source_sha256=manifest,
                source_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                cell_bytes=len(cell.encode()),resume_bytes=len(resume.encode()),default_roi_epochs=30,default_detr_epochs=30,
                default_total_training_epochs=60,default_phase='all',default_roi_encoder='resnet34',frozen_split=[616,91],
                T4_training_tested_here=False,new_model_PQ_measured=False)


if __name__=='__main__':print(json.dumps(build(),ensure_ascii=False,indent=2))
