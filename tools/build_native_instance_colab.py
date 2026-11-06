"""Self-contained, one-model native U-Net Colab experiment and durable resume."""
import base64
import hashlib
import json
from pathlib import Path
import textwrap
import zlib

ROOT = Path(__file__).resolve().parents[1]
MODULES = ('native_instance_model.py', 'native_instance_data.py', 'native_instance_training.py')
BOOTSTRAP_NAME = 'native_instance_colab_bootstrap.py'

BOOTSTRAP = r'''"""Restore immutable code/config from Drive, then run a streamed worker."""
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

def restore(root, work_root=Path('/content/solar_native_instance_code')):
    root=Path(root)
    pointer=json.loads((root/'current_code.json').read_text())
    if not re.fullmatch('[0-9a-f]{16}',pointer['version']):
        raise ValueError('Invalid saved code version')
    durable=root/'code'/pointer['version']
    manifest_path=durable/'source_manifest.json'
    if digest(manifest_path)!=pointer['manifest_sha256']:
        raise ValueError('Drive의 코드 목록이 변경되었습니다.')
    config_path=root/'experiment_config.json'
    if digest(config_path)!=pointer['config_sha256']:
        raise ValueError('저장된 학습 설정이 변경되었습니다. 실행 범위 RUN_UNTIL/PHASE만 바꾸세요.')
    cfg=json.loads(config_path.read_text())
    if Path(cfg['output_root'])!=root:
        raise ValueError('Saved output folder differs from this experiment')
    manifest=json.loads(manifest_path.read_text())
    if not manifest or any(Path(name).name!=name for name in manifest):
        raise ValueError('Invalid source manifest')
    work=Path(work_root)/pointer['version'];work.mkdir(parents=True,exist_ok=True)
    for name,expected in manifest.items():
        source=durable/name
        if not source.is_file() or digest(source)!=expected:
            raise ValueError('Drive 코드가 변경되었습니다: '+name)
        content=source.read_bytes()
        compile(content,name,'exec')
        pending=work/(name+'.pending');pending.write_bytes(content);pending.replace(work/name)
    return cfg,config_path,work

def check_paths(cfg):
    for key in ('annotations','train_split','val_split'):
        if not Path(cfg[key]).is_file():raise FileNotFoundError(key+': '+str(cfg[key]))
    if not Path(cfg['images']).is_dir():raise FileNotFoundError('images: '+str(cfg['images']))

def ensure_dependencies(work):
    versions={}
    for package in ('torch','torchvision'):
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError('Colab 기본 GPU PyTorch 런타임이 필요합니다: '+package) from exc
    required=[]
    for package,expected in [('segmentation-models-pytorch','0.5.0'),('timm','1.0.20'),('pycocotools','2.0.11')]:
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
groups_list=(1,3) if 'convnext' in sys.argv[2] else (1,)
for groups in groups_list:
    layer=torch.nn.Conv2d(3,3,3,padding=1,groups=groups).cuda()
    x=torch.randn(1,3,32,32,device='cuda',requires_grad=True)
    with torch.autocast('cuda',dtype=torch.float16): loss=layer(x).float().square().mean()
    loss.backward();torch.cuda.synchronize()
print('CUDA FP16 forward/backward 사전 검사 통과',flush=True)
"""
    subprocess.run([sys.executable,'-u','-c',probe,str(work),cfg['encoder']],env=environment,check=True)

def run(root, phase, run_until):
    cfg,config_path,work=restore(root)
    if phase not in ('all','train','evaluate'):raise ValueError('Unknown phase')
    if not 1<=run_until<=int(cfg['epochs']):raise ValueError('RUN_UNTIL은 1~고정 최종 epoch 범위여야 합니다.')
    check_paths(cfg);ensure_dependencies(work)
    environment=dict(os.environ)
    environment['TORCH_CUDNN_V8_API_DISABLED']='1'
    environment['PYTHONUNBUFFERED']='1'
    environment['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    check_gpu(environment,work,cfg)
    command=[sys.executable,'-u',str(work/'native_instance_training.py'),'--config',str(config_path),
             '--phase',phase,'--run-until',str(run_until)]
    print('한 모델만 실행:',cfg['encoder'],cfg['head_mode'],'| 이번 epoch:',run_until,'/ 고정 최종:',cfg['epochs'],flush=True)
    print('한 epoch: ',cfg['steps_per_epoch'],'개 mini-batch ×',cfg['batch'],'개 crop; 누적',cfg['accumulate'],flush=True)
    print('새 모델은 ImageNet encoder에서 시작합니다. 기존 YOLO/best84 가중치를 요구하지 않습니다.',flush=True)
    print('고정 train616/validation91, 실제 데이터만 사용. Drive 폴더:',root,flush=True)
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
    finally:
        process.stdout.close()
    return json.loads((Path(root)/'result.json').read_text())

if __name__=='__main__':
    signal.signal(signal.SIGTERM,interrupted)
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--phase',choices=('all','train','evaluate'),default='all')
    parser.add_argument('--run-until',type=int,default=30)
    args=parser.parse_args()
    result=run(Path(args.root),args.phase,args.run_until)
    print(json.dumps({key:result.get(key) for key in ('status','epochs_completed','fixed_final_epoch','archives')},ensure_ascii=False,indent=2),flush=True)
'''

CONTROLS = r'''# 한 번에 선택한 U-Net 한 모델만 학습합니다. 기본 최종 목표는 실제30epoch입니다.
RUN_UNTIL = 30       # 중간 정지는10/20, 이후30으로 높이면 동일 스케줄에서 이어갑니다.
PHASE = 'all'        # all: 학습+평가 / train: 학습만 / evaluate: 저장된 같은epoch 평가만
AUTO_DOWNLOAD = True
'''

FIRST = CONTROLS + r'''
ENCODER = 'resnet34' # resnet18 / resnet34 / tu-convnext_tiny 중 하나
HEAD_MODE = 'instance' # instance: 전경+경계+거리 / semantic: 전경 loss만
FINAL_EPOCHS = 30    # 최초실행 뒤 고정. 여러encoder를 자동으로 연달아 학습하지 않습니다.
CROP = 768
BATCH = 2
ACCUMULATE = 2
STEPS_PER_EPOCH = 256 # mini-batch 수, epoch당512 crop, 기본 optimizer update시도128회
LEARNING_RATE = 1e-4
BASE = '/content/drive/MyDrive/filament-segmentation-2026'
SPLIT_RUN = '/content/drive/MyDrive/SolarFilament_LimbOnly_YOLO26_mixed_FullRes'
OUTPUT = BASE + '/fixed_split_experiments/native_instance_unet_v1/' + ENCODER + '_' + HEAD_MODE

from google.colab import drive, files
from pathlib import Path
from datetime import datetime, timezone
import base64,hashlib,json,os,signal,subprocess,sys,zlib
drive.mount('/content/drive')
if ENCODER not in ('resnet18','resnet34','tu-convnext_tiny') or HEAD_MODE not in ('semantic','instance'):
    raise ValueError('ENCODER/HEAD_MODE 값을 확인하세요.')
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
data=Path(BASE)/'MAGFiLO_1.0_Kaggle_2026/train';split=Path(SPLIT_RUN)
cfg=dict(images=str(data/'train_images'),annotations=str(data/'MAGFiLO_1.0_Annotations_kaggle2026_train.json'),
    train_split=str(split/'train_split.csv'),val_split=str(split/'val_split.csv'),
    output_root=str(root),cache_root='/content/solar_native_instance_cache/'+ENCODER+'_'+HEAD_MODE,
    encoder=ENCODER,head_mode=HEAD_MODE,pretrained=True,epochs=FINAL_EPOCHS,
    crop=CROP,batch=BATCH,accumulate=ACCUMULATE,steps_per_epoch=STEPS_PER_EPOCH,
    lr=LEARNING_RATE,seed=42,device='cuda',amp=True,overlap=.25,threshold=.5,min_area=300,
    tta=False,preprocessing='limb',strict=True,eval_min_areas=[150,300],input_mean=.449,input_std=.226,
    code_dir=str(durable),code_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest())
config_bytes=(json.dumps(cfg,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode()
save_immutable(root/'experiment_config.json',config_bytes)
pointer=dict(version=__VERSION__,manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
             config_sha256=hashlib.sha256(config_bytes).hexdigest())
save_immutable(root/'current_code.json',(json.dumps(pointer,sort_keys=True,indent=2)+'\n').encode())
'''

RESUME = CONTROLS + r'''
# 완전히 초기화된 Colab에서도 Drive의 코드·설정을 불러옵니다.
OUTPUT = '/content/drive/MyDrive/filament-segmentation-2026/fixed_split_experiments/native_instance_unet_v1/resnet34_instance'
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
command=[sys.executable,'-u',str(durable/'native_instance_colab_bootstrap.py'),
         '--root',str(root),'--phase',PHASE,'--run-until',str(RUN_UNTIL)]
environment=dict(os.environ);environment['TORCH_CUDNN_V8_API_DISABLED']='1';environment['PYTHONUNBUFFERED']='1'
log_path=root/'colab_bootstrap.log'
print('전체 오류·실행 로그:',log_path)
with log_path.open('a',encoding='utf-8',buffering=1) as log:
    log.write('\n=== '+datetime.now(timezone.utc).isoformat()+' '+PHASE+' epoch'+str(RUN_UNTIL)+' ===\n')
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
    finally:
        process.stdout.close()
result=json.loads((root/'result.json').read_text())
print('실행 상태:',result.get('status'),'완료epoch:',result.get('epochs_completed'))
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
    payload='\n'.join(textwrap.wrap(base64.b64encode(zlib.compress(raw,9)).decode(),120))
    cell=FIRST.replace('__PAYLOAD__',payload).replace('__SHA__',repr(stamp)).replace('__FILES__',repr(tuple(sources))).replace('__VERSION__',repr(stamp[:16]))+EXECUTE
    resume=RESUME+EXECUTE
    compile(cell,'native-instance-first','exec');compile(resume,'native-instance-resume','exec')
    output=ROOT/'notebooks';output.mkdir(parents=True,exist_ok=True)
    first_path=output/'SolarFilament_NativeInstanceUNet_T4_CopyCell.py'
    resume_path=output/'SolarFilament_NativeInstanceUNet_T4_ResumeCell.py'
    notebook_path=output/'SolarFilament_NativeInstanceUNet_T4.ipynb'
    first_path.write_text(cell,encoding='utf-8');resume_path.write_text(resume,encoding='utf-8')
    intro=['# T4: ResNet U-Net의 전경·인스턴스 분리 실험\n',
           '선택한 한 모델만 실제30epoch 학습합니다. 원래 train616/validation91을 유지하고 실제 데이터만 사용합니다. 기본은 ResNet34 encoder, 전경·경계·거리3개 출력입니다.\n',
           '기존 YOLO/best84 대신 ImageNet encoder 가중치에서 시작하므로 첫 실행에는 가중치를 내려받습니다. ENCODER/HEAD_MODE를 바꾸면 별도 OUTPUT 폴더를 사용합니다. 여러 모델이나5fold를 자동 학습하지 않습니다.\n',
           '한 epoch는 전체 사진1회가 아니라256 mini-batch×2=512개768px crop 예산입니다. 기본30epoch는15,360개crop이며 gradient accumulation은2입니다.\n',
           'RUN_UNTIL=10/20으로 중간 정지 후30으로 재개할 수 있습니다. FINAL_EPOCHS=30은 전체 학습 스케줄이며 숨은100epoch 목표가 없습니다. 중단 시 마지막 완료epoch부터 복원하고 중간batch는 다시 실행합니다.\n',
           '처음에는 아래 최초 실행 셀을, 세션 종료 후에는 최초 셀 또는 재개 셀 하나를 실행하세요. 동시에 실행하지 마세요. 재개 셀 OUTPUT은 처음 선택한 모델 폴더와 같아야 합니다.\n',
           '이 노트북은 실행 코드입니다. 새 모델의 실제 T4 학습 시간·메모리·대회 PQ는 아직 측정하지 않았습니다.\n']
    notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python'),colab=dict(name=notebook_path.name)),
                  cells=[dict(id='intro',cell_type='markdown',metadata={},source=intro),
                         dict(id='first-run',cell_type='code',metadata={},execution_count=None,outputs=[],source=cell.splitlines(True)),
                         dict(id='resume-info',cell_type='markdown',metadata={},source=['## 새 세션 재개\n','처음 학습한 폴더를 OUTPUT에 지정합니다. 기본 ResNet34 instance를 바꾸지 않았다면 그대로 실행하세요.\n']),
                         dict(id='resume',cell_type='code',metadata={},execution_count=None,outputs=[],source=resume.splitlines(True))])
    notebook_path.write_text(json.dumps(notebook,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return dict(copy_cell=str(first_path),resume_cell=str(resume_path),notebook=str(notebook_path),
                source_bundle_sha256=stamp,source_count=len(sources),source_sha256={n:hashlib.sha256(s.encode()).hexdigest() for n,s in sources.items()},
                cell_bytes=len(cell.encode()),resume_bytes=len(resume.encode()),default_epochs=30,models_per_execution=1,
                default_encoder='resnet34',default_head_mode='instance',frozen_split=[616,91],
                T4_training_tested_here=False,new_model_PQ_measured=False)


if __name__=='__main__':print(json.dumps(build(),ensure_ascii=False,indent=2))
