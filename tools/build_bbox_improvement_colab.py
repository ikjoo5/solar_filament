"""Durable Colab delivery for frozen-ROI bbox probes and DETR continuation."""
import base64
import hashlib
import json
from pathlib import Path
import textwrap
import zlib

ROOT = Path(__file__).resolve().parents[1]
MODULES = ('bbox_probe_pipeline.py', 'bbox_detr_continuation.py',
           'roi_filament_core.py', 'roi_filament_pipeline.py', 'filament_detr.py',
           'native_instance_data.py', 'native_instance_training.py', 'native_instance_model.py')
BOOTSTRAP_NAME = 'bbox_improvement_colab_bootstrap.py'

BOOTSTRAP = r'''"""Restore immutable code/config, check original run, stream the worker."""
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

EXPECTED_BASE_CONFIG = '6d896724213e73810b0a41220769779576159456925d3ad3fa5acc657b2631a4'
EXPECTED_BASE_MANIFEST = '45cae00116599984add596946a47d8ef1b1ebe979ec57baf8ad7b7c831573715'
EXPECTED_CHECKPOINTS = {
    'detr/epoch030.pt':'4ea4a09244b24b00ac61a9618a11327a31f7156bcf279e7da87a1ea81b6f6565',
    'roi/epoch030.pt':'4d1a97c0baf3b7d820754de141d68f1f56cffb0f5fb1b06be6f1b93a9a90ad7f',
}

def interrupted(signum, frame):
    raise KeyboardInterrupt('Interrupted by signal '+str(signum))

def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()

def restore(root, work_root=Path('/content/solar_bbox_improvement_code')):
    root=Path(root);pointer=json.loads((root/'current_code.json').read_text())
    if not re.fullmatch('[0-9a-f]{16}',pointer['version']):raise ValueError('Invalid saved code version')
    durable=root/'code'/pointer['version'];manifest_path=durable/'source_manifest.json'
    if digest(manifest_path)!=pointer['manifest_sha256']:raise ValueError('Drive 코드 목록이 변경되었습니다.')
    config_path=root/'experiment_config.json'
    if digest(config_path)!=pointer['config_sha256']:
        raise ValueError('저장된 설정이 변경되었습니다. PHASE/RUN_UNTIL_ADDED만 실행 범위로 바꾸세요.')
    cfg=json.loads(config_path.read_text())
    if Path(cfg['output_root'])!=root:raise ValueError('Saved output folder differs from this experiment')
    if Path(cfg['base_root']).resolve()==root.resolve():raise ValueError('새 실험은 기존 실험과 다른 폴더여야 합니다.')
    manifest=json.loads(manifest_path.read_text())
    if not manifest or any(Path(name).name!=name for name in manifest):raise ValueError('Invalid source manifest')
    work=Path(work_root)/pointer['version'];work.mkdir(parents=True,exist_ok=True)
    for name,expected in manifest.items():
        source=durable/name
        if not source.is_file() or digest(source)!=expected:raise ValueError('Drive 코드가 변경되었습니다: '+name)
        content=source.read_bytes();compile(content,name,'exec')
        pending=work/(name+'.pending');pending.write_bytes(content);pending.replace(work/name)
    return cfg,config_path,work

def verify_base(cfg, phase):
    base=Path(cfg['base_root']);pointer_path=base/'current_code.json'
    if not pointer_path.is_file():raise FileNotFoundError('완료된 30에폭 실험 폴더가 필요합니다: '+str(base))
    pointer=json.loads(pointer_path.read_text());version=pointer['version']
    if not re.fullmatch('[0-9a-f]{16}',version):raise ValueError('Invalid base code version')
    config_path=base/'experiment_config.json';manifest_path=base/'code'/version/'source_manifest.json'
    if digest(config_path)!=pointer['config_sha256'] or digest(config_path)!=EXPECTED_BASE_CONFIG:
        raise ValueError('기존 30에폭 실험 설정이 검토한 결과와 다릅니다.')
    if digest(manifest_path)!=pointer['manifest_sha256'] or digest(manifest_path)!=EXPECTED_BASE_MANIFEST:
        raise ValueError('기존 30에폭 실험 코드 목록이 달라졌습니다.')
    for name,expected in json.loads(manifest_path.read_text()).items():
        if Path(name).name!=name or digest(manifest_path.parent/name)!=expected:
            raise ValueError('기존 실험 코드가 변경되었습니다: '+name)
    original=json.loads(config_path.read_text())
    for key in ('annotations','train_split','val_split','yolo'):
        if not Path(original[key]).is_file():raise FileNotFoundError(key+': '+str(original[key]))
    if not Path(original['images']).is_dir():raise FileNotFoundError('images: '+str(original['images']))
    for relative,expected in EXPECTED_CHECKPOINTS.items():
        path=base/relative
        if not path.is_file():raise FileNotFoundError('기존 학습 가중치가 필요합니다: '+str(path))
        if digest(path)!=expected:raise ValueError('검토한 30에폭 가중치와 다릅니다: '+str(path))
    if phase in ('all','extend') and not (base/'detr/last_full.pt').is_file():
        raise FileNotFoundError('optimizer를 포함한 DETR 재개 파일이 필요합니다: '+str(base/'detr/last_full.pt'))
    return original

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

def check_gpu(environment,work):
    probe="""import sys,torch,torchvision
sys.path.insert(0,sys.argv[1])
from native_instance_training import configure_cuda
configure_cuda()
assert torch.cuda.is_available(), 'Colab 런타임을 A100 GPU로 선택하세요.'
assert torch.cuda.is_bf16_supported() and torch.cuda.get_device_capability(0)[0]>=8, '추가 학습은 A100 이상 BF16 지원 GPU가 필요합니다. T4 대신 A100을 선택하세요.'
print('PyTorch:',torch.__version__,'CUDA:',torch.version.cuda,'GPU:',torch.cuda.get_device_name(0),flush=True)
layer=torch.nn.Conv2d(3,3,3,padding=1).cuda()
x=torch.randn(1,3,32,32,device='cuda',requires_grad=True)
with torch.autocast('cuda',dtype=torch.bfloat16): loss=layer(x).float().square().mean()
loss.backward();torch.cuda.synchronize()
print('CUDA BF16 forward/backward 사전 검사 통과',flush=True)
"""
    subprocess.run([sys.executable,'-u','-c',probe,str(work)],env=environment,check=True)

def run(root,phase,run_until_added):
    cfg,config_path,work=restore(root)
    if phase not in ('all','probes','extend'):raise ValueError('Unknown phase')
    if not 1<=run_until_added<=int(cfg['added_epochs']):raise ValueError('추가 학습 목표는1~30범위입니다.')
    verify_base(cfg,phase);ensure_dependencies(work)
    environment=dict(os.environ)
    environment['TORCH_CUDNN_V8_API_DISABLED']='1';environment['PYTHONUNBUFFERED']='1'
    environment['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    check_gpu(environment,work)
    print('실행 단계:',phase,'| 고정 ROI:30epoch | 고정 YOLO:기존 best.pt',flush=True)
    if phase in ('all','extend'):
        print('DETR 기존30 + 이번 추가 목표',run_until_added,'= 누적',30+run_until_added,
              '(추가 전체 스케줄30, 누적 최종60)',flush=True)
        print('추가 학습은 DETR만 합니다. ROI/YOLO 추가 학습은0epoch입니다.',flush=True)
        print('기존 optimizer 상태를 계승하고 추가30epoch용 학습률 스케줄을 새로 시작합니다.',flush=True)
        print('A100 추가 학습: 입력1280, batch2 × 누적2 = 유효batch4, BF16 본체 + FP32 bbox 헤드. 기존30의1280추론 대조군도 평가합니다.',flush=True)
    else:print('probes는 저장된 모델로 bbox 추론·평가만 합니다. 추가 학습0epoch입니다.',flush=True)
    if phase in ('all','probes'):
        print('타일 비교: 원본2048에서1024crop/stride768로9tiles + 전체사진, 검출기별 수행합니다.',flush=True)
        print('all은 타일 비교 결과ZIP을 먼저 저장한 뒤 DETR 추가 학습으로 넘어갑니다.',flush=True)
    print('세션 종료 후 같은 폴더로 재실행하면 저장된 후보/완료epoch를 재사용합니다:',root,flush=True)
    command=[sys.executable,'-u',str(work/'bbox_probe_pipeline.py'),'--config',str(config_path),
             '--phase',phase,'--run-until-added',str(run_until_added)]
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
    return json.loads((Path(root)/'latest_results.json').read_text())

if __name__=='__main__':
    signal.signal(signal.SIGTERM,interrupted)
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--phase',choices=('all','probes','extend'),default='all')
    parser.add_argument('--run-until-added',type=int,default=30)
    args=parser.parse_args();result=run(Path(args.root),args.phase,args.run_until_added)
    print(json.dumps({key:result.get(key) for key in ('status','phase','archives')},ensure_ascii=False,indent=2),flush=True)
'''

CONTROLS = r'''# all: 타일 검출 비교 후 DETR만 추가30epoch(기존30→누적60). ROI/YOLO는 고정.
PHASE = 'all'              # all / probes(추론만0epoch) / extend(DETR 추가 학습만)
RUN_UNTIL_ADDED = 30        # 15이면 누적45에서 정지;30으로 바꾸면 누적60까지 재개
AUTO_DOWNLOAD = True
'''

FIRST = CONTROLS+r'''
BASE_RUN = '/content/drive/MyDrive/filament-segmentation-2026/fixed_split_experiments/detection_roi_detr_v1'
OUTPUT = '/content/drive/MyDrive/filament-segmentation-2026/fixed_split_experiments/bbox_improvement_v1'

from google.colab import drive, files
from pathlib import Path
from datetime import datetime, timezone
import base64,hashlib,json,os,signal,subprocess,sys,zlib
drive.mount('/content/drive')
PAYLOAD = """__PAYLOAD__"""
raw=zlib.decompress(base64.b64decode(PAYLOAD))
if hashlib.sha256(raw).hexdigest()!=__SHA__:raise ValueError('실행 셀 전체를 다시 복사하세요: 코드 checksum 불일치')
sources=json.loads(raw)
if set(sources)!=set(__FILES__):raise ValueError('Invalid source bundle')
root=Path(OUTPUT)
if root.resolve()==Path(BASE_RUN).resolve():raise ValueError('OUTPUT은 기존 실험과 다른 폴더여야 합니다.')
durable=root/'code'/__VERSION__;durable.mkdir(parents=True,exist_ok=True)

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
cfg=dict(base_root=BASE_RUN,output_root=str(root),cache_root='/content/solar_bbox_improvement_cache',
    device='cuda',tile_size=1024,tile_stride=768,tile_nms=[.5,.7],tile_max_det=200,
    added_epochs=30,continuation_lr=1e-5,continuation_backbone_lr=1e-6,
    continuation_image_size=1280,continuation_batch=2,continuation_accumulate=2,continuation_precision='bf16',
    code_dir=str(durable),code_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest())
config_bytes=(json.dumps(cfg,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode()
save_immutable(root/'experiment_config.json',config_bytes)
pointer=dict(version=__VERSION__,manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
             config_sha256=hashlib.sha256(config_bytes).hexdigest())
save_immutable(root/'current_code.json',(json.dumps(pointer,sort_keys=True,indent=2)+'\n').encode())
'''

RESUME = CONTROLS+r'''
# /content가 초기화돼도 Drive의 코드·설정·가중치로 복원합니다.
OUTPUT = '/content/drive/MyDrive/filament-segmentation-2026/fixed_split_experiments/bbox_improvement_v1'
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
'''

EXECUTE = r'''
command=[sys.executable,'-u',str(durable/'bbox_improvement_colab_bootstrap.py'),
         '--root',str(root),'--phase',PHASE,'--run-until-added',str(RUN_UNTIL_ADDED)]
environment=dict(os.environ);environment['TORCH_CUDNN_V8_API_DISABLED']='1';environment['PYTHONUNBUFFERED']='1'
log_path=root/'colab_bootstrap.log'
print('전체 오류·실행 로그:',log_path)
with log_path.open('a',encoding='utf-8',buffering=1) as log:
    log.write('\n=== '+datetime.now(timezone.utc).isoformat()+' '+PHASE+' added'+str(RUN_UNTIL_ADDED)+' ===\n')
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
result=json.loads((root/'latest_results.json').read_text())
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
    compile(cell,'bbox-first','exec');compile(resume,'bbox-resume','exec')
    output=ROOT/'notebooks';output.mkdir(parents=True,exist_ok=True)
    first_path=output/'SolarFilament_BBoxImprove_A100_CopyCell.py'
    resume_path=output/'SolarFilament_BBoxImprove_A100_ResumeCell.py'
    notebook_path=output/'SolarFilament_BBoxImprove_A100.ipynb'
    first_path.write_text(cell,encoding='utf-8');resume_path.write_text(resume,encoding='utf-8')
    intro=['# A100: bbox 개선 — 고해상도 타일 검출 + DETR 추가30epoch\n',
        '완료한 detection_roi_detr_v1의 YOLO, DETR30, ROI U-Net30을 사용합니다. ROI U-Net과 YOLO는 추가 학습하지 않습니다.\n',
        '기본 all: 기존 모델의 전체사진/타일 bbox 비교를 먼저 저장하고, DETR만 기존30에서 추가30epoch(누적60) 학습합니다. 새 학습량은 DETR30epoch 하나입니다.\n',
        'probes는 학습0epoch로 검출 비교만 실행합니다. extend는 DETR 추가 학습과 같은 ROI를 통한 검증을 실행합니다.\n',
        '추가 DETR 학습은 A100에서1280입력, batch2×gradient accumulation2(유효4), BF16 본체와 FP32 bbox 헤드를 사용합니다. 기존 DETR30을1280으로 추론하는 대조군과 추가45/60에폭의1280결과를 비교합니다. 기존 DETR30의800타일 추론은 별도 실험입니다.\n',
        '전체사진에1024crop/stride768의9tiles를 더해 작은 필라멘트를 크게 봅니다. 원래2048좌표로 복원하고 bbox NMS0.5/0.7을 비교합니다. 여러 추론 뷰를 합치는 TTA와는 별개의 타일 실험입니다.\n',
        'RUN_UNTIL_ADDED=15이면 누적45에서 정지하고30으로 재실행하면 누적60까지 이어갑니다. 추가 전체 스케줄은30으로 고정하며 완료epoch의 optimizer/RNG 상태를 복원합니다.\n',
        'DETR 추가 단계는 기존 optimizer 상태를 계승하고 명시적인 추가30epoch용 학습률 스케줄을 시작합니다. 처음부터60epoch로 학습한 단일 스케줄과 같지는 않습니다.\n',
        '기존 train616/validation91 분할과 ROI checkpoint를 고정합니다. 결과는 bbox 위치·재현율과 최종 PQ·FP·FN을 함께 기록합니다. 정답 bbox oracle을 실제 추론 점수로 사용하지 않습니다.\n',
        'Drive의 별도 bbox_improvement_v1 폴더에 코드·설정·새 checkpoint·결과가 저장됩니다. 기존 실험 가중치는 복사하지 않고 참조합니다.\n',
        '세션 종료 뒤에는 최초 셀 또는 아래 재개 셀 하나를 실행합니다. 동시에 실행하지 마세요. 이 번들의 실제A100시간·메모리·새PQ는 여기서 측정하지 않았습니다.\n']
    notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python'),colab=dict(name=notebook_path.name)),
        cells=[dict(id='intro',cell_type='markdown',metadata={},source=intro),
               dict(id='first-run',cell_type='code',metadata={},execution_count=None,outputs=[],source=cell.splitlines(True)),
               dict(id='resume-info',cell_type='markdown',metadata={},source=['## 새 세션 재개\n','처음 실험한 Drive 폴더를 OUTPUT에 지정하고 PHASE/RUN_UNTIL_ADDED만 실행 범위로 변경합니다.\n']),
               dict(id='resume',cell_type='code',metadata={},execution_count=None,outputs=[],source=resume.splitlines(True))])
    notebook_path.write_text(json.dumps(notebook,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return dict(copy_cell=str(first_path),resume_cell=str(resume_path),notebook=str(notebook_path),
        source_bundle_sha256=stamp,source_count=len(sources),source_sha256=manifest,
        source_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),cell_bytes=len(cell.encode()),resume_bytes=len(resume.encode()),
        default_phase='all',default_added_detr_epochs=30,existing_detr_epochs=30,final_cumulative_detr_epochs=60,
        added_roi_epochs=0,added_yolo_epochs=0,default_total_additional_training_epochs=30,
        tile_size=1024,tile_stride=768,tiles_per_detector_image=9,frozen_split=[616,91],
        A100_training_tested_here=False,new_model_PQ_measured=False,default_continuation_image_size=1280,
        default_continuation_batch=2,default_continuation_accumulate=2,default_continuation_precision='bf16',bbox_head_precision='fp32')


if __name__=='__main__':print(json.dumps(build(),ensure_ascii=False,indent=2))
