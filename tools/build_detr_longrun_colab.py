"""Build the immutable A100 DETR60 -> at most DETR150 Colab delivery."""
import base64
import hashlib
import json
from pathlib import Path
import shutil
import textwrap
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
MODULES = ('detr_longrun.py', 'detr_longrun_eval.py', 'bbox_probe_pipeline.py',
           'bbox_detr_continuation.py', 'roi_filament_core.py', 'roi_filament_pipeline.py',
           'filament_detr.py', 'native_instance_data.py', 'native_instance_training.py',
           'native_instance_model.py')
BOOTSTRAP_NAME = 'detr_longrun_colab_bootstrap.py'

BOOTSTRAP = r'''"""Reconstruct local code from immutable Drive files and launch DETR only."""
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

EXPECTED_SOURCE_CONFIG = 'db31f83bd79f82240b07ef6b10e3da82ec12b7e5828a2672dcb8864d613cbc1e'
EXPECTED_SOURCE_MANIFEST = 'da5d0a7f6b036b2dc9aca05d5577d9e08cd4be096454fbfe6e65db5fbbe142e1'
EXPECTED_SOURCE_CHECKPOINT = '97155a443d0c161d4306625dcf5eecb68d0c5cf50dfa69b11c645c548fb92819'
EXPECTED_SOURCE_IDENTITY = 'a18cf39e98c9afc38a3e4229cc53612360806069e8a69d1a4c7b3446d104d557'
EXPECTED_ROI_CHECKPOINT = '4d1a97c0baf3b7d820754de141d68f1f56cffb0f5fb1b06be6f1b93a9a90ad7f'

def interrupted(signum, frame):
    raise KeyboardInterrupt('Interrupted by signal '+str(signum))

def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()

def distinct_output(output, source):
    output=Path(output).resolve();source=Path(source).resolve()
    if output==source or output in source.parents or source in output.parents:
        raise ValueError('새 OUTPUT은 기존 실험과 겹치지 않는 별도 폴더여야 합니다.')

def restore(root, work_root=Path('/content/solar_detr_longrun_code')):
    root=Path(root);pointer=json.loads((root/'current_code.json').read_text())
    if not re.fullmatch('[0-9a-f]{16}',pointer['version']):raise ValueError('Invalid saved code version')
    durable=root/'code'/pointer['version'];manifest_path=durable/'source_manifest.json'
    if digest(manifest_path)!=pointer['manifest_sha256']:raise ValueError('Drive 코드 목록이 변경되었습니다.')
    config_path=root/'experiment_config.json'
    if digest(config_path)!=pointer['config_sha256']:
        raise ValueError('저장된 설정이 변경되었습니다. RUN_UNTIL_EPOCH만 실행 범위로 바꾸세요.')
    cfg=json.loads(config_path.read_text())
    if Path(cfg['output_root'])!=root:raise ValueError('Saved output folder differs from this experiment')
    distinct_output(root,cfg['source_root'])
    manifest=json.loads(manifest_path.read_text())
    if not manifest or any(Path(name).name!=name for name in manifest):raise ValueError('Invalid source manifest')
    work=Path(work_root)/pointer['version'];work.mkdir(parents=True,exist_ok=True)
    for name,expected in manifest.items():
        source=durable/name
        if not source.is_file() or digest(source)!=expected:raise ValueError('Drive 코드가 변경되었습니다: '+name)
        content=source.read_bytes();compile(content,name,'exec')
        pending=work/(name+'.pending');pending.write_bytes(content);pending.replace(work/name)
    return cfg,config_path,work

def verify_source(cfg):
    source=Path(cfg['source_root']);pointer_path=source/'current_code.json'
    if not pointer_path.is_file():raise FileNotFoundError('완료된 DETR60 실험 폴더가 필요합니다: '+str(source))
    pointer=json.loads(pointer_path.read_text());version=pointer['version']
    if not re.fullmatch('[0-9a-f]{16}',version):raise ValueError('Invalid source code version')
    config_path=source/'experiment_config.json';manifest_path=source/'code'/version/'source_manifest.json'
    if digest(config_path)!=pointer['config_sha256'] or digest(config_path)!=EXPECTED_SOURCE_CONFIG:
        raise ValueError('기존 DETR60 실험 설정이 검토한 결과와 다릅니다.')
    if digest(manifest_path)!=pointer['manifest_sha256'] or digest(manifest_path)!=EXPECTED_SOURCE_MANIFEST:
        raise ValueError('기존 DETR60 코드 목록이 검토한 결과와 다릅니다.')
    for name,expected in json.loads(manifest_path.read_text()).items():
        if Path(name).name!=name or digest(manifest_path.parent/name)!=expected:
            raise ValueError('기존 실험 코드가 변경되었습니다: '+name)
    identity=json.loads((source/'detr_extension/training_identity.json').read_text())
    if identity['identity']!=EXPECTED_SOURCE_IDENTITY:raise ValueError('검토한 DETR60 학습 identity와 다릅니다.')
    checkpoint=source/'detr_extension/epoch060.pt'
    if digest(checkpoint)!=EXPECTED_SOURCE_CHECKPOINT:raise ValueError('검토한 DETR60 가중치와 다릅니다.')
    if not (source/'detr_extension/last_full.pt').is_file():
        raise FileNotFoundError('optimizer 재개 파일이 필요합니다: '+str(source/'detr_extension/last_full.pt'))
    base=Path(json.loads(config_path.read_text())['base_root']);distinct_output(cfg['output_root'],base)
    original=json.loads((base/'experiment_config.json').read_text())
    if digest(base/'roi/epoch030.pt')!=EXPECTED_ROI_CHECKPOINT:raise ValueError('고정 ROI30 가중치가 다릅니다.')
    for key in ('annotations','train_split','val_split'):
        if not Path(original[key]).is_file():raise FileNotFoundError(key+': '+str(original[key]))
    if not Path(original['images']).is_dir():raise FileNotFoundError('images: '+str(original['images']))
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
assert torch.cuda.is_bf16_supported() and torch.cuda.get_device_capability(0)[0]>=8, 'A100 이상 BF16 지원 GPU가 필요합니다. T4 대신 A100을 선택하세요.'
print('PyTorch:',torch.__version__,'CUDA:',torch.version.cuda,'GPU:',torch.cuda.get_device_name(0),flush=True)
layer=torch.nn.Conv2d(3,3,3,padding=1).cuda()
x=torch.randn(1,3,32,32,device='cuda',requires_grad=True)
with torch.autocast('cuda',dtype=torch.bfloat16): loss=layer(x).float().square().mean()
loss.backward();torch.cuda.synchronize()
print('CUDA BF16 forward/backward 사전 검사 통과',flush=True)
"""
    subprocess.run([sys.executable,'-u','-c',probe,str(work)],env=environment,check=True)

def run(root,run_until):
    cfg,config_path,work=restore(root)
    if not 60<=run_until<=int(cfg['max_epoch']):raise ValueError('누적 실행 목표는60~150범위입니다.')
    verify_source(cfg);ensure_dependencies(work)
    environment=dict(os.environ)
    environment['TORCH_CUDNN_V8_API_DISABLED']='1';environment['PYTHONUNBUFFERED']='1'
    environment['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    check_gpu(environment,work)
    print('DETR 기존60 → 누적 최대150: 추가 학습 최대90epoch. 이번 누적 상한:',run_until,flush=True)
    print('ROI30 고정. ROI/YOLO 추가 학습0epoch. 5-fold 추가 학습 없음.',flush=True)
    print('A100: 입력1280, batch2 × 누적2 = 유효batch4, BF16 본체 + FP32 bbox 헤드.',flush=True)
    print('학습률5e-6/backbone5e-7의 새90epoch cosine 단계; 기존 AdamW 상태는 계승합니다.',flush=True)
    print('먼저60epoch 기준선 재평가 → 매epoch 전체91장 PQ 검증. 고정confidence0.6/0.8/0.9 중 최고PQ로 판단합니다.',flush=True)
    print('confidence0.8의 PQ도 별도 기록. mask0.5, area300, quality0.6, NMS0.5는 고정합니다.',flush=True)
    print('PQ 0.001(0.1퍼센트포인트) 이상 개선 없는10연속epoch면 조기 종료. loss는 기록하되 종료 기준은 PQ입니다.',flush=True)
    print('최고 raw PQ checkpoint와 patience 개선 기준은 따로 저장합니다. 검증 도중 끊기면 그 검증부터 재개합니다.',flush=True)
    print('Drive 재개 폴더:',root,flush=True)
    command=[sys.executable,'-u',str(work/'detr_longrun.py'),'--config',str(config_path),
             '--run-until',str(run_until)]
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                             text=True,bufsize=1,env=environment,start_new_session=True)
    try:
        for line in process.stdout:print(line,end='',flush=True)
        code=process.wait()
        if code:raise RuntimeError('DETR 실행 실패(종료 코드 '+str(code)+'). 위 내부 traceback과 Drive colab_bootstrap.log를 확인하세요.')
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
    parser.add_argument('--run-until',type=int,default=150)
    args=parser.parse_args();result=run(Path(args.root),args.run_until)
    print(json.dumps({key:result.get(key) for key in ('status','completed_epoch','best_epoch','best_pq','archives')},ensure_ascii=False,indent=2),flush=True)
'''

CONTROLS = r'''# 기존 DETR60에서 누적 최대150까지만: 새 학습 최대90epoch; PQ 기준 조기 종료 적용.
RUN_UNTIL_EPOCH = 150       # 80이면 누적80에서 일시정지; 이후150으로 바꾸면 저장상태부터 재개
AUTO_DOWNLOAD = True       # 결과 ZIP만 다운로드; 가중치는 Drive에 보관
'''

FIRST = CONTROLS+r'''
SOURCE_RUN = '/content/drive/MyDrive/filament-segmentation-2026/fixed_split_experiments/bbox_improvement_v1'
OUTPUT = '/content/drive/MyDrive/filament-segmentation-2026/fixed_split_experiments/detr_a100_e150_v1'

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
root=Path(OUTPUT);source_root=Path(SOURCE_RUN)
original_source_config=source_root/'experiment_config.json'
if not original_source_config.is_file():raise FileNotFoundError('완료된 DETR60 실험이 필요합니다: '+str(original_source_config))
original_base=Path(json.loads(original_source_config.read_text())['base_root'])
for protected in (source_root,original_base):
    a=root.resolve();b=protected.resolve()
    if a==b or a in b.parents or b in a.parents:raise ValueError('OUTPUT은 기존 실험과 겹치지 않는 새 폴더여야 합니다.')
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
cfg=dict(source_root=SOURCE_RUN,output_root=str(root),cache_root='/content/solar_detr_longrun_cache',
    device='cuda',max_epoch=150,patience=10,min_delta=.001,
    continuation_lr=5e-6,continuation_backbone_lr=5e-7,continuation_lr_floor=.2,
    continuation_image_size=1280,continuation_batch=2,continuation_accumulate=2,continuation_precision='bf16',
    roi_eval_batch=8,eval_box_thresholds=[.6,.8,.9],eval_mask_threshold=.5,eval_min_area=300,
    eval_min_quality=.6,eval_nms_iou=.5,
    code_dir=str(durable),code_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest())
config_bytes=(json.dumps(cfg,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode()
save_immutable(root/'experiment_config.json',config_bytes)
pointer=dict(version=__VERSION__,manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
             config_sha256=hashlib.sha256(config_bytes).hexdigest())
save_immutable(root/'current_code.json',(json.dumps(pointer,sort_keys=True,indent=2)+'\n').encode())
'''

RESUME = CONTROLS+r'''
# /content가 초기화돼도 Drive의 코드·설정·모델·optimizer·조기종료 상태를 복원합니다.
OUTPUT = '/content/drive/MyDrive/filament-segmentation-2026/fixed_split_experiments/detr_a100_e150_v1'
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
if isinstance(RUN_UNTIL_EPOCH,bool) or not isinstance(RUN_UNTIL_EPOCH,int) or not 60<=RUN_UNTIL_EPOCH<=150:
    raise ValueError('RUN_UNTIL_EPOCH은60~150누적epoch 정수입니다. 추가150을 뜻하지 않습니다.')
command=[sys.executable,'-u',str(durable/'detr_longrun_colab_bootstrap.py'),
         '--root',str(root),'--run-until',str(RUN_UNTIL_EPOCH)]
environment=dict(os.environ);environment['TORCH_CUDNN_V8_API_DISABLED']='1';environment['PYTHONUNBUFFERED']='1'
log_path=root/'colab_bootstrap.log'
print('전체 오류·실행 로그:',log_path)
with log_path.open('a',encoding='utf-8',buffering=1) as log:
    log.write('\n=== '+datetime.now(timezone.utc).isoformat()+' total_until='+str(RUN_UNTIL_EPOCH)+' ===\n')
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
print('실행 상태:',result.get('status'),'완료epoch:',result.get('completed_epoch'),
      '최고epoch:',result.get('best_epoch'),'최고PQ:',result.get('best_pq'))
for archive in result.get('archives',[]):
    print('분석용 ZIP:',archive)
    if AUTO_DOWNLOAD:
        try:files.download(archive)
        except Exception as exc:print('Drive의 위 경로에서 받을 수 있습니다:',exc)
'''

GUIDE = '''# DETR 누적 최대150epoch·PQ 조기 종료 실험

- Colab 런타임을 **A100 GPU**로 설정하고 `SolarFilament_DETR150_A100_CopyCell.py` 전체를 한 셀에 붙여 실행합니다.
- 기존 `bbox_improvement_v1`에서 완료된 **60epoch 가중치와 optimizer 상태**를 계승합니다. 누적 상한150이므로 새 학습은 **최대90epoch**입니다. ROI U-Net30은 고정이며 YOLO·5-fold 학습은 수행하지 않습니다.
- 결과 위치: `/content/drive/MyDrive/filament-segmentation-2026/fixed_split_experiments/detr_a100_e150_v1`. 기존 실험 폴더는 변경하지 않습니다.
- 먼저 DETR60 기준선을 같은 조건으로 재평가하고 **매epoch 전체 validation91장**으로 최종 인스턴스 PQ를 계산합니다. 학습loss도 기록하지만 조기종료는 **검증PQ만** 판단합니다. loss가 내려가도 실제 찾기·분리 성능이 개선되지 않으면 종료합니다.
- 고정평가: 매epoch **동일한 detector confidence0.6/0.8/0.9 중 최고PQ**로 판단하고, confidence0.8의 PQ도 별도 기록합니다. 검출기의 확신도 변화만으로 조기종료하는 것을 줄이기 위한 작은 고정grid입니다. mask threshold0.5, mask area300px, ROI quality0.6, mask NMS0.5는 고정합니다. 학습 도중grid를 확장하지 않습니다.
- **10연속epoch 동안 PQ가0.001 이상 개선되지 않으면 조기 종료**합니다. 0.001은 PQ0.1퍼센트포인트입니다. 작은 개선은 누적해서 기준점과 비교합니다. 따라서 epoch61~70이 모두 개선되지 않으면70에서 멈출 수 있습니다.
- 최고 raw PQ 모델은 개선폭과 상관없이 별도 저장합니다. 조기종료의 유의미한 개선 기준점과 최고모델 기준은 다릅니다. 이후 모델이 기준선보다 나쁘면60epoch 모델이 최적으로 남을 수 있습니다.
- A100 설정: 입력1280, batch2×gradient accumulation2(유효batch4), BF16 본체+FP32 bbox 헤드. 학습률5e-6/backbone5e-7로 **새90epoch cosine 단계**를 시작하고 하한은초기값의0.2입니다. 기존 optimizer moments를 계승하지만 처음부터150epoch를 계획한 학습과 동일하지는 않습니다.
- **세션 종료 뒤** 최초 셀을 다시 실행하거나 `_ResumeCell.py`를 한 셀에 붙여 실행합니다. Drive의 코드·설정·가중치·optimizer·조기종료 상태를 복원합니다. 두 셀을 동시에 실행하지 마세요.
- `RUN_UNTIL_EPOCH=80`이면 누적80에서 일시정지합니다. 이후150으로 바꾸면 이어갑니다. 전체150스케줄·patience는 유지되며 조기종료가 우선입니다. 조기종료 완료 후 재실행해도 학습을 강제로 연장하지 않습니다.
- 저장된 마지막 완료epoch부터 재개하므로 미완료epoch의 batch는 다시 실행할 수 있습니다. 학습이 끝나고 검증 중 끊겼으면 그 검증부터 마친 뒤 다음epoch로 넘어갑니다. GPU 연산의 비트단위 동일성은 보장하지 않습니다.
- 재개 중 바꿀 값은 `RUN_UNTIL_EPOCH`와 `AUTO_DOWNLOAD`뿐입니다. 코드·학습 설정 변경은 기존 재개를 섞지 않도록 차단합니다. 다른 실험을 하려면 새OUTPUT이 필요합니다.
- 실행 오류의 내부 traceback은 셀 출력과 Drive `colab_bootstrap.log`에 함께 남습니다. Colab 기본 torch/torchvision CUDA 조합을 유지하면서 필요한 라이브러리를 설치합니다.
- 최적·최신 모델과 전체 재개 상태는 Drive에 보관합니다. 분석용 ZIP에는 수치history와 최고/최신 검증결과만 넣고 원본사진·모델가중치는 넣지 않습니다. ZIP은28MiB 이하로 나눕니다.
- validation은 기존 고정분할입니다. 이 실험의 최고검증PQ는 독립된 test/LB점수가 아니며 반복선택의 영향을 받습니다. 새 실제A100학습·소요시간·새PQ는 이 코드 생성 환경에서 측정하지 않았습니다.
'''


def build_sources():
    sources={name:(ROOT/'tools'/name).read_text(encoding='utf-8') for name in MODULES}
    sources[BOOTSTRAP_NAME]=BOOTSTRAP
    for name,source in sources.items():compile(source,name,'exec')
    return sources


def build(downloads=None):
    sources=build_sources()
    raw=json.dumps(sources,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
    stamp=hashlib.sha256(raw).hexdigest()
    manifest={name:hashlib.sha256(source.encode()).hexdigest() for name,source in sources.items()}
    manifest_bytes=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
    payload='\n'.join(textwrap.wrap(base64.b64encode(zlib.compress(raw,9)).decode(),120))
    cell=FIRST.replace('__PAYLOAD__',payload).replace('__SHA__',repr(stamp)).replace('__FILES__',repr(tuple(sources))).replace('__VERSION__',repr(stamp[:16]))+EXECUTE
    resume=RESUME+EXECUTE
    compile(cell,'detr150-first','exec');compile(resume,'detr150-resume','exec')
    output=ROOT/'notebooks';output.mkdir(parents=True,exist_ok=True)
    stem='SolarFilament_DETR150_A100'
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
        existing_detr_epochs=60,max_cumulative_detr_epochs=150,max_additional_detr_epochs=90,
        added_roi_epochs=0,added_yolo_epochs=0,patience_epochs=10,min_delta_pq=.001,
        validation_images=91,validation_interval_epochs=1,baseline60_recomputed=True,
        monitor_thresholds=dict(detector_grid=[.6,.8,.9],fixed_reference_detector=.8,mask=.5,area=300,quality=.6,nms=.5),
        A100_training_tested_here=False,new_model_PQ_measured=False,
        continuation_image_size=1280,continuation_batch=2,continuation_accumulate=2,
        continuation_precision='bf16',bbox_head_precision='fp32',roi_eval_batch=8)
    downloads=Path(downloads) if downloads is not None else ROOT.parent/'artifacts/downloads'
    downloads.mkdir(parents=True,exist_ok=True)
    files=[first_path,resume_path,notebook_path,guide_path]
    for path in files:shutil.copy2(path,downloads/path.name)
    metadata_path=downloads/(stem+'_metadata.json')
    metadata['downloads']={path.name:str(downloads/path.name) for path in files}
    metadata['package']=str(downloads/(stem+'_Package.zip'))
    metadata_path.write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with zipfile.ZipFile(metadata['package'],'w',zipfile.ZIP_DEFLATED) as archive:
        for path in files:archive.write(path,path.name)
        archive.write(metadata_path,metadata_path.name)
        for name,source in sources.items():archive.writestr('source/'+name,source)
        archive.writestr('source/source_manifest.json',manifest_bytes)
    return metadata


if __name__=='__main__':print(json.dumps(build(),ensure_ascii=False,indent=2))
