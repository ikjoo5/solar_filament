"""Build a self-contained T4 cell and Drive-only fresh-session resume cell."""
import base64
import hashlib
import json
from pathlib import Path
import textwrap
import zlib

ROOT = Path(__file__).resolve().parents[1]
MODULES = (
    'targeted_synthetic_launcher.py', 'targeted_synthetic_data.py',
    'targeted_synthetic_training.py', 'targeted_synthetic_hard_donors.json',
    'fixed_split_mask_quality.py', 'evaluate_grouped_oof.py',
    'grouped_cv_training.py', 'submission_inference.py',
    'submission_driver.py', 'submission_yolo_recipe.py',
    'submission_native_recipe.py', 'submission_native_process.py',
)
BOOTSTRAP_NAME = 'targeted_synthetic_colab_bootstrap.py'

BOOTSTRAP = r'''"""Restore reviewed code/dependencies from Drive and stream a separate worker."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

def interrupted(signum, frame):
    # The notebook stops this bootstrap process group. Raising here ensures
    # its separately grouped training worker is also stopped before exiting.
    raise KeyboardInterrupt('Interrupted by signal '+str(signum))

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def restore(root, work_root=Path('/content/solar_targeted_synthetic_v3')):
    root = Path(root)
    pointer = json.loads((root/'current_code.json').read_text())
    durable = root/'code'/pointer['version']
    manifest_path = durable/'source_manifest.json'
    if digest(manifest_path) != pointer['manifest_sha256']:
        raise ValueError('Drive 코드 목록이 변경되었습니다. 새 실험 폴더를 사용하세요.')
    manifest = json.loads(manifest_path.read_text())
    if not manifest or any(Path(name).name != name for name in manifest):
        raise ValueError('Invalid source manifest names')
    config_path = root/'experiment_config.json'
    if digest(config_path) != pointer['config_sha256']:
        raise ValueError('저장된 학습 설정이 변경되었습니다. RUN_UNTIL/PHASE/ARMS만 바꿀 수 있습니다.')
    cfg = json.loads(config_path.read_text())
    if Path(cfg['output_root']) != root:
        raise ValueError('Saved output root differs from the resume folder')
    work = Path(work_root)/pointer['version']
    work.mkdir(parents=True, exist_ok=True)
    for name, expected in manifest.items():
        source = durable/name
        if not source.is_file() or digest(source) != expected:
            raise ValueError('Drive에 저장한 코드가 달라졌습니다: '+name)
        content = source.read_bytes()
        if name.endswith('.py'):
            compile(content, name, 'exec')
        elif name.endswith('.json'):
            json.loads(content)
        (work/name).write_bytes(content)
    return cfg, config_path, work

def ensure_dependencies(work):
    # Keep the runtime's existing torch/torchvision pair, including its CUDA tag.
    torch_versions = {}
    for package in ('torch', 'torchvision'):
        try:
            torch_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError('Colab 기본 PyTorch GPU 런타임이 필요합니다: '+package) from exc
    pinned = [('ultralytics', '8.4.182'), ('timm', '1.0.20'),
              ('segmentation-models-pytorch', '0.5.0'), ('pycocotools', '2.0.11')]
    required = []
    for package, expected in pinned:
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        if actual != expected:
            required.append(package+'=='+expected)
    for module, package in [('numpy', 'numpy'), ('cv2', 'opencv-python-headless'), ('scipy', 'scipy')]:
        if importlib.util.find_spec(module) is None:
            required.append(package)
    if required:
        constraints = Path(work)/'keep_colab_torch.txt'
        constraints.write_text(''.join(k+'=='+v+'\n' for k,v in torch_versions.items()))
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                        '--upgrade-strategy', 'only-if-needed', '-c', str(constraints), *required], check=True)
    for package, expected in torch_versions.items():
        if importlib.metadata.version(package) != expected:
            raise RuntimeError('Colab PyTorch pair changed during dependency setup')

def check_paths(cfg, phase):
    keys = ['annotations', 'train_split', 'val_split', 'yolo']
    directories = ['images']
    if phase in ('all','prepare','train'):
        keys += ['synthetic_annotations','synthetic_split','hard_donors']
        directories.append('synthetic_images')
    if phase != 'prepare' and cfg.get('full_ensemble'):
        keys.append('native')
    for key in keys:
        if not Path(cfg[key]).is_file():
            raise FileNotFoundError(key+': '+str(cfg[key]))
    for key in directories:
        if not Path(cfg[key]).is_dir():
            raise FileNotFoundError(key+': '+str(cfg[key]))

def check_gpu(environment, work, cfg):
    probe = """import sys,torch,torchvision
sys.path.insert(0,sys.argv[1])
from grouped_cv_training import configure_cuda
configure_cuda()
torch.backends.cudnn.enabled = bool(int(sys.argv[2]))
assert torch.cuda.is_available(), 'Colab 런타임 유형을 T4 GPU로 선택하세요.'
print('PyTorch:',torch.__version__,'CUDA:',torch.version.cuda,'GPU:',torch.cuda.get_device_name(0),flush=True)
for groups in (1,3):
    layer=torch.nn.Conv2d(3,3,3,padding=1,groups=groups).cuda()
    x=torch.randn(1,3,32,32,device='cuda',requires_grad=True)
    with torch.autocast('cuda',dtype=torch.float16): loss=layer(x).float().square().mean()
    loss.backward();torch.cuda.synchronize()
print('CUDA FP16 일반/depthwise forward/backward 검사 통과; depthwise cuDNN 우회',flush=True)
print('일반 convolution cuDNN:',torch.backends.cudnn.enabled,flush=True)
"""
    subprocess.run([sys.executable, '-u', '-c', probe, str(work), str(int(cfg.get('cudnn_enabled',True)))],
                   env=environment, check=True)

def run(root, phase, run_until, arms):
    cfg, config_path, work = restore(root)
    if not 1 <= run_until <= int(cfg['epochs']):
        raise ValueError('RUN_UNTIL은 1~저장된 최종 epoch 범위여야 합니다.')
    if not arms or len(arms) != len(set(arms)) or set(arms)-{'control_v2', 'targeted_v3'}:
        raise ValueError('ARMS는 control_v2, targeted_v3 중 중복 없이 지정하세요.')
    check_paths(cfg,phase)
    ensure_dependencies(work)
    environment = dict(os.environ)
    environment['TORCH_CUDNN_V8_API_DISABLED'] = '1'
    environment['PYTHONUNBUFFERED'] = '1'
    environment['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    if phase != 'prepare':
        check_gpu(environment,work,cfg)
    command = [sys.executable, '-u', str(work/'targeted_synthetic_launcher.py'),
               '--config', str(config_path), '--phase', phase,
               '--run-until', str(run_until), '--arms', ','.join(arms),
               '--result', str(Path(root)/'result.json')]
    print('작업:',phase,'선택한 학습군:',','.join(arms),'이번 목표:',run_until,'/ 고정 최종:',cfg['epochs'],flush=True)
    print('기본 설정은 각 군 30epoch 추가 학습, 두 군 합계60epoch입니다. 학습은 5-fold가 아닙니다.',flush=True)
    print('Drive 재개 폴더:',root,flush=True)
    print('데이터 임시 캐시:',cfg['cache_root'],'(세션 종료 시 원본으로 재생성)',flush=True)
    log_path = Path(root)/('colab_'+phase+'.log')
    with log_path.open('a',encoding='utf-8',buffering=1) as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, env=environment, start_new_session=True)
        try:
            for line in process.stdout:
                print(line,end='',flush=True);log.write(line)
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)
        except BaseException:
            try: os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait()
            raise
    return json.loads((Path(root)/'result.json').read_text())

if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--phase',choices=('all','prepare','train','evaluate'),default='all')
    parser.add_argument('--run-until',type=int,default=30)
    parser.add_argument('--arms',default='control_v2,targeted_v3')
    args=parser.parse_args()
    result=run(Path(args.root),args.phase,args.run_until,args.arms.split(','))
    summary = {key:result.get(key) for key in ('status','phase','run_until','fixed_final_epoch','archives','output_root')}
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
'''

CONTROLS = r'''# Colab T4: 합성 v2 대조군과 어려운 작은 필라멘트 중심 v3를 비교합니다.
# 기본값: 각 군 30epoch 추가 학습, 두 군 합계60epoch. native best84는 고정합니다.
RUN_UNTIL = 30                 # 중간 점검은 10 또는 20; 이후30으로 높여 같은 스케줄에서 재개
PHASE = 'all'                  # 'prepare': 합성과 QC만 / 'train': 학습만 / 'evaluate': 평가만
ARMS = ['control_v2','targeted_v3']  # 순차 실행. 한 군만 지정해 다음 세션에 나머지 진행 가능
AUTO_DOWNLOAD = True
'''

FIRST = CONTROLS + r'''
BASE = '/content/drive/MyDrive/filament-segmentation-2026'
YOLO_RUN = '/content/drive/MyDrive/SolarFilament_LimbOnly_YOLO26_mixed_FullRes'
NATIVE_RUN = '/content/drive/MyDrive/SolarFilament_NativeUNet_A100/20261028T030808Z_bedf3e21'
FINAL_EPOCHS = 30              # 최초 실행 뒤 고정. 변경하려면 새 OUTPUT 폴더 사용
FULL_ENSEMBLE = True           # 원래 YOLO+고정 native 구성까지 평가. native가 없으면 시작 전 오류
IMAGE_SIZE = 2048              # 작은 객체 보존: 원래 YOLO 학습과 같은 해상도, batch1
CUDNN_ENABLED = True           # depthwise만 안전 경로로 우회; 처음 시작 전에만 변경 가능
OUTPUT = BASE + '/fixed_split_experiments/targeted_synthetic_v3_yolo_t4_v1'

from google.colab import drive, files
from pathlib import Path
import base64, hashlib, json, os, signal, subprocess, sys, zlib
drive.mount('/content/drive')
PAYLOAD = """__PAYLOAD__"""
raw = zlib.decompress(base64.b64decode(PAYLOAD))
if hashlib.sha256(raw).hexdigest() != __SHA__:
    raise ValueError('코드가 온전히 복사되지 않았습니다. 셀 전체를 다시 복사하세요.')
sources = json.loads(raw)
if set(sources) != set(__FILES__):
    raise ValueError('Invalid source bundle')
root = Path(OUTPUT); durable = root/'code'/__VERSION__
durable.mkdir(parents=True,exist_ok=True)
manifest = {}
for name, source in sources.items():
    if Path(name).name != name: raise ValueError('Invalid source name')
    if name.endswith('.py'): compile(source,name,'exec')
    else: json.loads(source)
    content = source.encode('utf-8'); manifest[name] = hashlib.sha256(content).hexdigest()
    path = durable/name
    if path.exists() and path.read_bytes() != content:
        raise ValueError('Drive에 보관한 코드가 변경되었습니다: '+name)
    if not path.exists(): path.write_bytes(content)
manifest_path = durable/'source_manifest.json'
manifest_bytes = (json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
if manifest_path.exists() and manifest_path.read_bytes() != manifest_bytes:
    raise ValueError('저장된 코드 manifest가 변경되었습니다.')
if not manifest_path.exists(): manifest_path.write_bytes(manifest_bytes)
base = Path(BASE); old = Path(YOLO_RUN); native = Path(NATIVE_RUN)
data = base/'MAGFiLO_1.0_Kaggle_2026/train'
synth = base/'synthetic_copy_paste_v2'
cfg = dict(images=str(data/'train_images'), annotations=str(data/'MAGFiLO_1.0_Annotations_kaggle2026_train.json'),
    train_split=str(old/'train_split.csv'), val_split=str(old/'val_split.csv'),
    synthetic_images=str(synth/'images'), synthetic_annotations=str(synth/'synthetic_annotations.json'),
    synthetic_split=str(synth/'split_info.json'), hard_donors=str(durable/'targeted_synthetic_hard_donors.json'),
    yolo=str(old/'best.pt'), native=str(native/'best.pt'),
    output_root=str(root), cache_root='/content/solar_targeted_synthetic_v3_cache',
    seed=42,epochs=FINAL_EPOCHS,imgsz=IMAGE_SIZE,batch=1,nbs=1,lr=1e-4,
    synthetic_samples=263,full_ensemble=FULL_ENSEMBLE,device='0',cudnn_enabled=CUDNN_ENABLED,
    code_dir=str(durable),code_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest())
config_path = root/'experiment_config.json'
config_bytes = (json.dumps(cfg,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode()
if config_path.exists() and config_path.read_bytes() != config_bytes:
    raise ValueError('기존 실험 설정과 다릅니다. 재개는 RUN_UNTIL/PHASE/ARMS만 변경하세요. 새 실험은 OUTPUT을 바꾸세요.')
if not config_path.exists(): config_path.write_bytes(config_bytes)
pointer = dict(version=__VERSION__, manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
               config_sha256=hashlib.sha256(config_bytes).hexdigest())
pointer_path = root/'current_code.json'
pointer_bytes = (json.dumps(pointer,sort_keys=True,indent=2)+'\n').encode()
if pointer_path.exists() and pointer_path.read_bytes() != pointer_bytes:
    raise ValueError('기존 실험의 코드 버전이 다릅니다. 원래 재개 셀 또는 새 OUTPUT 폴더를 사용하세요.')
if not pointer_path.exists(): pointer_path.write_bytes(pointer_bytes)
'''

RESUME = CONTROLS + r'''
# 새 런타임에서도 Drive에 저장된 코드·설정만으로 복원합니다. 최초 셀을 한 번 실행해야 합니다.
OUTPUT = '/content/drive/MyDrive/filament-segmentation-2026/fixed_split_experiments/targeted_synthetic_v3_yolo_t4_v1'
from google.colab import drive, files
from pathlib import Path
import hashlib, json, os, signal, subprocess, sys
drive.mount('/content/drive')
root = Path(OUTPUT)
pointer_path = root/'current_code.json'
if not pointer_path.is_file():
    raise FileNotFoundError('최초 실행용 복사 셀을 먼저 실행하세요: '+str(pointer_path))
pointer = json.loads(pointer_path.read_text())
durable = root/'code'/pointer['version']
manifest_path = durable/'source_manifest.json'
if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != pointer['manifest_sha256']:
    raise ValueError('Drive 코드 manifest가 변경되었습니다.')
for name,expected in json.loads(manifest_path.read_text()).items():
    if Path(name).name != name or hashlib.sha256((durable/name).read_bytes()).hexdigest() != expected:
        raise ValueError('Drive 코드가 변경되었습니다: '+name)
if hashlib.sha256((root/'experiment_config.json').read_bytes()).hexdigest() != pointer['config_sha256']:
    raise ValueError('Drive 학습 설정이 변경되었습니다.')
'''

EXECUTE = r'''
command = [sys.executable,'-u',str(durable/'targeted_synthetic_colab_bootstrap.py'),
           '--root',str(root),'--phase',PHASE,'--run-until',str(RUN_UNTIL),'--arms',','.join(ARMS)]
environment = dict(os.environ); environment['TORCH_CUDNN_V8_API_DISABLED']='1'
process = subprocess.Popen(command, env=environment, start_new_session=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, bufsize=1)
try:
    for line in process.stdout:
        print(line, end='', flush=True)
    code = process.wait()
    if code: raise subprocess.CalledProcessError(code,command)
except BaseException:
    try: os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError: pass
    try: process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL);process.wait()
    raise
result = json.loads((root/'result.json').read_text())
print('실행 상태:',result['status'])
dataset = result.get('dataset') or {}
preview = dataset.get('preview')
native_preview = dataset.get('native_preview')
if not native_preview and dataset.get('manifest'):
    native_preview = str(Path(dataset['manifest']).with_name('synthetic_native_preview.png'))
qc_images = [('합성 전체 영상·마스크 QC',preview),('원본 배율 donor/합성 ROI QC',native_preview)]
if any(path and Path(path).is_file() for _,path in qc_images):
    from IPython.display import display, Image
    for title,path in qc_images:
        if path and Path(path).is_file():
            print(title+': '+str(path));display(Image(filename=str(path)))
for archive in result.get('archives',[]):
    print('분석용 결과 ZIP:',archive)
    if AUTO_DOWNLOAD:
        try: files.download(archive)
        except Exception as exc: print('Drive의 위 경로에서 받을 수 있습니다:',exc)
'''


def build_sources():
    sources = {name:(ROOT/'tools'/name).read_text(encoding='utf-8') for name in MODULES}
    sources[BOOTSTRAP_NAME] = BOOTSTRAP
    for name, source in sources.items():
        if name.endswith('.py'):
            compile(source,name,'exec')
        else:
            json.loads(source)
    return sources


def build():
    sources = build_sources()
    raw = json.dumps(sources,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
    digest = hashlib.sha256(raw).hexdigest()
    payload = '\n'.join(textwrap.wrap(base64.b64encode(zlib.compress(raw,9)).decode(),120))
    cell = (FIRST.replace('__PAYLOAD__',payload).replace('__SHA__',repr(digest))
            .replace('__FILES__',repr(tuple(sources))).replace('__VERSION__',repr(digest[:16])))+EXECUTE
    resume = RESUME+EXECUTE
    compile(cell,'targeted-synthetic-first','exec')
    compile(resume,'targeted-synthetic-resume','exec')
    out = ROOT/'notebooks';out.mkdir(parents=True,exist_ok=True)
    first_path = out/'SolarFilament_TargetedSyntheticV3_T4_CopyCell.py'
    resume_path = out/'SolarFilament_TargetedSyntheticV3_T4_ResumeCell.py'
    nb_path = out/'SolarFilament_TargetedSyntheticV3_T4.ipynb'
    first_path.write_text(cell,encoding='utf-8')
    resume_path.write_text(resume,encoding='utf-8')
    intro = [
        '# T4: 작은 필라멘트 중심 합성 v3 추가 학습\n',
        '원래 train616 / validation91 분할을 유지합니다. 기존 YOLO에서 대조군과 v3군을 각각30epoch 추가 학습합니다. 총60epoch이며 5-fold 학습이 아닙니다. native best84는 고정합니다.\n',
        '첫 실행: T4 GPU 런타임에서 아래 최초 셀만 실행하세요. `PHASE="prepare"`면 합성/QC까지만 실행합니다. 기본 `all`은 합성→두 군 학습→검증까지 진행합니다.\n',
        '중단 후 최초 셀을 다시 실행하거나 마지막 재개 셀을 실행하면 마지막 완료 epoch부터 이어갑니다. 중간 배치 진행은 재실행합니다. `FINAL_EPOCHS=30`은 고정하고 `RUN_UNTIL=10→20→30`만 높일 수 있습니다.\n',
        'Drive에 코드·설정·checkpoint를 남기며 `/content` 데이터 캐시는 원본으로 재생성합니다. 최초 셀과 재개 셀을 동시에 실행하지 마세요.\n',
        '이 파일은 실행용 코드입니다. 새 합성 데이터의 실제 영상 품질·학습 성능·Colab T4 실행 시간은 아직 측정하지 않았습니다.\n',
    ]
    notebook = dict(nbformat=4,nbformat_minor=5,
        metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python'),
                      colab=dict(name=nb_path.name)),
        cells=[dict(id='intro',cell_type='markdown',metadata={},source=intro),
               dict(id='first-run',cell_type='code',metadata={},execution_count=None,outputs=[],source=cell.splitlines(True)),
               dict(id='resume-info',cell_type='markdown',metadata={},source=['## 세션 종료 후 재개용\n','최초 셀을 실행한 뒤에만 사용하세요. Drive에서 원래 코드와 설정을 검증하고 라이브러리·임시 파일을 복원합니다.\n']),
               dict(id='resume',cell_type='code',metadata={},execution_count=None,outputs=[],source=resume.splitlines(True))])
    nb_path.write_text(json.dumps(notebook,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    result=dict(copy_cell=str(first_path),resume_cell=str(resume_path),notebook=str(nb_path),
        source_bundle_sha256=digest,source_count=len(sources),cell_bytes=len(cell.encode()),
        resume_bytes=len(resume.encode()),default_epochs_per_arm=30,default_total_epochs=60,
        frozen_split=[616,91],models_trained_here=False,T4_execution_tested_here=False,
        source_sha256={name:hashlib.sha256(source.encode()).hexdigest() for name,source in sources.items()})
    return result


if __name__=='__main__':
    print(json.dumps(build(),ensure_ascii=False,indent=2))
