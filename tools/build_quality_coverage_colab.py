"""Package a resumable T4 sampler-coverage experiment; no public/test assets."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import zipfile
import zlib

REPO=Path(__file__).resolve().parents[1]
ART=REPO.parent/'artifacts'
DEST=ART/'downloads'
STEM='SolarFilament_QualityCoverage_T4'
MODULES=('quality_coverage_finetune.py','quality_coverage_sampling.py',
         'quality_append_api.py','filament_eval.py','submission_inference.py')
ASSETS={
 'full_baseline.csv':ART/'detr_best84_uploaded_d594c020e79f/predictions/full.csv',
 'validation_gt.csv':ART/'detr_best84_analysis_d594c020e79f/metrics/validation_gt.csv',
 'raw_views.jsonl':ART/'validation_results_20261027T120958Z_444da7f4/raw_views.jsonl',
}
for _name in ('corpus.json','corpus_identity.json','training_config.json',
              'best_validation_scores.json'):
    ASSETS['old_'+_name]=ART/'mask_quality_replay_uploaded_20261101'/_name

SETTINGS='''# 기존 품질 검증기 1개를 추가 학습합니다. YOLO/best84·5-fold는 다시 학습하지 않습니다.
# TOTAL_ADDITIONAL_EPOCHS=30은 전체 추가 학습 스케줄입니다.
# 중간에 멈추려면 RUN_UNTIL_EPOCH만 줄이세요. 재실행하면 마지막 완료 epoch부터 이어집니다.
TOTAL_ADDITIONAL_EPOCHS = 30
RUN_UNTIL_EPOCH = 30
BATCH_SIZE = 16
AUTO_DOWNLOAD = True

from pathlib import Path
from google.colab import drive, files
from datetime import datetime, timezone
from collections import deque
import base64, hashlib, importlib.util, json, os, signal, subprocess, sys, uuid, zlib
drive.mount('/content/drive')
BASE = Path('/content/drive/MyDrive/filament-segmentation-2026')
ROOT = BASE/'fixed_split_experiments/quality_coverage_t4_v1'
OLD_ROOT = BASE/'fixed_split_experiments/mask_quality_v1'
IMAGES = BASE/'MAGFiLO_1.0_Kaggle_2026/train/train_images'
CACHE = Path('/content/solar_quality_coverage_v1/roi')
if type(RUN_UNTIL_EPOCH) is not int or not 0 <= RUN_UNTIL_EPOCH <= TOTAL_ADDITIONAL_EPOCHS:
    raise ValueError('RUN_UNTIL_EPOCH은 0~TOTAL_ADDITIONAL_EPOCHS 정수여야 합니다.')
if type(BATCH_SIZE) is not int or BATCH_SIZE < 1:
    raise ValueError('BATCH_SIZE는 양의 정수여야 합니다.')
ROOT.mkdir(parents=True, exist_ok=True)
def atomic_bytes(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name+'.'+uuid.uuid4().hex+'.pending')
    pending.write_bytes(value)
    pending.replace(path)
def save_exact(path, value):
    if path.exists():
        if path.read_bytes() != value:
            raise ValueError('보관된 코드/입력이 다릅니다: '+str(path))
    else:
        atomic_bytes(path, value)
'''

INSTALL_AND_RUN='''
if importlib.util.find_spec('torch') is None:
    raise RuntimeError('PyTorch가 있는 Colab T4 GPU 런타임을 사용해 주세요. 이 셀은 torch/CUDA를 설치하거나 교체하지 않습니다.')
missing = [package for module,package in [('timm','timm==1.0.20'),
    ('cv2','opencv-python-headless'),('scipy','scipy'),
    ('pycocotools','pycocotools==2.0.11')] if importlib.util.find_spec(module) is None]
if missing:
    subprocess.run([sys.executable,'-m','pip','install',*missing],check=True)
for name,expected in manifest['sources'].items():
    path=CODE/name
    if Path(name).name != name or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise ValueError('보관된 실행 코드가 변경됐습니다: '+name)
for name,expected in manifest['assets'].items():
    path=ASSETS/name
    if Path(name).name != name or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise ValueError('보관된 실험 입력이 변경됐습니다: '+name)
if not (OLD_ROOT/'best.pt').is_file():
    raise FileNotFoundError('기존 품질 모델의 best.pt가 필요합니다: '+str(OLD_ROOT/'best.pt'))
if not (OLD_ROOT/'corpus.json').is_file() or not (OLD_ROOT/'proposals').is_dir():
    raise FileNotFoundError('기존 mask_quality_v1 후보 캐시가 필요합니다: '+str(OLD_ROOT))
if not IMAGES.is_dir():
    raise FileNotFoundError('원본 학습 이미지 폴더를 찾지 못했습니다: '+str(IMAGES))
environment=dict(os.environ,PYTHONUNBUFFERED='1',TORCH_CUDNN_V8_API_DISABLED='1',
    CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
# GPU 검사는 별도 프로세스에서 먼저 수행해, CPU 런타임에서 긴 ROI 준비를 시작하지 않습니다.
preflight = "import torch; print('PyTorch:',torch.__version__,'CUDA:',torch.version.cuda,flush=True); assert torch.cuda.is_available(), 'Colab GPU 런타임을 선택하세요. CUDA를 사용할 수 없습니다.'; print('GPU:',torch.cuda.get_device_name(0),'cuDNN:',torch.backends.cudnn.version(),flush=True)"
preflight_result=subprocess.run([sys.executable,'-u','-c',preflight],cwd=CODE,env=environment)
if preflight_result.returncode:
    raise RuntimeError('GPU 사전 확인에 실패했습니다. 위 오류를 확인하고 Colab T4 GPU 런타임에서 다시 실행해 주세요. ROI 준비와 학습은 시작하지 않았습니다.')
common=[sys.executable,'-u',str(CODE/'quality_coverage_finetune.py'),
    '--root',str(ROOT),'--old-root',str(OLD_ROOT),'--images',str(IMAGES),
    '--cache',str(CACHE),'--assets',str(ASSETS),'--epochs',str(TOTAL_ADDITIONAL_EPOCHS),
    '--run-until',str(RUN_UNTIL_EPOCH),'--batch-size',str(BATCH_SIZE)]
def run_phase(phase,device):
    command=common+['--phase',phase,'--device',device]
    log_path=ROOT/('colab_'+phase+'.log');tail=deque(maxlen=50)
    print('실행 단계:',phase,'/ 로그:',log_path,flush=True)
    with log_path.open('a',encoding='utf-8',buffering=1) as log:
        log.write('\\n=== '+datetime.now(timezone.utc).isoformat()+' ===\\n')
        process=subprocess.Popen(command,cwd=CODE,env=environment,start_new_session=True,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,errors='replace')
        try:
            for line in process.stdout:
                print(line,end='',flush=True);log.write(line);tail.append(line)
            status=process.wait()
            if status:
                print('\\n===== 실제 오류 마지막 출력 =====\\n'+''.join(tail),flush=True)
                raise RuntimeError(phase+' 종료 코드 '+str(status)+'; 위 traceback과 로그를 전달해 주세요: '+str(log_path))
        except BaseException:
            try: os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError: pass
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try: os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError: pass
                process.wait()
            raise
        finally:
            process.stdout.close()
print('원래 train616/validation91 고정분할, 품질 검증기 1개만 추가 학습합니다.',flush=True)
print('기존 YOLO+best84 출력과 Q50 추가 규칙은 고정합니다. 후보 노출을 개선한 추가 학습을 평가합니다.',flush=True)
print('전체 추가 스케줄:',TOTAL_ADDITIONAL_EPOCHS,'/ 이번 실행:',RUN_UNTIL_EPOCH,'epoch까지',flush=True)
print('ROI 배열은 /content에 생성합니다. 세션 종료 후 같은 셀을 실행하면 캐시를 복원합니다.',flush=True)
print('완료된 epoch의 모델·optimizer·스케줄·RNG는 Drive에서 재개합니다:',ROOT,flush=True)
# prepare와 train은 별도 프로세스라 준비 단계의 메모리를 학습에 남기지 않습니다.
run_phase('prepare','cpu')
run_phase('train','cuda')
result_path=ROOT/'result.json'
if not result_path.is_file():
    raise FileNotFoundError('실험 결과 파일이 생성되지 않았습니다: '+str(result_path))
result=json.loads(result_path.read_text())
print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
archive_value=result.get('archive')
if archive_value:
    archive=Path(archive_value).resolve()
    if not archive.is_file() or ROOT.resolve() not in archive.parents:
        raise ValueError('결과 ZIP 경로가 올바르지 않습니다: '+str(archive))
    if result.get('archive_sha256') and hashlib.sha256(archive.read_bytes()).hexdigest()!=result['archive_sha256']:
        raise ValueError('결과 ZIP의 해시가 다릅니다.')
    print('전달할 결과 ZIP:',archive,flush=True)
    if AUTO_DOWNLOAD:
        try: files.download(str(archive))
        except Exception as exc: print('자동 다운로드 실패. 위 Drive 경로에서 받을 수 있습니다:',exc,flush=True)
else:
    print('결과 ZIP 경로가 없습니다. result.json과 위 실행 로그를 확인해 주세요.',flush=True)
'''

FIRST=SETTINGS+'''
# 실행 코드와 고정 validation 입력을 Drive에 함께 보관합니다.
raw=zlib.decompress(base64.b64decode("""__PAYLOAD__"""))
if hashlib.sha256(raw).hexdigest()!='__DIGEST__':
    raise ValueError('복사된 셀 내용이 완전하지 않습니다. 파일 전체를 다시 복사해 주세요.')
bundle=json.loads(raw);version='__VERSION__'
CODE=ROOT/'code'/version;ASSETS=ROOT/'assets'/version
for name,source in bundle['sources'].items():
    if Path(name).name!=name: raise ValueError('잘못된 코드 파일명')
    compile(source,name,'exec');save_exact(CODE/name,source.encode())
for name,value in bundle['assets'].items():
    if Path(name).name!=name: raise ValueError('잘못된 입력 파일명')
    save_exact(ASSETS/name,base64.b64decode(value))
manifest=dict(sources={name:hashlib.sha256(source.encode()).hexdigest() for name,source in bundle['sources'].items()},
    assets={name:hashlib.sha256(base64.b64decode(value)).hexdigest() for name,value in bundle['assets'].items()})
manifest_bytes=json.dumps(manifest,sort_keys=True,indent=2).encode()
save_exact(CODE/'manifest.json',manifest_bytes)
config=dict(total_additional_epochs=TOTAL_ADDITIONAL_EPOCHS,batch_size=BATCH_SIZE,
    old_root=str(OLD_ROOT),images=str(IMAGES),scope='quality_coverage_fixed616_91_v1')
config_bytes=json.dumps(config,sort_keys=True,indent=2).encode()
save_exact(ROOT/'launch_config.json',config_bytes)
pointer=dict(version=version,bundle_sha256='__DIGEST__',manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    config_sha256=hashlib.sha256(config_bytes).hexdigest(),scope='quality_coverage_fixed616_91_v1')
pointer_bytes=json.dumps(pointer,sort_keys=True,indent=2).encode()
save_exact(ROOT/'current_code.json',pointer_bytes)
'''+INSTALL_AND_RUN

RESUME=SETTINGS+'''
# 최초 셀을 실행한 후에는 세션이 바뀌어도 이 짧은 셀로 재개할 수 있습니다.
pointer_path=ROOT/'current_code.json'
if not pointer_path.is_file():
    raise FileNotFoundError('최초 실행용 QualityCoverage 셀을 먼저 실행해 주세요: '+str(pointer_path))
pointer=json.loads(pointer_path.read_text())
version=pointer.get('version','')
if pointer.get('scope')!='quality_coverage_fixed616_91_v1' or len(version)!=16 or any(c not in '0123456789abcdef' for c in version):
    raise ValueError('이 실험의 코드 보관 폴더가 아닙니다.')
CODE=ROOT/'code'/version;ASSETS=ROOT/'assets'/version
manifest_path=CODE/'manifest.json'
if not manifest_path.is_file() or hashlib.sha256(manifest_path.read_bytes()).hexdigest()!=pointer['manifest_sha256']:
    raise ValueError('보관된 코드/입력 manifest가 다릅니다.')
manifest=json.loads(manifest_path.read_text())
config_path=ROOT/'launch_config.json'
if not config_path.is_file() or hashlib.sha256(config_path.read_bytes()).hexdigest()!=pointer['config_sha256']:
    raise ValueError('보관된 실행 설정이 다릅니다.')
expected=dict(total_additional_epochs=TOTAL_ADDITIONAL_EPOCHS,batch_size=BATCH_SIZE,
    old_root=str(OLD_ROOT),images=str(IMAGES),scope='quality_coverage_fixed616_91_v1')
if json.loads(config_path.read_text())!=expected:
    raise ValueError('기존 추가학습 스케줄/배치/입력 설정을 유지해 주세요. RUN_UNTIL_EPOCH만 바꿔 이어갈 수 있습니다.')
'''+INSTALL_AND_RUN

README='''# T4 품질 검증기 샘플 노출 개선 실험

- 원래 train616 / validation91을 유지합니다. 원본 YOLO와 best84는 고정합니다.
- 기존 품질 검증기 best epoch18에서 모델 가중치만 시작하고, 새 optimizer로 **추가30epoch의 단일 모델**을 학습합니다.
- 기존 고정 confidence 상위48개 선택을 바꿔 낮은 confidence의 정답 후보도 에폭별 순환 노출합니다. 백본과 회귀/BCE 목표는 유지합니다.
- 변경 전 epoch0 결과를 그대로 대조군으로 보관합니다. 이번에는 추가 학습하는 모델이1개이므로, 개선이 생겨도 **추가 epoch 효과와 샘플러 변경 효과를 각각 분리해 단정할 수는 없습니다.**
- 실제 full YOLO+best84 validation 기준선 PQ0.472273567 및 고정 Q50 추가 규칙으로 평가합니다. 새로 학습한 점수를 위해 Q50 임계값을 탐색하지 않습니다.
- 공개/test 예측은 학습 입력에 없습니다. validation 정답은 평가에만 사용합니다.

사용 방법:

1. Colab T4 GPU 런타임에서 `CopyCell.txt` 전체를 새 셀에 붙여 실행합니다.
2. 처음부터 동일 셀을 다시 실행해도 완료 epoch를 재개합니다. 최초 실행 이후에는 짧은 `ResumeCell.txt`도 사용할 수 있습니다.
3. 전체 스케줄은 `TOTAL_ADDITIONAL_EPOCHS=30`으로 유지합니다. 먼저 5epoch만 실행하려면 `RUN_UNTIL_EPOCH=5`로 바꾸고, 나중에30으로 변경합니다.
4. 종료 후 결과 ZIP을 전달합니다. 가중치·optimizer는 Drive에 남고, ZIP에는 평가 결과와 실행 기록을 넣습니다.

기본 경로:

- 원래 품질 모델과 후보: `MyDrive/filament-segmentation-2026/fixed_split_experiments/mask_quality_v1`
- 새 실험: `MyDrive/filament-segmentation-2026/fixed_split_experiments/quality_coverage_t4_v1`
- 원본 이미지: `MyDrive/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/train_images`
- 모든 후보의 ROI 배열은 약8GiB 규모의 `/content` 로컬 캐시로 준비합니다. 세션 종료 시 다시 만들며, 원래 데이터·학습 결과는 덮어쓰지 않습니다.
- ROI 준비 전에 별도 프로세스에서 CUDA 사용 가능 여부를 먼저 확인합니다. torch/CUDA는 설치하거나 교체하지 않습니다.

이 코드의 로컬 검증과 실제 Colab GPU 학습은 구분합니다. 생성 시점에는 T4 전체 학습과 PQ 개선을 확인하지 않았습니다. 기존 checkpoint는 같은 validation으로 선택했던 모델이므로 이후 결과도 고정 개발 분할 비교입니다.
'''


def build():
    sources={name:(REPO/'tools'/name).read_text() for name in MODULES}
    for name,source in sources.items():compile(source,name,'exec')
    assets={name:base64.b64encode(path.read_bytes()).decode() for name,path in ASSETS.items()}
    raw=json.dumps(dict(sources=sources,assets=assets),sort_keys=True,separators=(',',':')).encode()
    digest=hashlib.sha256(raw).hexdigest();version=digest[:16]
    first=FIRST.replace('__PAYLOAD__',base64.b64encode(zlib.compress(raw,9)).decode()).replace('__DIGEST__',digest).replace('__VERSION__',version)
    compile(first,'quality-coverage-first','exec');compile(RESUME,'quality-coverage-resume','exec')
    DEST.mkdir(parents=True,exist_ok=True)
    first_path=DEST/(STEM+'_CopyCell.txt');first_path.write_text(first)
    resume_path=DEST/(STEM+'_ResumeCell.txt');resume_path.write_text(RESUME)
    readme_path=DEST/(STEM+'_README.ko.md');readme_path.write_text(README)
    notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python')),
        cells=[dict(id='intro',cell_type='markdown',metadata={},source=README.splitlines(keepends=True)),
            dict(id='first',cell_type='code',execution_count=None,metadata={},outputs=[],source=first.splitlines(keepends=True)),
            dict(id='resume-note',cell_type='markdown',metadata={},source=['## 세션 종료 후 재개\n','최초 셀을 실행한 다음부터 사용합니다. 최초 셀을 다시 실행해도 재개합니다.\n']),
            dict(id='resume',cell_type='code',execution_count=None,metadata={},outputs=[],source=RESUME.splitlines(keepends=True))])
    nb=DEST/(STEM+'.ipynb');nb.write_text(json.dumps(notebook,ensure_ascii=False,indent=1))
    archive=DEST/(STEM+'_Package.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in (first_path,resume_path,nb,readme_path):z.write(path,path.name)
        for name,source in sources.items():z.writestr('tools/'+name,source)
        for name,path in ASSETS.items():z.write(path,'assets/'+name)
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    manifest=dict(bundle_sha256=digest,source_sha256={n:hashlib.sha256(s.encode()).hexdigest() for n,s in sources.items()},
        asset_sha256={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in ASSETS.items()},
        copy_cell=str(first_path),resume_cell=str(resume_path),notebook=str(nb),package=str(archive),
        copy_cell_bytes=first_path.stat().st_size,package_bytes=archive.stat().st_size,
        package_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (DEST/(STEM+'_manifest.json')).write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return manifest


if __name__=='__main__':print(json.dumps(build(),ensure_ascii=False,indent=2))
