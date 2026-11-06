"""Create a self-contained, single-code-cell Colab package for CHASE-only transfer."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import zlib

REPO = Path(__file__).resolve().parents[1]
ART = REPO.parent/'artifacts'
DEST = ART/'downloads'
ASSET_ROOT = ART/'chase_both_training_assets_20261104'
STEM = 'SolarFilament_CHASE_Only_BothModels_60Epoch_FullPass_FixPNG'
MODULES = ('chase_both_colab_bootstrap.py', 'chase_transfer_data.py',
    'chase_best84_training.py', 'chase_best84_evaluation.py',
    'chase_yolo_training.py', 'chase_yolo_predict.py',
    'solar_a100_train_standalone.py', 'targeted_synthetic_training.py',
    'submission_inference.py', 'submission_driver.py',
    'submission_yolo_recipe.py', 'submission_native_recipe.py')
ASSET_NAMES = ('frozen_yolo_validation.json.gz', 'train_split.csv', 'val_split.csv',
               'chase_ahas2024_2025_manifest.json')

CELL = '''# CHASE695장만 추가 학습: 기존YOLO26s-seg + 기존best84 U-Net을 각각 이어 학습합니다.
# PNG 읽기 오류 수정판: 기존 CHASE 원본과 다운로드 ZIP을 확인 후 재사용합니다.
# GONG 재학습·혼합·마지막 미세조정은 하지 않습니다. GONG91장은 검증에만 씁니다.
# 각60epoch, 두 모델 합계120epoch입니다. 매epoch CHASE695장 전부를 각각 한 번 사용합니다.
# 같은 셀을 다시 실행하면 마지막 완료epoch부터 재개합니다. 숨겨진100epoch 학습은 없습니다.
# 기존30epoch·400sample 실험과 다른 새 폴더에서 시작합니다. 예전checkpoint를 자동 승계하지 않습니다.
# 학습 종료 후 validation91 추론과 네 조합 평가가 이어집니다(학습epoch 증가 아님).
RUN_UNTIL_EPOCH = 60       # 1~60. 예:30까지 실행 후60으로 변경하면 같은60epoch일정으로 재개.
ARMS = ['yolo', 'native']  # 기본둘다. 이미 끝난 모델은 완료checkpoint를 재사용합니다.
PHASE = 'all'             # all / prepare(CPU자료준비만) / train / compare
AUTO_DOWNLOAD = True

from pathlib import Path
from google.colab import drive, files
from collections import deque
from datetime import datetime, timezone
import base64, hashlib, importlib.metadata, importlib.util, json, os, signal, subprocess, sys, uuid, zlib
drive.mount('/content/drive')
BASE = Path('/content/drive/MyDrive/filament-segmentation-2026')
ROOT = BASE/'fixed_split_experiments/chase_only_both_alignment_t4_60ep_fullpass_pngfix_v1'
IMAGE_DIR = BASE/'MAGFiLO_1.0_Kaggle_2026/train/train_images'
ANNOTATIONS = BASE/'MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'
YOLO_CHECKPOINT = Path('/content/drive/MyDrive/SolarFilament_LimbOnly_YOLO26_mixed_FullRes/best.pt')
NATIVE_CHECKPOINT = Path('/content/drive/MyDrive/SolarFilament_NativeUNet_A100/20261028T030808Z_bedf3e21/best.pt')
CHASE_ROOT = BASE/'external_CHASE/ahas_2024_2025'
CACHE_ROOT = Path('/content/chase_only_both_alignment_60ep_fullpass_cache')
# 위 CHASE_ROOT와 CACHE_ROOT는 오류가 난 버전과 동일하게 유지합니다.
# 학습 시작 전 준비 단계에서 실패했으므로 새 실험 폴더에서 수정된 코드를 실행합니다.
# 기존 코드·설정·원본 가중치와 이미 저장된 CHASE PNG는 변경하거나 삭제하지 않습니다.
if type(RUN_UNTIL_EPOCH) is not int or not 1 <= RUN_UNTIL_EPOCH <= 60:
    raise ValueError('RUN_UNTIL_EPOCH은1~60입니다. 전체스케줄은각60epoch로고정됩니다.')
if not ARMS or len(set(ARMS))!=len(ARMS) or not set(ARMS)<={'yolo','native'}:
    raise ValueError('ARMS에는 yolo/native를 중복 없이 넣으세요.')
if PHASE not in ('all','prepare','train','compare'):
    raise ValueError('PHASE는 all/prepare/train/compare 중 하나입니다.')
ROOT.mkdir(parents=True,exist_ok=True)
raw=zlib.decompress(base64.b64decode("""__PAYLOAD__"""))
if hashlib.sha256(raw).hexdigest()!='__SHA__':
    raise ValueError('복사 셀이 완전하지 않습니다. 셀 전체를 다시 복사해 주세요.')
bundle=json.loads(raw);version='__VERSION__'
CODE=ROOT/'code'/version;ASSETS=ROOT/'assets'/version
def save_exact(path,data):
    if path.exists():
        if path.read_bytes()!=data:
            raise ValueError('저장된 코드/자료가 변경됐습니다: '+str(path))
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_name(path.name+'.'+uuid.uuid4().hex+'.pending')
    pending.write_bytes(data);pending.replace(path)
for name,source in bundle['sources'].items():
    if Path(name).name!=name:raise ValueError('잘못된 코드 이름')
    compile(source,name,'exec');save_exact(CODE/name,source.encode())
for name,value in bundle['assets'].items():
    if Path(name).name!=name:raise ValueError('잘못된 자료 이름')
    save_exact(ASSETS/name,base64.b64decode(value))
manifest=dict(sources={name:hashlib.sha256(source.encode()).hexdigest() for name,source in bundle['sources'].items()},
    assets={name:hashlib.sha256(base64.b64decode(value)).hexdigest() for name,value in bundle['assets'].items()})
save_exact(CODE/'source_manifest.json',json.dumps(manifest,sort_keys=True,indent=2).encode())
for folder,key in ((CODE,'sources'),(ASSETS,'assets')):
    for name,expected in manifest[key].items():
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('코드 또는 자료 해시가 다릅니다: '+name)

# Colab의 CUDA PyTorch/torchvision은 교체하지 않습니다. 나머지는 필요한 경우에만 설치합니다.
for module in ('torch','torchvision'):
    if importlib.util.find_spec(module) is None:
        raise RuntimeError('PyTorch가 포함된 Colab 런타임을 사용하세요: '+module)
pinned={'ultralytics':'8.4.182','segmentation-models-pytorch':'0.5.0','timm':'1.0.20'}
install=[]
for package,expected in pinned.items():
    try:current=importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:current=None
    if current!=expected:install.append(package+'=='+expected)
for module,package in [('cv2','opencv-python-headless'),('numpy','numpy'),('scipy','scipy'),
    ('requests','requests'),('pycocotools','pycocotools'),('PIL','Pillow')]:
    if importlib.util.find_spec(module) is None:install.append(package)
if install:
    constraints=Path('/content/chase_torch_constraints.txt')
    constraints.write_text(''.join(package+'=='+importlib.metadata.version(package)+'\\n' for package in ('torch','torchvision')))
    subprocess.run([sys.executable,'-m','pip','install','--constraint',str(constraints),*install],check=True)

settings=dict(root=str(ROOT),assets=str(ASSETS),images=str(IMAGE_DIR),annotations=str(ANNOTATIONS),
    yolo_checkpoint=str(YOLO_CHECKPOINT),native_checkpoint=str(NATIVE_CHECKPOINT),
    chase_root=str(CHASE_ROOT),cache_root=str(CACHE_ROOT))
settings_path=ROOT/'bootstrap_settings.json'
encoded=json.dumps(settings,sort_keys=True,indent=2).encode()
save_exact(settings_path,encoded)
command=[sys.executable,'-u',str(CODE/'chase_both_colab_bootstrap.py'),'--settings',str(settings_path),
         '--run-until',str(RUN_UNTIL_EPOCH),'--arms',','.join(ARMS),'--phase',PHASE]
environment=dict(os.environ,PYTHONUNBUFFERED='1',TORCH_CUDNN_V8_API_DISABLED='1',
                 OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
log_path=ROOT/'training.log';tail=deque(maxlen=60)
print('추가 학습은 CHASE695장만 사용합니다. GONG91장은 검증 전용입니다.',flush=True)
print('PNG 수정판: 기존 Drive 원본과 같은 런타임의 다운로드 ZIP을 확인 후 재사용합니다.',flush=True)
print('기존 실험 폴더는 보존하고 수정판 폴더에서 실행합니다.',flush=True)
print('기존 두 가중치에서 이어 학습하며, GONG 재학습·혼합·마지막 미세조정은 없습니다.',flush=True)
print('각 모델 전체60epoch / 이번',RUN_UNTIL_EPOCH,'까지. 모델 순서:',ARMS,flush=True)
print('매epoch695장 전부 한 번씩 사용: 모델당41700표본 / 두 모델83400표본.',flush=True)
print('CHASE는 원반 정렬만 적용합니다. 명시된 겹침정보 없이 형태로 제외하지 않습니다.',flush=True)
print('로그:',log_path,'\\n저장 폴더:',ROOT,flush=True)
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
            raise RuntimeError('종료 코드 '+str(status)+'; 위 traceback과 로그를 확인하세요: '+str(log_path))
    except BaseException:
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            process.wait()
        raise
    finally:
        process.stdout.close()
result=json.loads((ROOT/'latest_result.json').read_text())
print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
if result.get('comparison_available'):
    report=json.loads(Path(result['comparison_report']).read_text())
    for row in report['results']:
        print(row['role'],'PQ',round(row['pq'],6),'FP',row['fp'],'FN',row['fn'],flush=True)
    print('위 점수는 고정 validation입니다. 새 리더보드 점수는 아직 없습니다.',flush=True)
print('업로드할 결과 ZIP:',result['archive'],flush=True)
if AUTO_DOWNLOAD:
    try:files.download(result['archive'])
    except Exception as exc:print('자동 다운로드가 중단됐습니다. 위 Drive ZIP을 직접 내려받으세요:',exc)
'''

README = '''# CHASE만 추가 학습 — 기존 YOLO와 best84 둘 다

## PNG 준비 오류 수정판

- 오류 원인은 모든 원본 이미지를2048×2048로 가정한 검증 조건입니다. 확인된2025년 원본에는 가로2304×세로2313 영상이 있고, 대응하는 마스크는2048×2048입니다.
- 저자의 공개 [마스크 생성 코드](https://github.com/irisaltHu/MORDEN/blob/e8186480a37395272ba414a0143b2b1232294ed1/segmentation/inference.py#L102)를 기준으로 원본 전체 영상을 마스크 좌표계에 bilinear resize한 뒤 원반 정렬을 함께 적용합니다. 이미지 일부를 임의로 잘라 맞추지 않으며, 전달된 이진 마스크는 그 좌표계에서 그대로 사용합니다. 기존 원본 PNG 바이트와 체크섬은 보존합니다.
- 기존 실행은 CHASE 파일을 읽는 준비 단계에서 중단됐으며 모델 학습은 아직 시작하지 않았습니다. 이 파일은 수정된 코드를 별도 실험 폴더에서 실행합니다. 예전 코드·설정 파일을 덮어쓰지 않습니다.
- **기존 셀을 이 수정판의 셀로 교체하고 `PHASE='all'`로 실행하세요.** 준비가 실패한 상태에서 `PHASE='train'`으로 건너뛰지 않습니다.
- CHASE 원본 폴더와 로컬 다운로드 캐시 경로는 이전 버전과 같습니다. 이미 저장한 원본 PNG는 CRC·크기 확인 후 재사용하고, 현재 런타임에 남아 있는 완성 ZIP은 MD5 확인 후 재사용합니다. 동일 자료를 처음부터 다시 받을 필요가 없습니다.
- Colab 런타임이 초기화되어 로컬 ZIP이 사라졌고 Drive 원본도 아직 일부만 있다면, 남은 파일을 복구하기 위해 ZIP을 다시 다운로드해야 합니다. Drive에695쌍이 모두 준비돼 있으면 ZIP 없이도 재개할 수 있습니다.

- 이 노트북은 기존 LB0.39 파이프라인의 **YOLO26s-seg와 ConvNeXt-Tiny U-Net(best84)을 각각 추가 학습**합니다. 별도의 학습형 refiner는 없으며 후처리 규칙은 그대로입니다.
- **추가 학습은 CHASE2024~2025 영상695장만 사용합니다. GONG 재학습·혼합·마지막 GONG 미세조정은 하지 않습니다.** GONG91장은 성능 검증에만 사용합니다. 기존 가중치가 이미 배운 GONG 특징을 초기값으로 사용합니다.
- CHASE 라벨은 자동 생성 후 사람이 선별한 이진 마스크이므로 객체 정답과 동등하다고 가정하지 않습니다.
- CHASE에는 원반 중심·크기 정렬과 모델 입력 수치 정규화만 적용합니다. CLAHE, 추가 limb 보정, 밝기·흐림 증강은 적용하지 않습니다. 겹침 여부가 명시되지 않은 영상을 모양만 보고 제외하지 않습니다.
- YOLO용 CHASE 라벨은 연결 영역에서 polygon을 만듭니다. 연결 영역은 검증된 필라멘트 개체 ID가 아닙니다. 변환 수량과 면적 차이를 감사 보고서에 기록합니다.

## 실행

1. Colab T4 GPU를 선택하고 `OneCell.ipynb`의 유일한 코드 셀을 복사해 실행합니다. 코드·분할·YOLO validation 캐시가 셀에 들어 있어 GitHub 다운로드에 의존하지 않습니다.
2. 기존 Drive의 train_images, 정답JSON, 두 best.pt 경로가 기본값입니다. 경로가 다르면 셀 위쪽 경로만 수정합니다. 원래 가중치는 해시를 확인하고 보존합니다.
3. 기본은 YOLO60epoch + U-Net60epoch, 총120epoch를 순서대로 실행합니다. **각 epoch마다 CHASE695장 전부를 정확히 한 번씩 사용**합니다. 모든 epoch가 CHASE 전용입니다. YOLO는 full-frame, U-Net은 사진마다1024crop 한 개를 사용합니다.
4. 중간에 멈춰도 같은 셀로 마지막 완료epoch부터 재개합니다. 진행 중이던 미완료epoch는 다시 실행합니다. 예를 들어 이 새 셀에서 `RUN_UNTIL_EPOCH=30`으로 실행하고 나중에60으로 바꿀 수 있습니다. 코사인 학습률 스케줄은 처음부터60epoch로 고정됩니다. 앞서 배포한 전체30epoch 또는 epoch당400sample 실험의 체크포인트를 새 full-pass 실험으로 옮겨 자동 재개하지 않습니다.
5. 학습 후 새 YOLO의 validation91장30뷰 추론과 **기존/새 YOLO × 기존/새 U-Net 네 조합** 평가가 이어집니다. 학습이60에 끝났는데 셀이 계속 실행되면 이 검증 단계의 로그를 확인합니다.
6. 결과ZIP을 이 대화에 올리면 기존 대비 PQ·FP·FN을 검토할 수 있습니다. ZIP에는 가중치·대량 이미지를 넣지 않습니다. 가중치는 Drive에 따로 남습니다.

`PHASE='prepare'`는 CPU로 외부 자료 다운로드·원반 정렬까지만 수행합니다. `PHASE='train'`은 학습만, `PHASE='compare'`는 두 모델의 해당epoch 결과가 있을 때 네 조합만 평가합니다. 기본 `all`은 전체 작업입니다. `ARMS=['yolo']` 또는 `['native']`로 한 모델씩 실행할 수도 있습니다.

## 매epoch 전체 데이터 사용과 실제 하이퍼파라미터

- 현재 대상695장 기준으로 매epoch 전부를 한 번씩 사용합니다. 순서는 섞지만 누락·중복 없이 한 바퀴 완료해야 해당epoch가 끝납니다. seed42로 재현할 수 있습니다.
- 60epoch에서 모델당41700개 표본, 두 모델 합계83400개 표본을 사용합니다. YOLO는 원반 정렬된2048 전체 이미지, U-Net은 매사진에서1024crop 한 개를 봅니다. U-Net의 사진 전체 순회는 해당사진의 모든 픽셀·객체를 매epoch 본다는 의미는 아닙니다. crop 위치는 epoch별로 달라질 수 있습니다. 현재 확인한 메타데이터에는 겹침 제외 표시가 없습니다. 이후 명시된 제외 정보로 실제 사용 장수가 달라지면 해당 수를 기준으로 전체 순회하고 보고서에 실제 장수를 기록합니다.
- 라이브러리 기본값 그대로 돌리는 설정이 아닙니다. 아래 학습 설정은 이 실험에서 지정한 값입니다.

| 설정 | YOLO26s-seg | best84 ConvNeXt-Tiny U-Net |
|---|---|---|
| 초기 가중치 | 기존 best.pt | 기존84epoch best.pt |
| epoch / 표본 수 |60 / 매epoch695|60 / 매epoch695|
| 입력 |2048전체 이미지|1024crop|
| batch / gradient accumulation |1 / 1|1 / 4|
| optimizer |AdamW|AdamW|
| 초기 learning rate / weight decay |2e-5 / 0.01|2e-5 / 0.01|
| 학습률 스케줄 |60epoch cosine, 최저비율0.1|60epoch cosine, 최저비율0.1|
| warmup / AMP |0epoch / 사용|별도warmup없음 / FP16|
| seed / workers |42 / 0|42 / 0|
| validation |학습 후 고정epoch 평가|시작 전, 매5epoch와 종료 시|

YOLO의 photometric·기하 증강과 mosaic/mixup/copy-paste는 끕니다. CHASE 원반 정렬과 U-Net crop은 적용합니다. 세부 loss와 구조는 기존 구현을 유지하며, 이 표에 없는 모든 옵션을 튜닝했다고 주장하지 않습니다.

## 저장 위치

- 외부 원본: `MyDrive/filament-segmentation-2026/external_CHASE/ahas_2024_2025`
- 실험: `MyDrive/filament-segmentation-2026/fixed_split_experiments/chase_only_both_alignment_t4_60ep_fullpass_pngfix_v1` — PNG 준비 오류가 난 이전 실험의 코드·설정은 보존합니다.
- 로컬 ZIP 캐시: `/content/chase_only_both_alignment_60ep_fullpass_cache/download/ahas_21582245_data.zip` — 이전 버전과 같은 위치입니다.
- `yolo/`: 완료epoch 재개 상태와 고정epoch YOLO 가중치.
- `native/last.pt`: U-Net 완료epoch 전체 재개 상태. `native/best.pt`: 기존 YOLO와 조합한 validation 최고 체크포인트(초기epoch0도 후보).
- `comparison/epoch060_comparison.json`: 고정60epoch 두 모델의 네 조합 비교. 이 비교는 native best.pt를 자동 선택하지 않고 last.pt의 고정epoch를 사용합니다.
- `latest_result.json`, `training.log`, `chase_only_both_fullpass_epoch060_results.zip`: 상태·오류 로그·검토용 결과.

## 해석

- 같은 실행 조건의 기존 두 모델을 기준으로 비교합니다. 과거0.472593/0.472274라는 수치를 새 실행의 측정값처럼 고정하지 않습니다.
- 전처리·후처리·임계값은 고정합니다. 네 조합은 각 모델 교체 효과를 보여주지만, CHASE의 효과만을 학습시간 효과와 분리한 GONG-only continuation 대조 실험은 아닙니다.
- U-Net은 CHASE의 foreground만 지도학습합니다. CHASE에 없는 spine 정답을 배경으로 만들지 않습니다. 공유 encoder·decoder가 바뀌므로 spine 예측이 완전히 고정되는 것은 아닙니다.
- 원래 train_split과 정답JSON을 사용하는 부분은 validation91의 분리·정답 검증용입니다. GONG train616장의 이미지·정답으로 추가 학습하는 단계는 없습니다.
- 공개 test참조216+102의 포함률은 이번 validation 실행에서 측정하지 않습니다. 새 test추론과 리더보드 확인은 이후 단계입니다.
- 여기서는 CPU 기능·패키징을 검증했습니다. 실제 GPU 추가 학습과 새 PQ 측정은 사용자의 Colab에서 진행됩니다.
'''


def build():
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO/'configs/chase_ahas2024_2025_manifest.json', ASSET_ROOT/'chase_ahas2024_2025_manifest.json')
    sources = {name:(REPO/'tools'/name).read_text() for name in MODULES}
    for name, source in sources.items(): compile(source, name, 'exec')
    assets = {name:base64.b64encode((ASSET_ROOT/name).read_bytes()).decode() for name in ASSET_NAMES}
    bundle = dict(schema_version=1, sources=sources, assets=assets)
    raw = json.dumps(bundle, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    sha = hashlib.sha256(raw).hexdigest(); version = sha[:16]
    cell = CELL.replace('__PAYLOAD__', base64.b64encode(zlib.compress(raw, 9)).decode()).replace('__SHA__', sha).replace('__VERSION__', version)
    compile(cell, STEM+'_CopyCell.py', 'exec')
    DEST.mkdir(parents=True, exist_ok=True)
    notebook = dict(nbformat=4, nbformat_minor=5,
        metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                      language_info=dict(name='python'), colab=dict(name=STEM+'_OneCell.ipynb', provenance=[]), accelerator='GPU'),
        cells=[dict(cell_type='code', execution_count=None, id='chase-both-onecell', metadata={}, outputs=[], source=cell.splitlines(keepends=True))])
    paths = dict(notebook=DEST/(STEM+'_OneCell.ipynb'), copy_cell=DEST/(STEM+'_CopyCell.txt'),
                 readme=DEST/(STEM+'_README.ko.md'), package=DEST/(STEM+'_Package.zip'))
    paths['notebook'].write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n')
    paths['copy_cell'].write_text(cell)
    paths['readme'].write_text(README)
    manifest = dict(bundle_sha256=sha, version=version,
        source_sha256={name:hashlib.sha256(source.encode()).hexdigest() for name,source in sources.items()},
        asset_sha256={name:hashlib.sha256(base64.b64decode(value)).hexdigest() for name,value in assets.items()},
        counts=dict(chase=695, gong_train=0, gong_validation=91, epochs_each=60,
                    samples_each_epoch=695, sample_exposures_each_model=41700, total_sample_exposures=83400),
        epoch_sampling='full_pass',
        training_domain='chase_only',
        **{k:str(v) for k,v in paths.items()})
    with zipfile.ZipFile(paths['package'], 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in ('notebook', 'copy_cell', 'readme'): archive.write(paths[name], paths[name].name)
        for name, source in sources.items(): archive.writestr('code/'+name, source)
        for name, value in assets.items(): archive.writestr('assets/'+name, base64.b64decode(value))
        archive.writestr('source_manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
    manifest['package_sha256'] = hashlib.sha256(paths['package'].read_bytes()).hexdigest()
    (DEST/(STEM+'_manifest.json')).write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    for key in ('notebook','copy_cell'):
        shutil.copyfile(paths[key], REPO/'notebooks'/paths[key].name)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


if __name__ == '__main__': build()
