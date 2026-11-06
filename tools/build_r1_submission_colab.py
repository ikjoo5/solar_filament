"""Package raw-image inference through the complete YOLO+best84 pipeline and frozen R1."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil
import textwrap
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
MODULES = ('r1_test_submission.py','experimental_consensus_tails.py','submission_inference.py',
           'submission_native_process.py','native_instance_training.py','r1_submission_colab_bootstrap.py',
           'submission_yolo_recipe.py','submission_native_recipe.py','submission_driver.py')
NAMES = ROOT.parent/'attachments/14e0ea8c-23c5-4cd5-9344-a783251831af/official_image_names.json'
STEM = 'SolarFilament_Best84Pipeline_R1_T4'

CONTROLS = '''# Colab T4: 원본 test 이미지 → 기존 YOLO+best84 전체 파이프라인 → R1.
MAX_IMAGES = None          # None=공식 test180장 전체. 2로 점검 후 None으로 바꾸면 이어서 실행
AUTO_DOWNLOAD = True       # 전체 완료 시 제출CSV+분석ZIP 다운로드. Kaggle 자동 제출은 하지 않음
'''

FIRST = CONTROLS+r'''
BASE = '/content/drive/MyDrive/filament-segmentation-2026'
TEST_IMAGES = BASE + '/MAGFiLO_1.0_Kaggle_2026/test/test_images'
YOLO_CHECKPOINT = '/content/drive/MyDrive/SolarFilament_LimbOnly_YOLO26_mixed_FullRes/best.pt'
NATIVE_CHECKPOINT = '/content/drive/MyDrive/SolarFilament_NativeUNet_A100/20261028T030808Z_bedf3e21/best.pt'
OUTPUT = BASE + '/submissions/best84_pipeline_r1_t4_v1'

from google.colab import drive, files
from pathlib import Path
from datetime import datetime, timezone
from collections import deque
import base64,hashlib,json,os,re,signal,subprocess,sys,zlib
drive.mount('/content/drive')
PAYLOAD = """__PAYLOAD__"""
raw=zlib.decompress(base64.b64decode(PAYLOAD))
if hashlib.sha256(raw).hexdigest()!=__BUNDLE_SHA__:
    raise ValueError('복사된 코드가 완전하지 않습니다. 최초 셀 전체를 다시 복사하세요.')
bundle=json.loads(raw);sources=bundle['sources']
if set(sources)!=set(__MODULES__):raise ValueError('코드 목록이 올바르지 않습니다.')
assets={name:base64.b64decode(value,validate=True) for name,value in bundle['assets_b64'].items()}
if {name:hashlib.sha256(value).hexdigest() for name,value in assets.items()}!=bundle['assets_manifest']:
    raise ValueError('내장 입력 파일 checksum 불일치')
root=Path(OUTPUT)
for protected in (Path(TEST_IMAGES),Path(YOLO_CHECKPOINT).parent,Path(NATIVE_CHECKPOINT).parent):
    a=root.resolve();b=protected.resolve()
    if a==b or a in b.parents or b in a.parents:
        raise ValueError('OUTPUT은 원본 이미지·가중치 폴더와 분리된 새 폴더여야 합니다.')
for label,path in [('YOLO_CHECKPOINT',YOLO_CHECKPOINT),('NATIVE_CHECKPOINT',NATIVE_CHECKPOINT)]:
    if not Path(path).is_file():raise FileNotFoundError(label+' 파일을 찾지 못했습니다: '+path)
if not Path(TEST_IMAGES).is_dir():raise FileNotFoundError('test_images 폴더를 찾지 못했습니다: '+TEST_IMAGES)
durable=root/'code'/__VERSION__
config=dict(output_root=str(root),images=TEST_IMAGES,yolo_checkpoint=YOLO_CHECKPOINT,
            native_checkpoint=NATIVE_CHECKPOINT,
            expected_names_path=str(root/'assets/official_test_names.json'),
            device='cuda',native_batch=1,precision='fp16',expected_images=180)
manifest={name:hashlib.sha256(source.encode('utf-8')).hexdigest() for name,source in sources.items()}
def encoded(value):return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode('utf-8')
manifest_bytes=encoded(manifest);asset_manifest_bytes=encoded(bundle['assets_manifest']);config_bytes=encoded(config)
pointer=dict(scope='best84_pipeline_r1_test_v1',version=__VERSION__,
             source_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
             assets_manifest_sha256=hashlib.sha256(asset_manifest_bytes).hexdigest(),
             config_sha256=hashlib.sha256(config_bytes).hexdigest())
pointer_bytes=encoded(pointer)
if (root/'current_code.json').exists() and (root/'current_code.json').read_bytes()!=pointer_bytes:
    raise ValueError('OUTPUT에 다른 코드·입력·설정이 저장되어 있습니다. 기존 셀 설정을 복원하거나 새 OUTPUT을 사용하세요. 기존 결과를 덮어쓰지 않습니다.')
def save_immutable(path,content):
    if path.exists():
        if path.read_bytes()!=content:raise ValueError('저장된 파일과 다릅니다. 원본을 복원하거나 새 OUTPUT을 사용하세요: '+str(path))
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        pending=path.with_name(path.name+'.pending');pending.write_bytes(content);pending.replace(path)
for name,source in sources.items():
    if Path(name).name!=name:raise ValueError('Invalid source name')
    compile(source,name,'exec');save_immutable(durable/name,source.encode('utf-8'))
for name,content in assets.items():
    if Path(name).name!=name:raise ValueError('Invalid asset name')
    save_immutable(root/'assets'/name,content)
save_immutable(durable/'source_manifest.json',manifest_bytes)
save_immutable(root/'assets/assets_manifest.json',asset_manifest_bytes)
save_immutable(root/'experiment_config.json',config_bytes)
save_immutable(root/'current_code.json',pointer_bytes)
'''

RESUME = CONTROLS+r'''
# 세션 종료로 /content가 지워져도 Drive의 코드·입력·설정·완료 캐시를 복원합니다.
OUTPUT = '/content/drive/MyDrive/filament-segmentation-2026/submissions/best84_pipeline_r1_t4_v1'
from google.colab import drive, files
from pathlib import Path
from datetime import datetime, timezone
from collections import deque
import hashlib,json,os,re,signal,subprocess,sys
drive.mount('/content/drive')
root=Path(OUTPUT);pointer_path=root/'current_code.json'
if not pointer_path.is_file():raise FileNotFoundError('최초 실행 셀을 먼저 실행하세요: '+str(pointer_path))
pointer=json.loads(pointer_path.read_text())
if pointer.get('scope')!='best84_pipeline_r1_test_v1' or not re.fullmatch('[0-9a-f]{16}',pointer.get('version','')):
    raise ValueError('best84 전체 파이프라인 + R1 실험의 OUTPUT 폴더가 아닙니다.')
durable=root/'code'/pointer['version']
for filename,folder,checksum in [('source_manifest.json',durable,pointer['source_manifest_sha256']),
                               ('assets_manifest.json',root/'assets',pointer['assets_manifest_sha256'])]:
    path=folder/filename
    if hashlib.sha256(path.read_bytes()).hexdigest()!=checksum:raise ValueError('Drive 파일 목록이 변경되었습니다: '+str(path))
    for name,expected in json.loads(path.read_text()).items():
        if Path(name).name!=name or hashlib.sha256((folder/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Drive 코드/입력이 변경되었습니다: '+name)
if hashlib.sha256((root/'experiment_config.json').read_bytes()).hexdigest()!=pointer['config_sha256']:
    raise ValueError('Drive 설정이 변경되었습니다. 원래 설정을 복원하거나 새 OUTPUT을 사용하세요.')
'''

EXECUTE = r'''
if MAX_IMAGES is not None and (type(MAX_IMAGES) is not int or not 1<=MAX_IMAGES<=180):
    raise ValueError('MAX_IMAGES는 None 또는1~180 정수입니다. 에폭 수가 아닙니다.')
command=[sys.executable,'-u',str(durable/'r1_submission_colab_bootstrap.py'),'--root',str(root)]
if MAX_IMAGES is not None:command+=['--max-images',str(MAX_IMAGES)]
environment=dict(os.environ,PYTHONUNBUFFERED='1',TORCH_CUDNN_V8_API_DISABLED='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
log_path=root/'colab_bootstrap.log';tail=deque(maxlen=60)
print('전체 실행·오류 로그:',log_path,flush=True)
with log_path.open('a',encoding='utf-8',buffering=1) as log:
    log.write('\n=== '+datetime.now(timezone.utc).isoformat()+' max_images='+str(MAX_IMAGES)+' ===\n')
    process=subprocess.Popen(command,env=environment,start_new_session=True,
                             stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,errors='replace')
    try:
        for line in process.stdout:
            print(line,end='',flush=True);log.write(line);tail.append(line)
        returncode=process.wait()
        if returncode:
            print('\n===== 실제 오류 마지막 출력 =====\n'+''.join(tail),flush=True)
            raise RuntimeError('실행 종료 코드 '+str(returncode)+'. 위 traceback과 로그를 그대로 전달해주세요: '+str(log_path))
    finally:
        # Include runner/native workers in cleanup; they inherit this process group.
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            process.wait()
        process.stdout.close()
result_path=root/'latest_results.json'
if not result_path.is_file():raise FileNotFoundError('상태 파일을 찾지 못했습니다: '+str(result_path))
result=json.loads(result_path.read_text())
complete=result.get('status')=='complete' and result.get('completed_images')==180 and result.get('expected_images')==180
print('완료 이미지:',result.get('completed_images'),'/180; 상태:',result.get('status'))
if not complete:
    print('부분 실행입니다. MAX_IMAGES=None으로 같은 최초 셀 또는 재개 셀을 실행하면 이어갑니다. 제출 파일은 내려받지 않습니다.')
else:
    run_manifest=json.loads((root/'run_manifest.json').read_text())
    if (run_manifest.get('status')!='complete' or run_manifest.get('official_images')!=180
            or run_manifest.get('identity')!=result.get('identity')):
        raise ValueError('완료 상태와 실행 manifest가 일치하지 않습니다.')
    output_hashes=run_manifest['output_sha256']
    if set(output_hashes)!={'submission_pipeline_baseline.csv','submission_pipeline_R1.csv','change_audit.csv'}:
        raise ValueError('완료 파일 목록이 올바르지 않습니다.')
    for name,expected in output_hashes.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('완료 후 파일이 변경되었습니다: '+name)
    expected_paths=dict(baseline_csv=root/'submission_pipeline_baseline.csv',r1_csv=root/'submission_pipeline_R1.csv',archive=root/'results.zip')
    for key,path in expected_paths.items():
        if Path(result[key]).resolve()!=path.resolve():raise ValueError('완료 파일 경로가 올바르지 않습니다: '+key)
    if hashlib.sha256(expected_paths['archive'].read_bytes()).hexdigest()!=result['archive_sha256']:
        raise ValueError('분석 ZIP이 생성 이후 변경되었습니다.')
    print('기존 파이프라인의 과거 public LB는0.39입니다. 이번 T4 재추론의 R1 적용 전/후 리더보드 점수는 모두 아직 측정하지 않았습니다.')
    for key,label in [('baseline_csv','같은 실행의 R1 적용 전 CSV'),('r1_csv','R1 적용 후 제출 CSV'),('archive','분석 ZIP')]:
        path=Path(result[key]).resolve()
        if not path.is_file() or root.resolve() not in path.parents:raise ValueError('완료 파일 경로가 올바르지 않습니다: '+str(path))
        print(label+':',path)
        if AUTO_DOWNLOAD:
            try:files.download(str(path))
            except Exception as exc:print('자동 다운로드 실패. 위 Drive 경로에서 직접 받을 수 있습니다:',exc)
'''

GUIDE = '''# 기존 YOLO+best84 전체 파이프라인 → R1

- Colab에서 T4 GPU를 선택하고 `CopyCell.txt` 전체를 한 셀에 붙여 실행합니다. 이 노트북의 최초 셀을 실행해도 같습니다.
- 입력은 원본 test 이미지와 기존 YOLO·best84 가중치입니다. 과거 제출 CSV를 입력하지 않습니다. 셀에는 공식 test180개 파일명과 실행 코드만 포함됩니다.
- 추가 학습은 0epoch입니다. 공식 test180장에 기존 전처리, YOLO의 전체30개 추론 뷰, native 전체 타일·TTA, 기존 후처리를 적용한 뒤 R1을 한 번 연결합니다.
- 같은 실행에서 기존 파이프라인 결과와 R1 적용 결과를 모두 생성합니다. 두 결과는 같은 모델 추론 캐시를 공유하며, R1 때문에 모델을 다시 학습하거나 전체 추론을 반복하지 않습니다.
- R1은 후보들이 지지하는 기존 마스크의 누락 영역을 복원합니다. 기존 ID·객체 수·기존 픽셀을 유지하며, 객체당 면적 증가는 최대25%입니다. 완전히 새 객체를 추가하는 규칙은 아닙니다.
- `MAX_IMAGES=None`이 전체 실행입니다. `2`로 먼저 점검한 경우에도 뒤에 None으로 변경하면 완료 캐시부터 이어갑니다. 180장 전체가 완료되기 전에는 제출 CSV/ZIP을 내려받지 않습니다.
- 세션이 끊기면 같은 최초 셀 또는 `ResumeCell.txt` 중 하나를 실행하세요. Drive에 저장한 코드·설정·입력·이미지별 완료 캐시를 사용하므로 `/content`가 지워져도 복원할 수 있습니다. 완료되지 않은 이미지 단계는 다시 처리할 수 있습니다.
- 두 실행 셀을 동시에 돌리지 마세요. 코드·경로·입력·가중치·추론 환경이 달라지면 다른 조건의 캐시를 섞지 않도록 재개를 차단합니다. 원래 설정을 복원하거나 별도의 새 OUTPUT을 사용합니다.
- 기본 입력: `filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images`. 가중치 경로는 최초 셀 위쪽에 표시되어 있습니다.
- 출력: `filament-segmentation-2026/submissions/best84_pipeline_r1_t4_v1`. `submission_pipeline_baseline.csv`는 이번 실행의 R1 적용 전 결과이고, `submission_pipeline_R1.csv`는 R1 적용 후 결과입니다. 변경 내역과 해시·재개 정보는 `results.zip`에 저장됩니다.
- 전체 오류 출력은 셀과 Drive의 `colab_bootstrap.log`에 함께 남습니다. Colab 기본 torch/torchvision을 유지하고 필요한 패키지만 설치한 뒤 새 프로세스에서 추론합니다.
- 전체 완료 시 적용 전/후 CSV 두 개와 분석 ZIP을 내려받습니다. Kaggle 제출은 자동으로 하지 않습니다.
- 기존 파이프라인의 과거 public LB는0.39입니다. 이번 T4 FP16 재추론은 과거 A100 실행과 수치가 달라질 수 있습니다. 이번 적용 전/후 CSV의 test PQ와 리더보드 점수는 모두 아직 측정하지 않았습니다.
'''


def digest(content):
    return hashlib.sha256(content).hexdigest()


def encoded(value):
    return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode()


def render_cells(sources,assets):
    """Render executable first/resume cells without writing files (also usable by fixtures)."""
    if set(sources) != set(MODULES):
        raise ValueError('Source bundle must contain the exact module set')
    if set(assets) != {'official_test_names.json'}:
        raise ValueError('Asset bundle must contain only official test names; no prediction CSV inputs')
    for name,source in sources.items():
        compile(source,name,'exec')
    asset_manifest = {name:digest(content) for name,content in assets.items()}
    source_manifest = {name:digest(source.encode()) for name,source in sources.items()}
    raw = json.dumps(dict(sources=sources,assets_b64={name:base64.b64encode(content).decode() for name,content in assets.items()},
                          assets_manifest=asset_manifest),ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
    bundle_sha = digest(raw)
    payload = '\n'.join(textwrap.wrap(base64.b64encode(zlib.compress(raw,9)).decode(),120))
    cell = FIRST.replace('__PAYLOAD__',payload).replace('__BUNDLE_SHA__',repr(bundle_sha)).replace('__MODULES__',repr(MODULES)).replace('__VERSION__',repr(bundle_sha[:16]))+EXECUTE
    resume = RESUME+EXECUTE
    compile(cell,'r1-first-cell','exec');compile(resume,'r1-resume-cell','exec')
    return dict(first=cell,resume=resume,bundle_bytes=raw,bundle_sha256=bundle_sha,
                source_manifest=source_manifest,assets_manifest=asset_manifest)


def build(downloads=None,names_path=None):
    names_document = json.loads(Path(names_path or NAMES).read_text())
    names = names_document['test_images']
    if len(names)!=180 or len(set(names))!=180 or any(Path(n).name!=n for n in names):
        raise ValueError('Expected180 unique official test filenames')
    sources = {name:(ROOT/'tools'/name).read_text(encoding='utf-8') for name in MODULES}
    assets = {'official_test_names.json':encoded(names)}
    rendered = render_cells(sources,assets)
    cell,resume,bundle_sha = rendered['first'],rendered['resume'],rendered['bundle_sha256']
    source_manifest,asset_manifest = rendered['source_manifest'],rendered['assets_manifest']
    output = ROOT/'notebooks';output.mkdir(exist_ok=True)
    content_files = {STEM+'_CopyCell.py':cell,STEM+'_CopyCell.txt':cell,
                     STEM+'_ResumeCell.py':resume,STEM+'_ResumeCell.txt':resume,STEM+'_실행안내.md':GUIDE}
    notebook = dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python3',language='python'),
                    colab=dict(name=STEM+'.ipynb')),
                    cells=[dict(id='guide',cell_type='markdown',metadata={},source=GUIDE.splitlines(True)),
                           dict(id='first-run',cell_type='code',metadata={},execution_count=None,outputs=[],source=cell.splitlines(True)),
                           dict(id='resume-info',cell_type='markdown',metadata={},source=['## 세션 종료 후 재개\n','최초 셀 또는 아래 재개 셀 중 하나만 실행합니다.\n']),
                           dict(id='resume',cell_type='code',metadata={},execution_count=None,outputs=[],source=resume.splitlines(True))])
    content_files[STEM+'.ipynb'] = json.dumps(notebook,ensure_ascii=False,indent=2)+'\n'
    for name,content in content_files.items():
        (output/name).write_text(content,encoding='utf-8')
    downloads = Path(downloads) if downloads is not None else ROOT.parent/'artifacts/downloads'
    downloads.mkdir(parents=True,exist_ok=True)
    for name in content_files:
        if (output/name).resolve() != (downloads/name).resolve():shutil.copy2(output/name,downloads/name)
    metadata = dict(scope='best84_pipeline_r1_test_v1',version=bundle_sha[:16],bundle_sha256=bundle_sha,
                    source_sha256=source_manifest,assets_sha256=asset_manifest,additional_training_epochs=0,
                    historical_pipeline_public_LB=.39,fresh_baseline_public_LB=None,new_public_LB_measured=False,expected_images=180,
                    input_prediction_CSV=False,full_original_pipeline=True,same_run_comparison=True,
                    only_official_test_names_embedded=True,device='cuda',native_batch=1,precision='fp16',
                    auto_Kaggle_submission=False,T4_inference_tested_here=False,
                    copy_cell_bytes=len(cell.encode()),resume_cell_bytes=len(resume.encode()),
                    downloads={name:str(downloads/name) for name in content_files},
                    package=str(downloads/(STEM+'_Package.zip')))
    metadata_path = downloads/(STEM+'_metadata.json')
    metadata_path.write_bytes(encoded(metadata))
    with zipfile.ZipFile(metadata['package'],'w',zipfile.ZIP_DEFLATED) as archive:
        for name in content_files:archive.write(output/name,name)
        archive.write(metadata_path,metadata_path.name)
        for name,source in sources.items():archive.writestr('source/'+name,source)
        archive.writestr('source/source_manifest.json',encoded(source_manifest))
        for name,content in assets.items():archive.writestr('assets/'+name,content)
        archive.writestr('assets/assets_manifest.json',encoded(asset_manifest))
    return metadata


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--downloads',type=Path)
    args=parser.parse_args()
    print(json.dumps(build(downloads=args.downloads),ensure_ascii=False,indent=2))
