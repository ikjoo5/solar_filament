"""Build a small, download-free Colab delivery for independent model tests."""
from __future__ import annotations
import base64
import gzip
import hashlib
import json
from pathlib import Path
import textwrap

REPO = Path(__file__).resolve().parents[1]
DEST = REPO.parent/'artifacts/downloads'
STEM = 'SolarFilament_CHASE_Independent'
NAMES = ('chase_independent_inference.py', 'chase_independent_decode.py',
         'chase_independent_launcher.py')

HEADER = '''# 독립 추론: 기존/CHASE YOLO와 U-Net × 원본/limb 입력. 추가학습은 하지 않습니다.
from google.colab import drive, files
drive.mount('/content/drive')
from pathlib import Path
import base64, gzip, hashlib, importlib.util, json, sys, uuid
ROOT = Path('/content/drive/MyDrive/filament-segmentation-2026/fixed_split_experiments/chase_only_both_alignment_t4_60ep_fullpass_pngfix_v1')
BASE = ROOT/'independent_models_v1'
'''

INSTALL = '''
raw = gzip.decompress(base64.b64decode(PAYLOAD))
if hashlib.sha256(raw).hexdigest() != BUNDLE_SHA:
    raise ValueError('복사한 실행 코드가 불완전합니다. 셀 전체를 다시 복사하세요.')
bundle = json.loads(raw)
expected = bundle['sha256']
required = {'chase_independent_inference.py', 'chase_independent_decode.py', 'chase_independent_launcher.py'}
if set(bundle['files']) != required or set(expected) != required:
    raise ValueError('독립 검증 파일 구성이 다릅니다.')
CODE = BASE/'code'/BUNDLE_SHA[:16]
for name, source in bundle['files'].items():
    value = source.encode('utf-8')
    if hashlib.sha256(value).hexdigest() != expected[name]:
        raise ValueError('검증 코드 해시가 다릅니다: '+name)
    compile(source, name, 'exec')
    path = CODE/name
    if path.exists() and path.read_bytes() != value:
        raise ValueError('저장된 독립 검증 코드를 보존합니다. 변경된 파일: '+str(path))
manifest = CODE/'source_manifest.json'
if manifest.exists() and json.loads(manifest.read_text()) != expected:
    raise ValueError('독립 검증 코드 목록이 변경됐습니다.')
if {p.name for p in CODE.glob('*.py')} - required:
    raise ValueError('독립 검증 폴더에 다른 Python 파일이 있습니다.')
CODE.mkdir(parents=True, exist_ok=True)
for name, source in bundle['files'].items():
    path = CODE/name
    if not path.exists():
        pending = path.with_name(path.name+'.'+uuid.uuid4().hex+'.pending')
        pending.write_text(source, encoding='utf-8'); pending.replace(path)
if not manifest.exists():
    pending = manifest.with_suffix('.pending')
    pending.write_text(json.dumps(expected, sort_keys=True, indent=2)); pending.replace(manifest)
index = BASE/'current_code.json'
pending = index.with_suffix('.pending')
pending.write_text(json.dumps(dict(version=BUNDLE_SHA[:16], bundle_sha256=BUNDLE_SHA, files=expected)))
pending.replace(index)
del raw, bundle, PAYLOAD
'''

RUN = '''
spec = importlib.util.spec_from_file_location('chase_independent_launcher', CODE/'chase_independent_launcher.py')
launcher = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = launcher; spec.loader.exec_module(launcher)
result = launcher.run_experiment(ROOT, CODE, expected)
for archive in result['archives']:
    print('이 채팅에 올릴 결과 ZIP:', archive)
    try: files.download(archive)
    except Exception as exc: print('Drive에 결과가 저장돼 있습니다:', archive, str(exc))
'''

README = '''# 독립 모델 검증

- 기존 best84 복합 파이프라인과 학습 파일을 보존합니다.
- 기존 YOLO / CHASE YOLO / 기존 U-Net / CHASE U-Net을 각각 별도 프로세스에서 실행합니다.
- 각 가중치는 원본 밝기 입력과 limb 보정 입력을 같은 런타임에서 새로 추론합니다. 총8조건이며 추가 에폭은 없습니다.
- 단일 base 뷰만 사용합니다. YOLO와 U-Net의 예측을 서로 전달하지 않습니다.
- YOLO: 2048 입력, 기존 모델 내부 NMS, confidence0.2, confidence 순 픽셀 소유권.
- U-Net: 기존1024타일·25%겹침·입력 정규화, foreground0.5·8연결요소·면적100이상.
- 후처리가 문자 그대로 없는 raw tensor 비교가 아니라, 같은 최소 디코더를 적용한 독립 비교입니다.

새 Colab 노트북에서 T4를 선택하고 CopyCell 전체를 한 셀에 복사해 실행하세요. ZIP 다운로드·업로드나 새 데이터 다운로드는 필요 없습니다. 이전 CHASE 실행의 Drive 코드·원본 이미지·네 가중치를 사용합니다.

중단 뒤에는 같은 셀 또는 ResumeCell을 실행합니다. 완료한 이미지의 캐시는 그대로 사용합니다. 가중치·입력·코드·런타임 조건이 바뀌면 다른 식별자로 구분합니다.

결과는 기존 실험 폴더 아래 independent_models_v1에 저장합니다. 각 역할의 role_results.json, 입력별 metrics.json/predictions.csv, 이미지별 원시 예측 캐시가 남습니다. 마지막에24MiB 이하 ZIP으로 나눠 내려받습니다. 생성된 모든 part를 채팅에 올리면 클라우드에서 분석할 수 있습니다.

이 환경에서는 코드/캐시/포장 동작을 CPU로 검사했습니다. 사용자 Colab에서의 GPU 추론 결과는 아직 없습니다. 기존 개발 검증91장을 반복 사용하는 실험이며 독립 OOF나 리더보드 점수가 아닙니다.
'''


def build(dest=DEST, *, source_dir=None):
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
    source_dir = Path(source_dir or REPO/'tools')
    files = {name:(source_dir/name).read_text(encoding='utf-8') for name in NAMES}
    for name, source in files.items(): compile(source, name, 'exec')
    hashes = {name:hashlib.sha256(source.encode()).hexdigest() for name, source in files.items()}
    raw = json.dumps(dict(files=files, sha256=hashes), ensure_ascii=False,
                     sort_keys=True, separators=(',', ':')).encode()
    digest = hashlib.sha256(raw).hexdigest()
    payload = base64.b64encode(gzip.compress(raw, compresslevel=9, mtime=0)).decode()
    packed = 'PAYLOAD = (\n'+''.join('    '+repr(line)+'\n' for line in textwrap.wrap(payload, 100))+')\n'
    cell = HEADER+'BUNDLE_SHA = '+repr(digest)+'\n'+packed+INSTALL+RUN
    resume = HEADER+'BUNDLE_SHA = '+repr(digest)+'\nexpected = '+repr(hashes)+'''
index = json.loads((BASE/'current_code.json').read_text())
if index != dict(version=BUNDLE_SHA[:16], bundle_sha256=BUNDLE_SHA, files=expected):
    raise ValueError('이 재개 셀과 저장된 코드 버전이 다릅니다. 해당 최초 실행 셀을 사용하세요.')
CODE = BASE/'code'/BUNDLE_SHA[:16]
if json.loads((CODE/'source_manifest.json').read_text()) != expected:
    raise ValueError('저장된 검증 코드 목록이 변경됐습니다.')
for name, sha in expected.items():
    if hashlib.sha256((CODE/name).read_bytes()).hexdigest() != sha:
        raise ValueError('저장된 검증 코드가 변경됐습니다: '+name)
'''+RUN
    for name, source in [('CopyCell', cell), ('ResumeCell', resume)]:
        compile(source, name, 'exec')
        (dest/(STEM+'_'+name+'.txt')).write_text(source, encoding='utf-8')
    if len(cell.encode()) > 30*1024:
        raise ValueError('Colab cell exceeded the compact delivery budget')
    notebook = dict(nbformat=4, nbformat_minor=5,
        metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                      accelerator='GPU'),
        cells=[dict(cell_type='code', id='independent-inference', execution_count=None,
                    metadata={}, outputs=[], source=cell.splitlines(keepends=True))])
    notebook_path = dest/(STEM+'_OneCell.ipynb')
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n')
    (dest/(STEM+'_README.ko.md')).write_text(README, encoding='utf-8')
    report = dict(version=digest[:16], bundle_sha256=digest, source_sha256=hashes,
                  raw_source_bytes=len(raw), cell_bytes=len(cell.encode()),
                  resume_bytes=len(resume.encode()), sources=list(files),
                  copy_cell=str(dest/(STEM+'_CopyCell.txt')),
                  resume_cell=str(dest/(STEM+'_ResumeCell.txt')), notebook=str(notebook_path),
                  independent_model_roles=4, input_conditions_each=2, training=False)
    (dest/(STEM+'_manifest.json')).write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    return report


if __name__ == '__main__':
    print(json.dumps(build(), ensure_ascii=False, indent=2))
