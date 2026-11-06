"""Deliver a small Colab cell plus a binary runtime ZIP, with exact old sources."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

REPO = Path(__file__).resolve().parents[1]
DEST = REPO.parent/'artifacts/downloads'
LEGACY_STEM = 'SolarFilament_CHASE_Only_BothModels_60Epoch_FullPass_FixPNG'
STEM = 'SolarFilament_CHASE_Light'
VERSION = '057b2b49a0cdf3e0'

CELL = '''# 작은 실행 셀: 원래 CHASE 학습 코드·설정·완료 에폭을 그대로 사용합니다.
RUN_UNTIL_EPOCH = 60
ARMS = ['yolo', 'native']
PHASE = 'all'
AUTO_DOWNLOAD = True

from pathlib import Path
from google.colab import drive, files
import hashlib, importlib.util, json, sys, uuid, zipfile
drive.mount('/content/drive')
BASE = Path('/content/drive/MyDrive/filament-segmentation-2026')
ROOT = BASE/'fixed_split_experiments/chase_only_both_alignment_t4_60ep_fullpass_pngfix_v1'
EXPECTED_SHA = '__ZIP_SHA__'
PACKAGE = BASE/'runtime_packages'/'chase_runtime___ZIP_VERSION__.zip'
if not PACKAGE.is_file():
    print('처음 한 번만 함께 제공된 SolarFilament_CHASE_Light_Runtime.zip을 선택하세요.')
    uploaded = files.upload()
    matched = [data for data in uploaded.values() if hashlib.sha256(data).hexdigest() == EXPECTED_SHA]
    if len(matched) != 1:
        raise ValueError('제공된 Runtime.zip 한 개를 선택해 주세요. 기존 큰 노트북 ZIP과 다릅니다.')
    PACKAGE.parent.mkdir(parents=True, exist_ok=True)
    temporary = PACKAGE.with_name(PACKAGE.name+'.'+uuid.uuid4().hex+'.pending')
    temporary.write_bytes(matched[0]); temporary.replace(PACKAGE)
    del uploaded, matched
with PACKAGE.open('rb') as stream:
    if hashlib.file_digest(stream, 'sha256').hexdigest() != EXPECTED_SHA:
        raise ValueError('Drive Runtime.zip의 체크섬이 다릅니다. 원본 파일을 보존하고 확인하세요.')
with zipfile.ZipFile(PACKAGE) as archive:
    namespace = {'__name__': 'chase_runtime_installer'}
    exec(compile(archive.read('install_runtime.py'), 'install_runtime.py', 'exec'), namespace)
launcher = namespace['install_runtime'](PACKAGE, ROOT, EXPECTED_SHA)
spec = importlib.util.spec_from_file_location('chase_light_runtime_monitor', launcher)
runtime = importlib.util.module_from_spec(spec); sys.modules[spec.name] = runtime; spec.loader.exec_module(runtime)
result = runtime.run_resume(ROOT, run_until=RUN_UNTIL_EPOCH, arms=ARMS, phase=PHASE)
if result and result.get('archive'):
    print('검토할 결과 ZIP:', result['archive'])
    if AUTO_DOWNLOAD:
        try: files.download(result['archive'])
        except Exception as exc: print('Drive에 결과 ZIP이 저장돼 있습니다:', exc)
'''

README = '''# CHASE 작은 실행 셀

큰 코드 셀을 편집하는 Chrome 페이지가 멈추는 문제를 줄이기 위한 실행 방식입니다. 약4.65MB의 압축 문자열을 셀에 넣지 않고, 작은 실행 셀이 Drive의 스크립트를 불러옵니다. 학습 코드·가중치·학습 설정은 이전 PNG 수정판과 동일합니다.

1. `SolarFilament_CHASE_Light_Runtime.zip`을 컴퓨터에 다운로드합니다. 압축을 풀 필요가 없습니다.
2. 기존 큰 셀이 들어 있는 노트북 대신 새 Colab 노트북을 열고 T4 GPU를 선택합니다.
3. 작은 `SolarFilament_CHASE_Light_OneCell.ipynb`를 열거나 `CopyCell.txt`의 셀을 복사해 실행합니다.
4. 처음 한 번 파일 선택 창이 나오면 내려받은 `Runtime.zip`을 선택합니다. Runtime ZIP은 Drive에 저장되며 다음 실행부터 다시 업로드하지 않습니다.

기존 PNG 수정판의 같은 실험 폴더를 사용하므로 완료한 학습을 초기화하지 않습니다. 원본 코드12개·분할·검증 캐시의 해시를 확인하고 기존 파일이 달라졌으면 덮어쓰지 않고 중단합니다. 큰 셀이 실행되기 전에 브라우저가 멈췄더라도 이 작은 셀로 최초 설정부터 진행할 수 있습니다.

- 학습: CHASE695장만, 두 모델 각각60epoch, 매epoch 전체사진1회. U-Net은 사진마다1024crop 하나입니다.
- 원본: 기존 `external_CHASE/ahas_2024_2025` 유지.
- 실험: 기존 `fixed_split_experiments/chase_only_both_alignment_t4_60ep_fullpass_pngfix_v1` 유지.
- 학습용12개 파일은 기존 `code/057b2b49a0cdf3e0`에 그대로 유지합니다. 화면 출력·메모리 기록용 helper는 별도 `launchers` 폴더에 둬 학습 식별자를 바꾸지 않습니다.
- 재개: 같은 작은 셀을 실행합니다. 완료epoch부터 재개하며 미완료epoch 일부는 다시 실행합니다.
- 전체 진행 로그는 Drive에 저장하고 화면 출력은 제한합니다. 연결이 끊겨도 남은 기록으로 어디서 멈췄는지 확인할 수 있습니다.
- 여기서는 기존자료·체크포인트 보존과 설치·실행기를 CPU로 검증했습니다. 실제 Colab 브라우저·GPU 학습을 이 환경에서 실행한 것은 아닙니다.
'''


def digest(value):
    return hashlib.sha256(value).hexdigest()


def build(dest=DEST, monitor_path=None):
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
    legacy = json.loads((DEST/(LEGACY_STEM+'_manifest.json')).read_text())
    if legacy['version'] != VERSION:
        raise ValueError('Expected the released PNG-fix training version')
    old_zip = Path(legacy['package'])
    if digest(old_zip.read_bytes()) != legacy['package_sha256']:
        raise ValueError('Released PNG-fix ZIP checksum differs')
    files = {}
    with zipfile.ZipFile(old_zip) as archive:
        for folder, key in (('code', 'source_sha256'), ('assets', 'asset_sha256')):
            for name, expected in legacy[key].items():
                value = archive.read(folder+'/'+name)
                if digest(value) != expected:
                    raise ValueError('Original source/asset changed: '+name)
                files[f'{folder}/{VERSION}/{name}'] = value
    original_manifest = dict(sources=legacy['source_sha256'], assets=legacy['asset_sha256'])
    files[f'code/{VERSION}/source_manifest.json'] = json.dumps(original_manifest, sort_keys=True, indent=2).encode()
    monitor_path = Path(monitor_path or REPO/'tools/chase_runtime_monitor.py')
    monitor = monitor_path.read_bytes(); compile(monitor, str(monitor_path), 'exec')
    launcher = f'launchers/{digest(monitor)[:16]}/chase_runtime_monitor.py'
    files[launcher] = monitor
    install_source = (REPO/'tools/chase_lean_install.py').read_bytes()
    compile(install_source, 'install_runtime.py', 'exec')
    manifest = dict(schema_version=1, training_version=VERSION, training_bundle_sha256=legacy['bundle_sha256'],
        launcher=launcher, files={name:digest(value) for name,value in files.items()},
        counts=legacy['counts'], changes='launcher and packaging only; original training modules unchanged')
    paths = dict(runtime_zip=dest/(STEM+'_Runtime.zip'), notebook=dest/(STEM+'_OneCell.ipynb'),
        copy_cell=dest/(STEM+'_CopyCell.txt'), readme=dest/(STEM+'_README.ko.md'))
    with zipfile.ZipFile(paths['runtime_zip'], 'w', zipfile.ZIP_DEFLATED) as archive:
        def write(name, value):
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 4, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, value)
        write('install_runtime.py', install_source)
        write('install_manifest.json', json.dumps(manifest, sort_keys=True, indent=2))
        for name, value in files.items(): write(name, value)
    zip_sha = digest(paths['runtime_zip'].read_bytes())
    cell = CELL.replace('__ZIP_SHA__', zip_sha).replace('__ZIP_VERSION__', zip_sha[:16])
    compile(cell, STEM+'_CopyCell.py', 'exec')
    notebook = dict(nbformat=4, nbformat_minor=5,
        metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
            language_info=dict(name='python'), colab=dict(name=STEM+'_OneCell.ipynb', provenance=[]), accelerator='GPU'),
        cells=[dict(cell_type='code', execution_count=None, id='chase-small-launcher', metadata={}, outputs=[], source=cell.splitlines(keepends=True))])
    paths['copy_cell'].write_text(cell)
    paths['notebook'].write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n')
    paths['readme'].write_text(README)
    report = dict(manifest, runtime_zip_sha256=zip_sha,
        runtime_zip_bytes=paths['runtime_zip'].stat().st_size, cell_bytes=len(cell.encode()),
        **{key:str(path) for key,path in paths.items()})
    (dest/(STEM+'_manifest.json')).write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('training_version','runtime_zip_bytes','cell_bytes','runtime_zip_sha256')}, indent=2))
    return report


if __name__ == '__main__': build()
