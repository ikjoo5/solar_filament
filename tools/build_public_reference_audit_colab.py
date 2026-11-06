"""Build a CPU-only reference comparison cell, separate from active inference."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import zipfile
import zlib

REPO = Path(__file__).resolve().parents[1]
DEST = REPO.parent / 'artifacts/downloads'
PUBLIC = REPO.parent / 'attachments/d03f8d87-a646-4f04-89dd-4ad6ab534cec/submission.csv'
NAMES = REPO.parent / 'attachments/14e0ea8c-23c5-4cd5-9344-a783251831af/official_image_names.json'
MODULES = ('public_reference_audit.py', 'public_reference_stage_trace.py',
           'filament_eval.py', 'submission_yolo_recipe.py', 'submission_native_recipe.py')
STEM = 'SolarFilament_Public055_ReferenceAudit'

CELL = '''# 현재 best84 + R1 추론이 180장 모두 완료된 다음에 실행하세요. GPU 불필요.
# 0.55 공개 예측을 비교 자료로 사용하며, 기존 모델/제출 CSV를 수정하지 않습니다.
RUN_ROOT = '/content/drive/MyDrive/filament-segmentation-2026/submissions/best84_pipeline_r1_t4_v1'
AUDIT_ROOT = '/content/drive/MyDrive/filament-segmentation-2026/diagnostics/public055_reference_audit_v1'
AUTO_DOWNLOAD = True

from pathlib import Path
from google.colab import drive, files
from datetime import datetime, timezone
from collections import deque
import base64, hashlib, importlib.util, json, os, signal, subprocess, sys, uuid, zipfile, zlib
drive.mount('/content/drive')
run_root = Path(RUN_ROOT)
state_path = run_root/'result.json'
if not state_path.is_file() or json.loads(state_path.read_text()).get('status') != 'complete':
    raise RuntimeError('현재 추론이 아직 완료되지 않았습니다. 기존 추론은 그대로 끝까지 실행하고 이 진단 셀을 나중에 실행하세요.')
audit_root = Path(AUDIT_ROOT)
a, b = audit_root.resolve(), run_root.resolve()
if a == b or a in b.parents or b in a.parents:
    raise ValueError('AUDIT_ROOT은 기존 추론 폴더와 겹치지 않는 별도 폴더여야 합니다.')
payload = """__PAYLOAD__"""
raw = zlib.decompress(base64.b64decode(payload))
if hashlib.sha256(raw).hexdigest() != '__SHA__':
    raise ValueError('복사된 셀 내용이 완전하지 않습니다.')
bundle = json.loads(raw)
version = '__VERSION__'
code = audit_root/'code'/version
assets = audit_root/'assets'/version
def save_exact(path, data):
    if path.exists():
        if path.read_bytes() != data: raise ValueError('저장된 진단 코드/참조 파일과 다릅니다: '+str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.'+uuid.uuid4().hex+'.pending')
    tmp.write_bytes(data); tmp.replace(path)
for name, source in bundle['sources'].items():
    if Path(name).name != name: raise ValueError('잘못된 파일명')
    compile(source, name, 'exec')
    save_exact(code/name, source.encode())
for name, content in bundle['assets'].items():
    if Path(name).name != name: raise ValueError('잘못된 파일명')
    save_exact(assets/name, base64.b64decode(content))
save_exact(code/'source_manifest.json', json.dumps({name:hashlib.sha256(source.encode()).hexdigest()
    for name,source in bundle['sources'].items()},sort_keys=True,indent=2).encode())
# 기존 torch/CUDA 패키지는 설치하거나 변경하지 않습니다.
missing = [package for module, package in [('numpy','numpy'), ('scipy','scipy'),
    ('cv2','opencv-python-headless'), ('pycocotools','pycocotools')]
    if importlib.util.find_spec(module) is None]
if missing:
    subprocess.run([sys.executable,'-m','pip','install',*missing],check=True)
output = audit_root/('review_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:6])
log_path = audit_root/('audit_'+output.name+'.log')
command = [sys.executable,'-u',str(code/'public_reference_audit.py'),
    '--reference-csv',str(assets/'reference_public055.csv'),
    '--cache-root',str(run_root),'--output-root',str(output),
    '--expected-names',str(assets/'official_test_names.json')]
tail = deque(maxlen=50)
print('CPU 진단 시작. 기존 추론 캐시/가중치/제출 파일은 읽기만 합니다.',flush=True)
print('오류가 나면 마지막 traceback과 로그를 전달하세요:',log_path,flush=True)
with log_path.open('w',encoding='utf8',buffering=1) as log:
    process = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
        text=True,bufsize=1,start_new_session=True)
    try:
        for line in process.stdout:
            print(line,end='',flush=True); log.write(line); tail.append(line)
        status = process.wait()
        if status:
            print(''.join(tail))
            raise RuntimeError('진단 종료 코드 '+str(status)+'; 로그: '+str(log_path))
    finally:
        try: os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError: pass
        try: process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL); process.wait()
        process.stdout.close()
if not (output/'summary.json').is_file():
    raise FileNotFoundError('진단 요약이 생성되지 않았습니다: '+str(output))
archive = audit_root/(output.name+'.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as target:
    for path in sorted(output.rglob('*')):
        if path.is_file(): target.write(path,str(path.relative_to(output)))
    target.write(log_path,'execution.log')
    for name in bundle['sources']: target.write(code/name,'code/'+name)
    target.write(code/'source_manifest.json','code/source_manifest.json')
print('완료:',archive)
print('이 ZIP은 진단 결과입니다. Kaggle 제출 파일이 아닙니다. 대화에 첨부해 주세요.')
if AUTO_DOWNLOAD: files.download(str(archive))
'''


def main():
    sources = {name: (REPO/'tools'/name).read_text() for name in MODULES}
    for name, source in sources.items():
        compile(source, name, 'exec')
    public = PUBLIC.read_bytes()
    assert hashlib.sha256(public).hexdigest() == '0ab3932122a563ba96f4d243934d5c00680dd49b9d1165912a4f180907d2c603'
    assets = {'reference_public055.csv': base64.b64encode(public).decode(),
              'official_test_names.json': base64.b64encode(NAMES.read_bytes()).decode()}
    bundle = json.dumps(dict(sources=sources,assets=assets),sort_keys=True,separators=(',',':')).encode()
    digest = hashlib.sha256(bundle).hexdigest()
    cell = CELL.replace('__PAYLOAD__',base64.b64encode(zlib.compress(bundle,9)).decode()).replace('__SHA__',digest).replace('__VERSION__',digest[:16])
    compile(cell,'colab_reference_audit','exec')
    DEST.mkdir(parents=True,exist_ok=True)
    copy = DEST/(STEM+'_CopyCell.txt'); copy.write_text(cell)
    notebook = dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3')),
        cells=[dict(cell_type='markdown',metadata={},source=['# Public 0.55 예측 참조 진단\n',
            '기존 best84 + R1의 전체 추론이 끝난 다음 실행합니다. CPU로 저장된 후보와 후처리를 비교합니다.\n',
            '실제 test PQ를 계산하거나 제출 CSV를 수정하는 셀이 아닙니다.\n']),
            dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=cell.splitlines(keepends=True))])
    nb = DEST/(STEM+'.ipynb'); nb.write_text(json.dumps(notebook,ensure_ascii=False,indent=1))
    package = DEST/(STEM+'_Package.zip')
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as archive:
        archive.write(copy,copy.name); archive.write(nb,nb.name)
        archive.write(REPO/'docs/PUBLIC055_REFERENCE_EXPERIMENTS.ko.md','README.ko.md')
        for name in MODULES: archive.write(REPO/'tools'/name,'tools/'+name)
        archive.writestr('assets/reference_public055.csv',public)
        archive.write(NAMES,'assets/official_test_names.json')
    result = dict(bundle_sha256=digest,source_sha256={n:hashlib.sha256(s.encode()).hexdigest() for n,s in sources.items()},
                  copy_cell=str(copy),notebook=str(nb),package=str(package),package_bytes=package.stat().st_size,
                  package_sha256=hashlib.sha256(package.read_bytes()).hexdigest())
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
