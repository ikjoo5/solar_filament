"""Build a single CPU Colab cell for exact candidate gate tracing."""
import base64
import hashlib
import json
from pathlib import Path
import zipfile
import zlib

import build_public_reference_audit_colab as old

REPO=old.REPO
STEM='SolarFilament_CandidateGateTrace'
MODULES=(*old.MODULES,'reference_panel_audit.py','candidate_gate_trace.py')


def main():
    cell=old.CELL.replace('public055_reference_audit_v1','candidate_gate_trace_v1')
    cell=cell.replace("str(code/'public_reference_audit.py')","str(code/'candidate_gate_trace.py')")
    cell=cell.replace("'--expected-names',str(assets/'official_test_names.json')]",
        "'--expected-names',str(assets/'official_test_names.json'),\n    '--panel216',str(assets/'reference216.json'),'--panel102',str(assets/'reference102.json')]")
    cell=cell.replace('# 현재 best84 + R1 추론이 180장 모두 완료된 다음에 실행하세요. GPU 불필요.',
        '# 후보별 실제 탈락 조건을 추적합니다. 기존 test180장 캐시를 사용하므로 GPU/재학습이 필요 없습니다.')
    cell=cell.replace("print('CPU 진단 시작. 기존 추론 캐시/가중치/제출 파일은 읽기만 합니다.',flush=True)",
        "print('CPU 후보 게이트 추적 시작. 원시 YOLO 후보와 실제 처리 조건을 저장합니다.',flush=True)\nprint('기존 추론 캐시/가중치/제출 파일은 읽기만 합니다.',flush=True)")
    sources={name:(REPO/'tools'/name).read_text() for name in MODULES}
    for name,source in sources.items():compile(source,name,'exec')
    public=old.PUBLIC.read_bytes()
    if hashlib.sha256(public).hexdigest()!='0ab3932122a563ba96f4d243934d5c00680dd49b9d1165912a4f180907d2c603':raise ValueError('Public source changed')
    assets={'reference_public055.csv':base64.b64encode(public).decode(),
        'official_test_names.json':base64.b64encode(old.NAMES.read_bytes()).decode()}
    for name in ('reference216.json','reference102.json'):
        assets[name]=base64.b64encode((REPO/'configs'/name).read_bytes()).decode()
    raw=json.dumps(dict(sources=sources,assets=assets),sort_keys=True,separators=(',',':')).encode()
    digest=hashlib.sha256(raw).hexdigest()
    cell=cell.replace('__PAYLOAD__',base64.b64encode(zlib.compress(raw,9)).decode()).replace('__SHA__',digest).replace('__VERSION__',digest[:16])
    compile(cell,STEM,'exec')
    old.DEST.mkdir(parents=True,exist_ok=True)
    copy=old.DEST/(STEM+'_CopyCell.txt');copy.write_text(cell)
    notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3')),
        cells=[dict(cell_type='markdown',metadata={},source=[
            '# 후보 탈락 조건 추적\n','기존 test180장 캐시를 CPU로 분석합니다. 새 학습/추론이나 제출 변경이 없습니다.\n',
            '전처리 rescue의 실제 confidence·면적·중복 조건과 최종 native 삭제 조건을 재현합니다.\n',
            '픽셀 투표로 변형되는 geometry 후보는 투표 자격과 면적 필터만 진단하며, 단일 후보의 전체 경로를 단정하지 않습니다.\n']),
            dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=cell.splitlines(keepends=True))])
    nb=old.DEST/(STEM+'.ipynb');nb.write_text(json.dumps(notebook,ensure_ascii=False,indent=1))
    package=old.DEST/(STEM+'_Package.zip')
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as archive:
        archive.write(copy,copy.name);archive.write(nb,nb.name)
        for name,source in sources.items():archive.writestr('tools/'+name,source)
        for name,data in assets.items():archive.writestr('assets/'+name,base64.b64decode(data))
    result=dict(bundle_sha256=digest,copy_cell=str(copy),notebook=str(nb),package=str(package),
        package_bytes=package.stat().st_size,package_sha256=hashlib.sha256(package.read_bytes()).hexdigest(),
        source_sha256={name:hashlib.sha256(source.encode()).hexdigest() for name,source in sources.items()})
    (old.DEST/(STEM+'_manifest.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
