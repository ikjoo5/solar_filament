"""Package the existing raw-stage audit plus frozen216/102 summaries in one cell."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import zipfile
import zlib

import build_public_reference_audit_colab as original
from reference_panel_audit import load_panel

STEM = 'SolarFilament_Reference216_102_Audit'
EXTRA = '''panel_command = [sys.executable,'-u',str(code/'reference_panel_audit.py'),
    '--audit-csv',str(output/'reference_instances.csv'),
    '--audit-summary',str(output/'summary.json'),
    '--panel216',str(assets/'reference216.json'),
    '--panel102',str(assets/'reference102.json'),
    '--output-root',str(output/'fixed_panels')]
subprocess.run(panel_command,check=True)
'''
README = '''# 216개·102개 고정 참조 진단

- 기존 best84 + R1의 180장 test 추론이 끝난 뒤, 단일 셀을 실행합니다.
- 기본 RUN_ROOT는 /content/drive/MyDrive/filament-segmentation-2026/submissions/best84_pipeline_r1_t4_v1 입니다.
- GPU 추론/학습 없이 Drive의 원시 후보와 중간 후처리 캐시를 읽습니다.
- 모델 가중치와 제출 CSV를 수정하지 않습니다.
- raw / baseline / R1의 공개 예측과의 IoU>0.5 일치 개수를 216개와102개로 나눠 출력합니다.
- 공개 예측은 정답이 아닙니다. 실제 test PQ, recall, FP, FN을 계산하는 코드가 아닙니다.
- 완료 시 review_*.zip을 자동 다운로드합니다. ZIP을 대화에 첨부해 주세요.
- 세션이 바뀌어도 Drive의 추론 캐시가 남아 있으면 이 셀 하나로 실행할 수 있습니다.
- 코드와 참조 CSV는 셀에 포함돼 있으므로 GitHub/별도 경로에서 스크립트를 받지 않습니다.
'''


def main():
    repo, dest = original.REPO, original.DEST
    names = (*original.MODULES, 'reference_panel_audit.py')
    sources = {name: (repo/'tools'/name).read_text() for name in names}
    for name, source in sources.items():
        compile(source, name, 'exec')
    public = original.PUBLIC.read_bytes()
    if hashlib.sha256(public).hexdigest() != '0ab3932122a563ba96f4d243934d5c00680dd49b9d1165912a4f180907d2c603':
        raise ValueError('Reference source changed')
    assets = {'reference_public055.csv': public, 'official_test_names.json': original.NAMES.read_bytes()}
    for count in (216, 102):
        path = repo/'configs'/f'reference{count}.json'
        load_panel(path, count)
        assets[path.name] = path.read_bytes()
    bundle = json.dumps(dict(sources=sources, assets={name: base64.b64encode(data).decode()
                         for name, data in assets.items()}), sort_keys=True, separators=(',', ':')).encode()
    digest = hashlib.sha256(bundle).hexdigest()
    cell = original.CELL.replace("archive = audit_root/(output.name+'.zip')", EXTRA+"archive = audit_root/(output.name+'.zip')")
    if EXTRA not in cell:
        raise ValueError('Original audit packaging seam changed')
    cell = cell.replace('public055_reference_audit_v1', 'reference216_102_audit_v1')
    cell = cell.replace('__PAYLOAD__', base64.b64encode(zlib.compress(bundle, 9)).decode()).replace('__SHA__', digest).replace('__VERSION__', digest[:16])
    compile(cell, 'reference_panels_colab', 'exec')
    dest.mkdir(parents=True, exist_ok=True)
    copy = dest/(STEM+'_CopyCell.txt'); copy.write_text(cell)
    notebook = dict(nbformat=4, nbformat_minor=5,
        metadata=dict(kernelspec=dict(name='python3', display_name='Python 3')),
        cells=[dict(cell_type='markdown', metadata={}, source=README.splitlines(keepends=True)),
               dict(cell_type='code', execution_count=None, metadata={}, outputs=[], source=cell.splitlines(keepends=True))])
    nb = dest/(STEM+'.ipynb'); nb.write_text(json.dumps(notebook, ensure_ascii=False, indent=1))
    package = dest/(STEM+'_Package.zip')
    with zipfile.ZipFile(package, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.write(copy, copy.name); archive.write(nb, nb.name)
        archive.writestr('README.ko.md', README)
        for name in names:
            archive.write(repo/'tools'/name, 'tools/'+name)
        for name, data in assets.items():
            archive.writestr('assets/'+name, data)
    result = dict(bundle_sha256=digest, copy_cell=str(copy), notebook=str(nb), package=str(package),
                  artifact_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (copy, nb, package)})
    (dest/(STEM+'_manifest.json')).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
