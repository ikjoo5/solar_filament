"""Build an inspectable, small Drive-only recovery section for the fixed run.

The prior full-pass cell was already run before its PNG shape error. Its Drive
sources/assets provide the large immutable payload. Only the audited PNG decoder
and coordinate mapping differ, so an exact checked line delta suffices.
"""
from __future__ import annotations

from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import zipfile

DEST = Path('/workspace/artifacts/downloads')
STEM = 'SolarFilament_CHASE_Only_BothModels_60Epoch_FullPass'
OLD_VERSION = 'ff1af6fc182be562'
VERSION = '057b2b49a0cdf3e0'
OLD_ROOT = 'chase_only_both_alignment_t4_60ep_fullpass_v1'
ROOT_NAME = 'chase_only_both_alignment_t4_60ep_fullpass_pngfix_v1'

RECOVERY = r'''
def recover_sources(base):
    root = Path(base)/'fixed_split_experiments'/ROOT_NAME
    old = root.parent/OLD_ROOT
    code, assets = root/'code'/VERSION, root/'assets'/VERSION
    pending = {}
    for folder, hashes in EXPECTED.items():
        for name, expected in hashes.items():
            target = root/folder/VERSION/name
            if target.is_file():
                value = target.read_bytes()
            else:
                previous = old/folder/OLD_VERSION/name
                if not previous.is_file():
                    raise FileNotFoundError('Drive에 복구할 코드/자료가 없습니다: '+str(previous))
                value = previous.read_bytes()
                old_sha = OLD_DATA_SHA if name == 'chase_transfer_data.py' else expected
                if hashlib.sha256(value).hexdigest() != old_sha:
                    raise ValueError('이전 코드/자료가 수정되어 자동 복구를 중단합니다: '+str(previous))
                if name == 'chase_transfer_data.py':
                    lines = value.decode('utf-8').splitlines(keepends=True)
                    for start, end, replacement in reversed(DATA_PATCHES):
                        lines[start:end] = [replacement]
                    value = ''.join(lines).encode('utf-8')
                pending[target] = value
            if hashlib.sha256(value).hexdigest() != expected:
                raise ValueError('저장된 코드/자료의 해시가 다릅니다. 덮어쓰지 않습니다: '+str(target))
    source_manifest = json.dumps(dict(sources=EXPECTED['code'], assets=EXPECTED['assets']),
                                 sort_keys=True, indent=2).encode()
    manifest_path = code/'source_manifest.json'
    if manifest_path.exists() and manifest_path.read_bytes() != source_manifest:
        raise ValueError('기존 source_manifest.json이 다릅니다. 원본을 보존합니다.')
    if {p.name for p in code.glob('*.py')} - set(EXPECTED['code']):
        raise ValueError('기존 코드 폴더에 예상하지 못한 Python 파일이 있습니다.')
    if not manifest_path.exists():
        pending[manifest_path] = source_manifest
    # Verify every source before writing anything. Never move old checkpoints.
    for path, value in pending.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name+'.recovery-'+str(os.getpid())+'.pending')
        temporary.write_bytes(value)
        if path.exists():
            temporary.unlink()
            if path.read_bytes() != value:
                raise ValueError('다른 실행이 파일을 변경했습니다: '+str(path))
        else:
            temporary.replace(path)
    print('Drive 코드·자료 검증 완료:', root, flush=True)
    return root, code, assets
'''


def render_recovery_section() -> str:
    manifests = [json.loads((DEST/(STEM+suffix+'_manifest.json')).read_text())
                 for suffix in ('', '_FixPNG')]
    old, new = manifests
    assert old['version'] == OLD_VERSION and new['version'] == VERSION
    assert old['asset_sha256'] == new['asset_sha256']
    changed = {name for name, sha in new['source_sha256'].items()
               if old['source_sha256'].get(name) != sha}
    assert changed == {'chase_transfer_data.py'}, changed
    with zipfile.ZipFile(DEST/(STEM+'_Package.zip')) as source:
        before = source.read('code/chase_transfer_data.py')
    with zipfile.ZipFile(DEST/(STEM+'_FixPNG_Package.zip')) as source:
        after = source.read('code/chase_transfer_data.py')
    assert hashlib.sha256(before).hexdigest() == old['source_sha256']['chase_transfer_data.py']
    assert hashlib.sha256(after).hexdigest() == new['source_sha256']['chase_transfer_data.py']
    a, b = before.decode().splitlines(True), after.decode().splitlines(True)
    patches = [(i,j,''.join(b[k:l])) for tag,i,j,k,l in SequenceMatcher(None,a,b,autojunk=False).get_opcodes()
               if tag != 'equal']
    probe = list(a)
    for i,j,replacement in reversed(patches): probe[i:j] = [replacement]
    assert ''.join(probe).encode() == after
    expected = dict(code=new['source_sha256'], assets=new['asset_sha256'])
    header = '# Drive에 저장된 이전 코드를 검증하고 PNG 수정판을 복구합니다.\n'
    header += 'from pathlib import Path\nimport hashlib, json, os\n'
    for key, value in dict(OLD_ROOT=OLD_ROOT, ROOT_NAME=ROOT_NAME, OLD_VERSION=OLD_VERSION,
                           VERSION=VERSION, EXPECTED=expected,
                           OLD_DATA_SHA=old['source_sha256']['chase_transfer_data.py'],
                           DATA_PATCHES=patches).items():
        header += key+' = '+repr(value)+'\n'
    result = header+RECOVERY
    compile(result, 'chase_nozip_recovery.py', 'exec')
    return result


def build(output=DEST, launcher=None):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    launcher = Path(launcher or Path(__file__).with_name('chase_inline_resume.py'))
    section = render_recovery_section()
    helper = launcher.read_text(encoding='utf-8')
    cell = '''# 다운로드 없이 기존 Drive 코드로 CHASE 두 모델을 각각60에폭까지 재개합니다.
RUN_UNTIL_EPOCH = 60
from google.colab import drive
drive.mount('/content/drive')
'''+section+'\n'+helper+'''
# 시작: 설치되어 있는 이전 코드/자료를 검증한 뒤 학습을 재개합니다.
BASE = Path('/content/drive/MyDrive/filament-segmentation-2026')
ROOT, CODE, ASSETS = recover_sources(BASE)
result = run_inline(ROOT, run_until=RUN_UNTIL_EPOCH)
'''
    compile(cell, 'CHASE_NoDownload.py', 'exec')
    stem = 'SolarFilament_CHASE_NoDownload'
    text_path = output/(stem+'_CopyCell.txt'); text_path.write_text(cell, encoding='utf-8')
    notebook = dict(nbformat=4, nbformat_minor=5,
        metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
                      accelerator='GPU'),
        cells=[dict(cell_type='code',id='chase-no-download',execution_count=None,metadata={},outputs=[],
                    source=cell.splitlines(keepends=True))])
    notebook_path = output/(stem+'_OneCell.ipynb')
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
    review = dict(copy_cell=str(text_path), notebook=str(notebook_path), cell_bytes=len(cell.encode()),
                  recovery_bytes=len(section.encode()), launcher_bytes=len(helper.encode()),
                  training_version=VERSION, fallback_version=OLD_VERSION,
                  no_download=True, no_embedded_assets=True,
                  cell_sha256=hashlib.sha256(cell.encode()).hexdigest())
    (output/(stem+'_manifest.json')).write_text(json.dumps(review, ensure_ascii=False, indent=2)+'\n')
    return review


if __name__ == '__main__':
    section = render_recovery_section()
    path = DEST/'SolarFilament_CHASE_NoDownload_RecoverySection.txt'
    path.write_text(section, encoding='utf-8')
    print(json.dumps(build()))
