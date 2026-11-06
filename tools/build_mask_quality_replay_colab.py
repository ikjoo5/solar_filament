"""Build the CPU-only single-cell exporter for frozen MaskQuality proposals."""
import base64
import hashlib
import json
from pathlib import Path
import zlib

ROOT = Path(__file__).resolve().parents[1]
CELL = r'''# CPU 전용: 추가 학습·추론 없이 검증91장의 기존 후보를 내보냅니다.
from google.colab import drive, files
from pathlib import Path
import base64, hashlib, json, subprocess, sys, zlib

drive.mount('/content/drive')
ROOT = Path('/content/drive/MyDrive/filament-segmentation-2026/fixed_split_experiments/mask_quality_v1')
OUTPUT = ROOT/'quality_replay_exports/v1'
source = zlib.decompress(base64.b64decode(__PAYLOAD__))
if hashlib.sha256(source).hexdigest() != __SHA__:
    raise ValueError('코드가 온전히 복사되지 않았습니다. 전체 셀을 다시 복사하세요.')
WORK = Path('/content/solar_mask_quality_replay')/__VERSION__
WORK.mkdir(parents=True, exist_ok=True)
script = WORK/'export_mask_quality_replay.py'
script.write_bytes(source)
print('기존 confidence와 학습한 품질 점수를 함께 시험할 후보 자료를 수집합니다.', flush=True)
print('원사진·가중치·원본 정답 JSON을 열지 않으며 학습과 모델 추론을 하지 않습니다.', flush=True)
subprocess.run([sys.executable, '-u', str(script), '--root', str(ROOT), '--output', str(OUTPUT)], check=True)
result = json.loads((OUTPUT/'result.json').read_text())
print('검증 사진:', result['validation_images'], '후보:', result['validation_proposals'],
      '품질 모델 epoch:', result['selected_epoch'], flush=True)
print('보내주실 ZIP:', result['archive'], flush=True)
try: files.download(result['archive'])
except Exception as exc: print('Drive에서 직접 다운로드할 수 있습니다:', exc, flush=True)
'''


def build():
    source = (ROOT/'tools/export_mask_quality_replay.py').read_bytes()
    compile(source, 'export_mask_quality_replay.py', 'exec')
    digest = hashlib.sha256(source).hexdigest()
    cell = (CELL.replace('__PAYLOAD__', repr(base64.b64encode(zlib.compress(source, 9)).decode('ascii')))
        .replace('__SHA__', repr(digest)).replace('__VERSION__', repr(digest[:12])))
    compile(cell, 'mask-quality-replay-cell', 'exec')
    output = ROOT/'notebooks'; output.mkdir(exist_ok=True)
    copy = output/'SolarFilament_MaskQuality_Replay_CopyCell.py'; copy.write_text(cell)
    notebook = dict(nbformat=4, nbformat_minor=5,
        metadata=dict(kernelspec=dict(name='python3', display_name='Python 3', language='python')),
        cells=[dict(id='intro', cell_type='markdown', metadata={}, source=[
            '# MaskQuality의 기존 검증 후보 수집\n',
            'CPU 런타임에서 실행할 수 있습니다. 추가 학습·추론 없이 기존91장 후보와 저장 점수만 ZIP에 담습니다.\n',
            '원본 사진과 가중치는 포함하지 않습니다. 완료 후 mask_quality_replay_validation.zip을 전달해 주세요.\n']),
            dict(id='run', cell_type='code', metadata={}, execution_count=None, outputs=[], source=cell.splitlines(True))])
    path = output/'SolarFilament_MaskQuality_Replay.ipynb'
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2)+'\n')
    return dict(copy_cell=str(copy), notebook=str(path), source_sha256=digest,
        cell_bytes=len(cell.encode()), standard_library_only=True, model_training=False, new_inference=False)


if __name__ == '__main__':
    print(json.dumps(build(), ensure_ascii=False, indent=2))
