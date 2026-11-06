"""Build a self-contained, one-paste Colab launcher from the frozen modules."""
import base64
import hashlib
import json
import lzma
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
MODULES = ('submission_driver.py','submission_inference.py',
           'submission_yolo_recipe.py','submission_native_recipe.py','submission_native_process.py')


def build_cell():
    files = {name:(ROOT/'tools'/name).read_text(encoding='utf-8') for name in MODULES}
    raw = json.dumps(files,ensure_ascii=False,separators=(',',':')).encode()
    payload = '\n'.join(textwrap.wrap(base64.b64encode(lzma.compress(raw,preset=9)).decode(),120))
    digest = hashlib.sha256(raw).hexdigest()
    prefix = '''# Colab 새 셀 하나에 전체 복사하세요. 추가 학습 없이 제출 CSV 하나를 생성합니다.
# 중단되면 같은 A100 환경에서 같은 셀을 다시 실행하세요.
BASE = "/content/drive/MyDrive/filament-segmentation-2026"
YOLO_RUN = "/content/drive/MyDrive/SolarFilament_LimbOnly_YOLO26_mixed_FullRes"
NATIVE_RUN = "/content/drive/MyDrive/SolarFilament_NativeUNet_A100/20261028T030808Z_bedf3e21"
TEST_IMAGES = BASE + "/MAGFiLO_1.0_Kaggle_2026/test/test_images"
OUTPUT = BASE + "/submissions/frozen_pq0472593_v1"

import base64, hashlib, json, lzma, subprocess, sys
from pathlib import Path
from google.colab import drive, files
drive.mount("/content/drive")

PAYLOAD = """'''
    suffix = f'''"""
raw = lzma.decompress(base64.b64decode(PAYLOAD))
assert hashlib.sha256(raw).hexdigest() == "{digest}", "코드가 온전히 복사되지 않았습니다. 전체 셀을 다시 복사하세요."
bundle = json.loads(raw)
assert set(bundle) == set({MODULES!r})
work = Path("/content/solar_submission_frozen")
work.mkdir(exist_ok=True)
for name, source in bundle.items():
    compile(source, name, "exec")
    (work/name).write_text(source, encoding="utf-8")

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "numpy==2.1.3", "ultralytics==8.4.182", "segmentation-models-pytorch==0.5.0",
    "timm==1.0.20", "huggingface-hub==0.34.4", "scipy==1.17.0",
    "pycocotools==2.0.11", "opencv-python==5.0.0.93"])
config = dict(test_images=TEST_IMAGES, output=OUTPUT,
    yolo_checkpoint=YOLO_RUN+"/best.pt", native_checkpoint=NATIVE_RUN+"/best.pt",
    train_split=YOLO_RUN+"/train_split.csv", val_split=YOLO_RUN+"/val_split.csv",
    keep_stages=False)
config_path = work/"config.json"
config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
result_path = work/"result.json"
# 새 프로세스를 사용해 설치 전 메모리에 남은 라이브러리와 섞이지 않게 합니다.
subprocess.run([sys.executable, "-u", str(work/"submission_driver.py"),
    "--config", str(config_path), "--result", str(result_path)], check=True)
result = json.loads(result_path.read_text(encoding="utf-8"))
print("제출용 CSV:", result["csv"])
print("테스트 이미지:", result["physical_test_images"], "제출 행:", result["prediction_rows"])
files.download(result["csv"])
'''
    cell = prefix+payload+suffix
    compile(cell,'colab-submission-copy-cell','exec')
    return cell, raw


def main():
    cell,raw = build_cell()
    output = ROOT/'notebooks'
    output.mkdir(exist_ok=True)
    copy = output/'SolarFilament_Submission_CopyCell.py'
    copy.write_text(cell,encoding='utf-8')
    notebook = dict(nbformat=4,nbformat_minor=5,metadata=dict(colab=dict(name='SolarFilament_Submission_Frozen.ipynb'),
        kernelspec=dict(name='python3',display_name='Python 3')),
        cells=[dict(cell_type='markdown',id='frozen-submission-info',metadata={},source=[
            '# 최고 검증 PQ 0.472593 파이프라인 제출 CSV\n',
            'Colab A100에서 다음 셀을 실행합니다. 학습 없이 test_images 전체를 추론합니다.\n',
            '같은 환경에서 재실행하면 Drive의 이미지·단계 캐시를 검증하고 재개합니다.\n',
            '전체 완료 후 submission_pq0472593.csv 하나를 내려받습니다. 검증 점수는 리더보드 점수가 아닙니다.']),
            dict(cell_type='code',id='frozen-submission-cell',metadata={},execution_count=None,outputs=[],source=cell.splitlines(keepends=True))])
    path = output/'SolarFilament_Submission_Frozen.ipynb'
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print(json.dumps(dict(copy_cell=str(copy),notebook=str(path),cell_bytes=len(cell.encode()),
        decoded_source_bytes=len(raw),decoded_sha256=hashlib.sha256(raw).hexdigest()),indent=2))


if __name__=='__main__':main()
