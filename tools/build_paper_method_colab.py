"""Build the fixed-best84/context-only experiment and a short transition cell."""
import base64
import hashlib
import json
import lzma
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
MODULES = ('paper_method_training.py', 'paper_multiscale.py', 'colab_setup_external_magfilo.py', 'paper_context_only.py')
BASE_RUNTIME_SHA = '6d60b5705fb0fbde03a6fd2922c8b83839fa49b5a78cc3434e7dcf4a1b4a08b4'


def build_cell():
    files = {name: (ROOT/'tools'/name).read_text(encoding='utf-8') for name in MODULES}
    raw = json.dumps(files, ensure_ascii=False, separators=(',', ':')).encode()
    payload = '\n'.join(textwrap.wrap(base64.b64encode(lzma.compress(raw, preset=9)).decode(), 120))
    source_sha = hashlib.sha256(raw).hexdigest()
    prefix = '''# Colab T4 GPU의 새 셀 하나에 전체 복사하세요.
# 기존 best84는 고정하고 multiscale_context만 총 100epoch 학습합니다.
# 중단 후 같은 셀을 실행하면 새 모델의 마지막 완료 epoch + optimizer 상태에서 재개합니다.
RUN_UNTIL_EPOCH = 20  # 우선 20epoch에서 저장·종료. 이후 40, 60, 100으로 올려 이어갑니다.
BASE = "/content/drive/MyDrive/filament-segmentation-2026"
YOLO_RUN = "/content/drive/MyDrive/SolarFilament_LimbOnly_YOLO26_mixed_FullRes"
NATIVE_RUN = "/content/drive/MyDrive/SolarFilament_NativeUNet_A100/20261028T030808Z_bedf3e21"
OUTPUT = BASE + "/paper_method_experiments/context_only_t4_v1"

import base64, hashlib, importlib.util, json, lzma, subprocess, sys
from pathlib import Path
from google.colab import drive, files
drive.mount("/content/drive")

PAYLOAD = """'''
    suffix = f'''"""
raw = lzma.decompress(base64.b64decode(PAYLOAD))
assert hashlib.sha256(raw).hexdigest() == "{source_sha}", "전체 셀을 다시 복사하세요: 코드 검증 실패"
bundle = json.loads(raw)
assert set(bundle) == set({MODULES!r})
work = Path("/content/solar_paper_methods")
work.mkdir(exist_ok=True)
for name, source in bundle.items():
    compile(source, name, "exec")
    (work/name).write_text(source, encoding="utf-8")
native_source = Path(NATIVE_RUN)/"run_source.py"
native_raw = native_source.read_bytes()
assert hashlib.sha256(native_raw).hexdigest() == "{BASE_RUNTIME_SHA}", "기존 best84 학습 소스가 확인했던 파일과 다릅니다."
(work/"base_runtime.py").write_bytes(native_raw)
assert (Path(NATIVE_RUN)/"best.pt").is_file(), "기존 best84 체크포인트를 찾지 못했습니다."

# 외부 데이터는 평가 전용 폴더에 보관합니다. 외부 이미지 다운로드/학습은 수행하지 않습니다.
spec = importlib.util.spec_from_file_location("external_folder_setup", work/"colab_setup_external_magfilo.py")
external_setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(external_setup)
print(json.dumps(external_setup.setup_external(base=BASE), ensure_ascii=False, indent=2))

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "numpy==2.1.3", "segmentation-models-pytorch==0.5.0", "timm==1.0.20",
    "huggingface-hub==0.34.4", "scipy==1.17.0", "pycocotools==2.0.11", "opencv-python==5.0.0.93"])
config = dict(
    images=BASE+"/MAGFiLO_1.0_Kaggle_2026/train/train_images",
    annotations=BASE+"/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json",
    train_split=YOLO_RUN+"/train_split.csv", val_split=YOLO_RUN+"/val_split.csv",
    synthetic_images=BASE+"/synthetic_copy_paste_v2/images",
    synthetic_annotations=BASE+"/synthetic_copy_paste_v2/synthetic_annotations.json",
    synthetic_split=BASE+"/synthetic_copy_paste_v2/split_info.json",
    initial_checkpoint=NATIVE_RUN+"/best.pt", base_runtime=str(work/"base_runtime.py"),
    external_root=BASE+"/external_MAGFiLO", data_role="competition_training",
    output_root=OUTPUT, cache_root="/content/filament_a100_cache", device="cuda",
    # methods는 기존 체크포인트 식별값을 유지합니다. 실제 학습은 execution_methods만 실행합니다.
    methods=["control", "multiscale_context"], execution_methods=["multiscale_context"],
    run_until_epoch=RUN_UNTIL_EPOCH, evaluate_baseline_current_device=True,
    paper_context_hidden=128, paper_context_rates=[1,3,5],
    encoder="tu-convnext_tiny", pretrained=False, preprocessing="limb_only",
    crop=1024, batch=1, accumulate=4, steps_per_epoch=400, epochs=100,
    validate_every=5, lr=1e-5, spine_weight=0.1, seed=42, workers=2, infer_batch=1,
    synthetic_ratio=0.2, warmup_epochs=3, real_finish_epochs=6, include_empty_synthetic=True,
    thresholds=[0.5], min_areas=[300], overlap=0.25,
    expected_source_sha256={{
        "annotations":"5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1",
        "train_split":"aa0733424815731c5c108ada91e45525eb17ca3a42b377bf19e3089d86aede7e",
        "val_split":"54c6a9aac5a3b70b3ecc6721109a8b0fe8ec020e90cb56e3848fccc92eadc61f",
        "synthetic_annotations":"5e7ad5f20fa12f0c820a82a315ba60e802480d54b4b3ecebd9fba25ae8fc6aea",
        "synthetic_split":"e8c9f3848c7b1fe1283167d973e459a5df7cd6336cc1d871268a34024d1ea881"}})
config_path = work/"experiment_config.json"
config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
print("기존 best84는 학습하지 않습니다. 새 블록 모델만 총 100epoch 학습/재개합니다.")
print("T4: batch1, 누적4, FP16. 이번 실행은", RUN_UNTIL_EPOCH, "epoch까지입니다.")
print("best84는 현재 GPU/정밀도로 한 번 재평가한 후 새 모델과 비교합니다.")
print("비교 점수는 native 단일 모델의 고정 검증 PQ이며, 기존 앙상블/LB 점수와 별개입니다.")
# 설치 전 커널에 남은 라이브러리와 섞이지 않도록 새 프로세스에서 실행합니다.
result_path = work/"result.json"
subprocess.run([sys.executable, "-u", str(work/"paper_context_only.py"),
    "--config", str(config_path), "--result", str(result_path)], check=True)
result = json.loads(result_path.read_text(encoding="utf-8"))
print(json.dumps(result["comparison"], ensure_ascii=False, indent=2))
print("체크포인트와 재개 기록:", result["output"])
print("업로드할 결과 ZIP:", result["archive"])
files.download(result["archive"])
'''
    cell = prefix+payload+suffix
    compile(cell, 'colab-paper-methods-copy-cell', 'exec')
    return cell, raw


def build_transition_cell():
    """Restore the user's already-copied/saved runtime; embed only the new driver."""
    raw=(ROOT/'tools/paper_context_only.py').read_bytes()
    payload='\n'.join(textwrap.wrap(base64.b64encode(lzma.compress(raw,preset=9)).decode(),120))
    prefix='''# 이전 학습 셀을 멈추고 Colab GPU를 T4로 선택한 뒤 전체를 실행하세요.
# best84는 고정. 우선 20epoch까지 새 블록만 학습하며, 같은 셀로 재개합니다.
RUN_UNTIL_EPOCH=20  # 이후 40, 60, 100으로 올리면 이어서 학습합니다.
import base64, hashlib, json, lzma, subprocess, sys
from pathlib import Path
from google.colab import drive, files
drive.mount('/content/drive')
BASE=Path('/content/drive/MyDrive/filament-segmentation-2026')
NATIVE=Path('/content/drive/MyDrive/SolarFilament_NativeUNet_A100/20261028T030808Z_bedf3e21')
OUTPUT=BASE/'paper_method_experiments/context_only_t4_v1'
PREVIOUS=BASE/'paper_method_experiments/multiscale_context_v1'
OLD=Path('/content/solar_paper_methods')
WORK=Path('/content/solar_paper_context_only')
WORK.mkdir(exist_ok=True)
# 기존 셀의 코드가 메모리에서 사라졌어도 Drive에 저장된 학습 소스로 복원합니다.
sources=[WORK,OLD]
for root in (OUTPUT,PREVIOUS):
    for name in ('context_only_progress.json','experiment_progress.json'):
        journal=root/name
        if journal.is_file():
            info=json.loads(journal.read_text())
            for method in ('multiscale_context','control'):
                saved=info.get('runs',{}).get(method,{}).get('path')
                if saved:sources.append(Path(saved))
config_source=next((folder/'experiment_config.json' for folder in sources
                    if (folder/'experiment_config.json').is_file()),None)
if config_source is None:
    config_source=next((folder/'config.json' for folder in sources
                        if (folder/'config.json').is_file()),None)
if config_source is None:
    raise RuntimeError('이전 학습 설정이 없습니다. 전체 새 Colab 셀을 사용해야 합니다.')
config=json.loads(config_source.read_text())
expected={
    'paper_method_training.py':'70a2c5a4293bcd2770f7e657f16d71a7eaaa08216862357cfa22734ab4715cf1',
    'paper_multiscale.py':'15cf2002b3dabcc53736c46815d0b8a868630b14c78cf85442f2f96d6041ee5c',
    'base_runtime.py':'6d60b5705fb0fbde03a6fd2922c8b83839fa49b5a78cc3434e7dcf4a1b4a08b4'}
for name,sha in expected.items():
    candidates=[folder/name for folder in sources]
    if name=='base_runtime.py':candidates.append(NATIVE/'run_source.py')
    source=next((path for path in candidates if path.is_file()
                 and hashlib.sha256(path.read_bytes()).hexdigest()==sha),None)
    if source is None:raise RuntimeError('원본 학습 소스를 복원하지 못했습니다: '+name)
    (WORK/name).write_bytes(source.read_bytes())
PAYLOAD="""'''
    suffix=f'''"""
raw=lzma.decompress(base64.b64decode(PAYLOAD))
assert hashlib.sha256(raw).hexdigest()=='{hashlib.sha256(raw).hexdigest()}', '전체 셀을 다시 복사하세요.'
compile(raw,'paper_context_only.py','exec')
(WORK/'paper_context_only.py').write_bytes(raw)
config.update(output_root=str(OUTPUT),base_runtime=str(WORK/'base_runtime.py'),
    initial_checkpoint=str(NATIVE/'best.pt'),resume_dir=None,
    execution_methods=['multiscale_context'],run_until_epoch=RUN_UNTIL_EPOCH,
    evaluate_baseline_current_device=True,
    batch=1,accumulate=4,steps_per_epoch=400,infer_batch=1,device='cuda')
assert config['epochs']==100, '기존 100epoch 스케줄을 유지해야 합니다.'
config_path=WORK/'experiment_config.json'
config_path.write_text(json.dumps(config,ensure_ascii=False,indent=2))
subprocess.check_call([sys.executable,'-m','pip','install','-q',
    'numpy==2.1.3','segmentation-models-pytorch==0.5.0','timm==1.0.20',
    'huggingface-hub==0.34.4','scipy==1.17.0','pycocotools==2.0.11','opencv-python==5.0.0.93'])
print('기존 모델 추가 학습은 건너뜁니다. best84 고정 기준선 vs 새 블록 모델만 총 100epoch.')
print('T4 배치1·누적4. 이번 실행은',RUN_UNTIL_EPOCH,'epoch까지 저장 후 종료합니다.')
print('같은 T4 실험의 체크포인트가 있으면 마지막 완료 epoch부터 이어갑니다.')
print('기존 A100 실험은 별도 폴더에 보존합니다. best84도 현재 GPU로 한 번 재평가합니다.')
result_path=WORK/'result.json'
subprocess.run([sys.executable,'-u',str(WORK/'paper_context_only.py'),
    '--config',str(config_path),'--result',str(result_path)],check=True)
result=json.loads(result_path.read_text())
print(json.dumps(result['comparison'],ensure_ascii=False,indent=2))
print('업로드할 결과 ZIP:',result['archive'])
files.download(result['archive'])
'''
    cell=prefix+payload+suffix
    compile(cell,'colab-context-only-transition','exec')
    return cell,raw


def main():
    cell, raw = build_cell()
    output = ROOT/'notebooks'
    output.mkdir(exist_ok=True)
    copy = output/'SolarFilament_PaperMethods_CopyCell.py'
    copy.write_text(cell, encoding='utf-8')
    notebook = dict(nbformat=4, nbformat_minor=5, metadata=dict(
        colab=dict(name='SolarFilament_ContextOnly_T4.ipynb'),
        kernelspec=dict(name='python3', display_name='Python 3')),
        cells=[dict(cell_type='markdown', id='paper-methods-info', metadata={}, source=[
            '# 기존 best84와 새 블록 모델 비교\n',
            '기존 best84는 고정 기준선으로 사용하고 새 문맥 블록 모델만 최대 100epoch 학습합니다.\n',
            'T4에서는 batch1·누적4·FP16으로 실행하고, 우선 20epoch에서 저장 후 종료합니다.\n',
            'RUN_UNTIL_EPOCH을 40, 60, 100으로 올리면 같은 100epoch 스케줄에서 재개합니다.\n',
            '기존 분할·합성 비율·검증 후처리를 유지합니다. 외부 MAGFiLO는 평가 전용입니다.\n',
            '중단 후 같은 셀을 실행하면 마지막 완료 epoch부터 optimizer/RNG 상태를 복원합니다.\n',
            '아직 측정된 새 성능이 없습니다. 결과 ZIP에는 기준 점수와 새 모델의 rich 검증 추론을 포함합니다.']),
            dict(cell_type='code', id='paper-methods-cell', metadata={}, execution_count=None,
                outputs=[], source=cell.splitlines(keepends=True))])
    path = output/'SolarFilament_PaperMethods_A100.ipynb'
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
    t4_path=output/'SolarFilament_ContextOnly_T4.ipynb'
    t4_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
    transition,_=build_transition_cell()
    transition_path=output/'SolarFilament_ContextOnly_SwitchCell.py'
    transition_path.write_text(transition,encoding='utf-8')
    print(json.dumps(dict(copy_cell=str(copy), notebook=str(path), cell_bytes=len(cell.encode()),
        transition_cell=str(transition_path),transition_bytes=len(transition.encode()),
        decoded_source_bytes=len(raw), decoded_sha256=hashlib.sha256(raw).hexdigest()), indent=2))


if __name__ == '__main__':
    main()
