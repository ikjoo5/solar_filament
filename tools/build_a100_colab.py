"""Build a standalone, plain-source A100 Colab notebook from reviewed modules."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def literal(filename, name):
    tree = ast.parse((ROOT/'tools'/filename).read_text())
    return next(ast.literal_eval(node.value) for node in tree.body
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == name)


def definitions(source, names):
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    return ast.unparse(ast.Module(body=nodes, type_ignores=[]))


def portable_module(filename):
    nodes = []
    for node in ast.parse((ROOT/'tools'/filename).read_text()).body:
        if isinstance(node, ast.ImportFrom) and node.module in {
                '__future__', 'train_filament_unet', 'filament_eval',
                'native_mixed_data', 'evaluate_yolo_baseline'}:
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef,
                             ast.ClassDef, ast.Assign, ast.AnnAssign)):
            nodes.append(node)
    return ast.unparse(ast.Module(body=nodes, type_ignores=[]))


VALIDATION = r'''
def validate_native(model, store, names, device, amp_dtype, cfg):
    thresholds, areas = cfg['thresholds'], cfg['min_areas']
    predictions = {(t, a): {} for t in thresholds for a in areas}
    for index, name in enumerate(names, 1):
        probability = predict_foreground(model, store.load(name), device,
            cfg.get('validation_tile', cfg['crop']), cfg['overlap'],
            cfg.get('infer_batch', cfg['batch']), amp_dtype)
        for threshold in thresholds:
            rles = [mu.encode(np.asfortranarray(mask.astype(np.uint8)))
                    for mask in iter_components(probability, threshold, min(areas))]
            for area in areas:
                predictions[threshold, area][Path(name).stem] = [
                    rle for rle in rles if int(mu.area(rle)) >= area]
        print(f'validation [{index}/{len(names)}] {name}', flush=True)
    results = []
    for (threshold, area), predicted in predictions.items():
        metric = score(predicted)
        results.append(dict(threshold=threshold, min_area=area,
            pq=metric['PQ'], sq=metric['SQ'], rq=metric['RQ'],
            tp_pairs=metric['TP'], fp=metric['FP'], fn=metric['FN']))
    return max(results, key=lambda row: row['pq']), results, predictions
'''


ENTRY = r'''
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--result', required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    result = run_training(cfg)
    Path(args.result).write_text(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
'''


CONFIG = r'''# A100: Runtime > Change runtime type > A100 GPU.
from pathlib import Path
from google.colab import drive
import json, subprocess, sys

drive.mount('/content/drive')

BASE = Path('/content/drive/MyDrive/filament-segmentation-2026')
DATA = BASE/'MAGFiLO_1.0_Kaggle_2026'
OLD_RUN = Path('/content/drive/MyDrive/SolarFilament_LimbOnly_YOLO26_mixed_FullRes')
SYNTHETIC_RATIO = 0.20  # Set 0.0 for a real-only comparison with the same budget.
RESUME_DIR = ''        # To resume, paste the printed run directory here.

CONFIG = dict(
    images=str(DATA/'train/train_images'),
    annotations=str(DATA/'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'),
    train_split=str(OLD_RUN/'train_split.csv'),
    val_split=str(OLD_RUN/'val_split.csv'),
    synthetic_images=str(BASE/'synthetic_copy_paste_v2/images'),
    synthetic_annotations=str(BASE/'synthetic_copy_paste_v2/synthetic_annotations.json'),
    synthetic_split=str(BASE/'synthetic_copy_paste_v2/split_info.json'),
    output_root='/content/drive/MyDrive/SolarFilament_NativeUNet_A100',
    cache_root='/content/solar_filament_a100_cache',
    resume_dir=RESUME_DIR or None,
    device='cuda', encoder='tu-convnext_tiny', pretrained=True,
    crop=1024, batch=2, accumulate=2, steps_per_epoch=200,
    epochs=100, validate_every=2, lr=1e-4, spine_weight=0.1,
    seed=42, workers=2, infer_batch=4,
    synthetic_ratio=SYNTHETIC_RATIO,
    warmup_epochs=3, real_finish_epochs=6, include_empty_synthetic=True,
    thresholds=[0.35, 0.5, 0.65], min_areas=[100, 200, 300], overlap=0.25)

EXPECTED = {
    'annotations':'5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1',
    'train_split':'aa0733424815731c5c108ada91e45525eb17ca3a42b377bf19e3089d86aede7e',
    'val_split':'54c6a9aac5a3b70b3ecc6721109a8b0fe8ec020e90cb56e3848fccc92eadc61f',
    'synthetic_annotations':'5e7ad5f20fa12f0c820a82a315ba60e802480d54b4b3ecebd9fba25ae8fc6aea',
    'synthetic_split':'e8c9f3848c7b1fe1283167d973e459a5df7cd6336cc1d871268a34024d1ea881'}
CONFIG['expected_source_sha256'] = {
    key:value for key,value in EXPECTED.items()
    if SYNTHETIC_RATIO > 0 or not key.startswith('synthetic_')}

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
    'numpy==2.1.3', 'segmentation-models-pytorch==0.5.0', 'timm==1.0.20',
    'huggingface-hub==0.34.4', 'pycocotools==2.0.11',
    'scipy==1.17.0', 'opencv-python==5.0.0.93'], check=True)
print('설정 완료. 다음 셀에서 학습을 시작하세요.')
'''


LAUNCH = r'''
from pathlib import Path
import json, subprocess, sys
from google.colab import files

work = Path('/content/solar_filament_a100')
work.mkdir(parents=True, exist_ok=True)
script = work/'solar_a100_train.py'
script.write_text(WORKER_SOURCE, encoding='utf-8')
config_path = work/'config.json'
config_path.write_text(json.dumps(CONFIG, ensure_ascii=False, indent=2), encoding='utf-8')
result_path = work/'result.json'
subprocess.run([sys.executable, str(script), '--config', str(config_path),
                '--result', str(result_path)], check=True)
result = json.loads(result_path.read_text())
print('체크포인트:', result['output'])
print('최고 검증 PQ:', result['summary']['best_development_pq'])
files.download(result['archive'])
'''


RECOVER = r'''# If the session expired, mount Drive and run this cell to fetch saved results.
from pathlib import Path
from google.colab import drive, files
drive.mount('/content/drive')
root = Path('/content/drive/MyDrive/SolarFilament_NativeUNet_A100')
archives = sorted(root.glob('*_native_unet_results.zip'))
assert archives, '첫 validation 저장 ZIP이 아직 없습니다.'
files.download(str(archives[-1]))
'''


def worker_source():
    core = literal('colab_train_native_unet_cell.py', 'CORE')
    old_training = literal('colab_train_native_unet_cell.py', 'TRAINING')
    prefix = literal('colab_train_native_unet_cell.py', 'PREFIX')
    minimal = literal('colab_restore_pixel_export_session_cell.py', 'SCRIPT')
    geometry = []
    for node in ast.parse(minimal).body:
        if isinstance(node, ast.Assign):
            names = {n.id for target in node.targets for n in ast.walk(target) if isinstance(n, ast.Name)}
            if names <= {'yy', 'xx', 'rr', 'disk', 'radii', 'bins'}:
                geometry.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == 'limb':
            geometry.append(node)
    source = '\n'.join([
        'from __future__ import annotations', prefix,
        ast.unparse(ast.Module(body=geometry, type_ignores=[])),
        'preprocess_limb_only = limb',
        definitions(core, {'digest', 'build_model', 'model_input', 'apply_d4', 'crop_origin',
                           'ImageStore', 'tile_positions', 'blend_window', 'predict_foreground',
                           'iter_components', 'score'}),
        definitions(old_training, {'prepare_split', 'write_csv', 'package_results'}),
        VALIDATION, portable_module('native_mixed_data.py'),
        portable_module('a100_training_engine.py'), ENTRY])
    compile(source, 'standalone-a100-training', 'exec')
    return source


def code_cell(source):
    return dict(cell_type='code', id=hashlib.sha256(source.encode()).hexdigest()[:12],
                execution_count=None, metadata={}, outputs=[],
                source=source.splitlines(keepends=True))


def markdown_cell(source):
    return dict(cell_type='markdown', id=hashlib.sha256(source.encode()).hexdigest()[:12],
                metadata={}, source=source.splitlines(keepends=True))


def main():
    worker = worker_source()
    assert "'''" not in worker
    worker_literal = "WORKER_SOURCE = r'''\n"+worker+"\n'''\n"
    title = '''# Solar Filament — A100 native U-Net + existing synthetic data

기존 split을 그대로 유지합니다: 실데이터 train616장 / validation91장.
합성462장(객체3,427개)은 train에만 사용합니다. 기본 샘플 확률은20%이며,
1–3epoch와95–100epoch는 실데이터만 사용합니다.

합성 라벨은 기존 생성기의 OpenCV rasterization을 유지하고, 없는 spine 정답의
보조 손실은 제외합니다. validation과 checkpoint 선택은 기존 공식 global PQ입니다.

**실행:** Colab에서 A100을 선택한 뒤 아래 세 코드 셀을 순서대로 실행하세요.
이 노트북은 기존 SCRIPT 없이 새 세션에서도 실행됩니다.
캐시는 /content 로컬에 두고 Drive에는 best/last만 갱신합니다.
체크포인트 약0.5GiB, atomic 저장 중 약0.9GiB가 필요합니다(직렬화 부가량 별도).
세션이 끊기면 같은 설정의 RESUME_DIR에 출력된 run 폴더를 넣어 완료 epoch부터 재개합니다.
새 모델의 전체 A100 학습/PQ는 아직 측정하지 않았습니다.
'''
    cells = [markdown_cell(title), code_cell(CONFIG),
             markdown_cell('## 2. 학습 코드 준비\n\n아래 코드는 노트북에 포함된 plain Python 소스입니다. 별도 repository 다운로드가 필요하지 않습니다.\n'),
             code_cell(worker_literal),
             markdown_cell('## 3. 학습 실행\n\n첫 epoch와 이후2epoch마다 전체91장을 검증합니다. 작은 결과 ZIP은 각 validation 후 Drive에 갱신됩니다.\n'),
             code_cell(LAUNCH),
             markdown_cell('## 세션 종료 후 저장 결과만 다운로드\n\n학습을 재개하려면 위 설정의 RESUME_DIR에 기존 run 폴더를 지정하세요. 아래 셀은 결과만 가져옵니다.\n'),
             code_cell(RECOVER)]
    notebook = dict(nbformat=4, nbformat_minor=5, cells=cells,
                    metadata=dict(accelerator='GPU', colab=dict(name='SolarFilament_A100_RealSynthetic.ipynb'),
                                  kernelspec=dict(display_name='Python 3', language='python', name='python3')))
    folder = ROOT/'notebooks'; folder.mkdir(exist_ok=True)
    (folder/'SolarFilament_A100_RealSynthetic.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=2)+'\n')
    # A readable standalone source and a one-cell copy version are provided too.
    (ROOT/'tools/solar_a100_train_standalone.py').write_text(worker+'\n')
    paste = CONFIG+'\n\n'+worker_literal+'\n'+LAUNCH
    compile(paste, 'a100-copy-cell', 'exec')
    (folder/'SolarFilament_A100_CopyCell.py').write_text(paste)
    print(json.dumps(dict(notebook=str(folder/'SolarFilament_A100_RealSynthetic.ipynb'),
                          worker_lines=len(worker.splitlines()), copy_cell=str(folder/'SolarFilament_A100_CopyCell.py'))))


if __name__ == '__main__':
    main()
