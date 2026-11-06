# Colab 새 셀 하나에 전체 복사: 추가 학습 없이 best84 추론만 실행합니다.
# 중단되면 같은 셀을 다시 실행하세요. 검증된 완료 이미지부터 건너뜁니다.
# 수치 라이브러리는 설치 후 새 Python 프로세스에서 불러옵니다.
EXPORT_SOURCE = r'''import csv, hashlib, importlib.util, json, subprocess, sys, zipfile
from pathlib import Path

RUN = Path('/content/drive/MyDrive/SolarFilament_NativeUNet_A100/20261028T030808Z_bedf3e21')
SOURCE_SHA = '6d60b5705fb0fbde03a6fd2922c8b83839fa49b5a78cc3434e7dcf4a1b4a08b4'
PINS = {
    'annotations': '5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1',
    'train_split': 'aa0733424815731c5c108ada91e45525eb17ca3a42b377bf19e3089d86aede7e',
    'val_split': '54c6a9aac5a3b70b3ecc6721109a8b0fe8ec020e90cb56e3848fccc92eadc61f',
}
FG_THRESHOLDS, SPINE_THRESHOLDS = [.2, .35, .5, .65, .8], [.3, .5, .7]
MIN_AREA, BATCH = 100, 4
VIEWS = [('base', ()), ('hflip', (-1,)), ('vflip', (-2,)), ('hvflip', (-2, -1))]

import cv2
import numpy as np
import torch
from pycocotools import mask as mu
from scipy import ndimage


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def atomic_text(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.pending')
    temporary.write_text(value, encoding='utf-8')
    temporary.replace(path)


def rle(mask):
    encoded = mu.encode(np.asfortranarray(mask.astype(np.uint8)))
    return dict(size=list(encoded['size']), counts=encoded['counts'].decode('ascii'))


@torch.inference_mode()
def predict_heads(source, model, image, device, tile, overlap, batch, amp_dtype):
    """Same tile order, sigmoid, Hann weights and FP32 accumulation as training."""
    model.eval()
    height, width = image.shape
    coordinates = [(y, x) for y in source.tile_positions(height, tile, overlap)
                   for x in source.tile_positions(width, tile, overlap)]
    weight = source.blend_window(tile)
    numerator = np.zeros((2, height, width), np.float32)
    denominator = np.zeros((height, width), np.float32)
    for start in range(0, len(coordinates), batch):
        selected = coordinates[start:start + batch]
        inputs = torch.from_numpy(np.stack([source.model_input(image[y:y+tile, x:x+tile])
                                           for y, x in selected])).to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype or torch.float32,
                            enabled=amp_dtype is not None):
            output = model(inputs)
        if tuple(output.shape) != (len(selected), 2, tile, tile):
            raise ValueError(f'Native two-head output required: {tuple(output.shape)}')
        probabilities = output.float().sigmoid().cpu().numpy()
        for (y, x), probability in zip(selected, probabilities):
            numerator[:, y:y+tile, x:x+tile] += probability * weight
            denominator[y:y+tile, x:x+tile] += weight
    if np.any(denominator <= 0):
        raise ValueError('Tile stitching left uncovered pixels')
    result = numerator / denominator
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite prediction probabilities')
    return result


def predict_views(source, model, image, device, tile, overlap, batch, amp_dtype, progress=None):
    base, total, unions = None, None, {}
    for view, axes in VIEWS:
        transformed = np.ascontiguousarray(np.flip(image, axes)) if axes else image
        heads = predict_heads(source, model, transformed, device, tile, overlap, batch, amp_dtype)
        restored = np.flip(heads, axes) if axes else heads
        if base is None:
            base, total = restored, restored.copy()
        else:
            total += restored
        unions[view] = rle(restored[0] >= .5)
        if progress is not None:
            progress(view)
    return base, total / np.float32(len(VIEWS)), unions


def prediction_record(heads, name, image_sha, identity, min_area=MIN_AREA):
    foreground, spine = heads
    result = dict(schema_version=1, identity=identity, file_name=name,
                  image_sha256=image_sha, size=list(foreground.shape), unions={}, components={})
    for head_name, probability, thresholds in (
            ('foreground', foreground, FG_THRESHOLDS), ('spine', spine, SPINE_THRESHOLDS)):
        result['unions'][head_name] = {f'{t:g}': rle(probability >= t) for t in thresholds}
    for threshold in FG_THRESHOLDS:
        labels, count = ndimage.label(foreground >= threshold, structure=np.ones((3, 3), np.uint8))
        areas = np.bincount(labels.ravel())
        components = []
        for label, region in enumerate(ndimage.find_objects(labels), 1):
            if region is None or areas[label] < min_area:
                continue
            local = labels[region] == label
            full = np.zeros_like(foreground, dtype=np.uint8)
            full[region] = local
            stats = {}
            for head_name, probability in [('foreground', foreground), ('spine', spine)]:
                values = probability[region][local]
                stats[head_name] = dict(mean=float(values.mean()), max=float(values.max()),
                                       p90=float(np.percentile(values, 90)))
            components.append(dict(rle=rle(full), area=int(areas[label]),
                bbox=[region[1].start, region[0].start,
                      region[1].stop-region[1].start, region[0].stop-region[0].start], stats=stats))
        result['components'][f'{threshold:g}'] = components
    return result


def verified_record(path, entry, identity, name, image_sha):
    if not entry or not path.is_file() or entry.get('image_sha256') != image_sha:
        return None
    if sha(path) != entry.get('record_sha256'):
        return None
    try:
        record = json.loads(path.read_text(encoding='utf-8'))
    except (ValueError, OSError):
        return None
    if (record.get('identity'), record.get('file_name'), record.get('image_sha256')) != (
            identity, name, image_sha):
        return None
    return record


def main():
    if not torch.cuda.is_available():
        raise ValueError('Colab 런타임에서 GPU를 선택하세요. A100 권장, 학습은 실행하지 않습니다.')
    source_path, checkpoint_path = RUN/'run_source.py', RUN/'best.pt'
    if sha(source_path) != SOURCE_SHA:
        raise ValueError('run_source.py differs from the reviewed training source')
    spec = importlib.util.spec_from_file_location('verified_native_export_source', source_path)
    source = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = source
    spec.loader.exec_module(source)  # Hash-verified definitions; __main__ training is not invoked.
    config = json.loads((RUN/'config.json').read_text())
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    saved = checkpoint['config']
    if config.get('run_source_sha256') != SOURCE_SHA or saved.get('run_source_sha256') != SOURCE_SHA:
        raise ValueError('Checkpoint/config source hash does not match the reviewed run')
    for key, expected in [('encoder', 'tu-convnext_tiny'), ('crop', 1024),
                          ('overlap', .25), ('preprocessing', 'limb_only')]:
        if config.get(key) != expected or saved.get(key) != expected:
            raise ValueError(f'Unexpected checkpoint/config: {key}')
    for key, expected in PINS.items():
        if sha(Path(config[key])) != expected or saved['source_sha256'].get(key) != expected:
            raise ValueError(f'Source/split hash mismatch: {key}')
    if checkpoint['epoch'] != 84 or abs(checkpoint['validation']['pq']-.4299305777121099) > 1e-12:
        raise ValueError('Expected the uploaded run best84 checkpoint')
    with Path(config['val_split']).open(newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    names = sorted({row['file_name'] for row in rows})
    if len(rows) != 101 or len(names) != 91 or any(Path(n).name != n for n in names):
        raise ValueError('Expected unchanged 91 physical validation images /101 groups')
    provenance = dict(schema_version=1, exporter_sha256=sha(Path(__file__)),
        source_sha256=SOURCE_SHA, checkpoint_sha256=sha(checkpoint_path),
        epoch=84, source_hashes=PINS, crop=1024, overlap=.25, batch=BATCH,
        foreground_thresholds=FG_THRESHOLDS, spine_thresholds=SPINE_THRESHOLDS, min_area=MIN_AREA,
        amp='bfloat16' if torch.cuda.is_bf16_supported() else 'float16', torch=str(torch.__version__),
        gpu=torch.cuda.get_device_name(), preprocessing='limb_only',
        tta=dict(views=[view for view, axes in VIEWS], inverse='native spatial flip', fusion='mean probability'))
    identity = hashlib.sha256(compact(provenance).encode()).hexdigest()
    output = RUN/'rich_predictions_best84_tta_v2'
    output.mkdir(exist_ok=True)
    manifest_path = output/'manifest.json'
    manifest = (json.loads(manifest_path.read_text()) if manifest_path.is_file()
                else dict(identity=identity, provenance=provenance, completed={}))
    if manifest['identity'] != identity:
        raise ValueError('Export configuration/GPU/weights changed; keep the same runtime type to resume.')
    records, pending = {}, []
    for name in names:
        image_sha = sha(Path(config['images'])/name)
        record_path = output/(Path(name).stem+'.jsonl')
        record = verified_record(record_path, manifest['completed'].get(name), identity, name, image_sha)
        if record is not None:
            records[name] = record
        else:
            pending.append((name, image_sha, record_path))
    print(f'완료 {len(records)}/91장: 남은 {len(pending)}장만 추론합니다.', flush=True)
    if pending:
        device = torch.device('cuda')
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = source.build_model(config['encoder'], pretrained=False).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        store = source.ImageStore(Path(config['images']), Path('/content/native_export_cache'), 'limb_only')
        for name, image_sha, record_path in pending:
            heads, averaged, view_unions = predict_views(source, model, store.load(name), device,
                1024, .25, BATCH, amp_dtype, lambda view: print(f'추론 {name}: {view}', flush=True))
            record = prediction_record(heads, name, image_sha, identity)
            record['tta'] = prediction_record(averaged, name, image_sha, identity)
            record['view_unions'] = view_unions
            atomic_text(record_path, compact(record)+'\n')
            manifest['completed'][name] = dict(record_sha256=sha(record_path), image_sha256=image_sha)
            atomic_text(manifest_path, compact(manifest)+'\n')
            records[name] = record
            print(f'저장 {len(records)}/91: {name}', flush=True)
    merged = output/'predictions.jsonl'
    atomic_text(merged, ''.join(compact(records[name])+'\n' for name in names))
    summary = dict(status='complete', physical_images=91, best_epoch=84,
        training_or_optimizer_updates=0, checkpoint_sha256=provenance['checkpoint_sha256'],
        native_size=[2048, 2048], no_ground_truth_masks_used=True,
        purpose='Base and four-view meanTTA foreground/spine RLE and confidence for local postprocessing')
    atomic_text(output/'summary.json', compact(summary)+'\n')
    archive = RUN/'best84_rich_predictions_tta.zip'
    temporary = archive.with_suffix('.pending.zip')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for path in [manifest_path, merged, output/'summary.json']:
            z.write(path, path.name)
    temporary.replace(archive)
    print(f'완료 ZIP: {archive} ({archive.stat().st_size/1024**2:.2f} MiB)', flush=True)
    return str(archive)


if __name__ == '__main__':
    Path('/content/native_export_result.json').write_text(json.dumps(dict(archive=main())))
'''

if __name__ == '__main__':
    import json, subprocess, sys
    from pathlib import Path
    from google.colab import drive, files
    drive.mount('/content/drive')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
        'numpy==2.1.3', 'segmentation-models-pytorch==0.5.0', 'timm==1.0.20',
        'huggingface-hub==0.34.4', 'pycocotools==2.0.11', 'scipy==1.17.0',
        'opencv-python-headless==5.0.0.93'])
    worker = Path('/content/native_export_worker.py')
    worker.write_text(EXPORT_SOURCE, encoding='utf-8')
    subprocess.run([sys.executable, '-u', str(worker)], check=True)
    result = json.loads(Path('/content/native_export_result.json').read_text())
    files.download(result['archive'])
