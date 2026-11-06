# 빈 Colab 셀 하나에 전체 코드를 붙여넣어 실행하세요.
import subprocess, sys, tempfile
from pathlib import Path
from google.colab import drive, files

drive.mount('/content/drive')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
    'ultralytics==8.4.182', 'pycocotools==2.0.11',
    'scipy==1.17.0', 'opencv-python==5.0.0.93'], check=True)

SCRIPT = r'''
import csv, hashlib, json, os, sys, uuid, zipfile, importlib.metadata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import cv2, numpy as np, torch
from pycocotools import mask as mu
from scipy.interpolate import UnivariateSpline
from ultralytics import YOLO
assert torch.cuda.is_available(), 'Colab 런타임에서 T4/A100 GPU를 선택하세요.'

DATA = Path('/content/drive/MyDrive/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026')
RUN = Path('/content/drive/MyDrive/SolarFilament_LimbOnly_YOLO26_mixed_FullRes')
IMAGES = DATA / 'train/train_images'
ANN = DATA / 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'
BEST, TRAIN, VAL = RUN/'best.pt', RUN/'train_split.csv', RUN/'val_split.csv'
THRESHOLDS = [.15, .25, .35, .45, .55]

def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

for path in (ANN, BEST, TRAIN, VAL):
    assert path.is_file(), f'파일 경로를 확인하세요: {path}'
assert digest(BEST) == '81c2a27d95821b5ede283169c0da8c08d360ecc4c2a60a1b9c6baf96d76bcccc', '업로드했던 best.pt와 다른 파일입니다.'
assert digest(ANN) == '5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1', '확인했던 annotation JSON과 다릅니다.'
data = json.loads(ANN.read_text())
image_map = {i['id']: i for i in data['images']}

def split_ids(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    ids = [r['image_id'] for r in rows]
    assert len(ids) == len(set(ids)), f'중복 ID: {path}'
    assert all(i in image_map for i in ids), f'알 수 없는 ID: {path}'
    assert all(image_map[r['image_id']]['file_name'] == r['file_name'] for r in rows), 'split 파일명 불일치'
    return set(ids)

train_ids, val_ids = split_ids(TRAIN), split_ids(VAL)
train_names = {image_map[i]['file_name'] for i in train_ids}
val_names = sorted({image_map[i]['file_name'] for i in val_ids})
assert val_names and not train_ids & val_ids and not train_names & set(val_names), '빈 validation 또는 train/val 중복'
assert val_ids == {i['id'] for i in data['images'] if i['file_name'] in val_names}, 'validation annotator 누락'
missing = [name for name in val_names if not (IMAGES/name).is_file()]
assert not missing, f'validation 이미지 누락: {missing[:10]}'
print(f'검증 대상: 영상 {len(val_names)}장, annotation 그룹 {len(val_ids)}개')

GT = defaultdict(list)
for ann in data['annotations']:
    if ann['image_id'] in val_ids:
        GT[ann['image_id']].append(mu.merge(mu.frPyObjects(ann['segmentation'], 2048, 2048)))
assert set(GT) == val_ids, '빈 GT 그룹을 확인하세요.'

yy, xx = np.ogrid[:2048, :2048]
rr = (np.sqrt((xx-1024)**2 + (yy-1024)**2)/900).astype(np.float32)
disk = rr <= 1
radii = (np.arange(100)+.5)/100
bins = np.minimum((rr[disk]*100).astype(np.int32), 99)

def limb(image):
    assert image is not None and image.shape == (2048, 2048) and image.dtype == np.uint8, '2048 원본 JPEG가 필요합니다.'
    hist = np.bincount(bins*256 + image[disk].astype(np.int32), minlength=25600).reshape(100, 256)
    count = hist.sum(axis=1)
    assert np.all(count > 0), '빈 radial bin'
    med = np.argmax(hist.cumsum(axis=1) >= ((count+1)//2)[:, None], axis=1).astype(float)
    spline = UnivariateSpline(radii, med, k=3, s=400)
    ref = float(np.median(spline(radii)[radii <= .15]))
    gain = np.clip(ref/np.maximum(spline(rr[disk]), 1), .25, 4)
    result = np.zeros_like(image)
    result[disk] = np.clip(image[disk].astype(np.float32)*gain, 0, 255).astype(np.uint8)
    return result

def postprocess(masks, mode):
    if mode == 'preserve':
        return [m for m in masks if int(m.sum()) >= 20]
    if not masks:
        return []
    union = np.logical_or.reduce(masks).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(union, connectivity=8)
    ids = sorted((i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 20),
                 key=lambda i: -int(stats[i, cv2.CC_STAT_AREA]))
    return [labels == i for i in ids]

def score(pred):
    tp = fp = fn = 0
    total_iou = 0.
    for group, truth in GT.items():
        predicted = pred.get(group.split('-', 1)[1], [])
        if not predicted:
            fn += len(truth)
            continue
        iou = np.asarray(mu.iou(truth, predicted, [0]*len(predicted)))
        hit = iou > .5
        tp += int(hit.sum())
        fp += int((~hit.any(axis=0)).sum())
        fn += int((~hit.any(axis=1)).sum())
        total_iou += float(iou[hit].sum())
    denominator = tp + .5*fp + .5*fn
    return dict(PQ=total_iou/denominator if denominator else 0.,
                SQ=total_iou/tp if tp else 0., RQ=tp/denominator if denominator else 0., TP=tp, FP=fp, FN=fn)

output = RUN/'official_pq_validation'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
output.mkdir(parents=True, exist_ok=False)
model = YOLO(str(BEST))  # SHA above pins this to the previously inspected checkpoint.
predictions = {(mode, t): {} for mode in ('union_cc', 'preserve') for t in THRESHOLDS}
for index, name in enumerate(val_names, 1):
    image = limb(cv2.imread(str(IMAGES/name), cv2.IMREAD_GRAYSCALE))
    result = model.predict(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), imgsz=2048,
                           conf=min(THRESHOLDS), iou=.7, retina_masks=True,
                           device=0, max_det=100, verbose=False)[0]
    raw = []
    if result.masks is not None and result.boxes is not None:
        for mask, confidence in zip(result.masks.data.float().cpu().numpy(), result.boxes.conf.cpu().numpy()):
            if mask.shape != (2048, 2048):
                mask = cv2.resize(mask, (2048, 2048), interpolation=cv2.INTER_NEAREST)
            raw.append((mask >= .5, float(confidence)))
    for (mode, t), predicted in predictions.items():
        masks = postprocess([m for m, confidence in raw if confidence >= t], mode)
        predicted[Path(name).stem] = [mu.encode(np.asfortranarray(m.astype(np.uint8))) for m in masks]
    print(f'[{index}/{len(val_names)}] {name}', flush=True)

comparisons = []
for (mode, t), predicted in predictions.items():
    metrics = score(predicted)
    name = f'{mode}_conf{t:.2f}.csv'
    with (output/name).open('x', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filament_id', 'segmentation_rle'])
        for image_id, rles in predicted.items():
            for i, rle in enumerate(rles, 1):
                writer.writerow([f'{image_id}_{i}', rle['counts'].decode('ascii')])
    comparisons.append(dict(postprocess=mode, confidence=t, **metrics, prediction_csv=name))
    print(mode, t, metrics)
summary = dict(status='complete', scope='user-supplied validation split; development tuning, not Kaggle LB reproduction',
               images=len(val_names), annotation_groups=len(GT), comparisons=comparisons,
               source_sha256={p.name: digest(p) for p in (ANN, BEST, TRAIN, VAL)},
               versions={name: importlib.metadata.version(name) for name in ['torch','ultralytics','numpy','scipy','opencv-python','pycocotools']},
               note='Preserve is diagnostic and can retain overlapping masks; verify submission policy before use.')
(output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
archive = output.parent/(output.name+'_results.zip')
with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(output.iterdir()):
        z.write(p, arcname=p.name)
print('결과 ZIP:', archive)
Path(sys.argv[1]).write_text(str(archive), encoding='utf-8')
'''

work = Path(tempfile.mkdtemp(prefix='solar_eval_'))
script = work/'evaluate.py'
script.write_text(SCRIPT, encoding='utf-8')
result_path = work/'result_zip.txt'
subprocess.run([sys.executable, str(script), str(result_path)], check=True)
files.download(result_path.read_text(encoding='utf-8'))
