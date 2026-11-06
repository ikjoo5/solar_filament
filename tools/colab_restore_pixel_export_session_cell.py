# Fresh Colab session: restore only what the pixel-tile exporter needs.
# Run this cell, then rerun the previous tile-export cell. CPU runtime is enough.
import subprocess
import sys
from google.colab import drive

drive.mount('/content/drive')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
    'numpy==2.1.3', 'pycocotools==2.0.11', 'scipy==1.17.0',
    'opencv-python==5.0.0.93'], check=True)

SCRIPT = r'''
from pathlib import Path
import numpy as np
from scipy.interpolate import UnivariateSpline

DATA = Path('/content/drive/MyDrive/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026')
RUN = Path('/content/drive/MyDrive/SolarFilament_LimbOnly_YOLO26_mixed_FullRes')
IMAGES = DATA / 'train/train_images'
ANN = DATA / 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'
BEST, TRAIN, VAL = RUN/'best.pt', RUN/'train_split.csv', RUN/'val_split.csv'

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
'''
compile(SCRIPT, 'restored-pixel-config', 'exec')
print('복구 완료: 이제 이전 타일 추출 셀을 실행하세요. GPU는 필요하지 않습니다.')
