# Detection → ROI U-Net / DETR 비교 실행 안내

## 기본 실행

이번 실험은 객체 위치 검출과 객체별 분할을 분리한다. 기존 YOLO의 **실제 bbox 출력**과 새로 학습한 DETR의 bbox를 동일한 ROI U-Net으로 처리한다. 기존 segmentation mask의 외접 박스로 검출을 대신하지 않는다.

1. Colab 런타임을 T4 GPU로 선택한다.
2. `SolarFilament_DetectionROI_DETR_T4_CopyCell.py` 전체를 셀 하나에 붙여 넣고 실행하거나 `.ipynb`의 최초 셀을 사용한다.
3. 기본 `PHASE='all'`은 YOLO bbox 수집 → **DETR 30에폭** → DETR bbox 수집 → **ROI U-Net 30에폭** → 전체 영상 평가 순서다.
4. 완료 후 생성되는 **결과 ZIP의 모든 part**를 전달한다. 가중치는 Drive에 보관하며 결과 ZIP에는 넣지 않는다.

**기본 전체 학습량은 서로 다른 두 모델의 30+30=60에폭이다.** 기존 YOLO는 재학습하지 않는다. 여러 encoder나 5-fold를 자동으로 연달아 실행하지 않는다.

## 비교 모델과 예산

| 항목 | DETR 검출기 | ROI 분할기 |
|---|---|---|
| 모델 | classic DETR / ResNet50 | 2채널 U-Net / ResNet34 |
| 초기화 | COCO-pretrained `facebook/detr-resnet-50`, 새 filament 분류 head | ImageNet encoder |
| 입력 | 800×800 태양 전체 사진 | 384×384 grayscale + bbox 위치 채널 |
| 목표 | filament bbox | 지정된 객체 하나의 mask |
| 학습 데이터 | 원래 train의 주석자별 1,053개 레코드 전체 | train의 GT jitter crop + 검출 FP의 빈 mask |
| 학습 예산 | 30회 전체 레코드 순회, batch1·누적4 | 30×256 mini-batch×4=30,720개 ROI, 누적1 |
| 기본 LR | backbone 1e-6, 나머지 1e-5 | 1e-4 |

DETR optimizer 시도는 기본 에폭당 최대 264회, 전체 최대 7,920회다. ROI optimizer 시도는 에폭당 256회, 전체 최대 7,680회다. AMP 생략은 history에 기록한다. 이 서로 다른 두 모델의 epoch를 같은 샘플 수나 계산량으로 해석하지 않는다.

원래 train616 / validation91 분할을 유지한다. 외부 MAGFiLO 전체나 SWEFil를 추가 학습하지 않는다. 논문의 초기화·전처리·학습 예산을 그대로 재현한 실험은 아니다. 특히 기존 Solar YOLO는 2048 입력, DETR는 800 입력이므로 실제 파이프라인 비교이며 detector 구조만의 통제 비교가 아니다.

## ROI 방식

입력 채널은 전처리 grayscale과 원 bbox 안이 1인 prompt다. 이미지 채널만 모델 안에서 정규화한다. 정답은 선택한 객체 하나이며 주변의 다른 객체를 함께 분할하도록 주지 않는다.

기본 ROI는 bbox 중심에서 `max(width,height)×1.5` 크기의 **정사각 실제 영상 문맥**을 가져와 384px로 변환한다. 이미지 밖은 padding한다. 논문의 `3width×3height` 직사각 문맥을 검게 정사각 padding하는 방식과 구분한다. 축을 따로 늘이는 공간 왜곡은 없다.

양성 예시는 bbox 위치·크기를 흔들어 검출 오차를 반영한다. 음성은 train 검출 FP나 안전한 배경 후보이며, **같은 train 사진의 모든 주석자 전경**을 확인해 다른 주석자의 실제 filament를 빈 정답으로 만들지 않게 한다. 음성을 안전하게 만들 수 없으면 양성으로 대체한다.

## 실행 범위와 재개

상단 기본값:

```python
PHASE = 'all'
DETR_UNTIL = 30
ROI_UNTIL = 30
```

- `all`: 두 검출기의 train 후보를 혼합해 ROI를 학습하고 두 검출기를 비교한다.
- `yolo_roi`: DETR를 생략하고 YOLO 후보로 ROI만 30에폭 학습한다.
- `compare_detr`: 이미 완료한 ROI를 그대로 사용해 DETR 학습·비교만 추가한다. ROI를 재학습하거나 기존 ROI corpus를 바꾸지 않는다.
- `evaluate`: 저장된 모델과 bbox로 평가만 한다.

`all`에서 DETR의 중간 점검을 원하면 `DETR_UNTIL=10` 또는 `20`으로 실행할 수 있다. **이때 ROI는 아직 시작하지 않는다.** 기본 전체 DETR 목표 30이 완료된 뒤에야 고정된 혼합 후보로 ROI 학습을 시작하므로, 나중에 DETR를 재개해도 ROI 학습 자료가 바뀌지 않는다. `ROI_UNTIL`도 10/20/30으로 진행할 수 있다.

`yolo_roi`로 만든 ROI에 DETR를 나중에 추가할 때는 `compare_detr`를 사용한다. 같은 폴더에서 `all`로 바꾸어 혼합 corpus로 재학습하려 하면 오류로 알린다. 혼합 corpus를 처음부터 비교하려면 새 `OUTPUT`을 사용한다.

중단 뒤 같은 최초 셀이나 `SolarFilament_DetectionROI_DETR_T4_ResumeCell.py`로 재개한다. 모델·optimizer·scheduler·AMP·RNG를 마지막 완료 에폭에서 복원한다. 중단된 에폭은 다시 실행한다. 코드·설정·원본·corpus가 달라지면 조용히 섞어 재개하지 않는다.

ROI encoder는 ResNet18/34/50을 선택할 수 있다. 기본은 34다. 바꾸는 새 실험은 별도 OUTPUT을 사용한다. 첫 실행 시 일반 사전학습 가중치를 내려받는다. 기존 Colab torch/torchvision CUDA 조합은 유지한다.

## 입력과 결과 경로

```text
/content/drive/MyDrive/filament-segmentation-2026/
  MAGFiLO_1.0_Kaggle_2026/train/
    train_images/
    MAGFiLO_1.0_Annotations_kaggle2026_train.json

/content/drive/MyDrive/SolarFilament_LimbOnly_YOLO26_mixed_FullRes/
  best.pt
  train_split.csv
  val_split.csv
```

기본 결과:

```text
filament-segmentation-2026/fixed_split_experiments/detection_roi_detr_v1/
  code/<version>/
  experiment_config.json
  detr/last_full.pt
  detr/epoch030.pt
  roi/last_full.pt
  roi/epoch030.pt
  proposals/
  evaluation/
  exports/
```

원본 복사·전처리 캐시는 `/content/solar_detection_roi_cache`에 두며 세션 초기화 뒤 다시 만든다. 설정·소스·입력 해시·가중치·bbox·평가 RLE는 Drive에서 복원한다. 초기 경로·설치·GPU 오류도 화면과 Drive 로그에 표시한다.

## 결과를 읽는 방법

- **YOLO bbox → ROI U-Net**: 실제 실행 가능한 전체 영상 결과다.
- **DETR bbox → 같은 ROI U-Net**: 위와 같은 분할기로 검출기 차이를 비교한다.
- **GT bbox → ROI U-Net oracle**: 정답 위치를 줬을 때의 진단이다. 주석자별 bbox를 따로 처리하며, 실제 검출 성능이나 제출 PQ로 취급하지 않는다.

실제 검출기는 bbox confidence 0.05/0.30 × 최종 mask 최소 면적 150/300의 네 설정을 보고한다. mask threshold0.5, ROI 평균 전경 확률0.6 조건을 사용한다. bbox를 mask 면적300 조건으로 미리 버리지 않는다. ROI의 낮은 quality 후보도 원자료 캐시에 남겨, 재추론 없이 후보 선택을 더 진단할 수 있다.

bbox proposal recall과 ROI crop의 GT 포함률은 전체 영상 PQ와 구분한다. 큰 잘못된 박스가 정답 영역을 덮을 수도 있다. 최종 채택은 실제 검출 bbox를 사용한 PQ·FP·FN·작은 객체 recall로 판단한다.

## 현재 검증된 범위

- 코어·DETR·파이프라인·Colab **31개 테스트 통과**.
- 실제 tiny HF-DETR의 forward/backward 및 연속 2에폭과 1+재개의 모델 가중치 일치를 확인했다.
- 실제 제공된 YOLO 가중치, tiny HF-DETR, 실제 ResNet18 ROI U-Net을 **모킹 없이** 연결한 CPU 통합 실행을 완료했다. 작은 2048 fixture에서 학습·중간 재개·두 bbox source 평가·주석자별 oracle·평가만 재실행·결과 ZIP 저장을 확인했다.
- train 정답 512개의 리샘플링 진단에서 256/context3의 평균 왕복 IoU는0.881108, 384/context1.5는0.980295였다. 이는 모델 성능·PQ·이론적 상한이 아닌 라벨 변환 진단이다.
- **대회 원사진에서 새 DETR·ROI U-Net을 GPU로 학습하거나 새 PQ를 측정하지는 않았다.** T4 시간·최대 메모리와 성능은 Colab 실행 결과로 확인한다.

Validation은 반복 사용한 고정 개발 세트다. 상승하더라도 새 LB나 독립 OOF 수치로 표현하지 않는다. 5-fold로 이동하면 detector와 ROI 분할기 모두 해당 fold의 train 범위를 지켜야 한다.
