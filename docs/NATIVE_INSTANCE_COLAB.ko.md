# 새 U-Net 모델 실험 — T4 실행 안내

## 이번에 실행할 내용

기존 최종 파이프라인과 별도로 **ResNet34 U-Net 한 모델을 실제 데이터만으로 30에폭 학습**한다. 같은 예측에서 전경 연결요소와 인스턴스 분리 디코딩을 비교한다. 새 모델의 성능은 아직 측정하지 않았다.

- 기본 모델: ImageNet encoder로 초기화한 ResNet34 U-Net.
- 학습 출력: 전경, 객체 경계, 객체별 정규화 내부 거리의 3채널.
- 손실: 전경 BCE+Dice, 경계 weighted BCE+Dice, 전경 내부 거리 SmoothL1.
- 원래 train 616장·주석 그룹 1,053개 / validation 91장·101그룹을 유지한다.
- 최초 실험은 실제 데이터만 사용한다. 가상 데이터·YOLO 후보·기존 best84 가중치를 입력으로 요구하지 않는다.
- 2048 사진을 축소하지 않고 768px crop으로 학습한다. 추론은 전체 원본에 타일을 적용한다.
- 배치 2, gradient accumulation 2, AdamW LR 0.0001, AMP를 사용한다.

**이번 epoch는 전체 사진을 한 번 통과한 단위가 아니라 crop 샘플링 예산이다.** 한 epoch에 256 mini-batch × 2 = 512개 crop, 30에폭에 총 15,360개 crop을 사용한다. 기본 optimizer update 시도는 에폭당 128회, 전체 최대 3,840회다. AMP overflow가 발생하면 실제 update 수는 줄며 history에 기록한다.

양성 crop은 약 70% 확률로 선택하고, 한 주석 그룹 내 객체를 균등하게 고른 다음 그 객체 안의 점을 중심으로 한다. 나머지는 무작위 crop이다. 사진·라벨에 동일한 회전·뒤집기를 적용한다. 같은 사진의 다른 주석자 결과는 별도 레코드이며 합치지 않는다.

## 실행 파일

1. Colab에서 **T4 GPU** 런타임을 선택한다.
2. `SolarFilament_NativeInstanceUNet_T4_CopyCell.py`를 열어 전체를 셀 하나에 붙여 넣고 실행한다. 또는 `.ipynb`의 최초 실행 셀을 사용한다.
3. Drive 연결 후 경로·메타데이터·사진 해시 검사, 패키지 설치와 GPU 검사, 학습, validation 평가, 결과 ZIP 저장 순서로 진행한다.

초기 encoder 가중치는 첫 실행 시 내려받는다. 기존 Colab PyTorch/torchvision CUDA 조합은 유지한다. 경로·설치·GPU 단계의 초기 오류도 화면과 Drive 로그로 전달한다.

기본 경로는 다음과 같다.

```text
/content/drive/MyDrive/filament-segmentation-2026/
  MAGFiLO_1.0_Kaggle_2026/train/
    train_images/
    MAGFiLO_1.0_Annotations_kaggle2026_train.json

/content/drive/MyDrive/SolarFilament_LimbOnly_YOLO26_mixed_FullRes/
  train_split.csv
  val_split.csv
```

학습 결과 기본 경로:

```text
/content/drive/MyDrive/filament-segmentation-2026/
  fixed_split_experiments/native_instance_unet_v1/resnet34_instance/
```

Drive에는 코드·설정·입력 사진의 해시·전체 재개 checkpoint·지정 에폭 가중치·평가 결과를 남긴다. 원사진 복사본과 전처리 uint8 NPY, validation probability 캐시는 `/content/solar_native_instance_cache`에만 둔다. 세션 종료로 로컬 캐시가 사라져도 원본에서 다시 만든다.

## 모델과 실행 범위 선택

최초 실행 셀 상단:

```python
ENCODER = 'resnet34'    # resnet18 / resnet34 / tu-convnext_tiny
HEAD_MODE = 'instance' # instance / semantic
RUN_UNTIL = 30
FINAL_EPOCHS = 30
PHASE = 'all'
```

- 기본 실행은 **선택한 한 모델의 30에폭**이다. 세 encoder를 연달아 학습하거나 5-fold를 실행하지 않는다.
- `HEAD_MODE='semantic'`은 같은 네트워크의 전경 손실만 사용한다. 학습하지 않은 보조 출력을 이용한 watershed 평가는 비활성화한다.
- encoder나 head mode를 바꾸면 별도 결과 폴더를 사용한다. 기존 설정을 바꿔 같은 실험에 섞어 재개하지 않는다.
- `RUN_UNTIL=10` 또는 `20`으로 중간 정지한 뒤 `30`으로 높여 재개할 수 있다. 고정 전체 스케줄은 30이다.
- `PHASE='train'`은 학습만, `'evaluate'`는 이미 완료한 지정 에폭의 평가만 한다. 기본 `'all'`은 학습과 평가를 모두 수행한다.
- 다른 학습 조건을 비교할 때는 새 `OUTPUT`을 사용한다. 처음부터 다른 epoch 예산을 정할 수 있지만, 시작한 실험의 전체 스케줄은 바꾸지 않는다.

## 평가와 재개

Instance 모델은 같은 확률 지도에 대해 **connected / watershed × 최소 면적 150 / 300**의 4개 고정 설정을 평가한다. 전경 임계값은 0.5다. Semantic 모델은 connected의 면적 150/300 두 설정만 평가한다. 최소 면적에 따라 watershed의 분리 허용 조건이 달라지므로 각 설정을 독립적으로 디코딩한다.

공식 strict IoU >0.5 기준의 PQ·SQ·RQ·FP·FN·작은 GT recall과 그룹별 결과, CSV를 저장한다. Validation은 모델 가중치 업데이트나 checkpoint 선택에 사용하지 않는다. 지정한 완료 에폭을 그대로 평가한다.

중단 후에는 동일 최초 셀 또는 `SolarFilament_NativeInstanceUNet_T4_ResumeCell.py`를 실행한다. 마지막 완료 에폭의 모델·optimizer·scheduler·AMP scaler·RNG를 복원한다. 진행 중이던 에폭은 다시 실행한다. 설정·소스·입력 픽셀·runtime이 달라지면 같은 실험으로 조용히 재개하지 않는다.

기본 재개 셀은 `resnet34_instance` 폴더를 가리킨다. 다른 모델을 골랐다면 재개 셀의 `OUTPUT`도 해당 폴더로 지정한다.

학습 후 `validation [1/91]` 등의 로그는 저장한 모델의 평가 단계다. 추가 학습 에폭이 아니다. 완료 시 생성되는 **결과 ZIP의 모든 part**를 전달하면 된다. 결과 ZIP에는 설정·history·마스크 RLE·평가 수치를 담고 가중치·원사진 전체는 넣지 않는다.

## 검증된 범위

- 모델·데이터·trainer·Colab 관련 **36개 테스트 통과**.
- 실제 ResNet18 U-Net, 실제 2048 fixture 데이터 로더, 실제 3출력 손실을 연결한 CPU 통합 학습에서 연속 2에폭과 1에폭 후 재개의 가중치 최대 차이 **0**.
- 가는 1px 객체 보존, 붙은 두 객체 분리, 작은 watershed 조각 때문에 원래 객체가 사라지는 경우의 부모 복원, 캐시 삭제 후 복원, 학습·추론 정규화 일치를 확인했다.
- 실행 파일의 압축 소스가 최종 모듈과 일치하고, 별도 폴더에서 세 모듈을 import할 수 있는지 확인했다.

**실제 대회 원사진의 새 모델 학습, T4 최대 메모리·소요 시간·새 PQ는 아직 측정하지 않았다.** 작은 CPU fixture에서의 정확한 재개 결과가 GPU bitwise 일치를 보장하지 않는다. 기존 91장은 반복 사용한 개발 세트이므로 향후 점수 상승도 새 LB나 독립 OOF 결과로 표현하지 않는다.

## 병행 실행한 저장 예측 실험

새 코드 준비와 함께 U-Net 단독 후처리 12개와 개별 뷰 합의 8개를 실행했다. 이 20개는 추가 GPU 학습 없이 계산했다.

| 구성 | 이번 검증 PQ | FP | FN |
|---|---:|---:|---:|
| 기존 YOLO+U-Net | 0.471930 | 204 | 221 |
| 과거 고정 U-Net 규칙을 현재 예측으로 재현 | 0.454038 | 210 | 247 |
| 새 개별 3뷰 medoid 합의 | 0.440157 | 260 | 241 |
| 실제 평균 TTA, foreground 0.5 | 0.431329 | 301 | 237 |

새 후처리만으로 기준선을 넘지 못했다. 과거 고정 규칙은 TTA foreground 0.5 연결요소 중 평균 전경 확률 0.9 이상·면적 300 이상만 유지한다. 과거 다른 실행에서 PQ 0.454582였으며 이번 재추론과 구분한다.

스키마도 다시 확인했다. 기존 export의 **최상위 `unions/components`는 base 출력이고, 실제 TTA 평균은 `tta.unions/components`**다. 앞선 12개 다양성 디코더는 base 출력을 사용한 결과로 정정해 기록했다. 실제 TTA 5개 임계값 후보는 GT 515개, 개별 뷰 후보는 518개, 합집합은 521개였다. 평균 TTA에 개별 뷰가 더하는 후보는 6개이며, base와 비교한 31개를 평균 TTA의 손실로 해석하면 안 된다.

이러한 후보 수는 저장된 임계값·연결요소 범위 안의 진단이며 모델 구조 자체의 성능 상한이 아니다. 새 모델은 실제 학습 결과로 판단한다.
