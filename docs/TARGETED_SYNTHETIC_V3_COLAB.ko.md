# 어려운 필라멘트 합성 데이터 비교 — Colab 실행 안내

이번 코드는 기존 고정 train 616장 / validation 91장을 사용해 **합성 데이터의 종류만 바꾸는 YOLO 추가 학습 비교**를 실행한다. Native best84와 기존 제출 후처리는 고정한다. 5-fold를 다시 학습하지 않는다.

## 실행

1. Colab의 런타임 유형을 **T4 GPU**로 선택한다.
2. `SolarFilament_TargetedSyntheticV3_T4_CopyCell.py`의 전체 내용을 셀 하나에 붙여 넣고 실행한다. 또는 같은 이름의 `.ipynb`를 연다.
3. Drive 연결을 승인하면 경로·원본 해시 확인, 합성 생성, 두 실험군 학습, 고정 validation 평가, 결과 ZIP 저장을 순서대로 실행한다.
4. 완료 시 내려받는 **결과 ZIP의 모든 part**를 전달하면 된다. 가중치와 원본 전체 사진은 ZIP에 넣지 않는다. 가중치는 Drive에 남는다.

설정과 코드는 처음 실행할 때 Drive에 함께 보관한다. 셀에 포함된 압축 문자열은 실행에 필요한 소스 코드와 train donor ID 목록이다. 테스트셋 예측이나 제출 RLE payload가 아니다.

## 이번에 고정한 조건

| 항목 | 설정 |
|---|---|
| 대조군 `control_v2` | 기존 합성 v2 사용 |
| 실험군 `targeted_v3` | train의 어려운 작은 객체를 우선 제안하는 새 합성 사용 |
| 시작 가중치 | 두 군 모두 같은 기존 YOLO `best.pt` |
| 학습량 | **각각 추가 30에폭, 두 군 합계 60에폭** |
| 입력 | 2048×2048, batch 1, gradient accumulation 없음 |
| 최적화 | AdamW, 시작 LR 0.0001, 최종 30에폭 기준 cosine schedule |
| 에폭당 데이터 | 실제 주석 레코드 1,053개 + 합성 263개 = 1,316개 |
| 합성 비율 | 263/1,316 ≈ 19.985%, 두 군 동일 |
| Validation 사용 | 학습·donor 선택·임계값 탐색에 사용하지 않고 지정 에폭의 평가에 사용 |
| 비교 지표 | 공식 PQ·FP·FN, GT recall, 작은 GT recall, 기존 대비 복구·손실 객체 수 |
| 최종 파이프라인 | YOLO 단독과 고정 native best84를 결합한 기존 제출 구성 둘 다 평가 |

같은 사진의 주석자별 라벨은 별도 학습 레코드로 유지한다. 원사진은 616장이다. 새 합성은 train 배경의 기존 라벨을 보존하고, 다른 주석자까지 포함한 알려진 전경 영역을 피해 train donor 2개를 삽입한다. 어려운 작은 donor의 **제안 확률**이 70%이며, 품질·배치 검사 후 실제 채택 비율은 결과에 기록한다.

두 군 모두 동일한 증강과 같은 데이터 수를 사용한다. 이번 비교에는 별도 real-only 마무리 구간을 추가하지 않았다. AMP overflow로 생긴 optimizer update 생략은 history에 따로 기록한다.

## 셀 맨 위에서 바꿀 수 있는 실행 범위

```python
RUN_UNTIL = 30
PHASE = 'all'
ARMS = ['control_v2', 'targeted_v3']
AUTO_DOWNLOAD = True
```

- `PHASE='all'`: 준비 → 두 군 학습 → 평가. 기본 실행이다.
- `PHASE='prepare'`: 합성과 품질 확인용 미리보기만 생성한다. 학습하지 않는다. 이후 `'all'`로 바꾸어 같은 셀을 실행하면 된다.
- `PHASE='train'`: 지정한 군을 학습하고 저장한다. 평가를 따로 할 때 사용한다.
- `PHASE='evaluate'`: 이미 저장한 두 군의 같은 에폭 가중치로 평가만 실행한다.
- `RUN_UNTIL=10` 또는 `20`: 30에폭 스케줄을 유지한 중간 정지다. 이후 `30`으로 바꾸면 이어서 학습한다. **기본값 30이면 실제 학습 목표도 30에서 종료된다.**
- `ARMS=['control_v2']`: 한 군만 먼저 실행할 수 있다. 다음 실행에서 `['targeted_v3']`를 지정하면 다른 군을 진행한다. 두 군의 같은 에폭이 준비되어야 비교 평가한다.

학습이 모두 끝나도 91장에 대한 원본 YOLO·대조군·실험군의 여러 추론 뷰 평가가 이어질 수 있다. 이때 `검증 추론 10/91` 같은 진행 로그가 나오며 추가 학습 에폭이 아니다. Native 추론은 세 모델이 공유한다.

## 기본 경로

```text
/content/drive/MyDrive/filament-segmentation-2026/
  MAGFiLO_1.0_Kaggle_2026/train/
    train_images/
    MAGFiLO_1.0_Annotations_kaggle2026_train.json
  synthetic_copy_paste_v2/
    images/
    synthetic_annotations.json
    split_info.json

/content/drive/MyDrive/SolarFilament_LimbOnly_YOLO26_mixed_FullRes/
  best.pt
  train_split.csv
  val_split.csv

/content/drive/MyDrive/SolarFilament_NativeUNet_A100/20261028T030808Z_bedf3e21/
  best.pt
```

최초 실행 셀의 `BASE`, `YOLO_RUN`, `NATIVE_RUN`으로 경로를 지정한다. 결과의 기본 위치는 다음과 같다.

```text
filament-segmentation-2026/fixed_split_experiments/targeted_synthetic_v3_yolo_t4_v1/
  experiment_config.json
  current_code.json
  code/<version>/
  data/<identity>/                 # 출처·변환·해시와 미리보기
  arms/control_v2/
    last_full.pt                  # 마지막 완료 에폭의 전체 재개 상태
    epoch030.pt                   # 30에폭 완료 후 평가용 가중치
    history.json
  arms/targeted_v3/
    last_full.pt
    epoch030.pt
    history.json
  evaluation/latest_evaluation.json
  exports/                        # 결과 ZIP 각 part < 28MiB
  result.json
```

`synthetic_preview.png`는 전체 합성 사진, `synthetic_native_preview.png`는 작은 객체 주변의 합성 전·후·정답 마스크를 원래 픽셀 크기로 보여준다. 두 미리보기도 결과 ZIP에 포함한다. 합성 전체 PNG와 학습 캐시는 `/content/solar_targeted_synthetic_v3_cache`에 두어 Drive 사용량을 줄인다.

## 중단 후 재개

최초 셀을 같은 설정으로 다시 실행하거나 `SolarFilament_TargetedSyntheticV3_T4_ResumeCell.py`를 실행한다. 런타임이 완전히 초기화돼도 Drive의 코드·설정·가중치를 불러오고 로컬 캐시를 다시 만든다.

- **마지막으로 완전히 저장된 에폭부터 재개**한다. 중단된 에폭의 일부 batch는 다시 학습한다.
- 모델·EMA·optimizer·scheduler·AMP scaler 상태를 함께 복원한다.
- 끝난 군/에폭은 재학습하지 않는다. 이미 완료된 동일 평가 결과도 재사용한다.
- 원본 데이터·코드·설정이 바뀌면 기존 실험에 섞어 재개하지 않고 오류로 알린다. 학습 조건을 바꾸는 새 실험은 새 `OUTPUT` 폴더를 사용한다.
- 기존 Colab의 PyTorch/torchvision CUDA 조합은 재설치하거나 낮추지 않는다. 세션 사이 패키지 버전이 달라지면 동일 학습 상태 비교를 위해 재개가 거부될 수 있다.

## 확인한 범위와 해석

- 작은 CPU 모델에서 연속 2에폭과 1에폭 후 재개한 2에폭 결과의 모델·EMA 가중치가 일치했다. GPU에서 bitwise 동일함을 보장하는 결과는 아니다.
- 실제 train GT 7,491개의 YOLO polygon 변환을 검사했다. 원래 마스크와 raster IoU 최소값은 0.97225였다. 기존 v2의 3,427개 객체는 최소 1.0이었다. YOLO polygon은 원래 마스크와 완전히 같은 형식이 아니므로 원래 RLE도 출처 기록에 보존한다.
- 2048px 합성 테스트 자료에서 전체 준비, 이미지·라벨 검사, 로컬 캐시 삭제 후 동일 복원을 확인했다. 실제 대회 원사진을 사용한 새 합성 생성과 T4 전체 학습은 Colab에서 실행해야 한다.
- **새 모델 PQ와 학습 시간·최대 GPU 메모리는 아직 측정하지 않았다.** 실행 결과로 v3가 v2보다 효과적인지 판단한다.
- 기존 91장 validation은 여러 번 사용한 개발 세트다. 개선이 확인되어도 곧바로 LB 상승이나 PQ 0.7 달성을 의미하지 않는다. 이 셀은 자동 제출·자동 채택을 하지 않는다.
