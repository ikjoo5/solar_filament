# 고정 616/91 분할의 마스크 품질 학습 실험

이 실험은 **아직 실제 태양 이미지로 학습하지 않았다.** T4 Colab 실행 코드와 기능 검증을 준비했다.
실제 PQ와 실행 시간은 Colab 결과를 받아 확인한다.

## 실험 이유

- 원영상 밝기·ridge·방향 일관성을 이용한 20개 추가 설정은 기존 최고 개발 PQ **0.436752508**을 넘지 못했다.
- 어두운 선 모양인 FP도 많았다. 밝기나 선 구조의 존재만으로는 정답과의 충분한 겹침을 판별하기 어려웠다.
- 5-fold 원시 후보를 정답과 대조한 원인 진단에서는 현재 FN 2,955개 중 1,965개가 4개 geometric 뷰 안에 IoU > 0.5 후보를 가지고 있었다. 최소 면적 300을 적용하면 1,911개였다. 전처리·tile까지 포함한 전체 후보군에서는 각각 2,388개와 2,306개였다.
- 위 수치는 정답을 이용해 후보의 존재를 확인한 진단이다. 후보 간 경쟁·FP 비용을 무시하므로 구현 가능한 복구량이나 PQ 상한이 아니다. 진단의 fold 모델과 이 실험의 고정 YOLO도 서로 다르다.
- 새 모델은 마스크와 주변 사진을 함께 보고 **정답과 IoU > 0.5로 매칭됐을 때의 예상 IoU**를 학습한다. 검출 confidence와 마스크의 겹침 품질을 구분하는 실험이다.

## 데이터와 모델

- 원래 고정 분할을 그대로 사용한다: train **616장·1,053주석그룹**, validation **91장·101주석그룹·708객체**.
- 원본 정답 JSON, 두 split CSV, 기존 태양 YOLO 가중치를 SHA256으로 확인한다. 같은 사진의 여러 주석은 같은 분할에 있어야 한다.
- 기존 YOLO는 고정한다. Limb 보정과 4개 뷰(base 2048, scale 2560, hflip 2048, vflip 2048)로 confidence ≥ 0.05 후보를 준비한다.
- 사진/마스크를 문맥이 포함된 256×256 ROI로 만든다. RGB로 복제한 흑백 사진 3채널과 마스크 1채널이 ResNet18의 입력이다.
- ImageNet 가중치를 사용하고 추가한 마스크 채널의 초기 가중치는 0으로 둔다. 예상 매칭 IoU 회귀와 매칭 확률 보조 학습을 함께 수행한다.
- 각 후보의 정답은 사진의 여러 주석에 대한 평균이다. 각 주석에서 최대 IoU가 **엄격히 0.5보다 클 때만** IoU를 회귀 정답에 반영한다.
- Train은 사진당 최대 48개 후보를 학습한다. Validation은 정답으로 후보를 골라내지 않고 **모든 원시 후보**를 평가한다.
- 5-fold의 예측이나 이번에 수집한 34,725개 특징을 새 모델의 학습 데이터로 사용하지 않는다. 이 중 일부는 원래 고정 validation 사진으로 학습된 fold 모델에서 나온 예측이기 때문이다.

## 비교 방법

- 같은 원시 후보에 같은 mask NMS, 점수 순서에 따른 픽셀 소유권 할당, 최소 면적 300을 적용한다.
- 비교 대상은 원래 YOLO confidence와 새 예상 IoU 점수다. 두 점수 모두 사전 지정한 5개 threshold(0.10/0.15/0.20/0.25/0.30)를 비교한다.
- 겹치지 않는 최종 마스크로 공식 all-pairs PQ를 계산한다. Validation은 gradient에 사용하지 않고 checkpoint와 threshold를 고르는 개발 평가에만 사용한다.
- 이 기준선은 **YOLO 단독의 동일 후보군 비교**다. 예전 YOLO+native PQ 0.472593이나 707장 OOF PQ 0.436753과 점수를 직접 비교하지 않는다.
- 기존 91장을 반복 사용한 개발 결과이므로, 향상이 나타나도 독립 검증이나 리더보드 향상으로 확정하지 않는다.

## 실행과 재개

1. Colab에서 **T4 GPU** 런타임을 선택한다.
2. `SolarFilament_FixedSplit_MaskQuality_CopyCell.py` 파일의 전체 내용을 한 셀에 복사해서 실행한다.
3. 후보 준비가 끝나면 별도 프로세스에서 품질 모델 1개를 학습한다. 처음부터 총 **40 epoch** 일정이다. 처음 2 epoch은 backbone을 고정하고 이후 38 epoch은 전체 품질 모델을 학습한다. 5-fold별로 반복하지 않는다.
4. 중단 후에는 같은 셀을 다시 실행한다. 준비한 원시 후보를 검증해 재사용하고 마지막으로 저장된 epoch부터 학습을 재개한다. 중단된 epoch 안의 배치는 다시 실행한다.
5. 기본값은 `TOTAL_EPOCHS=40`, `RUN_UNTIL=40`이다. 중간 점검을 원하면 `RUN_UNTIL=20`으로 실행하고 이후 40으로 바꾸어 재개한다. 이때 `TOTAL_EPOCHS=40`은 처음부터 끝까지 유지해 같은 learning-rate 일정으로 이어간다.
6. 완료 후 `mask_quality_results.zip`을 전달하면 기준 대비 PQ/FP/FN과 학습 곡선을 검토할 수 있다.

기본 입력 위치:

```text
/content/drive/MyDrive/filament-segmentation-2026/
  MAGFiLO_1.0_Kaggle_2026/train/train_images/
  MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json
/content/drive/MyDrive/SolarFilament_LimbOnly_YOLO26_mixed_FullRes/
  best.pt
  train_split.csv
  val_split.csv
```

저장 위치:

```text
filament-segmentation-2026/fixed_split_experiments/mask_quality_v1/
  code/<source_bundle_sha>/       # 세션 종료 후에도 남는 실행 코드
  proposals/                      # 원시 마스크·정답값, 사진 없이 저장
  corpus_identity.json            # 데이터·모델·코드·환경의 출처
  training_config.json
  baseline.json
  history.json
  last.pt                         # optimizer·scheduler·AMP·RNG 포함 재개 가중치
  best.pt                         # validation 개발 최고 품질 모델
  summary.json
  mask_quality_results.zip        # 평가용 소형 결과, 사진·가중치 제외
```

- 원본 사진은 복제하지 않는다. ROI는 `/content/solar_fixed_mask_quality_cache`에 임시 저장하므로 세션이 초기화되면 재생성한다. 이때 후보가 Drive에 남아 있으면 YOLO forward는 다시 하지 않는다.
- 첫 실행은 707장에 대한 4개 뷰 추론이 먼저 필요하다. `prepare` 로그는 학습 epoch가 아니다.
- 2 epoch마다 validation을 수행하고 epoch 1과 마지막 epoch에도 평가한다. 따라서 validation 로그가 출력되는 것은 정상이다.
- 매 epoch의 마지막 가중치와 검증 PQ 최고 가중치를 따로 저장한다. 오래 학습한다고 반드시 PQ가 오르는 것은 아니며, 검증 PQ가 떨어진 마지막 모델을 최고 모델 대신 선택하지 않는다.
- 이번 40 epoch 셀은 사용자가 이전 20 epoch 셀을 아직 실행하지 않았다고 확인한 뒤 설정했다. 엔진과 후보 생성 소스는 유지하고 실행 인자로만 일정을 40으로 지정했다.
- 기존 YOLO/native 가중치와 5-fold 폴더는 수정하지 않는다. 학습하는 새 품질 모델은 별도 경로에만 저장한다.
- ImageNet 초기 가중치는 최초 실행에 다운로드가 필요하다. Colab의 torch/CUDA는 유지하며 필요한 분석 패키지와 사용했던 YOLO 버전을 맞춘다.
