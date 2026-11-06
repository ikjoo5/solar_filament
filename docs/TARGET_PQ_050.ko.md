# 검증 PQ > 0.50 실험 기록

## 목표와 현재 상태

사용자는 로컬 검증 PQ를 실제 리더보드 개선의 실험 지표로 사용하며 0.50 초과까지 반복 실험하기로 했다. 기존 train 616장/validation 91장, 101 annotation group/708 GT와 제공 evaluator의 전역 all-pairs PQ를 유지한다. validation 정답으로 영상별 마스크를 고르거나 정답 위치를 예측에 넣지는 않는다. 전역 하이퍼파라미터 튜닝은 기록하며, 모델 학습과 구분한다.

**최신 결과:** 4-view 실험 ZIP 수신 후 전처리·후처리 중심 분석을 계속해 PQ **0.4571518474**, TP/FP/FN **467/199/241**까지 개선했다. 다음은 원본 영상에 전처리 8종을 적용하는 총 68개 설정이다. 상세 기록은 [2차 실험](PREPROCESS_POSTPROCESS_ROUND2.ko.md)을 우선한다. 아래 표는 1차 단계 당시 기록이다.

| 단계 | 설정 수 | 실제 전체 validation 최고 PQ | 겹침 없는 최고 PQ | 상태 |
|---|---:|---:|---:|---|
| 기존 threshold/후처리 | 10 | 0.439597 | 0.431752 | 사용자 GPU 결과, 로컬 재계산 완료 |
| NMS/겹침 해소/경계 보정 | 13 (기준선 포함) | 0.439597 | 0.431752 | 로컬 완료 |
| 면적 필터/overlap 병합 | 20 (원본 4개 포함) | **0.441682** | **0.434251** | 로컬 완료 |
| 낮은 confidence + 4-view 추론 | 22 | 미측정 | 미측정 | 실행 코드·좌표 테스트·CPU 1장 smoke 완료 |
| native U-Net + spine | 학습 실험 | 미측정 | 미측정 | 학습/전체 영상 검증/checkpoint I/O smoke 완료 |

아직 **0.50을 초과하지 않았다.** 단계별 설정에는 같은 기준선이 반복 포함되어 있어 표의 설정 수를 고유 실험 수로 단순 합산하지 않는다. 최고 raw preserve에는 겹침이 포함되며 이를 숨기거나 disjoint가 대회 필수라고 추정하지 않는다.

## 1. 새로 확인한 개선

- union+CC, conf 0.25, area≥100/200: PQ 0.434098, FP165/FN293. 기존보다 FP 7개 감소, FN 유지.
- union+CC, conf 0.25, area≥300: PQ **0.434251**, FP162/FN294. 기존 0.431752보다 +0.002499.
- preserve, conf 0.15, area≥300: PQ **0.441682**, FP378/FN203. overlap 71장, 추가 매칭쌍 77개는 남는다.
- complete-link mask-IoU 병합은 개선되지 않았다. 설정을 무한 확장하지 않고 다음 모델/추론 실험으로 이동한다.

근거는 `docs/evidence/object_filter_experiments.json`, 후보 CSV는 `/workspace/artifacts/object_filter_candidates_v1`에 저장했다.

## 2. 다음 GPU 실험: 재학습 없는 22개 설정

`tools/experiment_yolo_inference.py`는 다음을 실행한다.

- base 2048, scale 2560, horizontal flip 2048, vertical flip 2048의 4-view. 전체 validation에서는 91×4=364 forward pass다.
- 각 view의 conf≥0.05 원시 마스크와 실제 confidence를 원본 좌표의 compressed RLE JSONL로 저장한다.
- base conf 0.05/0.10/0.15/0.20/0.25 × area20/100: 10개.
- foreground binary vote의 conf 0.10/0.15/0.20 × vote≥50%/75% × area100/200: 12개.
- binary agreement를 mask logits 또는 보정된 probability라고 부르지 않는다.

파일 다운로드가 작동하지 않는 사용자에게는 기존 평가를 실행했던 같은 Colab의 `SCRIPT`를 재사용하는 `tools/colab_next_inference_cell.py`의 내용을 직접 붙여넣기로 전달한다. 새 작업 파일·새 결과 폴더만 만들며 기존 데이터와 모델을 수정하지 않는다. 실제 confidence와 raw RLE도 ZIP에 포함되어 이후 분석에 재사용할 수 있다.

이 cloud instance에는 CUDA 장치가 없어 전체 364-forward 실험을 대신 실행할 수 없다. CPU에서 공개 참조 영상 1장, 2-view, 22개 설정의 I/O를 검증했고 이 결과를 전체 validation 점수로 보고하지 않는다. Colab의 기존 Drive 이미지와 GPU가 실제 실행 위치다.

## 3. 새 학습: ConvNeXt-T U-Net + 직접 제공된 spine

`tools/train_filament_unet.py`와 `requirements-train.txt`를 준비했다.

- native 768² crop(T4 기본), 1024² crop(A100 메모리 확인 후).
- foreground/spine의 2-channel full-resolution output. foreground BCE+Dice, balanced spine BCE×0.1.
- 실제 영상 먼저 균등 선택 → annotator 하나 선택. annotation variant 개수 때문에 영상 가중치가 자동 증가하지 않는다.
- train-only 면적 하위 quartile 중심 crop, 일반 positive crop, uniform crop을 혼합한다.
- D4 영상/정답 동시 변환, 제한된 gamma/intensity 변화.
- 전체 영상 overlapping tile 추론과 양수 하한 Hann blending. 공식 전역 PQ로 best checkpoint를 고른다.
- AMP: 지원 시 BF16, 그 외 CUDA FP16+GradScaler. CPU smoke는 FP32다.
- dataset별 dense GT 전체를 캐시하지 않는다. processed image는 별도 cache에 저장한다.
- checkpoints는 state_dict와 plain metadata로 저장하며 `weights_only=True`로 다시 읽는다.

예시 본 실험(T4):

```bash
python tools/train_filament_unet.py \
  --images /content/drive/MyDrive/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/train_images \
  --annotations /content/drive/MyDrive/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json \
  --train-split /content/drive/MyDrive/SolarFilament_LimbOnly_YOLO26_mixed_FullRes/train_split.csv \
  --val-split /content/drive/MyDrive/SolarFilament_LimbOnly_YOLO26_mixed_FullRes/val_split.csv \
  --output-dir /content/drive/MyDrive/SolarFilament_LimbOnly_YOLO26_mixed_FullRes/unet_spine_trial_v1 \
  --cache-dir /content/filament_cache --encoder tu-convnext_tiny \
  --epochs 30 --steps-per-epoch 400 --crop 768 --batch 1 --accumulate 4 \
  --validate-every 2 --device cuda
```

출력 디렉터리는 새 경로여야 한다. A100에서는 프로파일 후 crop1024/batch2를 비교한다. 빠른 pilot은 epochs3/steps100으로 실행할 수 있지만 본 학습 결과라고 해석하지 않는다. steps_per_epoch는 전체 원본 영상 1회 순회를 의미하지 않는 microbatch 수다. 기본 본 실험은 epoch당 optimizer update 100회, 최대 30epoch의 3,000 update다.

구현 검증 중 spine polyline cap이 foreground 밖으로 나오는 사례를 테스트가 발견했다. 보조 target을 해당 foreground와 교차해 일치시키고 테스트를 통과시켰다. timm 1.0.20과 최신 Hugging Face Hub API 불일치도 발견하여 호환 버전 0.34.4를 고정했다. TLS/가중치 검증을 비활성화하지 않았다.

이 학습 코드는 CPU에서 실제 참조 JPEG 2장/정답 7개의 제한된 cohort로 1-step 학습, full-image tile validation, best/last checkpoint 저장 및 best checkpoint strict reload를 확인했다. encoder의 native output과 backward도 확인했다. 이 smoke의 PQ는 학습 성능을 의미하지 않으며, 신규 모델의 전체 validation 성능은 아직 미측정이다.
