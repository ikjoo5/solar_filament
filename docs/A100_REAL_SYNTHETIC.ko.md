# A100 학습 실행 안내

실행 파일은 `notebooks/SolarFilament_A100_RealSynthetic.ipynb`다. Colab에서 A100을 선택하고 세 개의 코드 셀을 위에서부터 실행한다. 기존 `SCRIPT`나 이전 평가 세션 없이 새 세션에서 동작한다. 한 셀로 복사하려면 `notebooks/SolarFilament_A100_CopyCell.py` 전체를 사용한다.

## 기본 학습

- ConvNeXt-T ImageNet encoder + 원본 크기 U-Net foreground/spine 2채널.
- A100: 1024 crop, batch2, gradient accumulation2, effective batch4, BF16.
- epoch당400개 crop/100개 optimizer update, 총100epoch.
- 실제 train616장과 기존 synthetic462장을 사용한다. 10개 빈 합성 영상도 정상적인 foreground 음성 샘플로 포함한다.
- 1–3epoch: 실데이터만. 4–94epoch: image-first 확률 real80/synthetic20. 95–100epoch: 실데이터만.
- 각 도메인에서 작은 객체 중심50%, 다른 positive 중심25%, uniform25% crop을 뽑는다. annotator 수가 많다고 같은 영상을 더 자주 고르지 않는다.
- foreground BCE+Dice +0.1 balanced spine BCE. Synthetic은 spine 정답이 없으므로 auxiliary loss만 제외한다.
- 합성 polygon은 원래 생성기와 같은 OpenCV 반올림+fillPoly, 실데이터는 공식 COCO rasterization을 사용한다.
- 실제 validation91장/101그룹/708GT만으로 첫 epoch, 이후2epoch마다 공식 global PQ를 계산한다. threshold0.35/0.5/0.65 × area100/200/300의 9개 설정에서 최고 checkpoint를 고른다.

## 경로와 분할

설정 셀에 다음 경로가 채워져 있다.

```text
/content/drive/MyDrive/filament-segmentation-2026/
  MAGFiLO_1.0_Kaggle_2026/train/train_images/
  MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json
  synthetic_copy_paste_v2/images/
  synthetic_copy_paste_v2/synthetic_annotations.json
  synthetic_copy_paste_v2/split_info.json

/content/drive/MyDrive/SolarFilament_LimbOnly_YOLO26_mixed_FullRes/
  train_split.csv
  val_split.csv
```

새 split을 만들지 않는다. 사용자 제공 metadata에서 source train924/val230 annotation IDs를 물리 파일로 확장하면 현재 train1,053그룹/616장 및 val101그룹/91장과 정확히 일치한다. 합성 donor와 현재 validation의 파일 겹침은0이다. 원본과 합성 JSON 및 split SHA256도 설정에 고정했다.

학습 전에 실제 Drive의 모든 합성 파일 존재 여부와 최대3장(양성 첫/마지막·빈 영상)의 2048×2048 uint8 형태를 검사한다. 이미지 전체의 화질을 사전에 검증했다는 뜻은 아니다.

## 비교 설정

첫 셀의 `SYNTHETIC_RATIO=0.20`이 기본값이다. 실데이터 단독 비교는 `0.0`으로 바꾸고 `RESUME_DIR=''`로 새로 실행한다. 같은 seed와 crop/update 예산을 유지한다. 합성 혼합이 기존 YOLO에서 도움이 됐다는 사용자 결과를 반영한 시작 설정이며, U-Net의 성능 향상이 이미 확인된 것은 아니다.

## 저장 공간과 재개

캐시와 native FG/spine target은 `/content/solar_filament_a100_cache`에만 저장한다. 새 결과는 다음 root 아래의 timestamp/UUID 폴더에 저장한다.

```text
/content/drive/MyDrive/SolarFilament_NativeUNet_A100/
```

각 run에는 최고 `best.pt`와 완료 epoch의 full `last.pt`만 유지한다. 모델 FP32 state 약121.8MiB, Adam 재개 checkpoint 약365.4MiB, 둘을 합친 정상 저장량 약487MiB다. atomic 교체 순간에는 약853MiB가 필요하며 파일 직렬화 부가량은 별도다. Drive의 용량 부족 표시를 고려해 약1GiB 이상 여유를 확인한다. Drive FUSE의 `disk_usage`를 실제 Google Drive quota로 단정하지 않는다.

세션이 종료되면 같은 노트북의 첫 셀 `RESUME_DIR`에 출력된 기존 run 폴더를 넣고 다시 실행한다. 완료 epoch부터 model/optimizer/scheduler/scaler/history/RNG를 복구한다. crop·혼합비율·학습 epoch 예산·분할·정밀도 등이 달라지면 잘못된 재개를 막는다. 같은 코드·환경의 CPU fixture에서 연속 학습과 중단 후 재개의 최종 가중치가 일치함을 확인했다. GPU·라이브러리 변경 사이의 bit 단위 동일성을 보장한다는 뜻은 아니다.

최고 epoch의 9개 CSV, `best_validation.csv`, config/history/source만 담은 작은 `_native_unet_results.zip`은 각 validation 후 갱신한다. checkpoint는 이 ZIP에 넣지 않는다. 학습이 끝나면 이 ZIP을 보내면 된다. 중간에 종료돼도 노트북 마지막 복구 셀로 최근 저장 ZIP을 받을 수 있다.

## 검증 범위

전체107개 테스트를 통과했다. 추가 검증에는 mixed real/synthetic native1024 spawn loader, synthetic spine gradient 차단, native label cache, 실제 JSON metadata의 CV/COCO 복원 비교, 완료 epoch 재개, 파일 누락 조기 실패, 저장 중 오류에서 이전 checkpoint 유지, 생성된 독립 worker의 2epoch CPU 실행·9개 validation grid·결과 ZIP이 포함된다.

실제 A100에서 새 모델을 전체 학습한 성능은 아직 측정하지 않았다. 현재 검증 최고는 기존 YOLO/tile ensemble PQ0.4649385759이며 목표0.50은 미달이다.
