# 원본 해상도 foreground/spine 모델 실험

현재 YOLO 계열 최고 validation PQ는 0.4649385759이며 목표 0.50은 아직 미달이다. 별도의 ConvNeXt-T U-Net에서 원본 픽셀 크기의 foreground와 spine을 학습하고, 이후 YOLO 예측과 결합할 수 있도록 준비했다. 새 모델의 전체 GPU 학습·성능 측정은 아직 실행하지 않았다.

## 실행

1. Colab에서 GPU를 선택한다. A100의 큰 메모리가 있으면 큰 crop을 사용하며, T4용 설정도 포함된다.
2. 기존 `SCRIPT` 복구 셀을 실행한다. 경로와 기존 limb 보정을 재사용한다.
3. `tools/colab_train_native_unet_cell.py` 전체를 새 셀에 붙여 실행한다. 이 파일은 plain Python이며 다른 repository 소스나 인코딩된 payload를 받지 않는다.

## 설정

| GPU 메모리 | crop | batch | gradient accumulation | microbatches/epoch |
|---|---:|---:|---:|---:|
| 32GiB 미만(T4 포함) | 768 | 1 | 4 | 400 |
| 32GiB 이상 | 1024 | 2 | 2 | 200 |

두 설정 모두 epoch당 400개 crop, optimizer update 100회다. 30epoch를 학습하고 처음 및 2epoch마다 전체 91장을 검증한다.

- 실제 train616개 영상만 사용하며 image-first/annotator-next로 샘플링한다.
- train 면적 하위25% 객체 중심 crop50%, 다른 positive crop25%, uniform crop25%다.
- 기존 limb 보정 후 ImageNet RGB 정규화한 grayscale3채널을 입력한다.
- ConvNeXt-T ImageNet encoder와 원본 해상도 U-Net2채널 출력이다.
- foreground BCE+Dice +0.1 balanced spine BCE를 사용한다.
- AdamW lr1e-4, cosine schedule, AMP, gradient clipping1.0이다.
- 각 전체 validation에서 threshold0.35/0.5/0.65 × minimum area100/200/300, 9개 전역 설정을 비교한다.
- strict IoU>0.5의 공식 global all-pairs PQ로 checkpoint를 선택한다.

## 저장과 세션 종료

Drive의 기존 `RUN/native_unet_runs/<새 timestamp_uuid>/` 아래에 저장한다. 기존 YOLO checkpoint를 덮어쓰지 않는다.

- `best.pt`: 최고 validation PQ의 새 모델 가중치.
- `last.pt`: 마지막 완료 epoch의 모델·optimizer·scheduler·scaler 상태.
- `config.json`, `history.json`, `best_epoch.json`, 실행 source.
- 최고 epoch의 9개 threshold/area CSV와 `best_validation.csv`.

각 validation 후 작은 `_native_unet_results.zip`을 자동으로 갱신하며, 종료 시에도 최종 ZIP을 만든다. 이 ZIP에는 큰 checkpoint가 포함되지 않으므로 그대로 업로드해 점수 재검증·YOLO 앙상블 비교에 사용한다. 완료 전에 세션이 끝나도 가장 최근에 저장한 ZIP은 Drive에 남는다.

이 셀은 **새 학습 시작용**이다. 셀을 다시 실행하면 새 폴더에서 ImageNet encoder로 새로 시작하며, optimizer 재개 기능을 제공한다고 주장하지 않는다. 저장된 last 상태를 이용하는 재개 코드는 필요할 때 별도로 구성할 수 있다.

Drive의 최근 저장 ZIP만 받으려면 mount 후 다음 셀을 실행한다.

```python
from pathlib import Path
from google.colab import files
root=Path('/content/drive/MyDrive/SolarFilament_LimbOnly_YOLO26_mixed_FullRes/native_unet_runs')
archives=sorted(root.glob('*_native_unet_results.zip'))
assert archives, '저장된 validation ZIP이 아직 없습니다.'
files.download(str(archives[-1]))
```

## 검증 범위

새 셀의 독립 동작·GPU별 설정·작은 native crop의 optimizer update·9개 검증 grid·공식 scorer의 strict0.5/annotator 반복·자동 ZIP 교체와 checkpoint 제외를 CPU에서 점검했다. 전체 테스트 88개를 통과했다. GPU 메모리와 전체 학습 wall time은 실제 Colab 실행에서 확인해야 한다. CPU 구현 검증을 새 모델의 성능 결과로 보고하지 않는다.
