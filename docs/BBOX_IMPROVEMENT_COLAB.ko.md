# A100 bbox 개선 실험 실행 안내

현재 ROI U-Net을 고정하고, 검출기가 제공하는 bbox를 개선하는 실험이다. **새로 학습하는 모델은 DETR 하나이며 추가 30에폭, 기존 30을 포함해 누적 60에폭**이다. YOLO와 ROI U-Net은 추가 학습하지 않는다.

## 실행 파일

- `SolarFilament_BBoxImprove_A100_CopyCell.py`: 전체 내용을 Colab 셀 하나에 붙여 넣어 실행한다.
- `SolarFilament_BBoxImprove_A100_ResumeCell.py`: 세션이 종료된 뒤 재개할 때 사용하는 짧은 셀이다.
- `SolarFilament_BBoxImprove_A100.ipynb`: 같은 최초 실행 셀과 재개 셀을 담은 노트북이다.

Colab에서 **A100 GPU 런타임**을 선택한다. 이번 추가 학습은 1280 입력, BF16 본체와 FP32 bbox 헤드를 사용하므로 T4에서 자동으로 낮은 설정으로 바꾸지 않는다. 코드가 BF16 지원 GPU인지 확인하고 실제 연산을 사전 검사한다.

최초 셀의 기본값은 다음과 같다.

```python
PHASE = 'all'
RUN_UNTIL_ADDED = 30
AUTO_DOWNLOAD = True
```

| PHASE | 실행 내용 | 추가 학습량 |
|---|---|---:|
| `all` — 기본값 | 기존 bbox/타일 비교 → 결과 저장 → DETR 추가 학습 및 같은 ROI로 비교 | DETR 최대 30에폭 |
| `probes` | 기존 가중치로 전체사진·타일 bbox 추론과 평가만 실행 | 0에폭 |
| `extend` | 기존 가중치의 대조 평가와 DETR 추가 학습·평가 | DETR 최대 30에폭 |

`RUN_UNTIL_ADDED=15`는 기존 30에서 **15에폭을 더 학습해 누적 45**에서 멈춘다는 뜻이다. 이후 값을 30으로 바꿔 재실행하면 남은 15에폭을 학습해 누적 60에 도달한다. 이미 완료된 에폭을 반복하거나 ROI를 다시 학습하지 않는다. 기본 실행에서도 누적 45와 60의 결과를 비교한다.

## 비교할 방법

- **기존 전체사진 검출:** YOLO와 DETR30의 현재 bbox 후보를 기준으로 사용한다.
- **타일 검출:** 원본 2048 사진에서 1024 크기의 crop을 stride 768로 9개 생성한다. 전체사진 후보와 타일 후보를 원본 좌표로 합치고 bbox NMS 0.5/0.7을 비교한다. DETR30 타일은 800 입력으로 추론한다.
- **입력 해상도 대조군:** 기존 DETR30 가중치를 전체사진 1280 입력으로 추론해, 해상도 변경 자체의 영향을 확인한다.
- **추가 학습:** DETR30 가중치에서 전체사진 1280 입력, batch 2, gradient accumulation 2, BF16 본체와 FP32 bbox 헤드로 추가 학습한다. 유효 배치는 4이며 기존 분할은 유지한다. 누적 45/60의 전체사진 1280 결과를 비교한다.
- **공통 분할기:** 모든 bbox를 기존 ROI U-Net30에 전달한다. ROI의 가중치·crop 크기·마스크 이진화 규칙을 바꾸지 않는다.

추가 학습은 원래 800 입력/FP16 실험에서 1280/BF16 본체·FP32 bbox 헤드로 바뀐 별도 단계다. 기존 optimizer의 모멘텀은 이어받고 학습률은 추가 30에폭용 스케줄을 명시적으로 시작한다. FP16에서 BF16으로 바뀌므로 기존 FP16 GradScaler 상태는 새 단계에서 이어 쓰지 않는다. 새 단계 중 세션이 종료되면 해당 단계의 optimizer·스케줄·RNG 상태를 복원한다. 처음부터 60에폭으로 설정한 단일 연속 스케줄과 동일한 실험은 아니다.

bbox 좌표를 계산하는 헤드는 FP32로 유지해 BF16 좌표 양자화가 작은 객체의 위치 정확도를 제한하지 않도록 한다.

검출의 박스 IoU·재현율·오탐과, 같은 ROI U-Net을 거친 **최종 PQ·FP·FN**을 함께 본다. 사진 91장과 주석 그룹 101개를 그대로 사용하며, 서로 다른 주석자는 합치지 않는다. 이 검증셋은 반복 사용한 개발용 분할이므로 새 public LB 또는 독립 5-fold 성능으로 해석하지 않는다.

## Drive 위치

기존 완료 실험:

```text
/content/drive/MyDrive/filament-segmentation-2026/fixed_split_experiments/detection_roi_detr_v1
```

새 실험:

```text
/content/drive/MyDrive/filament-segmentation-2026/fixed_split_experiments/bbox_improvement_v1
```

기존 폴더의 다음 파일을 사용한다.

- `experiment_config.json`, `current_code.json`, `code/.../source_manifest.json`
- `detr/epoch030.pt`: 비교할 기존 DETR
- `detr/last_full.pt`: optimizer를 포함한 DETR 추가 학습의 출발점
- `roi/epoch030.pt`: 모든 실험에 고정할 ROI U-Net
- 이전 결과·후보 캐시와 원본 이미지·정답·고정 train/validation CSV 경로

새 셀은 검토한 기존 코드·설정·가중치 해시와 일치하는지 검사한다. 기존 가중치를 새 Drive 폴더로 복제하지 않고 참조한다. 새 코드·설정·검출 후보·ROI 결과·DETR 추가 단계 checkpoint는 새 폴더에 보관한다.

## 중단과 재개

- 실행을 중단해도 **마지막 완료 에폭**부터 재개한다. 저장 중이던 에폭이나 중간 minibatch는 다시 실행될 수 있다.
- 추론 중단은 이미 저장된 사진·후보 캐시를 확인해 재사용한다.
- `/content`가 초기화돼도 Drive에 저장한 코드·설정에서 복원한다. 최초 셀을 같은 내용으로 실행하거나 재개 셀을 사용한다.
- `PHASE`, `RUN_UNTIL_ADDED`, `AUTO_DOWNLOAD`는 실행 범위다. 이미 저장한 학습 설정이나 코드가 달라지면 같은 실험으로 섞지 않도록 오류를 표시한다.
- 기존 DETR 30에폭 실험 셀의 최종 에폭을 60으로 수정하지 않는다. **새 bbox 개선 셀**이 기존 가중치에서 별도 추가 단계를 시작한다.

## 결과 전달과 오류 확인

`all` 실행은 타일 비교가 끝나면 그 결과 ZIP 경로를 먼저 출력·저장한 뒤 추가 학습으로 넘어간다. 중간 45에폭 및 최종 60에폭에서도 결과를 저장한다. 결과 ZIP은 모델 가중치와 원본 사진을 제외하며, 크기에 따라 여러 파트로 나뉜다. 완료 후 출력된 최신 결과의 ZIP 파트를 모두 전달하면 된다.

전체 실행 로그는 다음 위치에 남는다.

```text
bbox_improvement_v1/colab_bootstrap.log
```

하위 Python 프로세스의 실제 오류도 Colab 출력과 이 파일에 함께 기록된다. `CalledProcessError`만 보이면 그 앞의 traceback이나 로그 마지막 부분을 확인하면 된다.

이 안내와 코드는 실행 준비물이다. **이 환경에서 실제 A100 추가 학습이나 새 모델의 대회 검증 PQ는 아직 측정하지 않았다.**
