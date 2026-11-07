# 기술 가이드와 실험 기록

[한국어 프로젝트 소개](../README.ko.md) · [English introduction](../README.md) · [English technical guide](TECHNICAL_GUIDE.md)

GONG Hα 영상의 필라멘트를 검출·분할하고, 원본 2048×2048 좌표의 겹치지 않는 인스턴스 마스크를 제출하는 연구 코드입니다. 대회 지표는 PQ이며, 로컬 평가는 제공된 self-evaluation notebook의 **strict IoU > 0.5, 전체 주석 그룹 누적 방식**을 따릅니다.

현재 자체 제출 기준선은 **YOLO26s-seg + ConvNeXt-Tiny encoder U-Net(`best84`)**입니다. YOLO와 U-Net이 각각 영상에서 예측한 뒤 마스크를 결합합니다. YOLO 박스를 U-Net에 전달하는 ROI 모델과는 별도 구성입니다.

| 기록 | 값 | 의미 |
|---|---:|---|
| 자체 기준선 public LB | **0.39** | 사용자가 실제 제출에서 확인한 표시 점수 |
| 선택 당시 고정 검증 PQ | **0.472593** | 반복 사용한 개발 validation 결과 |
| CHASE 비교 실행의 기존 복합모델 PQ | **0.471930** | 해당 실행에서 다시 계산한 대조군 |
| 공개 예측을 추가·교체한 혼합 CSV public LB | **0.45** | 자체 모델이 독립적으로 생성한 성능과 구분 |
| 외부 공개 CSV public LB | **0.55** | 참조 예측이며 test 정답이 아님 |

전체 구조와 최신 실험 상태는 [현재 파이프라인](CURRENT_PIPELINE.ko.md)을 먼저 읽으세요. `docs/`의 다른 문서는 각 실험 당시의 기록이므로, 과거 문서의 “아직 실행하지 않음”이나 점수 표현을 최신 상태로 해석하지 않습니다.

Colab의 현재 시작점은 [GitHub 소스를 불러오는 실행 노트북](../notebooks/SolarFilament_GitHub_Colab.ipynb)입니다. 코드 전체를 압축 문자열로 복사하는 대신 저장소의 버전을 고정해 실행하며, 데이터와 가중치는 본인의 Drive에서 읽습니다.

[Colab에서 열기](https://colab.research.google.com/github/ikjoo5/solar_filament/blob/main/notebooks/SolarFilament_GitHub_Colab.ipynb) · [다음 CHASE/GONG 실험의 비교 설계](CHASE_GONG_NEXT_EXPERIMENT.ko.md)

## 저장소에 포함하는 것

- `tools/`: 평가, 모델 학습, 재개, 독립 추론, 제출 생성 및 실험 코드.
- `docs/`: 설계·실험 기록과 해석 범위.
- `tests/`: 작은 합성 사례로 평가·좌표·캐시·재개를 검사하는 테스트와, 별도 자료가 필요한 재현 검사.
- `requirements-*.txt`: 작업별 의존성.
- `configs/`: 공개 가능한 설정 예제. 실제 데이터 경로는 실행 환경에서 지정합니다.

대회 이미지·정답 파일·분할 CSV·합성 이미지·체크포인트·예측 RLE/CSV·원시 캐시·결과 ZIP은 별도로 보관합니다. 저장소를 clone하는 것만으로 과거 학습 가중치나 비공개 입력이 생기지는 않습니다. 필요한 파일을 본인의 Google Drive 또는 로컬 데이터 경로에 준비해야 합니다.

## 설치와 CPU 평가

Python 3.12의 별도 환경에서 시작할 수 있습니다. 아래 명령은 저장소 루트 기준입니다.

```bash
git clone https://github.com/ikjoo5/solar_filament.git
cd solar_filament
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-eval.txt
python -m unittest discover -s tests -p 'test_filament_eval.py' -v
```

제출 CSV의 형식·RLE·마스크 겹침만 확인하려면:

```bash
python tools/filament_eval.py --pred /path/to/submission.csv
```

정답이 있는 같은 validation의 PQ를 계산하려면:

```bash
python tools/filament_eval.py \
  --pred /path/to/validation_predictions.csv \
  --gt /path/to/validation_gt.csv
```

GT가 없으면 PQ를 계산하지 않습니다. 공개 test 예측과의 IoU는 참조 일치도이며 실제 test recall이나 PQ가 아닙니다. 정답 JSON의 변환은 `python tools/convert_annotations.py --help`에서 확인할 수 있습니다. 다른 split의 예측과 GT를 조합하지 마세요.

## 기존 기준선으로 제출 파일 생성

CUDA와 호환되는 PyTorch/torchvision이 설치된 GPU 환경이 필요합니다. Colab에서는 제공되는 CUDA 조합을 유지하고 필요한 추가 패키지를 설치합니다.

```bash
python -m pip install -r requirements-submission.txt
```

기본 제출 엔진은 A100에서 사용했던 native batch 4 구성입니다. 다음과 같이 **실제 경로로 수정한 JSON**을 저장합니다.

편집 가능한 예제는 [A100 설정](../configs/submission.example.json), [T4 설정](../configs/submission_t4.example.json)에 있습니다. `.local.json` 이름으로 복사한 개인 설정은 Git에서 제외됩니다. T4 Colab 노트북은 기존 CUDA 라이브러리를 유지하는 별도 설치 helper를 사용합니다. `requirements-eval.txt`와 `requirements-submission.txt`의 NumPy 고정 버전은 다르므로 CPU 평가 환경과 GPU 재현 환경을 분리하세요.

```json
{
  "test_images": "/data/MAGFiLO_1.0_Kaggle_2026/test/test_images",
  "yolo_checkpoint": "/checkpoints/original_yolo/best.pt",
  "native_checkpoint": "/checkpoints/original_native/best.pt",
  "train_split": "/data/splits/train_split.csv",
  "val_split": "/data/splits/val_split.csv",
  "output": "/outputs/frozen_baseline_v1",
  "keep_stages": false
}
```

```bash
python tools/submission_driver.py \
  --config /path/to/submission.local.json \
  --result /path/to/submission_result.json
```

- 입력은 원본 test 이미지와 두 가중치입니다. 기존 제출 CSV나 공개 0.55 RLE를 읽어 기준선 예측에 섞지 않습니다.
- 원래 가중치와 분할의 SHA를 검사합니다. 다른 모델을 이 고정 재현 경로에 넣고 SHA 검사만 해제하는 방식으로 비교하지 않습니다.
- 이미지별 완료 결과와 단계 캐시를 저장합니다. 같은 입력·설정·코드·런타임에서 재실행하면 완료 부분을 재사용합니다. 조건이 바뀌면 새 output 폴더를 사용합니다.
- 모든 test 이미지가 끝난 후 `submission_pq0472593.csv`를 씁니다. 파일명의 숫자는 과거 validation 기준선이며, 생성 파일의 LB 점수를 뜻하지 않습니다.
- Kaggle에 자동 제출하지 않습니다.

T4 batch 1, 원본 전체 파이프라인과 R1을 같은 추론에서 비교하는 별도 엔진은 `tools/r1_test_submission.py`입니다. `--help`와 [현재 파이프라인의 T4 설정](CURRENT_PIPELINE.ko.md#t4에서-기준선과-r1을-같이-만드는-경로)을 확인하세요. R1은 사용자가 확인한 표시 public LB에서 **0.39→0.39**였으므로 채택된 상승으로 기록하지 않습니다.

## 학습과 실험 진입점

| 목적 | 주요 소스 | 필요한 별도 자료 |
|---|---|---|
| native U-Net 학습·재개 | `solar_a100_train_standalone.py`, `native_mixed_data.py` | GONG train/valid, annotations, 선택한 합성 자료 |
| 고정 분할 합성 실험 | `targeted_synthetic_data.py`, `targeted_synthetic_training.py` | train 유래 donor·배경·출처 메타데이터 |
| 5-fold 학습·OOF 평가 | `fivefold_study.py`, `fivefold_full_pipeline.py`, `evaluate_fivefold_study.py` | 새 fold 분할 및 해당 fold의 가중치/캐시 |
| CHASE 데이터·학습 | `chase_transfer_data.py`, `chase_best84_training.py`, `chase_yolo_training.py` | CHASE 이미지/마스크와 원래 GONG 가중치 |
| CHASE 독립 비교 | `chase_independent_plain_worker.py` | 완료한 CHASE 실험의 Drive 폴더·manifest·4개 가중치 |
| detection → ROI segmentation 실험 | `roi_filament_pipeline.py`, `bbox_probe_pipeline.py`, `detr_best84_pipeline.py` | 실험별 detector/segmenter 가중치와 설정 |

모든 소스는 `tools/` 아래에 있습니다. 각 CLI의 `--help`와 해당 실험 문서를 확인하세요. 일부 과거 실행기는 입력 SHA·출처를 고정한 **기존 실행 재현용**이며 범용 학습 명령이 아닙니다. Colab 패키지 생성기에는 별도 assets가 필요한 경우가 있으므로, 파일이 없는데 빈 데이터나 임의 manifest로 우회하지 않습니다.

전체 테스트를 무조건 실행하면 외부 데이터·과거 실행 디렉터리·GPU가 필요한 검사가 포함될 수 있습니다. 새 환경에서는 먼저 위의 데이터 없는 평가 테스트를 실행하고, 작업에 필요한 테스트와 실제 입력 검증을 이어서 실행합니다. CPU 검사 통과는 새 GPU 학습 완료나 점수 향상의 증거가 아닙니다.

데이터 없이 실행하는 핵심 CPU 검사 모음은 다음과 같습니다. 이 설치 명령은 **별도 CPU 환경용**이며, 동작 중인 Colab CUDA 환경에 적용해 GPU용 PyTorch를 교체하지 않습니다.

```bash
python -m pip install torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
python tools/check_core.py
```

동일한 검사 목록을 GitHub Actions에서도 실행합니다. 데이터·과거 결과가 필요한 전체 이력 테스트를 이 핵심 검사에 포함한 것으로 표시하지 않습니다.

## 실험 원칙

- 재학습 비교에는 원래 고정 train 616장 / validation 91장을 사용합니다. validation은 101개 annotator 그룹과 708개 객체를 포함합니다.
- 5-fold OOF는 각 영상을 학습하지 않은 fold 모델의 예측으로만 구성합니다. `best84`나 다른 fold의 학습 가중치를 새 OOF에 끼워 넣지 않습니다.
- 합성의 donor·배경·영상 통계는 train에서만 가져옵니다. 검증·test 자료를 합성 학습 재료로 쓰지 않습니다.
- PQ와 함께 SQ/RQ·TP/FP/FN을 기록하고, 반복 사용한 validation/OOF의 개선과 실제 LB 개선을 구분합니다.
- CHASE는 계속 활용할 연구 대상입니다. CHASE-only 60에폭 가중치는 현재 기준선을 대체하지 않습니다. GONG 혼합·영상 적응·형태 합성은 별도 실험으로 설계하며, SWEFFI는 후속 단계로 보류합니다.
