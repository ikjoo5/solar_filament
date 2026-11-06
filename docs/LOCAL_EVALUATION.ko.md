# 로컬 annotation 변환과 PQ 검사

이 도구는 업로드된 self-evaluation notebook의 점수 의미를 재현한다. Kaggle backend와 최신 규정까지 확인한 것은 아니다. 기존 YOLO 체크포인트의 추론·검증은 [Drive 실행 안내](REMOTE_EVALUATION.ko.md)를 사용한다. 신규 고해상도 모델의 학습 파이프라인은 아직 구현하지 않았다.

현재 Python: `/workspace/.venvs/solar-eval/bin/python`.
현재 생성 GT: `/workspace/artifacts/solar_filament_gt_v1/gt.csv`.
원본 annotation 및 대용량 manifest는 Git checkout 밖에 있다.

## 설치

Python 3.12에서 검증했다. 새 환경에서는 저장소 디렉터리에서 다음을 실행한다.

```bash
python -m venv /workspace/.venvs/solar-eval
/workspace/.venvs/solar-eval/bin/python -m pip install -r requirements-eval.txt
```

현재 인스턴스에서는 기존 NumPy를 재사용하는 `--system-site-packages` venv를 사용했고, 고정한 NumPy/pycocotools 버전과 일치함을 확인했다. 새 설치는 위 requirements로 준비할 수 있다.

## 정답 JSON 변환

```bash
cd /workspace/solar_filament
/workspace/.venvs/solar-eval/bin/python tools/convert_annotations.py \
  --annotations /workspace/datasets/solar_filament/MAGFiLO_1.0_Annotations_kaggle2026_train.json \
  --output-dir /workspace/artifacts/solar_filament_gt_v2
```

새 output directory를 사용해야 한다. 기존 디렉터리와 사용자 파일은 덮어쓰지 않는다. 현재 완료된 변환은 `v1`이며 위 `v2`는 다시 실행할 때의 예다.

- `gt.csv`: evaluator-compatible 두 열, 원본 annotation 하나당 한 인스턴스.
- `manifest.json`: 원본 annotation UUID, 영상/annotator 그룹, category, bbox, spine mapping.
- `summary.json`: checksum·객체 수·누락/경고·변환 계약.

## 예측 CSV만 검사

```bash
/workspace/.venvs/solar-eval/bin/python tools/filament_eval.py \
  --pred /path/to/predictions.csv
```

2048² compressed COCO RLE 전수 디코딩, 빈 마스크·중복 ID·겹침을 확인한다. GT 없이 PQ를 출력하지 않는다. 필요하면 `--expected-images /path/to/image_stems.txt`에 한 줄당 image stem을 적어 전체 이미지 coverage를 확인한다. 객체가 없는 영상은 제출 행이 없을 수 있으므로 missing prediction을 곧바로 형식 오류로 간주하지 않는다. 반면 목록 밖의 예측 ID는 오류다.

## 검증 예측의 PQ

```bash
/workspace/.venvs/solar-eval/bin/python tools/filament_eval.py \
  --pred /path/to/validation_predictions.csv \
  --gt /path/to/validation_gt.csv
```

두 CSV의 image key가 대응해야 한다. 동일 validation split의 **모든 annotator 정답**을 포함한다. 다른 split/test 예측을 train GT에 대조하지 않는다. 현재 업로드 CSV의 177개 image key와 train의 707개 key는 교집합이 **0개**여서 이 두 파일을 바로 조합해 유효한 PQ를 구할 수 없다.

노트북이 GT에 없는 예측을 조용히 무시하는 것과 달리 도구는 기본 오류로 처리한다. 의도적인 subset 검증에만 `--filter-pred-to-gt`를 명시하고, report의 제외된 image/row 목록을 확인한다. 이 옵션은 test를 train에서 평가하는 우회책이 아니다.

평가는 strict `IoU > 0.5`인 **모든 pair**를 TP로 세고 전체 annotator-image에서 누적한다. 리포트의 per-group PQ는 진단용이며 평균값을 최종 PQ로 사용하지 않는다. RLE IoU의 float64 계산과 notebook의 torch float32 사이 작은 수치 차이는 가능하다.

## 테스트와 원본 notebook 동등성 검증

```bash
/workspace/.venvs/solar-eval/bin/python -m unittest discover -s tests -v
```

원본 notebook 비교에는 추가로 pandas/CPU torch가 필요하다.

```bash
/workspace/.venvs/solar-eval/bin/python -m pip install pandas==2.2.3
/workspace/.venvs/solar-eval/bin/python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu 'torch==2.9.1+cpu'
/workspace/.venvs/solar-eval/bin/python tools/check_notebook_parity.py \
  --notebook /path/to/self-evaluation-notebook.ipynb
```

checker는 실제 검토한 업로드 원본의 SHA-256을 확인하고 cell 3/4의 순수 평가 함수만 추출한다. 임의 notebook 전체를 실행하지 않으며, 다른 파일은 먼저 검토해야 한다. 2048² 합성 사례 8개에서 최대 차이 약 `2.71e-8`, 허용 오차 `1e-6`을 확인했다. 이 수치는 모델 PQ가 아니다.

대회 원본 JPEG 전체와 검증 예측 또는 baseline checkpoint가 준비되면, 동일 split에서 0.37 baseline과 신규 모델을 비교한다. public NSO 참조 JPEG는 GT 그림 확인용이며 대회 이미지와 동일한 학습 파일로 간주하지 않는다.

## annotation 전수 감사

연결성·면적·원반 제한·overlap·spine 길이와 GT union+CC 왕복을 다시 검사하려면 분석용 의존성을 설치한다.

```bash
/workspace/.venvs/solar-eval/bin/python -m pip install -r requirements-analysis.txt
/workspace/.venvs/solar-eval/bin/python tools/audit_annotations.py \
  /workspace/datasets/solar_filament/MAGFiLO_1.0_Annotations_kaggle2026_train.json \
  --submission /path/to/submission.csv \
  --output /workspace/artifacts/new_annotation_audit.json
```

감사는 원본 annotation을 변경하지 않는다. 기존 evidence는 `docs/evidence/annotation_audit.json`에 있다. GT 자체에서 생성한 union+CC 진단 점수는 **실제 모델 점수로 보고하면 안 된다**.
