# 고해상도 타일 추론과 객체 결합 5차 결과

전체 검증 최고가 **PQ0.4649385759 / TP481 / FP213 / FN227**로 올라갔다. 기존 FN 우선 기준 PQ0.4607591314 / TP479 / FP215 / FN229보다 PQ0.0041794 상승, FP와 FN 각각 2개 감소다. 기존에 맞춘 479개 GT는 모두 유지했고 2개를 추가로 맞췄다. 목표 PQ0.50은 아직 달성하지 못했다.

| 전역 설정 | PQ | TP | FP | FN |
|---|---:|---:|---:|---:|
| 기존 예측 | 0.460759 | 479 | 215 | 229 |
| 기존 + 두 타일 해상도 객체 다수결 | 0.462637 | 477 | 217 | 231 |
| 위 다수결에 기존 예측 IoU≥0.85 안전 조건 | **0.464939** | **481** | **213** | **227** |

분석 집합은 같은 91개 물리 영상/101개 annotator 그룹/708개 GT다. 개발 집합의 설정 비교이며 Kaggle 리더보드 수치가 아니다.

## 새 업로드 검증

`20261027T205426Z_eba01cfb_tiled_results.zip`은 기존 checkpoint를 T4에서 추론한 결과다. 1024×1024 원본 crop 9개를 1536/2048 입력으로 각각 추론했다.

- 91장 × 18뷰 = 1,638개 레코드, 원본 2048 좌표의 마스크 15,785개.
- 1536 입력 마스크 6,564개, 2048 입력 9,221개.
- 내부 crop 경계의 4픽셀 이내에 닿은 마스크 1,271개.
- 검출 개수 상한 도달 0개, 빈 마스크 0개.
- 모든 RLE, crop 범위, 경계 표시를 독립 재계산해 확인했다.
- annotation/checkpoint/train/val, JSONL, worker source 해시가 일치한다.
- 91장 모두 원본 JPEG 및 limb 영상 해시가 이전에 업로드한 실제 픽셀 manifest와 같다.

첨부 `worker_source.py`는 검증용 데이터로 읽었으며 업로드 소스를 실행하지 않았다.

## 완료한 분기

1. **신규 객체 추가 48개 설정**: 기존 마스크를 보존하고 tile confidence, 두 입력 해상도의 지지, 기존 raw 후보의 확인, 겹침 비율을 비교했다. 낮은 confidence는 FN을 189까지 줄였지만 FP577/PQ0.387603으로 악화됐다. 높은 confidence에서는 후보 0개 추가가 최고였다.
2. **객체별 결합 24개 설정 + 원본 기준**: 두 타일 해상도별 최고 confidence 마스크와 기존 마스크를 union/majority/intersection/best-tile로 결합했다. 최고는 confidence0.35/대응 IoU0.3/majority, PQ0.462637이었다.
3. **결합 후 안전 조건 13개 설정 + 2개 기준**: 앞 분기의 상한 confidence 및 기존 예측과의 IoU를 후속 비교했다. 최고는 confidence0.35/대응 IoU0.3/majority/기존 IoU0.85, PQ0.464939다. 최초 grid와 후속 탐색을 구분해 기록했다.
4. **전역 foreground 투표 41개 설정(기준 2개 포함)**: 실제 patch coverage로 정규화한 타일 투표와 기존 base/scale/vflip, 현재 foreground의 결합을 비교했다. 내부 crop margin128을 분자와 분모에서 같이 제외했다. 기존 기준을 넘지 못했다.

개별 정답이나 이미지별로 최적 마스크를 선택하지 않았다. 같은 결과를 내는 설정과 기준이 포함되어 있어 위 숫자를 고유 알고리즘 수 또는 독립 검증 횟수로 해석하지 않는다.

## 최고 규칙

기존 PQ0.460759 예측을 출발점으로 사용한다.

1. 잘림 표시가 없는 tile 후보 중 confidence≥0.35를 사용한다.
2. 각 tile 후보를 기존 인스턴스 중 IoU가 가장 높은 하나에만 대응시킨다. 대응 IoU가 0.3 이상이어야 한다.
3. 인스턴스별로 1536과 2048에서 각각 confidence가 가장 높은 마스크 하나를 고른다. 두 해상도 중 하나라도 없으면 기존 예측을 유지한다.
4. 기존 예측과 두 tile 마스크의 픽셀별 2-of-3 투표를 계산한다.
5. 후보의 면적 비율이 기존 대비 0.5–1.5이고, **기존 예측 마스크와의 IoU≥0.85**일 때만 수정한다. 이 IoU는 GT와의 IoU가 아니다.
6. 새 픽셀 충돌은 가까운 기존 예측에 배정하고, 기존 픽셀 소유권을 보호한다.

626개 출력 인스턴스 수를 유지했다. 330개가 변경되고, 두 해상도 지지가 없는 226개와 안전 조건을 통과하지 못한 70개는 원본을 유지했다. 출력 겹침·중복·추가 qualifying pair는 0개다.

저장 CSV를 부모 작업에서 다시 읽어 공식 global all-pairs PQ를 재계산했다. IoU 합은 322.992151→325.921942, SQ0.674305→0.677592, RQ0.683310→0.686163이다.

추가 매칭 두 건은 GT 진단에서 기존 최대 IoU0.480482→0.506579, 0.494460→0.504828로 나타났다. 이 GT 정보는 결과 설명에만 사용한다.

## 원시 후보 진단

confidence0.05에서 기존 12뷰의 후보가 IoU>0.5로 커버 가능한 GT는 650개, 새 잘리지 않은 타일 후보까지 포함하면 665개였다. 새 15개는 모두 기존 FN이며 중앙 면적은 691픽셀이다. 기존 FN229개 중 원시 후보가 있는 수는 171→186개다.

또한 이미 맞춘 GT479개 중 316개는 더 높은 IoU의 타일 마스크가 있었고, 101개는 IoU0.05 이상 높았다. 이는 **GT 기반 candidate availability 진단**이다. 달성한 recall이나 정답으로 후보를 고른 예측 결과가 아니다.

## 다음 단계

같은 YOLO의 confidence·영상 대비·단순 형태·타일 투표만으로 FN을 많이 줄이면 FP도 크게 늘었다. 추가 학습에서는 원본 크기의 foreground/spine decoder를 갖는 ConvNeXt-T U-Net을 별도 모델로 비교한다. 기존 616개 실제 학습 영상만 사용하고, 이미지부터 고른 뒤 annotator 하나를 뽑는다. native crop과 작은 객체 중심 샘플링, foreground BCE+Dice 및 보조 spine loss를 사용한다. 전체 91장 공식 PQ로 checkpoint를 고른다.

새 모델의 성능은 아직 측정하지 않았다. Colab에서 원본 데이터와 GPU를 사용해야 하며, 현재 최고 YOLO 결과는 독립 기준 및 후속 앙상블 재료로 보존한다.

## 결과와 근거

- 최고 validation CSV: `/workspace/artifacts/tiled_object_fusion_candidates_v2/c0.35_iou0.3_majority_guard0.85.csv`
- CSV SHA256: `2aebaff10bf14d7f18e544d6b28f730902fb2efa7a0555935f11ebf48474461f`
- `docs/evidence/tiled_candidate_audit.json`
- `docs/evidence/tiled_rescue_experiments.json`
- `docs/evidence/tiled_object_fusion_experiments.json`
- `docs/evidence/tiled_object_fusion_guard_experiments.json`
- `docs/evidence/tiled_pixel_consensus_experiments.json`
- `docs/evidence/tiled_best_verification.json`
