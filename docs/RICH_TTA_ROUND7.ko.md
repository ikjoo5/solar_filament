# Best84 반전 추론·확률 출력 후처리

업로드 `best84_rich_predictions_tta.zip`의 91장 추론 결과로 학습 없이 후처리를 비교했다.
최고 고정 개발셋 PQ는 **0.4725930445**이다. 기존 정답 매칭을 모두 보존하는 후보는
**PQ 0.4703621314, FN 218, FP 215**로, 직전 기준보다 FN 7개를 줄이고 FP를 유지했다.
목표 PQ 0.5는 아직 달성하지 못했다. 모든 수치는 반복 튜닝한 같은 91장 개발셋이며 Kaggle 점수가 아니다.

## 파일 검증

- 91개 이미지 / 101개 annotator 그룹 / 정답 708개를 그대로 사용했다.
- 91개 개별 기록의 SHA와 manifest가 일치하며, RLE 9,842개의 범위·크기·면적·중첩을 검사했다.
- exporter SHA는 직전 답변에서 제공한 셀과 정확히 일치한다.
- 원본 방향 base 마스크는 이전 84에폭 결과의 9개 threshold/area CSV와 모두 정확히 일치한다.
- 모델은 A100 BF16, native 2048 좌표의 base/hflip/vflip/hvflip 및 역변환 후 확률 평균을 사용했다.
- 원본 Drive 이미지 바이트·모델 가중치는 로컬에 없다. 이미지 해시는 export 내부 일치와 기존 결과
  재현을 확인했으며, 개별 연속 확률을 모델로 다시 계산한 검증은 아니다.

## 최종 저장 후보 독립 재평가

| 후보 | PQ | TP | FP | FN | 기존 483개 매칭 손실 |
|---|---:|---:|---:|---:|---:|
| 직전 기준 | 0.465562 | 483 | 215 | 225 | 0 |
| 최고 PQ | **0.472593** | 488 | **204** | 220 | 2 |
| 모든 기존 정답 보존 | **0.470362** | 490 | 215 | **218** | 0 |
| FN 최소 참고 후보 | 0.467443 | 499 | 242 | 209 | 0 |

최고 PQ 후보는 기존 483개 중 481개를 유지하고 7개를 복구한다. 직전 기준보다 FP 11개,
FN 5개가 줄었다. 정답 보존 후보는 483개를 모두 유지하고 7개를 복구하며 FP가 같다.
FN 최소 후보는 FP가 27개 증가해 PQ가 낮아, 참고용으로 보관한다.

22개 저장 CSV를 root에서 다시 읽어 독립 재채점했다. 모두 마스크 중첩·동일 마스크 중복·추가
매칭 쌍이 없으며, 공식 제공 평가 방식인 전체 IoU>0.5 쌍의 global PQ를 사용했다.

## 효과가 있었던 조합

1. 직전 630개 마스크를 base/TTA threshold 0.65 후보와 고유 매칭한다. 후보의 실제 foreground
   평균 출력은 0.8 이상, 원본과 IoU는 0.3 이상이어야 한다. 원본·base·TTA의 2/3 다수결을 쓰되,
   원본 대비 IoU 0.85 이상 및 면적 비율 0.5–1.5인 변화만 허용한다. 겹침은 원본까지의 거리로
   해소한다. 인스턴스 개수를 유지하면서 기존 매칭 2개를 추가로 복구했다.
2. TTA foreground threshold 0.35, 면적 300 이상, foreground 평균 출력 0.9 이상인 새 후보를
   검사한다. 4개 view 중 3개 이상이 후보 면적의 절반 이상을 덮어야 한다. 기존 마스크 최대 IoU
   0.1 이하·기존 합집합과 중첩 면적 25% 이하만 허용하고, 점유 픽셀을 제거한 뒤에도 면적
   300 이상이어야 한다. 경계 보정된 모든 원본을 기준으로 이 조건을 새로 계산한다.
3. 최고 PQ 후보는 TTA threshold 0.2의 foreground가 기존 마스크를 40% 미만 덮고, 픽셀별
   2-of-4 view foreground 지지가 25% 미만인 기존 마스크를 제거한다.
4. 정답 보존 후보는 위 제거를 면적 600 이하 마스크에만 적용하고, 각각 25%·5% 미만으로
   더 보수적으로 제한한다. 두 후보 모두 제거 전에 모든 경계 보정 원본의 픽셀을 예약해 새
   후보를 클리핑한다. 제거된 영역을 새 후보로 재활용하지 않는다.

실제 foreground 평균은 네트워크 출력의 요약값이며 인스턴스 정답 확률로 보정된 값은 아니다.
후보 생성과 모든 게이트에는 정답을 사용하지 않았고, 생성 후 정답으로 전역 설정을 비교했다.

## 효과가 없거나 제한적이었던 방법

- U-Net 단독 최선은 TTA·foreground 평균 필터를 쓴 PQ 0.454582이다. 기존 조합보다 낮다.
- 내보낸 spine 임계값 0.3·0.5는 foreground를 100% 덮었다. 0.7에서도 인스턴스별 최소
  coverage가 약 99.94%다. 현재 임계값 범위에서 가는 중심선이나 분할 씨앗으로 구분력이 없었다.
- spine 기반 52개 규칙은 실제 분할·제거가 없었다. 별도의 foreground 고확률 씨앗 watershed
  36개 규칙은 실제 분할을 만들었지만 FP 증가로 PQ가 좋아지지 않았다.
- 마지막 18개 Gaussian 경계 정리는 이미 보정된 최종 후보보다 모두 낮았다. 추가 smoothing을
  적용하지 않는다. 경계 다수결의 추가 15개 설정도 기존 최고점보다 좋아지지 않았다.

## 저장 위치와 재현

- 최고 PQ: `/workspace/artifacts/rich_boundary_composition_v1/tta_precision.csv`
- 정답 보존: `/workspace/artifacts/rich_boundary_composition_v1/tta_preserve.csv`
- FN 참고: `/workspace/artifacts/rich_boundary_composition_v1/base_preserve.csv`
- 고정 설정: `docs/evidence/rich_selected_recipe.json`
- 입력 검증: `tools/verify_rich_predictions.py`, `docs/evidence/rich_predictions_verification.json`
- 최종 독립 재평가: `tools/verify_rich_candidates.py`, `docs/evidence/rich_candidates_verification.json`
- 후보 생성: `tools/experiment_rich_boundary_refinement.py`,
  `tools/experiment_rich_boundary_composition.py`
- 분기별 비교: `docs/evidence/rich_*.json`

이번 결과에 추가 학습은 없다. 기존 체크포인트와 새로 제공된 91장 추론 캐시를 사용했으며,
후처리 분기·native 좌표·마스크 분할·저장 후 점수를 실제 코호트에서 검증했다.
