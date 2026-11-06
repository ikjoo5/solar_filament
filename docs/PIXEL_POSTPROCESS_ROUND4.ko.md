# 실제 검증 타일을 이용한 4차 실험

후속 고해상도 타일 추론으로 PQ/FN/FP를 함께 개선한 결과는 [5차 실험](TILED_INFERENCE_ROUND5.ko.md)에 정리했다. 아래는 픽셀 경계 보정 당시의 기록이다.

사용자가 보낸 `pixel_tiles_20261027T203313Z_fd420182_part01.zip`과 `part02.zip`을 함께 검증했다. 91장/101개 annotator 그룹에 대한 3,236개 무손실 PNG가 모두 정상이고, 입력 annotation/checkpoint/train/val SHA256이 이전 실험과 같다. 기존 두 626개 마스크 세트의 모든 보정 ROI가 완전히 포함되어 누락 픽셀로 인한 fallback은 없었다.

같은 검증 집합에서 픽셀 보정, 밝기 기반 후보 선별, 형태 필터를 비교했다. 이번 분기에서 FN과 PQ를 함께 개선하는 규칙은 찾지 못했다. 수치상 가장 높은 PQ는 **0.4608488351**이나 기존 최고보다 **0.0000897** 높고 FN이 3개 증가한다. FN 복구를 위한 다음 단계의 기준은 기존 **PQ0.4607591314 / FP215 / FN229**도 함께 유지한다. 목표 PQ0.50은 미달이다.

| 설정 | PQ | TP | FP | FN |
|---|---:|---:|---:|---:|
| 기존 복구 + 경계 평활화 | 0.460759 | 479 | 215 | 229 |
| 위 결과에서 둥글고 짧은 마스크 일부 제거 | **0.460849** | 476 | 209 | 232 |
| 픽셀 경계 보정 최고: Gaussian σ2 → GrabCut band3 | 0.452079 | 476 | 218 | 232 |
| 낮은 confidence + 대비 선별의 최저 FN 설정 | 0.450302 | 500 | 282 | 208 |

위 수치는 전체 91장/101그룹/708GT에서의 전역 누적 PQ다. 다른 이미지마다 정답을 보고 최적 규칙을 선택하지 않았다. 리더보드 점수로 보고하지 않는다.

## 수행한 비교

- 기존 복구 결과와 평활화 결과 각각에 13개 고정 픽셀 보정 설정: **26개**, 각 입력의 원본 유지가 최고였다. GrabCut band3/5/7 × erode3/relative core 및 photometric likelihood + signed-distance prior를 비교했다.
- 기존 인스턴스를 유지하고 새 전처리 후보만 추가하는 밝기 선별: confidence0.15/0.20/0.25/0.30 × 대비 z0/0.5/1 × support1/2, **24개**. 기존 복구 결과를 넘지 못했다.
- 같은 방식의 강한 대비 z2/3/4: **24개**. FP와 함께 TP도 제거했다. confidence0.30/support1에서 z2→3→4로 높이면 FP215→207→203, FN231→236→239다.
- 기존 최고와 낮은 confidence 복구 결과 각각에 16개 전역 형태 필터: **32개**. mask covariance elongation, contour compactness, 내부 두께, bbox 대각선을 사용했다. 소폭 상승한 최고 규칙은 elongation<1.5이면서 compactness>0.2인 8개 물리 인스턴스를 제거한다.
- Gaussian σ0/1/2/3 이후 좁은 band1/2/3 GrabCut + 원본 유지: **13개**. 표면 질감을 줄이면 직접 GrabCut보다 낫지만 기존 기준에 미치지 못한다.

총계에는 원본 유지 및 같은 결과를 내는 설정이 포함된다. 독립된 알고리즘 수나 독립 검증 횟수로 해석하지 않는다.

## 경계 보정이 실패한 이유의 진단

처음 26개 중 가장 높은 실제 보정 결과인 `rescue/grabcut_b3_erode3`를 기존 평활화 기준과 비교하면 3개 GT를 복구하고 9개를 놓친다. 470개 GT는 계속 매칭되지만 이들의 IoU 합도 5.1593 감소한다. 단순히 0.5 경계에서 몇 개가 교차한 문제만이 아니라 전체 마스크 모양도 악화됐다.

진단용 overlay에는 실제 타일 픽셀과 GT/기존/보정 contour를 그렸다. 이 사례 선정에만 GT를 사용했다. 픽셀의 어두운 부분만 좇는 경계가 수작업 정답의 객체 범위와 항상 같지는 않으며, 희미한 가지와 표면 질감의 혼동이 남는다.

공개 NSO 참고 8장은 대회 원본과 raw/limb SHA가 **0/8 일치**했다. 확보한 대회 타일과 픽셀을 직접 비교해도 값이 다르다. 따라서 이전의 공개 참고 8장 개선은 구현 시험으로만 남기고 실제 검증 개선으로 사용하지 않는다.

## 다음 비교: 원본 해상도 타일 추론

현재 후보의 confidence를 더 낮추거나 밝기/모양만으로 고르는 분기는 FP 또는 FN 손실 때문에 한계가 나타났다. 다음에는 동일 checkpoint에 **1024×1024 원본 crop을 1536/2048 입력으로 추론**해 다른 후보와 경계 정보를 확보한다. 전체 영상 limb 보정 후 crop하므로 방사형 보정의 문맥을 바꾸지 않는다.

1024 crop을 (0,512,1024) 좌표의 9개 위치에서 추론하고, 원본 2048 좌표로 마스크를 되돌린다. crop 내부 경계에 닿아 잘렸을 가능성이 있는 마스크는 표시한다. 원시 confidence와 마스크를 저장해 기존 geometric TTA/전처리 후보와 로컬에서 선별·앙상블을 비교한다. 학습이나 이전 view 추론은 반복하지 않는다.

이 단계는 사용자의 원본 이미지가 있는 Colab GPU에서 실행해야 한다. cloud에는 GPU와 전체 JPEG가 없다. 현재 업로드 타일은 기존 후보 주변만 포함하므로, 이를 2048 전체 영상인 것처럼 검은 부분으로 채워 모델에 넣지 않는다.

## 근거 파일

- `docs/evidence/pixel_tiles_verification.json`
- `docs/evidence/pixel_refinement_full91_experiments.json`
- `docs/evidence/pixel_refinement_diagnostics.json`
- `docs/evidence/photometric_rescue_experiments.json`
- `docs/evidence/photometric_rescue_highcontrast_experiments.json`
- `docs/evidence/shape_quality_experiments.json`
- `docs/evidence/smoothed_pixel_refinement_experiments.json`

최고 형태 필터 CSV는 `/workspace/artifacts/shape_quality_candidates_v1/current_best_elong_lt1.5_compact_gt0.2.csv`, FN 우선 기준 CSV는 `/workspace/artifacts/boundary_smoothing_candidates_v2/gaussian_s1.2_t0.6.csv`다. 둘 다 **validation 예측**이며 test 제출 파일은 아니다.
