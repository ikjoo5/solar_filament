# 전처리·후처리 3차 결과와 픽셀 경계 보정

후속 실제 픽셀 검증과 형태 필터 결과는 [4차 실험](PIXEL_POSTPROCESS_ROUND4.ko.md)에 정리했다. 아래 내용은 타일 업로드 전 3차 실험 당시의 기록이다.

현재 전체 검증 최고는 **PQ 0.4607591314**다. 91개 실제 영상, 101개 annotator 그룹, 708개 정답에서 계산했으며 TP479/FP215/FN229다. 목표 0.50은 아직 달성하지 못했다. 리더보드에서 확인한 점수는 사용자가 알려준 기존 제출의 0.37뿐이다.

| 전역 설정 | PQ | TP | FP | FN |
|---|---:|---:|---:|---:|
| 최초 union+CC/conf0.25 | 0.431752 | 415 | 172 | 293 |
| base/scale/vflip 합의 + closing5 + area300 | 0.457152 | 467 | 199 | 241 |
| 조건부 전처리 후보 복구 | 0.459118 | 479 | 215 | 229 |
| 위 결과의 Gaussian 경계 평활화 σ1.2/threshold0.6 | **0.460759** | **479** | **215** | **229** |
| 경계 평활화 threshold0.5 대안 | 0.459875 | 480 | 214 | 228 |

가장 높은 PQ와 가장 낮은 FN이 항상 같은 설정에서 나오지는 않는다. threshold0.5 대안은 FP와 FN을 하나씩 더 줄이지만, matched IoU 합계가 낮아 최고 PQ는 threshold0.6이다. 두 설정 모두 개별 이미지나 정답을 보고 마스크를 고르는 규칙을 사용하지 않는다.

## 새 업로드와 원시 캐시 검증

`20261027T125130Z_2312e759_results.zip`에는 68개 제출 CSV, summary, 8전처리 원시 캐시가 있었다. 모든 PQ/SQ/RQ/TP/FP/FN을 별도로 재계산해 1e-12 이내에서 일치함을 확인했다. 입력 annotation/checkpoint/train/val SHA256도 실제 파일 및 이전 실행과 같다. 728개 캐시 레코드의 13,188개 마스크가 91장을 포함하며 max_det에 도달한 레코드는 없다.

기존 base와 새 limb의 마스크·confidence는 91장 모두 정확히 같다. 따라서 이를 독립 투표로 중복 집계하지 않는다.

## 완료한 비교

- 새 전처리 68개: 전체 기존 최고 0.457152를 넘지 못했다. 대비 향상과 세부 구조 강조가 항상 검출 향상으로 이어지지는 않았다.
- 기존 geometric TTA와 새 전처리 혼합 84개 + reference 1개: 기존 최고를 넘지 못했다.
- 조건부 후보 복구 32개: 최고 0.459118. 물리 영상 기준 27개 인스턴스를 추가해 평가 FN12개 감소, FP16개 증가.
- 인스턴스별 경계 평활화·팽창·침식 11개 + reference 1개: 최고 0.460759. 마스크 수 626개 유지, SQ 0.671903→0.674305, FP/FN 유지. 단순 1픽셀 팽창은 PQ0.423121, 침식은 0.449013으로 악화됐다.

설정 수에는 reference와 동일 결과가 포함된다. 같은 development 집합에서 비교한 실험 수이며 독립 검증 횟수가 아니다.

## 현재 최고 설정의 순서

1. 기존 limb 보정을 그대로 적용한다.
2. base2048/scale2560/vflip2048의 confidence≥0.20 마스크를 각 view 안에서 union한다.
3. 3개 view 중 2표 이상인 foreground를 선택한다.
4. 사전 면적 필터 없이 5×5 ellipse closing을 적용한다.
5. 8-connected components 중 area≥300을 초기 인스턴스로 유지한다.
6. 새 전처리 후보를 confidence 내림차순으로 순회한다. confidence≥0.30, 기존 geometric view 후보(confidence≥0.10)와 IoU≥0.30, 기존 foreground와 겹치는 비율<0.10, 겹침을 뺀 면적≥300 조건으로 추가한다. 새 limb는 복구 후보에 중복 포함하지 않는다.
7. 각 인스턴스의 이진 마스크에 Gaussian σ1.2를 적용하고 0.6으로 이진화한다. 면적 비율이 원본 대비 0.5–1.5 밖이거나 원본 IoU<0.5이면 해당 원본을 유지한다.
8. 확장 영역이 충돌하면 가까운 원본 마스크에 픽셀을 배정한다. 원본 인스턴스의 픽셀 소유권을 보호하고, 정확한 거리 동점은 입력 순서를 유지한다.

정답은 전체 예측 생성 후 점수 계산에만 사용한다. 결과는 겹침·중복·추가 매칭쌍이 0이다.

## 원본 픽셀을 쓰는 다음 비교

`tools/refine_masks_with_pixels.py`는 GrabCut 6개, robust photometric likelihood + 거리 prior 6개, baseline 1개의 고정 grid다. 기존 복구 결과와 평활화 결과에 각각 적용할 예정이다. 얇은 구조의 core 소실, 밝기 대비 부족, 큰 면적 변화, 누락 픽셀은 원본 유지로 처리한다. 배경 표본을 포함한 ROI의 모든 픽셀이 있어야 보정한다. 보정 후 인스턴스를 다시 전체 CC로 분해하지 않는다.

공개 NSO 참고 JPEG 8장(9그룹/70GT)에서 코드를 실제 실행했다. 파일이 대회 이미지와 byte 단위로 동일한지 확인하지 못했으므로 **전체 91장 검증 점수가 아니다**.

| 입력 | 참고 8장 baseline PQ | 최고 픽셀 보정 PQ | FP/FN 변화 |
|---|---:|---:|---|
| 조건부 후보 복구 | 0.431287 | 0.444260 | 20/26→19/25 |
| 복구 + Gaussian 평활화 | 0.437531 | 0.445152 | 20/26→19/25 |

두 입력 모두 좁은 band3 + erode3 core GrabCut이 이 소규모 비교에서 가장 높았다. 넓은 band나 상대 core는 구조를 과도하게 줄여 악화되는 경우가 있었다. 전체 검증에서는 미리 정한 grid를 그대로 평가한다.

현재 cloud에는 사용자의 원본 JPEG/Drive 연결이 없다. `tools/colab_export_pixel_tiles_cell.py`를 기존 Colab에서 실행하면 후보 주변의 128×128 무손실 PNG만 추출한다. 캐시와 현재 입력 SHA256을 비교하고, confidence·후보 bbox만으로 타일을 선택한다. 정답 마스크로 위치를 고르지 않는다. 전체 캐시가 선택하는 타일은 3,236개이며 manifest에는 검증 91장을 모두 기록한다. ZIP은 각 24MiB 미만으로 나눈다. GPU 추론이나 재학습은 하지 않는다.

공개 참고 8장으로 실제 추출한 292개 타일은 약 2.51MiB였고 모든 픽셀·PNG SHA가 일치했다. 여러 ZIP에 동일한 manifest를 저장하므로, 전체 픽셀 검증을 위해 생성된 모든 part가 필요하다.

## 재현 근거

- `docs/evidence/preprocessing_results_verification.json`
- `docs/evidence/mixed_preprocessing_experiments.json`
- `docs/evidence/gated_fn_rescue_experiments.json`
- `docs/evidence/boundary_smoothing_experiments.json`
- `docs/evidence/pixel_refinement_reference_smoke.json`
- `docs/evidence/pixel_export_reference_smoke.json`
- `docs/evidence/target050_progress.json`

공식 평가의 strict IoU>0.5와 annotator별 반복 평가를 그대로 따른다. 모든 정답의 매칭 pair를 세는 공식 구현을 임의로 Hungarian matching으로 바꾸지 않았다.
