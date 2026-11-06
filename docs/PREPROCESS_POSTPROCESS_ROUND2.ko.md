# 전처리·후처리 중심 2차 실험

이 문서는 2차 실험 당시의 기록이다. 후속 전처리 결과와 현재 최고 PQ0.460759는 [3차 결과](PREPROCESS_POSTPROCESS_ROUND3.ko.md)에 정리했다.

## 현재 최고 검증 PQ: 0.4571518474

사용자는 FN을 줄이면서 FP의 무분별한 증가를 억제하고, 전처리·후처리를 특히 많이 비교하도록 요청했다. 같은 91장/101 annotation group/708 GT를 유지했다. 새 ZIP의 22개 점수와 입력 파일 hash가 기존 자료와 일치함을 확인했다.

| 설정 | PQ | TP / FP / FN |
|---|---:|---:|
| 최초 union+CC, confidence 0.25 | 0.431752 | 415 / 172 / 293 |
| 4-view 중 3표, confidence 0.20, area200 | 0.453525 | 455 / 193 / 253 |
| base+scale+vflip 중 2표, confidence 0.20, area300 | 0.454183 | 466 / 206 / 242 |
| 위 area300 결과에 closing5 후 다시 CC/area300 | 0.456621 | 467 / 200 / 241 |
| **면적 필터를 closing 뒤로 이동** | **0.457152** | **467 / 199 / 241** |

현재 최고는 겹침·추가 매칭쌍이 0이다. 최초 기준선보다 FN 52개 감소, FP 27개 증가이며 PQ는 +0.0254002다. 아직 목표 0.50은 초과하지 않았다.

최고 설정의 정확한 순서:

1. 기존 limb 보정.
2. base2048/scale2560/vflip2048을 각각 추론하고 confidence≥0.20인 객체들의 view별 foreground union 생성.
3. 3개 view 중 2개 이상이 지지하는 foreground 선택.
4. 면적 사전 필터 없이 5×5 ellipse kernel의 binary closing.
5. 8-connected components로 객체 생성.
6. area≥300인 객체 유지.

후처리 순서도 비교 대상이며, area 필터를 먼저 한 경우와 동일한 알고리즘이라고 취급하지 않는다.

## 완료된 실험 분기

- view 조합·confidence·면적 48개: 최고 0.454183.
- confidence-weighted foreground 26개 + reference 1개: 신규 최고 0.450594, 개선 없음.
- 객체 대응·view별 한 mask 제한 16개: 최고 0.445950, 개선 없음.
- 낮은 confidence 객체 대응 16개: 최고 0.434078, FP302/FN231. FN은 줄지만 FP 증가로 PQ가 낮아 채택하지 않는다.
- high-confidence seed에서 low-confidence 영역 복구 32개: 최고 0.451915, 개선 없음.
- closing/opening/hole 처리 12개(두 reference 포함): 최고 0.456621.
- morphology/area 순서 13개(reference 포함): 최고 0.457152.

설정에는 reference와 중복 결과가 포함되어 있어 고유 알고리즘 수로 합산하지 않는다. 전역 규칙만 비교했고, 정답별 최적 마스크 선택은 하지 않았다.

## 미검출 원인 분리

현재 FN241개 중 172개는 저장된 원시 후보(conf≥0.05, 4-view, max_det100/view)에 이미 IoU>0.5의 mask가 있었다. 28개는 best IoU 0.3~0.5, 41개는 0.3 미만이며 그중 28개는 raw 후보와도 overlap이 없었다.

이는 GT를 대조해 원인을 분석한 **candidate availability 진단**이다. 실제 선택 알고리즘의 recall이나 달성한 PQ가 아니다. 전체 708개 중 raw 후보가 매칭 가능한 GT는 639개지만 이를 정답으로 골라 예측하는 oracle는 사용하지 않는다.

이 결과는 후처리의 후보 선별에 여지가 크다는 근거다. 새로운 전처리로 낮은 신호의 검출을 안정화하고, 불안정한 후보를 합의·영상 근거로 억제하는 방향을 우선한다. 단순 confidence 하향은 FP와 객체 병합을 늘려 PQ가 하락할 수 있다.

## 다음 GPU 실험: 8개 전처리와 총 68개 고정 설정

모든 view는 기존 limb-corrected 영상에서 시작한다. 원본 좌표/2048 해상도와 uint8을 유지하고 원반 밖은 다시 0으로 만든다.

| view | 설정 |
|---|---|
| limb | 기존 입력 그대로 |
| clahe16 | clip1.5/grid16, 원본과 50% blend |
| clahe32 | clip2.0/grid32, 원본과 50% blend |
| percentile | 원반 내부 1–99% stretch, 원본과 25% blend |
| unsharp | Gaussian sigma1, gain0.5 |
| gaussian | sigma0.5 |
| bilateral | d5/color10/space3 |
| blackhat | ellipse15 응답을 0.5배 빼서 어두운 구조 강조 |

후처리는 single-view 48개, 대비 계열 3-of-4 6개, smoothing 계열 2-of-3 6개, 전체 8-view의 4/6표 8개, 총 68개다. 각 view의 confidence≥0.05 원시 mask/confidence/전처리 파라미터도 JSONL에 저장해 다음 로컬 비교에 사용한다.

`tools/colab_preprocessing_cell.py`는 다운로드가 필요 없는 기존 Colab의 새 셀용 코드다. 이전에 실행한 `SCRIPT`를 재사용해 새 subprocess/새 결과 폴더를 만들며, 이전 데이터·checkpoint는 수정하지 않는다. 유효한 reference preprocessing과 입력 배열·좌표·RLE·68개 candidate에 대한 테스트를 통과했다. 공개 참조 JPEG 1장에서 실제 checkpoint의 8-view CPU 추론도 완료했지만, 전체 91장 성능은 **아직 미측정**이다.

이 cloud machine에는 GPU와 사용자 Drive 연결이 없어 전체 cohort 추론은 기존 Colab GPU에서 실행해야 한다. 전처리 코드는 새 라이브러리를 요구하지 않는다. 돌아온 원시 캐시는 기존 scale/flip 캐시와 결합할 수 있으며, 동일한 baseline 입력을 독립 view처럼 중복 집계하지 않는다.

## 근거 파일

- `docs/evidence/four_view_results_verification.json`
- `docs/evidence/view_combination_experiments.json`
- `docs/evidence/confidence_fusion_experiments.json`
- `docs/evidence/instance_view_fusion_experiments.json`
- `docs/evidence/instance_view_lowconf_experiments.json`
- `docs/evidence/seeded_recovery_experiments.json`
- `docs/evidence/consensus_morphology_experiments.json`
- `docs/evidence/morphology_order_experiments.json`
- `docs/evidence/raw_candidate_availability.json`
- `docs/evidence/preprocessing_cpu_smoke.json`

검증 집합을 반복해 설정을 고르는 development 실험이다. 독립 CV나 실제 리더보드 상승으로 재분류하지 않는다.
