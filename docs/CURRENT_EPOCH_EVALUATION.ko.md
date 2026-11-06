# 중단 시점 가중치 평가와 새 OOF 실험

2026-10-31에 받은 `epoch30_review_20261031T132927Z_part01..13.zip`을 복원했다.
2,011개 파일·234,962,241바이트의 SHA-256과 읽기 중 ZIP CRC가 모두 일치했다.
수집 중 변경되거나 크기 제한으로 제외된 파일은 없다. 첨부 Python은 실행하지 않고,
소스를 데이터로 읽어 저장소의 검토된 코드와 비교했다. 버전은 `5c1fb752ac88`이다.

## 실행이 끝나지 않았던 이유 정정

- Native는 5개 fold 모두 30epoch 학습·rich 추론이 끝났다(707장).
- YOLO와 전체 파이프라인은 fold0~3이 30epoch·추론 완료다(566장).
- Fold4 YOLO는 history상25epoch 완료, 26epoch의743/902batch에서 중단됐다.
- `paused/fixed_final_epoch100`은 native component의 100epoch 스케줄 최종 표시다.
  Stage30 launcher는 학습을30에서 중단하고 별도 snapshot/추론을 수행한다.
  따라서 그 JSON만으로 100epoch 완료조건 오류라고 설명했던 이전 진단은 잘못이었다.
- 실행 순서는 각 fold의 native30→rich 추론→YOLO30→전체 추론이며, 5개 완료 후 통합 평가다.

## 실제 재채점

Native는707장·1,154주석그룹·8,199객체를 평가했다. Strict IoU>0.5 all-pairs와
global IoU합/TP/FP/FN 누적 방식을 유지했다. 실제 과거91장 개발PQ와 직접 비교하지 않는다.

| Native 단독 | PQ | TP | FP | FN |
|---|---:|---:|---:|---:|
| Base0.5·area300 | 0.389678 | 5193 | 4568 | 3006 |
| Four-flip TTA0.5·area300 | 0.396476 | 5217 | 4409 | 2982 |
| CLAHE 입력 | 0.299639 | 4779 | 7655 | 3420 |
| 기존 guarded boundary | 0.389701 | 5186 | 4575 | 3013 |
| 기존 guarded consensus | 0.389960 | 5194 | 4583 | 3005 |
| Base small-object recall | 0.368917 | 5319 | 5889 | 2880 |

TTA는5개 fold 모두 개선했다. 반면 native 입력의 CLAHE는 FP가 크게 늘어 미채택한다.

YOLO+native의 완료된4fold·566장·922주석그룹·6,546객체 결과는 다음과 같다.
이는 전체707장 OOF가 아니다.

| 전체 파이프라인 방법 | PQ | TP | FP | FN |
|---|---:|---:|---:|---:|
| 기존 선택 recipe | 0.434495 | 4228 | 2429 | 2318 |
| 마지막 추가/제거 생략 | 0.425548 | 4035 | 2280 | 2511 |
| native base만 CLAHE 교체 | 0.434641 | 4224 | 2437 | 2322 |
| YOLO 전처리 후보 limb만 사용 | 0.434259 | 4162 | 2315 | 2384 |

## 추가 8개 후처리

Native707장 전체에서, 추가 학습·신경망 추론 없이 캐시로 고정8개 규칙을 비교했다.
모든 후보 CSV 생성 후 채점했고, baseline5개 CSV는 기존 TTA 평가와 byte 단위로 같다.
결과는 현재 OOF로 선택한 개발 후보이며 독립 test 확인이나 자동 채택이 아니다.

- 최고: TTA threshold0.35·area300·foreground 평균≥0.8·4개 view 중2개 이상이
  객체 면적의50% 이상 지지. PQ **0.411520220**, TP5194/FP3666/FN3005.
- 기존 TTA 대비 ΔPQ **+0.015044185**, FP743개 감소·FN23개 증가다.
  주효과는 FP 억제이며 FN 문제를 해결했다고 해석하지 않는다.
- 5개 fold와6개 관측소 모두 PQ가 상승했다. 물리 이미지 단위 paired bootstrap2,000회
  조건부95% Δ구간은[+0.011910900,+0.018127604]. 설정 선택 편향과 시간 상관을 제거한
  독립 검증 구간은 아니다.
- 낮은 threshold를 그대로 쓰면 FN은2854로 줄지만 FP4729·PQ0.394453으로 악화됐다.
- 보수적인 novel candidate 추가는 FN13개 감소·FP22개 증가·PQ0.396670의 작은 개선이다.
- 이 후보를 기존 전체 YOLO+native recipe에 자동 적용하지 않았다.

## 추가 학습 없는 Colab 실행

`notebooks/SolarFilament_CurrentEpoch_Evaluate_CopyCell.py` 전체를 새 Colab T4 셀에 복사한다.
노트북 파일은 `notebooks/SolarFilament_CurrentEpoch_Evaluate.ipynb`다.

- Drive에 남은 원본 코드와 그 source manifest를 검증해 임시 작업 폴더에 복원한다.
- 이미 끝난 fold0~3 전체 결과와 native5fold rich를 그대로 재사용한다.
- Fold4는 `yolo/last_full.pt`의 **cv_epoch**와 저장 history/학습 identity를 확인해
  실제 완료 시점의 모델을 별도 복사한다. Checkpoint의 epoch를30으로 바꾸지 않는다.
- 저장된 모델로 fold4 검증141장의 YOLO 원시 추론과 기존 전체 recipe 조합만 수행한다.
- 실제 checkpoint가25라면 결과는 native30, YOLO[30,30,30,30,25]의 **혼합 학습량 OOF**다.
  5개 모두30epoch인 결과라고 표시하지 않는다. 모델·분할 출처는 같은 fold에 한정한다.
- 원본 가중치·학습 config·history·코드 포인터를 수정하지 않는다. 학습 runner를 호출하지 않는다.
- 출력은 `fivefold_study_v2/current_epoch_evaluation/native030_yolo_current_v1`이다.
  같은 셀을 다시 실행하면 완료한 이미지 추론을 재사용한다.
- 최종 CSV·평가·출처 metadata·실행소스와 **새 fold4 raw cache만**24MiB 미만 ZIP으로 묶는다.
  기존707장 native cache와566장 YOLO cache를 다시 업로드할 필요가 없다.
- 셀은 로그를 실시간으로 보여주고 완료 후 결과 ZIP을 자동 다운로드한다.

## 검증과 범위

새 exporter6개·기존 full pipeline3개·YOLO checkpoint3개, 총12개 CPU 테스트를 통과했다.
새 통합 fixture는 학습 runner가 호출되면 실패하도록 막은 상태에서 실제 checkpoint
snapshot→4fold 재사용→마지막fold 추론→전체 OOF→재개의 흐름을 검증했다.
원본 파일 SHA가 그대로이며 재개 시 완료한 추론이 반복되지 않았다.
실제 업로드의566장 결과 재사용도 새 driver에서 출처·RLE 검증을 통과했다.

실제 사용자 가중치와 전체 이미지는 Drive에 있어, fold4 GPU 추론은 이 환경에서 실행하지 않았다.
따라서 혼합 학습량707장 전체 파이프라인의 새 PQ는 아직 없다.

근거: `/workspace/artifacts/epoch30_restore_audit.json`,
`epoch30_native_independent/report.json`, `epoch30_full_partial_report.json`,
`epoch30_native_experiments_v1/report.json`, `epoch30_current_driver_reuse_audit.json`.
