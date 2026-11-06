# 첫 A100 100에폭 결과 및 교차 모델 후처리

업로드된 `labels_my-project-name_2026-10-13-10-06-14.zip`은 학습 결과 번들이다.
100에폭이 완료됐으며, 최고 체크포인트는 84에폭이다. ZIP에는 설정·로그·검증 CSV·소스가 있고
가중치는 포함하지 않는다. `best.pt`와 재개용 `last.pt`의 저장 위치는 설정의 Drive RUN이다.

## 학습 결과 재검증

- 실사 train 616장, validation 91장 / 101개 annotator 그룹 / 정답 708개를 유지했다.
- 실제 합성 이미지 462장과 정답 JSON·분할 SHA가 준비한 소스와 일치한다.
- 1–100 에폭이 연속이며 총 40,000개 crop, optimizer 10,000회 갱신을 기록했다.
- 1–3·95–100 에폭은 실사만, 4–94 에폭은 합성 비율 20% 설정대로 진행했다.
- 최고 PQ는 84에폭 0.4299305777, 마지막 100에폭은 0.4214823050이다.
- 같은 100에폭 스케줄 안의 30에폭 값은 0.4040564158이다. 더 오래 학습하면서 개선됐지만,
  별도 30에폭 스케줄과의 대조 실험은 아니다.
- 9개 검증 CSV의 PQ·SQ·RQ·TP·FP·FN을 모두 독립 재계산해 재현했다.
- 업로드 소스는 준비한 worker와 맨 앞 빈 줄 하나만 다르다. 업로드 Python을 실행하지 않고 확인했다.
- 합성 데이터의 순효과는 무합성 대조군이 없으므로 단정할 수 없다.

## 같은 고정 개발셋에서 얻은 결과

| 방식 | PQ | TP | FP | FN |
|---|---:|---:|---:|---:|
| 기존 tile 보정 최고 | 0.464939 | 481 | 213 | 227 |
| 학습한 U-Net 단독, 84에폭 | 0.429931 | 479 | 315 | 229 |
| U-Net 면적 400 이상 필터 | 0.431437 | 461 | 273 | 247 |
| 두 모델 지원도 기반 PQ 최고 | **0.468608** | 470 | 182 | 238 |
| 기존 정답 보존 + 확인된 U-Net 누락 복구 | **0.465562** | 483 | 215 | **225** |

PQ 최고 후보는 기존 매칭 11개를 잃는다. FN 감소를 이어갈 기준으로는 마지막 행을 보관한다.
기존 정답 481개를 모두 보존하고 2개를 복구하며, 평가 기준 FP는 2개 증가한다.
두 후보 모두 저장 CSV를 다시 읽어 재평가했고, 겹침·동일 마스크 중복·추가 매칭 쌍이 없다.
**목표 PQ 0.5는 아직 달성하지 못했다. 이 수치는 고정 개발셋 튜닝 결과이며 Kaggle 점수가 아니다.**

새 모델은 기존 모델이 놓친 정답 51개를 별도로 맞혔지만, 이는 GT를 이용한 사후 진단이다.
그 51개를 실제로 모두 복구한 결과가 아니다. 단순 추가·union·교차 모델 경계 보정·낮은 confidence
후보 추가는 대부분 FP 증가 또는 기존 매칭 손실로 성능을 떨어뜨렸다.

누락 복구 후보는 다음 전역 규칙으로 만든다. U-Net foreground 임계값 0.5의 면적 300 이상
연결요소 중 기존 인스턴스 최대 IoU가 0.1 이하이고 기존 foreground와의 면적 중첩이 25% 이하인
후보를 검사한다. 기존 geometric/preprocessing/native-tile YOLO 추론 중 적어도 한 계열에서
confidence 0.6 이상인 마스크와 IoU 0.3 이상일 때만 추가한다. 기존 점유 픽셀은 제거하고,
추가 후에도 면적 300 이상이어야 한다. 기존 626개 물리 마스크를 보존하고 4개를 추가했다.
이 규칙의 후보 생성에는 정답을 사용하지 않았다.

## 다음 추론

`tools/colab_export_native_predictions_cell.py`를 Colab 새 셀에 전체 복사한다.
다음 RUN의 84에폭 `best.pt`를 불러온다.

`/content/drive/MyDrive/SolarFilament_NativeUNet_A100/20261028T030808Z_bedf3e21`

추가 학습 없이 base/hflip/vflip/hvflip을 추론하고 native 좌표로 역변환한 확률 평균을 저장한다.
Base와 평균 TTA 각각 foreground 5개·spine 3개 임계값 RLE, 면적 100 이상 연결요소와 실제
foreground/spine 평균·최대·p90 확률을 제공한다. 개별 view의 foreground 임계값 0.5도 저장한다.
기존 학습 코드는 spine을 보조 손실로 학습했지만 검증 추론에는 foreground만 사용했다.
이번 출력으로 확률 기반 FP 필터와 spine 기반 분할을 시험할 수 있다. 개선 여부는 아직 측정하지 않았다.

완료된 이미지마다 SHA를 포함해 Drive에 원자적으로 저장한다. 같은 A100·설정·코드를 다시 실행하면
완료된 이미지는 건너뛰고, 중단 당시 이미지만 처음부터 처리한다. 가중치나 실행 조건이 바뀌면 기존
출력과 섞지 않는다. 결과는 RUN의 `best84_rich_predictions_tta.zip`이다. 모델 가중치는 넣지 않는다.

## 검증 근거

- `docs/evidence/a100_uploaded_results_verification.json`: 9개 제출 형식 CSV 독립 재평가.
- `docs/evidence/a100_uploaded_history_audit.json`: 학습 로그·스케줄·소스 감사.
- `docs/evidence/native_fusion_saved_verification.json`: 저장 후보 10개 독립 재평가와 GT 매칭 전이.
- `docs/evidence/native_*experiments.json`: 각 전역 후처리 비교 및 후보 생성 규칙.
- `tests/test_native_export_cell.py`: 기존 foreground와 배열 일치, native tile stitching,
  실제 확률 통계, 손상 감지·재개, 비대칭 TTA 역변환·평균을 검증했다.
- 전체 112개 테스트 통과. 새 export 셀의 실제 GPU 실행은 아직 하지 않았다.
