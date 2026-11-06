# 기존 YOLO 검증 결과와 다음 실험

## 검증 범위

사용자가 실행해 제공한 `20261027T112905Z_947c4b1f_results.zip`을 분석했다. 실제 validation은 영상 91장, annotation group 101개, 정답 객체 708개다. 정답·체크포인트·train/val split의 SHA-256이 앞서 확인한 파일들과 일치한다.

실행 환경은 torch 2.11.0+cu130, Ultralytics 8.4.182, NumPy 2.1.3, SciPy 1.17.0, OpenCV 5.0.0.93, pycocotools 2.0.11이다. 10개 CSV를 제공 evaluator 의미의 compressed-RLE 계산으로 재평가했고 모든 PQ/SQ/RQ/TP/FP/FN이 일치했다. 원본 이미지를 다시 추론하거나 새로운 모델을 학습한 것은 아니다.

이 점수는 같은 development validation에서 설정을 고르는 결과다. 리더보드 0.37보다 성능이 올랐다는 증거로 직접 비교하지 않는다. 같은 영상의 annotator는 독립 표본이 아니며, 기존 split에는 서로 다른 관측소의 train 영상과 20초 차이인 validation 영상 두 장도 있어 완전히 독립적인 일반화 점수로 해석하지 않는다.

## 1. 제공된 10개 설정

| confidence | union+CC PQ | 개별 mask 유지 PQ |
|---:|---:|---:|
| 0.15 | 0.42240 | 0.43960 |
| 0.25 | **0.43175** | 0.43908 |
| 0.35 | 0.40897 | 0.41341 |
| 0.45 | 0.37424 | 0.37583 |
| 0.55 | 0.32087 | 0.32132 |

기존 설정인 union+CC, confidence 0.25의 SQ는 0.67364, RQ는 0.64093, TP/FP/FN은 415/172/293이다. confidence를 올리면 마스크가 맞는 객체의 SQ는 조금 상승하지만 미검출이 늘어 전체 PQ는 낮아진다.

## 2. raw preserve 최고점을 바로 최종 설정으로 고르면 안 되는 이유

preserve 0.15는 PQ 0.43960, TP pairs 584, FP 391, FN 201이다. 실제 매칭된 고유 GT는 708−201=507개인데 TP pairs는 584개라 **77개의 추가 매칭쌍**이 존재한다. GT 71개가 복수 prediction과 각각 IoU>0.5로 매칭되었다.

제공 evaluator는 이를 문자 그대로 모두 집계한다. 이 계산을 임의로 Hungarian matching으로 바꾸지는 않지만, 겹침·중복을 허용하는 제출 정책과 실제 객체 품질을 별도로 확인한다. 보이는 PQ 차이가 전부 더 나은 객체 분리 때문이라고 단정하지 않는다.

91개 실제 영상을 단위로 annotator를 함께 묶어 paired bootstrap 5,000회를 수행했다.

- preserve 0.15 − union 0.25: development ΔPQ +0.00785, 95% resampling 구간 약 −0.01086~+0.02771.
- preserve 0.25 − union 0.25: development ΔPQ +0.00732, 구간 약 −0.00374~+0.01894.

두 구간 모두 0을 포함한다. 이 bootstrap은 설정 선택·학습 누수·다른 영상 사이 시간 상관을 보정한 독립 성능 검증이 아니다. 작은 차이를 확정된 일반화 이득으로 보고하지 않는다.

## 3. 실제 병목

### 선택된 예측에서 완전히 빠진 객체

기존 union+CC 0.25에서 미검출 293개 중:

- 어떠한 예측과도 겹치지 않음: **185개**.
- best IoU (0,0.1): 3개.
- best IoU [0.1,0.3): 41개.
- best IoU [0.3,0.4): 29개.
- best IoU [0.4,0.5]: 35개.

따라서 경계를 미세하게 바꾸는 것만으로 모든 문제를 해결할 수 없다. 185개는 해당 threshold/후처리의 출력에 덮이지 않았다는 뜻이며, neural network의 낮은 confidence proposal까지 전혀 없었다고 입증한 것은 아니다.

### 작은 객체

면적 quartile별 고유 GT 검출률은 39.5% / 62.7% / 67.2% / 65.0%다. 첫 quartile은 661.5px 미만이며 각 quartile은 GT 177개다. 원본 해상도 유지, 작은 객체 중심 crop 샘플링, mask supervision 해상도가 우선 실험 대상이다.

### category와 경계

Unidentifiable의 검출률은 47.6%, Left/Right는 65.4%/67.6%다. category는 크기 등과 혼재되어 있으므로 독립적인 category 효과나 밝기 효과로 단정하지 않는다. Unidentifiable를 학습에서 제거하지 않는다.

매칭된 마스크의 예측/GT 면적비 중앙값은 1.079다. 미검출 중 49개는 prediction union이 GT 픽셀의 80% 이상을 덮지만 개별 객체 매칭에 실패해, 일부 경계·병합·분리 문제도 있다. 다만 overlap graph만으로 오류 원인을 확정하지 않는다.

## 4. 실행 우선순위

1. **같은 checkpoint의 저비용 검증:** raw preserve의 중복 억제·겹침 해소와 제한적인 경계 보정을 비교한다. GT를 보고 객체별 규칙을 고르는 oracle는 사용하지 않는다.
2. **다음 추론에서 후보 정보 보존:** CSV 외에 원시 mask와 실제 confidence를 저장하고, confidence 0.05~0.25의 작은 후보군을 비교한다. 현재 CSV에는 정확한 confidence가 없어 중간 threshold를 임의 복원하지 않는다.
3. **YOLO 학습 해상도 비교:** 현재 mask_ratio=4와 2/1을 같은 split에서 비교하고 checkpoint를 전역 PQ로 선택한다. 메모리와 실제 target/prediction stride도 확인한다.
4. **주력 신규 모델:** native-resolution 1024² 타일의 ConvNeXt-T/S U-Net + 제공 spine 보조 head. 작은 GT의 샘플 빈도를 확보하고 배경 hard negative를 함께 학습한다. BCE+Dice 기준선에서 spine/boundary를 개별 ablation한다.
5. **검증·최종화:** 기존 91장은 baseline 비교용으로 유지하되, 최종 모델의 일반화는 관측 시간/관측소를 고려한 별도 grouped validation에서 확인한다. 새 분할에서 기존 checkpoint를 독립 평가 모델처럼 재사용하지 않는다. 상위 모델만 앙상블한다.

Mask2Former나 복잡한 global-instance branch를 먼저 추가하기보다, 실제 병목인 작은 객체 검출과 낮은 threshold의 FP 제어를 우선한다. 추가 architecture는 오류 분석과 같은 기준의 PQ 이득이 있을 때 채택한다.

## 5. 추가 후처리 13개 고정 후보의 실측

같은 binary mask CSV에 GT를 보지 않는 규칙으로 입력순서/면적순서 disjoint 할당, mask-IoU NMS 0.7/0.9, 1px cross dilation/erosion을 비교했다. 원래 confidence 숫자는 CSV에 없으므로 입력 순서를 confidence라고 단정하지 않았다. GT별 최적 규칙을 고르는 oracle나 추가 이미지 추론은 사용하지 않았다.

| 후보 | development PQ | overlap 영상 수 |
|---|---:|---:|
| 기존 union+CC 0.25 | **0.43175** | 0 |
| raw preserve 0.15 | 0.43960 | 71 |
| preserve 0.15 → 입력순서 겹침 해소 | 0.39504 | 0 |
| preserve 0.15 → 면적순서 겹침 해소 | 0.40581 | 0 |
| raw preserve 0.25 | 0.43908 | 48 |
| preserve 0.25 → 입력순서 겹침 해소 | 0.41642 | 0 |
| preserve 0.25 → 면적순서 겹침 해소 | 0.42380 | 0 |
| 기존 설정 → 1px cross 팽창 | 0.39823 | 0 |
| 기존 설정 → 1px cross 침식 | 0.43098 | 0 |

mask NMS 0.7/0.9에서도 overlap은 남았고 raw preserve를 넘는 후보는 없었다. 전체 13개 실험 중 **겹침 없는 후보의 최고값은 기존 설정 그대로**였다. 새롭고 더 나은 nonoverlap 후보를 찾았다고 보고하지 않는다.

여기서 겹침 유무는 분석상의 비교 조건이며, 대회가 disjoint masks를 요구한다고 확인한 것은 아니다. 겹침 마스크 제출이 허용된다면 raw preserve는 제공 scorer 기준으로 높은 development 점수를 낸 유효한 비교 후보로 남는다. 다만 추가 매칭쌍, FP 증가, bootstrap 변동과 독립 검증 미확인을 함께 고려한다. 원본의 literal 점수를 임의로 낮춰 보고하거나 규정을 추정해 후보를 금지하지 않는다.

결정: **기준선은 union+CC/0.25로 고정**, raw preserve/0.15와 0.25는 별도 후보로 기록한다. 다음 학습 실험의 주요 목표는 작은 GT 검출률을 현재 39.5%에서 개선하면서 FP를 억제하는 것이다. 아직 신규 모델의 성능이나 예상 리더보드 점수는 제시하지 않는다.

## 저장된 근거

- `docs/evidence/validation_score_verification.json`: 10개 점수 재계산과 bootstrap.
- `docs/evidence/validation_diagnostics.json`: GT별 best IoU·면적/category/site·매칭 진단.
- `docs/evidence/validation_summary.png`: confidence 곡선·미검출·크기·추가 매칭쌍 시각화.
- `docs/evidence/postprocess_experiments.json`: 13개 고정 후보와 overlap 진단.
- 원본 결과와 파생 후보 CSV는 `/workspace/artifacts`에 보관하며 학습 정답·이미지·체크포인트는 수정하지 않았다.
