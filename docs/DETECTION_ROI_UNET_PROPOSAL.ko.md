# Detection → box-conditioned ROI U-Net 검토

## 판단

대회 train만으로 검증할 가치가 높은 다음 구조다. 기존 YOLO와 full-image U-Net의 마스크를 병렬 결합하는 방법과 달리, 먼저 위치 후보를 만들고 객체별 영상으로 segmentation을 수행한다. 작은 객체의 상대적 해상도와 객체별 조건부 학습을 개선할 가능성이 있다. Bounding box만 학습하면 정확도가 반드시 높아진다고 가정하지 않는다.

## 첨부 논문에서 확인한 내용

Reche & Cid, *New dataset and framework for filament detection, classification, and segmentation using deep learning*, Acta Astronautica 249 (2026), 381–398. DOI: 10.1016/j.actaastro.2026.07.010.

- PDF 6쪽 §2.4: DETR 검출과 별도 U-Net 분할을 사용한다. Detector 입력은 800×800이다.
- 원 bbox a×b 주변을 3a×3b로 확장한다. 원 bbox 위치를 1로 표시한 추가 binary 채널을 입력해 crop 속 처리할 객체를 지정한다.
- 긴 변에 맞춰 정사각 padding 후 256×256으로 변환한다. U-Net 마스크를 원좌표로 복원한다.
- PDF 7쪽 Table 1: SWEFil bbox AP50 0.80, AR50 0.88. 11쪽 Table 3: MAGFiLO만 학습한 detector AP50 0.65, AR50 0.81.
- PDF 9쪽 Table 2: MAGFiLO 개별 crop segmentation IoU 0.70, Dice 0.81. 정답 annotation bbox 기반 crop의 분할 평가이며 전체 영상 검출 누락·FP를 포함한 PQ가 아니다.
- 논문은 SWEFil와 sunspot 라벨도 사용한다. 우리 적용은 기존 허용된 대회 train에 제한한다.

## 지금 수행한 후보 기하 진단

원래 고정 validation 91장·101주석그룹·GT 708개에 대한 기존 YOLO 4뷰 후보를 사용했다. **저장된 후보에는 confidence와 mask RLE만 있고 실제 detector bbox는 없다.** 아래 bbox는 마스크의 외접 사각형이므로 bbox-only detector 성능이 아니다. 후보 마스크 면적 ≥300을 유지했다.

| 후보 confidence | Mask IoU>0.5 후보 존재 | 외접 bbox IoU>0.5 후보 존재 |
|---|---:|---:|
| ≥0.05 | 635/708 (89.69%) | 643/708 (90.82%) |
| ≥0.30 | 498/708 (70.34%) | 488/708 (68.93%) |

낮은 confidence 범위에서 bbox는 맞지만 마스크가 틀린 GT는 16개, 반대는 8개였다. ROI 분할로 개선할 수 있는 사례가 있다는 단서지만 모든 미검출이 해결된다는 증거는 아니다. 낮은 confidence 후보는 사진당 평균 76.4개다.

모든 후보 bbox를 3배 확장하면 GT 픽셀 95% 이상을 포함하는 crop이 하나라도 있는 GT는 683/708이었다. 큰 오검출 박스나 이웃 객체 박스도 정답을 덮을 수 있으므로, 이 값은 실제 검출 recall이나 달성 가능한 최종 recall로 해석하지 않는다.

## 제안 실험 순서

1. **실제 detector bbox부터 수집한다.** 기존 YOLO-seg의 box 출력을 우선 이용해 초기 비용을 줄인다. 현재 마스크 외접 bbox와 구분하고, 후보를 마스크 면적 조건으로 버리지 않는다. 이후 bbox-only detector를 별도 학습해 비교할 수 있다. 논문의 DETR를 그대로 선택해야 하는 것은 아니다.
2. bbox confidence를 낮추어 후보를 확보하되 사진당 후보 수·정답 mask의 crop 포함률·작은 객체 coverage를 함께 측정한다. 검출 누락이 앞단에서 생기면 ROI 분할만으로 복구할 수 없다.
3. **ROI용 U-Net을 별도 학습한다.** 입력은 확장한 grayscale crop과 bbox 위치 채널, 정답은 지정된 한 객체의 마스크다. 현재 전체 영상용 1채널·3출력 U-Net에 crop만 넣는 것과 구분한다.
4. 정답 bbox crop으로 기본 분할 가능성을 먼저 확인하고, train에서 bbox 위치·크기 흔들기로 실제 검출 오차를 반영한다. train-only 검출 후보의 어려운 FP를 빈 마스크로 학습한다. 다른 주석자에게 실제 foreground로 표시된 영역을 단순 음성으로 만들지 않도록 검증한다.
5. 문맥 배율과 입력 해상도는 작은 수의 사전 지정 조건으로 비교한다. 논문의 3배·256px는 분류 문맥까지 위한 선택이므로, 길고 가는 filament에 무조건 적용하지 않는다. 종횡비는 padding으로 보존한다.
6. 실제 검출 bbox에서 원좌표 복원과 중복·겹침 처리를 수행하고 **전체 영상 PQ·FP·FN**을 계산한다. 정답 bbox로 얻은 segmentation IoU와 구분한다.
7. 원래 train616/validation91을 유지하고 검증 정답으로 crop을 제공한 결과는 oracle 진단으로만 표시한다. 향후 5-fold에서는 detector와 segmenter 둘 다 fold train 밖을 보지 않아야 한다.

이 검토에서는 논문 분석과 저장 후보의 CPU 기하 진단만 실행했다. 새 bbox detector나 ROI U-Net을 학습하지 않았고 실제 제출 구성도 바꾸지 않았다.

증거: `/workspace/artifacts/solarfilament_paper_review/paper_method_audit.json`, `box_proposal_geometry_audit.json` 및 재현 Python 코드.
