# Detection → ROI U-Net / DETR 비교 실험

## 목적과 비교 범위

전체 태양 영상에서 바로 모든 마스크를 만드는 모델과 별도로, **검출기로 위치 후보를 만들고 각 후보를 U-Net으로 분할**하는 구조를 시험한다. 원래 고정 train 616장·주석 그룹 1,053개 / validation 91장·101그룹을 유지한다. 같은 사진의 모든 주석자는 같은 분할에 둔다.

영역 검출과 마스크 분할의 실패를 구분하기 위해 다음 세 입력을 같은 ROI U-Net으로 평가한다.

| 입력 bbox | 용도 |
|---|---|
| 기존 YOLO의 실제 box 출력 | 기존 검출기를 활용한 실제 전체 파이프라인 |
| 새 DETR의 실제 box 출력 | 검출기 교체 효과 비교 |
| 각 주석자 그룹의 정답 bbox | 조건부 segmentation 가능성을 보는 oracle 진단 |

정답 bbox 결과는 실제 detector의 누락·FP를 포함하지 않는 별도 진단이다. 실제 검출 성능이나 제출 PQ로 표현하지 않는다. 실제 검출 예측은 사진당 한 번 만들고 각 주석자 그룹에 동일하게 평가한다. Oracle은 주석자 그룹별 정답 bbox를 각각 사용하며 여러 주석자의 정답을 합치지 않는다.

## 모델

### DETR 검출기

기본 검출 실험은 **기존 COCO 사전학습 `facebook/detr-resnet-50`**에서 시작한다. Transformer·bbox·backbone의 사전학습을 활용하고 분류 출력을 filament 한 클래스로 맞춘다. 논문의 DETR 계열 구조를 적용하지만 논문과 동일한 학습 데이터·초기화·정확한 재현 실험은 아니다.

- 실제 detector 입력 bbox를 원본 2048 좌표로 복원한다.
- mask 외접 사각형을 detector bbox로 대신하지 않는다.
- 후보 수집 단계에 segmentation mask 면적 300 조건을 적용하지 않는다.
- 대회 train의 bbox만으로 추가 학습한다. 외부 SWEFil·공개 MAGFiLO 자료를 추가 학습에 사용하지 않는다.

### Box-conditioned ROI U-Net

기본 모델은 ResNet34 encoder의 2채널 U-Net이다.

1. 후보 bbox 중심에, 긴 변에 문맥 배율을 곱한 변 길이의 정사각 문맥 영역을 잡는다.
2. 해당 실제 영상 영역을 잘라 384px로 변환한다. 영상 바깥 부분은 0으로 채운다.
3. 입력 채널은 grayscale 영상과 **원래 bbox 위치를 표시한 binary prompt**다.
4. 출력은 지정한 한 객체의 foreground mask다. 같은 crop의 다른 필라멘트를 전부 정답으로 합치지 않는다.
5. 결과를 원본 좌표로 되돌리고 중복 마스크·픽셀 겹침을 정리한다.

기본 문맥 배율 1.5·입력 384px는 분할 실험용 선택이다. 논문은 가로·세로를 각각 3배 확장한 직사각 문맥 영상을 자른 뒤 짧은 변을 padding하지만, 이번 구현은 처음부터 정사각 실제 문맥 영상을 자른다. 따라서 논문의 3배·256px·직사각 letterbox를 그대로 재현한 것은 아니다. 정사각 영역을 등비 변환하므로 영상 종횡비를 왜곡하지 않는다. 모델 내부에서 영상 채널만 정규화하고 binary prompt에는 영상 정규화를 적용하지 않는다.

Train ROI는 bbox 흔들기를 적용한 실제 양성 객체와 train 검출 후보의 어려운 FP·안전한 배경 음성으로 만든다. 다른 주석자가 표시한 실제 전경을 빈 마스크 정답으로 잘못 사용하지 않도록 모든 train 주석자의 전경과 음성 후보를 검사한다. Validation proposal은 학습용 음성 후보나 dataset에 들어가지 않는다.

## 실행 예산과 단계

기본 `all`은 **DETR 30에폭 + ROI U-Net 30에폭**, 서로 다른 두 모델의 총 60 학습 에폭이다. 기존 YOLO는 추가 학습하지 않는다. 다섯 fold나 여러 backbone을 자동으로 연속 학습하지 않는다.

- DETR epoch: 원래 train 주석 그룹 1,053개 전체를 대상으로 하는 epoch다. 기본 batch 1·누적 4이므로 에폭당 optimizer 시도는 최대 264회, 30에폭 최대 7,920회다.
- ROI epoch: 256 mini-batch × 4 = 1,024개 ROI 샘플링 예산이다. 30에폭 총 30,720개 ROI, 누적 1일 때 optimizer 시도는 최대 7,680회다. 원사진 전체를 한 번 통과한 단위와 구분한다.
- `yolo_roi`: DETR 학습 없이 YOLO bbox와 ROI U-Net으로 먼저 확인한다.
- `compare_detr`: 이미 학습한 ROI U-Net을 고정하고 DETR을 추가 학습·평가한다. 기존 ROI 학습 corpus를 새 DETR 후보로 바꾸어 재개하지 않는다.
- `evaluate`: 저장된 가중치와 후보를 이용해 평가한다.

각 모델의 최종 스케줄과 이번 실행 목표 에폭을 분리한다. 중단 시 마지막 완료 에폭의 모델·optimizer·scheduler·AMP·RNG를 복원한다. 중간에 중단된 에폭은 다시 실행한다. 단계별 checkpoint·설정·소스·데이터·입력 픽셀의 정체성을 확인한다.

## 평가

- 검출 진단: bbox IoU >0.5 정답 후보 존재율과 crop의 GT mask 포함률, 작은 객체·후보 수를 기록한다.
- 전체 결과: 실제 검출 bbox → ROI segmentation → 원좌표 마스크에서 공식 strict mask IoU >0.5의 PQ·FP·FN을 계산한다.
- ROI용 정답 bbox 실험은 oracle로 분리한다. 큰 박스가 정답을 포함한다는 이유로 올바른 객체를 선택했다고 간주하지 않는다.
- 같은 ROI 모델을 두 검출기에 사용해 detector 변경 효과를 비교한다. 모델 confidence 척도와 후보 분포가 다르므로 숫자가 완전히 같은 의미라고 가정하지 않는다.
- 기존 Solar 학습 YOLO의 2048 입력과 COCO 초기화 DETR의 800 입력·30에폭 학습은 해상도·이전 학습량이 다르다. 이는 각 검출 구성의 실용적인 전체 성능 비교이며, Transformer 대 CNN의 순수한 인과 비교나 모든 DETR의 성능 상한을 측정한 실험은 아니다.
- 모든 모델·규칙은 반복 사용한 고정 개발 validation에서 평가한다. 새 LB나 독립 5-fold OOF 결과로 표현하지 않는다.

## 논문과의 관계

첨부 논문은 SWEFil에서 DETR bbox AP50 0.80·AR50 0.88, MAGFiLO 개별 crop에서 segmentation IoU 0.70·Dice 0.81을 보고한다. 이 값은 우리 전체 영상 PQ가 아니다. MAGFiLO만 학습한 detector의 AR50은 0.81이었다. 검출기가 후보를 만들지 못한 객체는 순수 cascade의 다음 단계에서 복구할 수 없다.

이번 코드는 구조의 유용성을 대회 train에서 새로 검증하기 위한 것이다. 논문 점수의 직접 재현이나 PQ 0.7 달성을 사전에 주장하지 않는다.

## 검증 상태

독립 소스 검토와 **31개 테스트**를 통과했다. 실제 작은 HF DETR의 loss/backward·학습·재개, 실제 ResNet18 ROI gradient, 단일 객체 crop 정답과 좌표 복원, 다른 주석자의 전경을 음성으로 만들지 않는 검사, 실제 bbox 출력·마스크 비중첩, corpus 정체성과 단계별 재개·실패 로그를 확인했다.

검토 중 다음 사항을 보완했다.

- 검출 진단의 crop 포함률도 모델과 같은 정사각 문맥 geometry를 사용한다.
- `compare_detr`는 ROI checkpoint와 학습 설정을 먼저 검사하고, 잘못된 상태라면 DETR 30에폭을 시작하기 전에 실패한다.
- 기존 YOLO-only ROI corpus를 새 DETR 후보로 조용히 바꾸어 재개하지 않는다.
- 실제 YOLO bbox는 mask가 없거나 작아도 수집하며, DETR의 후보 confidence 범위가 평가 범위를 포함하는지 확인한다.

최종 소스 해시·검증 범위는 `docs/evidence/detection_roi_independent_review.json`에 기록했다. 추가로 root가 실제 원래 YOLO checkpoint(64px 추론), 작은 HF DETR, 실제 ResNet18 ROI U-Net을 연결한 CPU 통합 검사를 완료했다. DETR 2에폭·ROI 1→2에폭 재개, 두 검출기의 실제 bbox, 주석자별 oracle, 평가만 재실행할 때 checkpoint 보존, 결과 ZIP 생성을 확인했다. 해당 검사의 소스 해시는 독립 검토와 일치한다. 검사에는 가상 2048 fixture를 사용했으며 실제 대회 성능을 측정한 것은 아니다.

**실제 대회 T4 학습 시간·최대 메모리·새 모델 PQ는 아직 측정하지 않았다.**
