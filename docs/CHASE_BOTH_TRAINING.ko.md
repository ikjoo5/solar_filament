# CHASE만으로 YOLO와 best84 추가 학습

사용자 확정 조건: T4에서 기존 public LB0.39의 두 모델을 각각 CHASE만으로 이어 학습한다. GONG 재학습과 마지막 GONG 보정 단계는 사용하지 않는다. SWEFil은 제외한다. CHASE에는 원반 정렬만 적용하고 chirality는 학습하지 않는다. 겹침은 명시된 자료가 있을 때만 제외하며, 모양을 보고 자동 삭제하지 않는다.

## 학습하는 구성

- YOLO26s-seg: 기존 checkpoint에서 추가 학습한다. CHASE 이진 마스크의 연결 영역을 임시 객체 polygon으로 변환한다. 이 변환은 실제 객체 번호를 복원한 정답이 아니며, 구멍 채움·끊긴 영역·모델 내부 polygon resampling의 표현 차이를 보고한다. 작거나 가지가 있다는 이유로 제외하지 않는다.
- best84: 기존 ConvNeXt-Tiny encoder와 U-Net decoder를 포함한 모델을 추가 학습한다. CHASE는 foreground만 감독한다. CHASE에 없는 spine 라벨은 손실에서 제외하며 음성 정답으로 만들지 않는다. 공통 encoder/decoder가 갱신되므로 spine 출력까지 완전히 고정되는 것은 아니다.
- 0.39 파이프라인의 경계 refiner는 별도 학습 모델이 아니라 두 모델의 마스크와 뷰 지지를 조합하는 규칙이다. 해당 규칙과 임계값은 유지한다. 새 best84 출력이 경계 보정에도 쓰이므로 학습 효과는 후처리 결과까지 전달된다.
- 기존 YOLO와 기존 best84 파일은 보존하고 별도 실험 폴더에 새 가중치를 저장한다.

## 데이터와 입력

- CHASE AHAS2024~2025년695장: 원반 중심·크기 정렬만 적용. CLAHE, 추가 limb 보정, gamma, sharpening, blur, 초해상도와 photometric augmentation은 사용하지 않는다.
- PNG 규격 수정: 확인된2025 원본은2304×2313인 반면 공개 마스크는2048×2048이다. 저자의 [고정 버전 생성 코드](https://github.com/irisaltHu/MORDEN/blob/e8186480a37395272ba414a0143b2b1232294ed1/segmentation/inference.py#L102)에 따라 원본 전체를 마스크 격자로 bilinear resize한 다음, 이미지·마스크에 같은 원반 정렬을 적용한다. 마스크는 이 사전 변환에서 수정하지 않으며, 원반 정렬 시 nearest-neighbor를 유지한다. 저장된 원본 PNG와 CRC/SHA는 그대로 보존한다.
- GONG train616장: 이번 추가 학습에서 사용하지 않는다. 기존 train/validation 분할 파일은 데이터 식별과 누수 확인을 위해 보관한다.
- GONG validation91장,101 annotator 그룹,708GT는 평가에만 사용한다. 기존 파이프라인의 limb 보정도 평가 시 그대로 유지한다.
- 학습 후보 관측은 CHASE695장이다. 샘플링·crop을 늘린 수를 고유 사진 수로 세지 않는다.
- 공개 AHAS 전체815장 중 MHAS와 중복되는120장은 이번 대상이 아니다. 명시된 overlap 제외 정보가 없으면695장 모두 대상이다. 파일 손상이나 정렬 실패는 해당 관측 ID와 함께 오류로 표시하며 조용히 제외하지 않는다.
- CHASE 이진 마스크에서 겹침을 자동 판별하지 않는다. 추후 명시적 제외 정보가 있을 경우 native loss-ignore 또는 YOLO 전체 관측 제외로 처리하며, 제거한 필라멘트 위치를 배경 정답으로 바꾸지 않는다.
- 모델 입력의 고정 수치 정규화와 흑백 채널 복제는 유지한다. 기하 정렬 영상 보간과 마스크 nearest-neighbor 변환은 동일 좌표를 사용한다.

## 기본 추가 학습 예산

- 모델당60epoch, 두 모델을 순서대로 실행하므로 총120개 모델-epoch다. 기존 배포 가중치에서 추가 학습한다.
- 한 epoch는 CHASE695장을 각각 정확히 한 번 사용하는 단위다. `epoch_sampling='full_pass'`로 설정하고400개 제한을 제거한다.
- 처음부터 마지막60epoch까지 CHASE만 사용한다. 모델당 총41700개 입력이며 GONG 학습 샘플은0개다. 현재 선택된695장 각각을60번 사용한다.
- 매 epoch마다 입력 순서를 섞는다. Native의 사진 노출은 그 사진의 모든 픽셀·필라멘트를 crop으로 보았다는 뜻은 아니다.
- T4 기본 batch1. U-Net은 gradient accumulation4, YOLO는 완료epoch 재개가 검증된 accumulation1이다. 학습률2e-5의 추가 학습이다.
- 이번 실험은 사용자 선택에 따라 GONG replay 없이 CHASE 추가 학습의 효과를 확인한다. GONG에서의 성능 변화는 고정 validation으로 측정한다.
- 이 비교는 기존 가중치 대비 CHASE 추가 학습 결과다. 동일 학습량의 GONG-only continuation 대조를 실행하지 않으므로 다른 추가 학습 방식보다 우월하다고 단정하지 않는다.

## 에폭당 전체 이미지 사용

- 695장의 순서를 에폭마다 재현 가능한 난수로 섞고 전체를 한 번씩 사용한다. 에폭 안에서 일부 사진이 중복되거나 빠지지 않는다.
- 한 에폭을 마친 뒤 다음 에폭에서 순서를 다시 섞는다. 완료 에폭부터 재개할 때 같은 입력 순서를 복원한다.
- 각 사진의 누적 사용 횟수는 완료한 에폭 수와 같다. 기존400개 단위의 체크포인트를 이 설정으로 자동 승계하지 않는다.
- YOLO는 선택한 사진의2048×2048 전체 입력을 사용한다. Native는 선택한 사진에서1024×1024 crop을 사용하고, 에폭이 바뀌면 crop 위치가 달라질 수 있다.
- Native crop은 유효한 양성 픽셀이 있을 때75% 확률로 그중 한 픽셀을 포함하도록 선택하며, 나머지는 무작위 위치를 선택한다. 무작위 crop에도 필라멘트가 들어갈 수 있어75% 양성/25% 배경이라는 의미는 아니다.

## 실제 하이퍼파라미터

| 항목 | YOLO26s-seg | best84 ConvNeXt-Tiny U-Net |
|---|---|---|
| 초기 가중치 | 기존 배포 YOLO | 기존 epoch84 |
| 추가 학습 |60epoch ×695 full-frame|60epoch ×695 crop|
| 입력 크기 |2048×2048|1024×1024|
| Batch / 누적 |1 /1|1 /4|
| Optimizer |AdamW|AdamW|
| 초기 학습률 |0.00002|0.00002|
| 학습률 스케줄 |60epoch cosine, 최소 비율0.1|60epoch cosine, 최소 비율0.1|
| Weight decay |0.01, 라이브러리의 parameter group 정책|0.01|
| AMP |사용|사용|
| Warmup |없음|없음|
| Random seed / workers |42 /0|42 /0|
| 증강 |Mosaic·MixUp·회전·flip·밝기 변환 등 모두 끔|CHASE 증강 없이 crop 표집|
| 조기 종료 |60epoch 이내 조기 종료 없음|60epoch 이내 조기 종료 없음|
| 손실 |라이브러리의 YOLO26 segmentation 손실|foreground BCE + Dice; spine 감독 없음|

학습률·입력 크기·표집량·누적·전처리·증강 제한은 이번 추가 학습을 위해 지정한 값이다. 모든 하이퍼파라미터를 대회에서 탐색해 최적화한 값은 아니다. YOLO의 지정하지 않은 세부 손실 설정과 optimizer 세부값은 고정한 Ultralytics8.4.182의 기본값을 사용한다. Native AdamW의 betas와 eps도 PyTorch 기본값을 사용한다. 학습률은 기존 가중치를 작은 폭으로 갱신하려고2e-5로 시작하지만, CHASE-only에서 최적이라는 검증 결과는 아직 없다.

Native는695 crop을4개씩 누적하므로 한 에폭에174번 optimizer 업데이트를 시도한다. 마지막3개도 평균 손실로 계산해 버리지 않는다. YOLO는 한 에폭에695번 업데이트를 시도한다. AMP overflow가 발생하면 해당 업데이트가 건너뛰어질 수 있으며 실제 횟수를 기록한다.

## 평가

1. 기존 YOLO + 기존 best84.
2. 새 YOLO + 기존 best84.
3. 기존 YOLO + 새 best84.
4. 새 YOLO + 새 best84.

모든 조합은 같은 frozen finalizer와 공식 IoU>0.5 all-pairs PQ 계산을 사용한다. YOLO의 mAP나 U-Net의 Dice만으로 전체 파이프라인 개선을 판단하지 않는다. 같은 native 예측과 YOLO뷰 캐시를 공유해 불필요한 재추론을 줄인다.

- YOLO는 고정 완료epoch 가중치를 사용하며 mAP로 골랐다고 표시하지 않는다.
- Native는5epoch 간격으로 기존 YOLO와 조합한 full-pipeline PQ를 기록하고 best도 보존한다. 최종4조합은 두 모델의 같은 완료epoch 가중치를 사용해 비교한다.
- 과거 검증PQ0.472593과 publicLB0.39는 서로 다른 값이다. 같은 환경에서 생성하는 기준선과 새 조합의 검증 수치를 비교한다.
- 공개 test참조216+102는 이번 학습 라벨이 아니다. 이번 코드는 validation 추가 학습 평가용이며, 새 test포함률이나 새LB가 자동 측정되는 것은 아니다.

## 실행·재개와 확인 범위

- Colab에서 실행할 한 코드 셀 패키지를 제공한다. 코드와 설정, 가중치, 기록은 Drive에 보존하고 다운로드 ZIP·전처리 캐시는 로컬/content에 둔다.
- PNG 오류 수정본의 새 실험 폴더는 `fixed_split_experiments/chase_only_both_alignment_t4_60ep_fullpass_pngfix_v1`이다. 오류가 난 이전 실행은 자료 준비 중이었으므로 학습이 시작되지 않았다. 원본 CHASE 저장 폴더와 `/content/chase_only_both_alignment_60ep_fullpass_cache`는 유지해 받은 ZIP과 부분 추출된 파일을 재사용한다. 수정본 안에서는 같은 셀로 이어간다.
- 완료epoch의 optimizer, scheduler, AMP, RNG와 YOLO EMA를 복원한다. 중단된 epoch의 일부 batch는 다시 실행될 수 있다. 검증·추론의 완료 이미지와 단계도 재사용한다.
- 원본 가중치·정답·분할·이미지·코드의 식별자를 확인해 다른 실험 상태와 잘못 섞지 않는다.
- CPU 테스트와 실제 표본 감사는 수행했지만, 이 클라우드에서 전체695장 다운로드나 실제 T4 학습을 수행한 것은 아니다. 실제 추가 학습 PQ와 LB는 아직 없다.
