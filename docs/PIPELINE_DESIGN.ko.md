# Solar Filament Segmentation 2026: PQ 중심 전체 파이프라인 설계

작성일: 2026-10-27. 대상: https://www.kaggle.com/competitions/filament-segmentation-2026

업로드 자료 검토 후 수정: 사용자가 제공한 self-evaluation notebook으로 **전역 누적 PQ, strict IoU > 0.5, 모든 qualifying pair 집계**를 확인했다. 베이스라인 자체 평균 PQ와 다르다. 자세한 근거·우선 수정 실험은 [첨부 자료 분석](UPLOADED_FILES_AUDIT.ko.md)에 정리한다. 아래 일반적인 PQ 설명보다 제공 평가 코드의 실제 동작을 우선한다.

실제 validation 결과 반영: 사용자 실행 결과 91장/101 annotation group/708 GT를 재계산했다. 기존 union+CC 0.25는 PQ **0.43175**다. raw preserve 최고값 **0.43960**은 추가 매칭쌍 77개와 겹침을 동반하므로 즉시 최종 설정으로 채택하지 않는다. 면적 하위 25% GT 검출률 **39.5%**, 미검출 293개 중 출력과 겹치지 않는 GT **185개**로 작은 객체 검출을 우선 개선한다. 세부 결과는 [validation 분석](VALIDATION_RESULTS_20261027.ko.md)을 따른다.

추가 고정 후처리 13개 비교에서 겹침 없는 최고값은 기존 union+CC/0.25 그대로였다. raw preserve는 overlap 허용 규정을 추정해 배제하지 않고 별도 비교 후보로 남긴다. 이제 새 mask decoder·spine 보조학습과 작은 객체 샘플링의 실제 이득을 검증하는 단계다.

후속 사용자 지시에 따라 전처리·후처리 실험을 우선해, 동일 모델의 scale/flip 합의와 closing/면적 필터 순서로 검증 PQ **0.457152**까지 개선했다. 현재 FN241/FP199이며 다음 8개 전처리/68개 설정을 준비했다. 최신 우선순위·실측 결과는 [2차 전후처리 실험](PREPROCESS_POSTPROCESS_ROUND2.ko.md)을 따른다.

## 1. 결론과 현재 확인 범위

추천 출발점은 **원본 해상도 타일 기반 ConvNeXt U-Net/U-Net++ → 전체 영상에서 인스턴스 복원 → OOF PQ로 후처리 선택**이다. 이후 **전역 문맥 결합, 독립 백본, 객체 대응 앙상블**을 추가한다. 별도의 **Mask2Former 전역 인스턴스 모델 + 고해상도 보정**은 객체 정의와 오류 분석이 필요성을 뒷받침할 때 도입한다.

큰 모델 하나를 처음부터 최종안으로 고정하지 않는다. 채택 기준은 동일한 검증 분할과 공식 채점기로 측정한 PQ이며, 점수 상승은 실제 실험으로 확인해야 한다. annotation/CSV 전수 분석과 평가 도구 검증은 완료했지만 신규 모델 학습·제출은 수행하지 않았고, 최고점이나 예상 PQ를 제시할 근거는 없다.

확인된 사항:

- 사용자가 위 Kaggle 대회와 PQ 평가를 확인했다.
- 사용 가능한 학습 자원은 T4 또는 A100이다. GPU 개수와 A100 메모리 용량은 아직 미정이다.
- 공개 주최 측 안내는 GONG H-alpha 관측 영상의 필라멘트 분할 과제라고 설명한다.
- 최초 저장소에는 README.md만 있었으며, 현재 설계 문서와 데이터/평가 검증 도구를 추가했다. Windows의 `C:\Users\ikjoo\OneDrive\바탕 화면\UWM_IkjooOh\26Fall\공모전` 경로는 이 클라우드에서 직접 접근할 수 없다.
- 업로드 ZIP을 열어 2048² 영상 레코드 1,154개, 실제 파일 707개, polygon 인스턴스 8,199개와 `spine` 중심선 정보를 확인했다. 같은 실제 파일의 여러 annotator 레코드를 보존한다.
- 사용자가 제공 CSV의 리더보드 PQ를 0.37로 보고했다. 이는 직접 재현한 로컬 점수가 아니며 검증 점수와 분리 기록한다.

확인이 필요한 사항:

- 대회 원본 JPEG 전체의 픽셀·전처리 상태와 파일 coverage. 정답 metadata와 일부 공개 참조 JPEG의 크기는 2048²로 확인했다.
- annotator 간 불일치와 annotation 완전성, 서로 떨어진 조각의 동일 객체 처리, overlap/ignore 영역 정책.
- 제공 self-evaluation notebook과 실제 Kaggle backend의 일치 여부, 빈 이미지의 공식 제출 정책, 최신 규정. 제공 노트북의 계산·ID·RLE 계약은 아래와 같이 확인했다.
- 외부 데이터·사전학습 가중치·인터넷·추론 시간 제한, 최종 마감일.

제공 평가 노트북은 2048×2048 compressed COCO RLE를 디코딩한다. 제공 이미지 3장은 원본 2048×2048이며 채팅 표시용으로 1600×1600으로 축소되었다. 이름을 알려준 `20110109104734Ch.jpeg`는 GT 6개와 연결했고, 다른 두 장의 파일명은 미확인이다.

## 2. 먼저 고정할 평가 계약

일반적인 PQ의 식은 다음과 같다. 제공 평가 노트북은 아래 식에서 모든 annotator–이미지 entry의 TP IoU·FP·FN을 **전역 누적한 뒤 한 번** 나눈다. 이미지별 PQ의 평균이 아니다.

\[
PQ = \frac{\sum_{(p,g)\in TP}\operatorname{IoU}(p,g)}{|TP|+0.5|FP|+0.5|FN|}
    = SQ\times RQ
\]

- SQ: 매칭된 객체의 평균 IoU. TP=0일 때의 처리는 evaluator에서 확인한다.
- RQ: 객체 단위 검출·인스턴스 분리 품질.
- 제공 코드는 strict `IoU > 0.5`인 **모든 pair**를 TP로 센다. Hungarian/greedy 일대일 matching을 강제하지 않는다. 이를 임의 교체하지 않는다.
- FP는 qualifying GT가 없는 prediction column 수, FN은 qualifying prediction이 없는 GT row 수다. 노트북은 중복·겹침을 검사하지 않으며, overlap 허용 정책은 대회 규정에서 별도 확인한다.
- 예측 한 세트를 같은 이미지의 각 annotator GT에 반복 대조하고 전역 누적한다. annotator 수·객체 수가 점수 가중에 영향을 준다. 정답의 합집합이나 최고 annotator 점수로 대체하지 않는다.
- entry는 GT filament row에서만 발견한다. GT에 전혀 없는 이미지의 예측과 빈 GT 이미지를 정상적으로 평가하는 coverage를 이 노트북만으로 보장할 수 없다. 이미지 목록 검증을 별도로 둔다.

픽셀 Dice만으로 모델을 고르지 않는다. 정답 하나를 같은 크기 두 조각으로 분리하면 각 조각의 IoU가 0.5가 될 수 있다. strict `>0.5` 매칭에서는 픽셀 합집합이 정답과 같아도 모두 매칭에 실패할 수 있다. 반대로 별개 정답 두 개를 하나로 합치는 것도 PQ를 크게 떨어뜨릴 수 있다.

평가 검증 항목:

1. 완전 일치 마스크, 추가 FP, 누락 FN, 분리/병합 예제에서 공식 결과 확인.
2. 빈 정답/빈 예측, 작은 객체, ignore 영역, 겹친 마스크 처리 확인.
3. RLE 또는 제출 직렬화의 encode→decode가 원본 픽셀 위치·크기·ID를 보존하는지 확인.
4. 최종 제출 파일을 다시 읽어서 계산한 PQ가 메모리상의 예측 PQ와 같은지 확인.
5. 공식 코드·버전·파일 해시와 평가 옵션을 실험 결과에 남긴다.

## 3. 데이터 감사와 검증 분할

### 3.1 데이터 manifest

학습 전에 이미지 단위 manifest를 만든다. 필드는 다음을 권장한다.

`image_id, path, sha256, width, height, dtype, observation_time, observatory, sequence_id, annotator_ids, n_instances, foreground_fraction, group_id, fold`

- 이미지와 annotation ID 연결, 좌표계·RLE 순서·마스크 shape를 검증한다.
- 원본 이미지는 보존하고, 파생 캐시는 이미지 해시·전처리 버전으로 구분한다.
- 인스턴스 면적·폭·길이·대비·태양 중심으로부터의 거리 분포를 확인한다.
- 무작위 사례와 작은/희미한/접촉/분리된 사례를 포함하여 30–50장의 정답 overlay를 직접 살핀다.
- 검은 태양 외부, 흑점, 밝은 plage, 광학 아티팩트가 어떤 오탐을 만드는지 확인한다.
- 같은 영상에 여러 annotator가 있으면 원본 image_id와 각 정답 집합을 분리 보관한다. annotator별 샘플을 독립 이미지처럼 분할하지 않는다.
- annotator 간 PQ·객체 수·분리/병합 차이도 측정해 모델 오류와 정답 불일치를 구별한다. annotator 간 일치도를 모델 성능의 절대 상한으로 해석하지 않는다.

### 3.2 누수 방지

최소 분할 단위는 **실제 원본 영상**이다. 타일, 증강본, annotator 변형은 모두 같은 fold에 둔다.

- 정확 중복은 해시로 묶는다. 근접 프레임은 관측 시각·관측 시퀀스와 이미지 유사도를 함께 확인한다.
- 같은 필라멘트가 연속 프레임에 반복되면 날짜/관측 블록 단위 group 분할을 사용한다. 필요하면 인접 시점 사이 간격을 두고, 간격은 실제 유사도와 관측 주기로 결정한다.
- 관측소가 다르더라도 같은 태양·시점의 영상이면 독립 표본이라고 가정하지 않는다.
- 날짜 정보가 없으면 파일명 규칙을 확인하고, perceptual similarity와 수동 검토로 그룹을 만든다. 자동 유사도만으로 서로 다른 영상을 무조건 묶지는 않는다.
- 이후 발견한 중복을 반영하면 split 버전을 올리고 관련 실험을 재평가한다.

실험 순서:

1. 그룹 수가 충분하면 개발용 고정 fold 1개에서 baseline과 제한된 후보를 비교한다.
2. 유망 후보 2–3개를 같은 grouped 5-fold로 평가한다. 데이터 그룹이 적으면 fold 수를 줄인다.
3. 충분한 데이터가 있을 때 최종 확인용 독립 group holdout을 처음부터 남긴다. 적은 데이터에서는 nested/grouped 검증과 제한된 탐색을 우선한다.
4. 후처리·앙상블 가중치는 개발 OOF에서 결정하고 최종 holdout을 보고 다시 조정하지 않는다.
5. public LB는 제출 파이프라인과 분포 차이를 점검하는 데 사용한다. LB만 보고 반복적으로 파라미터를 고르지 않는다.

평균 PQ와 함께 관측 그룹 단위 bootstrap 구간, fold별 PQ, SQ/RQ, 크기·대비·관측소·가장자리별 결과를 남긴다. 비선형 집계에서는 공식 단위를 재표집해 scorer를 다시 계산한다.

## 4. 전처리와 샘플링

### 입력 표현

먼저 grayscale 원본을 학습 fold 통계로 안정적으로 정규화한 단순 입력을 만든다. pretrained RGB encoder를 쓰면 grayscale 3채널 복제로 시작하거나 첫 layer를 명시적으로 변환한다.

그다음 아래 두 입력을 별도 비교한다.

- A: 정규화한 원본.
- B: `[정규화 원본, 국소 대비 영상, limb-darkening 보정 대비 영상]`.

국소 대비는 CLAHE 또는 local mean/variance normalization을 작은 후보군으로 비교한다. 과한 CLAHE는 노이즈와 경계를 왜곡할 수 있다. 원본 정보는 반드시 보존한다. 채널 의미를 바꾸는 경우 RGB pretrained weights의 입력 초기화도 기록한다.

태양 원반이 확인되는 데이터라면 중심·반경을 추정하고, 반지름별 robust intensity profile로 limb darkening을 보정하는 후보를 만든다. 원반 추정 실패와 가장자리 노이즈는 fallback으로 처리한다. 물리적 마스크로 실제 정답을 지우지 않도록 한다. 전처리 통계는 기본적으로 train fold에서만 적합한다.

### 해상도

- 전체 영상을 512²로 축소하는 것만으로 최종 모델을 만들지 않는다. 얇은 구조가 사라질 수 있다.
- native-resolution 768² 또는 1024² crop을 주 기준으로 사용한다. A100에서는 1024²–1536²를 비교하되 실제 폭 분포와 메모리를 먼저 확인한다.
- train crop과 추론 타일 크기 차이, 필라멘트의 잘린 정도를 기록한다.
- 타일 정답이 일부 잘리는 학습과 전체 객체를 평가하는 단계의 차이를 오류 분석에 포함한다.

### 샘플링·증강

- 시작값은 foreground 중심 crop과 uniform crop을 약 1:1로 섞는다. 데이터의 실제 분포에 따라 조정한다.
- 전경 crop을 면적에만 비례해 고르면 작은 필라멘트가 소외될 수 있어 객체별 샘플링을 비교한다.
- 학습 데이터 안에서 오탐이 많은 배경을 hard negative로 재사용한다. validation/test 정답을 학습에 넣지 않는다.
- 회전·반전, 제한된 밝기/gamma/대비 변화, 현실적인 blur/noise를 사용한다. 모든 공간 변환은 마스크에도 동일하게 적용한다.
- augmentation 후에도 필라멘트가 실제 관측 형태와 폭을 유지하는지 확인한다. 강한 elastic deformation, 무차별 CutMix/Mosaic는 기본 설정에서 제외한다.

## 5. 모델 개발 순서

### M0: 반드시 먼저 만드는 기준선

**ResNet34 U-Net 또는 ConvNeXt-T U-Net + foreground head**.

- native crop으로 학습하고, 겹치는 타일을 전체 영상 확률맵으로 복원한다.
- 기본 loss는 BCE + soft Dice. 시작 가중치는 1:1이며 PQ로 조정한다.
- 가장 단순한 threshold + connected components(CC)로 인스턴스를 만든다.
- 이는 빠르게 오류를 측정하기 위한 기준선이다. CC를 대회의 객체 정의라고 가정하지 않는다.
- pretrained weights는 규정상 허용되는 경우에만 사용한다. 불허 시 random initialization과 augmentation을 별도 예산으로 평가한다.

### M1: 권장 주력 후보

**ConvNeXt-T/S encoder + U-Net 계열 고해상도 decoder + foreground/boundary/affinity heads**.

- foreground head: 필라멘트 픽셀 확률.
- boundary head: 인스턴스 경계·접촉부 구분을 돕는 보조 출력.
- local affinity head: 선택한 이웃 픽셀 쌍이 같은 인스턴스인지 예측. 다른 방향/간격의 pair를 포함하되 지나치게 많은 head는 피한다.
- 선택적 spine head: 업로드 정답의 실제 `spine` 좌표를 rasterize해 얇은 구조의 연속성을 보조한다. 제공 spine과 mask의 정렬·coverage를 검증하고, 유효하지 않은 spine을 임의의 음성 정답으로 만들지 않는다. 직접 제공된 중심선과 mask에서 파생한 skeleton은 구분한다.

실제 annotation 확인 후 권장 순서는 `foreground only → +제공 spine → +boundary → 필요한 경우 affinity`다. 여러 보조 손실을 처음부터 동시에 넣어 원인을 알 수 없게 만들지 않는다. 각 보조 출력은 학습 안정화에만 쓰는 경우와 decoder에 직접 쓰는 경우를 분리 평가한다.

Loss 후보:

\[
L = L_{BCE}+L_{Dice}+\lambda_b L_{boundary}+\lambda_a L_{affinity}+\lambda_t L_{topology}
\]

`lambda_b=0.1–0.3`, `lambda_a=0.1–0.3`, `lambda_t=0 또는 0.05–0.1`은 **탐색 시작 범위**다. gradient 크기·양성 빈도·loss 정규화에 따라 효과가 달라진다. topology/clDice는 서로 다른 객체를 연결할 위험이 있어 기본 필수 손실로 두지 않는다. binary union에 대한 연결성 손실은 객체 정체성을 학습하지 못한다.

다중 annotator인 경우:

- 같은 이미지에서 학습 step마다 유효한 annotator 하나를 뽑아 그 정답의 인스턴스 ID로 학습하는 방식을 기준으로 둔다.
- annotator 수에 비례해 영상 가중치가 커지지 않도록 image-first sampling을 사용하되, 공식 평가 가중치와의 차이를 확인한다.
- BCE의 soft consensus는 보조 후보로 사용할 수 있지만, 인스턴스 ID·경계·Dice를 단순 합집합 정답으로 대체하지 않는다.
- 부분 annotation인지 완전 annotation인지 확인하기 전에는 annotation 없는 영역을 모두 확실한 배경으로 가정하지 않는다.
- annotator-specific head는 일관된 스타일 차이가 충분히 확인되고 추론 시 최종 단일 예측으로 결합할 방법이 있을 때만 추가한다.

실제 데이터에서는 영상 411장이 정답 세트 1개, 145장이 2개, 151장이 3개를 가진다. prefix는 37종이지만 세부 개인 식별 의미를 숫자만으로 추정하지 않는다. 제공 spine은 8,199개 모두 비어 있지 않다. 길이는 중앙값 약 111px, 최장 약 1,105px로 고해상도 지역 정보와 긴 구조 문맥을 함께 비교할 근거가 된다.

### M2: 전역 문맥을 결합한 M1

전체 영상을 768²–1024² 수준으로 보는 경량 branch에서 문맥 feature를 만들고, 원본 좌표에 대응하는 feature를 local decoder에 결합한다. 실제 입력 크기는 원본 해상도와 자원에 맞춘다.

- local branch가 미세 경계를, global branch가 긴 구조의 전체 문맥을 담당한다.
- global feature를 캐시할 경우 encoder 업데이트와 augmentation 좌표 변환 때문에 캐시가 낡지 않도록 한다. 초기 구현은 end-to-end 또는 명시적으로 고정한 encoder를 사용한다.
- 저해상도 global prediction으로 local 후보를 강제로 삭제하지 않는다. 얇은 실제 필라멘트가 global 영상에서 보이지 않을 수 있다.
- semantic global feature만으로 떨어진 조각의 동일 객체성이 해결된다고 보장하지 않는다. 실제 분리 오류 감소를 측정한다.

### M3: 객체 정의가 요구하면 추가하는 인스턴스 후보

**Mask2Former 계열 전역 mask-query 모델 + 원본 해상도 local refinement**.

다음 경우에 우선순위를 높인다: 정답 객체가 여러 disconnected component로 구성되거나, semantic+CC가 객체 수를 반복적으로 잘못 추정하는 경우. 특히 데이터 감사에서 disconnected-instance가 흔하다고 확인되면 E7까지 기다리지 않고 M1과 조기 비교한다.

현재 GT 전수 검사에서는 8,199개 중 30개만 8-connected component가 2개이며 대부분은 극소수 고립 픽셀이다. 이 근거만으로 복잡한 query 모델을 우선 도입하지 않는다. 예측 자체의 끊김·병합 오류는 별도 validation 예측으로 확인해야 한다.

GT 자체의 union→8CC(min_area20) 왕복 진단은 PQ 약 0.9989로, 정답 기하에 대한 직접 손상은 작았다. 이는 실제 추론 성능이나 달성 가능한 점수가 아니다. 우선순위는 공식 PQ 기준 모델 선택과 고해상도 foreground/spine 학습에 두고, 후처리 교체의 이득은 실제 예측으로 검증한다.

- query 수는 학습 이미지당 객체 수의 상위 분위수와 최대값을 보고 여유 있게 정한다.
- no-object class와 Hungarian matching을 사용하고 작은 객체에 충분한 mask sampling을 확보한다.
- global mask와 이미지 feature로 local refinement를 조건화한다. refinement가 픽셀 정확도를 개선하면서 query의 객체 정체성을 보존하도록 설계한다.
- 고해상도 local-only 후보도 유지해 global branch가 놓친 작은 객체를 복구한다.
- backbone·입력 해상도·decoder stride 때문에 미세 필라멘트가 손실되지 않는지 확인한다.
- M1/M2보다 grouped OOF PQ가 좋거나, 앙상블에 추가할 때 개선될 경우에만 최종 사용한다. 구현 비용이 높아 초기 필수 단계로 두지 않는다.

## 6. 학습 설정과 GPU 사용

공통 시작 설정: AdamW, encoder LR 약 `1e-4`, decoder LR 약 `3e-4`, weight decay 약 `1e-2`, 짧은 warmup 후 cosine decay. 이는 고정 정답이 아니라 개발 fold에서 검증할 시작값이다. encoder freeze/unfreeze, EMA, deep supervision은 각각 별도 실험으로 추가한다.

에폭 수만 고정하기보다 optimizer step과 관찰한 이미지/crop 수를 기록한다. 작은 pilot으로 loss 감소·전체 영상 PQ·GPU 사용량을 확인하고 본 학습 예산을 정한다. checkpoint 선택 기준은 **전체 영상 PQ**이며, loss나 타일 Dice가 아니다.

| 항목 | T4 16GB | A100 40GB/80GB |
|---|---|---|
| 우선 용도 | 데이터/평가 검증, M0/M1 pilot, 후처리 탐색 | 고해상도 M1/M2, 최종 교차검증, 선택적 M3 |
| encoder 시작 후보 | ResNet34, ConvNeXt-T | ConvNeXt-T/S, 필요 시 더 큰 독립 후보 |
| crop 시작값 | 768², 여유 시 1024² | 1024², 여유 시 1280²–1536² |
| micro-batch 탐색 | 1부터 증가 | 2부터 증가; 모델·VRAM에 맞춰 측정 |
| mixed precision | FP16 autocast + GradScaler | BF16 가능 여부 확인 후 우선 비교 |
| effective batch 후보 | accumulation으로 약 8–16 | 약 8–16에서 비교 후 조정 |

실제 batch fit·속도·총 학습 시간은 모델/입력/channel/global branch를 포함한 짧은 프로파일 후 확정한다. A100이라는 이름만으로 80GB나 다중 GPU를 가정하지 않는다. gradient accumulation은 BatchNorm 통계를 큰 batch처럼 만들지 못하므로 작은 batch에는 GroupNorm/LayerNorm 또는 검증된 frozen BN을 사용한다. 데이터 로더 worker와 캐시는 CPU/RAM/스토리지 측정으로 결정한다.

초기 50–100 step에서 peak VRAM·처리량·optimizer update 수를 기록한다. Dice/topology loss의 reduction은 필요 시 FP32로 계산하고 공식 scorer의 수치형을 보존한다. 2048² FP16 확률맵은 head 하나당 약 8MiB이므로 모든 모델·TTA·head의 OOF를 무조건 저장하지 않고 후처리 탐색에 필요한 출력과 최종 예측을 구분한다.

여러 fold의 학습보다 **잘못된 metric·split·serialization을 고치는 것**이 먼저다. 동일 설정을 두 GPU에서 비교할 때는 step 수·effective batch·split을 맞춘다. 서로 다른 precision의 재현성 차이도 기록한다.

## 7. 추론과 인스턴스 복원

### 7.1 원본 전체 영상의 연속 예측

1. native-resolution 타일에 25–50% overlap을 둔다. overlap은 필라멘트 폭과 decoder의 경계 오류를 보고 결정한다.
2. 각 타일의 공간 출력을 원본 좌표로 복원한다.
3. Hann/Gaussian 계열 가중치로 중앙부를 더 신뢰하며 foreground와 boundary를 각각 합친다. 영상 가장자리의 누적 가중치가 0이 되지 않게 padding 또는 양수 하한을 둔다.
4. affinity는 global pixel-pair 좌표와 방향을 일치시켜 집계한다. 타일 경계를 넘는 edge의 누락도 처리한다.
5. **전체 영상에서 한 번 인스턴스를 생성한다.** 타일별 CC 결과를 그대로 이어 붙이지 않는다.

### 7.2 decoder 후보

- D0: threshold + CC. 최소 기준선.
- D1: foreground 지지 영역 안에서 boundary/affinity를 사용해 붙은 구조를 분리하는 graph decoder.
- D2: D1 + 제한적인 fragment merge. 끝점 거리·방향·그 사이 영상/foreground 근거·boundary·전역 객체 근거를 함께 사용한다.
- D3: query 인스턴스 마스크 + local refinement와 객체 중복 제거.

긴 객체에 여러 seed가 생겨 과분할될 수 있으므로 일반 watershed를 무조건 적용하지 않는다. merge도 가까운 조각의 전이적 연결만으로 결정하지 않는다. 병합 후보마다 경쟁 객체·경계 증거를 재검토하고 잘못된 연쇄 병합을 측정한다.

정답 객체가 disconnected라면 하나의 인스턴스에 여러 component를 허용할 수 있어야 한다. 반대로 정답이 component 단위라면 임의로 합치지 않는다. 이 선택은 정답 감사 결과에 따른다.

### 7.3 PQ 기준 보정

개발 OOF에서 다음을 **작고 제한된 탐색 공간**으로 조정한다.

- foreground threshold, hysteresis 유무와 두 threshold.
- object confidence/quality threshold, duplicate suppression.
- minimum area/length: 반드시 무필터 후보를 포함한다. 공개 타인의 특정 픽셀 수를 그대로 쓰지 않는다.
- boundary/affinity threshold와 fragment merge 기준.

하이퍼파라미터는 공식 PQ로 고른다. 작은 객체를 많이 지우거나 마스크를 일괄 팽창시켜 한 validation 점수만 올리는 설정은 strata별 결과와 holdout에서 검증한다. confidence를 쓰면 validation에서 매칭/IoU와 관계가 있는지 확인한다.

## 8. 앙상블과 TTA

1. 같은 검증 그룹을 제외하고 학습한 seed/backbone 모델들의 예측부터 결합한다. 각 평가 이미지에 대해 그 이미지와 동일 그룹을 학습에 사용한 모델은 절대 포함하지 않는다.
2. semantic head는 역변환한 foreground/boundary 확률을 평균한 뒤 인스턴스를 한 번 복원한다.
3. query 모델/완성 인스턴스를 섞는 경우, 인스턴스를 IoU·shape·global 근거로 대응시킨 후 대응 객체끼리 결합한다. instance ID 정수 자체를 평균하지 않는다.
4. 대응하지 않는 후보는 증거와 quality에 따라 유지/제거한다. 모든 모델의 마스크를 단순 합집합으로 제출하면 FP와 overlap이 늘 수 있다.
5. flip TTA부터 검증하고, 90도 회전을 포함한 D4는 좌표를 정확히 역변환한다. affinity 방향과 벡터 출력도 함께 역변환한다. scale TTA는 얇은 구조 손상과 비용을 따로 비교한다.
6. disjoint masks가 요구되면 겹친 픽셀을 confidence/quality와 픽셀 확률로 하나의 객체에 배정한다. 이 과정 후 IoU와 빈 객체 제거를 재평가한다.
7. 단일 모델 대비 이득을 누수 없는 검증에서 확인하고 추론 예산에 맞는 조합만 최종 채택한다. 일반적인 K-fold에서 한 이미지에 대한 OOF 모델은 그 이미지를 제외한 한 fold 모델뿐이다. 나머지 fold 모델들은 그 이미지를 학습했으므로 이들을 섞어 OOF 앙상블 점수를 계산하면 누수다. 여러 fold 모델 전체의 앙상블 이득은 모든 모델의 학습에서 제외한 독립 holdout이나 nested grouped 검증으로 확인한다. 모델별 OOF 예측을 결합할 때도 각 이미지의 예측이 모두 해당 그룹을 제외한 학습에서 나왔는지 검사한다.

## 9. 실험 우선순위와 채택 기준

| 순서 | 실험 | 확인할 문제 | 다음 단계 조건 |
|---|---|---|---|
| E0 | 공식 scorer/제출 round-trip, 데이터 감사, split 고정 | 점수와 좌표·누수 오류 | 평가 계약과 정답 mapping 정상 |
| E1 | M0, raw 입력, native tiles | 학습과 추론 baseline | 전체 이미지 PQ와 오류 사례 확보 |
| E2 | 입력 표현·해상도·샘플링 비교 | 희미한/작은 구조 미검출 | PQ 개선과 비용 비교 |
| E3 | M1 보조 head 개별 추가 | 붙음/끊김 오류 | RQ/PQ 개선, SQ 손상 확인 |
| E4 | M2 전역 문맥 | 긴 객체·타일 문맥 | 객체 오류 감소가 OOF에서 반복 |
| E5 | 제한된 decoder·threshold 탐색 | FP/FN, split/merge 균형 | holdout/다른 fold에서 유지 |
| E6 | 상위 2–3 후보 grouped folds | 모델 선택의 안정성 | fold별·전체 PQ 확인 |
| E7 | 필요 시 M3, 독립 백본 | 기존 모델이 해결 못 한 객체 정의 | 추가 비용 대비 OOF 이득 |
| E8 | fold/모델 앙상블·TTA | 오차 상보성 | 단일 모델 대비 실제 이득 |
| E9 | 고정 설정 최종 검증·전체 train 재학습 또는 fold ensemble | 제출 준비 | 제출 재로딩·shape·ID 검사 통과 |

자원이 제한되면 E7보다 E0–E6의 완성도를 우선한다. 더 큰 모델, 자기지도 사전학습, 외부 magnetogram 융합, pseudo-label은 기본 파이프라인이 안정화되고 규정·자원·추가 이득이 확인될 때만 도입한다. test pseudo-label을 한다면 허용 여부를 확인하고, validation 정답을 통한 누수를 막는다. 미검출을 숨기는 과도한 pseudo-label confidence는 피한다.

모든 실험은 `run_id, config, seed, split_hash, data_hash, evaluator_hash, checkpoint, OOF_prediction_path, PQ, SQ, RQ, FP, FN, runtime, peak_VRAM`을 남긴다. split/merge는 공식 매칭에서 직접 산출되지 않을 수 있어 GT-예측 overlap graph에 근거한 별도 진단 정의를 기록한다.

작은 PQ 차이는 한 번의 seed·한 fold 결과만으로 확정하지 않는다. 공식 집계 기준의 paired group 비교와 비용을 함께 본다. 개선 없는 head/증강/후처리는 최종 파이프라인에서 제거한다.

## 10. 구현 산출물과 제출 점검

아래는 **구현할 구조**이며 현재 실행 가능한 코드가 만들어진 것은 아니다.

```text
configs/                데이터·모델·추론 설정 및 고정 split 참조
src/data/               manifest, annotation decoder, grouped splits, crops
src/models/             M0/M1, 선택적 global context와 query 모델
src/losses/             foreground, instance boundary, affinity, topology
src/train.py            AMP, checkpoint, 전체 영상 validation
src/infer.py            native tiling, TTA 역변환, 전역 출력 복원
src/instances.py        decoder, quality filtering, 객체 대응·중복 제거
src/evaluate.py         공식 scorer 호출, PQ/SQ/RQ, strata·오류 리포트
src/submit.py           공식 형식 직렬화와 decode 검증
artifacts/              manifest, splits, OOF, reports, run metadata
```

- 원본 데이터·학습 가중치·예측 캐시는 Git에 커밋하지 않는다. 별도 저장소/허용된 로컬 경로와 checksum으로 관리한다.
- dependency와 pretrained weight 버전·라이선스·다운로드 출처를 고정한다.
- 테스트 image_id 누락/중복, 이미지 크기, 유효 픽셀 좌표, 인스턴스 overlap, mask encoding, 빈 예측 표현을 확인한다.
- ID map을 중간 저장할 때 객체가 255개를 넘을 수 있으면 uint8 손실을 피한다. 최종 포맷은 공식 제출 계약에 맞춘다.
- 실험 결과와 최종 모델의 run_id를 연결한다. 제출 폴더를 매번 새로 만들어 오래된 예측이 섞이지 않게 한다.
- clean process에서 checkpoint 로드→추론→저장→재로딩을 검증하고 T4/A100 중 실제 제출 환경의 시간·메모리를 측정한다.
- 모델 선택 이후 full-train을 재학습하면 calibration이 달라질 수 있으므로 ensemble/final checkpoint 전략을 사전에 정하고, test/LB를 보고 임의 조정하지 않는다.

## 11. 실제 데이터가 전달되면 먼저 할 일

정답 ZIP·제출 CSV·평가/베이스라인 노트북은 확보했다. 파일명을 알려준 `20110109104734Ch.jpeg`는 GT 그룹 `030101-20110109104734Ch`, 인스턴스 6개와 연결된다. 다음은 학습·기존 모델 비교에 필요한 자료다.

추가로 `best.pt`와 `train_split.csv`를 확보했다. 모델은 Ultralytics 8.4.182의 YOLO26s-seg이며 실제 저장 설정의 batch는 2다. split은 train 616개 실제 영상, metadata complement 91개로 구성된다. 원래 validation 재현 여부는 별도 확인한다. 사용자는 원본이 Google Drive에 있고 기존 노트북의 train_images 경로와 같다고 확인했다. **전체 이미지를 채팅에 업로드할 필요 없이 [자체 포함 Colab 노트북](../notebooks/Evaluate_Existing_YOLO_Colab.ipynb)으로 기존 위치에서 평가한다.**

1. 최신 규정과 전체 test image 목록/빈 예측 제출 정책.
2. 파일명이 보존된 대회 원본 train/test JPEG. 소규모 smoke test에는 5–10장으로 시작할 수 있지만, 경쟁 모델 학습에는 전체 train이 필요하다.
3. 확보한 `best.pt`와 train split으로 추론·후처리 비교를 수행한다. 원래 `val_split.csv`가 있으면 우선 사용하고, 없으면 train-complement 진단이라고 명시한다. 현재 제출 CSV만으로는 union 이전 객체와 confidence를 복구할 수 없지만 체크포인트에서 새 추론이 가능하다.

개인 문서가 섞인 Windows 폴더 전체를 공유할 필요는 없다. 대회 관련 원본 파일을 클라우드에 배치한 뒤 E1 학습을 시작한다. JSON의 공개 NSO URL에서 가져온 참조 영상은 바깥쪽 로고/텍스트 등이 대회 영상과 다를 수 있어 동일 파일로 간주하거나 학습 데이터를 무단 대체하지 않는다.

## 12. 근거와 해석 한계

- 공식 대회: https://www.kaggle.com/competitions/filament-segmentation-2026
- 주최 측 공개 안내: https://solarnews.aas.org/2026/solar-filament-segmentation-challenge-2026/
- PQ 원 논문: https://arxiv.org/abs/1801.00868
- Mask2Former: https://arxiv.org/abs/2112.01527
- clDice: https://arxiv.org/abs/2003.07311

본 설계에서는 공개 안내와 업로드 정답 ZIP·평가/베이스라인 노트북·제출 CSV·표시용 이미지 및 공개 NSO 참조 JPEG 1장을 활용했다. Kaggle 최신 규정과 backend 구현은 확인하지 못했다. 사용자 보고 LB 0.37을 로컬에서 재현하지 않았으며, 신규 모델 학습도 수행하지 않았다. 공개 참가자 결과나 논문 성능을 본 대회의 최고점 근거로 사용하지 않는다.
