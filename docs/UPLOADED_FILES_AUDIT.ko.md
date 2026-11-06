# 업로드 파일과 이미지 분석

## 범위와 실행 상태

사용자의 요청은 PQ 중심 모델/파이프라인 설계이며, 첨부 노트북 내부 주석·설치·Drive mount·학습 명령을 별도의 사용자 실행 지시로 취급하지 않았다. 노트북을 통째로 실행하지 않고 JSON의 source cell을 읽어 분석했다.

- `self-evaluation-notebook.ipynb`: 평가 계약을 확인했다. 아래 cell 번호는 0부터 시작한다.
- `SolarFilament_LimbOnly_YOLO26_Train.ipynb`: 라벨 변환·전처리·학습·후처리·자체 평가를 확인했다. 저장된 실행 결과가 없어 학습 성공·점수를 입증하지 않는다.
- `yoloseg_limb_conf0.25.csv`: 1,145행, 필수 두 열, 중복 ID·빈 문자열 없음. 실제 RLE 검사는 별도 실행 결과를 따른다.
- `MAGFiLO_1.0_Annotations_kaggle2026_train.json`: 처음 32MiB 전송 제한으로 막혔으나, 추가 ZIP(약 7.7MB)에서 정상적으로 읽었다. JSON은 48,616,700 bytes이며 SHA-256은 `5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1`이다.
- 이미지 3장: 채팅에서 시각 검토했다. 원본 2048² → 표시용 1600² 축소 안내가 있다. 대회 원본 파일은 현재 파일시스템에 없지만, 이름을 확인한 영상의 공개 NSO 참조 JPEG를 별도로 받아 GT overlay를 만들었다. 대회 JPEG와 byte-level 동일성을 확인한 것은 아니다.
- 사용자는 제공 CSV의 리더보드 점수가 **0.37**이라고 보고했다. 로컬 재현 점수와 구분한다.

## 1. 제공 평가기의 실제 계약

### 입력

두 dataframe 모두 `filament_id`, `segmentation_rle` 열을 사용한다.

- 예측: `<image>_<instance>`.
- GT: `<annotator>-<image>_<instance>`.
- parser는 첫 `_` 앞을 잘라 사용한다. GT는 그다음 첫 `-` 뒤를 image key로 사용한다. 실제 JSON의 image_id 숫자와 이 key를 같다고 가정해서는 안 된다.
- RLE는 **compressed COCO counts 문자열**이다. Kaggle의 다른 대회에서 쓰는 단순 `start length` 목록과 다르다.
- decode 크기는 기본 2048×2048에 고정된다. CSV 자체에는 shape가 없다.

근거: cell 2, cell 3의 `rles_to_layers`, `process_entry`, `get_overlap_df`.

### 점수

cell 4 `get_pq_score`는 다음을 수행한다.

1. annotator–image별 IoU 행렬에 `IoU > 0.5`를 적용한다.
2. qualifying pair **전부**를 TP IoU 목록에 추가한다. 명시적인 일대일 matching은 없다.
3. 매칭이 하나도 없는 prediction column을 FP, GT row를 FN으로 센다.
4. 모든 entry를 누적해 `sum(TP IoU)/(TP + 0.5 FP + 0.5 FN)`을 한 번 계산한다.
5. denominator가 0이면 0을 반환한다.

즉 `mean(PQ_per_image)` 또는 `mean(PQ_per_annotator_image)`와 다르다. 같은 영상의 예측은 해당 영상의 여러 annotator 정답에 반복 비교된다. **checkpoint·confidence·후처리 선택 모두 전역 누적 PQ로 통일**해야 한다.

평가기의 한계도 보존·분리한다.

- `get_overlap_df`는 GT row에서만 entry를 발견한다. GT에 없는 이미지의 prediction은 이 노트북에서 순회되지 않는다.
- GT 객체가 0개인 영상/annotator는 row가 없어 기본 dataframe만으로는 entry가 생성되지 않는다.
- 전체 GT가 비면 overlap table 생성이 실패할 수 있다.
- 겹친/중복 마스크, 잘못된 ID, 빈 RLE, 전체 test coverage를 검증하지 않는다.
- 노트북의 실행 호출은 주석 처리되어 있다. Run All만으로 실제 점수가 계산되지 않는다.
- dense mask를 CPU torch tensor로 만들어 overlap을 계산한다. GPU로 보내는 코드는 없다.

이 한계를 이용해 점수를 올리는 예측을 만들지 않는다. 원본의 점수 의미는 유지하되, ID coverage·중복·마스크 유효성을 별도 검사한다. 규정 확인 전에는 겹친/중복 예측을 허용된 최적화라고 가정하지 않는다.

## 2. YOLO 베이스라인에서 우선 비교할 부분

| 우선순위 | 코드 근거 | 실제 동작 | 권장 비교 |
|---|---|---|---|
| 1 | cell 22 | greedy matching 후 record별 PQ 평균 | 제공 평가기의 전역 누적으로 다시 평가·threshold 선택 |
| 2 | cell 20 `build_final_instances` | YOLO 마스크 전체 union → connected components | 개별 mask 보존·중복/overlap 처리 대 union+CC |
| 3 | cell 14 `mask_to_yolo_segments` | annotation 하나도 component별 별도 YOLO 객체로 분리 | 원본 GT와 polygon round-trip의 객체 수·마스크 손실 감사 |
| 4 | cell 8 | 중심 (1024,1024), 반경 900 밖을 모두 0 | GT 보존율·실제 원반 geometry 확인; raw 입력과 비교 |
| 5 | cell 18 | `mask_ratio=4`로 축소된 mask supervision | 메모리 내에서 mask_ratio 2/1과 native-resolution decoder 비교 |
| 6 | cell 18 | 기본 Ultralytics validation 기준 best.pt/patience | checkpoint를 전체 영상 공식 PQ로 재선택 |

`LimbOnly`는 limb-darkening 보정만 사용한다는 뜻이다. 태양 가장자리만 학습한다는 뜻은 아니다.

### 객체 정체성 손실

학습 변환은 `RETR_EXTERNAL` contour를 사용하므로 hole을 버리고 작은 component, 폭/높이가 2 미만인 것, contour 면적 0인 가느다란 구조를 제거할 수 있다. 추론에서도 union+CC가 YOLO의 객체 ID를 버린다.

- 서로 닿아도 다른 정답 객체면 union이 잘못 병합할 수 있다.
- 떨어진 조각이 하나의 정답 객체면 component 분리가 잘못 과분할할 수 있다.
- 실제 정답이 component 단위라면 이 후처리가 유리할 수도 있으므로, 정답을 보기 전에는 일괄 제거하지 않고 비교한다.

### 분할·synthetic·캐시

- 같은 `file_name`의 COCO records를 같은 split에 둔 점은 유지한다.
- 다른 이름의 근접 관측 영상·동시 관측소 간 유사 영상은 추가 그룹화가 필요할 수 있다.
- `USE_SYNTHETIC_DATA=True`가 기본이며, 현재 공유되지 않은 synthetic 이미지·JSON·split 파일이 필요하다. 우선 original-only 기준선을 둔다.
- synthetic의 source뿐 아니라 background 출처와 미라벨 실제 필라멘트도 확인한다.
- split을 바꿔도 기존 YOLO images/labels 폴더가 남을 수 있다. 설정별 새 디렉터리와 manifest 검증을 사용한다.
- 캐시 key에 이미지 내용·전처리·annotation·변환 버전을 넣는다. 기존 사용자 파일을 지우는 방식으로 해결하지 않는다.
- `REBUILD_YOLO_LABELS=False` 상태로 변환 규칙을 바꾸면 오래된 라벨이 재사용될 수 있다.
- `MAX_DET=100`, `MIN_COMPONENT_AREA=20`의 영향은 실제 객체 통계와 무필터 비교로 검증한다.

## 3. 제공 CSV의 해석

- 열 이름: `filament_id`, `segmentation_rle`.
- 1,145개의 예측 인스턴스와 177개의 image key를 포함한다. 전수 디코딩 결과 유효한 2048² compressed RLE이며, 최소 면적 21·최대 23,384 pixels, 중복 ID·빈 마스크·겹친 마스크는 없다.
- 객체가 하나도 예측되지 않은 영상은 행이 없을 수 있다. 177개를 전체 test 영상 수라고 단정할 수 없다.
- 이 CSV는 예측 결과의 형식을 보여준다. 전체 test image 목록 또는 반드시 채워야 할 sample submission 명세를 증명하지 않는다.
- train GT와 이 CSV의 image key가 대응하는지 확인하기 전에는 PQ를 계산하거나 보고하지 않는다. test 예측을 train GT에 임의로 대조하면 유효한 검증이 아니다.

## 4. 이미지 3장이 설계에 주는 정보

눈으로는 희미하고 가는 어두운 구조, 강한 표면 질감, 밝은 원반 경계, 검은 외부와 밝기 변화가 확인된다. 이 관찰만으로 어떤 픽셀이 정답 필라멘트인지는 확정할 수 없다.

- 원본 2048² 정보와 희미한 선을 유지하는 native-resolution 학습을 우선한다.
- `raw`, `limb-corrected`, `raw+contrast`를 독립적으로 비교한다. 이미 보정된 영상인지도 원본 metadata와 처리 이력에서 확인한다.
- 임의의 밝기 threshold나 모든 영상에 고정한 원반 geometry로 정답을 지우지 않는다.
- 원반 가장자리, 흑점/활동 영역, 표면 질감을 hard negative 분석에 포함한다.
- 채팅 축소본을 annotation에 맞추기 위해 다시 2048²로 확대해 원본인 것처럼 사용하지 않는다.

## 5. 확정된 다음 실험 순서

1. 원본 파일명과 압축 정답 JSON으로 image/annotator mapping을 검증한다.
2. 기존 validation 예측 또는 best.pt가 있으면 동일 split에서 **제공 evaluator 의미의 점수**를 기준선으로 만든다.
3. 동일 YOLO 출력에 `union+CC`와 `개별 instance 보존`을 비교한다. 새 모델 학습보다 먼저 오류 원인을 분리할 수 있다.
4. 원본 GT→YOLO polygon→복원 mask 손실을 수치화한다. disconnected-instance가 많으면 query/embedding 인스턴스 모델을 조기에 비교한다.
5. split/cache를 고정하고 raw 대 limb 입력, mask supervision 해상도를 비교한다. 실제 정답에서 반경 제한·20px 필터 손실이 작으므로 이를 성능 저하의 주요 원인으로 미리 단정하지 않는다.
6. 결과를 기준으로 고해상도 ConvNeXt U-Net + boundary/affinity, 전역 문맥 branch와 비교한다.
7. 상위 후보만 grouped 교차검증·독립 holdout 앙상블 평가로 확장한다.

## 6. ZIP에서 확인한 실제 annotation 구조

- image records 1,154개, 고유 file_name 707개, annotation 8,199개. image metadata 크기는 모두 2048²이다.
- image `id`가 이미 `<prefix>-<physical_image_stem>` 형식이다. prefix의 세부 의미는 별도 데이터 설명 없이 숫자만으로 추정하지 않는다.
- segmentation은 실제 list이며, 현재 8,199개 모두 polygon 1개를 가진다. spine과 bbox도 JSON 문자열이 아닌 list다.
- category 정의는 Left/Right/Unidentifiable/Ambiguous이다. 실제 annotation 수는 각각 2,535/2,590/3,074/0개다. 제출에는 class 열이 없으므로 segmentation 기준선에서는 임의로 특정 category를 제외하지 않는다.
- 변환 시 원본 annotation 하나를 RLE 인스턴스 하나로 유지하고, annotation UUID와 생성 filament_id의 대응을 manifest에 남긴다. 여러 polygon인 annotation도 별도 객체로 쪼개지 않고 합쳐 처리한다.
- `20110109104734Ch.jpeg` → image ID `030101-20110109104734Ch`, annotation 6개. 실제 raster 면적은 317/533/846/270/465/2,644 pixels이다.
- [정답과 spine 참조 그림](evidence/20110109104734Ch_gt_reference.png)은 공개 NSO JPEG를 사용했다. 외부 로고/텍스트가 사용자 이미지와 다르므로 학습 데이터 대체본이나 대회 원본 일치 증거로 사용하지 않는다.

## 7. 평가 도구 검증

`tools/filament_eval.py`는 dense Torch mask 대신 compressed RLE IoU를 사용해 제공 노트북의 전역 집계 의미를 재현한다. ID/CSV/RLE·겹침·coverage 검사를 추가하되 마스크를 자동 수정하지 않는다.

- 평가 계약·입력 검증 unittest 13개 통과.
- 실제 notebook의 검토된 cell 3/4 함수와 2048² 합성 fixture 8개를 비교했다. 최대 PQ 차이는 약 `2.71e-8`로 설정한 `1e-6` 허용 오차 이내였다.
- 이는 도구의 합성 사례 검증이다. 실제 모델 PQ나 GPU 학습 준비가 입증된 것은 아니다.
- train GT와 test submission의 image key가 다르면 재채점할 수 없다. 그런 입력은 도구가 기본적으로 거부하여 노트북의 무음 누락을 방지한다.

## 8. 전수 정답 감사로 교정한 가설

정량 결과는 [annotation 감사 JSON](evidence/annotation_audit.json)에 저장했다.

- 관측 기간: 2011-01-09–2022-08-03. 고유 관측 날짜 656개이며, 관측소 suffix는 6종이다.
- annotation variant가 1개인 영상 411장, 2개 145장, 3개 151장이다.
- 객체 면적 중앙값 1,228px, 1% 분위 약 209px, 최대 37,739px. 전체 annotation group의 객체 수 중앙값 7개, 최대 26개다. `MAX_DET=100`이 GT 객체 수 때문에 제한되는 증거는 없다.
- 8-connected 기준 단일 component GT는 8,169개, 두 component GT는 30개다. 대부분 작은 raster 고립 픽셀을 동반하므로 disconnected GT를 근거로 고비용 query 모델을 첫 순위로 정하지 않는다.
- `<20px` component는 29개·총 55px이며, 이 필터가 완전히 없애는 GT 객체는 1개다. 이 항목의 영향은 현재 GT에서 작다.
- 중심 (1024,1024), 반경 900 밖의 GT 손실은 5개 객체·총 116px이다. 총 annotation 면적 대비 약 **0.000667%**로 작다. 다른 이미지 분포의 위험은 별도 확인하되 현재 baseline 점수의 주원인으로 과장하지 않는다.
- hole이 있는 raster GT는 917개다. hole 총면적과 실제 YOLO polygon round-trip IoU는 아직 별도 측정이므로, 개수만으로 큰 손실이라고 해석하지 않는다.
- 동일 annotation group 안의 객체 overlap은 5개 group·10개 객체·5,041px에서 확인되었다. GT도 항상 disjoint라는 가정은 성립하지 않는다.
- spine은 모두 유효한 짝수 좌표 list이며 비어 있지 않다. 길이 중앙값 약 111px, 95% 분위 약 387px, 최대 약 1,105px다.
- train 707개 stem과 업로드 제출 CSV 177개 stem의 교집합은 **0개**다. 따라서 로컬 모델 점수는 아직 측정하지 않았다.

추가로 각 annotation group의 GT 자체에 `union → 8-connected components → min_area=20`을 적용했다. 원본과 복원 마스크의 **annotation round-trip 진단 PQ는 0.998903**이며 객체 수가 변한 그룹은 12/1,154개였다. 이는 annotator마다 자기 정답에서 예측을 만든 진단으로, 실제 모델 PQ·성능 상한·도달 가능한 점수가 아니다. 따라서 union+CC 자체가 baseline 0.37의 주원인이라고 단정할 근거는 없다. 실제 예측의 끊김·붙음에서는 별도 비교가 필요하다.

우선적인 모델 개선 가설은 **공식 전역 PQ와 모델 선택의 일치, 원본 mask supervision 해상도, 제공 spine 활용, annotation 불일치를 고려한 검증**이다. 후처리·polygon 손실도 비교하되 그 영향의 크기는 실험 결과로 판단한다.

## 9. 추가 체크포인트·split과 이미지 이동 없는 검증

- 업로드 `best.pt`: 23,824,733 bytes, SHA-256 `81c2a27d95821b5ede283169c0da8c08d360ecc4c2a60a1b9c6baf96d76bcccc`.
- 실제 로드한 모델은 YOLO26s-seg, 11,434,463 parameters, 저장된 Ultralytics 버전 8.4.182다. batch=2, imgsz=2048, mask_ratio=4, mixed run 설정이다. 예시 notebook의 batch=1과 실제 저장 설정은 다르다. stripped checkpoint의 epoch=-1을 학습 종료 epoch로 해석하지 않는다.
- train split은 1,053개 annotation group/616개 실제 영상/7,491개 객체다. 나머지는 101개 group/91개 영상/708개 객체이며 물리 파일명 중복은 없다.
- 나머지 91장을 원래 validation이라고 확정하지 않는다. 또한 두 영상은 다른 관측소 train 영상과 20초 차이라, 독립 일반화 검증에서는 근접 관측 누수에 주의한다. 세부 사항은 [split 감사](evidence/train_split_audit.json)에 있다.
- 체크포인트 실제 로드와 2048² 공개 참조 JPEG 추론, 원시 mask confidence·후처리 출력, 1장 제한 CLI의 CSV·JSON 저장까지 확인했다. 참조 JPEG는 대회 원본과 byte 동일성을 확인하지 않았고 이 점검은 전체 validation이나 LB 0.37 재현이 아니다.
- 평가·변환·split·후처리·checkpoint 정책 테스트 36개를 통과했다. 자체 포함 Colab notebook의 6개 코드 셀과 내장 4개 모듈의 문법·소스 일치를 확인했다. 실제 Colab GPU/Drive 전체 평가는 아직 실행되지 않았다.
- 사용자 확인 Drive 경로를 사용하는 [Colab notebook](../notebooks/Evaluate_Existing_YOLO_Colab.ipynb)을 준비했다. 원본 이미지 전체 업로드 없이, 실행 후 작은 결과 JSON/CSV ZIP만 공유하면 된다.
