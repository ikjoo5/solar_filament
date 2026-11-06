# 외부 GONG 라벨과 영상 복원 조사

조사일: 2026-11-04, America/Chicago. 공개 공식 페이지·데이터 저장소·논문 원문을 확인했다. 이번 작업은 조사와 파일 목록/라벨 감사이며, 새 학습·추론·PQ 측정이나 제출 변경을 수행하지 않았다.

## 먼저 확인된 전처리 문제

- [MAGFiLO 원 논문](https://www.nature.com/articles/s41597-024-03876-y)의 **Data Conversion** 절에 따르면, 원래 FITS를 JPEG로 바꿀 때 Fourier 기반 디지털 필터로 limb darkening 등 큰 규모의 밝기 변화를 제거한다. `fits2jpeg`로 만든 영상이며 GONG 공개 JPEG와 같은 처리에 워터마크만 없다.
- [NSO H-alpha 설명](https://nso.edu/data/nisp-data/h-alpha/)도 현장 sharpening과 JPEG용 wavelet 기반 밝기 평탄화를 설명한다. 과학 분석용 FITS에는 후자의 wavelet 처리가 없다고 명시한다. 즉, 대회 JPEG와 무가공 관측을 같은 것으로 취급하면 안 된다.
- 우리의 `submission_inference.limb`는 이 JPEG 위에 반지름별 중앙값과 spline으로 추정한 보정을 추가한다. 따라서 추가 보정의 강도가 이득을 주는지, 남은 대비를 약화시키는지 비교할 필요가 있다. 이미 처리된 JPEG라는 사실만으로 현재 구현이 잘못됐다고 결론 내릴 수는 없다.
- 앞서 제공한 318개 시각화의 원본 칸은 **제공 JPEG 그대로**이며 우리 모델 입력의 추가 limb 보정을 적용하지 않는다. 작은 crop은 표시 크기에 맞춰 Lanczos로 확대된다. 화면의 흐림만으로 실제 모델 입력의 전처리 성공 여부를 판단하기 어렵다.
- best84는 `limb_only` 입력으로 학습됐으므로, 추론에서만 보정을 없애는 비교에는 학습/추론 입력 분포 변화도 포함된다. 전처리 자체의 최종 학습 효과를 주장하려면 입력 조건을 맞춘 학습 비교가 필요하다.

## 대회 규정과 관측 기간

**사용자 확인 반영:** 2026-11-04 사용자가 주최 측 이메일로 외부 데이터 사용 허용을 확인했다고 알렸다. 이 회신을 작업 조건으로 반영하며, 공개 페이지의 아래 문구만을 이유로 외부 데이터 활용을 보류하거나 재확인을 요구하지 않는다. 메일 원문을 직접 검토했다고 주장하는 것은 아니다. 날짜·물리 관측 중복과 validation 누수는 데이터 구성에서 별도로 점검한다.

[공식 Evaluation 페이지](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/evaluation), 2026-08-07 수정 문구:

> Although participants may use other external data sources, please note that: (1) for inference of ML models, only the H-alpha images provided in the test directory will be used; and (2) the models may not use any other ground-truth metadata for training.

- 일반 Rules의 외부 데이터 허용에는 주최 측 별도 제한이 우선한다는 조건이 있다.
- 이 공개 문구만으로는 **촬영일이 겹치지 않으면 외부 정답 학습이 자동으로 허용된다**고 해석할 수 없었다. 이후 사용자가 주최 측의 허용 회신을 확인했으므로 현재 작업은 그 조건을 따른다. 이번 조사에서는 아직 외부 데이터를 학습에 넣지 않았다.
- 사용자 제공 공식 파일명 목록 전체를 검사한 결과: train/validation707장은 **2011-01-09~2022-08-03**, test180장은 **2011-01-20~2022-07-28**이다.
- 2023년 이후 관측은 날짜 분리 후보가 된다. 데이터셋 공개 연도와 촬영 연도는 다르며, 이름이 다르더라도 같은 태양 현상을 시간차·다른 관측소에서 담을 수 있다.

## 외부 라벨 후보

### SWEFil: 가장 직접적인 후보

- [공식 공개 저장소](https://huggingface.co/datasets/antonio-reche/SWEFil), CC-BY-4.0. GONG H-alpha 영상의 **수동 필라멘트·흑점 라벨**, COCO bbox와 polygon segmentation을 제공한다.
- 2026년 후속 [논문](https://doi.org/10.1016/j.actaastro.2026.07.010)과 관련되지만, 확인한 공개 저장소는 2025-10 업데이트된 첫 버전이다. 확장판 계획을 공개 완료로 해석하지 않는다.
- 저장소 revision `bc818c69a79266ad0fa91a539359485af57967cf`의 실제 파일·COCO를 검사했다. 고유 관측277개, raw/processed 버전까지 세면554개 이미지 레코드다. 이 중 annotation이 있는 고유 관측은250개다.
- 전체의 중복 제거 라벨은 필라멘트1670개, 흑점402개다. README의3340/804는 같은 객체를 raw/processed 양쪽에 세어 두 배가 된 수치다. 같은 좌표·종류의 annotation 반복을 직접 확인했다.
- 촬영 기간은2010-06-04~2023-12-02이며, **2023년 관측14장에는 필라멘트300개와 흑점99개**가 있다. 이 수치는 두 영상 버전의 반복을 제거한 것이다. 최신 연도의 실제 라벨 후보를 확인했다는 점에서 의미가 있다.
- 대회887개 이미지와 정확한 파일명 stem 중복은0개였다. 전체 SWEFil의 같은 날·인접 시각·같은 현상 중복은 아직 감사하지 않았다. 사용자 확인에 따라 외부 데이터 활용을 진행하되, 우선 촬영 기간이 명확히 분리되는2023년14장을 후보로 삼는다.
- 흑점과 필라멘트 구별 연구에 특히 잘 맞는다. 다만 대회 정답과 객체 경계·분리 기준이 다른지도 검토해야 한다.

### CHASE 계열: 최신 공개지만 GONG과 구분

- [MHAS / Compound U-Net, Zenodo17230605](https://zenodo.org/records/17230605): 2025년 공개된 CHASE H-alpha120장과 수동 수정 binary mask 자료. GONG이 아니며 개체별 instance 라벨로 바로 간주할 수 없다.
- [MORDEN/AHAS 자료, Zenodo21582245](https://zenodo.org/records/21582245): 2026-07-26 공개. CHASE2024~2025 관측695장과 MHAS120장을 합친815장이다. 실제 압축파일 목록의 날짜·분할 수를 확인했다. 모델+CRF 출력 중 사람이 선별한 자동 라벨이며 최신 수동 GONG instance 데이터로 소개하면 안 된다.
- [Flat U-Net 자료](https://doi.org/10.12149/101545): 2025년 공개된 BBSO2022~2024 FITS 기반 자료다. 전통적 방법으로 생성한 이진 마스크 중 사람이 품질을 선별했다. 7.8GB 다운로드 링크는 확인했으나 파일 전체와 별도 데이터 라이선스는 검증하지 않았다.
- 같은 H-alpha라도 관측 장비·파장 대역폭·해상도 차이가 있다. 공개 시점이 최신이라는 이유만으로 대회 입력과 같은 분포가 되는 것은 아니다.
- 기존 MAGFiLO1.0은 대회 데이터와 대부분 겹치므로 새 독립 학습 자료로 재사용하지 않는다.

## 업스케일링과 복원

- **단순 확대**는 모델이 보는 픽셀 크기를 바꾼다. 우리 YOLO는 이미2048→2560 전체 영상 추론과1024 crop→1536/2048 확대 추론을 사용한다. 같은 확대를 다시 추가하는 것을 새로운 세부 정보 복원으로 해석할 수는 없다.
- **흐림 복원/초해상도**는 대기·광학·샘플링으로 약화된 정보를 추정한다. 유용할 수 있지만 가늘고 어두운 구조가 인공적으로 생기면 필라멘트 FP가 늘 수 있어, 선명도 점수보다 실제 분할 PQ·FP·FN으로 평가해야 한다.
- 다른 파장의 선명한 태양 사진은 같은 필라멘트의 고해상도 정답이 아니다. 복원용 대응 자료는 우선 같은 H-alpha, 관측 시각·파장 대역폭·정렬·스케일을 검토해야 한다.
- [ITI, Nature Communications2025](https://www.nature.com/articles/s41467-025-58391-4)는 KSO의 저품질/고품질 H-alpha 자료로 seeing·구름 영향을 줄이는 실험을 제시한다. [공개 코드](https://github.com/RobertJaro/InstrumentToInstrument)와 예제가 있어 방법론 후보로 적합하다. 그러나 관측은2008~2019년, KSO FITS 전제이며 GONG2048 JPEG에 바로 넣어 PQ가 개선되는 것으로 검증된 모델은 아니다.
- [NeuralBD2026](https://arxiv.org/abs/2603.05033)는 여러 단노출 프레임으로 PSF와 영상을 복원하는 방법이다. 여러 프레임이 가진 정보를 단일 대회 JPEG에서 이용할 수 있다고 가정하지 않는다.
- 이번 조사에서 바로 사용할 수 있는 검증된 **단일 GONG JPEG→고해상도 H-alpha 정답 쌍**은 확보하지 못했다.

## 권장 실험 순서

1. 같은 위치·표시 배율로 제공 JPEG / 현재 limb / 약한 limb를 나란히 확인한다. 반지름별 보정량, 포화 비율, 희미한 필라멘트와 주변 배경의 대비도 기록한다.
2. 기존 모델과 후처리를 고정해 입력 보정 강도를 비교한다. 이 결과는 기존 가중치에 대한 입력 민감도이며, 새 전처리로 재학습한 성능을 대신하지 않는다.
3. best84의1배/2배 보간 대조군과 보수적인 흐림 복원 후보를 비교한다. 원래 영상에서 보는 범위와 출력2048좌표를 맞추고 후처리 조건을 고정한다.
4. 복원 학습이 필요하면 우선 기존 train 안의 비교적 선명한 패치에 흐림·축소·잡음·압축을 가해 복원하도록 학습한다. 목표 영상 역시 GONG 품질이므로 실제 광학 해상도가 더 높아졌다고 주장하지 않는다. 손실은 픽셀·구조 보존과 재열화 일관성을 우선한다.
5. 새 학습은 기존 고정 train/validation으로 비교하고, 전처리·후처리 채택 후보는5-fold에서 재확인한다. 학습 기반 복원 전처리는 해당 fold의 train으로 학습한 복원 모델을 사용한다. 고정 분할에서 학습한 복원 모델은 우선 그 고정 validation에서 평가하며 모든 fold에 그대로 재사용하지 않는다. 전체PQ·TP·FP·FN과 희미한/작은 객체의 검출률을 함께 기록한다. 공개 참조216+102는 별도 후보 포함률 진단이며 test정답 PQ로 해석하지 않는다.

## 조사 근거

`/workspace/artifacts/external_data_research_20261104/`에 다음을 보관했다.

- `kaggle_pages.json`, `kaggle_page_Evaluation.md`, `rules_overlap.md`: 공식 규정 원문과 기간 감사.
- `magfilo_paper.html/txt`, `gong_halpha.html/txt`: 영상 생성·보정 설명 원문.
- `swefil_actual_audit.json`, SWEFil COCO JSON과 저장소 목록: 실제 날짜·중복·객체 수 감사.
- `gong_labels.md`: 외부 데이터 후보 상세 조사.
- `superresolution.md`: ITI/NeuralBD 등의 입력 조건, 공개 코드, 적용 한계와 실험 설계.
