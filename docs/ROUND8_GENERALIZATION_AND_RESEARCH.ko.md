# Round 8: 외부 평가와 일반화 점검

사용자가 보고한 public LB는 0.37에서 **0.39로 0.02 상승**했다. 현재 고정 개발셋 PQ는
0.472593이다. 차이는 0.06175에서 0.08259로 커졌다. 이 차이만으로 과적합을 확정하지 않는다.
표본 구성·관측소·촬영 시기 차이와 반복된 설정 선택이 모두 영향을 줄 수 있다.
사용자가 알려준 public 1위 약 0.7은 독립 확인하지 않았다.

현재 파이프라인과 두 체크포인트는 그대로 보존했다. 이후 논문의 DenseCRF 방법을 현재
파이프라인에 실제 적용한 4개 설정에서 최고 PQ는 0.471758로 낮아져 채택하지 않았다.
세부 결과와 다중 규모 문맥 블록의 학습 비교 설계는 `docs/PAPER_METHOD_APPLICATIONS.ko.md`에 있다.
새 GPU 학습 또는 외부 평가에서 개선을 확인한 결과는 아직 없다.

## 실제 수행한 재분석

- 기존 train 616장과 val 91장의 파일명·annotation group 중복은 없다. 다만 val 13장은 train과
  같은 날짜, 38장은 1일 이내, **72장은 3일 이내**에 촬영됐다. 이는 시간 의존성 위험의 진단이며
  동일 픽셀 중복 또는 잘못된 분할이 확인됐다는 뜻은 아니다. 기존 분할을 변경하지 않았다.
- 물리 이미지 단위 paired bootstrap 10,000회에서 현재 고정 파이프라인과 최초 YOLO의 개발셋
  PQ 차이는 +0.04084, 조건부 95% 구간은 [+0.02325,+0.05903]이다. 이미 이 개발셋으로 설정을
  반복 선택했으므로 이 구간은 선택 편향을 포함하지 않으며 독립적인 일반화 보장이 아니다.
- 전체로는 개선됐지만 Uh 관측소의 PQ 차이는 -0.01804였다. 다른 5개 관측소의 차이는
  +0.03475~+0.07936이었다. 표본별 불확실성이 있으므로 관측소별 게이트를 새로 만들지 않는다.
- 현재 FN 220개 중 기존 모든 raw 후보를 사후 검색하면 178개에서 IoU>0.5 후보가 있다.
  26개는 최대 IoU 0.3~0.5, 16개는 0.3 미만이다. **178개를 실제 복구했다는 뜻이 아니다.**
  후보의 정확한 선별·품질 예측을 개선할 근거이며 정답을 이용한 선택기를 만들지 않는다.
- train에서 계산한 면적 사분위 경계는 670 / 1234 / 2442.5픽셀이다. 최하위 크기군의 val recall은
  57.8%, 최상위는 74.4%다. 작은 필라멘트가 주요 약점이다.
- train의 다중 주석 이미지 287장에 대한 587개 주석쌍을 점검했다. foreground union IoU 중앙값은
  약 0.4702다. 서로 다른 주석을 단순 합집합으로 붙이거나 특정 주석을 잘못됐다고 단정하지 않는다.
  이미지 우선 표본추출과 annotation set 별 감독을 유지하고, 향후 공식 train만으로 품질 학습을 검토한다.

근거: `docs/evidence/round8_generalization_audit.json`,
재현: `tools/analyze_round8_generalization.py`.

## 확인한 논문과 적용 범위

| 논문 | 확인한 내용 | 실험에 가져올 부분 |
|---|---|---|
| Ahmadzadeh et al., *A dataset of manually annotated filaments from H-alpha observations*, Scientific Data (2024), [DOI](https://doi.org/10.1038/s41597-024-03876-y) | 공개 원문·공식 페이지. MAGFiLO v1.0은 958개 고유 관측, 1593개 주석 관측, 10244개 필라멘트. 관측당 최대 3명 독립 주석 | N은 주석 수 대신 고유 관측·촬영 기간 블록을 함께 보고한다. 주석 변이를 임의로 정답 하나로 병합하지 않는다 |
| Shang et al., *Solar Filament Detection Based on an Improved Deep Learning Model*, Applied Sciences (2024), [DOI](https://doi.org/10.3390/app14093745) | 원문 PDF 확인. CNN–Transformer, 다중 규모 residual block, deformable large-kernel attention. 512 입력의 150장 데이터를 90/30/30으로 분할하고 train에 밝기·대비·방향 증강 | 작은 구조의 고해상도 경로와 큰 문맥을 함께 사용하고, 관측소/노출 차이에 견디는 train 증강을 고정 비교한다 |
| Diercke et al., *A universal method for solar filament detection from Hα observations using semi-supervised deep learning*, A&A (2024), [DOI](https://doi.org/10.1051/0004-6361/202348314) | 공개 DOI 초록 확인. YOLOv5 검출 후 U-Net 분할, ChroTel/GONG/KSO 간 적용 | 검출·분할의 역할 분리와 관측소별 외부 평가를 참고한다. 외부 MAGFiLO pseudo-label 학습은 채택하지 않는다 |
| You & Shang, *Solar Filament Detection Based on Improved DeepLab V3+*, PASP (2022), [DOI](https://doi.org/10.1088/1538-3873/ac6e07) | 공개 DOI 초록 확인. 저수준 특징과 확장된 dilation으로 작은 필라멘트를 보완, DenseCRF 경계 보정 | 작은 구조 경로·넓은 수용영역을 학습 비교군으로, CRF는 별도 평가할 후처리 가설로 둔다 |
| Hu et al., *A Modern ConvNet for Solar Filament Detection*, RAA (2026), [DOI](https://doi.org/10.1088/1674-4527/ae983a) | 공개 DOI 초록 확인. MORDEN 다중 규모 특징과 DenseCRF/DBSCAN 후처리 | 다중 규모 표현과 경계 정리의 별도 ablation을 설계한다. 논문 전체 구현을 재현했다고 주장하지 않는다 |

위 논문의 pixel accuracy·F1·IoU는 대회의 instance PQ와 직접 비교하지 않는다. 논문의 외부
태양 데이터셋이나 태양 데이터로 학습된 가중치를 새 훈련 데이터의 대체물로 사용하지 않는다.
철회 표기가 있는 2025 논문과 검증되지 않은 검색 결과는 채택 근거에서 제외했다.
일부 원문 서버는 접근을 거절했으므로 확인하지 못한 원문 세부사항을 추정하지 않는다.

읽은 자료와 DOI 응답은 `/workspace/artifacts/round8_research`에 보관했다.

## 외부 MAGFiLO: 평가 전용

사용자 제약을 고정했다. **외부 MAGFiLO의 이미지·주석·예측은 학습, pseudo-label 학습,
미세조정 또는 테스트 시 파라미터 갱신에 사용하지 않는다.** 현재 가중치는 eval/inference로만 사용한다.

공식 출처는 [MAGFiLO 소개](https://www.mlecofi.net/magfilo),
[공개 데이터 DOI](https://doi.org/10.7910/DVN/J6JNVK)다. 공개 전체 958장이 그대로 추가 검증 N이
되는 것은 아니다. 실제 대회 train/val/test와의 중복을 제외한 뒤 추가 N을 산정한다.
공식 데이터 원본의 개별 파일 목록은 현재 실행 환경에서 HTTP403으로 받지 못했다.
사용자가 정한 보관 위치는 기존 대회·합성 데이터와 같은 Drive 상위 폴더 아래의
`/content/drive/MyDrive/filament-segmentation-2026/external_MAGFiLO`다. 경로는 확정됐으며,
주석·이미지의 실제 다운로드와 중복 제외 후 추가 N 산정은 아직 수행하지 못했다.

구현한 평가 절차:

1. 공식 train_images 전체(train·val 포함)와 test_images 전체의 관측 ID·파일 SHA·디코딩된
   grayscale 픽셀 SHA를 만든다. 외부 자료의 동일 관측·이름이 바뀐 동일 픽셀·서로 다른 버전 충돌을 제거한다.
2. 외부 자료의 같은 물리 관측에 붙은 여러 annotator는 하나의 N으로 묶는다. 각 주석에 대한 PQ는
   대회의 all-pairs IoU>0.5 방식으로 계산하되 신뢰구간에서는 함께 묶는다.
3. 정답 마스크를 보지 않고 27일 촬영 블록으로 development 약 70%, confirmation 약 30%를
   배정한다. 두 부분 경계에 3일 embargo를 적용한다. 같은 태양의 여러 관측소 이미지는 같은
   촬영 블록을 공유한다. 27일 블록 수가 정확한 통계적 유효 표본 수라고 가정하지 않는다.
4. 기본 실행은 `external_dev`다. confirmation 정답은 이 실행의 점수·GT 내보내기에 포함하지 않는다.
   최종 설정을 고정한 뒤 `external_confirm`을 명시적으로 평가한다.
5. 공식 관측일과 3일 넘게 떨어진 부분집합의 PQ, 관측소별 PQ, blank GT까지 포함한 보조 PQ,
   27일 블록 bootstrap을 함께 출력한다. 반복 튜닝한 91장의 PQ와 한 점수로 합치지 않는다.
6. 현재 고정 파이프라인의 예측을 전부 끝낸 뒤 선택된 외부 GT를 rasterize/score한다.
   결과 ZIP에는 해당 split의 예측·GT·지표만 담는다. 원본 이미지는 재인코딩하지 않는다.

기존 모델이 이미 학습한 이미지나 실제 대회 test에 해당하는 주석으로 성능 근거를 만들지 않는다.
중복 제외 후 남는 샘플이 없으면 점수를 계산하지 않고 그 사실을 보고한다.

## 준비된 코드와 다음 실행

- 메타데이터·중복·분할·평가: `tools/external_evaluation_data.py`
- 모델을 켜지 않는 표본 자격 검사: `tools/prepare_external_evaluation.py`
- 고정 모델 평가/재개/결과 ZIP: `tools/external_eval_runner.py`
- 확정 Drive 경로 설정: `notebooks/ExternalMAGFiLO_Evaluation_Config.json`
- Colab 폴더 생성 셀: `tools/colab_setup_external_magfilo.py`

외부 설정의 `external_annotations`는 `external_MAGFiLO/annotations.json`, `external_images`는
`external_MAGFiLO/images`로 채웠다. 생성 셀은 `images`, `manifests`, `evaluations`와 평가 전용
역할 파일을 만들며 기존 파일을 덮어쓰지 않는다. 공개 API에서 주석 JSON이 하나로 확인될 때만
다운로드하고, HTTP403이면 폴더와 상태 기록을 유지한다. 대용량 이미지를 무조건 내려받지 않는다.
실제 Drive 폴더 생성은 Colab에서 셀을 실행할 때 이루어진다.

외부 자료가 확보되면 먼저 독립적인 추가 N을 보고하고, 고정 LB0.39 파이프라인의 external_dev
기준선을 실행한다. 논문 간 성능 비교 대신 논문에서 가져온 방법 하나를 우리 모델에 적용하고
동일한 평가 자료에서 기존 방식과 비교한다. 모든 새 학습은 공식 train과 기존 누수 방지된 합성
학습 데이터 범위 안에서만 진행한다.

현재 외부 데이터 실제 점수·새 학습 성능은 아직 없다. 준비 코드의 중복 제거·보류 split 제외·
동일 픽셀 보존·평가 지표·기존 추론 회귀 테스트 41개를 통과했다.
