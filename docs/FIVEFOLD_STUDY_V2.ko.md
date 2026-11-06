# 새 실험 기준: 5-fold 전처리·후처리 / 고정 분할 재학습

확정한 기준선은 실제 제출했던 **YOLO26s-Seg + native ConvNeXt-Tiny U-Net**이다.
기존 검증PQ0.472593, 사용자 보고publicLB0.39를 기록한 추론 파이프라인을 사용한다.
추가context모델을 쓰는 개발후보0.473599는 기본값에 넣지 않는다.

| 실험 | 사용할 데이터와 가중치 |
|---|---|
| 추론 전처리·후처리 | 새5-fold의 fold별 고정 모델과 OOF예측 |
| 모델 구조·loss·학습률·학습 증강 등 재학습 | 기존train616장/valid91장 고정 분할 |

5-fold의 모든검증사진은 해당사진을 학습하지 않은 fold모델로만 예측한다. 다른4개fold의 모델은
그사진을 학습했을 수 있으므로 OOF에서5개모델을 섞어 ensemble하지 않는다.

## 학습량과 시작 가중치

각fold에서 YOLO와native를 **각각30epoch** 학습한다. 따라서5개fold 전체는
**5×2×30=300 model-epoch**다. 서로다른모델의epoch는 업데이트 수나 실행시간이 같다는 뜻은 아니다.
모든fold를 같은30epoch에서 비교하고, 한fold만100epoch까지 먼저 진행하지 않는다.

YOLO는 공개범용 `yolo26s-seg.pt`, nativeencoder는 범용ImageNet에서 시작한다.
기존YOLO solar `best.pt`, best84, context가중치와 기존fold0실험의 가중치를 가져오지 않는다.
같은native모델구조·limb보정·D4증강·작은객체crop학습·YOLO학습설정과 기존추론규칙을 옮긴다.
학습데이터와가중치·예산은 새로 만들어지므로 과거PQ를 재현한다는 보장은 없다.

학습률스케줄의 상한은100으로 고정해 나중에30→60으로 이어갈 때 스케줄을 바꾸지 않는다.
현재셀은30에서 종료한다. YOLO라이브러리 로그의100은 스케줄설정이며,30epoch저장후 중단하는
콜백을 적용했다. 표준라이브러리의 종료후optimizer제거를 막고 완료상태를 별도로 보존한다.

새fold의 합성donor·배경출처를 확인할 수 없어5-fold기준선은 원본만 학습한다. 원래분할용
합성462장을 새fold에 그대로 넣지 않는다. 기존616/91재학습용 템플릿은 원래donor616장이
validation91장과 겹치지 않는지 검사한 뒤 합성을 사용할 수 있도록 원래메타데이터SHA를 지정했다.

## 폴더

```
/content/drive/MyDrive/filament-segmentation-2026/fivefold_study_v2/
  experiment_policy.json
  fold_plan.json
  current_request.json
  code/<코드버전>/
  fold0/
    train_split.csv
    val_split.csv
    excluded_embargo.csv
    split_manifest.json
    last.pt                       # native 전체학습상태
    yolo/
      config.json
      last_full.pt                # YOLO 전체학습상태
      stage030.json
    stage030/
      model.pt                    # native 30epoch가중치
      yolo.pt                     # YOLO 30epoch가중치
      rich_records.jsonl
      yolo_records/*.json.gz       # YOLO원시예측압축캐시
      full_pipeline/
        full_selected.csv
        full_records.jsonl
  fold1/ ... fold4/
  comparisons/
  logs/
```

처음 실행할 때5개fold의train/valid/excluded목록과 원본사진명을 먼저 저장한다.
가중치는 순서대로 학습하면서 생성된다. 같은원본의 모든주석자를 같은fold에 넣고,
27일시간블록을 묶으며, validation촬영일3일이내의 다른블록은 학습에서 제외한다.
검증사진수는143/142/140/141/141장이다.4일차이가 완전한통계적독립성을 보장하지는 않는다.

## 실행과 재개

기존v1학습셀을 중단하고 ColabT4에서
`notebooks/SolarFilament_5Fold_Study_v2_CopyCell.py` 내용을 한셀에 복사해 실행한다.
노트북 파일은 `notebooks/SolarFilament_5Fold_Study_v2.ipynb`다.
기본 `STAGE_EPOCH=30`이며5개fold가 모두 순서대로 실행된다.

각fold에서 native학습·rich추론 → YOLO학습 → 전체파이프라인추론 순서로 진행한다.
모델분기는 별도Python프로세스에서 실행해YOLO가native의전역Torch설정을 바꾸지 않도록 했다.
T4는nativeFP16·depthwiseCUDA우회·batch1/누적4를 사용하며, YOLO는원래2048해상도·batch1이다.

세션이 종료되면 `notebooks/SolarFilament_5Fold_Study_v2_Resume.py`를 실행한다.
기본값None은 직전공통목표30을 불러온다. 폴드번호를 매번 선택할 필요가 없다.
완료fold는 확인후 건너뛰고, 진행중인학습은 마지막저장epoch부터 재개한다. 현재epoch의
미완료부분은 다시 계산한다. 추론은 이미지별캐시에서 이어가며 가중치·입력·코드SHA를 확인한다.

완료epoch마다 native의optimizer/scheduler/scaler/RNG를 저장한다. YOLO도 live가중치·EMA·
optimizer·scheduler·scaler·RNG를 별도로 보존해라이브러리의checkpoint정리로 재개상태가 사라지지
않도록 했다. 작은CPU모델에서 실제1→2epoch YOLO학습·재개와native상태 재개를 검증했다.
GPU·라이브러리버전이 바뀐 실행의 bit단위동일성까지 보장하는것은 아니다.

## 이번 전처리·후처리 비교

모든방법은 같은fold의 같은YOLO/native가중치를 사용한다.

1. 기존선택파이프라인:4개기하view·8개전처리view·18개tileview의YOLO결합과native4방향TTA,
   경계보정·후보추가·제거규칙을 그대로 적용한다.
2. 마지막native후보추가·제거단계를 생략한 결과를 비교한다.
3. native의base입력을limb+CLAHE로 바꾸되 나머지TTA지지는 유지한 전처리view를 비교한다.
4. YOLO전처리후보 중limb외의후보를 제거하는 ablation을 비교한다.

확률·RLE·view별원시예측을 저장하므로 후처리변경은 재학습하지 않는다. 새로운전처리입력은
기존가중치로 추가추론이 필요할 수 있다. YOLO원시캐시는gzip으로저장하고 최종결과마다 중복복사하지 않는다.
모든후보CSV를 생성한 뒤 정답을 읽어 PQ·SQ·RQ·TP·FP·FN을 계산한다. 전체5개가 같은epoch에서
완료된 뒤에만707장의 전체OOF로 표시하고, fold·관측소·시기별 결과도 함께 보고한다.

전체OOFPQ는foldPQ의 단순평균이 아니라 IoU합·TP·FP·FN을 합산해 계산한다. 전처리·후처리도
OOF점수를 반복해서보며 선택하므로 이자료는 개발검증셋이며 독립적인최종test로 설명하지 않는다.
publicLB는 별도확인한다.

## 모델·하이퍼파라미터 실험

`experiment_policy.json`에 실험용도를 저장하고, `fixed_split_model_template.json`을 생성한다.
재학습은 `fixed_split_runner.py`로 원래616/91분할과 입력SHA를 확인해 별도
`fixed_split_experiments/실험이름/`에 저장한다. 이템플릿을 자동실행하지는 않는다.

새native모델을 비교할 때 원래91장에 대한 기존YOLO예측을 재사용할 수 있다. 고정분할에서 학습한
새가중치를 기존5-fold의 validation에 넣으면 누수가 생기므로5-fold모델을 덮어쓰지 않는다.
모델개선이 충분히확인되어 새기준선으로 승격할 때만, 별도버전으로 fold모델을 다시 만든다.

## 결과와 검증범위

완료하면 `fivefold_stage030_results.zip`을 자동다운로드한다. Drive에도 같은파일이 남는다.
ZIP에는 분할·설정·학습이력·native예측·전체파이프라인CSV·비교결과를 넣고 가중치와사진은 제외한다.
YOLO원시예측압축캐시는Drive에 보존하며, 추가후처리는 그캐시로CPU에서 실행할 수 있다.

현재확인한 것은 CPU학습·재개·fold출처·예측캐시·평가기능과 기존파이프라인 구현일치다.
기존91장의원시예측을 새조합코드로 재생해625개마스크·PQ0.472593의 일치를 확인했다.
이검증에 사용한과거예측을 새fold의OOF로 쓰지 않는다. 실제T4/30epoch의 새5-fold성적은 아직 없다.
