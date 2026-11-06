# 다양한 validation으로 옮기는 T4 학습·평가

공식 train707장을 다섯 번에 나눠 검증하는 실행 코드를 준비했다. 기본 실행은
**fold0 하나: 학습549장·검증143장·촬영일 근접 제외15장**이다. 새로운5-fold 성적은 아직 없다.
클라우드는CPU 환경이며 원본 이미지는 사용자Drive에 있으므로 실제 학습·추론은 Colab에서 수행한다.

## 처음 실행

`notebooks/SolarFilament_GroupedCV_T4_CopyCell.py`를 열어 내용을 모두 Colab 한 셀에 복사한다.
노트북 형식은 `notebooks/SolarFilament_GroupedCV_T4.ipynb`다. ColabGPU를T4로 선택한다.

기본값은 `FOLDS=[0]`, `EPOCHS=100`, `RUN_UNTIL_EPOCH=100`이다. 한 번에5개fold를
시작하지 않는다. 짧게 점검하려면 `RUN_UNTIL_EPOCH=20`으로 실행하고 나중에100으로 올린다.
같은100epoch 학습률 스케줄의 완료epoch부터 이어가며, `EPOCHS`를 바꿔서 재개하지 않는다.

ConvNeXt-Tiny U-Net 단일native모델을 범용ImageNetencoder에서 새로 시작한다. 기존
best84·context·YOLO 가중치를 사용하지 않는다. 새분할의 단일모델 기준선을 만드는 실행이며,
이PQ를 기존YOLO+native 앙상블 검증0.473599나 공개LB0.39와 직접 비교하지 않는다.

T4는FP16·batch1·누적4·1024crop·epoch당400개crop·lr1e-4다. 기존에 성공했던
depthwise CUDA 경로를 적용하고 학습 전에forward/backward를 점검한다. 환경의
torch·torchvision을 다시 설치하지 않고 필요한 모델 라이브러리만 설치한다.

## 저장과 재개

코드·분할·설정·로그·학습 상태는 다음Drive 폴더에 저장된다.

```
/content/drive/MyDrive/filament-segmentation-2026/grouped_cv_t4_v1/
  code/
  launch_config.json
  splits/fold0/
  runs/fold0/last.pt
  logs/fold0.log
```

완료epoch마다 model·optimizer·scheduler·scaler·RNG를 함께 원자적으로 저장한다.
중간epoch에서 끊기면 마지막 저장epoch부터 시작하므로 그epoch 일부는 다시 계산한다.
첫epoch 저장 전에 끊기면 새로 시작한다. 세션이 사라지면
`notebooks/GroupedCV_T4_Resume_CopyCell.py`를 실행해Drive의 소스와 설정을 복구한다.
기존 `/content` 디렉터리나 config가 없어도 된다. 같은세션의 중복학습은 잠금으로 거부하고,
셀 중단 시 자신이 띄운 학습프로세스도 종료한다.

fold0 완료후 재개셀의 `FOLDS`를 `[1]`, `[2]`, `[3]`, `[4]`로 바꿔 순서대로 실행한다.
진행 중인fold는 같은번호로 재개한다. 코드·학습 설정·split이 다르면 기존체크포인트를
덮어쓰지 않는다. CPU 테스트의 완전일치는 같은환경의 재개에 관한 결과이며,
라이브러리나GPU를 바꾼 실행의 bit단위 일치를 보장하지 않는다.

## validation을 사용하는 방식

같은 이미지의 모든 주석자를 같은fold에 두고27일 단위로 묶었다. 다른 블록이어도
검증 촬영일3일 이내인 이미지는 해당fold의 학습에서 제외한다. 모든fold의 최소학습/검증
날짜차이는4일이다. 이 날짜 차이가 완전한 통계적 독립성을 보장하지는 않는다.

기존합성462장은 새fold별 donor·배경출처를 확인할 수 없어 새실행에서 사용하지 않는다.
원래616/91분할에서 합성이 안전했던 사실과 새fold의 안전성은 별개다. 합성을 다시 쓰려면
해당fold의 학습사진으로 재생성하고 개별출처를 기록한다.

20epoch마다 진단PQ를 기록하지만 그중 최고를OOF결과로 고르지 않는다. 미리 정한100epoch의
모델만 최종OOF예측을 만든다. 중간성적에 맞춰 정책을 계속 변경하면 해당결과도 개발용
성적이 된다. 기존91장도 이미 방법선택에 쓰였으므로 완전히 미사용한test라고 부르지 않는다.

## 결과 전달과 FP·FN 비교

실행한fold의 작은결과ZIP을 올리면 된다. fold0 파일은 다음위치다.

```
.../grouped_cv_t4_v1/runs/fold0/fold0_cv_results.zip
```

ZIP에는 설정·history·fold계획·CSV·모든검증이미지의 예측기록이 들어간다. 이미지·가중치는
넣지 않는다. 중간정지ZIP은 학습진단이며 최종OOF가 아니다.100epoch체크포인트가 있고
추론이 끊겼다면 같은셀을 실행해서 학습을 반복하지 않고 이미지별 추론을 재개한다.

최종기록은0.35/0.5/0.65의 연결성분과 평균확률을 저장한다. 최소100pixel 후보를 보관하되
기준선CSV는0.5·최소면적300으로 고정한다. GPU를 다시 쓰지 않고 작은객체 보완·보수적검출·
고확률seed 기반후처리를 비교하는 고정4개 설정을 `experiment_grouped_oof_postprocess.py`에 준비했다.
모든후보CSV를 생성한 후 정답을 읽어 채점하며 설정을 자동 채택하지 않는다.

`tools/evaluate_grouped_oof.py`는 원본정답으로 재채점하며 무검출 이미지도 확인한다.
전체5fold가 없으면 `partial_folds`로 보고한다. 모두완료된 뒤707장의 IoU합·TP·FP·FN을
합산해서 전체PQ를 계산하며 foldPQ의 단순평균을 쓰지 않는다. fold별·관측소6개·
2011–14/2015–18/2019–22의 시기별 결과도 함께 보고한다.

## 실제로 확인한 범위

기존91장을 촬영일별로 다시나눈 진단에서 최근경계보정은 학습사진과3일이내72장의
PQ0.468995→0.470300을 보였지만,3일보다떨어진19장은0.488519→0.488199였다.
새관측의 검증이 아니며19장만으로 일반화악화를 확정하지도 않는다.
출처는 `docs/evidence/validation_strata_audit.json`이다.

CPU에서 실제ConvNeXt-Tiny U-Net의 forward/backward를 확인했다. 작은모델의 연속3epoch와
1epoch후재개에서 가중치·Adam상태·LR·예측이 같았고 결과를OOF평가기까지 연결하는
통합테스트도 통과했다. 실제T4·100epoch성능은 아직 측정하지 않았다.
