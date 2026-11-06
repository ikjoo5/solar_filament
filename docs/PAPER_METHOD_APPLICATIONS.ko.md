# 논문 방법을 현재 필라멘트 모델에 적용하는 실험

목적은 논문의 점수를 비교하는 것이 아니라, 논문에서 가져온 방법이 **우리 모델의 PQ를
높이는지** 같은 평가 자료에서 확인하는 것이다. 한 번에 방법 하나를 바꾸고 FN 감소와 함께
FP·SQ 변화를 기록한다. 개발셋 PQ 0.472593과 사용자가 보고한 public LB 0.39의 기존
파이프라인을 기준선으로 보존한다. 아래 새 방법의 public LB 개선은 아직 확인되지 않았다.

## 완료: DenseCRF 경계 보정

[개선된 DeepLab V3+ 연구](https://doi.org/10.1088/1538-3873/ac6e07)와
[MORDEN 연구](https://doi.org/10.1088/1674-4527/ae983a)의 DenseCRF 후처리 아이디어를
현재 최고 파이프라인의 625개 인스턴스에 적용했다. 실제 C++ DenseCRF2D solver로
Gaussian·bilateral pairwise 항의 4개 고정 설정을 실행했다.

| 방식 | PQ | SQ | TP | FP | FN |
|---|---:|---:|---:|---:|---:|
| 현재 기준선 | **0.472593** | **0.677900** | 488 | 204 | 220 |
| DenseCRF 4개 중 최고 | 0.471758 | 0.675318 | 489 | 203 | 219 |

평가 범위는 기존 91개 물리 이미지·101개 주석 그룹·708개 정답 인스턴스다. 최고 설정에서
정답 3개를 복구하고 2개를 잃어 FN과 FP는 각각 1개 줄었지만, 경계 IoU가 낮아져 PQ는
0.000835 감소했다. 물리 이미지 단위 paired bootstrap의 조건부 95% 차이 구간은
[-0.004176, +0.002766]이다. 이미 반복 사용한 개발셋이므로 독립 검증으로 해석하지 않는다.
**이번 DenseCRF 설정은 채택하지 않았다.**

원래 모델의 연속 확률 맵이 저장돼 있지 않아 이진 마스크의 signed-distance로 고정 unary를
만들었다. 따라서 이 결과는 이 unary와 현재 픽셀 자료에서의 적용 결과이며, 연속 확률을 쓰는
모든 DenseCRF 설정이 효과가 없다는 결론은 아니다. ROI 픽셀이 없으면 기존 마스크로 되돌렸고,
면적·원래 마스크 IoU·다른 인스턴스 소유 영역을 보호했다. 4개 결과 모두 저장 후 재채점했으며
중복 마스크·인스턴스 간 중첩·추가 매칭 쌍은 없었다.

근거: `docs/evidence/paper_densecrf_experiments.json`.
실행 코드: `tools/experiment_paper_densecrf.py`.

## 준비: 기존 U-Net에 다중 규모 문맥 추가

[2024년 다중 규모 연구](https://doi.org/10.3390/app14093745)와
[확장 수용영역 연구](https://doi.org/10.1088/1538-3873/ac6e07)를 참고해 기존
ConvNeXt-Tiny U-Net bottleneck에 dilation 1/3/5 depthwise 경로와 crop 내부 global pooling을
추가했다. 512,641개 파라미터가 늘어 기존 대비 약 1.606% 증가한다. 전체 Transformer·DLKA·MORDEN
모델의 재현을 주장하지 않는다. crop pooling은 입력 crop의 문맥이며 태양 전체 문맥은 아니다.

게이트를 0에서 시작하므로 초기 출력은 원래 모델과 같다. CPU FP32·BF16에서 출력의 정확한
일치, 초기화 난수 보존, 게이트 및 분기 역전파, batch 1, 전체 상태 재개를 확인했다.
실제 best84 가중치는 Colab Drive에서 SHA를 확인한 뒤 불러온다. 이 환경에서는 해당 가중치의
새 GPU 학습을 실행하지 않았다.

비교 기준은 **학습하지 않은 원래 best84**로 고정한다. 기존 모델을 추가로 100epoch
학습하는 control 실험은 실행하지 않고, best84에 새 문맥 블록을 넣은 모델만 학습한다.
기존 control 학습 파일과 A100 실험 폴더는 삭제하거나 덮어쓰지 않는다.

기본 실험 환경은 사용자 요청에 따라 T4로 변경했다. 원본 해상도와 crop 1024를 유지하며
batch 1, gradient accumulation 4, epoch당 400 batch, inference batch 1을 사용한다.
기존 batch2×200과 같이 epoch당 400개 crop, optimizer update 100회, effective batch 4다.
T4에서는 FP16을 사용한다. learning rate 1e-5, 합성 표본 최대 20%, 5epoch마다 고정
threshold 0.5·최소 면적 300으로 검증한다. 최대 100epoch cosine 스케줄은 유지하되
**우선 20epoch에서 체크포인트를 저장하고 종료**한다. `RUN_UNTIL_EPOCH`을 40, 60, 100으로
올리면 optimizer·scheduler·RNG와 함께 이어가며 스케줄을 다시 시작하지 않는다.

T4 전용 결과는
`/content/drive/MyDrive/filament-segmentation-2026/paper_method_experiments/context_only_t4_v1`
에 저장한다. 이전 A100/BF16·batch2 실험은 정밀도와 배치가 다르므로 여기에 자동으로
합치지 않는다. A100 최종 학습은 선택한 방법과 설정으로 별도 실험에서 진행할 수 있다.

원래 best84 단독 모델의 저장된 PQ 0.429931을 기록하고, 현재 GPU/정밀도에서 한 번 더
평가하여 같은 FP16 조건의 새 모델과 비교한다. 전체 앙상블 PQ 0.472593과는 분리한다.
이 비교에는 블록 추가와 추가 학습의 효과가 함께 포함되므로, 블록만의 인과 효과라고
주장하지 않는다. best 체크포인트와 4방향 flip 추론을 내보내 전체 고정 후처리에서도
이후 비교할 수 있도록 한다. 외부 데이터는 계속 평가에만 사용한다.

기존 학습 소스 3개는 byte 단위로 유지하고, 새 실행 모듈 `tools/paper_context_only.py`에서
실제로 새 모델만 선택한다. 기존 `methods` 메타데이터는 재개 식별값 보존용이며,
`execution_methods=['multiscale_context']`가 실제 실행 대상이다.

- 전체 Colab 셀: `notebooks/SolarFilament_PaperMethods_CopyCell.py`.
- T4 노트북: `notebooks/SolarFilament_ContextOnly_T4.ipynb`.
- 기존 셀에서 전환하는 복사 셀: `notebooks/SolarFilament_ContextOnly_SwitchCell.py`.

전환 셀은 기존 Colab 파일 또는 Drive에 저장한 학습 소스를 SHA 검증 후 복원하므로
세션이 바뀐 뒤에도 재사용할 수 있다. 새 성능과 T4 실제 메모리 사용은 이 CPU 환경에서
측정하지 않았다.
새 실행·복원·pilot 재개·출력 관련 테스트를 포함해 총 26개가 통과했다.
검증 범위: `docs/evidence/paper_multiscale_readiness.json`.
**GPU 학습 후 PQ 개선과 외부 자료에서의 개선은 아직 측정하지 않았다.**

## 외부 MAGFiLO 보관과 사용 범위

사용자가 정한 폴더는 다음과 같다. 기존 대회·합성 데이터와 같은 상위 폴더를 사용한다.

```text
/content/drive/MyDrive/filament-segmentation-2026/
├── MAGFiLO_1.0_Kaggle_2026/
├── synthetic_copy_paste_v2/
└── external_MAGFiLO/
    ├── annotations.json
    ├── images/
    ├── manifests/
    └── evaluations/
```

경로 설정과 Colab 생성 셀을 준비했다. 클라우드에서 사용자 Drive에 실제 폴더를 만들거나
원본 이미지를 내려받지는 않았다. 현재 공개 Dataverse API는 HTTP403을 반환했으므로 실제
데이터 확보, 중복 제외 후 추가 N, 외부 PQ는 미확정이다.

외부 MAGFiLO는 평가에만 쓴다. 학습·pseudo-label 학습·미세조정·테스트 시 파라미터 갱신에
넣지 않는다. 대회 train/val/test와 관측 ID·파일 SHA·디코딩 픽셀 SHA가 겹치는 자료를 제외한 뒤
27일 촬영 블록별 external_dev와 보류 external_confirm으로 나눈다. 3일 경계 embargo를 적용하고,
기존 91장과 외부 점수를 섞지 않는다. 최종 설정을 고정하기 전에는 confirmation을 반복 평가하지 않는다.

외부 개발셋에서 기준선과 같은 방법으로 평가하고, PQ·FP/FN·관측소별 차이·시간적으로 떨어진
부분집합·블록 단위 불확실성을 확인한다. 같은 외부 개발셋을 반복 보고 설정을 고르면 그 역시
과적합될 수 있으므로 외부 자료가 늘었다는 이유만으로 독립성이 보장된다고 보지 않는다.

보관 설정: `docs/evidence/external_layout.json`.
평가 준비 상태: `docs/evidence/round8_external_evaluation_readiness.json`.
