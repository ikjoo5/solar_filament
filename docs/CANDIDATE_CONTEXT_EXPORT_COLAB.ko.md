# Colab CPU에서 문맥 특징 수집

기존 모델을 다시 학습하지 않고, Drive의 원사진·후보에서 다음 선택기 실험에 필요한 특징을 모은다. 전체 사진을 업로드할 필요는 없다. 결과에는 원시 후보 마스크와 별도 정답 target이 들어가지만, 사진과 가중치는 포함하지 않는다.

1. `SolarFilament_CandidateContextFeatures.ipynb`를 Colab에서 열고 **CPU 런타임**으로 실행한다. 또는 `SolarFilament_CandidateContextFeatures_CopyCell.py` 전체를 한 셀에 복사해 실행한다.
2. Drive를 연결하고 첫 실행 셀을 실행한다. `MAX_IMAGES=None`은 707장 전체다. 먼저 확인하려면 1로 실행한 뒤 None으로 바꿔 재실행할 수 있다.
3. 마지막에 `status: complete`, 사진 707장, train 후보 53,535개 / validation 후보 7,172개인지 확인한다.
4. 출력된 `candidate_context_..._partNN.zip`을 **모든 part** 전달한다. 파일당 28 MiB 이하이며, 브라우저 다운로드가 실패해도 Drive에 남는다. `paused`는 일부 사진만 완료된 상태다.

기본 경로:

```text
원사진:
/content/drive/MyDrive/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/train_images

기존 원시 후보:
/content/drive/MyDrive/filament-segmentation-2026/fixed_split_experiments/mask_quality_v1

코드·캐시·결과 저장:
/content/drive/MyDrive/filament-segmentation-2026/fixed_split_experiments/mask_quality_v1/candidate_context_features/v1
```

세션이 종료되면 최초 셀을 다시 실행해도 된다. `SolarFilament_CandidateContextFeatures_ResumeCell.py`의 짧은 셀도 사용할 수 있다. Drive에 보관한 코드와 설정을 읽고, 완료된 사진은 해시·후보 순서·특징 스키마를 검사해 재사용한다. 진행 중 끊긴 사진은 다시 처리한다.

Python·NumPy·OpenCV·pycocotools 버전이 이전 실행과 달라지면 같은 결과 폴더의 캐시를 섞지 않고 중단한다. 이 경우 오류와 `request.json`을 확인해 이전 환경을 복원하거나 새 결과 폴더에서 수집해야 한다. 단순 세션 종료와 데이터·코드·실행 환경 변경을 구분한다.

이번에 검증한 것은 작은 입력의 기능·재개·분할·무결성 검사와 실제 corpus metadata다. 사용자의 Drive 전체에서 특징을 추출하거나 새 모델을 학습한 결과는 아직 없다. 이 수집은 GPU가 필요 없지만, 실제 전체 실행 시간은 Drive 읽기 속도와 CPU에 따라 달라진다.
