# 최고 검증 PQ 0.472593 파이프라인 제출 CSV

`notebooks/SolarFilament_Submission_CopyCell.py` 전체를 Colab A100 새 셀에 복사해 실행한다.
노트북 형태는 `notebooks/SolarFilament_Submission_Frozen.ipynb`다. 두 파일은 같은 실행 셀을 담는다.
필요한 추론·후처리 코드를 셀 안에 압축해 포함하므로 이전 노트북의 변수나 클라우드 저장소에
추가로 스크립트를 업로드할 필요가 없다. 별도 Python 프로세스에서 라이브러리를 불러온다.

## 입력과 출력

기본 테스트 이미지 폴더:

`/content/drive/MyDrive/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images`

기존 YOLO 체크포인트와 train/val 파일명 분할:

`/content/drive/MyDrive/SolarFilament_LimbOnly_YOLO26_mixed_FullRes`

새 모델의 최고 84에폭 체크포인트:

`/content/drive/MyDrive/SolarFilament_NativeUNet_A100/20261028T030808Z_bedf3e21/best.pt`

최종 CSV:

`/content/drive/MyDrive/filament-segmentation-2026/submissions/frozen_pq0472593_v1/submission_pq0472593.csv`

전체 실제 테스트 이미지가 끝나면 이 CSV 하나를 원자적으로 저장하고 자동 다운로드한다.
CSV 열은 `filament_id,segmentation_rle`이며 ID는 원본 stem + `_1`, `_2`부터 시작한다.
원본 2048×2048 compressed COCO RLE를 쓴다. 검출이 없는 이미지는 완료로 기록하되 가짜 행을 만들지 않는다.
원본 제출 파일의 177개 예측 이미지 수를 테스트 총수로 강제하지 않는다.

추가 학습은 없다. GT 마스크는 읽지 않는다. train/val CSV에서는 파일명만 읽어 테스트 폴더에
기존 학습·검증 이미지가 잘못 섞이지 않았는지 검사한다. 두 모델 가중치와 고정 분할의 SHA를 확인한다.

## 고정 파이프라인

- 기존 YOLO: geometric 4개, preprocessing 8개, native tile 18개 추론.
- 기존 최고 합의·전처리 복구·Gaussian·tile 경계 보정 및 새 모델의 확인된 후보 추가.
- 새 ConvNeXt-T U-Net: base/hflip/vflip/hvflip, native 1024 tile·Hann blending·BF16 확률 평균.
- 최종 base/TTA 2-of-3 경계 보정, 평균 확률·view 일치도로 확인된 새 후보 추가,
  최고 PQ 설정의 낮은 지원도 마스크 제거.

모든 수치·게이트는 `docs/evidence/rich_selected_recipe.json`의 최고 PQ 후보로 고정한다.
이 중간 제출 생성에는 새로운 전후처리 탐색이나 재학습을 섞지 않는다.

YOLO 생성자는 cuBLAS 환경 변수를 변경하므로 새 모델은 독립적인 지속 Python 프로세스에서
추론한다. YOLO 생성 전의 환경을 사용하며, 원본 grayscale 배열을 로컬 NPY로 값 변화 없이 전달한다.
새 모델을 이미지마다 다시 불러오지 않는다. 학습용 DataLoader나 로컬 네트워크 소켓은 쓰지 않는다.

## 중단과 재개

같은 A100·설정·셀을 다시 실행한다. Drive에 저장한 이미지와 모델·코드·런타임 해시를 확인해
완료 이미지는 건너뛰고, geometric/preprocessing/tiled/native 중 이미 끝난 단계도 재사용한다.
중단 당시 완료하지 못한 단계부터 다시 처리한다. 세션이 끝나 `/content` 파일이 사라져도 셀이 복구한다.

완료한 이미지의 최종 RLE는 Drive에 계속 보관한다. 해당 이미지의 중간 원시 캐시만 삭제해 저장 공간을
절약한다. 체크포인트·원본 데이터·다른 실행 폴더는 삭제하지 않는다. 손상된 gzip·JSON·기본 스키마는
재계산하고, 손상된 진행 manifest는 개별 검증 캐시에서 복구한다.

모델·입력 이미지·코드·정밀도·라이브러리/GPU 조건이 달라지면 기존 작업에 섞지 않고 중단한다.
환경을 되돌리거나 셀 위쪽 `OUTPUT`을 새 폴더로 바꿔 별도 실행한다.

## 검증 범위

- 추론 전처리·native tile 좌표·반전 역변환·실제 확률 통계는 기존 실행 코드와 CPU fixture에서 일치한다.
- 실제 기존 YOLO CPU forward도 공개 참고 이미지에서 실행해 원본 좌표 RLE를 확인했다.
- 검증 91장 전체의 캐시를 새 제출용 코드로 재실행해 모든 단계가 기존 마스크와 정확히 일치했다.
  최종 CSV 625행 전체가 최고 PQ 후보 CSV와 바이트 단위로 같으며 PQ 0.4725930445를 재현했다.
- 재개·중간 실패·캐시 손상·빈 이미지·CSV 원자적 저장·압축 셀 무결성을 테스트했다.
- 실제 테스트 이미지와 새 모델 가중치는 Drive에 있으므로 테스트셋 GPU 추론과 CSV 생성은 Colab에서 실행한다.
  검증 점수는 리더보드 점수가 아니다.

구현은 `tools/submission_*.py`, 패키지 생성은 `tools/build_submission_colab.py`,
캐시 재현 검증은 `tools/verify_submission_pipeline.py`다.
근거는 `docs/evidence/submission_*.json`에 보관한다.
