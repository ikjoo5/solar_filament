# 기존 YOLO+best84 파이프라인에 R1 연결

## 실행 구조

`원본 test 이미지 → 기존 전처리 → YOLO 30뷰 + best84 타일/TTA → 기존 후처리 → R1 한 번 → 제출 CSV`

- 입력은 원본 test180장, 기존 YOLO·best84 가중치, 공식 test 파일명 목록이다. **제출 CSV를 입력으로 읽지 않는다.**
- 기존 `submission_driver.finalize_frame`을 그대로 사용한다. YOLO의 기하4뷰·전처리8뷰·타일18뷰와 native 전체 rich 결과를 모두 계산하며, 기존 후보 선택·마스크 보정 과정을 생략하지 않는다.
- R1은 같은 추론에서 얻은 기하 뷰와 native base/TTA 지지를 재사용한다. 기존 후처리의 최종 마스크에 한 번 적용하며 모델을 다시 학습하지 않는다.
- R1 적용 전후 객체 ID·개수·기존 픽셀을 보존하고, 마스크 간 겹침을 허용하지 않는다. 각 객체 면적 증가는 최대25%다. 기존과 이번 추론의 객체 수가 반드시1,347개로 같다고 가정하지 않는다.

## 실행

1. Colab에서 T4 GPU를 선택한다.
2. [최초 실행 셀](/workspace/artifacts/downloads/SolarFilament_Best84Pipeline_R1_T4_CopyCell.txt)을 한 셀에 복사하거나 [노트북](/workspace/artifacts/downloads/SolarFilament_Best84Pipeline_R1_T4.ipynb)의 최초 셀을 실행한다.
3. `MAX_IMAGES=None`이면 전체180장을 처리한다. `2`로 점검한 경우 None으로 바꾸어 같은 셀을 다시 실행하면 완료된 단계부터 이어간다.
4. 세션 종료 후에는 같은 최초 셀 또는 [재개 셀](/workspace/artifacts/downloads/SolarFilament_Best84Pipeline_R1_T4_ResumeCell.txt)을 실행한다. 코드·설정·이미지별 단계 캐시는 Drive에 저장한다.

기본 입력 경로:

- 이미지: `/content/drive/MyDrive/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images`
- YOLO: `/content/drive/MyDrive/SolarFilament_LimbOnly_YOLO26_mixed_FullRes/best.pt`
- best84: `/content/drive/MyDrive/SolarFilament_NativeUNet_A100/20261028T030808Z_bedf3e21/best.pt`

## 생성 파일

Drive의 `filament-segmentation-2026/submissions/best84_pipeline_r1_t4_v1`에 저장한다.

- `submission_pipeline_baseline.csv`: 같은 실행에서 얻은 R1 적용 전 결과.
- **`submission_pipeline_R1.csv`**: R1 적용 후 제출 후보.
- `change_audit.csv`: 객체별 면적·추가 픽셀·변경 여부.
- `results.zip`: 두 CSV, 변경 내역, 실행·입력·코드·환경 기록.

180장 전체가 완료되어야 제출 CSV를 생성한다. 같은 추론으로 두 CSV를 만들기 때문에 R1만 추가했을 때의 효과를 비교할 수 있다. 기존 파이프라인의 과거 public LB는0.39다. 이번 T4 재추론 파일 두 개의 LB 점수는 아직 측정하지 않았다.

## 확인 범위

- 고정 R1의 기존707장 재현 검증을 유지했다.
- 실제 검증 추론 캐시8장에서 기하4·전처리8·타일18·native full 결과를 다시 기존 finalizer에 넣어 **51개 기준 마스크의 RLE 일치**를 확인했다. 그 뒤 R1을 적용한 결과도 앞서 검증한 복원 결과와 일치했다. 이 확인에는 GT나 새 신경망 추론을 사용하지 않았다.
- CPU fixture 검사는 전체 단계 호출, CSV 입력 차단, 같은 실행의 기준선 보존, 빈 이미지 처리, 단계별 중단·재개, R1 중복 적용 방지, 입력·가중치·캐시 변경 감지, 동시 실행 차단, 실제 Colab 셀의 복원·다운로드 조건을 확인한다.
- 현재 클라우드에는 원본 test 이미지와 CUDA GPU가 없어 **실제 T4 test 추론은 Colab에서 실행해야 한다.** 여기서 새 제출 CSV나 새 리더보드 점수를 생성했다고 주장하지 않는다.

[실행 패키지 ZIP](/workspace/artifacts/downloads/SolarFilament_Best84Pipeline_R1_T4_Package.zip)
