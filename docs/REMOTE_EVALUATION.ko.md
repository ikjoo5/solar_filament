# 이미지 업로드 없이 Colab에서 기존 모델 평가

사용자가 확인한 Drive 경로는 다음과 같다.

`/content/drive/MyDrive/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/train_images`

원본 이미지는 Drive와 Windows에 그대로 둔다. **추가 이미지 ZIP을 채팅에 올릴 필요가 없다.**

1. [독립 실행 Colab 노트북](../notebooks/Evaluate_Existing_YOLO_Colab.ipynb)을 내려받아 Colab에서 연다.
2. GPU를 T4/A100으로 설정하고 위에서부터 실행해 Drive를 연결한다.
3. 경로 설정에서 정답 JSON, `best.pt`, `train_split.csv` 위치를 확인한다. 기본값은 업로드 베이스라인의 Drive 폴더와 같다. annotation ZIP도 JSON 하나가 들어 있다면 사용할 수 있다.
4. 평가를 끝낸 뒤 내려받는 **결과 ZIP만** 채팅에 첨부한다. 이미지와 모델은 결과 ZIP에 넣지 않는다.

노트북 자체에 평가 스크립트를 내장했으므로 Git clone이나 저장소 push가 필요하지 않다. 코드가 바뀌면 저장소에서 `python tools/build_colab_notebook.py`로 내장 사본을 다시 생성한다. 사용자가 만든 다른 노트북은 덮어쓰지 않는다.

실행은 기존 checkpoint의 추론과 평가이며 새 모델 학습이나 Kaggle 제출은 하지 않는다. confidence 후보별로 기본 `union_cc`와 개별 마스크를 유지하는 `preserve`를 비교한다. `preserve`에는 overlap이 남을 수 있어 진단용으로 해석하고 최종 제출 전에 규정을 확인한다.

`val_split.csv`가 기존 run 폴더에 있으면 자동 사용한다. 없다면 명시적 `USE_TRAIN_COMPLEMENT=True` 설정으로 train 목록에서 제외된 metadata 집합을 평가한다. 현재 업로드 목록 기준으로 이는 **91개 실제 영상, 101개 annotation group, 정답 708개**다. 당시 validation과의 일치나 synthetic 원천을 입증하지 않았으며 **리더보드 0.37 재현으로 부르지 않는다**. validation이 train과 같은 실제 파일이나 annotation variant를 공유하면 중단한다.

기본은 선택된 집합 전체 평가다. `MAX_IMAGES=2`로 바꾸면 빠른 smoke test가 되며 전체 validation으로 보고하지 않는다. 결과는 매번 새 Drive 하위 폴더에 저장해 기존 결과를 덮어쓰지 않는다. `summary.json`에 split 출처·실행 범위·checkpoint/입력 해시·실행 버전·TP/FP/FN·PQ가 남는다.

생성된 노트북의 코드 문법과 내장 모듈 소스는 정적으로 검사했다. 노트북에는 실행 출력이 없으며, 실제 Colab GPU와 사용자의 전체 Drive 데이터에서 실행되었다고 주장하지 않는다. 기존 Colab CUDA용 PyTorch를 유지하며 CPU wheel은 설치하지 않는다.
