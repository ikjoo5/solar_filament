# 기존 synthetic_copy_paste_v2 데이터의 확인 범위

업로드된 `SolarFilament_LimbOnly_YOLO26_Train.ipynb`를 JSON 소스로 읽었으며, 이후 사용자가 제공한 실제 `synthetic_annotations.json`과 `split_info.json`도 직접 검사했다. 첨부 파일은 데이터로 읽었으며 노트북을 통째로 실행하지 않았다. **합성 이미지 픽셀 자체는 현재 cloud에 없으므로 실제 파일 존재·영상 품질을 검사했다고 주장하지 않는다.** 정량 근거는 `docs/evidence/synthetic_metadata_audit.json`에 저장했다.

**사용자는 기존 train/valid 및 synthetic의 누수 방지 구성을 이미 완료했고, 이전 YOLO에서도 합성 혼합으로 점수가 상승했다고 명시했다. 이 세션의 사용자 설명을 적용해 새 코드의 기본값은 기존 synthetic 사용·crop 샘플20% 혼합으로 둔다. 추가 provenance 승인이나 배경 재확인을 학습 시작 조건으로 요구하지 않는다.** 자동 metadata/split 검사는 파일 선택 오류를 잡는 실행 검증이며, 사용자의 설명을 다시 승인받는 절차가 아니다.

## 제공 노트북으로 확인한 경로와 형식

노트북 cell6의 경로는 다음과 같다.

```text
/content/drive/MyDrive/filament-segmentation-2026/synthetic_copy_paste_v2/
  images/
  synthetic_annotations.json
  split_info.json
```

`synthetic_annotations.json`은 COCO 형태의 `images`/`annotations` 배열이다. 실제 파일은 **이미지462장·인스턴스3,427개**이며, 모든 이미지 metadata가2048×2048이다. `images[].id`는 정수1..462, 파일명은 `synthetic_00000.png`부터 `synthetic_00461.png`까지 연속적이다. 기존 loader(cell16)는 이미지를 `images/`에서 읽으며 segmentation을 native2048로 복원한다.

annotation 필드는 `id,image_id,category_id,segmentation,area,bbox,iscrowd`이고 **spine은3,427개 모두 없다**. donor/background/source 관련 필드도 없다. Left1,091·Right1,039·Unidentifiable1,297개이며 Ambiguous는0개다. 새 foreground 학습에서는 모두 filament로 취급한다.

image ID1..10(`synthetic_00000.png`..`synthetic_00009.png`)은 annotation이 없는10장이고, 나머지452장은1..24개의 인스턴스를 가지며 중앙값은7개다.

`split_info.json`에는 원본 annotation image ID 목록인 train924개/val230개가 있으며, 두 목록은 겹치지 않고 합집합이 원본1,154개 ID와 정확히 같다. seed42, val_ratio0.2다. 이 원시 ID 분할의 물리 파일은 train616개/val206개이고, annotator 중복 때문에115개 파일명이 양쪽에 있었다.

기존 loader(cell10)처럼 source train ID가 가리키는 물리 파일명을 train에 고정해 같은 파일의 다른 annotator records도 승격하면129개 ID가 추가되어 **현재 train1,053그룹/616장, val101그룹/91장과 정확히 일치**한다. source train donor 파일이 현재 validation91장에 포함되는 경우는0개다. 따라서 합성을 켜기 위해 validation을 다시 나누거나 변경할 필요가 없다.

이는 실제 metadata로 검증한 **원본 source pool의 분할 검사**다. 합성 이미지별 실제 donor IDs, 배경 원본 ID, 생성 이력은 두 JSON에 없다. 이 부분의 분리 근거는 현재 사용자의 누수 방지 완료 설명으로 기록하며, cloud에서 생성 이력을 독립 검증했다고 주장하지 않는다.

## 실제 polygon·면적 검사와 rasterizer 선택

3,427개 annotation은 모두 하나의 유효한 polygon을 갖고 있으며 좌표는148..1901 범위다. polygon은 유한한 짝수 개 xy좌표·3개 이상의 점을 만족하고, native raster가 비어 있는 경우나 orphan image ID는0개다. `iscrowd`는 모두0이다. OpenCV 복원에서 각 인스턴스는 모두 연결되어 있다. 저장 면적은145..33,689픽셀, 중앙값1,393픽셀이다.

**합성 polygon은 기존 loader와 같은 OpenCV `np.rint`→`fillPoly`로 복원하는 것이 적절하다.** 저장 면적과 비교한 결과는 다음과 같다.

| 복원 방식 | 저장 면적과 정확히 일치 | 평균 상대 면적 차이 | bbox 정확히 일치 |
|---|---:|---:|---:|
| 기존 OpenCV fillPoly | 2,556/3,427 | +0.0456% | 3,426/3,427 |
| COCO frPyObjects | 0/3,427 | −10.10% | 0/3,427 |

OpenCV 면적 오차는2,975개가1픽셀 이내다. COCO raster는 polygon 경계 처리 차이로 중앙9.72%, 최대23.65% 면적이 작아져 가는 구조의 synthetic 라벨을 의도와 다르게 줄일 수 있다. 실제 이미지 픽셀은 보지 않았지만, 저장 면적·bbox 및 기존 loader 동작과의 일치가 OpenCV 방식을 뒷받침한다. **실데이터 정답은 기존 공식 pycocotools 방식 그대로 사용하고, synthetic만 legacy fillPoly를 적용한다.**

## 새 학습에서 고정할 metadata SHA256

```text
split_info.json (31,303 bytes)
e8c9f3848c7b1fe1283167d973e459a5df7cd6336cc1d871268a34024d1ea881

synthetic_annotations.json (12,514,857 bytes)
5e7ad5f20fa12f0c820a82a315ba60e802480d54b4b3ecebd9fba25ae8fc6aea
```

## 배경과 라벨에 관한 기존 기록

노트북 cell15와27은 `clean_solar.jpg`에 실제 필라멘트가 남으면 미라벨 전경이 잘못된 음성으로 학습될 수 있다고 기록한다. 특히 빈 annotation인 synthetic 이미지10장을 검토하라고 적혀 있다. 사용자가 올린 Drive 화면에도 `clean_solar.jpg`가 있지만, **파일명이나 화면만으로 필라멘트가 없는지 또는 검증 영상과 독립적인지는 증명할 수 없다**.

`tools/inspect_synthetic_data.py`는 해당 Colab 경로의 metadata·파일 목록·원본 split과의 관계를 읽기 전용으로 보고한다. provenance 필드가 없을 때 임의의 생성 이력을 추정하지 않으며, 사용자 설명과 독립 파일 검증을 구별해 기록한다. 이 보고서의 독립 검증 여부가 합성 데이터 사용을 차단하는 조건은 아니다.

## 권장 실험

1. 현재 고정 616개 real train/91장 real validation을 바꾸지 않는다. synthetic은 train에만 추가한다.
2. 기존 synthetic을 기본으로 사용하며 **crop 샘플 확률20%**, real80%로 제한한다. 합성 파일 개수가 많다고 학습 대부분을 차지하게 하지 않는다.
3. 초기 real-only warmup 뒤20% 혼합, 마지막 real-only 보정의 스케줄을 사용한다. original-only는 같은 조건의 선택적 대조군으로 제공하고, 같은 공식 PQ/FN/FP로 효과를 확인한다.
4. synthetic spine이 없으면 해당 sample의 auxiliary spine loss를 제외한다. `spine=0`을 정답으로 넣으면 실제 합성 필라멘트의 중심선을 잘못된 음성으로 학습한다. foreground segmentation loss는 정상 적용한다.
5. on-the-fly copy-paste는 기존 synthetic과 별도 실험 옵션이다. 필요할 때 현재 616개 train의 donor와 label이 있는 train 배경만 사용하고 배경의 기존 정답도 보존한다. 현재 사용자 설정에서는 기존 synthetic을 그대로 사용하므로 이를 필수 선행 작업으로 만들지 않는다.

현재 단계의 권장은 **기존 synthetic20% 혼합을 기본으로 준비하고, 실데이터만 사용하는 대조군도 선택할 수 있게 하는 것**이다. 이전 YOLO의 합성 효과가 새 U-Net에서도 유지되는지는 같은 91장 검증 PQ로 측정한다.

## A100/Drive 저장

화면에는 Drive 용량 부족 경고가 보인다. processed PNG/cache는 Colab `/content`에 두고, Drive에는 현재 best와 완료 epoch 재개용 checkpoint, 작은 CSV/JSON/결과 ZIP을 제한된 수로 저장하는 것이 적절하다. 합성 이미지 전체를 새 Drive run마다 다시 복사하지 않는다.
