# 공개 MAGFiLO의 실제 외부 평가 가능성

업로드한 `official_image_names.json`의 전체 목록을 대조한 결과, 공개 MAGFiLO에서
대회 train·test를 제외한 **외부 평가 후보 71장**이 확인됐다. 원본 71장도 실제 확보했다.
사용자 Drive의 대회원본과 파일·픽셀 중복 검사를 마치기 전이므로 확정 독립 평가 N은 미정이다.

공식 Dataverse의 `magfilo_2024_v1.0.json`은 60,602,626바이트이며
공식 MD5 `bfa414567b4dfa2a6b4a9eccedb6aa20`과 일치한다.

| 관측 ID 비교 | 물리 이미지 수 |
|---|---:|
| 공개 MAGFiLO 전체 | 958 |
| 그중 공식 대회 train/validation과 겹침 | 707 |
| 그중 전체 공식 test와 겹침 | 180 |
| 대회 train·test와 관측 ID가 다른 후보 | **71** |

train707은 기존 정답 JSON과 정확히 같으며 test180과의 교집합은 없다. 각 목록 안의 파일명·
관측 ID 중복도 없다. 기존 제출 CSV의 test177은 모두 test180에 들어 있다. 이전 미확정74 중
3장이 test로 판명됐고 71장이 남았다. 남은 이미지의 주석자별 레코드는 121개이며,
이를 독립 이미지 121장으로 세지 않는다.

후보 원본 JPEG 71장(총29,608,976바이트)을 GONG 출처에서 다운로드했다. 모두 2048×2048
uint8로 디코딩됐고 실패0·후보끼리 파일 중복0·회색조 픽셀 중복0이다. 대회 train/test 이미지를
이 과정에서 다운로드하거나 해당 정답으로 채점하지 않았다. 아직 외부 PQ는 계산하지 않았다.

시간적 관련성도 남아 있다. 실제 가중치 학습616장을 기준으로 후보16장이 촬영일3일 이내이며,
55장은3일보다 멀다. 대회 train/validation707장을 기준으로 하면22장이3일 이내이고,
test까지 포함한887장과3일보다 떨어진 후보는43장이다. 이 거리는 근접 날짜 분석이며,
3일을 넘었다는 이유만으로 완전한 통계적 독립성을 보장하지 않는다.

`notebooks/External71_Prepare_CopyCell.py`는 기존 Drive 경로에서 전체 대회원본887장의
파일 SHA와 디코딩된 픽셀 SHA를 계산하고, 후보71장을 별도 `external71_images` 폴더에
받아 비교한다. 정답은 중복 제외를 통과한 외부 관측만 별도 파일에 보관한다. 이름·픽셀 기준을
통과한 관측을 27일 시간 블록으로 development/confirmation에 나누고 경계3일은 보류한다.
GPU·학습·추론·채점 없이 준비하며, 이미지별 다운로드는 재개 가능하다. 생성된 작은 audit ZIP에는
이미지·마스크·가중치를 넣지 않는다. 클라우드에서 사용자 Drive의 최종 검사를 대신 실행하지는 않았다.

외부 데이터의 전처리·후처리 실험 허용은 그대로 반영한다. 해당 실험도 정답 점수를 반복해서
보며 설정을 선택하면 평가 자료에 적응하므로, 알려진 대회 test는 제외하고 별도 확인 부분집합은
보류한다. 외부 데이터는 학습·pseudo-label 학습·테스트 시 파라미터 갱신에 사용하지 않는다.

현재 비교 근거: `docs/evidence/official_image_manifest_audit.json`.
실제 다운로드 근거: `docs/evidence/external_magfilo_candidates_acquisition.json`.
이전 부분 test 목록으로 진행한 계산은 `docs/evidence/external_magfilo_retrieval_audit.json`에
역사적 기록으로 남긴다. 확정 외부 N·PQ는 Drive 검사와 실제 모델 추론 이후에만 보고한다.
