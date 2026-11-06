# 실제 제출 유효성 규칙: 마스크 겹침 금지

- 사용자가 실제 Kaggle 제출에서 다음 오류를 확인했다: `Invalid Submission! Submissions may not contain overlapping masks. That is, predictions of the same filament may not overlap.`
- 같은 이미지의 서로 다른 인스턴스가 **한 픽셀이라도 겹치면 안 된다**. mask IoU가0.1미만이라는 조건은 이 규칙을 보장하지 않는다. 큰 마스크 안에 작은 마스크가 포함되어도 IoU는 작을 수 있다.
- 제공된 self-evaluation 노트북은 마스크 간 겹침을 거부하지 않는다. 평가 수식의 재현과 실제 제출 유효성 검사를 구분한다. `filament_eval.evaluate`의 과거 진단 계산 방식은 유지하되 제출 파일을 만들 때 별도로 겹침0을 강제한다.
- 생성 파일의 모든 마스크를 같은 이미지별로 묶고, `sum(instance_area) == area(union(instances))`를 만족해야 배포한다. 최종CSV를 다시 읽어 이미지명·RLE 크기·중복ID·빈 마스크도 검사한다.
- `build_public_addition_submission.py`는 기존0.39 마스크를 보존하고, 공개 후보225개 중 겹치는9개를 제외한216개만 추가한다. 겹침 검사를 통과하기 전에는 다운로드용 제출 파일을 저장하지 않는다.
- 이전 `submission_own039_plus_public225_exact.csv`는 **제출 거절된 파일**이다. 수정본은 `submission_own039_plus_public216_no_overlap.csv`이다.
