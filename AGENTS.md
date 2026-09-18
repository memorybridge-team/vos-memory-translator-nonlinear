# CMMT v2 작업 규칙

- 작업 전에 `PROJECT_CONTEXT.md`를 끝까지 읽고, 현재 실행 계획은 `docs/experimental_plan.md`를 따른다. 사용자의 최신 요청이 우선한다.
- 답변과 문서는 한국어를 기본으로 한다. 논문·코드로 확인된 사실, 가설, 제한적인 pilot 결과를 구분한다.
- 주 연구 범위는 SAM 2.1 Tiny→Base+ nonlinear memory handoff 및 DAVIS 2017/MOSEv2/LVOS v2 평가다. 이전 Tiny→Large 결과는 과거 pilot으로 표시한다.
- 상태 복원 오차보다 후속 객체 분할·재등장, 부재 중 오류와 handoff 비용을 우선한다. 모든 방법에 같은 split·객체·switch를 적용하고 미래 GT로 입력을 고르지 않는다.
- 코드 변경은 테스트와 실행 방법·seed·checkpoint/data provenance를 남긴다. 원본 데이터, checkpoint, raw cache, SSH key는 Git에서 제외한다.
- 연구 범위·방법·실험 결과가 바뀌면 출처와 날짜를 포함해 `PROJECT_CONTEXT.md`를 갱신한다. 과거 기여와 판단 변경 이력을 지우지 않는다.
- GitHub에 게시할 때에는 작성자와 출처를 정확히 기록한다. 이관된 파일의 과거 작성자를 새 저장소의 contributor UI가 표시하지 않더라도 `MIGRATION.md`의 provenance를 유지한다.
