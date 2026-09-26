# 연구 보고서 인덱스

이 디렉터리는 Project #2의 Task 번호를 기준으로 연구 결과와 재현 증거를 찾는
공식 진입점이다. 설계 계약은 `docs/design/`, 현재 프로젝트 상태는
`docs/research_progress_summary.md`, 실행 결과와 완료 증거는 이 디렉터리에 둔다.

## 읽는 순서

1. 저장소 `README.md`와 `docs/research_progress_summary.md`로 전체 상태를 확인한다.
2. Project Board의 현재 Task와 canonical Issue에서 범위와 완료 기준을 확인한다.
3. 아래 Task `README.md`에서 현재 결론, 남은 gate와 증거 목록을 읽는다.
4. 완료된 Task는 `FINAL_REPORT.md`, 세부 재현은 날짜별 `runs/`를 확인한다.
5. 실제 변경 diff와 검토 기록은 연결된 PR을 확인한다.

## Task별 보고서

| Task | 상태 | 현재 보고서 | 최종 통합본 |
|---|---|---|---|
| 02 State I/O | Done | [`tasks/02_state_io/README.md`](tasks/02_state_io/README.md) | [`FINAL_REPORT.md`](tasks/02_state_io/FINAL_REPORT.md) |
| 03 Benchmark | In Progress | [`tasks/03_benchmark/README.md`](tasks/03_benchmark/README.md) | v1.1 gate 완료 후 작성 |
| 06 Runtime | Done | [`tasks/06_runtime/README.md`](tasks/06_runtime/README.md) | [`FINAL_REPORT.md`](tasks/06_runtime/FINAL_REPORT.md) |
| 07 Paired state | Todo | Task 시작 시 생성 | Task 완료 시 작성 |
| 08 Baselines | Todo | Task 시작 시 생성 | Task 완료 시 작성 |
| 09 Nonlinear candidates | Todo | Task 시작 시 생성 | Task 완료 시 작성 |
| 13 Final evaluation | Todo | Task 시작 시 생성 | Task 완료 시 작성 |

`reports/legacy/`는 이전 모델 쌍의 역사적 결과이며 현재 Small→Base+ 성능 주장에
사용하지 않는다.

## 작성 규칙

- Task 디렉터리는 `NN_short_name` 형식을 사용한다.
- `README.md`는 현재 상태, 핵심 결과, 남은 gate와 증거 링크를 갱신하는 인덱스다.
- `runs/YYYY-MM-DD_slug/`는 실행 명령, 환경, 원본 수치, JSON, 실패와 한계를 보존하는
  불변에 가까운 실행 기록이다. 실질적인 실행이 없는 날을 위해 빈 일지를 만들지 않는다.
- 중간 동결본은 `milestones/YYYY-MM-DD_slug.md`에 보존한다.
- Task를 Done으로 옮기기 전에 `FINAL_REPORT.md`에서 목적, 결정, 전체 결과, 실패,
  한계와 완료 기준별 증거를 통합한다.
- dataset 원본, checkpoint, paired state, 대형 cache와 raw log는 RunPod Network
  Volume에 둔다. Git에는 작은 보고서·JSON·그림과 checksum·보관 경로만 남긴다.
- 날짜별 원본을 최종 통합본에 복제하지 않는다. 최종본은 손실 없이 해석하고 원본에
  링크해, 읽기 쉬움과 감사 가능성을 함께 유지한다.

## Done 전 증거 순서

1. 날짜별 실행 보고서와 산출물 저장
2. Task `FINAL_REPORT.md` 작성
3. canonical Issue의 각 완료 기준에 증거 링크 연결
4. PR test·review와 `main` 병합
5. Issue checklist 갱신과 완료 처리
6. 마지막으로 Project Status를 `Done`으로 변경
