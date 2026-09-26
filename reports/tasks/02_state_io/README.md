# Task 02 — Small/Base+ State I/O

> 상태: **Done**
> Canonical Issue: [#2](https://github.com/memorybridge-team/vos-memory-translator-nonlinear/issues/2)
> 관련 PR: [#3](https://github.com/memorybridge-team/vos-memory-translator-nonlinear/pull/3), [#8](https://github.com/memorybridge-team/vos-memory-translator-nonlinear/pull/8)

## 한눈에 보기

Task 02는 SAM 2.1 Small과 Base+의 temporal-memory 경계를 확인하고 translator가
번역할 값, 정확히 복사할 metadata, Target이 생성할 값과 전달하지 않을 값을 동결했다.

- 번역: `maskmem_features`, `obj_ptr`
- 정확히 복사·조립: frame, object, slot, conditioning, validity, switch metadata
- Target 생성: `maskmem_pos_enc`
- 비전송: 과거 mask·score·prompt dictionary, 영상 크기·fingerprint

자세한 결과와 완료 기준별 증거는 [`FINAL_REPORT.md`](FINAL_REPORT.md)에 있다.

## 기준 문서

- [Small/Base+ State I/O 계약](../../../docs/design/small_base_state_io_contract.md)
- [State Assembly Map](../../../docs/architecture/cmmt-state-assembly-map.html)

## 날짜별 증거

- [2026-09-20 Small/Base+ runtime inventory](runs/2026-09-20_runtime_inventory/README.md)
- [2026-09-21 minimal-history contract](runs/2026-09-21_minimal_history_contract/report.md)

Task 02는 계약과 정적·runtime inventory를 담당한다. 실제 fresh Target continuation,
correction과 repeated switch closure는 [Task 06](../06_runtime/README.md), 학습용 paired
state 대량 수집은 Task 07의 책임이다.
