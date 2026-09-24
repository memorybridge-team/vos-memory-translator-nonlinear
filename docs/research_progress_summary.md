# CMMT 현재 연구 진행 요약

> 팀원이 현재 상태를 처음 확인할 때 읽는 canonical snapshot입니다. 세부 task 상태는 [Project #2](https://github.com/orgs/memorybridge-team/projects/2/views/1), 실행 범위와 완료 조건은 연결된 Issue, 변경 검토는 PR, 재현 증거는 `reports/`를 따릅니다. 이 문서는 보드 항목을 나열하는 대신 현재까지 검증된 사실과 다음 gate를 요약합니다.

## 한눈에 보기

- 완료: Task 01 연구 범위, Task 02 State I/O, Task 06 runtime export·assembly·injection
- 진행 중: Task 03 benchmark protocol v1.1 — VOST external zero-shot onboarding이 남음
- 다음: Task 07 paired-state 수집 → Task 08 baselines → Task 09–11 nonlinear 학습·동결 → final/external 평가
- 현재 결과 해석: runtime은 exact하게 동작하지만, Small→Base+ nonlinear translator의 성능은 아직 검증하지 않음

# 1. 연구 문제와 범위

- 작업:
  - 연구 모델을 SAM 2.1 Small → Base+로 확정
  - Nonlinear Translator를 이용한 memory handoff 문제로 범위 고정
  - 과거 frame replay 없이 다음 frame부터 추론하는 것을 목표로 설정
- 결과:
  - 연구 범위와 성공 기준 확정
  - tensor 복원보다 전환 후 mask·객체 정체성·재등장 성능을 우선 평가
- 증빙 링크:
  - [README](../README.md)
  - [연구 범위·baseline·성공 기준](design/01_scope_baselines_success_stop.md)

# 2. State I/O API·상태 계약

- 작업:
  - Translator 입력·출력 state 확정
  - metadata 복사·Target 재생성 정책 확정
- 결과:
  - 번역 대상: `maskmem_features`, `obj_ptr`
  - 복사 대상: frame/object/slot/conditioning/validity/switch metadata
  - Target 생성: `maskmem_pos_enc`
  - 과거 mask·score·prompt history·영상 fingerprint는 handoff에서 제외
  - `02 State I/O` 완료
- 증빙 링크:
  - [Small→Base+ State I/O 계약](design/small_base_state_io_contract.md)
  - [State Assembly Map](architecture/cmmt-state-assembly-map.html)

# 3. Runtime·상태 주입 검증

- 작업:
  - Base+ self-injection과 여러 runtime 상황 검증
  - GPU→CPU memory export 동기화 오류 수정
  - Small→Base+ Direct Copy 예비 실험 수행
- 결과:
  - Base+ self-injection: 후속 frame MSE 0, binary IoU 1.0
  - 다객체·late prompt·객체 재등장·반복 switch 검증 완료
  - Small→Base+ Direct Copy: binary IoU 0.0
  - 같은 tensor shape라도 모델 간 표현 의미가 다를 수 있음을 확인
- 증빙 링크:
  - [Task 06 최종 통합 보고서](../reports/tasks/06_runtime/FINAL_REPORT.md)
  - [Base+ self-injection 원본 실행](../reports/tasks/06_runtime/runs/2026-09-20_base_plus_self_injection/README.md)
  - [Task 06 edge case·Direct Copy 원본 실행](../reports/tasks/06_runtime/runs/2026-09-21_edge_case_and_direct_injection/README.md)

# 4. 벤치마크·데이터셋

- 작업:
  - DAVIS·MOSEv2·LVOS v2 train/validation manifest와 loader 검증
  - MOSEv2/LVOS v2 train을 video-disjoint fit/dev로 분리
  - in-domain held-out와 external cross-dataset zero-shot 역할 분리
- 결과:
  - DAVIS validation: 30개 영상, 249개 case
  - MOSEv2 validation: 433개 영상, 1,720개 case
  - LVOS v2 validation: 140개 영상, 714개 case
  - 세 데이터셋 train/validation loader 전수 검증 `failure_count=0`
  - MOSEv2/LVOS v2를 fit/dev 및 sealed in-domain final로 사용
  - VOST를 primary external zero-shot, DAVIS를 engineering-seen external로 고정
  - VOST manifest·loader·공식 `J/J_tr` 검증은 진행 전
- 증빙 링크:
  - [Benchmark protocol](design/03_benchmark_protocol.md)
  - [Manifest 설명](../manifests/README.md)
  - [MOSEv2 manifest](../manifests/mosev2_valid_v1.json)

# 5. 비교군·평가지표

- 작업:
  - 비교군과 평가 기준 고정
  - 불필요한 `Last-Mask`와 original prompt 없는 `Recent Replay` 제외
- 결과:
  - 비교군: Source-only, Base+-native/Full Replay, Direct Copy, Moment-Matched Copy, Original-Prompt, Last-Visible, Original+Last-Visible, Original+Replay-4/8/16, Nonlinear Translator
  - 정확도: J&F, J, F, switch shock, recovery length, identity continuity
  - 비용: latency, replay frame 수, VRAM, state bytes, translator parameter 수
  - 모든 비교군은 같은 영상·객체·switch 조건 사용
- 증빙 링크:
  - [Benchmark protocol](design/03_benchmark_protocol.md)
  - [Baseline·성공·중단 기준](design/01_scope_baselines_success_stop.md)

## 다음 단계

1. VOST license/download/checksum, manifest, loader, 공식 `J/J_tr`를 검증해 Task 03 v1.1을 닫는다.
2. MOSEv2/LVOS v2 fit/dev에서만 Small/Base+ paired state를 수집한다.
3. 같은 evaluator에서 baseline을 구현하고 fit 통계와 dev model selection을 검증한다.
4. nonlinear 후보를 학습·선정한 뒤 config를 동결한다.
5. MOSE/LVOS sealed in-domain final과 VOST/DAVIS external evaluation을 순서대로 실행한다.
