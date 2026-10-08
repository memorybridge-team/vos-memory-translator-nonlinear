# Repository responsibilities

## Public experiment core: `vos-memory-translator-nonlinear`

The public core contains only reusable Small-to-Base+ state handoff code:
CanonicalState, SAM 2 export/materialize/inject, translator definitions,
runtime continuation, frozen manifests/splits, and regression tests.  It does
not contain datasets, paired-state tensors, checkpoints, raw run logs, or
dataset-specific evaluation orchestration.

## Fixed-split training contract (2026-10-09)

사용자 승인으로 `state_training/`에 재사용 가능한 고정 split·cached state
학습·validation R² 선정 API를 추가한다. 모델팀의 frozen Transformer body를
그대로 사용하며 데이터 수집, baseline, rollout/J&F, 외부 benchmark는 추가하지
않는다. 원본 영상·cache·checkpoint·실행 로그는 계속 Network Volume에서 관리한다.
실행/미실행 경계와 명령은 [운영 안내](state_training_r2_operator_guide.md)에 있다.
기존 frozen manifests의 fit/development는 저장 위치를 찾는 입력이며 새 학습
역할을 결정하지 않는다. MOSE official train의 새 90/5/5 membership은 별도
immutable manifest로 고정한다.

## Local Task 07/09 workspace

`CMMT/scripts/task07/` is the operational home for paired-state collection and
state-only development checks.  It reads frozen MOSEv2/LVOS v2 manifests from
the core, calls the core's `cmmt-sam2-prepare-case` runtime entry point, and
writes all `.pt` payloads, checksums, and collection status to the RunPod
Network Volume.  These scripts are not a public-main API.

## Benchmark repository: `vos-memory-benchmark`

The benchmark repository owns dataset-specific manifest builders and Task
08/13 aggregation: official metrics, baseline sweeps, temporal summaries,
confidence intervals, and full-evaluation evidence.  It consumes core runtime
reports but does not copy the model runtime or tensor payloads into Git.

## Evidence archive: `vos-memory-system`

Completed Task 02/06 runtime evidence, historical DAVIS support code, legacy
KV-cache utilities, project workflow helpers, and presentations are archived
there.  Archive material is traceable by source revision but is not imported
by the public core.  The frozen visual State Assembly Map is also kept there:
[cmmt-state-assembly-map.html](https://github.com/memorybridge-team/vos-memory-system/blob/chore/archive-runtime-evidence/cmmt-evidence/nonlinear-v2/architecture/cmmt-state-assembly-map.html).

## Merge rule

Before removing a public-core file, copy it to its assigned destination,
verify its SHA-256, remove every public-core import and console entry point,
then run compile and regression tests.  A large tensor or dataset is never
committed to any repository.
