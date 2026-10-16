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

2026-10-13 게시 검토에서는 이 경로에 필요한 변경만 선별했다.
collector·이전 LVOS/official 학습 및 J&F 운영 branch 전체를 가져오지 않았다.
기존 runtime/manifest와 모델팀의 frozen body는 보존했다. 검토 및 실제 CPU/GPU
증거의 범위는 [학습 PR 검토 기록](../reports/training_r2/REVIEW_AND_RUNTIME_2026-10-13_KO.md)을 따른다.

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

## 2026-10-05 operational clarification

The dated [training/evaluation pipeline v2](training_evaluation_pipeline_v2.md)
documents the current RunPod recipe without restoring orchestration code removed
from this core. The deployed official trainer is a separately pinned local
snapshot; a new official-validation rollout adapter is being prepared in an
isolated evaluation workspace. Neither deployment is implied by a documentation
merge. Pending evaluation work is not labelled as completed benchmark evidence.

The two-GPU training Pod and the two-GPU evaluation Pod have different jobs.
They share read access to completed artifacts, not a four-rank training process.
Publishing core documentation never updates the running trainer. Runtime changes
require their own verification, immutable source binding and, where necessary,
a checkpoint-boundary continuation receipt.
