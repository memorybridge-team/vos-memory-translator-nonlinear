# Repository responsibilities

## Public experiment core: `vos-memory-translator-nonlinear`

The public core contains only reusable Small-to-Base+ state handoff code:
CanonicalState, SAM 2 export/materialize/inject, translator definitions,
runtime continuation, frozen manifests/splits, and regression tests.  It does
not contain datasets, paired-state tensors, checkpoints, raw run logs, or
dataset-specific evaluation orchestration.

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
