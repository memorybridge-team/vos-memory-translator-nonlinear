# Task 07 paired-state collector repair — 2026-10-01

## Scope and result

- Local branch: `fix/paired-state-collection-contract`; no main merge, remote push, PR, or RunPod worker change.
- Reconciled the upstream Task 07 collector through `9492bfcf` with the local repair contract selectively; historical Task 06 runtime evidence was retained.
- Production selection accepts only signed MOSEv2/LVOS v2 official train manifests and complete, disjoint frozen fit/development video splits. DAVIS is excluded from active collection.
- The Task 07 entry point now builds `cmmt.paired_generating.v1`, records official/runtime prompt and switch mapping, and passes `--prompt-frame-index` plus `--generating-json` to the prepared-case subprocess. Only the first object prompt mask is opened; future GT is not passed to SAM 2.
- State-only + active-memory collection uses the bounded lazy JPEG loader and retains the Task 06 default path. The strict cache writer and checksum/generating-hash skip logic preserve incompatible or partial prior files rather than overwriting them.

## CPU verification

From the repository root with the current Python environment:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp 'C:\Users\SAMSUNG\Documents\Codex\pytest-task07-20260930-full' tests
```

Result: **120 passed in 122.15s**. Additional `--dry-run --max-cases 1` selections passed against both frozen MOSEv2 and LVOS v2 train manifests. `compileall` and Git diff whitespace checks also passed after cleanup.

After adding the final manifest-tampering and 8-shard union checks and legacy labeling, the final Task 07 suite passed **11 tests in 8.12s**:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp 'C:\Users\SAMSUNG\Documents\Codex\pytest-task07-20261001-final' tests/test_task07_collector.py
```

Added/retained checks cover manifest hash, DAVIS rejection, split overlap/coverage, sparse official-to-runtime mapping, prompt/generating CLI flags, cache checksum plus generating identity, atomic JSON, queue exclusivity, and prior repair/case-cache regressions.

## Canonical command (RunPod, review before starting)

The following command is the production entry point. Paths reflect the team's documented `/workspace/CMMT` layout and need local confirmation on the Pod. Start with `--dry-run`; then run a new single-case pilot with `--max-cases 1` after confirming checkpoints and policy. Remove `--max-cases` only for approved full collection. Each static worker must use its own run directory and shard index; give each worker one visible GPU.

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python scripts/task07/prepare_paired_state_dataset.py \
  --source-manifest manifests/mosev2_train_v1.json \
  --fit-split manifests/mosev2_train_v1_fit.json \
  --development-split manifests/mosev2_train_v1_development.json \
  --dataset-root /workspace/CMMT/data/MOSEv2 \
  --sam2-repo /workspace/CMMT/sam2 \
  --source-config configs/sam2.1/sam2.1_hiera_s.yaml \
  --source-checkpoint /workspace/CMMT/checkpoints/sam2.1_hiera_small.pt \
  --source-model-id sam2.1_small \
  --target-config configs/sam2.1/sam2.1_hiera_b+.yaml \
  --target-checkpoint /workspace/CMMT/checkpoints/sam2.1_hiera_base_plus.pt \
  --target-model-id sam2.1_base_plus \
  --read-policy-json configs/paired_read_policy.example.json \
  --network-volume-root /workspace/CMMT/task07-review-cache \
  --run-directory /workspace/CMMT/task07-review-runs/shard-00 \
  --paired-split all --shard-count 8 --shard-index 0 --device cuda:0 --dry-run
```

For LVOS v2, use the three `lvosv2_train_v1*.json` manifests and `data/LVOSv2`. The provided directory names are execution examples, not evidence that those files currently exist on the Pod.

## Files and baseline relationship

Added canonical collector, README, bounded loader and collector tests; modified `case_cache.py`, `cli.py`, `roundtrip.py`, `sam2_state.py`, cache/state regression tests and `PROJECT_CONTEXT.md`. The original `scripts/prepare_paired_state_dataset.py` is retained with a legacy docstring. The preexisting untracked operator guide is preserved separately.

The local repair base is `e7bca7a`. Upstream `backup/main-before-cleanup-20260929` was inspected at `29a88c5c`, and the Task 07 branch at `9492bfcf`. Selected Task 07 changes were integrated with conflict resolution; unrelated benchmark reports and cleanup-main deletions were not merged. Existing repair validation/ownership logic remains in use. Its `apply_memory_policy` is the single production selection step; an additional selection pass would truncate the memory twice.

`_prepare_command` forwards the state-only/active-memory flags. `prepare_handoff_case_main` parses them, `prepare_cross_model_case_reference` stops target collection at the switch and applies the policy, and `write_case_cache` omits both mask groups while validating `cache_mode`. The state schema remains part of canonical validation.

## Unverified gates

- Actual SAM 2.1 Small/Base+ GPU collection, late-prompt LVOS case, Network Volume restart/resume and multi-worker queue behavior have not been executed in this local environment.
- No downstream VOS J&F, identity, or translator performance claim follows from these CPU tests.
- Before starting or changing a RunPod job, review the code and operator guide, confirm the pinned SAM checkout/checkpoint SHA and volume namespace, then run a single-case GPU pilot. Do not modify a currently running worker from this checkout.
