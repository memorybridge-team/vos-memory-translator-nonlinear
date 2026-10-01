# Task 07 local paired-state workflow

This directory is the local operational home for Task 07. It is not copied to
the public translator core repository.

- Code: this directory.
- Frozen input manifests: this checkout's `manifests/` (MOSEv2/LVOS v2 train only).
- Paired-state cache (`.pt`, checksum, collection status): RunPod Network Volume.
- Raw collection evidence and final benchmark reports: `vos-memory-benchmark`.

`prepare_paired_state_dataset.py` accepts only MOSEv2 or LVOS v2 train manifests
plus their frozen `fit` and `development` video split manifests. It supplies
only the first object-prompt mask to SAM 2 and never reads future masks.
For production it builds the repair branch's `cmmt.paired_generating.v1` contract,
maps official frame IDs to contiguous runtime indices, and passes both the
prompt index and generating JSON to `cmmt-sam2-prepare-case`. The default
read policy is `configs/paired_read_policy.example.json`; use
`--read-policy-json` only for an explicitly frozen alternative. Both predictors
are checked against that policy at runtime.

Outputs are placed under `<network-volume-root>/<run-identity-sha>/<fit|development>/`.
A cache is skipped only if its file checksum and `.complete.json` generating
hash match the current case. An existing incomplete or incompatible cache is
preserved and collection stops; use a new volume namespace after audit. The
`--dry-run` mode validates frozen manifests and prints a selection, but does
not check checkpoint files or run SAM 2. Successful CPU tests do not establish
GPU collection success or downstream J/F.

The older `paired_collection_cli.py` collector remains a repair/audit path;
Task 07 production collection starts here, with no compact-cache conversion.

`evaluate_nonlinear_collection.py` is state-only development validation. It
does not make a downstream J/F claim; official evaluation happens later through
the benchmark repository after model selection is frozen.
