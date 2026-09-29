# Task 07 local paired-state workflow

This directory is the local operational home for Task 07. It is not copied to
the public translator core repository.

- Code: this directory.
- Frozen input manifests: `v2/manifests/` until the cleanup PR replaces `main`.
- Paired-state cache (`.pt`, checksum, collection status): RunPod Network Volume.
- Raw collection evidence and final benchmark reports: `vos-memory-benchmark`.

`prepare_paired_state_dataset.py` accepts only MOSEv2 or LVOS v2 train manifests
plus their frozen `fit` and `development` video split manifests. It supplies
only the first object-prompt mask to SAM 2 and never reads future masks.

`evaluate_nonlinear_collection.py` is state-only development validation. It
does not make a downstream J/F claim; official evaluation happens later through
the benchmark repository after model selection is frozen.
