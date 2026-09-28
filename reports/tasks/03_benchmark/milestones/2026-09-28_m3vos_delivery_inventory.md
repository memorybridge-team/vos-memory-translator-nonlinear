# M³-VOS immutable delivery inventory

## Purpose

Verify the completed M³-VOS Hugging Face delivery before it is used for a
CMMT prompt loader, switch manifest, or evaluator result. This is an inventory
gate, not a model evaluation.

## Reproducibility

- Dataset: `Lijiaxin0111/M3_VOS`
- Pinned revision: `5deb15b2baeaaa294ca168b789537729f7fb53a5`
- Delivery root: `/workspace/CMMT/data/M3VOS-manual`
- Validator: `scripts/validate_m3vos_inventory.py`
- Command:

  ```bash
  python3 scripts/validate_m3vos_inventory.py \
    /workspace/CMMT/data/M3VOS-manual \
    --revision 5deb15b2baeaaa294ca168b789537729f7fb53a5 \
    --output reports/tasks/03_benchmark/runs/2026-09-28_m3vos_delivery_validation/inventory.json
  ```

The direct file-level Hugging Face delivery has no single archive to checksum.
The immutable revision and inventory digest are the reproducibility anchor.
Raw per-sequence output remains in the ignored RunPod run directory;
`inventory.json` SHA-256 is
`2a6c98fede7e7d5ce6e85f4ae093a106842111a5906becc0abb7db6bbd73ad41`.

## Result

| Check | Result |
| --- | ---: |
| declared split / JPEG / annotation / viewer / target-object set agreement | pass |
| sequences | 471 |
| RGB JPEG frames | 202,577 |
| GT PNG annotations | 202,577 |
| object records | 530 |
| core members | 68 |
| stem mismatches or metadata disagreement | 0 |

The received `target_object.json` uses per-sequence mappings such as
`{"obj_1": {...}}`, rather than the list format initially assumed by the
validator. The parser was corrected and the full scan rerun; this is not a
data redownload.

## Metric contract correction

Official evaluator checkout `M3VOS_Experiment`
`8cf8f9b3cb069d8476ef6c3c0b8f11b8337c3b56` actually emits `J`, `J_last`,
and `J_cc`. `J_last` is the last quarter after temporal sampling and endpoint
removal. The dataset reader detects label 255 as void, while the default
evaluator call passes no void array to its metric function. CMMT will retain a
separate official-output smoke and void-aware engineering check.

## Interpretation and remaining gate

The literature/project figure (479 videos, 205,181 dense masks) differs from
this fixed delivery (471 sequences, 202,577 RGB/GT pairs). Neither figure is
silently substituted for the other: manifests and denominators use the
received delivery inventory, while the literature figure is cited as context.

Task 03 remains open for the object-level first-prompt loader and fixed switch
manifest, official evaluator GT-copy/shard-merge smoke, and conditional
boundary F/J&F integrity gate.
