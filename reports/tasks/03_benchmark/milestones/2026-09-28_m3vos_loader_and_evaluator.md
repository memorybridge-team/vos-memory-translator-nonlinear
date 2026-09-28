# M³-VOS loader and evaluator contract closure

## Scope

This milestone validates dataset plumbing only. It is not a learned-translator
result and no external benchmark result was used for model selection.

## Frozen manifest and loader

- Immutable delivery revision: `5deb15b2baeaaa294ca168b789537729f7fb53a5`
- Manifest: `manifests/m3vos_external_v1.json`
- Manifest content SHA-256:
  `b71c4af8634b53668ed1e74ef51234d815499d48d7a93ab290b4d0884632b612`
- 471 sequences, 1,590 cases, three fixed 25/50/75% switches per actual
  annotation object.
- Loader result: 1,590/1,590 cases checked, `failure_count=0`.

The loader reads only a first non-empty GT prompt mask and the RGB at the
switch frame; future GT is not read. Label 255 is never a prompt object. Four
metadata/annotation discrepancies are retained in the manifest instead of
creating empty-object cases:

| Sequence | Metadata-only labels | Annotation-only labels |
| --- | --- | --- |
| `0374_assemble_machinery_4` | — | 3 |
| `0375_assemble_machinery_5` | — | 3 |
| `0406_assemble_machinery_13` | 3 | — |
| `0453_erupt_foam_2` | 2 | — |

## Official evaluator smoke

Official evaluator checkout: `M3VOS_Experiment`
`8cf8f9b3cb069d8476ef6c3c0b8f11b8337c3b56`.

- GT-copy sequence `0001_open_cup_1`, object 1: `J=1.0`, `J_last=1.0`,
  `J_cc=0.9999999999990095` (floating-point epsilon from the evaluator's
  denominator).
- Two GT-copy shards (`0001_open_cup_1`, `0002_open_cup_2`) merged per object
  exactly match a single combined evaluator invocation: maximum absolute error
  `0.0`.
- The released evaluator emits `J/J_last/J_cc`; `J_last` is the final quarter
  after its sampling and endpoint removal. It is not named `J_tr` in code.

## Boundary integrity (non-official supplementary metric)

On five native-resolution GT-copy frames of `0001_open_cup_1`, PNG
export/reload produced exactly `J=F=J&F=1.0`. A half-resolution nearest-neighbor
round trip lowered J to roughly 0.986 while F remained 1.0; one-pixel dilation
or erosion lowered J to roughly 0.966–0.973. This demonstrates expected
serialization/morphology sensitivity, so F/J&F stays a conditional appendix
metric rather than a primary M³-VOS claim. No label-255 void pixel was observed
in the received delivery; official-output and void-aware semantics remain
documented separately.

## Remaining scope

M³-VOS onboarding is complete. Full per-video evaluation, clustered confidence
intervals and CMMT method comparisons belong to Task 13 after configuration is
frozen, not to this Task 03 plumbing gate.
