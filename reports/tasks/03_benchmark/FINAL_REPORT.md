# Task 03 최종 보고서 — Benchmark protocol and external evaluation onboarding

## 목적

Translator 품질을 비교하기 전에 학습·선정·최종 평가·외부 zero-shot 역할을 분리하고,
각 dataset에서 같은 first prompt·switch 조건을 재현 가능하게 읽을 수 있는지 검증했다.
Task 03은 모델 성능을 보고하는 task가 아니라, 이후 Task 07–13이 leakage 없이 실행될
benchmark 계약을 동결하는 task다.

## 최종 데이터 역할

| 역할 | Dataset | 사용 규칙 |
| --- | --- | --- |
| Translator fit / development | MOSEv2·LVOS v2 train의 video-disjoint fit/dev | paired state primary loss에는 future GT를 쓰지 않고, dev만 구조·checkpoint 선택에 사용 |
| Sealed in-domain final | LVOS v2 validation, MOSEv2 official server | config freeze 뒤 한 번만 실행 |
| Primary external zero-shot | VOST validation | 학습·통계·threshold·replay-k 선택에 사용하지 않음 |
| Complementary external stress | PUMaVOS, M³-VOS | 각각 partial/unusual-mask, material phase-transition 조건; config freeze 뒤 평가 |

DAVIS는 historical runtime engineering evidence로만 남고, 현재 translator 학습·평가·성능
주장 범위에서 제외한다.

## 완료된 benchmark gates

| Dataset | 동결된 산출물 | 검증 결과 |
| --- | --- | --- |
| MOSEv2/LVOS v2 | train manifest, video-level fit/dev split | dense GT prompt·RGB·object·switch loader 계약을 전수 검증; fit/dev와 sealed final 역할 분리 |
| VOST | 70-val sequence, 210-case fixed 25/50/75% switch manifest | official `J/J_last` GT-copy smoke와 input contract 통과 |
| PUMaVOS | 24 sequence, 78-case external manifest | 21,187 RGB/GT paired frames, inventory failure 0; local GT-copy J/F/J&F smoke 통과 |
| M³-VOS | 471 sequence, 1,590-case external manifest | 202,577 RGB/GT pairs, loader 1,590/1,590, official `J/J_last/J_cc` GT-copy/shard merge, boundary integrity 통과 |

M³-VOS immutable HF delivery revision은
`5deb15b2baeaaa294ca168b789537729f7fb53a5`이며 manifest content SHA-256은
`b71c4af8634b53668ed1e74ef51234d815499d48d7a93ab290b4d0884632b612`이다.
문헌/project page의 479 videos/205,181 dense masks와 받은 delivery의 471 sequences/
202,577 pairs는 구분해서 기록한다. 실험의 manifest denominator는 받은 delivery를 따른다.

## M³-VOS evaluator와 boundary 정책

공식 evaluator checkout `8cf8f9b3cb069d8476ef6c3c0b8f11b8337c3b56`의 실제
출력은 `J`, `J_last`, `J_cc`다. `J_last`는 해당 코드가 temporal sampling과 endpoint
removal 뒤 마지막 25%에 계산하는 값이며, released code의 명칭은 `J_tr`가 아니다.

GT-copy `0001_open_cup_1` object 1은 `J=1.0`, `J_last=1.0`,
`J_cc=0.9999999999990095`였다. 두 prediction shard를 합친 per-object 결과와 combined
evaluation 결과의 maximum absolute error는 `0.0`이었다.

`F/J&F`는 M³-VOS의 official primary metric이 아니다. 다섯 native-resolution GT-copy
frame에서 export/reload는 정확히 `J=F=J&F=1.0`이었고 resize 및 1-pixel morphology
sensitivity도 기록했다. 따라서 향후 appendix에만 조건부 보조 metric으로 보고한다.

## 발견한 데이터 계약 이슈

M³-VOS `target_object.json`과 annotation label이 일치하지 않는 4개 sequence를 발견했다.
metadata-only object는 빈 prompt case로 만들지 않고, annotation에 실제 존재하는 non-void
label만 prompt/evaluation 객체로 썼다. 해당 mismatch는 manifest에 그대로 남겼다. 수신
delivery에서는 void label 255이 관찰되지 않았으며, official-output와 void-aware semantics는
후속 implementation에서도 분리한다.

## 완료 기준과 증거

| 완료 기준 | 판정 | 증거 |
| --- | --- | --- |
| fit/dev·sealed·external 역할 분리 | 완료 | [benchmark protocol](../../../docs/design/03_benchmark_protocol.md) |
| VOST onboarding | 완료 | [v1.1 addendum](milestones/2026-09-24_v1_1_addendum.md) |
| PUMaVOS inventory·manifest·metric smoke | 완료 | [PUMaVOS source record](runs/2026-09-27_external_onboarding_sources.md) |
| M³-VOS immutable delivery inventory | 완료 | [inventory milestone](milestones/2026-09-28_m3vos_delivery_inventory.md) |
| M³-VOS loader/evaluator/boundary contract | 완료 | [contract milestone](milestones/2026-09-28_m3vos_loader_and_evaluator.md) |

## 한계와 후속 책임

Task 03은 CMMT method의 accuracy를 주장하지 않는다. full per-video predictions,
video-clustered confidence intervals, method comparison과 external evaluation 결과는
configuration freeze 뒤 Task 13에서 수행한다. 다음 구현 순서는 Task 07 paired-state
collection, Task 08 baseline evaluator, Task 09 nonlinear translator 학습이다.
