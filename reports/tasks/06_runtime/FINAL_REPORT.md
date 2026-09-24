# Task 06 최종 보고서 — Memory extraction and target injection

## 목적

Task 02의 최소 State I/O 계약을 실제 SAM 2.1 checkpoint runtime에서 사용해 Source state를
export하고 fresh Target에 과거 RGB replay 없이 조립·주입한 뒤, 다음 frame부터 continuation이
정확히 동작하는지 검증했다.

## 구현 범위

- Canonical state export와 schema validation
- fresh Target registry/history materialization과 injection
- 전환 이후 correction의 Target-native 처리
- 전환 이전 correction의 안전한 prompt-anchor replay
- injected state의 재-export와 repeated handoff
- Small→Base+ Direct Copy를 통한 cross-model injection 기계 검사

## 통합 결과

| Gate | 사례 | 결과 |
|---|---|---|
| Base+ single-object self-injection | DAVIS `walking`, switch 10 | 이후 61 frames MSE 0, max error 0, binary IoU 1.0, 과거 backbone call 0 |
| 다객체·late prompt | `bike-packing`, object 1@0, object 2@10, switch 20 | 이후 48 frames exact, replay 0 |
| 부재·재등장 | `india`, object 3, switch 35 | 이후 45 frames exact, replay 0 |
| 전환 이후 correction | `walking`, switch 10, correction 20 | 52 frames exact, history replay 0 |
| 전환 이전 correction | switch 20, correction 10 | anchor부터 21 frames Target replay 후 51 frames exact |
| repeated handoff | switch 10→20 | 첫 전환 뒤 61 frames, 두 번째 뒤 51 frames exact |
| Small→Base+ Direct Copy | `walking`, switch 10 | 주입은 성공했으나 Base+-native 대비 이후 61 frames binary IoU 0.0 |

최종 RunPod 환경의 전체 회귀검사는 `57 passed`였다. 반복 handoff 중 diagnostic인
`object_score_logits`를 exporter가 필수로 요구하던 결함을 발견했고, continuation 필수
field를 `maskmem_features`와 `obj_ptr`로 바로잡았다. score를 payload에 재도입하지 않아
최소 계약은 유지됐다.

## 해석

Base+→Base+ exact 결과는 state export·조립·주입 경로가 원래 Target continuation을
손상하지 않는다는 구현 closure다. Small→Base+ Direct Copy의 실패는 tensor shape가
같아도 두 모델의 표현 의미가 호환되지 않음을 보인 한 사례의 pilot이다. learned
Nonlinear Translator의 필요 가능성을 보여 주지만 성공이나 전체 benchmark 성능을
증명하지 않는다.

## 완료 기준과 증거

| 완료 기준 | 판정 | 증거 |
|---|---|---|
| 최소 payload refactor 전체 test | 완료 | [2026-09-23 run](runs/2026-09-23_correction_and_repeated_switch/README.md), `57 passed` |
| Base+ checkpoint same-model smoke | 완료 | [2026-09-20 run](runs/2026-09-20_base_plus_self_injection/README.md) |
| 전환 이후 correction | 완료 | [correction run](runs/2026-09-23_correction_and_repeated_switch/README.md) |
| 전환 이전 correction replay | 완료 | 같은 run의 replay JSON |
| 여러 영상·switch 반복 검증 | 완료 | single/multi-object, reappearance, repeated switch runs |

## 한계와 다음 단계

Runtime gate는 DAVIS의 제한된 engineering 사례를 이용한 구현 검증이며 GT 일반화 성능이
아니다. Task 07은 MOSEv2/LVOS v2 fit/dev에서 Small/Base+ paired state를 수집하고,
Task 08–13은 공정한 비교군, nonlinear 학습, sealed in-domain과 external 평가를 수행한다.
