# Task 02 최종 보고서 — Small/Base+ State I/O 계약

## 목적

SAM 2.1 Small의 상태를 Base+로 전달하기 전에 두 모델의 runtime state 경계를 확인하고,
nonlinear translator의 입력·출력과 Target state 조립 규칙을 결과 확인 전에 고정했다.

## 최종 계약

| 구분 | 값 | 처리 |
|---|---|---|
| 번역 대상 | `maskmem_features`, `obj_ptr` | Small 표현을 Base+ 표현으로 변환 |
| 필수 metadata | `frame_indices`, `slot_order`, `is_conditioning`, `validity`, `object_ids`, `switch_frame` | 값과 순서를 정확히 보존 |
| Target 생성 | `maskmem_pos_enc` | Base+ 정책과 shape로 재생성 |
| 외부 interaction | 실제 RGB·prompt/correction timeline | 과거 correction replay에만 사용 |
| 비전송 | Source mask/score archive, prompt/tracking dictionary, 영상 크기·길이·fingerprint | handoff payload와 bytes에서 제외 |

Source와 Target이 동일 영상, frame 순서, preprocessing과 switch를 사용한다는 조건은
실험 manifest가 보장한다. 영상 checksum은 paired-data 무결성 검사이며 handoff API에
포함하지 않는다.

## 수행 내용과 결과

1. 공식 SAM 2 revision과 Small/Base+ config를 기준으로 memory boundary의 shape, dtype,
   conditioning/non-conditioning history와 object pointer 구조를 inventory했다.
2. CanonicalState exporter, pair validator, Target materializer/injector의 역할을 분리했다.
3. switch, slot, validity, conditioning과 object registry가 다르면 pair를 거부하도록
   fail-closed validation을 고정했다.
4. 같은 shape를 갖는 두 모델 사이에서도 표현 의미는 호환된다고 가정하지 않으며,
   translator는 shape 변환기가 아니라 component-wise semantic mapper로 정의했다.
5. Task 06의 Base+ self-injection이 Target-generated PE와 최소 history만으로 exact하게
   이어지는 것을 확인해 계약이 실제 continuation에 충분함을 검증했다.

## 완료 기준과 증거

| 완료 기준 | 판정 | 증거 |
|---|---|---|
| Small/Base+ memory boundary 표 | 완료 | [State I/O 계약](../../../docs/design/small_base_state_io_contract.md) |
| 번역·복사·Target 생성 field 정책 | 완료 | 계약 문서와 [Assembly Map](../../../docs/architecture/cmmt-state-assembly-map.html) |
| pair 정렬 fail-closed 검증 | 완료 | validator와 회귀 test |
| CPU test | 완료 | Issue #2 및 PR #3 증거 |
| 실제 checkpoint inventory | 완료 | [runtime inventory](runs/2026-09-20_runtime_inventory/README.md) |
| Base+ export→inject 경계 | 완료 | [Task 06 최종 보고서](../06_runtime/FINAL_REPORT.md) |

## 한계와 후속 책임

Task 02는 번역 품질을 검증하지 않는다. Small→Base+ learned translation, 데이터셋 전체의
paired state 수집과 downstream 성능은 Task 07 이후의 책임이다. 과거 frame correction은
최소 history를 직접 수정하지 않고 실제 RGB·prompt timeline을 이용한 Target replay로
라우팅한다.
