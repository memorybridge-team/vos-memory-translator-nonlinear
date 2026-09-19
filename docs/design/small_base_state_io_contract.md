# Small → Base+ 상태 I/O 계약

> 상태: **FROZEN v1.0 — 2026-09-20**
>
> 계약 ID: `cmmt.small_to_base_plus.io.v1`
>
> 근거: 정적 config/code, 실제 Small/Base+ runtime inventory, paired dump, fail-closed validator
>
> 단일 객체 Base+ self-injection과 확장 continuation 검증은 이 계약을 소비하는 task 06의 증거다.
> 기준 upstream: Meta SAM 2 commit `2b90b9f5ceec907a1c18123530e92e794ad901a4`

## 결론

SAM 2.1 Small과 Base+는 backbone 크기는 다르지만 video-memory 경계의 기본 shape가 같다. 따라서 첫 nonlinear translator는 크기 변환기가 아니라 **같은 shape 안의 표현 의미를 Small 공간에서 Base+ 공간으로 바꾸는 component-wise mapper**로 정의한다.

Translator가 학습할 연속 입력은 `spatial_memory`, `object_pointer`, `presence_logits` 세 종류다. Frame 번호, 객체 ID, conditioning 여부, record 순서와 validity는 학습하지 않고 정확히 복사한다. Spatial positional encoding은 source 것을 번역하거나 복사하지 않고 Base+가 다시 생성한다.

## 1. 확인된 모델 경계

| 항목 | Small | Base+ | handoff 판단 |
|---|---:|---:|---|
| backbone 시작 차원 | 96 | 112 | translator 입력이 아님 |
| image neck / memory attention `d_model` | 256 | 256 | object pointer 경계와 연결 |
| memory encoder 출력 channel | 64 | 64 | spatial mapper 입력·출력 |
| 기본 memory grid | 64×64 | 64×64 | 1024 입력과 stride 16 기준 |
| object pointer 차원 | 256 | 256 | pointer mapper 입력·출력 |
| `num_maskmem` | 7 | 7 | 한 번의 memory read 범위; 전체 저장 history 상한이 아님 |
| `max_obj_ptrs_in_encoder` | 16 | 16 | 시간축 pointer read 범위; 객체 수 제한이 아님 |

위 표는 pinned config와 `SAM2Base` 생성 코드의 정적 계약이다. 2026-09-20 DAVIS `walking`, object 1, switch frame 10의 실제 checkpoint runtime에서도 Small과 Base+ 모두 spatial memory `[1,1,11,64,64,64]` bfloat16, object pointer `[1,1,11,256]` float32, presence logits `[1,1,11,1]` float32로 확인했다. 이 한 사례의 shape·dtype 일치는 경계를 확인하지만 표현 의미의 동일성을 뜻하지 않는다.

## 2. CanonicalState의 입력과 처리 정책

| Canonical field | 대표 shape | 출처 | 처리 정책 | Target 조립 위치 |
|---|---|---|---|---|
| `spatial_memory` | `[B,O,K,64,64,64]` | `maskmem_features` | nonlinear translation | `maskmem_features` |
| `object_pointer` | `[B,O,K,256]` | `obj_ptr` | 별도 nonlinear translation | `obj_ptr` |
| `presence_logits` | `[B,O,K,1]` | `object_score_logits` | continuous I/O는 고정; direct/calibration/nonlinear/ablation 선택은 task 09 | `object_score_logits` |
| `frame_indices` | `[B,O,K]` | history dict key | 정확히 복사 | cond/non-cond frame key |
| `slot_order` | `[B,O,K]` | canonical record 순서 | 정확히 복사·검증 | materialization 순서 |
| `is_conditioning` | `[B,O,K]` | cond/non-cond dictionary | 정확히 복사 | history dictionary 선택 |
| `validity` | `[B,O,K]` | padding 여부 | 정확히 복사·loss mask | 유효 record만 조립 |
| `object_ids` | 길이 `O` | predictor registry | 정확히 복사 | `obj_id_to_idx`, `obj_idx_to_id` |
| `switch_frame` | scalar | handoff 시점 | 정확히 복사·pair 검증 | continuation 시작점 |

`B`는 현재 injection 구현에서 1, `O`는 prompt로 등록한 객체 수, `K`는 객체별 저장 history를 맞추기 위한 padded record 축이다. `K=7`로 고정하지 않는다. Predictor가 저장한 전체 history와 한 frame에서 실제 attention이 읽는 memory 수는 다르기 때문이다.

Injection 직전에는 Target runtime 정책에 따라 `spatial_memory`와 보존된 `pred_masks`를 Target `storage_device`로, `object_pointer`, `presence_logits`, Target-generated PE를 Target compute device로 이동한다. Source device 문자열은 provenance일 뿐 Target 장치 설정을 덮어쓰지 않는다. Runtime에서 확인한 dtype은 spatial `bfloat16`, pointer/presence `float32`이며 translator 내부 정밀도와 최종 cast는 학습 config에 기록한다.

## 3. Translator에 넣지 않는 상태

다음 데이터는 handoff에는 필요하지만 learned tensor 입력으로 사용하지 않는다.

- `maskmem_pos_enc`: Base+ memory encoder의 position module로 grid에 맞춰 재생성한다.
- temporal position: `frame_indices`와 현재 frame의 차이로 Base+가 자체 정책에 따라 만든다.
- `pred_masks`: 현재 구현은 source의 low-resolution logits를 opaque payload로 보존한다. 일반 propagation의 핵심 read tensor는 아니지만 switch 이후 prompt correction을 continuation-closed하게 유지하는 데 필요하다.
- original point/mask prompts와 `frames_tracked_per_obj`: 객체 등록·상호작용 이력을 그대로 복사한다.
- `cached_features`: 과거 RGB backbone feature를 전달하지 않는다. Fresh target runtime의 cache는 비운다.
- `temp_output_dict_per_obj`: 미완성 상호작용 상태는 전달하지 않고 target에서 빈 dictionary로 시작한다.

## 4. Paired-state 정렬 규칙

학습 pair `(Small state, Base+ native state)`는 다음이 모두 같아야 한다.

1. video와 preprocessing
2. prompt timeline과 object ID
3. switch frame
4. `[B,O,K]` record 축
5. `frame_indices`, `slot_order`, `is_conditioning`, `validity`
6. video frame 수와 원본 높이·너비 metadata

Continuous channel/grid/pointer 차원은 model pair에 따라 달라도 된다. 반대로 위 이산 계약이 다르면 tensor shape가 맞더라도 서로 다른 시간·객체 record를 학습시키는 것이므로 fail closed한다. 이를 위해 `validate_paired_state_contract`가 padding, slot, switch mismatch까지 거부하도록 구현했다.

## 5. 현재 구현과 검증 상태

| 항목 | 상태 |
|---|---|
| Canonical export·history materialization | 구현됨 |
| Target PE 재생성 | 구현됨 |
| object registry·prompt/tracking metadata 복원 | 구현됨 |
| Pair discrete timeline validator | 구현·CPU unit test 추가 |
| Base+ checkpoint same-model export→inject | DAVIS `walking`, object 1, switch 10 통과 |
| Small/Base+ 실제 runtime shape inventory | 단일 paired case 확인 |
| 다객체·prompt correction 뒤 continuation closure | task 06 구현·동작 검증 범위 |
| Small→Base+ paired-state 예시 dump | 생성·checksum 기록 완료 |

## 6. GPU 재개 시 첫 실행

2026-09-20에 `scripts/runpod_preflight.sh`로 revision·checkpoint·storage를 확인한 뒤 `scripts/runpod_base_plus_roundtrip.sh`를 실행했다. DAVIS `walking`, object 1, switch frame 10에서 이후 61 frames의 native/injected 결과가 binary IoU 1.0, MSE 0.0, max absolute error 0.0이었고 injection 중 과거 backbone 호출은 0회였다. 원본 증거는 [`reports/runtime/2026-09-20_base_plus_self_injection/`](../../reports/runtime/2026-09-20_base_plus_self_injection/)에 있다.

이 결과로 단일 객체·첫 frame prompt의 실행 경계는 통과했다. 같은 조건의 Small/Base+ paired example도 생성해 두 모델의 shape·dtype·timeline 일치를 확인했다. `.pt` cache는 166,882,485 bytes이므로 Git에는 checksum만 남기고 RunPod network volume에 보존한다. 상세 보고서는 [`reports/runtime/2026-09-20_small_base_runtime_inventory/`](../../reports/runtime/2026-09-20_small_base_runtime_inventory/)에 있다.

다객체, late prompt, absent/reappearance, prompt correction과 여러 sequence/switch에서의 반복 검증은 이 계약을 소비하는 task 06의 완료 조건이다. task 02는 실제 Small/Base+ inventory, paired dump, 필드 정책, fail-closed validator와 State Assembly Map을 기준으로 검토·동결한다.

## 7. 동결 근거와 변경 관리

- Small/Base+ 실제 경계: spatial `[1,1,11,64,64,64]` bfloat16, pointer `[1,1,11,256]` float32, presence `[1,1,11,1]` float32
- paired cache: `166,882,485 bytes`, SHA-256 `5e9bca17217d522335acf80a454834cf2beab22d4dd8f6204fcd221d2bf1a5f0`
- validator: schema, switch frame, object/frame/slot/conditioning/validity, 영상 크기 불일치를 fail closed
- 시각 계약: [`State Assembly Map`](../architecture/cmmt-state-assembly-map.html)
- runtime inventory: [`reports/runtime/2026-09-20_small_base_runtime_inventory/`](../../reports/runtime/2026-09-20_small_base_runtime_inventory/)

이 계약 이후 새 field, dtype, shape, copy/translate/regenerate 정책을 바꾸면 계약 버전을 올리고 다음을 함께 갱신한다: validator test, Map, example dump, checksum, 영향받는 paired-state shard 목록. Task 06의 edge-case 실패가 현재 계약의 누락을 드러낸 경우에도 조용히 덮어쓰지 않고 v1.1 이상의 변경 기록을 남긴다.

## 8. Task 02 완료 선언

- [x] continuous translator I/O와 shape·dtype 경계 고정
- [x] discrete metadata의 copy·alignment·validity 규칙 고정
- [x] Target PE 재생성과 runtime device 정책 고정
- [x] prompt/prediction history의 continuation payload 정책 고정
- [x] 실제 Small/Base+ inventory와 paired example/checksum 기록
- [x] fail-closed validator와 State Assembly Map 일치 검토

따라서 task 02의 계약 작업은 Done으로 판정한다. 다객체·late prompt·재등장·correction에서 실제 continuation이 맞는지는 task 06의 구현 검증으로 남긴다.
