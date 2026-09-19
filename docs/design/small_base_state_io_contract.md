# Small → Base+ 상태 I/O 계약

> 상태: **정적 코드·config 검토 완료, checkpoint runtime 검증 보류**  
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

위 표는 pinned config와 `SAM2Base` 생성 코드의 정적 계약이다. 실제 checkpoint에서 나온 dtype, grid, record 수와 값의 의미가 같은지는 GPU runtime inventory로 다시 확인해야 한다.

## 2. CanonicalState의 입력과 처리 정책

| Canonical field | 대표 shape | 출처 | 처리 정책 | Target 조립 위치 |
|---|---|---|---|---|
| `spatial_memory` | `[B,O,K,64,64,64]` | `maskmem_features` | nonlinear translation | `maskmem_features` |
| `object_pointer` | `[B,O,K,256]` | `obj_ptr` | 별도 nonlinear translation | `obj_ptr` |
| `presence_logits` | `[B,O,K,1]` | `object_score_logits` | calibration 또는 component ablation | `object_score_logits` |
| `frame_indices` | `[B,O,K]` | history dict key | 정확히 복사 | cond/non-cond frame key |
| `slot_order` | `[B,O,K]` | canonical record 순서 | 정확히 복사·검증 | materialization 순서 |
| `is_conditioning` | `[B,O,K]` | cond/non-cond dictionary | 정확히 복사 | history dictionary 선택 |
| `validity` | `[B,O,K]` | padding 여부 | 정확히 복사·loss mask | 유효 record만 조립 |
| `object_ids` | 길이 `O` | predictor registry | 정확히 복사 | `obj_id_to_idx`, `obj_idx_to_id` |
| `switch_frame` | scalar | handoff 시점 | 정확히 복사·pair 검증 | continuation 시작점 |

`B`는 현재 injection 구현에서 1, `O`는 prompt로 등록한 객체 수, `K`는 객체별 저장 history를 맞추기 위한 padded record 축이다. `K=7`로 고정하지 않는다. Predictor가 저장한 전체 history와 한 frame에서 실제 attention이 읽는 memory 수는 다르기 때문이다.

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
| Base+ checkpoint same-model export→inject | GPU 대기 |
| Small/Base+ 실제 runtime shape inventory | GPU 대기 |
| 다객체·prompt correction 뒤 continuation closure | GPU 대기 |
| Small→Base+ paired-state 수집 | 위 gate 통과 후 시작 |

## 6. GPU 재개 시 첫 실행

GPU가 생기면 `scripts/runpod_preflight.sh`로 revision·checkpoint·storage를 확인하고, `scripts/runpod_base_plus_roundtrip.sh`로 Base+ same-checkpoint export→inject를 실행한다. 통과 기준은 다음 frame 이후 native/injected mask agreement, 과거 backbone 호출 0회, target-generated positional encoding, 저장 record 수와 object registry 일치다. 이 gate가 실패하면 paired-state 수집이나 nonlinear 학습으로 넘어가지 않는다.

