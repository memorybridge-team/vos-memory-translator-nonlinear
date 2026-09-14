# Cross-Model KV Cache Transfer 분석과 SAM 2 Memory Translator 구현 보고서

작성일: 2026-08-25
구현 기준 브랜치: `main`
SAM 2 고정 소스: `facebookresearch/sam2@2b90b9f5ceec907a1c18123530e92e794ad901a4`

## 1. 결론

Cross-Model KV Cache Transfer의 직접 재사용 가능한 핵심은 “KV라는 이름”이 아니라 다음 세 가지다.

1. paired calibration으로 source state와 target-native/oracle state를 같은 시점에 수집한다.
2. target read path가 실제 소비하는 표현을 대상으로 component별 affine/ridge baseline을 먼저 푼다.
3. 위치 정보는 source 표현에 결합된 채로 억지로 옮기지 않고, source 위치 성분을 제거하거나 target 규칙으로 다시 만든다.

SAM 2에는 LLM의 layer/head/token 축이 그대로 존재하지 않는다. 따라서 LLM mapper를 복사하는 대신 spatial memory, object pointer, presence를 분리한 canonical state와 target-owned positional regeneration 경계를 구현했다. conditioning/non-conditioning frame 역할, object/frame 매핑, slot order, validity 같은 discrete state는 학습하지 않고 그대로 보존한다.

현재 저장소에는 nested state inspector, Hugging Face legacy/cache-object normalizer, SAM 2 multi-object history canonicalizer, Direct/Ridge/Linear/residual two-layer MLP, paired-state metrics와 synthetic smoke가 들어갔다. 실제 SAM 2 체크포인트와 로컬 checkout이 없으므로 checkpoint 기반 Tiny↔Large paired collection과 next-frame injection은 실행하지 않았다. synthetic 결과는 코드 경로 검증일 뿐 SAM 2 성능 근거가 아니다.

## 2. 근거 등급

이 문서는 다음 표기를 엄격히 구분한다.

- **[논문 확인]**: arXiv v1 본문·표·부록에 명시된 사실
- **[공식 코드 확인]**: 고정된 SAM 2 commit 또는 공식 Hugging Face 문서/소스에서 확인한 사실
- **[독립 구현 확인]**: 저자 구현이 아닌 제3자 공개 구현에서 확인한 구체적 engineering choice
- **[구현 결정]**: 이 저장소가 채택한 인터페이스·baseline
- **[미검증]**: 실제 checkpoint runtime으로 아직 확인하지 못한 항목

분석 대상 논문은 Heo et al., *Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for Prefill Reuse*, arXiv:2608.03893v1이다. 원문은 [공식 arXiv 페이지](https://arxiv.org/abs/2608.03893)에서 확인할 수 있다.

2026-08-25 기준 arXiv 페이지와 논문에서 저자 공식 code repository 링크를 찾지 못했다. 따라서 아래 “공개 구현” 분석은 `souvikDevloper/kvbridge`의 독립 구현 commit `949d81d7861e998d5c42db68d7567cc70e2e58c5`를 engineering corroboration으로만 사용하며, 논문의 공식 코드라고 부르지 않는다.

## 3. LLM KV cache의 실제 tensor contract

### 3.1 외부 cache 형식과 mapper 내부 형식

Hugging Face의 논리적 per-layer cache tensor는 K와 V 각각 다음 형식이다.

```text
external key/value: [B, H_kv, T, D_h]
axes:               batch, KV head, sequence, head dimension
dynamic axes:       B, T
```

legacy cache는 layer 순서의 `tuple[(key, value), ...]`다. Transformers 버전에 따라 cache object는 `key_cache`/`value_cache` list를 갖거나 `layers` 아래 각 layer의 `keys`/`values`를 갖는다. 이 저장소의 `normalize_hf_cache`는 Transformers를 import하지 않고 이 세 layout을 duck typing으로 읽는다.

논문 수식은 한 source/target layer·head·K/V에 대해 token matrix `C ∈ R[T,D_h]`로 기술한다. production mapper의 입력은 선택된 source layer마다 모든 source KV head를 펼친 뒤 layer 축을 feature 방향으로 이어 붙인 행렬이다.

```text
[B,Hs,T,Ds]
  --permute(0,2,1,3)--> [B,T,Hs,Ds]
  --reshape------------> [B*T,Hs*Ds]
  --concat k layers----> X: [B*T,k*Hs*Ds]

per target (layer, head, K-or-V):
Y_hat = X W + b,  W: [k*Hs*Ds, Dt], b: [Dt]

stack target heads: [B*T,Ht*Dt]
  --reshape------------> [B,T,Ht,Dt]
  --permute------------> [B,Ht,T,Dt]
```

`flatten_kv_tokens`와 `unflatten_kv_tokens`가 이 permute/reshape를 명시적으로 구현하고 unit test에서 exact round-trip을 확인한다. 별도 normalization은 사용하지 않는다. affine centering은 fitting 수치 절차이며 runtime input normalization layer가 아니다.

### 3.2 MHA, GQA, MQA와 shape mismatch

- MHA에서는 `H_kv = H_query`다.
- GQA에서는 여러 query head가 더 적은 KV head를 공유하므로 `1 < H_kv < H_query`다.
- MQA에서는 `H_kv = 1`이다.
- cache mapper가 직접 다루는 것은 query head 수가 아니라 `H_kv`와 `D_h`다. target attention 내부의 KV repeat/broadcast는 target model이 수행한다.

**[논문 확인]** 평가한 모든 model pair는 GQA이며 source/target 모두 `H_kv=8`, `D_h=128`이다. 논문의 production 수식은 `Hs`, `Ds`, `Ht`, `Dt`가 다른 경우에도 차원상 정의되지만, head 수나 head dimension이 다른 pair의 실증 결과는 없다. 따라서 “mismatch를 지원하는 수식”과 “mismatch에서 검증됨”을 구분해야 한다.

Layer 수 차이는 target layer별 top-k source layer 선택으로 처리한다. source의 모든 head를 feature로 쓰고 target head별 독립 output map을 두므로 head correspondence를 미리 가정하지 않는다. 같은 target layer에 속한 target heads는 선택된 source layer ID만 공유하고 weight는 공유하지 않는다.

Tokenizer와 token sequence가 같아야 동일 token observation을 paired calibration할 수 있다. 논문 평가 범위는 같은 LLM family·호환 tokenizer·dense attention이다. cross-family tokenizer alignment, sliding-window/dense hybrid, 서로 다른 token position은 검증 범위 밖이다.

### 3.3 논문에서 평가한 source/target pair

| Family | Source → Target | Parameter ratio | KV heads | Head dim | Depth ratio |
|---|---|---:|---:|---:|---:|
| Qwen3 | 8B → 32B | 약 4× | 8 → 8 | 128 → 128 | 약 1.8× |
| Qwen3 | 14B → 32B | 약 2.3× | 8 → 8 | 128 → 128 | 약 1.6× |
| Llama 3.1 | 8B → 70B | 약 8.8× | 8 → 8 | 128 → 128 | 약 2.5× |
| Ministral 3 | 3B → 8B/14B | 약 2.7×/4.7× | 8 → 8 | 128 → 128 | 약 1.2×/1.5× |
| Ministral 3 | 8B → 14B | 약 1.8× | 8 → 8 | 128 → 128 | 약 1.2× |

### 3.4 RoPE와 position-free mapping

K cache에는 RoPE가 적용된 key가 저장되고 V에는 positional rotation이 없다. 논문 mapper는 source key의 rotation을 먼저 역변환하고, content-space key를 mapping한 뒤 target model의 RoPE를 같은 token position에 다시 적용한다.

```text
K_source_content(t) = K_source_cached(t) R_source(t)^-1
K_target_content_hat(t) = K_source_content(t) W_K + b_K
K_target_cached_hat(t) = K_target_content_hat(t) R_target(t)

V_target_hat(t) = V_source_features(t) W_V + b_V
```

RoPE가 결합된 cache에 직접 fit한 실험은 calibration length 1024에서는 오차 범위 내 결과를 보였지만, position-free formulation이 길이 변화에 구조적으로 더 안전하다. 이 원칙은 SAM 2에서 “source `maskmem_pos_enc`를 learned translator 입력/출력으로 취급하지 않고 target이 다시 생성”하는 정책으로 대응한다. 두 positional system이 동일하다는 주장은 아니다.

### 3.5 dtype, device, dynamic axis, injection

**[논문 확인]** calibration forward는 BF16, covariance 누적은 FP32, 일부 분석은 FP64다. sequence length `T`는 tokenwise mapping의 observation 축이므로 mapper parameter와 무관하게 동적이다. calibration은 FineWeb-Edu 500 sequences, sequence length 1024, stride 4로 target head당 약 128K token observations를 만든다.

**[독립 구현 확인]** `kvbridge`는 source/target cache를 `[B,H,T,D]`로 검증하고, content key와 rotary factor를 별도 container에 둔다. final mapper는 `einsum("nf,hfd->nhd")` 후 external layout으로 되돌린다. target injection 시 마지막 token을 prefix cache에서 제외하고 target이 그 token을 mapped prefix에 대해 직접 처리하도록 한다. 이 구체적인 “withhold last token” 경로는 독립 구현의 선택이지 논문 저자 코드에서 확인된 사실은 아니다.

논문의 latency 측정은 8×H100, BF16, 50 warmups, 30 trials이며 source-cache cross-GPU 이동은 mapper latency에 포함하지만 mapped cache를 별도 target process로 전송하는 비용은 포함하지 않는다. 보고된 2.7–25× 수치는 이 저장소의 측정값이 아니다.

## 4. Linear/OLS/Ridge mapper의 정확한 구조

### 4.1 구조 probing과 production mapper

논문은 먼저 per `(source layer, target layer, head, K/V)` affine OLS로 선형 구조를 조사한다.

```text
C_target = C_source W + b
```

이 probe에는 ridge regularization이 없다. production mapper는 target layer별로 source layer 후보의 head-averaged `R²`를 K와 RoPE-stripped K/V에 대해 평균해 top-k source layers를 고른다. 같은 target layer의 모든 target heads는 source layer IDs를 공유한다.

Final production fit은 target `(layer l, head h, cache type c∈{K,V})`마다 독립이다.

```text
X ∈ R[N, d_s], d_s = k Hs Ds
Y ∈ R[N, Dt]

Xc = X - mean(X)
Yc = Y - mean(Y)
W = (Xc^T Xc + λI)^-1 Xc^T Yc
b = mean(Y) - mean(X) W
λ = 0.01
```

K와 V는 weight와 bias를 공유하지 않는다. target head끼리도 weight를 공유하지 않는다. source head는 input feature로 모두 결합된다. Layer-selection sweep은 `k ∈ {1,2,4,6,8,10,12,16,20,24,all}`이고 validation log-likelihood benchmark로 pair별 k를 고른다. GSM8K, CoQA와 latency는 k 선택에서 제외했다.

### 4.2 parameter와 compute

Bias를 포함한 mapper parameter 수는 다음이다.

```text
P_affine = 2 Lt Ht (d_s Dt + Dt)
         = 2 Lt Ht ((k Hs Ds) Dt + Dt)
```

논문의 headline formula와 1.01B–3.36B parameter/4–12GB 표기는 작은 bias 항을 생략한다. Token T개 runtime multiply-add 규모는 대략 `2 T Lt Ht d_s Dt`이며 여기서 앞의 2는 K/V 두 cache type이다. 실제 FLOPs convention에 따라 multiply와 add를 각각 세면 두 배가 된다.

Centered sufficient statistics를 누적하면 calibration memory를 N에 비례해 저장하지 않아도 되지만, 각 design dimension에 대해 `XᵀX` 누적 `O(N d_s²)`, `XᵀY` 누적 `O(N d_s Dt)`, solve `O(d_s³)`가 든다. 논문은 한 pair fitting에 single 8×H100 node에서 약 47–87분을 보고한다.

### 4.3 observed success와 failure

Average task retention은 Qwen3 14B→32B 97.6%, Qwen3 8B→32B 87.5%, Llama 3.1 8B→70B 72.8%, Ministral 3 3B→8B 76.2%였다. 반면 Ministral 3 3B→14B 44.2%, 8B→14B 41.6%로 실패했다. 같은 KV shape만으로 transfer 가능성이 보장되지 않는다.

Raw calibration `R²`는 downstream HellaSwag와 약한 음의 상관(`r=-0.20`)을 보인 반면 attention-output cosine은 12 directions에서 `r=+0.57`이었다. 따라서 CMMT도 tensor MSE만으로 성공을 선언하면 안 된다.

## 5. Nonlinear mapper의 정확한 구조

논문 nonlinear 대안은 같은 source feature와 target `(layer,head,K/V)` 단위를 사용하는 independent MLP다.

```text
Linear(d_s, 1024)
ReLU
Linear(1024, 1024)
ReLU
Linear(1024, Dt)
```

- optimizer: Adam
- learning rate: `1e-3`
- epochs: 20
- batch size: 4096 token observations
- objective: MSE
- dropout, normalization, residual connection: 논문에 명시되지 않음
- target head/K/V/layer 간 parameter sharing: 없음

Per mapper parameter 수는 `d_s·1024+1024 + 1024²+1024 + 1024·Dt+Dt`이고 전체는 `2 Lt Ht`를 곱한다.

MLP는 ridge가 이미 잘 되는 Qwen3 14B→32B에서 HellaSwag retention을 0.3 percentage points 낮췄고 Ministral 3 3B→8B에서도 1.5 points 낮췄다. 실패 pair인 Ministral 3B→14B에서는 +24.3 points, 8B→14B에서는 +36.8 points 개선했지만 완전 복구 근거는 아니다. 즉 nonlinear capacity는 보편적인 이득이 아니라 linear structure가 깨지는 pair에서의 fallback이다.

Multi-turn drift는 Qwen3 14B↔32B 한 pair의 CoQA 100 conversations, 약 15 turns에서만 조사됐다. Small→Large gap은 turn 1에서 turn 10까지 1.7 points 늘고 Large→Small은 약 0.33 points/turn drift를 보였다. 이를 SAM 2 long-video drift의 정량 근거로 직접 옮길 수 없다.

## 6. SAM 2.1 continuation-closed state

### 6.1 predictor container

**[공식 코드 확인]** pinned `sam2_video_predictor.py`의 `init_state`는 최소한 다음 container를 유지한다.

| Container | 역할 | handoff 정책 |
|---|---|---|
| `images`, `num_frames`, video H/W | 현재 video와 frame bounds | target runtime가 소유; 경로/identity 검증 |
| `device`, `storage_device`, offload flags | tensor placement | target 환경으로 재설정 |
| `point_inputs_per_obj`, `mask_inputs_per_obj` | 사용자 interaction history | 보존; 학습 금지 |
| `cached_features` | image backbone cache | switch 이후 target frame에는 target이 생성; source cache 직접 번역 안 함 |
| `constants` | frame-independent cached values, PE 등 | target이 재생성 |
| `obj_id_to_idx`, `obj_idx_to_id`, `obj_ids` | client/model object registry | exact 보존 |
| `output_dict_per_obj` | finalized cond/non-cond histories | continuous component 번역 + discrete key 보존 |
| `temp_output_dict_per_obj` | preflight 전 임시 interaction output | 정상 handoff 전 consolidate; 비어 있지 않으면 fail closed 권장 |
| `frames_tracked_per_obj` | tracked frame와 reverse direction | exact 보존 |

각 object의 `output_dict_per_obj[obj_idx]`는 `cond_frame_outputs`와 `non_cond_frame_outputs`를 갖고, 각 frame output은 다섯 key를 갖는다.

| Tensor | 공식 저장/소비 의미 | 예상 standard SAM 2.1 contract | 동적 축 | 정책 |
|---|---|---|---|---|
| `maskmem_features` | memory encoder output; 이후 memory attention의 spatial memory | `[1,64,64,64]` | object/frame record | **translate** |
| `maskmem_pos_enc` | spatial memory의 positional encoding list | list of `[1,64,64,64]` | list level, object view | **target regenerate** |
| `pred_masks` | stored low-resolution mask logits; correction/output path | runtime inspect | H/W may depend on config | preserve or target-owned recompute; translator input 아님 |
| `obj_ptr` | SAM output token projection; later pointer tokens | `[1,256]` | object/frame record | 별도 **translate** |
| `object_score_logits` | object presence/no-object signal | `[1,1]` | object/frame record | scalar calibrate/preserve ablation |

Tiny/Small/Large official SAM 2.1 config는 memory boundary에서 `d_model=256`, memory encoder `out_dim=64`, `num_maskmem=7`, 64×64 memory grid로 정렬돼 있다. 위 shape는 config와 공식 code path에서 도출한 expected contract이며 이 작업환경에서 checkpoint runtime으로 측정한 값은 아니다.

`maskmem_features`가 BF16이면 한 record/object는 `64·64·64·2 = 524,288 bytes`다. 실제 total handoff bytes는 object 수, 보존할 cond history, active non-cond slots, pointer/score, dtype, metadata와 필요 시 mask logits를 합산해야 한다. 코드의 inspector는 가정 대신 runtime `numel·element_size`를 기록한다.

### 6.2 history와 active memory selection

Finalized history와 다음 frame에서 실제 읽는 active memory를 구분해야 한다.

- Conditioning frame outputs는 anchor memory다.
- Non-conditioning outputs는 `num_maskmem-1`과 temporal stride 정책에 따라 최근/선택 record만 다음 memory attention에 들어간다.
- `obj_ptr`는 frame selection 후 token으로 split/stack되어 spatial memory token 뒤에 붙는다.
- frame index와 cond/non-cond role이 temporal positional encoding 및 selection에 영향을 준다.
- 여러 object는 client ID와 internal object index의 안정적인 mapping을 필요로 한다.

완전한 interactive continuation을 위해서는 현재 active tensor만이 아니라 object registry, frame-keyed histories, prompt history, reverse tracking metadata가 닫혀 있어야 한다. 반대로 translator 학습 input은 continuous read-state로 제한한다.

### 6.3 canonical schema

구현한 `CanonicalState`는 다음 축을 고정하고 크기는 runtime에서 검증한다.

```text
spatial_memory       [B,O,K,C,H,W]  translate
object_pointer       [B,O,K,D]      translate with separate head
presence_logits      [B,O,K,1]      scalar calibration
frame_indices        [B,O,K]        preserve
slot_order           [B,O,K]        preserve
is_conditioning      [B,O,K]        preserve
validity             [B,O,K]        preserve/padding mask
object_ids           tuple[O]       preserve
switch_frame         scalar int     preserve
positional_information             target regenerate/provided-target only
metadata                              prompt/frame/object state preservation
```

`b_t`는 source-native canonical state, `a_t`는 같은 frame prefix를 target이 직접 처리한 native/oracle state, `hat a_t=T(b_t)`는 translated state다. Paired fit은 object ID, frame index, conditioning role이 정확히 일치하지 않으면 중단한다.

`canonicalize_sam2_inference_state`는 object별 cond/non-cond 기록을 `(frame index, role)`로 안정 정렬하고 ragged record 수를 K축에 padding한다. 불완전한 memory record는 기본 `strict=True`에서 오류를 낸다. `materialize_sam2_history`는 target-owned `positional_factory` 없이는 history를 만들지 않으므로 source PE가 조용히 주입되지 않는다. Stored mask가 없거나 positional factory가 `None`을 반환해도 fail closed하며, target mask grid가 다를 때는 명시적 `mask_factory`를 전달한다.

## 7. LLM KV와 SAM 2 memory 비교

| 구분 | 직접 재사용 가능 | SAM 2에 맞게 수정 | 직접 적용하면 안 됨 |
|---|---|---|---|
| paired calibration | 동일 prefix 시점의 source/target-native state | token pairing → object/frame/role/validity pairing | frame alignment 없는 독립 dump 회귀 |
| affine/ridge | centered closed-form, bias, lambda audit | per layer/head/KV → spatial/pointer/presence component head | 한 global tensor flatten 후 단일 map |
| position | source position 제거/target position 적용 원칙 | inverse RoPE → target spatial/temporal PE 재생성 | source `maskmem_pos_enc` 직접 복사 |
| selection | target read path 기준 source context 선택 | source layer top-k → cond/non-cond frame/slot selection | layer index를 frame index처럼 해석 |
| dynamic axis | mapper weight와 token count 분리 | T → variable object/record/grid axes와 validity mask | fixed K/O/H/W 상수 하드코딩 |
| shape mismatch | input concat/output projection | head/dim → C/H/W/D projection + grid resampling | same shape를 same semantic basis로 간주 |
| injection | target-native cache container 복원 | target predictor history/materialization + PE factory | transient assembled attention memory만 삽입 |
| evaluation | raw reconstruction 외 downstream readout/drift | language accuracy → J&F, Shock-k, identity break, recovery | HellaSwag retention을 VOS 성능으로 해석 |

LLM의 per-token map은 token 사이를 섞지 않는다. 이 저장소의 spatial translator도 첫 baseline에서는 global attention을 추가하지 않고 모든 object/record/pixel에 공유되는 channel map을 쓴다. Grid가 다르면 먼저 명시적 bilinear resampling을 한다. 이는 유효한 baseline이지만 장거리 object memory relation을 충분히 모델링한다는 뜻은 아니다.

## 8. 구현

| 파일 | 역할 |
|---|---|
| `state_inspector.py` | nested mapping/list/tuple/dataclass/HF cache object 재귀 inventory, JSON/Markdown |
| `hf_cache.py` | legacy/current HF cache normalization과 exact axis round-trip |
| `state_schema.py` | runtime-validated `CanonicalState`, `StateSpec`, bytes와 policy |
| `sam2_state.py` | multi-object/frame SAM 2 history canonicalization과 target PE materialization boundary |
| `translators.py` | Direct, closed-form OLS/Ridge, Linear, residual two-layer MLP |
| `metrics.py` | component MSE/cosine/relative error, bytes, latency, direct improvement |
| `paired_experiment.py` | paired-state training/evaluation과 explicitly synthetic smoke |
| `tests/test_state_transfer.py` | cache/schema/SAM 2/ridge/residual contract tests |

### 8.1 translator ladder

- **Direct**: grid는 bilinear resample, channel/pointer dimension은 truncate 또는 zero-pad한다. 학습 parameter는 없다.
- **Ridge/OLS**: spatial pixel observations에 공유 channel affine map, pointer affine map, scalar presence affine map을 독립 fitting한다. `lambda=0`은 centered `torch.linalg.lstsq`, `lambda>0`은 centered closed-form ridge다.
- **Linear**: 위 세 component에 별도 `nn.Linear`를 둔다.
- **Residual MLP**: feature와 pointer에 `Linear→GELU→Linear`, presence에 scalar affine를 둔다. Identity residual은 input/output dimension이 같은 component에만 켠다. Grid mismatch는 MLP 앞에서 명시적으로 resample한다.

모든 learned baseline은 object/record/pixel 사이 global attention을 사용하지 않는다. Discrete metadata는 clone/copy하고 positional output은 target policy만 기록한다.

### 8.2 CLI

실제 SAM 2 probe는 checkpoint, config, video, switch frame, output과 seed를 모두 받는다. checkpoint나 dataset을 자동 다운로드하지 않는다.

```powershell
sam2-memory-probe `
  --sam2-repo C:\path\to\sam2 `
  --config configs/sam2.1/sam2.1_hiera_t.yaml `
  --checkpoint C:\checkpoints\sam2.1_hiera_tiny.pt `
  --model-id sam2.1-hiera-tiny `
  --video-dir C:\data\video_frames `
  --prompt-mask C:\data\first_mask.png `
  --object-id 1 `
  --switch-frame 30 `
  --jsonl outputs\tiny.jsonl `
  --csv outputs\tiny.csv `
  --canonical-state outputs\tiny_state.pt `
  --seed 7
```

Nested `.pt` state/cache inspection:

```powershell
cmmt-state-inspect `
  --input outputs\state.pt `
  --key b_t `
  --json outputs\state_contract.json `
  --markdown outputs\state_contract.md
```

Synthetic paired-state smoke:

```powershell
cmmt-synthetic-experiment --output-dir outputs\synthetic_cmmt --seed 7
```

여러 video/switch에서 source와 target canonical state를 별도로 수집한 뒤 train/test를 섞지 않고 offline tensor baseline을 실행한다.

```powershell
cmmt-paired-experiment `
  --train-source outputs\train01_tiny.pt `
  --train-target outputs\train01_large.pt `
  --test-source outputs\test01_tiny.pt `
  --test-target outputs\test01_large.pt `
  --output-dir outputs\paired_tiny_to_large `
  --seed 7
```

이 command는 serialized state metric만 계산하며 실제 target next-frame injection/J&F를 대신하지 않는다.

## 9. 실행 결과

환경: Windows 11, Python 3.13.0, PyTorch 2.13.0+cpu, CUDA 없음, seed 7.

```text
pytest: 12 passed in 5.64s (latest warm rerun; cold run 13.05s)
```

Synthetic pair는 source `[1,2,4,6,4,5]` spatial / pointer D=8에서 target C=7, grid 6×4, pointer D=5로 이동하며 고정된 affine+quadratic 관계를 사용한다. Training 6 pairs, test 3 pairs, 120 epochs다.

| Translator | Aggregate MSE | Direct 대비 개선 | Parameters | Median CPU latency |
|---|---:|---:|---:|---:|
| Direct | 1.447550 | 0.00% | 0 | 0.3752 ms |
| Ridge | 0.030507 | 97.89% | 96 | 0.5881 ms |
| Linear | 0.111864 | 92.27% | 96 | 0.5934 ms |
| residual MLP | 0.019728 | 98.64% | 910 | 1.1130 ms |

이 표는 synthetic regression/shape adapter가 작동한다는 증거다. SAM 2 J&F, identity continuity, occlusion recovery 또는 real handoff latency를 입증하지 않는다. Synthetic continuation readout은 finite output과 axis continuity만 확인한다.

## 10. 실제 실험 blocker와 다음 명령

로컬 검색에서 official SAM 2 checkout과 Tiny/Large checkpoint를 찾지 못했다. 사용자 지시에 따라 checkpoint/dataset을 자동 다운로드하지 않았다. 따라서 다음은 **미실행**이다.

- Tiny와 Large가 같은 prefix를 처리한 paired `b_t/a_t` runtime dump
- same-checkpoint export→materialize→inject round-trip
- Tiny→Large/Large→Tiny Direct/Ridge/MLP next-frame smoke
- 실제 frame에서의 component MSE/cosine/relative error
- J&F, switch shock, identity break, recovery length, replay count, CUDA latency/VRAM

필요 입력이 준비되면 먼저 같은 checkpoint round-trip으로 state closure를 검증한다. 그 뒤 같은 video/prompt/switch frame으로 Tiny와 Large probe를 각각 실행하고, object/frame/role key로 paired state를 만든다. Actual injection은 target predictor가 생성한 `maskmem_pos_enc`를 `positional_factory`로 전달해야 하며, source PE를 사용하면 안 된다.

## 11. 연구 판단

1. SAM 2.1 Tiny↔Large는 boundary shape가 같아 Direct Copy가 매우 강할 수 있다. 이 경우 learned mapper가 필요 없다는 결과도 정당하다.
2. Tensor reconstruction이 좋아도 target memory attention output이나 future mask가 나쁠 수 있다. 최소 one-step readout와 +1/+5/+20 frame downstream 검증이 필요하다.
3. Tiny→Large는 source가 버린 정보를 회귀로 복구할 수 없는 방향이다. failure 시 translated memory + replay-1/2/4 hybrid를 평가한다.
4. Target reset, Last-Mask, replay-k, Direct, target-native oracle가 우선 비교군이다.
5. Same-family Tiny↔Large의 성공만으로 cross-architecture claim을 하지 않는다. 최종 novelty는 SAM 2↔Cutie/XMem 같은 heterogeneous spatial/object-centric memory handoff에서 입증해야 한다.

## 12. Primary sources와 구현 근거

- [Cross-Model KV Cache Transfer arXiv abstract](https://arxiv.org/abs/2608.03893)
- [Cross-Model KV Cache Transfer arXiv HTML](https://arxiv.org/html/2608.03893v1)
- [Hugging Face cache explanation](https://huggingface.co/docs/transformers/v4.57.0/cache_explanation)
- [Hugging Face cache strategies](https://huggingface.co/docs/transformers/kv_cache)
- [SAM 2 predictor at pinned commit](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/sam2_video_predictor.py)
- [SAM 2 base at pinned commit](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/modeling/sam2_base.py)
- [SAM 2.1 Tiny config at pinned commit](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/configs/sam2.1/sam2.1_hiera_t.yaml)
- [SAM 2.1 Large config at pinned commit](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/configs/sam2.1/sam2.1_hiera_l.yaml)
- [Independent kvbridge implementation](https://github.com/souvikDevloper/kvbridge/tree/949d81d7861e998d5c42db68d7567cc70e2e58c5)
