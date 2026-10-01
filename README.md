# vos-memory-translator-nonlinear

SAM 2.1 Small이 switch 시점까지 만든 객체별 memory를 translator로 바꾼 뒤 SAM 2.1 Base+ predictor에 주입해 `switch frame + 1`부터 이어 추적하는 코드입니다.

This repository exports per-object memory from SAM 2.1 Small up to a switch frame, translates it, injects it into a SAM 2.1 Base+ predictor, and continues tracking from `switch frame + 1`.

아직 검증되지 않음

## 모델 구조

기본 모델은 `TransformerStateTranslator`의 `base` 프리셋입니다. 클래스는 두 헤드로 나뉩니다. 공간 메모리를 다루는 `LightweightSpatialMemoryTranslator`와, 객체 포인터를 다루는 `ResidualPointerTranslator`입니다. 두 헤드는 가중치를 공유하지 않습니다.

Small과 Base+의 메모리 텐서 크기가 같아서, 모델은 격자를 키우거나 채널을 늘리지 않습니다. 같은 칸의 값을 고칩니다. 학습 전 출력은 입력과 같습니다.

### 입력과 출력

SAM 2 `inference_state`에서 switch까지의 기록을 꺼내면 축은 다음과 같습니다.

```text
spatial_memory   [B, O, K, 64, 64, 64]   maskmem_features, 보통 bfloat16
object_pointer   [B, O, K, 256]          obj_ptr, 보통 float32
validity         [B, O, K]               실제 기록과 패딩을 구분
```

`B`는 배치, `O`는 객체 수, `K`는 그 시점까지 쌓인 메모리 기록 수입니다. `K`는 7로 고정되지 않습니다. 프레임 번호, slot 순서, conditioning 여부, validity, object id, switch frame은 학습하지 않고 복사합니다. presence logit은 진단용으로만 남기고 target history에는 넣지 않습니다. 위치 인코딩은 번역하지 않으며, 주입 때 Base+ memory encoder가 64×64 격자로 다시 만듭니다.

`validity`가 참인 기록만 `[N, ...]`로 펼칩니다. `N`은 유효한 (배치, 객체, 기록)의 개수입니다. 패딩 칸은 forward에 들어가지 않고, 출력에서 source 값을 복사합니다. 모델 안의 계산 dtype은 파라미터 dtype인 float32이고, handoff로 내보낼 때 공간 메모리는 source dtype, 포인터도 source dtype으로 되돌립니다.

### 전체 식

```text
M_hat = M + alpha * Delta(M)
p_hat = p + fc2(GELU(fc1(p)))
```

`M`은 한 기록의 공간 메모리 `[64, 64, 64]`입니다. `alpha`는 `[1, 64, 1, 1]`이라 채널마다 스칼라 하나이고, 초기값은 0입니다. 포인터 MLP의 마지막 층 weight와 bias도 0으로 초기화합니다. 그래서 학습 전 `M_hat = M`, `p_hat = p`입니다. 0으로 두는 것은 `alpha`와 포인터 마지막 층뿐이고, Transformer 가중치는 일반적인 초기화를 유지합니다.

`Delta`는 로컬 갈래와 공간 문맥 갈래의 합입니다.

```text
Delta = Conv1x1(M) + Upsample(Conv1x1(Conv3x3(tokens)))
```

앞의 `Conv1x1`에는 bias가 있습니다. 문맥 갈래의 `Conv1x1`에는 bias가 없습니다. 업샘플은 채널별 bilinear이고 `align_corners=False`입니다.

### 공간이 거치는 텐서

유효 기록 `N`장에 같은 가중치를 적용합니다. 한 장의 경로는 아래와 같습니다.

```text
M                          [64, 64, 64]
├─ 로컬
│    Conv2d 1×1, 64→64     [64, 64, 64]
│
└─ 문맥
     Conv2d 4×4, stride 4  [64, 16, 16]
     토큰으로 펼침          [256, 64]
     + 위치 임베딩          [256, 64]
     Transformer block ×2  [256, 64]
     LayerNorm             [256, 64]
     격자로 복원            [64, 16, 16]
     Conv2d 3×3, 64→64     [64, 16, 16]
     Conv2d 1×1, bias 없음 [64, 16, 16]
     bilinear → 64×64      [64, 64, 64]
             │
             └─ 두 갈래를 더해 Delta
M + alpha * Delta          [64, 64, 64]
```

패치 임베딩은 겹치지 않는 4×4 칸 하나를 토큰 하나로 만듭니다. 16×16 격자이므로 토큰 수는 256이고, 토큰 차원 `d_model`은 64입니다. 위치 임베딩은 학습 가능한 `[1, 256, 64]`이며 표준편차 0.02로 초기화합니다. 모든 프레임이 같은 위치 임베딩을 공유합니다. dropout은 0입니다.

Transformer 블록은 Pre-LN입니다. 각 층은 다음 순서입니다.

```text
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

Attention은 토큰마다 `Linear(64 → 192)`로 query, key, value를 만들고 4개 헤드로 나눕니다. 헤드 차원은 16입니다. 헤드마다 행렬은 `[256, 256]`이고, 출력은 `Linear(64 → 64)`로 다시 합칩니다. `N`은 이 연산의 배치 축입니다. 기록이 여러 장이어도 프레임 사이, 객체 사이에 attention 값은 흐르지 않습니다. FFN은 토큰마다 따로 `64 → 128 → GELU → 64`를 적용합니다. `128`은 `d_model * mlp_ratio`이고 `mlp_ratio`는 2입니다.

3×3 conv는 16×16 토큰 격자에서 이웃 8칸을 봅니다. 그 뒤 bias 없는 1×1 conv가 채널을 64로 유지한 채, bilinear가 64×64 픽셀 격자로 올립니다. 1×1 conv를 업샘플 앞에서 하는 이유는 bilinear가 채널별로 선형이라 순서를 바꿔도 값이 같고, 16×16에서 곱하는 편이 연산이 적기 때문입니다.

로컬 갈래의 1×1 conv는 원본 64×64의 각 위치에서 채널 벡터만 봅니다. 이 출력과 업샘플된 문맥을 원소별로 더한 값이 `Delta`입니다.

### 포인터

포인터는 기록마다 길이 256 벡터 하나를 독립적으로 바꿉니다.

```text
256 → Linear → 512 → GELU → Linear → 256
p_hat = p + 그 출력
```

공간 토큰, 다른 기록, 다른 객체와 연결되지 않습니다. 마지막 Linear의 weight와 bias는 0에서 시작합니다.

### 크기

| 모듈 | 파라미터 | 프레임당 MAC |
|---|---:|---:|
| patch embed 4×4 | 65,600 | 16.8M |
| 위치 임베딩 | 16,384 | — |
| Transformer 2층 | 66,944 | 33.6M |
| 마지막 LayerNorm | 128 | — |
| 3×3 conv | 36,928 | 9.4M |
| 문맥 1×1 conv | 4,096 | 1.0M |
| 로컬 1×1 conv | 4,160 | 16.8M |
| alpha | 64 | — |
| 공간 합계 | 194,304 | 77.6M |
| 포인터 MLP | 262,912 | 기록당 0.3M |
| 전체 | 457,216 | |

공간 77.6M MAC 중 문맥 갈래가 60.8M, 로컬 갈래가 16.8M입니다. 포인터는 기록당 `2 × 256 × 512` MAC입니다. LayerNorm, GELU, bilinear, residual 덧셈, `alpha` 곱은 이 MAC 합계에 넣지 않습니다.

## Model structure

The default model is the `base` preset of `TransformerStateTranslator`. It has two heads that do not share weights: `LightweightSpatialMemoryTranslator` for spatial memory and `ResidualPointerTranslator` for the object pointer.

Small and Base+ use the same memory tensor size, so the model does not resize the grid or expand the channels. It changes the values in place. Before training, the output equals the input.

### Input and output

Records exported from a SAM 2 `inference_state` up to the switch have these axes:

```text
spatial_memory   [B, O, K, 64, 64, 64]   maskmem_features, usually bfloat16
object_pointer   [B, O, K, 256]          obj_ptr, usually float32
validity         [B, O, K]               real records versus padding
```

`B` is the batch, `O` is the object count, and `K` is the number of memory records stored up to the switch. `K` is not fixed at 7. Frame indices, slot order, conditioning flags, validity, object ids, and the switch frame are copied and are not learned. Presence logits remain diagnostic and are not written into the target history. Positional encoding is not translated; at injection the Base+ memory encoder rebuilds it for the 64×64 grid.

Only records whose `validity` is true are flattened to `[N, ...]`, where `N` counts valid (batch, object, record) tuples. Padding never enters the forward pass and is copied from the source into the output. Computation uses the parameter dtype, float32. On handoff, spatial memory and the pointer are cast back to their source dtypes.

### Equations

```text
M_hat = M + alpha * Delta(M)
p_hat = p + fc2(GELU(fc1(p)))
```

`M` is one record's spatial memory, `[64, 64, 64]`. `alpha` has shape `[1, 64, 1, 1]`, one scalar per channel, and is initialized to 0. The pointer MLP's last-layer weight and bias are also initialized to 0. Before training, `M_hat = M` and `p_hat = p`. Only `alpha` and that last pointer layer start at zero; the Transformer weights keep a standard initialization.

`Delta` is the sum of a local branch and a spatial-context branch.

```text
Delta = Conv1x1(M) + Upsample(Conv1x1(Conv3x3(tokens)))
```

The first `Conv1x1` has a bias. The context branch's `Conv1x1` does not. Upsampling is per-channel bilinear with `align_corners=False`.

### Spatial tensor path

The same weights are applied to all `N` valid records. One record follows this path:

```text
M                          [64, 64, 64]
├─ local
│    Conv2d 1×1, 64→64     [64, 64, 64]
│
└─ context
     Conv2d 4×4, stride 4  [64, 16, 16]
     flatten to tokens     [256, 64]
     + positional embedding[256, 64]
     Transformer block ×2  [256, 64]
     LayerNorm             [256, 64]
     restore grid          [64, 16, 16]
     Conv2d 3×3, 64→64     [64, 16, 16]
     Conv2d 1×1, no bias   [64, 16, 16]
     bilinear → 64×64      [64, 64, 64]
             │
             └─ sum the branches to form Delta
M + alpha * Delta          [64, 64, 64]
```

Patch embedding turns each non-overlapping 4×4 cell into one token. The 16×16 grid yields 256 tokens, and `d_model` is 64. The positional embedding is a learnable `[1, 256, 64]` parameter initialized with standard deviation 0.02 and shared by every frame. Dropout is 0.

Each Transformer block is Pre-LN:

```text
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

Attention maps each token through `Linear(64 → 192)` into query, key, and value, then splits them into 4 heads of dimension 16. Each head uses a `[256, 256]` matrix, and `Linear(64 → 64)` merges the heads. `N` is only the batch axis of this operation: several records can be present while no attention value moves between frames or between objects. The FFN applies `64 → 128 → GELU → 64` independently to each token. The width 128 is `d_model * mlp_ratio` with `mlp_ratio` 2.

The 3×3 convolution sees the eight neighbors on the 16×16 token grid. A bias-free 1×1 convolution keeps 64 channels, and bilinear upsampling restores the 64×64 pixel grid. The 1×1 convolution runs before upsampling because per-channel bilinear is linear, so the order does not change the value, and the multiply is cheaper on 16×16.

The local 1×1 convolution reads only the channel vector at each position of the original 64×64 map. `Delta` is the elementwise sum of that output and the upsampled context.

### Pointer

The pointer head maps one 256-dimensional vector per record:

```text
256 → Linear → 512 → GELU → Linear → 256
p_hat = p + that output
```

It is not connected to the spatial tokens, to other records, or to other objects. The last Linear layer's weight and bias start at 0.

### Size

| Module | Parameters | MAC per frame |
|---|---:|---:|
| 4×4 patch embed | 65,600 | 16.8M |
| positional embedding | 16,384 | — |
| 2 Transformer blocks | 66,944 | 33.6M |
| final LayerNorm | 128 | — |
| 3×3 conv | 36,928 | 9.4M |
| context 1×1 conv | 4,096 | 1.0M |
| local 1×1 conv | 4,160 | 16.8M |
| alpha | 64 | — |
| spatial total | 194,304 | 77.6M |
| pointer MLP | 262,912 | 0.3M per record |
| total | 457,216 | |

Of the 77.6M spatial MAC, the context branch accounts for 60.8M and the local branch for 16.8M. The pointer costs `2 × 256 × 512` MAC per record. LayerNorm, GELU, bilinear upsampling, the residual additions, and the `alpha` multiply are outside this MAC total.
