# vos-memory-translator-nonlinear

SAM 2.1 Small이 switch 시점까지 만든 객체별 memory를 translator로 바꾼 뒤 SAM 2.1 Base+ predictor에 주입해 `switch frame + 1`부터 이어 추적하는 코드입니다.

새 MLP와 `mlp_context`의 실제 데이터 학습 및 후속 추적 성능은 아직 검증되지 않았습니다.

## 모델별 모듈

세 모델의 진입 클래스를 `src/vos_memory_inspector/models/`로 분리했습니다.

| 모듈 | 클래스 | 모델 |
|---|---|---|
| `residual_mlp.py` | `ResidualMLPStateTranslator` | 위치별 잔차 MLP |
| `transformer.py` | `TransformerStateTranslator` | 로컬 Linear + 공간 Transformer |
| `mlp_context.py` | `MLPContextStateTranslator` | 독립 로컬 MLP + gated 공간 문맥 |

`_base.py`는 유효 기록 선택·패딩·dtype 처리를, `_spatial.py`는 공간 backbone·
fusion 연산·포인터 헤드·설정을 공유합니다. `registry.py`는 모델과 비교 preset을
선택합니다. 학습 루프·손실·평가 경로도 공통으로 유지합니다.

```python
from vos_memory_inspector.models import (
    ResidualMLPStateTranslator,
    TransformerStateTranslator,
    MLPContextStateTranslator,
)

mlp = ResidualMLPStateTranslator(spec, spec)
transformer = TransformerStateTranslator(spec, spec)
mlp_context = MLPContextStateTranslator(spec, spec)
```

`build_translator("mlp_context", ...)`와 저장된 additive-MLP payload를 읽는
`TransformerStateTranslator.from_payload(...)`는 새 `MLPContextStateTranslator`를
반환합니다. 기존 `translators`·`transformer_translator` import 경로, CLI, payload
schema와 가중치 key는 유지합니다. 모델 구조와 학습 조건은 바꾸지 않았습니다.

## MLP only 모델

`ResidualMLPStateTranslator`의 기본 설정은 공간 hidden 128, 포인터 hidden 512입니다.
두 경로는 독립적이며 Linear와 GELU만 사용합니다. 공간 경로는 모든 위치에 같은
가중치를 적용하고, 위치·객체·기록 사이의 정보를 섞지 않습니다.

```text
공간 메모리 [B,O,K,64,64,64]
  각 위치의 64차원 벡터 x → Linear(64→128) → GELU → Linear(128→64) → +x
객체 포인터 [B,O,K,256]
  기록별 256차원 벡터 p → Linear(256→512) → GELU → Linear(512→256) → +p
```

두 경로의 마지막 Linear weight와 bias는 0으로 초기화합니다. 첫 층은 일반
초기화를 유지합니다. 같은 크기의 입력에서는 학습 전 출력이 입력과 같습니다.
전체 파라미터는 279,488개입니다(공간 16,576개, 포인터 262,912개).
공간 MAC은 기록당 67,108,864, 포인터 MAC은 262,144입니다. GELU·덧셈은 제외한
계산값이며, 측정한 실행 시간은 아닙니다.

```python
from vos_memory_inspector import ResidualMLPStateTranslator, StateSpec

spec = StateSpec(feature_channels=64, height=64, width=64, pointer_dim=256)
model = ResidualMLPStateTranslator(spec, spec)
# 학습: FP32 출력과 기울기를 유지한다.
spatial_hat, pointer_hat = model.translate_tensors(spatial, pointer, validity)
# 주입: 원본 dtype으로 출력한다. 기본값은 output_dtype="source"다.
translated_state = model.translate(source_state)
```

유효 기록만 계산하고 패딩은 복사합니다. 메타데이터와 presence logit은 보존하고
positional encoding은 target에서 다시 생성합니다. 기존 v2 MLP payload도 읽을 수
있습니다. `pointer_hidden_dim`이 없는 과거 payload는 기존처럼 공간 hidden 값을
사용하고, `output_dtype`이 없으면 float32로 복원합니다.

### 학습 실행

아래 명령은 `cmmt.paired_state_selection.v1` manifest와 checksum이 있는
`cmmt.paired_state_cache.v1` 쌍을 사용합니다. 캐시는
`CACHE_ROOT/{train,validation}/SEQUENCE_objID_switchFRAME.pt` 형식입니다.
같은 영상의 train/validation 중복과 동일 case 중복은 거부합니다.

```bash
PYTHONPATH=src python scripts/train_nonlinear_from_collection.py \
  --selection-manifest /path/to/selection.json \
  --cache-root /path/to/paired-cache \
  --output-dir /path/to/mlp-run \
  --hidden-dim 128 --pointer-hidden-dim 512 \
  --epochs 8 --learning-rate 0.001 \
  --whole-frames --device cuda --seed 7
```

기본값은 8 epoch, 학습률 0.001, 전체 공간 위치 학습입니다. `--whole-frames`는
생략할 수 있습니다. `--spatial-samples-per-pair N`을 명시하면 공간 위치만 일부
추출합니다. 포인터는 두 모드 모두 모든 유효 기록을 사용합니다. 위치 일부 추출은
공간 손실의 추정 분산을 바꾸지만, 각 헤드의 평균 MSE 계수는 둘 다 1입니다.
파라미터 개수로 손실 가중치를 정하지 않습니다.

손실은 FP32 spatial MSE + pointer MSE이며 pair별 균등 가중치입니다. Adam으로
epoch당 한 번 갱신합니다. 매 epoch 갱신 후 validation의 모든 유효 위치·포인터를
평가하고, 합산 손실이 가장 낮은 epoch의 가중치를 복원해 저장합니다. 동률이면
먼저 나온 epoch를 선택합니다. Training history는 각 epoch 갱신 전 손실이고,
validation history는 갱신 후 손실이므로 평가 시점은 다릅니다.

`paired_report.json`의 v2 schema는 기존 `aggregate_mse` 대신 `spatial_mse`,
`pointer_mse`, `state_loss`(두 MSE의 합)를 기록합니다. 학습·checkpoint 선택·
최종 리포트 모두 FP32 `translate_tensors()` 출력으로 계산하며 handoff의 bf16
변환은 거치지 않습니다. `training.model_selection`에 validation 이력, 선택 epoch,
선택 손실을 기록하고, `paired_translators.pt`에도 `training_selection`으로 보존합니다.
가중치는 선택 epoch의 값이며, 마지막 epoch의 값으로 덮어쓰지 않습니다.

Collection의 validation은 checkpoint 선택용이므로 독립 test 점수로 부르지 않습니다.
일반 `cmmt-paired-experiment` CLI는 `--validation-source/--validation-target` 또는
`--validation-case-cache`를 별도로 받습니다. `--test-*`는 선택에 쓰지 않으며,
test 입력이 없으면 리포트의 `evaluation_split`을 `validation`으로 표시합니다.
이 CLI와 collection 스크립트는 기본 epoch·학습률·전체 위치 학습 설정을 공유합니다.

이 경로는 모든 pair를 CPU 메모리에 읽는 소규모 실험용입니다. `1차 학습 종료`의
raw index를 직접 읽는 대규모 DDP 경로는 아니며, 그 실행의 정규화·AdamW·배치
정책과도 다릅니다. test10의 8-epoch 실행과 전체 위치·학습률·validation 선택
원칙은 맞췄지만, test10은 AdamW와 record batch 4, 영상별 validation 평균을
사용했습니다. 이 경로는 Adam과 epoch당 한 번 갱신, pair별 평균이므로 동일
가중치 재현을 보장하지 않습니다. 실제 데이터 학습과 J&F 평가는 실행하지 않았습니다.

```python
import torch
from vos_memory_inspector import ResidualMLPStateTranslator

artifact = torch.load("/path/to/mlp-run/paired_translators.pt",
                      map_location="cpu", weights_only=True)
model = ResidualMLPStateTranslator.from_payload(artifact["residual_mlp"]).eval()
```

`scripts/evaluate_nonlinear_collection.py --translator residual_mlp
--translator-artifact /path/to/mlp-run/paired_translators.pt`로 기존 handoff 평가 경로에
연결할 수 있습니다. 이 명령의 나머지 데이터·SAM 2 인자는 `--help`를 참고합니다.

## 개선 모델: 독립 MLP + Transformer 문맥

`mlp_context`는 로컬 보정과 문맥 보정을 독립적으로 더합니다.
기존 `base` Transformer와 별도 preset으로 추가했으며, 기존 payload의 구조는
유지합니다. 새 구조에는 새 가중치 학습이 필요합니다.

```text
M_hat = M + LocalMLP(M) + alpha * Context(M)
LocalMLP = Conv1x1(64→128) → GELU → Conv1x1(128→64)
Context = patch embed → Transformer → LayerNorm → Conv3x3 → Conv1x1 → Upsample
p_hat = p + Linear(512→256)(GELU(Linear(256→512)(p)))
```

1×1 conv는 각 위치에 같은 MLP를 적용합니다. 로컬 MLP는 gate 없이 보정하고,
채널별 `alpha`는 문맥 보정만 조절합니다. 로컬 MLP의 마지막 층, 포인터 MLP의
마지막 층, `alpha`를 0으로 초기화하여 처음에는 입력을 그대로 출력합니다.
문맥 projection은 일반 초기화를 유지합니다. 첫 update에서 로컬 마지막 층과
문맥 gate가 학습되며, 이후에는 각 분기의 내부 가중치에도 기울기가 전달됩니다.

문맥 설정은 patch 4, 256토큰, 차원 64, 2층, 4헤드, 학습 가능한 위치 임베딩을
유지합니다. 포인터 hidden은 512입니다. attention은 한 기록의 공간 안에서만
작동하며 객체·기록·시간 사이를 섞지 않습니다.

| 모델 | 공간 파라미터 | 전체 파라미터 | 공간 MAC / 기록 |
|---|---:|---:|---:|
| MLP only (`residual_mlp`) | 16,576 | 279,488 | 67.1M |
| 기존 Transformer (`transformer`, preset `base`) | 194,304 | 457,216 | 77.6M |
| 독립 MLP + 문맥 (`mlp_context`) | 206,720 | 469,632 | 127.9M |

모든 모델의 포인터 파라미터는 262,912개입니다. MAC은 conv·Linear·attention
행렬곱을 센 계산값이며 LayerNorm·GELU·softmax·업샘플·덧셈·gate는 제외합니다.
실행 시간이나 정확도 측정값이 아닙니다. 개선 의도는 픽셀별 비선형 보정을
로컬 MLP가 담당하고, Transformer가 공간 문맥에 따른 보정을 추가하도록 하는 것입니다.
기존 `base_mlp`는 두 분기를 GELU 전에 합치므로 이 구조와 다릅니다.

```bash
PYTHONPATH=src python scripts/train_nonlinear_from_collection.py \
  --selection-manifest /path/to/selection.json \
  --cache-root /path/to/paired-cache \
  --output-dir /path/to/mlp-context-run \
  --translator mlp_context \
  --epochs 8 --learning-rate 0.001 --whole-frames --device cuda --seed 7
```

`--translator transformer`로 기존 구조도 같은 학습·validation 선택 경로에서
학습할 수 있습니다. 두 Transformer 모델은 공간 위치 추출을 거부하고 전체
기록으로 학습합니다. 학습·평가 손실은 FP32 spatial MSE + pointer MSE이며,
validation 최적 가중치를 저장합니다. 위의 소규모 pair 학습 경로와 같은 제한이
적용됩니다. 일반 `cmmt-paired-experiment`에서도 두 이름을 선택할 수 있습니다.

```python
from pathlib import Path
from vos_memory_inspector.translator_artifact import load_learned_translator

model = load_learned_translator(
    Path("/path/to/mlp-context-run/paired_translators.pt"), "mlp_context"
)
```

`scripts/evaluate_nonlinear_collection.py`와 `cmmt-sam2-cached-handoff`도
`--translator mlp_context --translator-artifact ...`를 받습니다.
평가 결과의 재사용 여부는 파일 hash뿐 아니라 모델 이름과 component policy까지
비교합니다. 합성 테스트는 초기 identity, 각 분기의 학습, 기록 간 독립성,
validation 선택, 저장·복원을 확인합니다. 실제 J&F 개선은 별도 학습·평가가 필요합니다.

## 비교 모델: 기존 Transformer 구조

`TransformerStateTranslator`는 두 헤드로 나뉩니다. spatial memory를 다루는 `SpatialMemoryTranslator`와, 객체 포인터를 다루는 `ResidualPointerTranslator`입니다. 두 헤드는 가중치를 공유하지 않습니다.

Small과 Base+의 메모리 텐서 크기가 같아서, 모델은 격자를 키우거나 채널을 늘리지 않습니다. 학습 전 출력은 입력과 같습니다.

### 입력과 출력

SAM 2 `inference_state`에서 switch까지의 기록을 꺼내면 축은 다음과 같습니다.

```text
spatial_memory   [B, O, K, 64, 64, 64]   maskmem_features, 보통 bfloat16
object_pointer   [B, O, K, 256]          obj_ptr, 보통 float32
validity         [B, O, K]               실제 기록과 패딩을 구분
```

`B`는 배치, `O`는 객체 수, `K`는 그 시점까지 쌓인 메모리 기록 수입니다. `K`는 기본값이 7이지만 (객체에 대해서 7개의 프레임을 저장) 7로 고정되지 않습니다. 메타데이터인 프레임 번호, slot 순서, conditioning 여부, validity, object id, switch frame은 학습하지 않고 복사합니다. presence logit은 진단용으로만 남기고 target history에는 넣지 않습니다. positional encoding은 번역하지 않으며, 주입 때 Base+ memory encoder가 64×64 격자로 다시 만듭니다.

`validity`가 참인 기록만 `[N, ...]`로 펼칩니다. `N`은 유효한 (배치, 객체, 기록)의 개수입니다. 패딩 칸은 forward에 들어가지 않고, 출력에서 source 값을 복사합니다. 모델 안의 계산 dtype은 파라미터 dtype인 float32이고, handoff로 내보낼 때 spatial memory는 source dtype, 포인터도 source dtype으로 되돌립니다.

### 전체 식

```text
M_hat = M + alpha * Delta(M)
p_hat = p + fc2(GELU(fc1(p)))
```

`M`은 한 기록의 spatial memory `[64, 64, 64]`입니다. `alpha`는 `[1, 64, 1, 1]`이라 채널마다 스칼라 하나이고, 초기값은 0입니다. 포인터 MLP의 마지막 층 weight와 bias도 0으로 초기화합니다. 그래서 학습 전 `M_hat = M`, `p_hat = p`입니다. 0으로 두는 것은 `alpha`와 포인터 마지막 층뿐이고, Transformer 가중치는 일반적인 초기화를 유지합니다.

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

