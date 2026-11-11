# vos-memory-translator-nonlinear

SAM 2.1 Small이 switch 시점까지 만든 객체별 memory를 translator로 바꾼 뒤 SAM 2.1 Base+ predictor에 주입해 `switch frame + 1`부터 이어 추적하는 코드입니다.

새 translator의 실제 데이터 학습 및 후속 추적 성능은 아직 검증되지 않았습니다.

## 모델 구조

Translator는 두 가지입니다. 둘 다 "선형 연결층 하나 + Pre-LN 블록 하나"이며,
`TransformerStateTranslator`는 블록 안에 self-attention 부분층이 더 있습니다.
각 부분의 근거와 출처는 [`docs/translator_model_design.md`](docs/translator_model_design.md),
이전 구조에서 바뀐 내용과 이유는 [`docs/refactor_decisions.md`](docs/refactor_decisions.md)에 있습니다.

| 모듈 | 클래스 | 모델 |
|---|---|---|
| `residual_mlp.py` | `ResidualMLPStateTranslator` | 연결층 + 위치별 feed-forward 블록 |
| `transformer.py` | `TransformerStateTranslator` | 연결층 + attention·feed-forward 블록 |

`_base.py`는 유효 기록 선택·패딩·dtype 처리를, `_stitched.py`는 두 모델이 공유하는
모든 부품(연결층, 블록, attention, 포인터 헤드, `TranslatorConfig`, payload)을 가집니다.
`registry.py`는 모델과 비교 preset을 선택합니다. 학습 루프·손실·평가 경로는 공통입니다.

```python
from vos_memory_inspector.models import (
    ResidualMLPStateTranslator,
    TransformerStateTranslator,
)

mlp = ResidualMLPStateTranslator(spec, spec)
transformer = TransformerStateTranslator(spec, spec)
```

구조 설정은 `TranslatorConfig(num_layers=1, mlp_ratio=4.0, attention=<클래스가 정함>,
num_heads=1, sr_ratio=4, output_dtype="source")` 하나입니다. `attention`은 호출자가
정하지 않고 모델 클래스가 정합니다. 이전의 `SpatialTransformerConfig`와 MLP 모델의
`hidden_dim`, `pointer_hidden_dim`, `skip_mode`, `output_dtype` 인자는 없습니다.
학습 설정 JSON의 `model_options`는 `TranslatorConfig` 필드입니다.

기존 `translators`·`transformer_translator` import 경로와 모델 이름(`residual_mlp`,
`transformer`)은 유지합니다. payload schema는 `cmmt.stitched_translator.v1`로 바뀌었습니다.

> 이 브랜치의 구조 변경 전에 만든 payload와 checkpoint
> (`cmmt.residual_mlp_translator.v2/.v3`, `cmmt.spatial_context_transformer_translator.v1/.v2`)는
> 이 브랜치에서 읽을 수 없습니다. 커밋 `913e35b` 또는 `main`에서 읽으세요.
> 모든 모델을 새로 학습해야 합니다.

### 공통 식

한 기록은 spatial memory `M`(`[64, 64, 64]`, 채널·높이·너비)과 객체 포인터 `p`(길이 256)입니다.
공간 메모리는 4,096개 위치(토큰), 위치마다 64채널로 봅니다.

```text
Z     = S(M) = W M + b            위치마다 같은 선형 연결층, W와 b는 ridge 해에서 시작
Z     = Z + Attn(LN(Z))           Transformer 모델에만 있음
M_hat = Z + FFN(LN(Z))            두 모델 모두
p_hat = A p + c                   포인터는 선형, A와 c는 ridge 해에서 시작
```

- `residual_mlp`: 블록에 feed-forward 부분층(64 → 256 → 64, GELU)만 있습니다.
- `transformer`: 같은 블록에 feed-forward 앞의 self-attention 부분층이 더해집니다.
  쿼리는 4,096개이고, 키·값은 축마다 4배 줄여 256개입니다(stride 4 합성곱 + LayerNorm).
  head는 1개이며, 고정 sine 위치 인코딩을 쿼리와 키에만 더합니다.
- 각 부분층의 마지막 선형층은 0에서 시작합니다. 그래서 학습 전 모델은 연결층 출력과
  같습니다. ridge를 넣으면 닫힌 형태의 선형 기준선이고, 넣지 않으면 항등(자리표시자)입니다.
- 마지막 LayerNorm은 두지 않습니다. 출력은 target memory이므로 크기를 유지해야 하기 때문입니다.
- 기록 하나씩 독립으로 계산합니다. 위치 사이는 attention 안에서만 섞이고, 기록·객체·프레임
  사이에는 정보가 흐르지 않습니다. source와 target의 공간 격자는 같아야 하며(SAM 2.1 세 크기 모두 64×64)
  다르면 오류입니다.
- 프레임 번호, slot 순서, validity, 객체 id는 번역하지 않고 복사합니다. 위치 인코딩은 target이
  다시 만듭니다.
- 같은 구조와 설정을 세 쌍(Small → Base+, Base+ → Large, Small → Large)에 씁니다.
  쌍마다 다른 것은 가중치와 학습 설정입니다.

### 크기

기록당 수치이며 코드에서 센 값입니다. MAC은 선형층·축소 합성곱·attention 두 곱을 세고,
LayerNorm·GELU·softmax·위치 인코딩·덧셈은 제외합니다. 측정한 실행 시간이 아닙니다.

| 모델 | 공간 파라미터 | 포인터 파라미터 | 전체 | 공간 MAC | 포인터 MAC |
|---|---:|---:|---:|---:|---:|
| `linear` (ridge 기준선) | 4,160 | 65,792 | 69,952 | 16,777,216 | 65,536 |
| `residual_mlp` | 37,376 | 65,792 | 103,168 | 150,994,944 | 65,536 |
| `transformer` | 119,872 | 65,792 | 185,664 | 337,641,472 | 65,536 |
| `residual_mlp_matched` (대조군) | 119,936 | 65,792 | 185,728 | 486,539,264 | 65,536 |

`residual_mlp_matched`는 feed-forward 비율 14(은닉 폭 896)로 Transformer 모델과 공간
파라미터 수를 맞춘 대조군입니다. attention 때문인지 파라미터 수 때문인지를 가를 때 씁니다.
두 모델은 attention 부분층 하나만 다르므로, 둘의 비교는 그 부분층을 뺀 ablation입니다.

### Preset

`direct`, `moment_match`, `linear`, `residual_mlp`, `transformer`, `residual_mlp_matched`입니다.
깊이·head 수·축소 비율·feed-forward 비율은 `TranslatorConfig` 값이므로 ablation은 preset이 아니라
설정 덮어쓰기로 합니다. 이전 구조의 ablation preset은 모두 없어졌습니다.

`python -m vos_memory_inspector.transformer_translator --preset {residual_mlp,residual_mlp_matched,transformer}`는
더미 입력의 forward 요약을 출력합니다.

### 사용

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
positional encoding은 target에서 다시 생성합니다.

### 입력과 출력

SAM 2 `inference_state`에서 switch까지의 기록을 꺼내면 축은 다음과 같습니다.

```text
spatial_memory   [B, O, K, 64, 64, 64]   maskmem_features, 보통 bfloat16
object_pointer   [B, O, K, 256]          obj_ptr, 보통 float32
validity         [B, O, K]               실제 기록과 패딩을 구분
```

`B`는 배치, `O`는 객체 수, `K`는 그 시점까지 쌓인 메모리 기록 수입니다. `K`는 기본값이 7이지만 (객체에 대해서 7개의 프레임을 저장) 7로 고정되지 않습니다. 메타데이터인 프레임 번호, slot 순서, conditioning 여부, validity, object id, switch frame은 학습하지 않고 복사합니다. presence logit은 진단용으로만 남기고 target history에는 넣지 않습니다. positional encoding은 번역하지 않으며, 주입 때 target memory encoder가 64×64 격자로 다시 만듭니다.

`validity`가 참인 기록만 `[N, ...]`로 펼칩니다. `N`은 유효한 (배치, 객체, 기록)의 개수입니다. 패딩 칸은 forward에 들어가지 않고, 출력에서 source 값을 복사합니다. 모델 안의 계산 dtype은 파라미터 dtype인 float32이고, handoff로 내보낼 때 spatial memory는 source dtype, 포인터도 source dtype으로 되돌립니다.

### 구형 학습 실행(소규모)

아래 명령은 `cmmt.paired_state_selection.v1` manifest와 checksum이 있는
`cmmt.paired_state_cache.v1` 쌍을 사용합니다. 캐시는
`CACHE_ROOT/{train,validation}/SEQUENCE_objID_switchFRAME.pt` 형식입니다.
같은 영상의 train/validation 중복과 동일 case 중복은 거부합니다.

```bash
PYTHONPATH=src python scripts/train_nonlinear_from_collection.py \
  --selection-manifest /path/to/selection.json \
  --cache-root /path/to/paired-cache \
  --output-dir /path/to/mlp-run \
  --translator residual_mlp \
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

이 경로는 ridge 시작을 적용하지 않는 구형 학습기입니다. 연결층이 항등 자리표시자에서
시작하므로 `docs/translator_model_design.md`가 설명하는 구조대로 학습한 모델이 아닙니다.
논문에 쓸 모델은 `cmmt-state-training`으로 학습합니다(`docs/state_training.md`).
`--hidden-dim`과 `--pointer-hidden-dim` 플래그는 없어졌고, `run_paired_experiment`와
`run_synthetic_experiment`는 `translator_config: TranslatorConfig | None`을 받습니다.

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


## 구조 문서

구조의 각 부분과 출처는 [`docs/translator_model_design.md`](docs/translator_model_design.md),
이전 구조(항등/affine skip, alpha 게이트, 패치 임베딩, 문맥 합성곱, 업샘플, 학습형 위치 임베딩,
포인터 MLP)를 없앤 결정은 [`docs/refactor_decisions.md`](docs/refactor_decisions.md)를 참고하세요.

## 전체 데이터용 학습 경로

v2 `state_only` 로더, bounded record batch, streaming ridge 초기화와 공통 AdamW
trainer는 `cmmt-state-training`으로 실행합니다. 기존 소규모 collection 경로는 유지합니다.
모델별 설정과 CPU 검사·전체 데이터 preflight·CUDA 실행 준비 명령은
[전체 데이터 학습 안내](docs/state_training.md)를 참고하세요.

200GB 다운로드는 형식·실행 검사 샘플입니다. 공식 validation을 포함한 전체 index가
준비되지 않으면 전체 학습 준비 완료로 표시하지 않습니다. FiLM·부재 전용 분기·
새 attention 손실은 추가하지 않았으며, 실제 spatial 번역 생략도 비활성화 상태입니다.
