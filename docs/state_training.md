# 전체 데이터 학습 준비

PyTorch 2.4 이상이 필요하다. 검증 환경은 SAM2에 설치된 PyTorch를 사용한다.

`cmmt-state-training`은 v2 `state_only` 캐시를 기록 단위로 읽는다. 설치 전에는
`PYTHONPATH=src python -m vos_memory_inspector.state_training.cli`로 같은 CLI를 실행한다.
RGB·mask는 상태 학습에 필요하지 않다. J&F 평가에는 별도 영상·주석과 predictor 실행이 필요하다.

## 데이터와 preflight

입력은 1차 학습의 `cmmt.official_state_training_index.v1` 원본 `raw_index.json`이다.
MOSE/LVOS의 fit·development는 모두 train이다. LVOS official valid만 validation으로
사용한다. 200GB 다운로드는 형식·실행 검사 샘플이며 별도 validation을 만들지 않는다.
캐시를 수정하거나 원본 index의 역할을 다시 지정하지 않는다.

원래 경로가 달라지면 root-map JSON을 제공한다. 가장 긴 prefix부터 치환하고 나머지
상대 경로는 유지한다. 예:

```json
{
  "/workspace/CMMT-task07-artifacts/production": "/data/CMMT-task07-artifacts/production"
}
```

CLI에서 root-map의 상대 경로는 현재 작업 디렉터리를 기준으로 해석한다.
파일명 `switch1221`과 runtime `switch_frame`은 다를 수 있다. Case의 원래 frame 번호와
캐시의 runtime 인덱스를 따로 보존하며 파일명을 runtime 번호로 사용하지 않는다.

```sh
cmmt-state-training preflight \
  --index /data/run_binding/raw_index.json --root-map /data/root-map.json \
  --config configs/state_training/transformer.json \
  --world-size 2 --output runs/preflight-data.json
```

기본 preflight는 모든 case를 checksum·tensor 수준에서 읽고 train-only target RMS를
재계산한다. `--sample-cases 8`은 검사량을 제한하므로 샘플만 성공해도 `launch_ready`는
false다. 누락 목록과 손상 원인을 JSON에 기록한다. 준비 미완료의 종료 코드는 2다.
두 모델(`residual_mlp`, `transformer`) 모두 ridge 시작이 필요하므로 `--ridge-init` 없이는
`ridge_initialization_ready: false`로 표시하고 종료 코드 2를 낸다. 순서는 `preflight` →
`ridge-fit` → `preflight --ridge-init runs/ridge.pth` → `train --ridge-init runs/ridge.pth`다.
재개 검사에는 `--checkpoint`를 사용하고
`--world-size`를 실제 실행 process 수에 맞춘다.

```sh
cmmt-state-training preflight \
  --index /data/run_binding/raw_index.json --root-map /data/root-map.json \
  --config configs/state_training/transformer.json --ridge-init runs/ridge.pth \
  --world-size 2 --output runs/preflight-data-ridge.json
```

전체 검사는 `torchrun`을 띄우기 전에 단일 process로 수행한다. 보고서는 v2 schema이며,
검사한 case ID·SHA256·파일 크기/mtime·train RMS를 포함한다. `train`에는 반드시
`--preflight-report`(위 두 번째 preflight의 `runs/preflight-data-ridge.json`)를 전달한다. 시작/재개 시에는 index SHA,
보고서 digest·정규화·현재 파일 stamp/sidecar만 확인하고 tensor를 다시 읽지 않는다.
CLI는 이 검사를 process group 생성 전에 완료한다. 실제 case를 로드할 때의 SHA256
검사는 계속 수행한다. 오래된 v1 보고서나 바뀐 파일은 거부하므로 preflight를 다시 실행한다.
데이터 보고서는 모델별로 공유할 수 있으며 모델·ridge·checkpoint 설정은 실행 시 별도로 대조한다.
라이브러리 호출도 `readiness=`가 필수다. DDP caller는 group 생성 전에
`validate_readiness(index, report)`로 검사한 객체를 만들어 전달한다.

메모리 예상량은 process당 worker cache·전송 대기 tensor와 모델/optimizer 추정치다.
전체 데이터가 RAM에 올라가지 않지만, index·Python overhead·활성값은 추가 공간이 필요하다.
64MiB보다 큰 case는 LRU에 보관하지 않는다. 한 번 읽는 case와 현재 global window는
일시적으로 메모리에 존재한다. 실제 GPU peak와 처리량은 CUDA 실행 전까지 미검증이다.

## Ridge와 모델 초기화

```sh
cmmt-state-training ridge-fit \
  --index /data/run_binding/raw_index.json --root-map /data/root-map.json \
  --config configs/state_training/transformer.json \
  --preflight-report runs/preflight-data.json --output runs/ridge.pth
```

FP64 centered sufficient statistics를 누적하고 `lambda=alpha*trace(G)/D`를 적용한다.
Bias는 정규화하지 않는다. 후보 alpha는 `0,1e-8,1e-6,1e-4,1e-2,1e-1,1`이다.
공간·pointer별 validation MSE로 선택하고 상대 1e-8 이내 동률이면 작은 alpha를 고른다.
선택 후 재적합하지 않는다. FP32 계수·train RMS·후보 점수·충분통계·index identity를 저장한다.
`ridge-fit`도 보고서를 전달하면 중복 전체 검사를 생략한다. 보고서를 생략한 기존 단일
process 명령은 전체 검사를 수행한다. 충분통계 계산 자체에는 train/validation 읽기가 필요하다.
충분통계 구현은 기존 `linear-transfer/regression.py`에서 가져왔으며 sibling 저장소를 import하지 않는다.

비교 설정은 `configs/state_training/`에 있다.

| 설정 | 공간 | Pointer |
|---|---|---|
| `residual_mlp.json` | 선형 연결층 + feed-forward 블록(64→256→64, GELU) | 선형 `A p + c` |
| `transformer.json` | 선형 연결층 + self-attention·feed-forward 블록 | 선형 `A p + c` |
| `residual_mlp_matched.json` | `residual_mlp`의 feed-forward 비율을 14로 넓힌 대조군(공간 파라미터 119,936개, Transformer 모델은 119,872개) | 선형 `A p + c` |

두 모델은 attention 부분층 하나만 다르다. 구조와 크기(파라미터 `residual_mlp` 103,168,
`transformer` 185,664, 공간 MAC 기록당 150,994,944와 337,641,472)는
[`translator_model_design.md`](translator_model_design.md) 1·3·4절에, 바뀐 이유는
[`refactor_decisions.md`](refactor_decisions.md)에 있다. 설정 파일의 `model_options`는
`TranslatorConfig` 필드(`num_layers`, `mlp_ratio`, `num_heads`, `sr_ratio`, `output_dtype`)이며
두 파일 모두 비어 있다(기본값 사용). 세 모델 쌍에 같은 구조를 쓰고, 가중치와 학습 설정만 다르다.

두 모델 모두 ridge artifact의 해로 연결층과 포인터 선형층을 초기화한다. 각 부분층의
마지막 선형층은 0으로 초기화하므로 초기 출력은 FP32 연산 허용오차 내에서 ridge와
일치한다. Ridge의 데이터 identity가 다르면 학습을 거부한다. `train`은 `--ridge-init`이 없으면
오류를 낸다. `cpu-smoke`는 `--ridge-init`이 없으면 선택된 샘플로 ridge 해를 직접 적합한다.

Payload schema는 `cmmt.stitched_translator.v1`이다. 이 브랜치의 구조 변경 전에 만든 payload와
checkpoint(`cmmt.residual_mlp_translator.v2/.v3`, `cmmt.spatial_context_transformer_translator.v1/.v2`)는
읽을 수 없다. 커밋 `913e35b` 또는 `main`에서 읽고, 모든 모델을 새로 학습한다. 새 가중치는
기존 handoff CLI의 `--translator-artifact`로 읽을 수 있다. Streaming ridge에는
`--translator ridge --presence-policy direct`를 사용한다. v2 state-only 캐시만으로
기존 mask 기반 handoff/J&F CLI를 실행할 수 있다는 뜻은 아니다.

## CPU 확인과 상태 평가

```sh
cmmt-state-training cpu-smoke \
  --index /data/run_binding/raw_index.json --root-map /data/root-map.json \
  --config configs/state_training/residual_mlp.json \
  --sample-cases 8 --steps 2 --output runs/cpu-smoke-residual-mlp
```

이 명령은 사용 가능한 train 샘플만으로 CPU forward/backward를 검사한다. Validation
역할을 만들지 않는다. 전체 ridge가 없으면 선택된 train 샘플의 고정 alpha=1e-4 해로
일시 초기화한다. `sample_initialization.pth`에는 `sample_only` identity가 붙으며 공식
학습 초기화로 사용할 수 없다. 보고서는 학습/validation/benchmark 결과가 아니다.
공식 ridge가 있으면 `--ridge-init runs/ridge.pth`로 실제 초기화를 검사할 수 있다.

```sh
cmmt-state-training evaluate \
  --index /data/run_binding/raw_index.json --root-map /data/root-map.json \
  --model residual_mlp --model-artifact runs/residual-mlp/paired_translators.pt \
  --output runs/residual-mlp-state-evaluation.json
```

`evaluate`는 CPU에서 모든 validation record의 상태 오차를 집계한다. 새 schema
(`cmmt.stitched_translator.v1`)의 payload, `paired_translators.pt`, epoch checkpoint,
streaming ridge를 지원한다. 구조 변경 전에 저장한 가중치(예: 1차 학습 epoch 27)는 읽지 못한다. `--model ridge`는
ridge artifact용이다. 원시/정규화 MSE와 source presence·conditioning·공간 사용 여부별
통계를 기록한다. Source/target presence 불일치도 기록한다. J&F를 계산하지 않는다.

평가는 validation case만 읽는다. 공식 index의 train-only 정규화 값을 사용하므로 train
캐시가 마운트되어 있지 않아도 된다. 정규화를 다시 추정하거나 train 파일을 검사하지 않는다.
`--preflight-report runs/preflight-data.json`을 추가하면 해당 보고서의 RMS와 validation
stamp를 대조한다. 정규화가 없는 sample index는 이 보고서가 필수다. Validation 데이터의
target으로 RMS를 추정하는 fallback은 없다.

## CUDA 실행 명령: 준비만 제공, 이번 작업에서 실행하지 않음

```sh
cmmt-state-training train \
  --index /data/run_binding/raw_index.json --root-map /data/root-map.json \
  --config configs/state_training/residual_mlp.json --ridge-init runs/ridge.pth \
  --preflight-report runs/preflight-data-ridge.json \
  --device cuda --output runs/residual-mlp

python -m torch.distributed.run --standalone --nproc-per-node=2 \
  -m vos_memory_inspector.state_training.cli train \
  --index /data/run_binding/raw_index.json --root-map /data/root-map.json \
  --config configs/state_training/residual_mlp.json --ridge-init runs/ridge.pth \
  --preflight-report runs/preflight-data-ridge.json \
  --device cuda --output runs/residual-mlp-ddp
```

Transformer 모델은 `transformer.json`을 사용하며 `--ridge-init`은 똑같이 필요하다
(위 두 명령에서 설정 파일만 바꾼다). 시작 시 기존 보고서와
파일 metadata를 대조하며 전체 해시/tensor 검사는 반복하지 않는다. 기존 run 폴더에
덮어쓰지 않으며, 재개에는 `--resume runs/RUN/last.ckpt`를
명시한다. Index·모델 옵션·정규화·학습 정책·ridge hash·world size가 달라지면
재개를 거부한다. 실행 결정성 profile과 PyTorch 버전도 checkpoint identity에 포함한다.
이 정보가 없는 이전 학습 checkpoint는 새 trainer에서 재개할 수 없으며 새 run이 필요하다.
기존 모델 가중치의 추론·상태 평가는 계속 지원한다. Max epochs와 microbatch/worker tuning은 바꿀 수 있다.

결정적 알고리즘을 강제하고 cuDNN benchmark는 끄며, DDP `broadcast_buffers=False`를
사용한다. CUDA 초기화 전에 `CUBLAS_WORKSPACE_CONFIG=:4096:8`을 설정한다. 미설정이면
CLI가 설정하며, 충돌하는 설정 또는 잘못된 설정으로 이미 초기화한 CUDA는 거부한다.
명시적으로 환경 변수를 지정할 때도 이 값을 사용한다. 이는 실행 정책을 고정하는 장치이며,
GPU 종류·process 수가 다른 실행 사이의 비트 일치를 보장하지 않는다.

통신 timeout 기본값은 기존의 180초를 유지한다. `--distributed-timeout-seconds`로 바꿀 수
있지만 긴 preflight를 기다리는 수단으로 사용하지 않는다.

AdamW lr 3e-4, weight decay 1e-4, FP32, global batch 64, microbatch 32, clip norm 1.0,
seed 7이 기본이다. Bias와 LayerNorm 파라미터에는 weight decay를 적용하지 않는다.
연결층 weight를 포함한 나머지 weight에는 AdamW 정책대로 weight decay를 적용한다. Ridge는 초기값이며
학습 중 고정된 prior는 아니다. 비교 조건 유지를 위해 이번 timeout/평가 수정에서 이 정책을 바꾸지 않았다.
최대 60 epoch, 최소 15 epoch 이후 meaningful improvement 0.2%가 없는 validation 8회로
중단한다. Warmup은 min(epoch update 수,1000), plateau는 factor 0.5 / patience 3 /
relative threshold 0.002 / cooldown 1 / min_lr 1e-5다.

각 global window의 실제 기록 수로 gradient accumulation/DDP 평균을 보정한다. 마지막
batch는 복제하지 않고 기록이 없는 rank는 0 기울기로 collective에 참여한다.
매 epoch 전체 validation의 최솟값을 선택하며 동률이면 이전 epoch다. Checkpoint에는
optimizer·scheduler·rank별 Python/NumPy/Torch RNG와 완료 epoch·step·best payload가 있다.
Mid-epoch exact resume는 지원하지 않는다. `paired_translators.pt`는 선택된 가중치다.

기본 손실은 모든 유효 spatial/pointer 기록의 train RMS 정규화 MSE 합이다.
`training.spatial_loss_records="used"`는 별도 실험 설정이다. Validation 선택·보고는
모든 유효 기록을 유지하므로 train 목적함수는 `training_objective`로 별도 표시한다.

순방향·stride 1·conditioning 1개일 때 다음 프레임의 공간 사용 마스크는 conditioning과
switch s의 s-5…s, pointer 마스크는 conditioning과 s-14…s다. 다른 정책은 보수적으로
전체 기록을 표시한다. 이 정책 계산은 runtime logits 동등성의 증거가 아니므로 실제
번역 생략은 비활성화되어 있다.

1차 0.4926과의 데이터 일치 표시는 원본 index SHA256에 묶는다. 학습 설정까지 달라진
실험을 동일 조건 재현이라고 부르지 않는다. CUDA/NCCL·GPU 메모리·처리량·후속 logits·
J&F 검증은 실제 실행 전까지 완료로 표시하지 않는다.

## 회귀 검사

```sh
PYTHONPATH=src python -m pytest -q
```

CPU Gloo 2-process 검사는 로컬 loopback 통신이 필요하다. macOS에서는 테스트 worker가
`GLOO_SOCKET_IFNAME=lo0` 기본값을 사용한다. CUDA 학습을 실행하는 테스트는 없다.
