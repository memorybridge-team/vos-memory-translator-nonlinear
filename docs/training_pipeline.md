# Small → Base+ state-supervised 학습 파이프라인

기준일: 2026-09-28. `cmmt.small_to_base_plus.io.v1.1`을 소비한다.
한 방향·한 번의 runtime handoff만 구현하며 continuation은 `switch_frame + 1`에서 시작한다.
이 문서의 CPU synthetic 검증과 실제 SAM 2 GPU gate는 별도로 기록한다.

## 계약 감사

| 항목 | 확인 및 처리 |
|---|---|
| 연속 I/O | `CanonicalState.spatial_memory`, `object_pointer`; 실제 shard에서 `StateSpec`와 dtype을 읽는다. `K`를 7로 가정하지 않는다. |
| 정렬 | object registry, switch, schema, `[B,O,K]`, frame/slot/conditioning/validity 전부 exact 검사. duplicate·prefix 밖 record·nonfinite valid tensor 거부. |
| 비학습 필드 | presence는 diagnostic-only, discrete metadata exact copy, PE는 Target 재생성. `contract_dict()`의 오래된 `calibrate` 표기를 수정했다. |
| 기존 cache 결함 | compact cache가 frame/slot/conditioning/validity를 검사하지 않던 부분을 v1.1 validator에 연결했다. |
| 오래된 경로 | `prepare_paired_state_dataset.py`는 DAVIS 전용이고 `train_nonlinear_from_collection.py`는 residual MLP에 결합돼 있다. 현재 pipeline은 새 CLI를 쓴다. 과거 재현 경로는 보존했다. |
| 문서 충돌 | `AGENTS.md`의 Tiny/DAVIS, `PROJECT_CONTEXT.md`의 Linear 제외는 이번 Small→Base+ 및 learned Linear 요청보다 이전 정책이다. |
| 모델팀 미제공 계약 | nonlinear importable factory, constructor kwargs, forward adapter, 버전/소스, context record batching 정책. 기존 residual MLP를 최종 모델팀 구조로 간주하지 않는다. |

## 구성과 데이터 정책

- `training_collection.py`: pinned SAM 2 checkpoint로 두 모델을 순차 실행하고 switch까지의 snapshot만 수집한다. model parameters는 모두 동결한다.
- `training_storage.py`: 같은 filesystem에 고유 temporary file → flush/fsync → atomic replace → directory fsync. shard/checkpoint를 먼저 저장하고 checksum을 포함한 JSON pointer를 나중에 commit한다.
- `training_data.py`: tensor-only safe `weights_only=True` shard, checksummed manifest, lazy CPU loader, 영상 단위 fit/dev 검사.
- `training_runner.py`: learned Linear 또는 외부 nonlinear factory, valid-record loss, fit-only RMS loss scale, optimizer/RNG/history 포함 checkpoint, epoch 경계 재개.
- `training_rollout.py`: checkpoint → 기존 `inject_sam2_canonical_state` → 다음 frame. warmup 없는 Target 초기화, injection 전/중 과거 backbone call 0 검사.

기본 fit/dev는 MOSEv2 및 LVOS v2 **official train**만 허용한다. official validation,
DAVIS, VOST, PUMaVOS, M³-VOS는 collection plan 및 학습 loader에서 거부한다. 새 dataset의 학습 역할을 추가하려면
별도 프로토콜 변경이 필요하다. `--allow-synthetic`은 명시적 synthetic fixture에만 쓰며 실제 데이터 실험에서는 사용하지 않는다.

`native_history`는 두 모델이 동일 최초 mask/correction timeline에서 각자 추적한 state다.
`controlled_same_mask`는 매 active object/frame에 **같은 명시적 mask**를 두 predictor의
`add_new_mask`에 주는 intervention history다. 빈 mask도 허용한다. 이 버전은 두 native trajectory의
mask만 사후 교체한 것으로 해석하지 않는다. 두 mode는 shard directory, manifest, train run으로
분리되며 controlled checkpoint를 native downstream rollout에 사용하지 않는다.

RGB는 numeric `000000.jpg..` staging view로 제공한다. `build-plan`은 원래 frame stem과 staged
index 대응을 저장한다. LVOS sparse official frame ID는 전체 sorted RGB에서의 index로 정확히 변환한다.
객체별 첫 prompt는 기존 manifest가 지정한 frame을 사용한다. 미래 GT나 event label로 입력을 고르지 않는다.
GT mask는 최초 prompt 및 명시적으로 분리한 controlled prefix intervention에만 사용한다.

## 모델팀 인터페이스

```python
def build_translator(*, source_spec: StateSpec, target_spec: StateSpec, **kwargs) -> torch.nn.Module:
    ...

# nn.Module.forward(source: CanonicalState) -> CanonicalState
# spatial_memory/object_pointer만 학습하여 반환한다.
# source.with_continuous(..., presence_logits=source.presence_logits.clone(),
#     positional_information={"policy": "regenerate_at_target"}) 사용 가능.
```

runner는 discrete/diagnostic copy, target spec, finite output을 검사한다. private feature head에는
의존하지 않는다. factory는 `--factory model_team.adapter:build_translator`와 JSON constructor kwargs로
선택한다. factory 및 Module graph의 class source SHA-256과 state_dict를 저장하며 로드 시 source hash를 검사한다.
`--record-batch-size 4`는 K축 minibatch를 사용하므로 cross-record context를 요구하는 모델팀 구조는
`--record-batch-size 0`으로 전체 history를 받거나 모델팀이 승인한 batching adapter를 제공해야 한다.
Linear와 nonlinear 비교는 같은 manifest·mode·loss scale·batching·seed·학습량 및 downstream 조건을 사용한다.
runtime facade도 저장된 record batching 정책을 사용하고 translated chunk를 CPU로 모은 뒤 Target runtime
정책으로 이동한다. 모델팀의 전체-history context 구조는 명시적으로 batch size 0을 사용한다.

loss는 spatial MSE와 pointer MSE를 각각 valid record에서 평균낸 뒤 fit-only target RMS²로 나누어 더한다.
presence, padding은 loss에 넣지 않는다. 이 RMS는 feature normalization이 아니라 loss scale이다.
dev는 gradient/statistics fitting에 사용하지 않고 checkpoint 선택에만 쓴다. state loss는 VOS 품질 증거가 아니다.

## 로컬 CPU smoke

저장소 root의 PowerShell에서:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install numpy pillow pytest
$env:PYTHONPATH = 'src;.'
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m vos_memory_inspector.training_cli --persistent-root .\artifacts\training synthetic-smoke --output smoke-001
.\.venv\Scripts\python.exe -m vos_memory_inspector.training_cli --persistent-root .\artifacts\training verify --collection smoke-001/collection --allow-synthetic
```

smoke는 one-video overfit → video-disjoint dev → 3+3 epoch resume/연속 6 epoch parity →
저장 checkpoint의 실제 materializer/injector 연결을 수행한다. SAM predictor와 PE module은 synthetic fixture다.
출력 directory는 새 이름이어야 한다. `smoke_report.json`, `collection/manifest.json`, `splits/*.json`,
`overfit-run`, `dev-run`, `uninterrupted-run`에 산출물이 생긴다.

## RunPod 실행 순서

아래는 기존 bootstrap이 완료된 **독립 checkout**에서의 MOSE 소규모 real-GPU 명령이다.
현재 팀원 작업 checkout에 reset/overwrite하지 않는다. mount와 dataset 실제 경로는 환경에 맞춰 지정한다.
GPU image의 CUDA PyTorch를 쓰고, bootstrap inventory와 `pip freeze`를 보존한다.

```bash
export CMMT_VOLUME_ROOT=/workspace
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=7
export PYTHONPATH=src:.
export PYTHON_BIN="$PWD/.venv/bin/python"
export SAM2_REPO="$PWD/.external/sam2"
export DATASET_ROOT="$CMMT_VOLUME_ROOT/CMMT/data/MOSEv2"
export SOURCE_CKPT="CMMT/checkpoints/sam2.1_hiera_small.pt"
export TARGET_CKPT="CMMT/checkpoints/sam2.1_hiera_base_plus.pt"

# 기존 preflight는 checkout 내부 checkpoint를 검사하므로 아래 shared paths를 직접 확인한다.
findmnt "$CMMT_VOLUME_ROOT"
nvidia-smi
sha256sum "$CMMT_VOLUME_ROOT/$SOURCE_CKPT" "$CMMT_VOLUME_ROOT/$TARGET_CKPT"
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
"$PYTHON_BIN" -m pytest -q
mkdir -p "$CMMT_VOLUME_ROOT/cmmt/small-dev"
"$PYTHON_BIN" -m pip freeze > "$CMMT_VOLUME_ROOT/cmmt/small-dev/environment.txt"

"$PYTHON_BIN" -m vos_memory_inspector.training_cli build-plan \
  --manifest manifests/mosev2_train_v1.json --dataset-root "$DATASET_ROOT" \
  --fit-videos manifests/mosev2_train_v1_fit.json \
  --dev-videos manifests/mosev2_train_v1_development.json \
  --videos-per-split 1 --switch-limit 1 \
  --source-checkpoint "$SOURCE_CKPT" --target-checkpoint "$TARGET_CKPT" \
  --output cmmt/small-dev/plan.json

"$PYTHON_BIN" -m vos_memory_inspector.training_cli collect \
  --plan "$CMMT_VOLUME_ROOT/cmmt/small-dev/plan.json" \
  --collection cmmt/small-dev/collection --sam2-repo "$SAM2_REPO" --device cuda --seed 7
"$PYTHON_BIN" -m vos_memory_inspector.training_cli verify --collection cmmt/small-dev/collection
"$PYTHON_BIN" -m vos_memory_inspector.training_cli split \
  --collection cmmt/small-dev/collection --output cmmt/small-dev/splits \
  --fit-videos manifests/mosev2_train_v1_fit.json \
  --dev-videos manifests/mosev2_train_v1_development.json

export FIT_VIDEO="$("$PYTHON_BIN" -c 'import json; print(json.load(open("manifests/mosev2_train_v1_fit.json"))["videos"][0])')"
"$PYTHON_BIN" -m vos_memory_inspector.training_cli overfit-manifest \
  --collection cmmt/small-dev/collection --video-id "$FIT_VIDEO" --output cmmt/small-dev/overfit.json
"$PYTHON_BIN" -m vos_memory_inspector.training_cli train \
  --collection cmmt/small-dev/collection --fit "$CMMT_VOLUME_ROOT/cmmt/small-dev/overfit.json" \
  --overfit --output cmmt/small-dev/linear-overfit --device cuda --epochs 100 --seed 7
"$PYTHON_BIN" -m vos_memory_inspector.training_cli train \
  --collection cmmt/small-dev/collection \
  --fit "$CMMT_VOLUME_ROOT/cmmt/small-dev/splits/native_history.fit.json" \
  --dev "$CMMT_VOLUME_ROOT/cmmt/small-dev/splits/native_history.dev.json" \
  --output cmmt/small-dev/linear-dev --device cuda --epochs 10 --seed 7
"$PYTHON_BIN" -m vos_memory_inspector.training_cli rollout \
  --plan "$CMMT_VOLUME_ROOT/cmmt/small-dev/plan.json" --collection cmmt/small-dev/collection \
  --checkpoint "$CMMT_VOLUME_ROOT/cmmt/small-dev/linear-dev/best.json" \
  --sam2-repo "$SAM2_REPO" --pair-index 0 --output cmmt/small-dev/rollout-linear.json
```

LVOS는 dataset root를 `/workspace/CMMT/data/LVOSv2`로, 세 manifest를 `lvosv2_train_v1*.json`으로 바꿔 별도 collection을 만든다.
전체 collection은 `--videos-per-split`/`--switch-limit` 제한을 제거한다.
`split`에는 기존 frozen fit/development video manifests를 제공한다. 하나의 collection에서 MOSE+LVOS를
모으려면 같은 checkpoint/config의 plans를 case 목록으로 합치고, 각각 frozen 역할을 유지하는 combined
fit/dev video manifest를 명시해야 한다. 현재 `split --fit-videos/--dev-videos`는 단일 dataset manifest를 받는다.

모델팀 nonlinear은 같은 `train` 명령에서 `--factory model_team.adapter:build_translator
--factory-kwargs '{"...": "..."}'`와 새 `--output`을 사용한다. factory와 kwargs가 제공되기 전에는
임의 nonlinear 구조를 최종 결과로 생성하지 않는다.

## 재시작·중단 재개 및 무결성

Pod 재시작 후 동일 persistent volume이 mount됐는지 provider 설정과 `findmnt "$CMMT_VOLUME_ROOT"`로
확인한다. CLI는 디스크의 영속성 자체를 추론하지 않는다. 아래 `verify`는 새 process에서 모든 shard를
CPU로 읽고 SHA-256·size·모델·shape/dtype·record identity를 재검증한다.

```bash
"$PYTHON_BIN" -m vos_memory_inspector.training_cli verify --collection cmmt/small-dev/collection
"$PYTHON_BIN" -m vos_memory_inspector.training_cli collect \
  --plan "$CMMT_VOLUME_ROOT/cmmt/small-dev/plan.json" --collection cmmt/small-dev/collection \
  --sam2-repo "$SAM2_REPO" --seed 7
"$PYTHON_BIN" -m vos_memory_inspector.training_cli train \
  --collection cmmt/small-dev/collection \
  --fit "$CMMT_VOLUME_ROOT/cmmt/small-dev/splits/native_history.fit.json" \
  --dev "$CMMT_VOLUME_ROOT/cmmt/small-dev/splits/native_history.dev.json" \
  --output cmmt/small-dev/linear-dev --device cuda --epochs 20 --seed 7 \
  --resume "$CMMT_VOLUME_ROOT/cmmt/small-dev/linear-dev/latest.json"
```

완료 shard도 checksum/read 검증 후에만 skip한다. mismatch는 수집과 학습을 중지한다.
manifest에 없는 orphan shard/partial file은 다음 수집에서 다시 계산할 수 있다.
SIGKILL은 `.writer.lock`을 남길 수 있다. 해당 writer가 실제로 종료됐는지 확인한 뒤 그 directory의
lock만 운영자가 제거한다. CLI는 active/stale lock을 임의로 추정해 지우지 않는다.
checkpoint는 optimizer, Python/NumPy/Torch/CUDA RNG, loss scales, data/factory/source hashes,
환경과 Git provenance, metrics history를 포함한다. epoch 도중 중단되면 마지막 완료 epoch에서 재개하고
중단 epoch를 다시 수행한다. same run의 latest만 resume한다. epochs 상한 외의 config/data/code/environment
변경은 거부하므로 업데이트된 코드/환경은 새 run으로 기록한다.

`rollout`은 Target-native 대비 후속 mask agreement를 출력한다. 현재 이는 engineering 비교이며 GT J&F나
공식 benchmark가 아니다. GT rollout evaluator 연결·실제 dev rollout gate 완료 뒤에 attention/logit
distillation을 추가한다. 향후 Target operations로 gradient를 보낼 때 `freeze_model`만 적용하고,
injected state를 detach하거나 Target forward 전체에 `no_grad`/`inference_mode`를 적용하지 않는다.
pinned predictor의 inference decorator를 그대로 사용하는 경로는 distillation용 differentiable API가 아니다.

## 비용 및 남은 gate

`manifest.resources`는 collection wall time/누적 shard bytes, epoch metrics는 wall time/peak CUDA allocation을
기록한다. 현재 측정한 CPU synthetic fixture는 source/target 공간이 `C=4,H=W=3,D=6`인 테스트 입력이다.
실제 SAM 2 storage나 GPU 비용으로 외삽하지 않는다. 실측 JSON은 실행 보고서에 보존한다.
실제 GPU wall time·VRAM·volume restart 증거·요금은 실행 전 미측정이다.

사용자가 2026-09-28 전달한 팀 정보에서 Network Volume은 `KNSW_DATASET`, mount는 `/workspace`,
프로젝트는 `/workspace/CMMT`, 데이터는 `/workspace/CMMT/data/{MOSEv2,LVOSv2}`,
checkpoint root는 `/workspace/CMMT/checkpoints`다. 실제 Pod host/port와 각 파일의 현재 존재/hash는
미확인이다. 팀원은 개인 public SSH key만 등록하며 private key를 공유하지 않는다.
실제 gate에 필요한 것은 활성 Pod endpoint/SSH, 해당 파일의 원격 확인, pinned SAM 2 checkout,
모델팀 nonlinear factory이다. 사용자가 nonlinear 구조는 아직 제작 전이라고 확인했다.
one-video real overfit, video-disjoint real dev 학습과 checkpoint runtime rollout을 통과하기 전에는
해당 milestone이나 VOS 개선을 완료로 표시하지 않는다. 데이터·states·checkpoint·raw logs는 Git 밖에 두고
코드·테스트·lightweight provenance만 commit한다.
