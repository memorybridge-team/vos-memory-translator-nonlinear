# vos-memory-translator-nonlinear

SAM 2.1 Small이 switch 시점까지 만든 객체별 memory를 translator로 바꾼 뒤 SAM 2.1 Base+ predictor에 주입해 `switch frame+ 1`부터 이어 추적하는 코드입니다. 

## 코드가 하는 일

한 번의 handoff는 네 단계입니다.

```text
SAM 2 inference_state
        │  export
        ▼
handoff state           spatial_memory [B, O, K, C, H, W]
                        object_pointer [B, O, K, D]
                        frame / slot / conditioning / validity
        │  Translator.translate
        ▼
translated handoff state
        │  inject
        ▼
target inference_state  maskmem_features, target PE, obj_ptr
        │  propagate_in_video(start_frame_idx = switch + 1)
        ▼
future masks → J&F, temporal metrics
```

`handoff state`는 switch 시점까지의 객체별 memory를 translator가 읽고 쓸 수 있게 묶은 텐서다. 축은 batch, object, record다. 번역하는 값은 spatial memory와 object pointer뿐이고, frame 번호·slot·conditioning 여부·valid mask·object id·switch frame은 그대로 복사한다. Target positional encoding은 target memory encoder가 다시 만든다. Source `pred_masks`와 `object_score_logits`는 진단용으로만 남기고 target에는 넣지 않는다.

## 저장소 구조

```text
src/vos_memory_inspector/   라이브러리. CLI도 여기 있다
scripts/                    수집·학습·sweep처럼 여러 모듈을 묶는 실행 스크립트
tests/                      CPU에서 도는 계약·회귀 테스트
configs/                    재현용 설정
manifests/                  GT 픽셀이 없는 switch/object manifest
docs/design/                동결된 I/O·baseline·protocol 계약
docs/architecture/          state assembly map
reports/                    실행 기록. 새 실험 결과가 코드 설명은 아니다
```

패키지 import는 torch 심볼을 지연 로딩합니다. `cmmt-task`처럼 저장소 관리 명령은 GPU 환경이 없어도 동작합니다.

```python
from vos_memory_inspector import ResidualMLPStateTranslator
```

## 패키지 지도

### 상태 계약과 SAM 2 I/O

| 모듈 | 역할 |
|---|---|
| `state_schema.py` | handoff state의 필드, shape 검사, source/target pair가 같은 timeline인지 검사. Python 타입 이름은 `CanonicalState` |
| `sam2_state.py` | predictor `inference_state`에서 handoff state를 뽑고, target의 fresh state에 다시 넣는다. export 전에 producer CUDA device를 동기화한다 |
| `upstream.py` | checkout이 pinned commit인지, private source contract 문자열이 있는지 확인 |
| `inventory.py` | compact output과 memory-attention 입력의 producer/consumer 경로 |

`init_sam2_inference_state_without_warmup`은 pinned `init_state`와 같은 컨테이너를 만들되 frame 0 warmup을 건너뜁니다. injection 대상은 등록 객체와 temporary output이 없는 fresh state여야 합니다.

### Translator

`translators.py`의 translator는 `translate(source)`로 spatial memory와 object pointer만 바꾸고, frame·slot·conditioning·validity는 그대로 복사합니다.

| 클래스 | 하는 일 |
|---|---|
| `DirectCopyTranslator` | grid는 bilinear, channel/pointer 길이는 truncate 또는 zero-pad |
| `RidgeStateTranslator` | spatial·pointer에 대해 closed-form affine/ridge. `to_payload` / `from_payload` |
| `LinearStateTranslator` | component별 `nn.Linear` |
| `ResidualMLPStateTranslator` | hidden MLP. 입출력 shape가 같을 때만 residual. schema `cmmt.residual_mlp_translator.v2` |
| `LearnedComponentPolicyTranslator` | 고른 component만 learned, 나머지는 Direct Copy |

`translator_training.py`의 `state_mse_loss`는 valid record의 spatial MSE와 pointer MSE를 더합니다. `fit_gradient_translator`는 Adam 루프이고, `spatial_samples_per_pair`가 있으면 spatial token만 샘플링합니다. `paired_experiment.py`가 synthetic smoke와 held-out pair 학습을 이 둘 위에 올립니다. Ridge·Linear는 회귀와 과거 보고서 재현용으로 남아 있고, 현재 학습 스크립트의 기본 후보는 residual MLP입니다.

### 캐시

| 모듈 | 파일에 들어가는 것 |
|---|---|
| `case_cache.py` | source/target handoff state, source prefix mask, target oracle future mask, metadata. SHA-256 sidecar |
| `paired_state_cache.py` | 학습에 쓰는 source/target handoff state pair만 |

둘 다 `*.partial`에 쓴 뒤 rename하고, load 시 sidecar checksum을 맞춥니다. `weights_only=False`이므로 신뢰하는 로컬 파일만 읽습니다.

### 평가

| 모듈 | 역할 |
|---|---|
| `metrics.py` | valid record 기준 MSE, cosine, relative error, `handoff_bytes`, CPU/CUDA latency |
| `davis_evaluation.py` | 공식 DAVIS `db_eval_iou` / `db_eval_boundary`로 switch 이후 구간만 채점. GT-visible과 GT-absent를 분리 |
| `temporal_evaluation.py` | target-native 대비 +1/+5/+20, switch shock, identity-loss proxy, recovery |
| `artifacts.py` | oracle/candidate PNG와 GT·native·candidate 비교 캔버스 |
| `baseline_sweep.py` | case 선택, suite 완료 판정, 재개, case 평균 aggregate |

### Manifest와 데이터셋

| 모듈 | 역할 |
|---|---|
| `evaluation_manifest.py` | DAVIS 2017. 영상 단위로 regular quantile과 occlusion/reappearance/area/motion tag를 붙인다. GT는 case를 고르는 데만 쓴다 |
| `mose.py` | MOSEv2. validation은 first-frame annotation만 있으므로 DAVIS event tagger를 타지 않는다 |
| `lvos.py` | LVOS v2. 공식 split metadata의 객체별 frame range 안에서 switch를 고른다 |
| `davis.py` | DAVIS sequence 검증과 trainval 480p download |

### Probe

`runner.py`가 영상 하나를 끝까지 돌리며 `probe.py`의 `StateProbe`와 `attention_hook.py`의 `MemoryAttentionProbe`를 붙입니다. 기록 형식은 `manifest.py`의 JSONL/CSV입니다. `compatibility.py`는 source/target manifest의 shape·dtype을 맞춰 disposition을 붙이며, shape가 같다고 표현이 정렬됐다고 보지 않습니다. `state_inspector.py`와 `hf_cache.py`는 nested state / HF KV cache를 재귀적으로 inventory합니다.

### CLI

인자 파서는 `cli.py`에 있고, console script 이름은 `pyproject.toml`의 `[project.scripts]`입니다.

| 명령 | 함수 | 용도 |
|---|---|---|
| `sam2-memory-probe` | `probe_main` | 한 영상의 compact state와 attention 입력 inventory |
| `sam2-memory-compare` | `compare_main` | 두 manifest의 shape 비교 |
| `cmmt-state-inspect` | `state_inspect_main` | `.pt` 안 텐서 inventory |
| `sam2-roundtrip-smoke` | `roundtrip_main` | 같은 checkpoint export → inject |
| `sam2-prompt-timeline-roundtrip` | `prompt_timeline_roundtrip_main` | 서로 다른 frame의 mask prompt |
| `sam2-correction-roundtrip` | `correction_roundtrip_main` | switch 이전 correction |
| `sam2-repeated-switch-roundtrip` | `repeated_switch_roundtrip_main` | 반복 switch |
| `cmmt-sam2-prepare-case` | `prepare_handoff_case_main` | source prefix와 target oracle를 한 번 계산해 cache |
| `cmmt-sam2-cached-handoff` | `cached_handoff_main` | cache에서 Direct / Ridge / residual MLP 주입 |
| `cmmt-sam2-cached-baseline` | `cached_baseline_main` | reset, last-mask, replay-k, full replay |
| `sam2-direct-handoff-smoke` | `direct_handoff_main` | cache 없이 Small → Base+ Direct Copy |
| `sam2-ridge-handoff-smoke` | `ridge_handoff_main` | 저장된 ridge payload로 handoff |
| `cmmt-davis-build-manifest` | `davis_manifest_main` | DAVIS case manifest |
| `cmmt-mose-build-manifest` | `mose_manifest_main` | MOSEv2 manifest |
| `cmmt-lvos-build-manifest` | `lvos_manifest_main` | LVOS v2 manifest |
| `cmmt-davis-future-eval` | `davis_future_evaluation_main` | 저장된 mask의 partial J&F |
| `cmmt-paired-experiment` | `paired_experiment_main` | pair로 translator 학습·평가. injection은 하지 않음 |
| `cmmt-synthetic-experiment` | `synthetic_experiment_main` | checkpoint 없는 CPU smoke |
| `cmmt-task` | `research_task.main` | 완료 증거 없는 task를 `done`으로 닫지 않는 로컬 기록 |

`research_task.py`는 GitHub Issue URL, acceptance criteria, artifact 경로를 검사합니다. 연구 보드 운영 순서는 [docs/github_task_workflow.md](docs/github_task_workflow.md)에 있습니다.

## 스크립트

`scripts/`는 라이브러리를 조합합니다. 패키지를 설치한 뒤 `python scripts/...`로 실행합니다.

| 스크립트 | 호출하는 것 |
|---|---|
| `prepare_paired_state_dataset.py` | 영상 단위 train/validation split 후 `cmmt-sam2-prepare-case`. 이어서 compact `paired_state_cache`를 쓸 수 있다. 입력 경로는 DAVIS layout |
| `train_nonlinear_from_collection.py` | selection manifest의 train pair로 `run_paired_experiment(..., translator_names=("residual_mlp",))` |
| `evaluate_nonlinear_collection.py` | validation case마다 `run_cached_translator_handoff` 후 DAVIS J&F와 temporal metric. 같은 `translator_evaluation_id`면 skip |
| `run_cached_baseline_suite.py` | 한 cache에 Direct, reset, last-mask, replay-1/2/4, full replay를 돌리고 last-mask == replay-1을 검사 |
| `run_rare_event_baseline_sweep.py` | manifest의 rare-event case를 재개 가능하게 sweep |
| `backfill_temporal_metrics.py` | 이미 있는 `davis.json`만으로 temporal 필드를 다시 계산 |
| `diagnose_same_checkpoint_history.py` | 주입 직후 `maskmem_features` / PE / `obj_ptr`가 native와 같은지 기록 단위로 비교 |
| `build_baseline_suite_gallery.py` | suite PNG로 HTML 갤러리와 mp4 |
| `make_synthetic_video.py` | probe용 작은 JPG 시퀀스와 첫 frame mask |
| `validate_benchmark_data.py` | manifest의 영상·prompt mask가 실제 파일로 열리는지 검사 |
| `build_video_splits.py` | train manifest를 영상 단위 fit/development split으로 나눔 |
| `runpod_bootstrap.sh`, `runpod_preflight.sh`, `runpod_base_plus_roundtrip.sh` | GPU pod에서 SAM 2, checkpoint, self-injection smoke |

## 설치와 테스트

Python 3.10 이상, PyTorch 2.3 이상이 필요합니다. SAM 2 checkpoint와 DAVIS/MOSEv2/LVOS 원본은 Git에 넣지 않습니다.

```bash
git clone https://github.com/memorybridge-team/vos-memory-translator-nonlinear.git
cd vos-memory-translator-nonlinear
python -m pip install -e ".[dev]"
python -m pytest -q
```

`tests/`는 synthetic state와 fixture로 계약만 검사합니다. GPU round-trip은 테스트에 없고 `sam2-roundtrip-smoke`와 `scripts/diagnose_same_checkpoint_history.py`로 확인합니다. pytest가 `vos_memory_inspector`를 못 찾으면 패키지가 설치되지 않은 것입니다. `PYTHONPATH=src`로도 같은 import가 됩니다.

GPU pod에서는 `bash scripts/runpod_bootstrap.sh "$PWD"` 다음 `bash scripts/runpod_preflight.sh "$PWD"`를 실행합니다. 절차는 [docs/runpod_l4_operations.md](docs/runpod_l4_operations.md)에 있습니다.

## 관련 문서

코드 계약을 바꿀 때는 README보다 아래 문서를 먼저 맞춥니다.

- [Small → Base+ I/O 계약](docs/design/small_base_state_io_contract.md) — 어떤 필드를 번역·복사·재생성하는지
- [State assembly map](docs/architecture/cmmt-state-assembly-map.html)
- [Baseline·성공·중단 기준](docs/design/01_scope_baselines_success_stop.md)
- [Benchmark protocol](docs/design/03_benchmark_protocol.md)
- [실험 계획](docs/experimental_plan.md)
