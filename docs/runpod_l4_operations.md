# RunPod L4 운영 및 첫 검증

## 고정 환경

2026-09-19 사용자가 선택한 Pod는 NVIDIA L4 1장(24 GB VRAM), vCPU 16개, RAM 62 GB다. Container disk는 20 GB이고 Network Volume 200 GB가 `/workspace`에 mount되어 있으며, 현재 24 GB를 사용한다. 프로젝트 코드·데이터·checkpoint·실험 산출물은 Container disk가 아니라 `/workspace` 아래에 둔다.

권장 프로젝트 경로는 다음과 같다.

```text
/workspace/vos-memory-translator-nonlinear/  # Git checkout과 .venv
/workspace/vos-memory-translator-nonlinear/data/      # DAVIS/MOSEv2/LVOS v2
/workspace/vos-memory-translator-nonlinear/checkpoints/
/workspace/vos-memory-translator-nonlinear/outputs/   # raw cache, logs, run status
```

GitHub에는 코드·설정·작은 Markdown/JSON·선별 gallery만 push한다. dataset, checkpoint, raw state cache, 전체 frame dump는 Network Volume에만 둔다.

## 첫 실행 순서

```bash
cd /workspace/vos-memory-translator-nonlinear
git fetch origin
git switch codex/project-board-workflow
git pull --ff-only

bash scripts/runpod_bootstrap.sh "$PWD"
bash scripts/runpod_preflight.sh "$PWD"
```

bootstrap은 공식 SAM 2 revision, SAM 2.1 Small/Base+ checkpoint, Python package를 준비하고 CUDA·checkpoint SHA-256·GPU 정보를 `outputs/inventory/`에 기록한다. preflight는 `/workspace` mount, 40 GiB 이상의 여유 공간, checkpoint와 venv 존재를 검사한다.

## 첫 GPU gate: Base+ same-checkpoint round-trip

DAVIS가 `/workspace/vos-memory-translator-nonlinear/data/DAVIS`에 있고 `blackswan`, object 1, switch frame 10을 사용한다면 다음을 실행한다.

```bash
bash scripts/runpod_base_plus_roundtrip.sh \
  /workspace/vos-memory-translator-nonlinear \
  /workspace/vos-memory-translator-nonlinear/data/DAVIS \
  blackswan 1 10
```

성공 조건은 다음 세 가지다.

1. `report.json`의 `mean_binary_iou >= 0.999`
2. `max_abs_error <= 1e-4`
3. `backbone_calls_during_injection = 0`

이 조건은 translator 성능이 아니라 **Base+ 자신의 state를 꺼냈다 넣는 통로가 정상인지** 확인한다. 실패하면 paired-state 수집이나 nonlinear 학습을 시작하지 않는다.

## 장시간 실행 확인

각 실행 폴더의 `status.txt`가 가장 먼저 볼 파일이다.

```bash
cat outputs/roundtrip/<RUN_ID>/status.txt
tail -n 30 outputs/roundtrip/<RUN_ID>/stdout.log
nvidia-smi
```

- `state=running`: 실행 중이다. GPU utilization과 log 갱신 시각을 확인한다.
- `state=completed`: 종료 코드와 acceptance gate를 통과했다.
- `state=failed`: `stdout.log` 마지막 부분과 report를 보존하고 다음 작업 전에 원인을 분리한다.

이후 paired-state 수집·학습은 같은 방식으로 `status`, log, checksum manifest를 남기는 background job으로 실행한다. 장시간 job이 정상 실행 중이고 독립적으로 판단할 일이 없다면, Codex는 polling하지 않고 사용자가 완료 여부를 확인한 뒤 재개한다.
