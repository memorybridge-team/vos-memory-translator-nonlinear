# LVOS paired-state → 학습 → no-replay 평가 운영 가이드

- 날짜: **2026-10-03 KST**
- 상태: **READY_AFTER_INPUTS**. 로컬 CPU 구현 검증과 실제 checkpoint GPU 검증은 별개입니다.
- 작업 branch: `feat/lvos-training-gates`; 통합 기준 `4a1bb1b06ec8d676184a8f6e770550e27aafacbd`.
- 모델 소스: `746ea3e7d84c366c2d7ac06159e90a1f684bca56`; benchmark 소스: `bcf0a0f6a36c4129d487e5e58e151468d7ca714b`.
- 실제 수행한 검사와 최종 로컬 commit: `reports/tasks/07_paired_training/LVOS_PIPELINE_REPORT_2026-10-03.md` 및 공유용 `LVOS_FINAL_HANDOFF_2026-10-03.json` 참조.
- 아직 필요한 입력: 현재 SSH endpoint/접근 승인, 비어 있는 GPU UUID, 시간·요금 한도, 완료 cache와 generating inventory, checkpoint SHA, 모델 config 승인, metric·export adapter 승인, 고정 monitor 영상 목록.

이 문서는 명령을 설명합니다. 오늘 원격 작업·유료 실행·Pod 변경·원격 push/PR은 수행하지 않았습니다. 기존 수집 작업과 `/workspace/CMMT` 환경을 유지하십시오. 명령의 `REQUIRED_*`는 **미확인 필수 입력**이며 실행 전에 바꿉니다.

### 빠른 순서 (첫 실행)

1. §2의 CPU 테스트와 smoke 결과를 확인합니다.
2. §3의 SSH·PID/GPU/mount 읽기 전용 점검을 합니다.
3. §4의 새 checkout/interpreter가 기존 worker와 분리됐는지 확인합니다.
4. §5의 완료 inventory → 전체 audit/view를 만듭니다. 실제 cache가 없으면 여기서 BLOCKED입니다.
5. §6의 모델/metric 승인과 monitor 동결 → epoch-0 checkpoint → real-pair G2/G4 및 one-video overfit을 진행합니다.
6. §7의 identity/Direct Copy handoff·dev 평가 후 G0–G6를 확인합니다.
7. 모든 real gate PASS 후에만 §8의 profile/A/B를 시작합니다. Eval worker는 별도 output을 사용합니다.
8. §10의 complete full-dev shortlist만 export합니다. 중단·재개·과금 확인은 §9를 따릅니다.

## 1. 현재 계약과 입력 표

Paired state는 같은 영상·등록 객체·prompt history·switch의 Small/Base+ 내부 상태 쌍입니다. 이번 데이터는 native-history, 독립 `O=1`입니다. Source/target의 object/frame/slot/conditioning/validity가 정확히 대응해야 합니다. Spatial `[B,O,K,64,64,64]`와 pointer `[B,O,K,256]`만 학습합니다. K는 실행에서 관측하며 7로 고정하지 않습니다. Presence와 mask는 진단용이며 target history에 주입하지 않습니다. Target positional encoding은 Base+에서 재생성합니다.

기존 LVOS **official train** membership에서 fit/development를 읽습니다. Official validation과 외부 benchmark는 normalization·학습·early stopping·checkpoint 선택에 사용하지 않습니다. Sparse official ID와 연속 runtime index는 다릅니다. Target는 switch `t`까지의 encoder를 실행하지 않고 `t+1`부터 이어갑니다.

|입력|현재 근거/설정|담당자·확인 방법|
|---|---|---|
|SSH host/port|미확인, 내일 제공 예정|Pod 운영자; 현재 Dashboard 값|
|Topology/기존 workers|실측 미확인; 이전 요청에는 8 GPU 수집 중으로 기술|수집 담당자; PID·cwd·UUID·session 읽기 전용 확인|
|Persistent mount|사용자가 `/workspace`로 제공; 이 작업에서 원격 mount 검증 안 함|Pod 운영자; `findmnt`|
|기존 프로젝트|사용자 제공 `/workspace/CMMT`|수집 담당자; 기존 interpreter/install path 보존|
|LVOS RGB/GT root|사용자 제공 `/workspace/CMMT/data/LVOSv2/extracted/`; 세부 train/JPEG/PNG 경로 미확인|데이터 담당자; `configs/lvos_runtime.template.json`의 영상별 실제 디렉터리 입력|
|완료 native cache|사용자 제공 `/workspace/CMMT-task07-artifacts/production/lvos/cache/{fit,development}/`; 실제 namespace/inventory 미확인|수집 담당자; 완료 status·conditions·checksum/marker 제공|
|Checkpoint root|사용자 제공 `/workspace/CMMT/checkpoints`; Small/Base+ 파일명·SHA 미확인|모델/Pod 담당자|
|SAM2 checkout|경로 미확인; pin `2b90b9f5ceec907a1c18123530e92e794ad901a4`|환경 담당자; HEAD와 tracked source 확인|
|최종 translator|모델팀 `base` 원본 구조 이관; 최종 preset/config 승인자 미입력|모델팀(용재, 담당 확인 필요); `model-lock.json` 실제 승인|
|Primary metric|고정 benchmark 함수 확인; mean video retention의 25/50/75 평균(%)|Benchmark 팀; 계약 승인·Full Replay 결과 제공|
|Monitor|12개 dev 영상, 기존 50% switch, 최대 64 runtime 미래 frame의 제안; 아직 동결 안 함|연구/Benchmark 담당자; score 전에 영상·GT coverage 승인|
|Export adapter|`TransformerStateTranslator.from_payload` strict loader 구현; Benchmark adapter 수용 승인 미확인|모델/Benchmark 팀; 실제 호출 규약 승인|
|시간·요금|미확인. 기존 개인 150,000 KRW를 이번 팀 작업 한도로 적용하지 않음|예산/Pod 운영자; 초 단위 cap, 실제 Pod/GPU 요금|

아래 경로는 **새 출력 배치 제안**입니다. 기존에 쓰는지 운영자가 확인한 후 사용합니다.

|이름|제안 경로|
|---|---|
|격리 checkout|`/workspace/CMMT-lvos-isolated`|
|입력 JSON|`/workspace/CMMT-training-inputs/lvos`|
|Training view|`/workspace/CMMT-training-views/lvos/full-v1`|
|독립 runs|`/workspace/CMMT-training-runs/lvos-only`|

State-only cache에는 미래 oracle mask나 GT J&F가 없습니다. 단일 객체 cache는 joint multi-object equivalence를 입증하지 않습니다.

## 2. 로컬 CPU 확인 — 검증된 명령

**위치: 로컬 PowerShell, 이 문서의 작업 저장소.** 명령별 output이 이미 있으면 새로운 이름을 사용합니다. `cpu-test`는 pytest log/JUnit/report를 만들며 synthetic 검사입니다. 실패 시 log의 첫 reason을 수정하고 GPU로 넘어가지 않습니다.

```powershell
Set-Location C:\Users\SAMSUNG\Documents\Codex\2026-09-27\you-are-the-implementation-workspace-for\work\repo
```

```powershell
$env:PYTHONPATH='src;.'
```

```powershell
$env:PYTHONUTF8='1'
```

```powershell
.\.venv\Scripts\python.exe -m vos_memory_inspector.lvos_cli --help
```

```powershell
.\.venv\Scripts\python.exe -m vos_memory_inspector.lvos_cli cpu-test --output C:/Users/SAMSUNG/Documents/Codex/lvos_cpu_next_evidence
```

```powershell
.\.venv\Scripts\python.exe scripts/lvos_cpu_smoke.py --output ../../outputs/lvos_cpu_next_smoke
```

성공: exit 0, `report.json`의 exit_code=0, smoke의 `summary.json`에 `execution_kind=cpu_synthetic`. `snapshot/`, `overfit/`, `dev/`, `draft.md`가 생깁니다. 실제 weights/GPU/VOS 성능 PASS로 해석하지 않습니다.

## 3. SSH 후 첫 읽기 전용 점검 — 원격 실행은 대기

**위치: 로컬 PowerShell.** 현재 endpoint와 개인 key 경로를 입력합니다. Private key는 공유하지 않습니다. 실패 시 운영자가 Dashboard·public key 등록을 확인합니다.

```powershell
ssh root@REQUIRED_CURRENT_HOST -p REQUIRED_CURRENT_PORT -i REQUIRED_PERSONAL_PRIVATE_KEY_PATH
```

**위치: RunPod bash.** 한 명령씩 실행하고 결과를 운영 담당자에게 전달합니다. 기존 PID/session의 code revision과 환경을 확인하기 전에는 설치하거나 job을 변경하지 않습니다.

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv
```

```bash
nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name,used_memory --format=csv
```

```bash
ps -eo pid,ppid,lstart,etime,args
```

```bash
readlink -f /proc/REQUIRED_WORKER_PID/cwd
```

```bash
readlink -f /proc/REQUIRED_WORKER_PID/exe
```

```bash
findmnt -T /workspace
```

```bash
df -h /workspace /dev/shm
```

성공: UUID/PID/기존 interpreter/mount/free space가 확인됨. 불일치·공유 volume 포화 시 GPU 작업을 시작하지 말고 운영자에게 결과를 공유합니다. 진행 중 수집에 경쟁하는 전체 cache SHA scan은 이 단계에서 하지 않습니다. Source ZIP은 실제 worker 실행 revision 증거가 아닙니다.

## 4. 격리 코드·환경 — 운영자 승인 후 준비

**위치: RunPod bash.** 검토된 로컬 commit/source artifact를 운영자가 **새 checkout**에 배치한 뒤 사용합니다. 이 작업의 GitHub 게시 명령은 없습니다. 기존 수집 환경에서 `git pull` 또는 `pip install -e`를 실행하지 않습니다.

```bash
cd /workspace/CMMT-lvos-isolated
```

```bash
git rev-parse HEAD
```

```bash
git status --short
```

```bash
python3 -m venv .venv
```

```bash
.venv/bin/python -m pip install -e '.[dev,lvos]'
```

```bash
.venv/bin/python -m vos_memory_inspector.lvos_cli --help
```

```bash
.venv/bin/python -m vos_memory_inspector.lvos_cli cpu-test --output /workspace/CMMT-training-inputs/lvos/cpu-suite
```

성공: 검토한 revision과 HEAD 일치, 새 interpreter에서 help와 CPU test report 표시. G6에는 이 host에서 생성한 `cpu-suite/report.json`을 입력합니다. Windows report의 절대 경로를 Pod에 그대로 사용하지 않습니다. Code/test SHA가 변한 report는 거절됩니다. `ModuleNotFoundError`이면 새 interpreter/설치 경로를 확인합니다. 기존 worker의 parent가 새 CLI를 호출할 경로를 사용하지 않는지 운영자가 확인합니다. 실제 SAM2 dependency 설치와 CUDA/PyTorch 호환성은 Pod 환경 담당자의 gate 입력입니다.

## 5. 완료 데이터 audit → immutable view — CPU 작업

**위치: RunPod bash, 격리 checkout.** `selection.json`은 기존 source/fit/dev manifest로 만든 전체 frozen selection입니다. 새로 분할하지 않습니다. 아래 plan 명령은 기존 CLI이며, **새 파일**에만 씁니다.

```bash
.venv/bin/python -m vos_memory_inspector.paired_collection_cli plan --source-manifest manifests/lvosv2_train_v1.json --fit-split manifests/lvosv2_train_v1_fit.json --development-split manifests/lvosv2_train_v1_development.json --output /workspace/CMMT-training-inputs/lvos/selection.json
```

수집 담당자가 `/workspace/CMMT-training-inputs/lvos/requests.json`을 준비합니다. 배열 항목은 아래 schema이며, `expected_generating`에는 **해당 완료 파일의 검증된 conditions JSON 전체**를 넣습니다. 아직 자동 inventory 생성 helper는 구현하지 않았습니다. 정확한 requirement: 완료 worker status·case binding과 conditions를 묶고 case_id/namespace/digest를 검증하여 이 배열을 생성해야 합니다. 현재는 담당자가 명시적 inventory를 제공합니다. 경로나 checkpoint provenance를 추정하지 않습니다.

```json
[
  {
    "path": "REQUIRED_ACTUAL_COMPLETED_CACHE_PT",
    "case_id": "train:0ClBYzYm:obj1:switch1221",
    "sha256": "REQUIRED_ACTUAL_CACHE_SHA256",
    "expected_generating": null
  }
]
```

`null`을 실행 가능한 generating 계약으로 취급하지 않습니다. Legacy pickle는 팀이 신뢰한 원본에 한해서 아래 flag로 열립니다. 완료 marker 또는 reviewed immutable run evidence, checksum, 안정 구간이 없으면 중단합니다. 원본은 변경하지 않습니다.

```bash
.venv/bin/python -m vos_memory_inspector.lvos_cli audit --requests /workspace/CMMT-training-inputs/lvos/requests.json --selection /workspace/CMMT-training-inputs/lvos/selection.json --output /workspace/CMMT-training-views/lvos/full-v1 --trusted-team-legacy --stable-seconds 60 --max-shard-mib 256
```

성공: `audit.json`의 completed=expected, rejected/unknown=0, zero video overlap, `snapshot.json` state=`ready`, `views/*.pt`와 각 completion marker. Tensor payload는 최대 256MiB이며 archive header overhead가 추가됩니다. 모든 valid record를 유지하고 shared prefix 중복 수만 기록합니다. 학습에서는 완료 **case만 shuffle**하며 내부 시간/slot 순서를 유지합니다. Case가 큰 경우 bounded shard로 나누고 같은 worker가 그 case의 모든 shard를 순서대로 읽습니다.

실패: `audit.json`의 case_id/reason을 공유합니다. 누락/불완전/손상/외부 split/정렬 오류를 건너뛰지 않습니다. 실패 view를 재사용·덮어쓰기하지 않고, 입력을 수정한 뒤 새 view namespace에 재실행합니다. Serialization-only 변경은 CPU import/repack 대상입니다. 잘못된 prompt frame은 선택 재수집 근거이며 이번 학습 작업에서 원본을 수정하지 않습니다. Unknown checkpoint provenance는 추가 근거가 필요합니다.

**재시작 후 같은 view의 CPU 읽기 확인:** 수집/원본 writer가 종료하고 안정 상태임을 확인한 후 실행합니다.

```bash
.venv/bin/python -m vos_memory_inspector.lvos_cli verify --snapshot /workspace/CMMT-training-views/lvos/full-v1
```

성공: exit 0. 시작 시 전체 SHA를 검사하고 epoch에서는 immutable stat/manifest/marker 변화를 검출합니다. Source/view가 바뀌면 snapshot을 무효화합니다. `weights_only=True` view를 사용하며 record마다 큰 raw cache를 다시 읽지 않습니다.

## 6. 계약 동결 → 실제 pair → 짧은 학습 → checkpoint

### 6.1 모델·metric·monitor 입력

**위치: RunPod bash, CPU.** 모델 승인자를 임의로 쓰지 않습니다. 다음은 **미승인 proposal**을 만드는 명령입니다. 모델팀이 생성된 complete config/source SHA를 검토하고 실제 승인자를 기록해야 합니다.

```bash
.venv/bin/python -m vos_memory_inspector.lvos_cli freeze-model --output /workspace/CMMT-training-inputs/lvos/model-proposal.json
```

승인 후 별도 `model-lock.json`은 `freeze-model --approved-by REQUIRED_ACTUAL_APPROVER`로 생성합니다. 기본 구조를 바꾸면 `FROZEN_ARCHITECTURE`로 중단합니다. A/B는 lr만 다르며 architecture/seed/data는 같습니다.

```bash
.venv/bin/python -m vos_memory_inspector.lvos_cli metric-template --benchmark-root REQUIRED_PINNED_BENCHMARK_CHECKOUT --output /workspace/CMMT-training-inputs/lvos/metric-contract.json
```

성공: JSON 생성. **생성은 metric/export 승인 PASS가 아닙니다.** Benchmark 팀이 primary aggregation, visible/undefined/zero-replay 규칙, monitor [0,1] 점수, export_loader와 승인자를 확인합니다. `approved_by`/`export_approved_by`가 비어 있으면 promotion이 막힙니다. Benchmark repo의 Full Replay 생성 코드가 없어 Job C는 BLOCKED입니다.

`configs/lvos_runtime.template.json`을 복사해 실제 영상별 RGB/GT와 SAM2/checkpoint 경로, 승인자를 채웁니다. 경로를 확인한 `runtime.json`을 입력하고, **score를 보기 전에** 12개 영상 ID를 모두 반복 지정합니다.

```bash
.venv/bin/python -m vos_memory_inspector.lvos_cli freeze-protocol --snapshot /workspace/CMMT-training-views/lvos/full-v1 --source-manifest manifests/lvosv2_train_v1.json --runtime /workspace/CMMT-training-inputs/lvos/runtime.json --scope monitor --video-id REQUIRED_DEV_VIDEO_01 --video-id REQUIRED_DEV_VIDEO_02 --video-id REQUIRED_DEV_VIDEO_03 --video-id REQUIRED_DEV_VIDEO_04 --video-id REQUIRED_DEV_VIDEO_05 --video-id REQUIRED_DEV_VIDEO_06 --video-id REQUIRED_DEV_VIDEO_07 --video-id REQUIRED_DEV_VIDEO_08 --video-id REQUIRED_DEV_VIDEO_09 --video-id REQUIRED_DEV_VIDEO_10 --video-id REQUIRED_DEV_VIDEO_11 --video-id REQUIRED_DEV_VIDEO_12 --output /workspace/CMMT-training-inputs/lvos/monitor.json
```

성공: sparse official/runtime map과 GT-visible coverage를 확인한 immutable protocol. `INSUFFICIENT_SCORED_FRAMES`이면 운영/연구 담당자가 score 이전의 영상·coverage 설계를 검토합니다. 낮은 score를 이유로 영상을 교체하거나 window를 늘리지 않습니다.

### 6.2 Gate용 epoch-0 checkpoint — CPU

```bash
.venv/bin/python -m vos_memory_inspector.lvos_cli initialize --snapshot /workspace/CMMT-training-views/lvos/full-v1 --model-lock /workspace/CMMT-training-inputs/lvos/model-lock.json --config configs/lvos_job_a.json --output /workspace/CMMT-training-runs/lvos-only/epoch-zero --device cpu
```

성공: `INITIALIZED.json`, fit-only RMS/config/provenance, `checkpoints/epoch-00000.ckpt`와 completion marker. 아직 학습한 후보가 아닙니다. 이 파일로 target handoff gate를 실행할 수 있습니다.

### 6.3 실제 G2/G4 및 one-video overfit — **별도 GPU 승인 후 대기 명령**

재현 가능한 CUDA matmul을 위한 환경 설정입니다. 승인된 새 작업용 shell에만 적용합니다. 실제 GPU 동작은 아직 검증 전입니다.

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

아래 `REQUIRED_APPROVED_SECONDS`는 실제 허용 초로 바꿉니다. `CUDA_VISIBLE_DEVICES`에 확인된 **free UUID 하나**를 지정하면 process 내부 device는 `cuda:0`입니다. `cuda:7`로 쓰지 않습니다. G5 입력이 없는 첫 gates 결과에는 BLOCKED가 포함되며 unattended 학습은 승인되지 않습니다.

```bash
CUDA_VISIBLE_DEVICES=REQUIRED_FREE_GPU_UUID .venv/bin/python -m vos_memory_inspector.lvos_cli gates --snapshot /workspace/CMMT-training-views/lvos/full-v1 --model-lock /workspace/CMMT-training-inputs/lvos/model-lock.json --metric-contract /workspace/CMMT-training-inputs/lvos/metric-contract.json --cpu-test-report REQUIRED_PASSED_CPU_REPORT_JSON --device cuda:0 --execute-approved --max-wall-seconds REQUIRED_APPROVED_SECONDS --output /workspace/CMMT-training-inputs/lvos/pair-gates
```

성공 기준: G1 실제 전체 audit, G2 real-pair full-record forward/backward, G4 checkpoint/output/update/optimizer/scheduler/RNG parity의 실제 근거 파일. G3는 CPU reference, G6는 실행한 CPU safety suite입니다. G2 loss 감소는 연결성 검사입니다. 실패하면 해당 case/reason/gradient/delta log를 공유합니다.

One-video overfit에는 audit 때 **사전 지정한 fit 영상의 모든 pilot case IDs**를 `--pilot-case-id`로 반복해 새 snapshot을 만듭니다. 원본 full selection을 바꾸지 않습니다. 아래 `REQUIRED_ONE_VIDEO_PILOT_VIEW`는 그 출력입니다.

```bash
CUDA_VISIBLE_DEVICES=REQUIRED_FREE_GPU_UUID .venv/bin/python -m vos_memory_inspector.lvos_cli overfit --snapshot REQUIRED_ONE_VIDEO_PILOT_VIEW --model-lock /workspace/CMMT-training-inputs/lvos/model-lock.json --config configs/lvos_job_a.json --output /workspace/CMMT-training-runs/lvos-only/overfit-real --device cuda:0 --execute-approved --max-wall-seconds REQUIRED_APPROVED_SECONDS --max-micro-iterations 20
```

성공: finite loss/gradients와 checkpoint 또는 명시된 partial `STOPPED.json`. `max_micro_iterations`가 epoch 중간이면 완성 epoch로 세지 않습니다. OOM이면 해당 run을 보존하고 동일 architecture로 microbatch를 줄이는 **진단 config**를 별도로 기록합니다. 본 A/B 설정 변경은 연구 담당자 검토 대상입니다.

## 7. 실제 handoff → dev evaluator → G5

**위치: RunPod bash, 승인된 GPU. 이 문서 작성 중 실행하지 않았습니다.** Epoch-0 identity와 Direct Copy를 같은 frozen monitor에서 각각 실행합니다. Whole rollout은 runtime frame을 순서대로 처리하고 GT는 frozen sparse PNG만 scoring에 사용합니다.

```bash
CUDA_VISIBLE_DEVICES=REQUIRED_FREE_GPU_UUID .venv/bin/python -m vos_memory_inspector.lvos_cli evaluate --snapshot /workspace/CMMT-training-views/lvos/full-v1 --protocol /workspace/CMMT-training-inputs/lvos/monitor.json --checkpoint /workspace/CMMT-training-runs/lvos-only/epoch-zero/checkpoints/epoch-00000.ckpt --metric-contract /workspace/CMMT-training-inputs/lvos/metric-contract.json --benchmark-root REQUIRED_PINNED_BENCHMARK_CHECKOUT --method learned --output /workspace/CMMT-training-runs/lvos-only/epoch-zero/identity-worker-0 --device cuda:0 --execute-approved --max-wall-seconds REQUIRED_APPROVED_SECONDS
```

```bash
CUDA_VISIBLE_DEVICES=REQUIRED_FREE_GPU_UUID .venv/bin/python -m vos_memory_inspector.lvos_cli evaluate --snapshot /workspace/CMMT-training-views/lvos/full-v1 --protocol /workspace/CMMT-training-inputs/lvos/monitor.json --checkpoint /workspace/CMMT-training-runs/lvos-only/epoch-zero/checkpoints/epoch-00000.ckpt --metric-contract /workspace/CMMT-training-inputs/lvos/metric-contract.json --benchmark-root REQUIRED_PINNED_BENCHMARK_CHECKOUT --method direct_copy --output /workspace/CMMT-training-runs/lvos-only/epoch-zero/direct-worker-0 --device cuda:0 --execute-approved --max-wall-seconds REQUIRED_APPROVED_SECONDS
```

성공: case별 실제 encoder frame IDs가 전부 `>t`, 처음 continuation=`t+1`, expected frame/object/GT coverage 일치, `result.json`/`READY.json`. `cases/*.json`은 완료된 case 근거이며 전체 READY와 다릅니다. 낮은 J&F도 coverage와 no-replay 조건이 맞으면 engineering gate를 통과할 수 있습니다. Native mask agreement를 GT J&F로 표현하지 않습니다.

단일 worker도 먼저 merge해 완전한 protocol임을 확인합니다. 아래는 CPU 명령입니다. Direct Copy에도 같은 방식으로 별도 출력 폴더를 사용합니다.

```bash
.venv/bin/python -m vos_memory_inspector.lvos_cli merge-eval --protocol /workspace/CMMT-training-inputs/lvos/monitor.json --metric-contract /workspace/CMMT-training-inputs/lvos/metric-contract.json --benchmark-root REQUIRED_PINNED_BENCHMARK_CHECKOUT --result /workspace/CMMT-training-runs/lvos-only/epoch-zero/identity-worker-0/result.json --output /workspace/CMMT-training-runs/lvos-only/epoch-zero/identity-merged
```

마지막 gates 명령에 `--handoff-result .../identity-merged/result.json`과 `--direct-copy-result REQUIRED_DIRECT_MERGED_RESULT_JSON`을 추가하고 **새** `real-gates` output으로 실행합니다. G0–G6가 모두 PASS이고 근거 SHA/model/data/code digest가 현재와 같아야 main training을 허용합니다. CPU fake predictor는 G5 실제 GPU PASS가 아닙니다.

## 8. Profile → A/B 독립 학습/평가

**GPU 승인 후:** 대표 snapshot의 100–300 micro-iterations, real dev rollout의 cold/warm I/O와 records/sec를 측정합니다. 아래 100은 profile 상한이며 dataset epoch를 완성했다는 뜻이 아닙니다.

```bash
CUDA_VISIBLE_DEVICES=REQUIRED_FREE_GPU_UUID .venv/bin/python -m vos_memory_inspector.lvos_cli profile --snapshot /workspace/CMMT-training-views/lvos/full-v1 --model-lock /workspace/CMMT-training-inputs/lvos/model-lock.json --config configs/lvos_job_a.json --output /workspace/CMMT-training-runs/lvos-only/profile-100 --device cuda:0 --execute-approved --max-wall-seconds REQUIRED_APPROVED_SECONDS --max-micro-iterations 100
```

GPU profile 전에는 GPU ETA·속도·요금을 측정값으로 쓰지 않습니다. FP32 v1이며 AMP/DDP/augmentation은 구현 범위 밖입니다. DataLoader는 workers=4/prefetch=2/persistent/pin, CPU 검사는 workers=0입니다. Case 단위 정렬과 전체 valid coverage를 유지합니다.

`configs/lvos_jobs.template.json`을 복사하여 host/UUID/경로/초 제한을 채우고 `jobs.json`으로 저장합니다. 실행 계획 생성 자체는 CPU이며 job을 시작하지 않습니다. Duplicate job/device/output namespace를 거절합니다. 여러 Pod면 host를 각 Pod로 입력하고 각 명령을 그 Pod에서 실행합니다. 기존 8-GPU 수집과 이 A/B/C/D plan은 별도입니다.

```bash
.venv/bin/python -m vos_memory_inspector.lvos_cli launch-plan --jobs /workspace/CMMT-training-inputs/lvos/jobs.json --output /workspace/CMMT-training-inputs/lvos/launch-plan.json
```

|Job|역할|Device/output ownership|
|---|---|---|
|A|lr=3e-4, seed=7|독립 free UUID·`job-a` writer|
|B|lr=1e-4, 같은 architecture/seed/data|독립 free UUID·`job-b` writer|
|C|Benchmark-owned baseline|생성 코드/계약 미제공 → BLOCKED|
|D|완료 checkpoint 평가|독립 free UUID·각 immutable eval directory|

승인 후 plan의 A/B command를 해당 host에서 실행합니다. 자동 실행 기능은 없습니다. Job D는 `evaluation/requests/epoch-*.json`을 읽어 해당 immutable checkpoint를 평가하고, coordinator가 `evaluation/results/epoch-NNNNN/result.json`으로 merge합니다. Checkpoint `.complete.json`이 없으면 읽지 않습니다. 두 run의 results/status를 공유하지 않습니다.

Evaluation을 여러 worker로 나눌 때 모두 같은 `--shard-count N`과 서로 다른 `--shard-index 0..N-1`을 씁니다. 영상별 완전하고 disjoint한 case assignment를 검증합니다. Coordinator의 `merge-eval --result ...`를 worker마다 반복합니다. 누락·중복 shard는 nonzero exit이며 worker mean을 평균하지 않습니다.

Training은 monitor 매 2 epoch, 최대 outstanding 2로 대기합니다. 결과를 epoch 순서로 한 번만 수용하고, incomplete/foreign 결과는 patience에 반영하지 않습니다. 기본 min_epoch=6/patience=5/delta=.001의 stopping-reference와 raw maximum이 다릅니다. Budget 종료는 early stopping과 구분합니다.

## 9. 로그·재개·종료

`logs/events.jsonl`, `metrics/train.json`/`train.csv`, `metrics/case_records.json`, `checkpoints/epoch-*.ckpt`, `last.ckpt.json`, `best_state_loss.ckpt.json`, `best_monitor.ckpt.json`을 봅니다. State-loss/monitor best는 `best_model.pth`가 아닙니다. `STATUS.json`이 현재 상태이고 예전 실패 marker는 재개 이력으로 보존됩니다.

**재개 명령:** 같은 code/model/data/config/runtime cap을 사용하고, 같은 run 폴더의 `last.ckpt.json`이 가리키는 checkpoint를 입력합니다. 부분 epoch의 optimizer update는 버리고 마지막 완료 epoch에서 다시 시작합니다. Mid-epoch exact resume는 지원하지 않습니다.

```bash
CUDA_VISIBLE_DEVICES=REQUIRED_FREE_GPU_UUID .venv/bin/python -m vos_memory_inspector.lvos_cli train --snapshot /workspace/CMMT-training-views/lvos/full-v1 --model-lock /workspace/CMMT-training-inputs/lvos/model-lock.json --config configs/lvos_job_a.json --gates /workspace/CMMT-training-inputs/lvos/real-gates/gates.json --monitor-protocol /workspace/CMMT-training-inputs/lvos/monitor.json --metric-contract /workspace/CMMT-training-inputs/lvos/metric-contract.json --output /workspace/CMMT-training-runs/lvos-only/job-a --resume REQUIRED_LAST_COMPLETE_CHECKPOINT --device cuda:0 --execute-approved --max-wall-seconds REQUIRED_APPROVED_SECONDS
```

성공: optimizer/scheduler/RNG/RMS 복원, 동일 identity, 다음 epoch 완전 coverage. Config/data/code mismatch면 새 연구 run을 검토하며 기존 run을 강제로 재개하지 않습니다. 실패/손상 checkpoint를 수정하거나 `.partial`을 완료 파일로 바꾸지 않습니다. `.writer.lock`은 정확한 host/PID/writer 소유권과 종료를 확인한 후 운영자가 처리합니다. `pkill`, `killall`, `rm -rf`는 사용하지 않습니다.

Training wall cap은 microbatch 경계에서, eval cap은 frame 경계에서 검사합니다. GPU kernel·모델 로드·한 번의 file read를 즉시 선점하지 않으므로 cap을 조금 초과할 수 있습니다. 승인된 중단은 정확한 process/session을 확인해 해당 작업만 종료하고 마지막 complete checkpoint를 보존합니다.

**Python 종료와 Pod 과금 종료는 별개입니다.** 전용 Pod 종료는 운영자가 (1) 모든 필요한 run/weights/manifest가 persistent mount에 저장됨, (2) completion/SHA 검증, (3) log와 next resume pointer 기록, (4) 같은 Pod에 다른 팀 작업이 없음, (5) volume 유지 조건과 Dashboard billing 상태를 확인한 후 따로 승인·수행합니다. 이 파이프라인은 Pod를 구매·종료하지 않습니다.

## 10. Full-development shortlist와 export

`shortlist --result REQUIRED_MONITOR_MERGED_RESULT ... --output .../shortlist.json`은 상위 3개 distinct **trained** epoch를 동결합니다. Monitor 영상 전체가 완료됐는지 먼저 검증합니다. Full protocol은 아래 CPU 명령이며 subset을 지정하지 않습니다.

```bash
.venv/bin/python -m vos_memory_inspector.lvos_cli freeze-protocol --snapshot /workspace/CMMT-training-views/lvos/full-v1 --source-manifest manifests/lvosv2_train_v1.json --runtime /workspace/CMMT-training-inputs/lvos/runtime.json --scope full_development --output /workspace/CMMT-training-inputs/lvos/full-development.json
```

각 shortlisted checkpoint를 full protocol로 evaluate/merge합니다. Full Replay merged artifact를 `merge-eval --replay REQUIRED_REPLAY_RESULT_JSON`에 입력해야 primary retention이 계산됩니다. Baseline도 같은 case/object/window/GT 규칙과 모델/checkpoint provenance를 제공해야 합니다. 현재 generator/export acceptance 승인은 BLOCKED이며 임시 baseline을 사용하지 않습니다.

```bash
.venv/bin/python -m vos_memory_inspector.lvos_cli select --protocol /workspace/CMMT-training-inputs/lvos/full-development.json --metric-contract /workspace/CMMT-training-inputs/lvos/metric-contract.json --shortlist REQUIRED_SHORTLIST_JSON --result REQUIRED_FULL_DEV_MERGED_RESULT_01 --result REQUIRED_FULL_DEV_MERGED_RESULT_02 --result REQUIRED_FULL_DEV_MERGED_RESULT_03 --output /workspace/CMMT-training-runs/lvos-only/final-selection
```

성공: 완전한 full-dev coverage·candidate/checkpoint/model/data/metric digest, 승인된 strict export를 확인한 `selection.json`, `best_model.pth`, SHA completion marker. Tie 1e-6이면 이른 epoch를 선택합니다. 후보가 3개 미만이면 실제 shortlist 개수만 입력합니다. Monitor/state loss/CPU synthetic만으로 publish하지 않습니다. 이미 best가 있으면 덮어쓰지 않고 새 selection namespace를 사용합니다.

```bash
.venv/bin/python -m vos_memory_inspector.lvos_cli draft-export --run /workspace/CMMT-training-runs/lvos-only/job-a --run /workspace/CMMT-training-runs/lvos-only/job-b --result REQUIRED_REAL_FULL_DEV_MERGED_RESULT --output /workspace/CMMT-training-runs/lvos-only/draft-summary.md
```

성공: 실제 artifacts의 coverage/config/SHA/곡선/점수만 Markdown에 기록. Missing 결과는 대기이며 0으로 채우지 않습니다. Epoch-0/Direct Copy보다 나아졌는지는 실제 비교 결과가 있어야 기술합니다.

## 11. 시간·비용과 문제 해결

현재 GPU 측정값은 없습니다. User-reported LVOS 약 29GB/전체 artifacts 약 400GB는 학습 record 수나 실측 storage cost가 아닙니다. ETA는 남은 valid records와 measured end-to-end throughput, 실제 dev evaluation/checkpoint/I/O 시간을 사용합니다. Cold/warm/shard/case별 편차로 범위를 제시하고 fit-only ETA가 rollout까지 포함한다고 주장하지 않습니다.

전체 Pod 시간당 요금이면 `Pod rate × reserved Pod-hours`를 한 번 적용합니다. 따로 청구되는 Pod/GPU면 각각의 실제 요금·시간을 합산합니다. 8 GPU×72h=576 GPU-hours는 모두 72시간 실행된 경우입니다. Idle reserved 시간·추가 derived view/storage·필요 시 환율은 별도 입력입니다. 기존 cap을 임의로 적용하거나 현재 수집을 종료하지 않습니다.

|증상|확인|안전한 조치와 공유 근거|
|---|---|---|
|Module/entrypoint 없음|새 cwd/interpreter/source SHA/help|격리 환경만 재설치; 실제 실행 경로 공유|
|Sparse/late prompt 오류|official→runtime map, prompt digest|잘못된 case만 수집 담당자 재검토; cache 수정 금지|
|미완료/손상|marker/SHA/stat, `audit.json` reason|중단·원본 보존; 정확한 case와 checksum 공유|
|Legacy provenance 미확인|conditions 또는 reviewed immutable run evidence|담당자 근거 수집; metadata 추정 금지|
|Split 혼입/overlap|frozen source/membership SHA와 video IDs|기존 manifest 보존; 입력 inventory 수정 후 새 audit|
|Policy/shape/dtype 불일치|생성 read-policy·실제 tensor shape/dtype|혼합하지 않음; 모델/수집 담당자 판단|
|OOM|peak memory/microbatch/record shape|종료 근거 보존; 같은 architecture의 진단 batch 축소|
|Volume/RAM/shm 압박|free space/RSS/data wait|승인된 새 storage 계획; cache 일괄 삭제 금지|
|GPU 낮은 사용률|data wait/H2D/compute, case 모델 reload|대표 profile 후 workers/I/O 계획 검토|
|Duplicate worker/writer|host/UUID/output/shard map, lock PID|추가 시작 중단; 정확한 소유자 확인|
|재개 code/data mismatch|identity·SHA·RMS/config|강제 재개 금지; 새 run 또는 원 revision 복원 검토|
|Eval 누락/undefined|expected case/frame, GT-visible/absent counts|실패 case 보고; 평가 shard 완전 merge 전 promotion 금지|
|과거 encoder 호출|trace의 실제 frame IDs|G5 실패 유지; injector/upstream 계약 검토|

## 12. 복사 가능한 인계 양식

```text
[LVOS 진행]
code branch/commit/package SHA:
model preset/config digest/승인자:
collection/membership digest, native/정책/O=1:
실행 종류: CPU synthetic / CPU real cache / real-checkpoint GPU
fit: completed/expected cases, valid records, videos:
development: completed/expected cases, valid records, videos:
rejected/unknown/recollect 및 정확한 case IDs:
G0–G6 상태와 실제 evidence 파일:
GPU host/UUID, run/output 소유자, eval shard union/disjoint:
마지막 complete checkpoint/SHA, STATUS, partial epoch 여부:
measured cold/warm records/sec, remaining work/ETA 범위:
실제 요금·reserved runtime·storage 입력 및 비용 추정:
다음 checkpoint/담당자/필수 입력:
```

**검증됨:** 모든 CLI help, CPU synthetic audit/loader/loss/gradient/checkpoint/resume/worker·selection guard 테스트, 별도 CPU overfit/dev smoke. **대기:** 실제 cache 전체 audit, Small/Base+ weights/GPU G2/G4/G5, real dev scoring/profile/A/B/C/D/full-dev export adapter 수용. 구체적 실행 log와 gate 표는 관련 보고서에 있습니다. 문서 생성은 배포·GPU 속도 개선·VOS 품질 증거가 아닙니다.
