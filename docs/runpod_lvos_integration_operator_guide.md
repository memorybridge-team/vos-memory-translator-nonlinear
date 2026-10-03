# LVOS integration 운영 가이드 — 2026-10-03

**READY_AFTER_INPUTS**. Branch `feat/lvos-training-gates`, 기반 `f77965bd060017beedb930d1916e6ce948a4861c`. 최종 로컬 commit·현재 source SHA·실제 CPU 결과는 공유 파일 `LVOS_INTEGRATION_HANDOFF_2026-10-03.json`에 기록합니다. 이 문서의 원격/GPU 명령은 **운영자 별도 승인 후 실행할 절차**입니다. 문서 작성 중 SSH/GPU/Pod 작업을 수행하지 않았습니다.

Metric pin `bcf0a0f6a36c4129d487e5e58e151468d7ca714b`, baseline pin `dcd335380dbe62f3299dbc5d4456b5ac8a12b4b4`, model pin `746ea3e7d84c366c2d7ac06159e90a1f684bca56`. Benchmark 원본은 수정하지 않습니다. `REQUIRED_*`는 미확인 필수 입력입니다. 숫자 자리의 placeholder를 그대로 실행하면 CLI가 중단합니다.

실제 수행한 검사: working-tree 전체 CPU 회귀 196 PASS(355.08초), 이후 request/source 검사 14 PASS(92.37초), 최종 source-bound LVOS suite 77 PASS(246.99초), help 24개·문서 CLI parser 29개·JSON fixture helper 2개·4-job dry-run planner. Fresh checkout/archive 결과는 동봉 handoff/log를 확인합니다. 서로 중복되는 검사의 PASS 수를 합산하지 않습니다. 실제 cache/GPU/volume 검증은 미수행입니다.

## 1. 한 페이지 시작 순서와 필요한 값

Paired state는 같은 객체·prompt history·switch에 대응하는 Small/Base+ 내부 상태입니다. 이번 학습은 native-history·독립 O=1이며 공간/포인터만 번역합니다. Official ID는 PNG/JPEG 이름의 sparse 번호이고 runtime index는 모델에 들어가는 연속 번호입니다. 모델팀 `base` 구조, fit/development 영상 membership과 원본 cache는 유지합니다.

1. 현재 Pod와 기존 수집 worker의 읽기 전용 상태를 확인합니다.
2. 새 checkout/interpreter에서 CPU 검사를 합니다.
3. `discover` → 전체 `audit` → `verify`로 실제 완료 pair를 확인합니다.
4. 모델/metric 승인과 monitor/full-dev protocol을 동결합니다.
5. 실제 pair update·strict reload·identity/Direct Copy no-replay gate를 수행합니다.
6. 먼저 순차 train/eval을 검증합니다. 별도 free GPU가 승인된 경우 explicit worker/video shards를 사용합니다.
7. Full Replay denominator를 동결된 full-dev protocol/target당 한 번 생성합니다.
8. shortlist의 완전한 full-dev만 canonical retention으로 재선정하고 bundle을 백업합니다.

|필수 값|현재 근거|담당자|
|---|---|---|
|현재 SSH host/port·접근 범위|미확인|Pod 운영자|
|Topology·기존 PID/cwd/interpreter·GPU UUID·free device|미확인, 기존 8 GPU 수집은 보존 대상|수집/Pod 운영자|
|실제 Network Volume 종류·이름·mount|사용자 제공 `/workspace`, KNSW_DATASET; 실제 mount 미검증|Pod 운영자/Dashboard|
|완료 cache root|후보 `/workspace/CMMT-task07-artifacts/production/lvos/cache/{fit,development}/`; 실측 미검증|수집 담당자|
|worker conditions/status/case-bindings root|미확인, discovery `--evidence-root`에 입력|수집 담당자|
|LVOS RGB/GT|후보 `/workspace/CMMT/data/LVOSv2/extracted/`; 영상별 디렉터리 미검증|데이터 담당자|
|checkpoint|예상 `sam2.1_hiera_small.pt`, `sam2.1_hiera_base_plus.pt`; 실제 경로/SHA 미확인|모델/환경 담당자|
|SAM2 revision/install|pin `2b90b9f5ceec907a1c18123530e92e794ad901a4`; 실제 checkout 미확인|환경 담당자|
|model-lock v2 승인|base 구조는 고정 source에서 확인, 승인 기록은 미제공|모델팀|
|metric/export 승인·12개 monitor 영상|승인 근거 미제공|연구/Benchmark 팀|
|시간(초)·실제 요금·통화·job/전체 예산·유지보수 창|미제공, 기존 개인 150,000 KRW를 팀 한도로 적용하지 않음|예산/Pod 운영자|

### 운영자 설정표 — 실제 값을 채워 보관

|항목|입력란|
|---|---|
|Host/port/승인 범위|`REQUIRED_CURRENT_ENDPOINT_AND_SCOPE`|
|Host ID/physical GPU UUID|`REQUIRED_HOST_ID` / `REQUIRED_FREE_GPU_UUID`|
|현재 worker PID/session/cwd/code SHA|`REQUIRED_ACTIVE_WORKER_EVIDENCE`|
|새 code revision/interpreter|최종 handoff commit / `REQUIRED_ISOLATED_PYTHON`|
|cache/evidence/RGB/GT roots|`REQUIRED_VERIFIED_ROOTS`|
|Small/Base+ checkpoint·config·upstream SHA|`REQUIRED_ACTUAL_HASHES`|
|selection/split digest·memory/read policy|`REQUIRED_GENERATING_EVIDENCE`|
|초 상한·요금/시간·통화·허용 총액|`REQUIRED_SECONDS` / `REQUIRED_RATE` / `REQUIRED_CURRENCY` / `REQUIRED_BUDGET`|

## 2. SSH 이후 첫 확인 — 읽기 전용, 원격 승인 후

**위치:** 로컬 PowerShell. **수정:** 현재 endpoint와 본인의 key 경로. **기대:** 현재 Pod shell. **성공:** 승인된 host와 일치. **실패:** Dashboard/접근 범위를 확인합니다. Private key를 공유하거나 문서에 붙이지 않습니다.

```powershell
ssh root@REQUIRED_CURRENT_HOST -p REQUIRED_CURRENT_PORT -i "$env:USERPROFILE\.ssh\id_ed25519"
```

다음 명령은 **RunPod bash**입니다. 수정할 값은 없습니다. GPU UUID/PID, cwd/interpreter, mount/공간을 운영자 설정표에 기록합니다. 확인이 안 되면 production 실행을 보류합니다. 파일 전체 hash scan을 live 수집과 동시에 시작하지 않습니다.

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv
```

```bash
nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name,used_memory --format=csv
```

```bash
ps -eo pid,lstart,etime,args
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
df -h /workspace
```

`/workspace`라는 이름만으로 Network Volume임을 증명할 수 없습니다. Dashboard에서 KNSW_DATASET 연결을 확인합니다. ZIP이 있는 사실도 현재 실행 interpreter의 revision 증거가 아닙니다. 기존 workers가 쓰는 환경에 `git pull`/`pip install -e`를 하지 않습니다. 배포 승인 후 필요한 작업만 case 종료까지 drain하고, 구 evidence를 보존한 별도 checkout/환경을 준비합니다. legacy parent가 새 CLI를 호출하지 않는지 PID·module path를 함께 확인합니다.

## 3. 새 환경 CPU 검사 — 검증된 로컬 명령 / Pod 실행 대기

**위치:** 로컬 PowerShell, 작업 repo. `Set-Location`은 본인 checkout으로 수정합니다. **기대:** 두 LVOS suite의 실제 JUnit/log/report. **성공:** exit 0와 현재 source/suite SHA. **실패:** 첫 오류를 고친 뒤 다시 CPU gate를 수행합니다.

```powershell
Set-Location C:\Users\SAMSUNG\Documents\Codex\2026-09-27\you-are-the-implementation-workspace-for\work\repo
```

```powershell
$env:PYTHONPATH = 'src;.'
```

```powershell
.\.venv\Scripts\python.exe -m vos_memory_inspector.lvos_cli --help
```

```powershell
.\.venv\Scripts\python.exe -m vos_memory_inspector.lvos_cli cpu-test --output ..\..\outputs\REQUIRED_NEW_CPU_EVIDENCE_DIRECTORY --basetemp C:/Users/SAMSUNG/Documents/Codex/tNEW
```

**Pod:** 승인된 격리 checkout에서 새 venv와 GPU PyTorch 환경을 준비합니다. 프로젝트 optional dependency `lvos`는 OpenCV를 포함합니다. 실제 설치 버전은 `provenance.json`/G0로 확인합니다. 위 CPU 명령을 `.venv/bin/python`으로 다시 실행하여 **Pod의 새 증거**를 생성합니다. Windows 보고서를 Pod G6로 재사용하지 않습니다.

Source-lock v2는 UTF-8 source의 CRLF만 LF로 정규화합니다. 의미 있는 code 변경은 계속 검출합니다. 데이터·weights·ZIP·JSON evidence는 원시 SHA를 사용합니다. 구 v1 model-lock은 재생성 후 다시 승인합니다. 구 run을 새 code identity로 강제 resume하지 않습니다.

Windows의 `tNEW`는 placeholder입니다. 존재하지 않는 짧은 전용 이름으로 바꾸고 절대 temp 경로 길이를 40자 이하로 유지합니다. 긴 output 경로 안에 temp를 두면 worker namespace가 Windows 제한을 넘을 수 있습니다. 기존 temp는 삭제하지 않고 `TEST_TEMP_EXISTS`로 중단합니다. Local PASS를 Pod executable 검증으로 재사용하지 않습니다.

**RunPod bash, 새 격리 환경에서 CPU 검사:** 다음 명령은 GPU를 실행하지 않습니다. 원격 실행 승인 후 새 output을 지정합니다. **기대:** `cpu-suite/report.json`, `pytest.log`, `junit.xml`. **성공:** exit 0와 실제 Pod source/suite SHA. **실패:** report의 오류를 보존하고 수정합니다.

```bash
.venv/bin/python -m vos_memory_inspector.lvos_cli cpu-test --output /workspace/CMMT-training-inputs/lvos/cpu-suite
```

아래 경로들은 아직 원격에서 검증하지 않은 **새 배치 제안**입니다: `/workspace/CMMT-lvos-isolated`, `/workspace/CMMT-training-inputs/lvos`, `/workspace/CMMT-training-views/lvos/full-v1`, `/workspace/CMMT-training-runs/lvos-only`.

## 4. Discovery → audit → loader

**위치:** RunPod bash, 새 checkout. **수정:** 실제 root·selection·output. Collector `plan`의 frozen selection JSON이 필요합니다. **성공:** selection digest가 고정 membership과 일치. **실패:** production cache/split을 수정하지 말고 입력 파일을 확인합니다.

```bash
.venv/bin/python -m vos_memory_inspector.paired_collection_cli plan --help
```

`discover`는 원본에 lock/status/새 파일을 쓰지 않습니다. `--trusted-team-legacy`는 **팀 legacy pickle 읽기를 승인한 경우에만** 붙입니다. 승인 없이는 metadata만 확인하고 TRUST_REQUIRED로 unknown을 남깁니다. 실제 worker evidence root가 cache 밖에 있으면 `--evidence-root`를 추가합니다. 파일명으로 checkpoint/case를 추정하지 않습니다.

**기대:** `discovery/requests.json`, `inventory.json`. **성공:** 전체 expected와 completed가 일치, rejected/unknown/duplicate=0. Partial fixture/누락은 BLOCKED이며 실제 완료 수로 부풀리지 않습니다. **실패:** inventory의 정확한 case/reason을 수집 담당자에게 보냅니다.

```bash
.venv/bin/python -m vos_memory_inspector.lvos_cli discover --cache-root /workspace/CMMT-task07-artifacts/production/lvos/cache/fit --cache-root /workspace/CMMT-task07-artifacts/production/lvos/cache/development --selection /workspace/CMMT-training-inputs/lvos/selection.json --output /workspace/CMMT-training-inputs/lvos/discovery-v2 --rgb-root /workspace/CMMT/data/LVOSv2/extracted --trusted-team-legacy --require-complete
```

Expected cases, 완료 pair, 유효 memory record, 파일 bytes는 다른 수치입니다. Status의 dry-run 개수는 pair 완료 증거가 아닙니다. 미완료 `.pt`, checksum/marker 불일치, 중복 semantic case는 requests에 유효 입력으로 승격하지 않습니다. Legacy provenance가 없으면 explicit reviewed run-evidence가 필요합니다. Filename에서 SHA/승인자를 만들어 넣지 않습니다.

**기대:** immutable `snapshot.json`, `audit.json`, `views/*.pt`와 completion marker. **성공:** 전체 expected=completed, fit/dev video overlap=0, padding 제외, generating policy/model 일치. **실패:** 원본을 수정하지 않고 rejected/unknown case만 조사합니다. Serialization-only는 import 가능하지만 잘못된 prompt frame은 selective recollection 판단이 필요합니다.

```bash
.venv/bin/python -m vos_memory_inspector.lvos_cli audit --requests /workspace/CMMT-training-inputs/lvos/discovery-v2/requests.json --selection /workspace/CMMT-training-inputs/lvos/selection.json --output /workspace/CMMT-training-views/lvos/full-v1 --trusted-team-legacy
```

```bash
.venv/bin/python -m vos_memory_inspector.lvos_cli verify --snapshot /workspace/CMMT-training-views/lvos/full-v1
```

Loader는 weights_only-safe tensor dictionary `source_spatial/target_spatial/source_pointer/target_pointer/frame/slot/conditioning/refs`를 읽습니다. 완성된 case 단위로만 shuffle하고 내부 시간·slot 순서는 유지합니다. State-only cache에는 미래 oracle mask가 없습니다. 단일 객체 cache로 joint multi-object equivalence를 주장하지 않습니다.

## 5. 계약·protocol·CPU report gate

**수정:** 실제 승인자와 pinned metric checkout. 승인이 아직 없으면 `freeze-model`의 `--approved-by`를 빼 proposal만 만듭니다. **성공:** digest/config/source pin 확인 후 모델팀 승인. **실패:** 임의 preset/factory를 넣지 않습니다.

```bash
.venv/bin/python -m vos_memory_inspector.lvos_cli freeze-model --output /workspace/CMMT-training-inputs/lvos/model-lock.json --approved-by REQUIRED_MODEL_TEAM_APPROVAL_REFERENCE
```

```bash
.venv/bin/python -m vos_memory_inspector.lvos_cli metric-template --benchmark-root REQUIRED_PINNED_METRIC_CHECKOUT --output /workspace/CMMT-training-inputs/lvos/metric-contract.json
```

`metric-contract.json`의 `approved_by`, `export_approved_by`는 실제 승인 후 입력합니다. `runtime.json`은 `configs/lvos_runtime.template.json`에서 영상별 RGB/GT·SAM2·target 경로를 실제 값으로 채웁니다. Monitor 12개 영상 목록은 score를 보기 전에 결정합니다. 아래 `--video-id`는 12개를 모두 반복해야 하므로 현재 placeholder 1개만으로는 실행 준비가 아닙니다.

```bash
.venv/bin/python -m vos_memory_inspector.lvos_cli freeze-protocol --snapshot /workspace/CMMT-training-views/lvos/full-v1 --source-manifest manifests/lvosv2_train_v1.json --runtime /workspace/CMMT-training-inputs/lvos/runtime.json --output /workspace/CMMT-training-inputs/lvos/monitor.json --scope monitor --video-id REQUIRED_MONITOR_VIDEO_01
```

```bash
.venv/bin/python -m vos_memory_inspector.lvos_cli freeze-protocol --snapshot /workspace/CMMT-training-views/lvos/full-v1 --source-manifest manifests/lvosv2_train_v1.json --runtime /workspace/CMMT-training-inputs/lvos/runtime.json --output /workspace/CMMT-training-inputs/lvos/full-dev.json --scope full_development
```

Full-dev는 모든 frozen dev case를 포함합니다. Protocol v2는 원본 initial prompt PNG SHA/runtime ID와 sparse scored-frame map을 보존합니다. Baseline MD5 재분할과 자체 switch rounding을 사용하지 않습니다. 고정 420 영상에서 MD5를 적용하면 fit 64개가 dev로, dev 60개가 fit으로 바뀌므로 그 CLI를 이 경로에 적용하지 않습니다.

**기대:** `gates.json`과 G3/G6 evidence. **성공:** FAIL 없음; CPU report-only에서 BLOCKED는 exit 0이지만 준비 완료가 아닙니다. `--require-ready`는 BLOCKED/미승인 상태 exit 3, FAIL은 항상 exit 2입니다.

```bash
.venv/bin/python -m vos_memory_inspector.lvos_cli gates --output /workspace/CMMT-training-inputs/lvos/cpu-gates --cpu-test-report /workspace/CMMT-training-inputs/lvos/cpu-suite/report.json
```

## 6. 실제 GPU vertical slice — 별도 실행 승인 필요

아래 GPU 명령은 모두 **pending**입니다. `REQUIRED_*`를 실제 숫자/UUID/host/quote로 바꾸고 별도 실행 승인을 받습니다. GPU process 안의 device는 `cuda:0`입니다. `CUDA_VISIBLE_DEVICES=GPU-...`이면 physical index와 local index가 다릅니다. Lease root는 승인된 공유 persistent root이며 advisory lock을 기존 collectors가 자동으로 따르는 것은 아닙니다. Free UUID 확인이 선행합니다.

**기대:** G2 spatial/pointer 업데이트, G4 weights/output/optimizer/scheduler/RNG/RMS parity. **실패:** GPU OOM/계약 오류를 로그로 보낸 뒤 새 namespace에서 승인된 조치만 수행합니다.

```bash
CUDA_VISIBLE_DEVICES=REQUIRED_FREE_GPU_UUID CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m vos_memory_inspector.lvos_cli gates --snapshot /workspace/CMMT-training-views/lvos/full-v1 --model-lock /workspace/CMMT-training-inputs/lvos/model-lock.json --metric-contract /workspace/CMMT-training-inputs/lvos/metric-contract.json --output /workspace/CMMT-training-inputs/lvos/pair-gpu-gates --device cuda:0 --execute-approved --max-wall-seconds REQUIRED_APPROVED_SECONDS --lease-root /workspace/CMMT-training-inputs/lvos/gpu-leases --host-id REQUIRED_HOST_ID --physical-gpu-uuid REQUIRED_FREE_GPU_UUID --quoted-hourly-rate REQUIRED_RATE --total-budget REQUIRED_BUDGET --budget-currency USD
```

One-video overfit는 frozen fit 영상 `0ClBYzYm`의 실제 case 3개를 명시한 **별도 pilot audit/view**를 사용합니다. 다음 case ID는 로컬 manifest에서 확인했으며 실제 cache의 완료 여부는 아직 미확인입니다. 먼저 전체 requests에서 이 3개만 별도 파일에 복사합니다. 누락/중복이면 assert에서 중단합니다. 원본 requests는 변경하지 않습니다. 최소 미래 길이와 맞지 않는 임의 20-frame 예제를 사용하지 않습니다. Pilot view는 main train용 full snapshot으로 승격하지 않습니다.

```bash
.venv/bin/python -c "from pathlib import Path; from vos_memory_inspector.lvos_contract import json_read; from vos_memory_inspector.training_storage import write_json; ids={'train:0ClBYzYm:obj1:switch621','train:0ClBYzYm:obj1:switch1221','train:0ClBYzYm:obj1:switch1821'}; rows=[r for r in json_read('/workspace/CMMT-training-inputs/lvos/discovery-v2/requests.json') if r['case_id'] in ids]; output=Path('/workspace/CMMT-training-inputs/lvos/overfit-requests.json'); assert len(rows)==3 and {r['case_id'] for r in rows}==ids and not output.exists(); write_json(output,rows)"
```

```bash
.venv/bin/python -m vos_memory_inspector.lvos_cli audit --requests /workspace/CMMT-training-inputs/lvos/overfit-requests.json --selection /workspace/CMMT-training-inputs/lvos/selection.json --output /workspace/CMMT-training-views/lvos/overfit-0ClBYzYm --trusted-team-legacy --pilot-case-id train:0ClBYzYm:obj1:switch621 --pilot-case-id train:0ClBYzYm:obj1:switch1221 --pilot-case-id train:0ClBYzYm:obj1:switch1821
```

**위치:** 승인된 RunPod bash. **수정:** 실제 free UUID/host/rate/budget와 초 상한. **기대:** spatial/pointer loss, gradient, 유효 record 수와 checkpoint. **성공:** 3개 case coverage와 유한 update, strict reload. **실패:** 해당 입력/로그를 보존합니다. 아래는 승인 대기 GPU 명령이며 3 epoch의 짧은 진단만 수행합니다. Config의 나머지 설정은 고정 Job A를 사용합니다.

```bash
.venv/bin/python -c "from pathlib import Path; from vos_memory_inspector.lvos_contract import json_read; from vos_memory_inspector.training_storage import write_json; config=json_read('configs/lvos_job_a.json'); config['max_epochs']=3; output=Path('/workspace/CMMT-training-inputs/lvos/overfit-3epochs.json'); assert not output.exists(); write_json(output,config)"
```

```bash
CUDA_VISIBLE_DEVICES=REQUIRED_FREE_GPU_UUID CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m vos_memory_inspector.lvos_cli overfit --snapshot /workspace/CMMT-training-views/lvos/overfit-0ClBYzYm --model-lock /workspace/CMMT-training-inputs/lvos/model-lock.json --config /workspace/CMMT-training-inputs/lvos/overfit-3epochs.json --output /workspace/CMMT-training-runs/lvos-only/overfit-0ClBYzYm --device cuda:0 --execute-approved --max-wall-seconds REQUIRED_APPROVED_SECONDS --lease-root /workspace/CMMT-training-inputs/lvos/gpu-leases --host-id REQUIRED_HOST_ID --physical-gpu-uuid REQUIRED_FREE_GPU_UUID --quoted-hourly-rate REQUIRED_RATE --total-budget REQUIRED_BUDGET --budget-currency USD
```

**Epoch-0:** 모델 승인 후 CPU initialize가 untrained checkpoint를 만듭니다. **성공:** body/complete/journal/last pointer가 존재하고 strict load. **실패:** 원본을 삭제하지 말고 recovery report를 확인합니다.

```bash
.venv/bin/python -m vos_memory_inspector.lvos_cli initialize --snapshot /workspace/CMMT-training-views/lvos/full-v1 --model-lock /workspace/CMMT-training-inputs/lvos/model-lock.json --config configs/lvos_job_a.json --monitor-protocol /workspace/CMMT-training-inputs/lvos/monitor.json --metric-contract /workspace/CMMT-training-inputs/lvos/metric-contract.json --output /workspace/CMMT-training-runs/lvos-only/identity
```

**Handoff:** 아래 learned epoch-0와 같은 명령의 `--method direct_copy`, 별도 output을 각각 실행합니다. **성공:** encoder IDs는 모두 `>t`, 시작은 `t+1`, prompt/object/frame/slot/validity/PE 계약 일치. **실패:** target 과거 encoder 호출이 있으면 G5 PASS로 표시하지 않습니다.

```bash
CUDA_VISIBLE_DEVICES=REQUIRED_FREE_GPU_UUID CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m vos_memory_inspector.lvos_cli evaluate --snapshot /workspace/CMMT-training-views/lvos/full-v1 --protocol /workspace/CMMT-training-inputs/lvos/monitor.json --checkpoint /workspace/CMMT-training-runs/lvos-only/identity/checkpoints/epoch-00000.ckpt --metric-contract /workspace/CMMT-training-inputs/lvos/metric-contract.json --benchmark-root REQUIRED_PINNED_METRIC_CHECKOUT --output /workspace/CMMT-training-runs/lvos-only/identity-learned-shard0 --method learned --device cuda:0 --execute-approved --max-wall-seconds REQUIRED_APPROVED_SECONDS --lease-root /workspace/CMMT-training-inputs/lvos/gpu-leases --host-id REQUIRED_HOST_ID --physical-gpu-uuid REQUIRED_FREE_GPU_UUID --quoted-hourly-rate REQUIRED_RATE --total-budget REQUIRED_BUDGET --budget-currency USD
```

```bash
.venv/bin/python -m vos_memory_inspector.lvos_cli merge-eval --result /workspace/CMMT-training-runs/lvos-only/identity-learned-shard0/result.json --protocol /workspace/CMMT-training-inputs/lvos/monitor.json --metric-contract /workspace/CMMT-training-inputs/lvos/metric-contract.json --benchmark-root REQUIRED_PINNED_METRIC_CHECKOUT --output /workspace/CMMT-training-runs/lvos-only/identity-learned-merged
```

Direct Copy도 따로 merge합니다. 실제 두 merged result를 `gates --handoff-result ... --direct-copy-result ... --cpu-test-report ... --require-ready`에 넣어 새 output에서 G0–G6를 실행합니다. GPU device/rate/lease/time 인자는 위 GPU gate 명령과 동일하게 필수입니다. 실제 PASS가 없으면 main train은 실행되지 않습니다.

## 7. Full Replay 전용 denominator와 canonical primary

Full Replay는 **target Base+와 original initial prompt만** 사용하여 prompt부터 전개합니다. 과거 frame 처리는 이 방법에서 허용합니다. 미래 GT는 입력으로 전달하지 않고 scoring에만 씁니다. 같은 영상/객체의 세 fraction은 한 rollout에서 계산합니다. Protocol/target/read-policy/source pin/coverage가 모두 같은 완료 결과만 재사용합니다. 재학습 epoch마다 denominator를 다시 만들 필요가 없습니다.

**기대:** per-case J/F/JF·visible/absent/undefined·encoder IDs, `result.json`, `READY.json`. **성공:** frozen full-dev expected coverage 일치. **실패:** 구 baseline row의 부족한 provenance를 새 SHA로 소급 보완하지 않습니다. Bare `video/object/switch_name/jf`는 import 근거가 아닙니다.

```bash
CUDA_VISIBLE_DEVICES=REQUIRED_FREE_GPU_UUID CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m vos_memory_inspector.lvos_cli full-replay --snapshot /workspace/CMMT-training-views/lvos/full-v1 --protocol /workspace/CMMT-training-inputs/lvos/full-dev.json --metric-contract /workspace/CMMT-training-inputs/lvos/metric-contract.json --benchmark-root REQUIRED_PINNED_METRIC_CHECKOUT --baseline-root REQUIRED_PINNED_BASELINE_CHECKOUT --output /workspace/CMMT-training-runs/lvos-only/replay-shard0 --device cuda:0 --execute-approved --max-wall-seconds REQUIRED_APPROVED_SECONDS --lease-root /workspace/CMMT-training-inputs/lvos/gpu-leases --host-id REQUIRED_HOST_ID --physical-gpu-uuid REQUIRED_FREE_GPU_UUID --quoted-hourly-rate REQUIRED_RATE --total-budget REQUIRED_BUDGET --budget-currency USD
```

```bash
.venv/bin/python -m vos_memory_inspector.lvos_cli merge-eval --result /workspace/CMMT-training-runs/lvos-only/replay-shard0/result.json --protocol /workspace/CMMT-training-inputs/lvos/full-dev.json --metric-contract /workspace/CMMT-training-inputs/lvos/metric-contract.json --benchmark-root REQUIRED_PINNED_METRIC_CHECKOUT --output /workspace/CMMT-training-runs/lvos-only/replay-merged
```

Final primary는 `selection_score`: (video,fraction) 내부 object 평균 → video별 Method/Replay → video 평균 → 25/50/75% 평균입니다. `.9,.3,.6` 대비 `.6,.3,.6`의 canonical retention은 **88.8889%**, legacy pooled는 **83.3333%**입니다. Legacy pooled/gap/PGR를 primary로 대체하지 않습니다. Replay=0 영상은 해당 fraction ratio에서 제외하고, 정의 가능한 영상이 없으면 primary=null입니다. Missing coverage는 중단합니다. 유효 retention >100을 clipping하지 않으며 Full Replay를 수학적 상한으로 가정하지 않습니다.

Monitor는 기존 **50%·≤64 runtime frame의 `monitor_jf_proxy` [0,1]**입니다. `min_delta=.001`과 early-stop 조건을 유지합니다. 최종 full-dev primary와 같은 지표라고 보고하지 않습니다.

## 8. 순차 학습/평가, explicit worker와 merge

**우선 순차 모드:** 하나의 승인된 free GPU lease를 가진 process가 학습 → 평가를 순서대로 수행합니다. RNG는 callback 전후 보존합니다. CPU fake-worker 검증은 실제 SAM2 GPU 처리량 증거가 아닙니다.

```bash
CUDA_VISIBLE_DEVICES=REQUIRED_FREE_GPU_UUID CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m vos_memory_inspector.lvos_cli train --snapshot /workspace/CMMT-training-views/lvos/full-v1 --model-lock /workspace/CMMT-training-inputs/lvos/model-lock.json --config configs/lvos_job_a.json --gates /workspace/CMMT-training-inputs/lvos/real-gates/gates.json --monitor-protocol /workspace/CMMT-training-inputs/lvos/monitor.json --metric-contract /workspace/CMMT-training-inputs/lvos/metric-contract.json --benchmark-root REQUIRED_PINNED_METRIC_CHECKOUT --evaluation-mode sequential --output /workspace/CMMT-training-runs/lvos-only/job-a --device cuda:0 --execute-approved --max-wall-seconds REQUIRED_APPROVED_SECONDS --lease-root /workspace/CMMT-training-inputs/lvos/gpu-leases --host-id REQUIRED_HOST_ID --physical-gpu-uuid REQUIRED_FREE_GPU_UUID --quoted-hourly-rate REQUIRED_RATE --total-budget REQUIRED_BUDGET --budget-currency USD
```

**기대:** 독립 run/config/logs/metrics/checkpoints/evaluation requests/results. **성공:** component loss/gradient/LR/valid records와 현재 checkpoint 출력. **실패:** `STATUS.json`의 단계/이유를 보존합니다. State loss 감소를 VOS 개선으로 해석하지 않습니다. Linear는 기존 training path에 있으며 이번 고정 Transformer repair에서 architecture suite를 확장하지 않습니다.

Parallel은 별도 승인된 free GPU UUID/host를 사용합니다. `train --evaluation-mode queue`는 immutable 요청만 내고, 다음 worker는 **dry-run이 기본**입니다. 같은 mutable selection/status 파일을 여러 writer가 공유하지 않습니다.

```bash
.venv/bin/python -m vos_memory_inspector.lvos_cli eval-worker --run /workspace/CMMT-training-runs/lvos-only/job-a --snapshot /workspace/CMMT-training-views/lvos/full-v1 --benchmark-root REQUIRED_PINNED_METRIC_CHECKOUT --shard-index 0 --shard-count 1
```

**실제 worker 승인 후:** `--execute`와 GPU 승인/rate/lease/time 값이 모두 있어야 실행합니다. `--drain --merge-shards`는 budget 안에서 남은 요청을 처리합니다. 그렇지 않으면 한 번의 scan/실행 후 종료합니다. 전부 완료되지 않으면 pending을 보존합니다.

```bash
CUDA_VISIBLE_DEVICES=REQUIRED_EVALUATOR_GPU_UUID CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m vos_memory_inspector.lvos_cli eval-worker --run /workspace/CMMT-training-runs/lvos-only/job-a --snapshot /workspace/CMMT-training-views/lvos/full-v1 --benchmark-root REQUIRED_PINNED_METRIC_CHECKOUT --shard-index 0 --shard-count 1 --execute --execute-approved --drain --merge-shards --device cuda:0 --max-wall-seconds REQUIRED_APPROVED_SECONDS --lease-root /workspace/CMMT-training-inputs/lvos/gpu-leases --host-id REQUIRED_HOST_ID --physical-gpu-uuid REQUIRED_EVALUATOR_GPU_UUID --quoted-hourly-rate REQUIRED_RATE --total-budget REQUIRED_BUDGET --budget-currency USD
```

Multiple shards는 영상 단위 disjoint assignment입니다. 모든 worker에서 같은 shard-count, 서로 다른 index/UUID를 사용하고 dry-run case ID의 완전/중복 없는 coverage를 먼저 확인합니다. 한 8-GPU host인지 여러 Pods인지는 실제 topology로 결정합니다. Independent processes이며 torchrun/DDP가 아닙니다.

**수동 CPU merge:** 모든 shard의 verified pointer와 READY가 필요합니다. 같은 입력으로 다시 merge하면 SHA를 재검증하여 같은 결과를 재사용합니다. Foreign/stale/incomplete shard는 중단합니다.

```bash
.venv/bin/python -m vos_memory_inspector.lvos_cli eval-merge --run /workspace/CMMT-training-runs/lvos-only/job-a --request /workspace/CMMT-training-runs/lvos-only/job-a/evaluation/requests/epoch-00002.json --benchmark-root REQUIRED_PINNED_METRIC_CHECKOUT --shard-count 1
```

```bash
.venv/bin/python -m vos_memory_inspector.lvos_cli eval-status --run /workspace/CMMT-training-runs/lvos-only/job-a --output /workspace/CMMT-training-inputs/lvos/job-a-eval-status.json
```

`training_status=COMPLETED`와 `evaluation_status=PENDING/COMPLETED`, final promotion은 별개입니다. Python이 종료되었다고 evaluation이나 Pod 과금까지 완료된 것은 아닙니다. `launch-plan`은 계획만 생성하며 executor로 바뀌지 않았습니다.

## 9. 중단·복구·비용

Epoch checkpoint는 journal → staged body → body → completion marker → last pointer 순서입니다. `--resume latest`는 동일 identity의 journal/완료 chain을 검증하여 최신을 복구합니다. 구 pointer가 이전 완료 epoch를 가리켜도 새 완료 checkpoint를 확인합니다. Body가 아직 없는 journal은 보존하며 이전 완료 epoch에서 다시 시작합니다. Corrupt/foreign/unbound orphan body는 임의 덮어쓰기 없이 중단합니다. Epoch-0 initialize를 같은 입력으로 다시 실행해 복구할 수 있습니다. 학습이 진행된 run을 initialize로 초기화하지 않습니다.

다음 명령은 `train --help`로 검증된 resume flag 확인입니다. 실제 재개는 §8의 동일 config/output/계약/GPU 승인 명령에 `--resume latest`를 추가합니다. Config나 code identity가 바뀌면 새 run을 만듭니다. Mid-epoch exact resume를 지원한다고 주장하지 않습니다.

```bash
.venv/bin/python -m vos_memory_inspector.lvos_cli train --help
```

하나의 deadline은 snapshot hashing, RMS, loader/batch, state-dev, eval frame/wait, finalization에 전달됩니다. 상한이면 `STOPPED/budget_stop/incomplete_stage`, exit 4를 남기고 부분 dev 점수를 best로 사용하지 않습니다. 개별 hash/load/serialization/kernel은 선점하지 못하므로 한 연산만큼 초과할 수 있습니다. DataLoader worker prefetch/종료도 즉시 선점하지 않습니다. 완료 파일은 보존합니다.

Stale lock은 자동 제거 대상이 아닙니다. 정확한 PID/host/UUID/writer 소유권을 확인한 뒤 승인된 graceful shutdown을 수행합니다. Blanket `pkill`/`killall`/`rm -rf`를 사용하지 않습니다. 기존 collectors와 다른 Pod 작업을 종료하지 않습니다.

GPU CLI는 초 cap·실제 rate·통화·job budget과 UUID/lease를 요구하며 `seconds/3600 × rate <= job budget`을 검사합니다. 이 guard는 shared Pod 전체 과금 종료나 여러 job 총예산의 중앙 관리 기능이 아닙니다. 운영자가 실제 합계 예산을 승인해야 합니다. Whole-Pod 요금은 Pod-hour당 한 번 적용하고 GPU 8개를 다시 곱하지 않습니다. 별도 GPU/Pod 과금이면 실제 rate/runtime 합계를 사용합니다. Idle reserved time, 추가 storage, 필요 시 현재 FX는 명시 입력으로 기록합니다. 8×72=576 GPU-hours는 8개가 실제로 전 구간 예약된 조건입니다. 실제 처리량/남은 weighted work를 측정하기 전에는 dataset GB만으로 ETA를 만들지 않습니다.

## 10. Shortlist → full-dev → final export → backup

**수정:** 실제 완료된 모든 monitor result 목록. **기대:** top-3 trained epoch shortlist. **성공:** frozen proxy identity/coverage 동일. Epoch-0은 trained 후보에서 제외합니다.

```bash
.venv/bin/python -m vos_memory_inspector.lvos_cli shortlist --result REQUIRED_MONITOR_RESULT_1 --result REQUIRED_MONITOR_RESULT_2 --result REQUIRED_MONITOR_RESULT_3 --output /workspace/CMMT-training-inputs/lvos/shortlist.json
```

후보별 §6 `evaluate`를 full-dev protocol/후보 checkpoint로 실행하여 disjoint shards를 merge합니다. 이때 `merge-eval --replay /workspace/CMMT-training-runs/lvos-only/replay-merged/result.json`을 추가합니다. Missing/undefined coverage는 임의 제거하지 않습니다. 모든 shortlist 후보의 complete full-dev가 필요합니다.

```bash
.venv/bin/python -m vos_memory_inspector.lvos_cli select --protocol /workspace/CMMT-training-inputs/lvos/full-dev.json --metric-contract /workspace/CMMT-training-inputs/lvos/metric-contract.json --shortlist /workspace/CMMT-training-inputs/lvos/shortlist.json --benchmark-root REQUIRED_PINNED_METRIC_CHECKOUT --result REQUIRED_COMPLETE_CANDIDATE_1 --result REQUIRED_COMPLETE_CANDIDATE_2 --result REQUIRED_COMPLETE_CANDIDATE_3 --output /workspace/CMMT-releases/REQUIRED_RELEASE_ID
```

**기대:** strict schema/config/weights `best_model.pth`, completion/SHA, `selection.json`. **성공:** canonical primary를 원본 Replay artifact에서 재계산, 실제 Benchmark loader acceptance 승인. 선정 주장은 **평가한 shortlist 중 최고**이며 전체 epoch의 global best 증명은 아닙니다. Last/state-loss-best/monitor-best/final-best는 서로 다른 artifact입니다.

```bash
.venv/bin/python -m vos_memory_inspector.lvos_cli compare --protocol /workspace/CMMT-training-inputs/lvos/full-dev.json --metric-contract /workspace/CMMT-training-inputs/lvos/metric-contract.json --benchmark-root REQUIRED_PINNED_METRIC_CHECKOUT --replay /workspace/CMMT-training-runs/lvos-only/replay-merged/result.json --result REQUIRED_COMPLETE_CANDIDATE_1 --result REQUIRED_COMPLETE_CANDIDATE_2 --result REQUIRED_COMPLETE_CANDIDATE_3 --output /workspace/CMMT-releases/REQUIRED_RELEASE_ID/comparison.json
```

```bash
.venv/bin/python -m vos_memory_inspector.lvos_cli bundle --release /workspace/CMMT-releases/REQUIRED_RELEASE_ID --run /workspace/CMMT-training-runs/lvos-only/REQUIRED_WINNER_RUN --output /workspace/CMMT-releases/REQUIRED_RELEASE_ID-backup --include-resume
```

**기대:** small `bundle.zip`, sanitized `artifact-index.json`, `checksums.json`. **성공:** strict load와 bytes/SHA 확인 후 승인된 로컬/기관 storage에 별도 사본을 보관합니다. 이 명령은 업로드/Pod 변경을 하지 않습니다. Translator 457,216개 FP32 weight는 원소만 약 1.83MB이며 긴 학습 시간만으로 checkpoint가 커지는 것은 아닙니다. 실제 파일 크기는 index를 봅니다. Raw/cache/view/prediction과 weights는 Git에 넣지 않습니다.

## 11. 문제 해결과 Slack 인계

|증상|확인|안전한 조치|공유 증거|
|---|---|---|---|
|Sparse/late prompt mismatch|official/runtime map, initial PNG SHA|정확한 해당 case만 재검토/재수집 승인|case ID·map·reason|
|module/entrypoint 없음|cwd/interpreter/help/install path|새 격리 환경 경로 수정|revision·help log|
|구 lock/schema|v1/v2, source hash policy|새 source-bound proposal/승인/gates|old/new SHA|
|checksum/partial shard|marker/sidecar/stability/writer|원본 보존, 해당 파일 조사|bytes·SHA·PID|
|legacy provenance unknown|conditions/bindings/reviewed evidence|명시된 proof 확보, filename 추정 금지|unknown inventory|
|policy/empty prompt|READ_FIELDS·O=1·prompt object pixel|계약/annotation 원인 확인|config SHA·prompt SHA|
|OOM/shared I/O 포화|VRAM/CPU/I/O/RSS/free space|승인된 microbatch/worker 변경은 새 run|profile·rate·elapsed|
|중복 shard/foreign result|index/count/run/checkpoint/protocol SHA|worker 소유권 확인, 새 attempt|assignment·READY SHA|
|resume code mismatch|code/lock/config/data identity|동일 revision 복원 또는 새 run|journal·recovery report|
|학습 완료, eval pending|eval-status·request/worker/result READY|승인된 worker drain/merge|pending 리스트|

```text
[LVOS 진행]
Code revision / source SHA:
Model / metric / baseline pin:
Generating/read policy / selection digest:
Fit: completed pairs / expected cases; valid records; bytes:
Development: completed pairs / expected cases; valid records; bytes:
Rejected / recollect / unknown / missing:
Shard coverage (index/count, 중복/누락):
Training status / evaluation pending / final promotion:
Measured throughput / weighted remaining work / ETA range:
Actual quoted Pod/GPU rate / reserved idle time / elapsed cost:
다음 점검 시각·필요 입력·담당자:
```

검증 구분: 로컬 help/CPU fault/fixture/metric/worker/bundle와 clean archive 검사는 실제 log로 보고합니다. Production discovery, 실제 pair update, real checkpoint reload/no-replay, Full Replay GPU, full-dev VOS quality, Network Volume restart/실제 비용은 별도 입력·실행이 준비될 때까지 **BLOCKED**입니다.
