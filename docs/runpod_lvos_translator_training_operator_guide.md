# LVOS 단일 translator DDP 운영 안내

2026-10-03 · **READY_AFTER_INPUTS** · `feat/lvos-pilot-preflight`, base `f2632d7`.
최종 로컬 commit/소스 ZIP SHA는 공유용 `outputs/LVOS_DDP_HANDOFF_2026-10-03.json`에 기록한다.
최종 Windows CPU **9 tests PASS**. 앞선 Linux Python 3.12/Torch 2.8 source는 CPU 6 tests 및 실제 cache 2-rank 검사 PASS.
최종 revision의 Linux CPU 재검사는 새 Pod에서 먼저 수행한다. 실제 GPU 실행은 0회다.

## 1. 현재 상태와 필요한 입력

|항목|확인값 / 담당자|
|---|---|
|목표|고정 base translator 하나, fit 학습/dev spatial·pointer loss; J&F는 평가팀|
|현재 GPU|RTX 4090 1개. 동일 Pod의 2 GPU는 아직 준비되지 않음 — 운영자|
|Volume|new / j700qtcgrg / EUR-IS-1 / `/workspace`|
|fit|`/workspace/CMMT-task07-artifacts/production/lvos/cache/fit`, 실제 1,488 files|
|development|`/workspace/CMMT-task07-artifacts/production/lvos/cache/development`, 실제 315 files|
|RGB|`/workspace/CMMT/data/LVOSv2/extracted/train/JPEGImages`|
|입력 검사|전체 1,803 files CPU 읽기/SHA/tensor 대응 통과. fit 9개의 prompt 조건은 불일치|
|9개 처리|데이터팀 확인 대기. `configs/lvos_ddp_prompt_exclusions.proposed.json`은 미승인 제외안|
|새 Pod 입력|SSH endpoint, free GPU UUID 2개, host ID, Pod 전체 USD/hour, 시간·USD 상한 — 운영자|
|코드/환경|격리 source 경로·ZIP SHA·Python/Torch import 경로 — 학습 담당|

Pod 설정: 동일 host GPU 2개, 위 Volume/지역/mount, Python 3.12/Torch 2.8 환경.
정확한 container image 이름과 새 가격은 미확인이다. 현재 1-GPU USD 0.744/hour를 새 Pod 단가로 사용하지 않는다.
Pod 생성·Stop·Terminate는 운영자가 별도 승인한다. SSH 접속 정보는 로컬 기록에만 보관한다.

## 2. 새 Pod의 CPU 검사

이하 **RunPod bash**. 운영자가 실제 절대 경로를 `SOURCE_ROOT`(격리 repo), `PY`(별도 venv Python),
`RUN_ROOT`(원본 밖 새 출력)에 설정한다. 기존 worker 환경에서 git pull/editable install 하지 않는다.
현재 CPU 증거는 `/workspace/CMMT-lvos-isolated/lvos-ddp-cpu-20261003T111600Z`에 있다.
새 Pod에서는 기존 venv의 system-site-packages 경로도 재확인한다.

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH="${SOURCE_ROOT:?검증된 source 경로 필요}/src" "${PY:?격리 Python 필요}" -m vos_memory_inspector.lvos_ddp train --help
```

성공: DDP flags 표시, exit 0. 실패: import/interpreter 경로를 확인하고 GPU 실행 보류.

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="${SOURCE_ROOT:?}/src" "${PY:?}" -m pytest "${SOURCE_ROOT}/tests/test_lvos_ddp.py" -q -p no:cacheprovider --basetemp "${RUN_ROOT:?새 경로 필요}/cpu-tmp" --junitxml "${RUN_ROOT}/cpu-tests.xml"
```

성공: **9 passed**, exit 0. CPU 2/4-rank, 불균등 valid count, accumulation, 단일 global batch 대비 update,
양 branch, strict reload, epoch 중단/재개, 미승인 제외안 차단을 확인한다. 실패: log/JUnit 전달 후 중단.
Windows test는 해당 Torch TCPStore/libuv 문제 때문에 FileStore+spawn을 사용한다. Linux test는 torchrun이다.
CPU PASS를 GPU PASS로 기록하지 않는다.

## 3. 기존 cache를 loader에 연결

구형 v2 cache의 CanonicalState/StateSpec만 allowlist하여 weights_only=True/CPU로 읽는다.
실제로 읽은 bytes의 SHA와 `.sha256`, 안정된 stat, object/frame/slot/conditioning/validity,
finite, BF16 spatial/FP32 pointer, official ID→RGB runtime index, frozen fit/dev를 검사한다.
Marker 부재를 미완성으로 취급하지 않는다. Unsafe pickle fallback은 없다.
생성 revision·weights·mask/pair history가 없으면 UNKNOWN으로 기록하며 만들어 넣지 않는다.

`raw_index.json`은 학습 코드가 만드는 derived 관리 파일이다. 원본에 snapshot.json/audit.json이 있다고 가정하지 않는다.
Tensor 복제 변환 없이 원본을 직접 읽고 작은 index 및 fit-only normalization만 별도 저장한다.

**CPU 전체 검사:** `INDEX_ROOT`는 schema가 섞이지 않는 새 출력 경로다.
현재 9개 prompt 문제가 해결되지 않으면 이 명령은 실패한다.

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="${SOURCE_ROOT:?}/src" "${PY:?}" -m vos_memory_inspector.lvos_ddp index --fit-root /workspace/CMMT-task07-artifacts/production/lvos/cache/fit --development-root /workspace/CMMT-task07-artifacts/production/lvos/cache/development --rgb-root /workspace/CMMT/data/LVOSv2/extracted/train/JPEGImages --output "${INDEX_ROOT:?새 index 경로 필요}" --operator-completed user_confirmed_completed_2026-10-03 --max-wall-seconds 1200
```

성공: raw_index.json, INSPECTION.json, cases/*.json, normalization, exit 0.
실패: FAILURES.json에 case ID/사유를 기록하고 학습 index를 발행하지 않는다. 원본 수정·새 completion marker 부착 없음.
미완료 index는 동일 인자·원본 stat에서 같은 명령으로 재개한다. 완성 index는 재발행하지 않는다.
학습 시 원본 bytes SHA도 재확인한다.

**확인 지연 시 제외안:** 예상 fit 1,479 / development 315 cases. 계획 수이며 성공 tensor 수가 아니다.
운영자가 제외안을 승인하면 proposal의 복사본에 실제 approved_by/recorded_at을 기록한다.
위 index 명령에 `--exclusion-plan "$APPROVED_EXCLUSION_PLAN"`을 추가하고 새 INDEX_ROOT를 사용한다.
Frozen video membership은 유지하며 case별 제외 이유/실제 사용 목록을 run manifest에 저장한다.
미승인 proposal로 CPU index를 만들 수 있으나 본 학습은 거부된다.
`--prompt-decision`은 특정 관측 prompt history를 수용하는 명시 승인 경로이며 현재 사용하지 않는다.

## 4. 2-GPU smoke → 본 학습

새 Pod에서 먼저 읽기 전용 GPU/프로세스 확인. 전체 process args/환경변수는 출력하지 않는다.

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv
```

```bash
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
```

승인된 free UUID를 GPU_UUID_0/GPU_UUID_1에 설정한다. HOST_ID, LEASE_ROOT, POD_RATE_USD,
SMOKE_SECONDS/SMOKE_BUDGET_USD, TRAIN_SECONDS/TRAIN_BUDGET_USD, SMOKE_ROOT/TRAIN_ROOT도 실제 승인값이어야 한다.
Smoke 계획은 최대 600초로 제안하되 실제 상한은 운영자가 정한다. 미입력 변수는 즉시 안내하고 중단한다.

|world|rank/local CUDA|physical UUID|microbatch|accumulation|global batch|
|---|---|---|---|---|---|
|2|0 / 0|GPU_UUID_0|16|2|64|
|2|1 / 1|GPU_UUID_1|16|2|64|
|4|0..3 / 0..3|승인 UUID 4개|16|1|64|

완성 case 순서만 shuffle한다. rank는 global batch의 연속 valid-record 구간을 맡고 source/target/slot 대응을 함께 유지한다.
Case가 rank 경계에 걸릴 수 있으나 동일 record는 중복하지 않는다.
DDP 평균 gradient에 world_size/global_valid_records를 적용한다. 마지막 batch는 실제 유효 수로 가중하며
부족한 rank는 loss 0으로 collective에 참여한다. Fit-only normalization을 rank 0이 broadcast한다.

**아래 GPU 명령은 운영 승인 후 실행할 명령이며 아직 검증/실행하지 않았다.**

```bash
 timeout --signal=TERM --kill-after=30s "${SMOKE_SECONDS:?승인 초 필요}s" env CUDA_VISIBLE_DEVICES="${GPU_UUID_0:?UUID 필요},${GPU_UUID_1:?UUID 필요}" CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 PYTHONPATH="${SOURCE_ROOT:?}/src" "${PY:?}" -m torch.distributed.run --standalone --nproc_per_node=2 -m vos_memory_inspector.lvos_ddp train --index "${INDEX_ROOT:?}/raw_index.json" --output "${SMOKE_ROOT:?새 출력 필요}" --mode smoke --global-batch 64 --device cuda --max-wall-seconds "$SMOKE_SECONDS" --execute-approved --gpu-uuid "$GPU_UUID_0" --gpu-uuid "$GPU_UUID_1" --host-id "${HOST_ID:?}" --lease-root "${LEASE_ROOT:?}" --pod-hourly-rate "${POD_RATE_USD:?Pod 전체 단가 필요}" --budget-usd "${SMOKE_BUDGET_USD:?승인 USD 필요}"
```

성공: exit 0, STATUS.json=REAL_DDP_SMOKE_PASS, 양 branch update, strict model/optimizer/scheduler reload,
rank weights 최대 차이 0, assignments/*-coverage.json의 unique/expected 일치. Diagnostic epoch 3개 저장.
본 학습과 같은 index/global batch/optimizer 정책이어야 한다. 실패/OOM/통신 오류: FAILED.json/log 전달 후 본 학습 보류.
Worker deadline exit 3은 torchrun parent에서 exit 1일 수 있고 outer timeout은 124일 수 있다.

```bash
 timeout --signal=TERM --kill-after=30s "${TRAIN_SECONDS:?승인 초 필요}s" env CUDA_VISIBLE_DEVICES="${GPU_UUID_0:?},${GPU_UUID_1:?}" CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 PYTHONPATH="${SOURCE_ROOT:?}/src" "${PY:?}" -m torch.distributed.run --standalone --nproc_per_node=2 -m vos_memory_inspector.lvos_ddp train --index "${INDEX_ROOT:?}/raw_index.json" --output "${TRAIN_ROOT:?새 출력 필요}" --config "${SOURCE_ROOT}/configs/lvos_ddp_2gpu.json" --mode train --global-batch 64 --device cuda --smoke-evidence "${SMOKE_ROOT:?}/STATUS.json" --max-wall-seconds "$TRAIN_SECONDS" --execute-approved --gpu-uuid "$GPU_UUID_0" --gpu-uuid "$GPU_UUID_1" --host-id "${HOST_ID:?}" --lease-root "${LEASE_ROOT:?}" --pod-hourly-rate "${POD_RATE_USD:?}" --budget-usd "${TRAIN_BUDGET_USD:?}"
```

성공: REAL_DDP_TRAIN_COMPLETED, fit/dev 두 loss, epoch별 checkpoint/export.
4 GPU는 UUID 4개/nproc_per_node=4/configs/lvos_ddp_4gpu.json으로 변경한다. CPU 4 rank는 확인했으나 NCCL 4 GPU는 미검증.

## 5. 모니터링·재개·평가팀 전달

logs/steps.jsonl: 두 raw/normalized component loss, gradient/LR, epoch/step, records/sec, ETA,
GPU UUID별 utilization/VRAM, rank별 valid 수, checkpoint 경로. metrics/history.json: fit/dev epoch 결과.
ETA는 실제 처리 속도 추정이며 dev/저장 시간은 추가될 수 있다. GPU 속도와 48시간 내 종료는 미측정이다.

Rank 0만 atomic checkpoint/export를 저장한다. 재개는 동일 TRAIN_ROOT/설정에 --resume을 추가한다.
마지막 완성 epoch의 model/optimizer/scheduler/rank별 RNG/normalization에서 재개하며 partial epoch는 다시 실행한다.
Mid-epoch exact resume는 지원하지 않는다. GPU 수 변경 시 global batch는 유지하지만 동일 trajectory를 보장하지 않는다.
Stale lease/손상 파일은 자동 삭제하지 않는다. 정확한 writer/소유권 확인 없이 재시작하지 않는다.

평가팀 전달:

- translator/epoch-000NN.pth + .complete.json: tensor-only canonical weights/config/spec.
- translator/epoch-000NN.json, translator/normalization.json.
- checkpoints/epoch-000NN.ckpt + marker, last.ckpt.json: 재개용 optimizer/scheduler/RNG.
- metrics/history.json, logs/steps.jsonl, assignments/, run.json, raw_index.json 및 사용/제외 승인안.

Export는 기존 TransformerStateTranslator.from_payload()로 strict load한다.
Best_state_loss는 dev state loss 기준이며 최종 VOS best_model이 아니다. 외부 J&F early stopping은 적용하지 않는다.
Self-injection/Direct Copy/handoff/J&F는 평가팀 담당이며 학습 조건이 아니다.
Python 종료와 Pod 과금 종료는 별개다. Pod 전체 단가는 Pod 시간에 한 번만 곱하고 Volume/idle 예약 비용은 별도 계산한다.
