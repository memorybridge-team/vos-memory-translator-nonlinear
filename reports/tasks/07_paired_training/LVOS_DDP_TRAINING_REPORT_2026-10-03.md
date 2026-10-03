# LVOS 기존 paired-state → 단일 translator DDP 구현 보고

2026-10-03 · 상태 **READY_AFTER_INPUTS**. Branch `feat/lvos-pilot-preflight`, base `f2632d7`.
최종 local commit/source artifact는 `outputs/LVOS_DDP_HANDOFF_2026-10-03.json`에 기록한다.
Push/PR/merge 없음. 기존 teammate의 untracked paired-state guide는 보존·미stage.

## 1. 변경 내용

|파일|구현|
|---|---|
|`lvos_raw_cache.py`|기존 v2 직접 loader, dataclass safe allowlist/weights_only, bytes SHA/stat, 실제 tensor 대응/finite/shape/dtype, frozen split, resumable CPU index, fit-only RMS|
|`lvos_ddp.py`|같은 translator의 DDP.forward adapter, rank별 disjoint valid records, no_sync accumulation, global valid count 가중 gradient, normalization broadcast, rank 0 atomic checkpoint/export, 완성 epoch resume|
|`configs/lvos_ddp_2gpu.json`|16 × 2 × 2 = global 64, 기존 LR/loss, FP32, augmentation 없음|
|`configs/lvos_ddp_4gpu.json`|16 × 1 × 4 = global 64, 동일 모델/loss|
|`tests/test_lvos_ddp.py`|CPU 2/4-rank, 불균등 count·padding, 단일 global batch 기준 update, 양 branch, reload, epoch 중단/재개, 제외안 차단|
|기존 `lvos_gates.py`, `lvos_cli.py`, `lvos_training.py`|앞선 공식 G0/G1 dependency 수정, state 학습 분리 및 epoch weights/config/normalization export 보존|

Model body/API는 revision `746ea3e7d84c366c2d7ac06159e90a1f684bca56`의 고정 base 그대로다.
DDP state 학습은 G5/handoff/J&F/metric adapter를 호출하거나 기다리지 않는다.
Source/config pin과 데이터 검사, 실제 DDP smoke binding은 유지한다. 정식 승인자 이력은 만들어 넣지 않는다.
전체 연구 gate 통과/최종 VOS best_model을 주장하지 않는다.

## 2. 실제 cache 검사

- 원본 roots: `/workspace/CMMT-task07-artifacts/production/lvos/cache/{fit,development}`.
- 실제 flat inventory: fit 1,488, development 315, 총 **1,803 files / 30,322,070,943 bytes**.
- 실제 CPU loading/SHA/Small→Base+ object/frame/slot/conditioning/validity/finite/모델 spec 검사를 실행했다.
  Full source 기대 목록과 정확히 일치했고 validation 혼입·fit/dev 영상 중복을 발견하지 않았다.
- **1,794 cases**가 전체 기존 prompt 조건까지 통과. 나머지 **fit 9 cases**도 tensor 대응은 통과했으나 conditioning 0이 frozen first prompt보다 앞선다.
- 실제 검사 시간 **362.455초**. 약 4.97 cases/sec, raw bytes 기준 약 79.8 MiB/sec. 학습/GPU 속도가 아니다.
- 원본 수정·삭제·재생성 없음. Marker 없음만으로 미완성 판정하지 않았다.
- 생성 revision/weights SHA/mask provenance/pair mode 미확인은 tensor 유효성과 별도로 UNKNOWN이다.
  현재 weights SHA나 `.prepare.json`만으로 생성 당시 weights를 입증하지 않는다. 다른 Pod writer 존재는 미확인이다.

검사 샘플에서 실제 관측한 spatial은 `[1,1,16,64,64,64]` BF16, pointer는 `[1,1,16,256]` FP32이며 source/target가 대응한다.
Loader는 padding을 valid indices로 제외하고 이 shape/dtype을 runtime에서 검사한다.
`raw_index.json`은 새 내부 derived 관리 파일이다. 원본 snapshot/audit가 제공됐다는 가정은 없다.
기존 strict importer의 expected_generating/completion/run_binding 요구는 공식 연구 경로에 유지하고,
운영 완료 확인 + 실제 검사를 위한 별도 state 학습 경로를 구현했다. Unsafe pickle fallback 없음.

증거:

- 로컬 `outputs/lvos_pilot_20261003T095455Z/cache_full_findings.local.json`, `prompt_details.local.json`.
- 원격 `/workspace/CMMT-lvos-isolated/lvos-ddp-inspection-20261003T110200Z/index/INSPECTION.json`, `FAILURES.json`, `prompt_details.json`.
- 초기 검사 source ZIP SHA `de6c0dce841a709ba162c7d551af9c16c9a8a5ca64bc38effc762bfbd2d09d40`.

## 3. 실제 실행 증거와 gate 상태

|검사|상태|실행 근거|
|---|---|---|
|공식 gate dependency 회귀|CPU PASS|6 passed / 13 deselected, 7.27초, exit 0; `gate_regression.log/xml`|
|최종 loader/DDP policy·2/4-rank|CPU PASS|9 passed, 78.21초, exit 0; `ddp_cpu_commitable.log/xml`|
|앞선 Linux Torch 2.8 CPU|CPU PASS|6 tests, CLI help 3개 exit 0; `.../lvos-ddp-cpu-20261003T111600Z/command-3.log`, `ddp_cpu.xml`|
|실제 cache CPU 2-rank|CPU PASS|torchrun exit 0, 3 epochs/24 updates, 양 branch 갱신, rank weights 최대 차이 0, strict model/optimizer/scheduler reload|
|전체 입력 prompt 조건|BLOCKED|fit 9개 확인 대기, 무조건 포함/자동 제외하지 않음|
|최종 Linux source 재검사|PENDING|최신 policy 추가는 로컬에서 검증; 새 Pod에서 최종 9 tests 실행 필요|
|실제 2-GPU NCCL/VRAM|BLOCKED|현재 Pod는 1 GPU. 새 endpoint/UUID 2개/Pod 전체 단가/시간·예산 없음|
|본 학습|BLOCKED|입력 정책 승인과 actual DDP smoke 필요|
|handoff/J&F/VOS best|외부 담당|평가팀 업무이며 학습 시작 조건 아님|

Local 환경: **Python 3.14.0 / Torch 2.14.0+cpu**, Windows, CUDA_VISIBLE_DEVICES="".
Linux 실제 환경: **Python 3.12.3 / Torch 2.8.0+cu128**, 별도 venv, global Torch 재사용, CUDA_VISIBLE_DEVICES="".
최종 local source와 앞선 remote source를 동일 revision PASS로 섞지 않는다.
Windows TCPStore/libuv 환경 실패는 log에 보존하고 FileStore+spawn으로 실제 Gloo 검증했다.
Linux에서는 실제 torchrun 2-rank를 사용했다.

실제 cache CPU diagnostic은 정상 fit 4 + dev 1을 index로 검사하고, fit 앞 2 cases를 학습/diagnostic dev 1 case로 사용했다.
Global batch 4/microbatch 1은 **CPU 검사 설정**이며 본 학습 설정 64/16과 구분한다.
Core 학습·reload 구간 4.392초, launcher 포함 11.239초. GPU profile/실제 속도 증거가 아니다.
Normalization은 해당 diagnostic fit 4 cases에서 한 번 계산했다. 전체 fit normalization으로 간주하지 않는다.
현재 smoke는 full fit normalization을 재사용하되 dev는 첫 case만 diagnostic으로 기록해 full-dev 반복 비용을 제한한다.
본 학습은 전체 development state loss를 기록한다.

## 4. 9개 제외안과 다음 실행

사용자는 데이터팀에 frame 0의 의미/실제 mask/Small·Base+ prompt 동일 여부를 확인하도록 지시했다.
정확한 9개 목록·질문은 `LVOS_CACHE_PROMPT_CONFIRMATION_2026-10-03.md`에 저장했다.
수신 채널/대상이 없어서 실제 외부 메시지는 보내지 않았다.
지연 시 `configs/lvos_ddp_prompt_exclusions.proposed.json`으로 **fit 1,479 / dev 315**를 제안한다.
Frozen membership은 유지하고 run별 실제 사용/제외 목록을 저장한다. Proposal 승인 없이 본 학습을 시작할 수 없다.

실행 순서: 새 Pod metadata 확인 → 최종 CPU tests → 승인된 입력 정책의 raw index → 2-GPU smoke → 고정 한도의 본 학습 → 평가팀 전달.
정확한 bash 명령·필수 변수·기대 파일은 `docs/runpod_lvos_translator_training_operator_guide.md`에 있다.
2-GPU smoke는 최대 600초를 제안하나 실제 단가/시간/예산은 운영자가 확정한다.
본 학습 30 epochs 또는 승인된 wall deadline. Mid-epoch exact resume는 지원하지 않는다.
No J&F early stopping. Best_state_loss와 평가팀 VOS best_model은 분리한다.

## 5. 비용·저장·운영 제한

GPU 작업 시간 **0초**. 현재 1-GPU Pod의 사용자 확인 예약 단가는 USD 0.744/hour, Volume 별도다.
362.455초 CPU 검사 시간에 대한 단순 Pod 예약 환산은 약 **USD 0.0749**이며 실제 invoice/전체 세션 비용이 아니다.
Reserved idle/설치/업로드/CPU 검사 시간도 Pod 과금에 포함된다. Python 종료는 Pod 과금 종료가 아니다.
새 2-GPU Pod는 **전체 Pod-hour 단가를 한 번만 적용**하고 GPU 수를 다시 곱하지 않는다.
최종 단계에서는 이전 원격 120분 창이 지나 로컬 작업만 수행했다. 마지막 원격 CPU job은 이미 exit 0으로 종료됐다.
종료/추가 Pod/Volume 변경은 하지 않았다.

Raw cache 약 28.24 GiB는 복제하지 않았다. Derived index는 작은 JSON이고,
실제 CPU diagnostic checkpoint body 하나는 **5,560,451 bytes**였다. 미완성/완성 파일은 보존했다.
최종 GPU VRAM/records/sec/epoch 시간/48시간 내 종료 예측은 실제 smoke/profile 후 확인한다.

## 6. 검증 명령

Local PowerShell, repo에서 실행한 최종 command:

```powershell
$env:PYTHONPATH='src'; $env:CUDA_VISIBLE_DEVICES=''; $env:PYTHONUTF8='1'; $env:PYTHONDONTWRITEBYTECODE='1'; .\.venv\Scripts\python.exe -m pytest tests/test_lvos_ddp.py -q -p no:cacheprovider --basetemp C:\Users\SAMSUNG\Documents\Codex\pddp103l --junitxml ..\..\outputs\lvos_pilot_20261003T095455Z\ddp_cpu_commitable.xml
```

다시 실행할 때는 기존 temp를 삭제하지 않고 새 전용 basetemp로 변경한다.
CLI index/train --help도 exit 0으로 검증했다. GPU 명령은 미실행이다.
실제 SAM2 실행·추론·GT J&F·전체 연구 gate PASS는 이 보고의 근거에 포함되지 않는다.
