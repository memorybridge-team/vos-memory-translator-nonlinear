> **갱신 2026-10-03:** 이 문서는 `f77965b` 당시의 기록입니다. 현재 integration의 source-lock/worker/budget CLI는 [새 운영 가이드](runpod_lvos_integration_operator_guide.md)를 따릅니다. 아래 과거 GPU 명령은 현재 runtime에 바로 적용하지 않습니다.

# 내일 LVOS GPU gate를 시작하는 짧은 안내

**2026-10-03 · READY_AFTER_INPUTS · `feat/lvos-training-gates`**
통합 기준 `4a1bb1b`; 최종 로컬 commit은 공유용 `LVOS_FINAL_HANDOFF_2026-10-03.json` 참조.

로컬 CPU suite **31 PASS**, 전체 회귀 실행 **151 PASS**와 synthetic overfit/dev/strict reload를 확인했습니다. 실제 LVOS cache·SAM weights·GPU handoff 결과는 아직 없습니다. **G3/G6는 CPU PASS, G0/G1/G2/G4/G5는 BLOCKED**입니다. 상세 [검증 보고서](../reports/tasks/07_paired_training/LVOS_PIPELINE_REPORT_2026-10-03.md)에 실행 log와 한계가 있습니다.

## 1. 먼저 받을 값

|담당자|필수 값|
|---|---|
|Pod 운영자|현재 SSH host/port, 접근 승인, 기존 PID/session과 topology, free GPU UUID, 허용 초·요금, `/workspace` mount/free space|
|수집 담당자|완료 cache 경로/SHA/marker, case별 generating conditions·producer revision, frozen inventory|
|모델팀|최종 base config/factory 승인, Small/Base+ checkpoint 파일·SHA, pinned SAM2 checkout 경로|
|Benchmark/연구팀|12개 monitor dev 영상/GT 경로·coverage, metric/export 승인, Full Replay 생성 코드/결과 계약|

`REQUIRED_*`는 미확인 값입니다. 입력 전에는 GPU 명령을 실행하지 않습니다. 기존 cache/split/수집 환경을 유지합니다. Python 종료와 Pod 과금 종료는 별도입니다.

## 2. 접속 후 첫 확인

**로컬 PowerShell:** 현재 endpoint와 개인 key 경로만 바꿉니다. Private key는 공유하지 않습니다. 접속 실패 시 Pod 운영자가 public key 등록과 Dashboard endpoint를 확인합니다.

```powershell
ssh root@REQUIRED_CURRENT_HOST -p REQUIRED_CURRENT_PORT -i REQUIRED_PERSONAL_PRIVATE_KEY_PATH
```

**RunPod bash:** GPU/PID/mount가 실제 입력과 일치하는지 확인합니다. 실패/충돌 시 새 job을 시작하지 않고 결과를 운영자에게 전달합니다.

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv
```

```bash
ps -eo pid,ppid,etime,args
```

```bash
findmnt -T /workspace
```

## 3. 실행 순서

새 checkout/interpreter를 준비한 후 아래 순서대로 진행합니다. 제안 checkout `/workspace/CMMT-lvos-isolated`는 원격에서 아직 확인하지 않은 경로입니다. 기존 worker가 쓰는 환경에 `git pull`/`pip install -e`를 하지 않습니다.

|순서|작업·성공 기준|명령 위치|
|---:|---|---|
|1|새 환경 help/CPU suite; exit 0, 현재 source/test SHA report|[상세 가이드 §4](runpod_lvos_training_operator_guide.md#4-격리-코드환경--운영자-승인-후-준비)|
|2|완료 inventory audit→새 view; 전체 expected=completed, unknown/rejected=0|가이드 §5 `audit`/`verify`|
|3|모델/metric 승인·monitor GT coverage 동결→epoch-0 checkpoint|가이드 §6 `freeze-model`/`metric-template`/`freeze-protocol`/`initialize`|
|4|승인된 free GPU에서 실제 pair G2/G4→짧은 one-video 학습|가이드 §6.3 `gates`/`overfit`|
|5|Saved checkpoint를 실제 Base+에 reload→identity/Direct Copy handoff→dev 점수|가이드 §7 `evaluate`/`merge-eval`; encoder frame IDs `>t`, 처음 `t+1`, complete coverage|
|6|G0–G6 모두 실제 근거 PASS→profile→독립 A/B와 평가 D|가이드 §8; Job C baseline producer는 미제공으로 BLOCKED|
|7|상위 trained epoch shortlist의 complete full-dev만 export|가이드 §10 `shortlist`/`select`; state/monitor best와 구분|

**새 환경의 첫 CPU 명령 (RunPod bash):** cwd가 검토된 격리 checkout인지 먼저 확인합니다. 성공하면 실제 cache inventory 단계로 이동합니다. 실패하면 `pytest.log`의 첫 오류부터 수정합니다.

```bash
.venv/bin/python -m vos_memory_inspector.lvos_cli cpu-test --output /workspace/CMMT-training-inputs/lvos/cpu-suite
```

GPU 명령은 `CUDA_VISIBLE_DEVICES=확인한_UUID`, process 내부 `cuda:0`, `--execute-approved`, 실제 숫자 `--max-wall-seconds`와 별도 output을 사용합니다. 명령 전문·expected files·실패 조치는 [전체 운영 가이드](runpod_lvos_training_operator_guide.md)에 있습니다. Partial epoch는 마지막 complete epoch부터 재개하며 mid-epoch exact resume는 지원하지 않습니다. State loss 감소를 VOS 개선으로 보고하지 않습니다.
