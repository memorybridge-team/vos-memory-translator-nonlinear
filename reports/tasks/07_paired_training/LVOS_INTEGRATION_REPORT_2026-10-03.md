# LVOS 운영 복구·benchmark 연결 로컬 보고

**2026-10-03 KST · READY_AFTER_INPUTS · branch `feat/lvos-training-gates`**

기반 commit은 `f77965bd060017beedb930d1916e6ce948a4861c`다. 최신 사용자 승인에 따라 로컬 구현·CPU 검사·로컬 commit을 수행한다. 원격 push/PR/merge, SSH, GPU, Pod/storage 변경과 기존 수집 변경은 수행하지 않았다. 최종 commit·전달 ZIP/patch SHA와 fresh checkout/archive 실행 결과는 공유용 `LVOS_INTEGRATION_HANDOFF_2026-10-03.json` 및 Markdown에 저장한다. 실제 runtime 경로/weights는 접근할 수 없어 검증하지 못했다.

## 1. 실제 근거와 결정

|항목|읽기 전용 확인|적용/남은 결정|
|---|---|---|
|Training main|`55b117dcae4a0102b92382ed06cbee4f1fc3956c`|Main 직접 수정/merge하지 않음|
|Published training branch|`5be4fba5ec2e7df3c95178639f02b3078388b0f3`|기반 f77965b의 ancestor, 0/3 commits 차이. dependency를 포함한 정상 PR 검토 필요|
|Model reference|`746ea3e7d84c366c2d7ac06159e90a1f684bca56`|Spatial Transformer base·별도 pointer MLP와 tensor API body 유지. 실제 승인자/승인 기록 미제공|
|Benchmark remote HEAD|`d268d82ca5293cc66a3bfbd7d194b952083bd820`|Default HEAD와 실험용 reference를 혼동하지 않음|
|Metric reference|`bcf0a0f6a36c4129d487e5e58e151468d7ca714b`|실제 `best_model.selection_score`를 final/table 공통 primary로 호출|
|Baseline reference|`dcd335380dbe62f3299dbc5d4456b5ac8a12b4b4`|`baseline/no_handoff.py::full_replay`가 존재. Generator 자체 부재가 blocker가 아님|
|Historical benchmark|`8d0a5cd9f019c9270e4917e3dbe7a479d0cf9276`|과거 결과를 현재 model/protocol의 실측으로 재사용하지 않음|
|Frozen split|LVOS official train 420 영상, fit 347/development 73|Membership/source manifest 변경 없음. Baseline MD5 재분할을 적용하면 fit→dev 64/dev→fit 60이므로 bypass|
|Scientific contract|Small→Base+ one-way, no-replay, native-history independent O=1, `cmmt.small_to_base_plus.io.v1.1`|K를 고정하지 않음. ID/frame/slot/conditioning/validity 보존. Target PE 재생성, source masks/presence 미주입|
|Legacy metadata|확인 가능한 generating/conditions/case-binding 또는 reviewed immutable run evidence|Filename에서 model SHA/승인/완료를 만들어 넣지 않음|

모델 구조와 frozen manifest의 diff가 비어 있는지 확인했다. 기존 untracked `docs/runpod_paired_state_operator_guide.md`는 수정/stage하지 않는다. Git reference는 source 근거이며 배포된 Pod executable revision 증거는 아니다.

### 줄바꿈 결함의 실제 관찰

PM이 지적한 raw source SHA의 플랫폼 의존성은 재현됐다. 다만 보존된 f77965b ZIP을 직접 읽으면 model 1,079줄/API 356줄이 CRLF이고 현재 Git blob/작업 파일은 LF였다. 변환 방향에 관계없이 바이트 차이가 원인이다. UTF-8 검증 후 **CRLF만 LF로 정규화**하는 `utf8_crlf_to_lf_v1`을 source에 적용했다. 실제 weights/shard/checkpoint/ZIP은 원래 raw SHA256을 유지한다. BOM/공백/코드 변경을 제거하거나 integrity gate를 생략하지 않는다. Model-lock v1은 v2로 재생성하고 실제 source-bound 승인을 새로 받아야 한다.

## 2. 구현한 순서와 파일

1. **운영 결함 4개:** `training_storage.py`/`training_runner.py`의 canonical source SHA, `.gitattributes`, `lvos_checkpoint.py`의 journal/latest-valid 복구, `lvos_budget.py`의 단일 deadline, `lvos_cli.py`의 gate FAIL=2/BLOCKED report-only=0/require-ready=3. Checkpoint identity·optimizer·scheduler·RNG·RMS를 유지한다. Epoch-0도 복구하며 corrupt/foreign/orphan 근거를 자동 삭제/덮어쓰지 않는다.
2. **Discovery:** `lvos_discovery.py`의 explicit root·completion/stability/SHA·generating/frozen case 검사와 audit requests/inventory. 원본에 쓰지 않는다. Unknown/rejected/duplicate/missing를 분리하고 expected case/완료 pair/valid record/bytes를 별도로 기록한다. Legacy pickle은 명시된 trusted-team 승인만 허용한다.
3. **Full Replay/adapter:** `lvos_benchmark.py`는 별도 pinned baseline의 실제 entrypoint만 호출한다. Frozen full-dev prompt/switch/scored-frame map과 target/read-policy를 사용하고, 같은 video/object의 세 fraction은 한 rollout에서 채점한다. Original initial prompt 외 GT는 모델 입력에 넣지 않는다. Learned/Direct Copy는 기존 canonical injector와 target t+1 경로를 사용한다. 과거 target encoding은 Full Replay에서만 허용한다.
4. **Canonical metric:** `lvos_metrics.py`를 final selection/통합 비교표 모두에서 사용한다. Object 평균→video Method/Replay→video 평균→fraction 평균이다. Raw J/F/JF를 유지하고 valid >100을 clipping하지 않는다. Undefined/missing/zero Replay 규칙을 검사한다. 기존 50%/≤64-frame `monitor_jf_proxy` [0,1]과 min_delta=.001은 유지한다. State best/monitor best/strict full-dev shortlist best를 구분한다.
5. **Explicit worker/merge 이후 순차 자동 연결:** `lvos_jobs.py`의 immutable request v2, dry-run default worker, disjoint video shards, per-run/worker/request/attempt outputs, verified exactly-once merge, crash/retry/timeout/recovery를 먼저 검증했다. Training consume에서도 request marker/schema/digest와 evaluator source SHA를 검사한다. `train --evaluation-mode sequential`은 exclusive host/UUID lease 아래 실행하며 callback 전후 RNG를 보존한다. Planner는 실행기로 바꾸지 않았다.

추가 전달 산출물: `lvos_release.py`의 small strict reproducibility bundle, `configs/lvos_benchmark_reference_lock.json`, updated job template, 기존 가이드의 historical banner와 신규 한국어 운영 가이드. 원본 benchmark의 fixture 사본 3개의 저자는 RohSeongmin이며 출처를 `MIGRATION.md`에 기록했다.

## 3. 실제 CPU 실행 증거

전체 working-tree CPU 회귀는 `196 passed in 355.08s`였다. 이후 최종 request/source 검증 보완은 focused regression `14 PASS, 92.37초`와 최종 source-bound CPU driver `77 PASS, 246.99초`로 검증했다. Source SHA는 `00413bbfe9b3421a506a8c550cc577135da49448b939d678b539eef63a096ac3`, suite SHA는 `dc9c2b2c53d2392be074428ada9b1ee7df1d883ff9085a7eddfd6ee3de2836d1`이다. Fresh checkout/archive 결과는 동봉 handoff/index를 따른다. 서로 중복되는 suite의 PASS 개수를 더하지 않는다.

```powershell
$env:PYTHONPATH = 'src;.'
```

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider --basetemp C:/Users/SAMSUNG/Documents/Codex/it10j --junitxml ../../outputs/lvos_integration_full_regression.xml
```

```powershell
.\.venv\Scripts\python.exe -m vos_memory_inspector.lvos_cli cpu-test --output ../../outputs/lvos_integration_cpu_verified --basetemp C:/Users/SAMSUNG/Documents/Codex/ti03
```

CPU 검사는 canonical source bytes/architecture, checkpoint 저장 중단 경계, deadline/exit, fixture discovery/원본 보존, sparse/late prompt Full Replay trace, split/metric 반례, worker 소유권/미완료/중복/foreign source, pending status, strict bundle load를 대상으로 수행했다.

초기 결함과 수정 중 실패 log/XML도 보존했다. 최종 driver의 한 차례 실패(`75 PASS/1 FAIL`, 270.38초)는 긴 Windows temp 아래 worker namespace가 파일명 길이 제한을 넘은 경우였다. `cpu-test --basetemp`에 새 짧은 전용 경로를 지정하도록 보완한 후, 기존 temp 보존 검사와 전체 suite를 다시 실행하여 77 PASS를 확인했다. 기존 cache/temp를 삭제하여 해결하지 않았다.

문서 검사에서는 `--help` 24개, CLI block 29개의 parser, JSON helper 2개, 4-job dry-run planner를 실행했다. GPU/remote 명령은 syntax/help만 확인했으며 실제 운영 준비 완료는 아니다. 실제 GPU 속도·storage cost·J&F 개선은 미측정이다. CPU fixture 파일 크기와 경과 시간은 handoff/index에 실측을 기록한다.

## 4. GPU gate와 필요한 입력

|검증|로컬 확인 범위|실제 상태/담당|
|---|---|---|
|G0 model/metric freeze|고정 source/config와 미승인 proposal|BLOCKED: 모델/Benchmark 팀 승인 기록|
|G1 real audit|fixture alignment/SHA/fit-dev disjoint와 거절 동작|BLOCKED: 수집 담당의 완료 cache/conditions/bindings와 실제 읽기 접근|
|G2 real pair update|CPU forward/backward·component loss/padding/RMS/accumulation|BLOCKED: 실제 pair, Small/Base+ weights SHA, free GPU|
|G3 loss logic|CPU synthetic 실행|PASS 실행 evidence는 final gates/log 참조|
|G4 actual checkpoint roundtrip|CPU strict load/output/optimizer/scheduler/RNG/RMS parity|BLOCKED: real-checkpoint GPU roundtrip|
|G5 target handoff|CPU injector/fake Replay/frame trace·frozen protocol 검사|BLOCKED: learned epoch-0와 canonical Direct Copy의 실제 encoder IDs/continuation/완전 coverage|
|G6 CPU safety|source/suite-bound pytest report|PASS 실행 evidence는 final gates/log 참조|
|Full Replay/full-dev promotion|CPU adapter/reducer/merge/strict bundle|BLOCKED: 승인된 target/policy/protocol의 실제 GPU rows, Benchmark strict loader acceptance|

Pod 운영자는 현재 endpoint/접근 범위, topology/free physical UUID, 시간·실제 rate/통화·job과 전체 예산, persistent Network Volume/mount 실측을 제공해야 한다. 데이터 담당자는 영상별 RGB/PNG paths, 기존 worker code/PID/install path와 selection digest를 제공해야 한다. `/workspace`나 과거 ZIP만으로 network volume/deployed revision을 확인했다고 판단하지 않는다.

후보 root는 `/workspace/CMMT-task07-artifacts/production/lvos/cache/{fit,development}/` 와 `/workspace/CMMT/data/LVOSv2/extracted/`다. Checkpoint 기대 이름은 `sam2.1_hiera_small.pt`/`sam2.1_hiera_base_plus.pt`이지만 실제 path/SHA는 미확인이다. Placeholder를 추정으로 채우지 않는다. 상세 절차는 [신규 운영 가이드](../../../docs/runpod_lvos_integration_operator_guide.md)에 둔다.

## 5. 비용·중단·결과의 한계

단일 deadline은 준비/hash/RMS/shard/batch/state-dev/eval frame/wait/finalization에서 검사한다. 개별 hash/load/serialization/kernel이나 DataLoader 종료는 선점하지 못해 해당 처리만큼 초과할 수 있다. STOPPED와 부분 dev를 완전 평가로 취급하지 않으며 mid-epoch exact resume를 주장하지 않는다.

Python 종료와 Pod 과금 종료는 별개다. 기존 collector/shared Pod를 자동 중지하지 않는다. Whole-Pod rate는 Pod-hour당 한 번, 별도 과금은 각 실제 rate/runtime을 합산한다. GPU 8개×72시간=576 GPU-hours는 모두 전 구간 예약된 경우다. 과거 150,000 KRW는 개인 예산 가정이며 현재 team run에 새 상한으로 적용하지 않는다. 실제 throughput/비용/재시작 후 volume 가독성은 BLOCKED다.

PASS는 실행 log/XML/checksum을 동반하는 CPU/fixture engineering 결과다. 실제 checkpoint GPU 검증이나 VOS 개선 증거는 아직 없다. 최종 selection 주장은 “평가한 shortlist 안에서 최고”다. Joint multi-object equivalence, global best epoch, Full Replay가 항상 상한이라는 주장은 하지 않는다.

## 6. 원격 공개 검토용 계획 — 미실행

현재 published training branch보다 본 branch가 dependency를 포함해 진행되어 있다. f77965b만 cherry-pick하는 절차를 제안하지 않는다. 팀의 PR base/merge 순서를 확인하고 승인 후 정상 push→PR로 진행할 수 있다. 다음 명령은 현재 **승인 대기·미실행**이다.

```powershell
git push origin HEAD:refs/heads/feat/lvos-training-gates
```

```powershell
gh pr create --base feat/training-pipeline-small-baseplus --head feat/lvos-training-gates --title "LVOS operations recovery and canonical benchmark bridge" --body-file C:/Users/SAMSUNG/Documents/Codex/2026-09-27/you-are-the-implementation-workspace-for/outputs/lvos_integration_proposed_pr_body.md
```

PR base는 published training branch를 임시 후보로 삼았으며 main 채택 순서는 팀 결정이 필요하다. 검토 항목은 architecture/split diff=0, source hash portability, fault recovery/deadline/exit, native-history 분리, canonical metric, request/source identity, 원본/현재 worker 보존, 실제 GPU BLOCKED 명시, 출처와 artifact 제외다.
