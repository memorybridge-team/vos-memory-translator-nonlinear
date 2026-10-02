# LVOS 고정 모델 pipeline 로컬 구현·검증 보고

**2026-10-03 KST · READY_AFTER_INPUTS · branch `feat/lvos-training-gates`**

요청한 문서와 당일 추가 지시에 따라 로컬 구현·CPU 검사·로컬 commit까지 수행했다. 실제 LVOS cache/Small·Base+ weights/현재 SSH endpoint는 제공되지 않았다. GPU·원격 배포·push/PR·Pod 변경·기존 수집 변경은 수행하지 않았다. 최종 commit과 patch/source artifact SHA는 공유용 `outputs/LVOS_FINAL_HANDOFF_2026-10-03.json`에 기록한다.

## 1. 확인한 계약과 남은 계약

|항목|확인 근거·판단|
|---|---|
|기존 코드 기준|Collector 통합 `4a1bb1b06ec8d676184a8f6e770550e27aafacbd` 위에 task branch 생성. Main과 teammate 변경 보존|
|모델 구조/API|모델팀 `feat/transformer-adapted-translator` pin `746ea3e7d84c366c2d7ac06159e90a1f684bca56`. Spatial context Transformer/tensor API 이관, architecture class AST 동일. Base params 실제 CPU count **457,216**|
|입출력|Spatial `[B,O,K,64,64,64]`, pointer `[B,O,K,256]`; train FP32, handoff spatial bf16/pointer fp32. Validity/frame/slot/conditioning/object/switch 유지. Source mask/presence 미주입, target PE 재생성|
|데이터 계약|기존 LVOS official-train source/fit/dev manifest의 content SHA와 membership 유지. Native-history·독립 O=1, mixed conditions 거절. Controlled same-mask 미혼합|
|낡거나 다른 계약|AGENTS의 Tiny/DAVIS 범위는 최신 Small/LVOS 지시로 대체. 기존 sampled Linear runner는 whole-record spatial Transformer에 적합하지 않아 기존 코드를 보존하고 별도 LVOS runner 사용|
|Benchmark|별도 checkout `bcf0a0f6a36c4129d487e5e58e151468d7ca714b`의 실제 `best_model.py` 확인. 영상별 method/replay 비율→fraction 평균 함수 사용. README의 다른 집계 설명 대신 실제 함수 pin을 계약에 명시|
|아직 미확정|모델팀 최종 base preset/config 승인, benchmark metric/monitor 승인, strict export loader의 Benchmark adapter 수용. 모두 입력 전 BLOCKED|
|Job C|해당 benchmark checkout은 점수 함수/문서만 제공. Baseline 생성기/실행 adapter 없음. 임시 baseline이나 가짜 점수를 구현 완료로 표시하지 않음|

최종 검사 package source SHA:
`add5a452fd1ede61f613a2729a1dd3ab5005f9f1b10e752f39561d0f8811f3a4`.
Code/test SHA를 CPU report에 기록하고 G6에서 현재 파일과 비교한다. 검증 후 source 변경 시 기존 PASS를 재사용할 수 없다.

## 2. 구현 파일

|파일|실제 기능|
|---|---|
|`transformer_translator.py`, `frozen_tensor_api.py`|모델팀 구조/gradient-preserving tensor API. 기존 `translators.py` 미수정. Source pin/SHA는 `configs/lvos_reference_lock.json`|
|`lvos_contract.py`|Complete model lock, strict/weights-only artifact·completion/SHA, evidence·JSONL·resource 기록, pinned metric loader|
|`lvos_snapshot.py`|완료 cache audit→원본 밖 bounded tensor views. Full selection/membership 고정, actual/padded/bytes/duplicates 집계. Source/target metadata 대응 검사|
|`lvos_training.py`|Fit-only streaming target RMS, separate per-record spatial/pointer MSE, AdamW/no-decay, record-weighted accumulation, warmup/cosine/clip, epoch-boundary strict resume, dev-state loss|
|`lvos_evaluation.py`|Score 이전 sparse GT protocol 동결, 실제 target injector 진입점, encoder frame-ID hook, disjoint video shards/merge, ordered early stopping, top-3 shortlist, full-dev-only strict export|
|`lvos_gates.py`|G0–G6 상태·실행 종류·근거 SHA. Real G1/G2/G4/G5와 G0/G3/G6를 확인하기 전 unattended main 학습 거절|
|`lvos_cli.py`, `pyproject.toml`|실행 가능한 audit/verify/initialize/gates/overfit/profile/train/resume/evaluate/merge/shortlist/select/launch-plan/draft-export. `cmmt-lvos-pipeline` entry point|
|`sam2_lazy_loader.py`|No-warmup initializer가 실제 쓰는 `sam2.utils.misc` alias도 current process에서 bounded loader로 연결|
|`configs/lvos_job_{a,b}.json`|lr 3e-4/1e-4만 다르게 설정. Seed/data/order/architecture 공통. FP32, augmentation none, micro16/accum4, max30|
|`configs/lvos_{runtime,jobs}.template.json`|필수 unknown 값을 `REQUIRED_*`로 표시. A/B/C/D independent command plan, free UUID/namespace 확인 필요|
|`tests/test_lvos_pipeline.py`, `scripts/lvos_cpu_smoke.py`|CPU synthetic 회귀와 보존 가능한 one-video overfit/video-disjoint dev smoke|
|`docs/runpod_lvos_training_operator_guide.md`|첫 실행 순서, 정확한 명령/placeholder/담당자/실패 조치/재개/과금·Pod 종료 별도 확인|

Sampler는 완료 **paired case만 shuffle**한다. 한 case 내부의 시간·slot 순서는 유지하며, 큰 case의 bounded shard를 같은 worker가 순서대로 읽는다. Valid absent record를 제외하지 않는다. Statistical duplicate는 보고하고 기존 record weight를 유지한다. Pod 기본 workers=4/prefetch=2/persistent/pin, CPU workers=0. Custom batch의 pin callback을 검사했다.

`last.ckpt.json`, `best_state_loss.ckpt.json`, `best_monitor.ckpt.json`을 구분한다. `best_model.pth`는 real, complete full-development shortlisted candidate·known checkpoint SHA·matching digests·승인된 strict loader가 있어야 생성한다. Immutable checkpoint/eval 결과를 worker별 별도 output에 기록하고 coordinator가 case 단위로 합친다. Launcher는 계획만 저장하며 자동 실행 기능은 없다.

## 3. 이번에 실제 실행한 검사

위치: 로컬 PowerShell, 저장소 `work/repo`. Python **3.14.0**, Torch **2.14.0+cpu**, NumPy **2.5.3**. 실제 SAM checkpoint나 GPU 미사용.

```powershell
$env:PYTHONPATH='src;.'
```

```powershell
$env:PYTHONUTF8='1'
```

```powershell
.\.venv\Scripts\python.exe -m vos_memory_inspector.lvos_cli cpu-test --output C:/Users/SAMSUNG/Documents/Codex/lvos_cpu_release_evidence_20261003
```

**PASS: 31 tests, 127.34s**. Driver wall 131.01s. 실제 log/JUnit/checksum/source·test SHA:
`C:/Users/SAMSUNG/Documents/Codex/lvos_cpu_release_evidence_20261003/{pytest.log,junit.xml,report.json}`.

검사 내용: identity/초기 gate·output layer 및 이후 deep gradient, NaN padding, independent loss reference, pointer element 수 독립성, unequal accumulation/partial window, fit-only RMS(다른 dev target), full-record metadata 정렬, frozen membership/official validation 혼입 방지, absent valid record 유지, persistent 2-worker coverage/slot 순서, completion/손상/원본 변경, exact epoch-boundary weight/optimizer/scheduler/RNG/output/history resume, incomplete/foreign eval·duplicate shard·namespace/early-stop/full-dev promotion 거절, strict export payload reload, CLI nonzero failure/help.

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider --basetemp C:/Users/SAMSUNG/Documents/Codex/lvfull3 --junitxml ../../outputs/lvos_full_regression_2026-10-03.xml
```

**PASS: 전체 151 tests, 307.70s**. 이 실행은 LVOS 29개를 포함한다. 이후 target config hash 경로·Pod 간 namespace 회귀 2개와 evidence SHA 보호를 검토하고, 최종 LVOS 31개를 위 명령으로 재검사했다. 서로 다른 실행의 test 수를 합산하지 않는다.

첫 LVOS 실행은 **22 PASS / 1 FAIL**이었다. Existing lock의 실제 예외가 `FileExistsError`인데 test가 `RuntimeError`를 기대한 오류를 수정했다. Model import의 미존재 MomentMatched dependency는 해당 registry 분기로 이동했다. 이후 최종 테스트에는 실패가 없다.

Benchmark pin의 실제 함수도 CPU에서 4×4 synthetic mask와 synthetic score rows로 실행했다. `lvos_frozen_benchmark_cpu_2026-10-03.json`에 source SHA/입력 종류/함수 결과를 저장했다. 이는 actual metric code 연결 검사이며 실제 VOS/Full Replay 성적이 아니다.

## 4. 실제 저장한 CPU vertical slice

```powershell
.\.venv\Scripts\python.exe scripts/lvos_cpu_smoke.py --output ../../outputs/lvos_cpu_release_smoke_2026-10-03
```

**PASS — cpu_synthetic**. `snapshot/`, `overfit/`, `dev/`, strict checkpoint, `draft.md`, `summary.json` 저장.

|관측 항목|실제 값/한계|
|---|---|
|선택한 synthetic case|Fit `train:0ClBYzYm:obj1:switch1221`; dev `train:1umdNtrE:obj1:switch961`. 원 frozen case ID 사용, 픽셀/state는 명시적 synthetic|
|실제 완료 tensor|각 split 1 case, 1 video, valid 2 records, padded 1 record. 전체 LVOS 1,803개 완료로 표시하지 않음|
|Normalization|Fit target RMS spatial=2/pointer=2; dev target=3을 통계에 포함하지 않음|
|Fit-only overfit|3 complete epochs; normalized fit loss `0.5 → 0.48178786 → 0.45995635`|
|Video-disjoint dev state loss|`1.96323466 → 1.91818643 → 1.89120698` (toy target) |
|Saved dev checkpoint SHA|`0807ce64f0ec2f54bead4efa56c2f4c8b06f202c1351f1f045adc8574218cde8`|
|전체 smoke wall|**20.2456s**; fixture RGB 생성/audit/학습/dev/checkpoint/draft 포함. GPU throughput으로 환산하지 않음|
|저장 비용|전체 smoke outputs **50,440,267 bytes** (약 48.1MiB). Raw cache 6,422,462 bytes; train tensor payload 4,202,564 bytes. 실제 Pod 데이터 크기/요금 아님|
|실제 GPU 비용/ETA|미측정·미확인. 유료 GPU 실행 없음. 요금·reserved 시간·storage/환율은 운영자 입력 필요|

**State loss 감소는 VOS 개선 증거가 아니다.** 실제 LVOS GPU rollout/J&F/성능 비교 값은 대기이다. CPU fake predictor의 frame-ID/no-replay 테스트는 실제 Base+ G5 PASS가 아니다. GPU profile 100–300 micro-iterations, source/target checkpoint consistency, active policy equivalence, persistent-volume restart는 아직 확인하지 않았다.

## 5. Gate 상태와 실제 근거

```powershell
.\.venv\Scripts\python.exe -m vos_memory_inspector.lvos_cli gates --output ../../outputs/lvos_release_gates_2026-10-03 --cpu-test-report C:/Users/SAMSUNG/Documents/Codex/lvos_cpu_release_evidence_20261003/report.json
```

|Gate|현재 상태|실행 종류·근거/필수 입력|
|---|---|---|
|G0|BLOCKED|Model/metric 최종 승인·Pod 환경/weights provenance 미제공. Pin/config proposal과 source AST는 CPU에서 확인|
|G1|BLOCKED|실제 완료 LVOS cache inventory 없음. CPU synthetic audit/누수·정렬 실패 검사는 JUnit에 있음|
|G2|BLOCKED|Real cache + approved frozen model + GPU 필요. Full64 records CPU 3-step gradient/delta 검사는 실행됨|
|G3|PASS|`cpu_synthetic`; `lvos_release_gates_2026-10-03/G3.json`, actual independent numeric reference·padding·window 검사|
|G4|BLOCKED|Real-checkpoint GPU round trip 필요. CPU strict reload/output/update/epoch resume/손상 거절은 JUnit과 saved smoke checkpoint로 확인|
|G5|BLOCKED|실제 Base+ identity/Direct Copy monitor handoff results 없음. Hook/injector 진입점은 CPU fake predictor로만 검사|
|G6|PASS|`cpu_synthetic`; `.../G6.json` 및 source/test SHA가 일치하는 실제 31-test report/log/JUnit. 실제 다중 GPU 성능이나 network-volume 실측 PASS 아님|

Machine-readable: `outputs/lvos_release_gates_2026-10-03/gates.json`와 `logs/events.jsonl`.
각 PASS는 실제 evidence 파일 SHA와 연결되어 있다. `unattended_training_ready=false`. Missing stage를 0점·성공으로 채우지 않았다.

## 6. 내일 필요한 입력·담당자와 실행 순서

1. **Pod 운영자:** 현재 SSH host/port·접근 범위, topology/기존 workers·free UUID, `/workspace` mount/free space, 허용 wall seconds/요금. Private key 공유 불필요.
2. **수집 담당자:** 완료 case별 경로/SHA/completion·generating conditions·원 실행 revision/모델 SHA, frozen selection inventory. Unknown provenance는 근거로 해결하며 metadata를 추정하지 않는다.
3. **모델팀:** 최종 base preset/factory/complete config/source SHA 승인, Small/Base+ 파일명·SHA/upstream pin. API는 이관됐으나 최종 승인자는 미입력이다.
4. **Benchmark/연구 담당자:** GT source/영상별 경로, score 이전 monitor 12개 영상·coverage, frozen primary/undefined/zero-replay 규칙과 monitor metric 승인, Full Replay producer/result schema, export adapter 수용 승인.

순서: 읽기 전용 환경 점검 → 격리 interpreter CPU tests → real full audit/view → model/metric/monitor 동결 → epoch-0 initialization → real pair G2/G4 및 one-video overfit → saved checkpoint real target reload/no-replay identity+Direct Copy → complete monitor merge/G5 → 모든 gate PASS → profile → independent A/B + benchmark C + eval D → full-dev shortlist/export.

정확한 명령·수정할 값·expected files·실패 조치는 [운영 가이드](../../../docs/runpod_lvos_training_operator_guide.md)에 있다. 모든 GPU 명령은 별도 승인과 실제 숫자 cap 입력 전에는 대기이다. Default launcher는 dry-run plan이며 Pod 구매/종료를 하지 않는다.

## 7. 남은 구현·검증 제한

- Automatic inventory generator는 구현하지 않았다. 현재 CPU audit는 명시적 `requests.json`을 받는다. 요구 schema는 operator guide의 배열이며, 기존 status/case binding/conditions와 실제 완료 checksum을 case_id로 묶어야 한다. 이는 수집 담당자 입력 또는 후속 helper 작업이다.
- Job C baseline generator는 benchmark source에 없어 연결 계약/구현을 해당 팀이 제공해야 한다. 임시 J&F/baseline adapter로 대체하지 않는다.
- Export는 모델팀의 실제 payload/from_payload 형식을 사용하지만 benchmark adapter 수용은 승인 전이다. 단순 파일명 변경을 호환 검증으로 표시하지 않는다.
- FP32 v1이다. AMP/augmentation/cosine loss/DDP/mid-epoch exact resume는 추가하지 않았다. Partial epoch는 last complete epoch부터 다시 시작한다.
- Runtime cap은 micro/frame 경계에서 검사한다. 진행 중 kernel/model load/file read를 선점하지 않으므로 hard real-time 상한이 아니다. Python 종료와 Pod 과금 종료는 별도이다.
- Local outputs에는 synthetic states/checkpoints가 있으며 Git에 포함하지 않는다. 기존 untracked `docs/runpod_paired_state_operator_guide.md`를 변경하거나 commit하지 않는다.

## 8. 공유 파일과 검증된/대기 명령

- 저장 guide: `docs/runpod_lvos_training_operator_guide.md`; 동일 outputs copy 제공.
- 첫 접속용 짧은 guide: `docs/runpod_lvos_quickstart_ko.md`; 동일 outputs copy 제공.
- Saved sample manifest: `outputs/lvos_cpu_release_smoke_2026-10-03/snapshot/snapshot.json` (**synthetic pilot**).
- Actual draft: `outputs/lvos_cpu_release_smoke_2026-10-03/draft.md`. GPU comparisons는 대기.
- Patch/source ZIP·local commit·SHA: `outputs/LVOS_FINAL_HANDOFF_2026-10-03.json` 참조.
- 검증됨: compileall, 전체/최종 CPU tests, 모든 CLI help, actual synthetic audit/loader/overfit/dev/checkpoint, local gate runner, benchmark 고정 metric CPU 함수, 문서 CLI syntax/dry-run 검토.
- 대기: 실제 Pod 명령·mount/restart·real cache audit·weights/GPU G2/G4/G5·실측 profile·A/B/C/D·real full-dev best export. 배포·GPU 속도 개선·VOS 개선을 주장하지 않는다.
