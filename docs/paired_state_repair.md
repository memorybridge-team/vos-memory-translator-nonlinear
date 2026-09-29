# ZIP 단일 객체 paired-state 수집 repair 운영 지침

2026-09-29. 개발 기준 `5be4fba5ec2e7df3c95178639f02b3078388b0f3`.
ZIP SHA-256 `312702886b9ae7c231fb1abcaee69e35eec9fa5822c33c9333b88fd24dc193aa`.
ZIP의 수집 경로를 보완한 replacement이며 현재 Pod 배포본과 동일하다고 가정하지 않는다.

## 보존하는 의미와 수정

수집 단위는 `(dataset, video, object, switch)`의 독립 단일 객체 `O=1`이다.
`single_object_independent_v1`을 기록하며, 따로 추론한 객체들을 concatenate해서 joint native state로 취급하지 않는다.
Production의 mask 조건은 `native_history`다. Controlled same-mask는 기존 loader에서 별도 mode로 유지한다.
기존 active 선택은 `active_window_v1`로 명명한 임시 정책이다. Conditioning 전체와 최근
non-conditioning `max(num_maskmem-1, max_obj_ptrs_in_encoder-1)` records를 보존하고 새 slot을 기록한다.
`full_history`로 자동 전환하거나 둘을 동일 lineage에서 섞지 않는다.

Late prompt와 switch는 실제 JPEG 목록에서 **각각** 해석한다. MOSE manifest 값은 RGB-order index,
LVOS는 공식 frame ID다. JPEG 외 파일은 frame 수에 넣지 않고, 중복 numeric ID·비숫자 JPEG·누락 ID는 거부한다.
첫 RGB의 공식 ID가 1이면 runtime index 0일 수 있다. ID 26/51/1491도 실제 목록으로 해석하며 offset을
subtract한 뒤 전체 영상을 그대로 처리하는 방법은 사용하지 않는다. Staging은 영상 전체의 numeric symlink view이며 trim하지 않는다.
미래 annotation을 읽지 않고 manifest가 지정한 첫 prompt mask만 읽는다.

`--state-only`는 두 predictor를 switch까지 propagation한다. Inclusive API에
`max_frame_num_to_track = switch - prompt_index`를 전달하고 실제 처리/inference frame과 backbone call을 기록한다.
`init_state`의 영상 전체 로딩·frame-0 warmup은 별도 시간/범위로 기록한다. 전체 미래 oracle consumer는
기존 `handoff_full` 기본 경로를 사용한다. Mask-free cache를 완전한 미래 oracle로 사용하지 않는다.

최신 write는 unique temp → fsync → replace → checksum → 완료 marker 순서다.
원자적 replace 및 O_EXCL이 모든 참여 worker 사이에서 일관적인 **같은 filesystem**이어야 한다.
KNSW_DATASET/공유 filesystem의 이 성질은 provider와 실제 동시 writer smoke로 확인해야 한다.
Case별 lock과 shard별 status/run lock을 사용하며 stale lock은 자동 삭제하지 않는다.
중단 시 marker가 없거나 checksum이 어긋난 원본도 덮어쓰지 않고 audit 대상으로 보존한다.
Event JSON은 append-only이고 status는 그 projection이다. 다른 worker와 mutable status를 공유하지 않는다.

## 실제 배포를 먼저 확인

현재 endpoint와 read-only 승인 전에는 과거 SSH endpoint에 접속하지 않는다.
Topology(한 Pod의 8 GPUs인지 여러 Pods인지), GPU UUID/physical assignment, PID/session, cwd,
Python/설치 module 경로·실제 소스 hashes, dataset/split, shard count/index, cache/log/run roots,
실패 사례와 현재 쓰는 파일을 기록한다. 전체 environment·private key를 출력하지 않는다.

승인 후 operator가 현재 Pod에서 실행할 제한된 inventory 예시:

```bash
python scripts/inspect_collection_workers.py --help
python scripts/inspect_collection_workers.py --pid "$CURRENT_WORKER_PID" \
  --source-file "$ACTUAL_COLLECTOR_SCRIPT" --source-file "$ACTUAL_ROUNDTRIP_MODULE" \
  --status-file "$ACTUAL_SHARD_STATUS" --log-file "$ACTUAL_SHARD_LOG" > "$NEW_AUDIT_ROOT/worker.json"
```

이 도구는 명시한 PID/파일만 읽는다. 각 실제 interpreter/package의 CLI 파일과 source hash는 추가 `--source-file`로 지정한다.
Module probe는 동일 Python 경로로 시작한 새 process의 설치 경로다. 실행 중 worker에 이미 import된
module revision을 증명하지 않으므로 시작 시 run 기록·파일 hash와 함께 검토한다.
다른 Pod에서는 그 Pod의 승인된 endpoint에서 별도 inventory를 만든다. 처음부터 500GB tensor/cache 재귀 검사를 하지 않는다.
Stable completed cache 소수만 아래 audit에 명시한다. Legacy writer는 새로운 lock을 존중하지 않으므로,
mtime 안정성 검사만으로 완료를 증명하지 않는다. Operator의 완료/run 기록과 before/after SHA 검사도 필요하다.

## 로컬 CPU 검증 및 실행 가능한 package

전체 repository source ZIP을 새 directory에 풀거나 기준 repo+review patch를 사용한다. ZIP에는 `src`,
`pyproject.toml`, scripts, tests, frozen manifests가 포함된다. Data/weights/environment는 포함되지 않는다.
요구 사항은 Python ≥3.10, Torch ≥2.3, NumPy ≥1.24, Pillow ≥9.4, pytest ≥8이다.
GPU 환경의 SAM 2 의존성은 **별도의 pinned checkout/venv**에 설치한다. Live worker 환경은 변경하지 않는다.

```powershell
$env:PYTHONPATH='src;.'
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp 'C:\Users\SAMSUNG\Documents\Codex\repair-test-new'
.\.venv\Scripts\python.exe -m vos_memory_inspector.paired_collection_cli --help
.\.venv\Scripts\python.exe -m compileall -q src scripts
git diff --check
```

Windows에서는 짧은 temp/artifact root를 사용한다. Pytest `--basetemp`는 기존 내용을 지울 수 있으므로 새 전용 경로를 지정한다.
Linux의 새 isolated checkout에서 `python -m venv .venv`, CUDA에 맞는 Torch와 pinned SAM 의존성을
설치하고 `.venv/bin/python -m pip install -e .`를 실행하면 console entrypoints도 설치된다.
실제 실행 interpreter로 `import vos_memory_inspector; print(vos_memory_inspector.__file__)`을 확인한다.

## 동결 selection과 8-way 계획

```bash
export PYTHONPATH=src:.
PYTHON=.venv/bin/python
"$PYTHON" -m vos_memory_inspector.paired_collection_cli plan \
  --source-manifest manifests/mosev2_train_v1.json \
  --fit-split manifests/mosev2_train_v1_fit.json \
  --development-split manifests/mosev2_train_v1_development.json \
  --output "$NEW_PLAN_ROOT/mose-global.json"
"$PYTHON" -m vos_memory_inspector.paired_collection_cli shard-plan \
  --plan "$NEW_PLAN_ROOT/mose-global.json" --shard-count 8 --output "$NEW_PLAN_ROOT/mose-eight.json"
```

LVOS는 세 파일명을 `lvosv2_train_v1*.json`으로 바꿔 별도 global selection을 만든다.
기존 frozen membership 파일 자체를 바꾸거나 fallback random split을 사용하지 않는다.
각 dataset/run 그룹의 전체 selection을 먼저 동결한 후 legacy modulo로 분할한다.
전체 union과 intersection=empty를 검증하며, 실행 도중 shard-count/order를 바꾸지 않는다.
실제 topology/기존 assignment가 확인되기 전에는 두 dataset에 각각 새 8-worker job을 시작하지 않는다.
실측 불균형이 없으므로 balanced 재분할은 추가하지 않았다. `cost_proxy`는 총 frame 수 기반 참고값으로 ETA가 아니다.

## CPU audit/import/verify

Requests JSON은 `{"requests": [...]}`다. 각 항목은 명시적 completed `path`, 알려진 `case_id`, 가능하면
review된 `sha256`, `expected_generating`을 둔다. Expected 조건은 실제 RGB mapping·모델/config SHA·run
증거에서 만들어야 하며, schema를 통과시키려 placeholder 값을 넣지 않는다.
수정 cache의 generating/marker가 없으면 `run_evidence: {path, sha256}`와 `completed_evidence`가 필요하다.
Run evidence는 `status=reviewed_immutable`, reviewer, reviewed_at, source_evidence, 그리고
`bindings: [{cache_sha256, generating}]`를 포함한다. 이는 실제 생성 run에 적용된다는 운영자 검토 기록이어야 한다.
파일명·model-id 문자열만으로 생성을 추정하지 않는다. 새 collector의 conditions JSON은 repaired run의
조건이며 오래된 cache의 누락 이력을 소급 증명하지 않는다.

```bash
"$PYTHON" -m vos_memory_inspector.paired_collection_cli audit \
  --requests "$AUDIT_REQUESTS" --output "$NEW_AUDIT_ROOT" --trusted-team-legacy --stable-seconds 60
"$PYTHON" -m vos_memory_inspector.paired_collection_cli import \
  --requests "$AUDIT_REQUESTS" --plan "$NEW_PLAN_ROOT/mose-global.json" \
  --output "$NEW_IMPORT_ROOT" --trusted-team-legacy --stable-seconds 60
"$PYTHON" -m vos_memory_inspector.paired_collection_cli verify --collection "$NEW_IMPORT_ROOT"
```

`.pt`의 CanonicalState object는 pickle 실행을 요구할 수 있으므로 `--trusted-team-legacy`는 팀이
신뢰한 파일에만 사용한다. SHA를 먼저 확인하고 읽으며 원본에는 lock/metadata/내용을 쓰지 않는다.
Output은 `weights_only=True`로 읽을 수 있는 tensor/dict shards와 `manifest.json`, `splits/fit.json`,
`splits/dev.json`, `migration-ledger/events/*.json`이다. Original checksum·frame mapping·prompt timeline·
generating provenance·source 정책·split을 보존한다. Legacy의 이전 slot map이 없으면 `UNKNOWN`으로 기록한다.
같은 semantic case+generating 조건은 경로가 달라도 deduplicate한다. 같은 identity에서 tensor가 다르면 거부한다.
Fit/dev는 정확한 frozen memberships를 소비하며 다른 switch도 같은 영상 역할을 유지한다.

Full→active는 `import --target-policy active_window_v1`로 명시적 CPU 변환할 수 있다. Stride=1을 요구하고
선택/slot map을 남긴다. Active→full은 삭제한 record를 복구할 수 없으므로 변환하지 않는다.
Format 변환 또는 diagnostic 표기 수정만으로 전체 GPU 재수집을 요구하지 않는다.
`reuse-plan.json`의 `recollection_case_ids`는 확인된 invalid/missing 사례만 포함한다.
근거 부족은 `PENDING_EVIDENCE`이며 임의로 전체 재수집 목록에 넣지 않는다.

## 승인된 새 worker의 dry-run / 제한 수집

```bash
"$PYTHON" -m vos_memory_inspector.paired_collection_cli collect \
  --plan "$NEW_PLAN_ROOT/mose-global.json" --dataset-root "$VERIFIED_DATASET_ROOT" \
  --sam2-repo "$ISOLATED_PINNED_SAM2" --source-checkpoint "$VERIFIED_SMALL_CKPT" \
  --target-checkpoint "$VERIFIED_BASEPLUS_CKPT" --read-policy-json configs/paired_read_policy.example.json \
  --cache-root "$NEW_CACHE_ROOT" --run-root "$NEW_RUN_ROOT" \
  --shard-count 8 --shard-index 0 --max-cases 1 --device cuda:0 --dry-run
```

Read-policy example은 pinned config/constructor 기준의 기대값이며 배포 관측값이 아니다.
Actual predictor 속성이 다르면 추론 전에 거부한다. `active_window_v1`은 evaluation stride=1만 받는다.
Worker별 CUDA_VISIBLE_DEVICES를 운영자가 실제 UUID/assignment에 맞게 지정한다.
Visible GPU 하나를 배정한 후 내부 device는 일반적으로 `cuda:0`이다. 이 작업은 독립 inference 수집이며 DDP가 아니다.
실제 수집은 현재 endpoint·free assignment·시간·비용 승인을 받은 후에만 `--dry-run`을 제거한다.
`--case-timeout-seconds`로 subprocess 상한을 둔다. `--worklist`는 confirmed recollection case IDs를
기존 global assignment와 교차하여 처리한다. 코드/조건/모델별 새 namespace와 shard별 logs/status/manifest를 사용한다.

## Same-checkpoint 정책 gate와 배포 판단

```bash
"$PYTHON" -m vos_memory_inspector.memory_policy_smoke --help
"$PYTHON" -m vos_memory_inspector.memory_policy_smoke \
  --sam2-repo "$ISOLATED_PINNED_SAM2" --checkpoint "$VERIFIED_BASEPLUS_CKPT" \
  --video-dir "$VERIFIED_VIDEO_DIR" --prompt-mask "$VERIFIED_FIRST_PROMPT_MASK" \
  --object-id "$OBJECT_ID" --prompt-frame-index "$RESOLVED_PROMPT_INDEX" --switch-frame "$RESOLVED_SWITCH_INDEX" \
  --future-frames 3 --max-video-frames 1000 --max-input-bytes 1000000000 \
  --max-wall-seconds 300 --max-cuda-bytes 8589934592 --output "$NEW_SMOKE_REPORT"
```

기본은 dry-run이다. 별도 승인 후에만 `--execute`를 추가한다. Exact 동등성을 사전에 요구할 때만
`--require-exact`를 사용하며, 차이가 나면 measured report를 보존하고 정책 채택을 명시적으로 결정한다.
Allocator 상한은 전체 GPU/RAM 또는 총 요금 상한을 대신하지 않는다. 운영자가 승인한 free GPU/비용 ledger가 필요하다.
CPU 테스트는 실제 continuation parity를 증명하지 않는다. Spatial/pointer loss 감소도 VOS 개선 증거가 아니다.

## Pinned read 정책의 근거와 남은 검증

Upstream `2b90b9f5ceec907a1c18123530e92e794ad901a4`의
[SAM2Base](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/modeling/sam2_base.py)는
spatial memory에서 evaluation stride와 conditioning 선택을 적용하며 pointer에는 별도 window와 과거-only 조건을 적용한다.
[SAM2VideoPredictor](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/sam2_video_predictor.py)의
propagation 종료는 inclusive다. 실제 predictor 속성은 수집 전에 기대값과 비교한다.
이 정적 코드 확인은 `active_window_v1`의 모든 runtime read/equivalence를 증명하지 않는다.
Read-set 변화나 stride 변경은 별도 정책과 continuation 검증을 요구한다.

## Pod 재시작 뒤 읽기 검증 — 아직 미실행

종료 전 imported collection의 `manifest.json` SHA와 shard 검증 결과를 원본 밖 run 기록에 보관한다.
새 Pod에 같은 Network Volume을 연결한 뒤 실제 mount와 경로를 확인하고 격리 환경을 재구성한다.
동일 collection에 `verify --collection "$NEW_IMPORT_ROOT"`를 다시 실행하여 모든 shard checksum,
weights-only load 및 pair 계약을 검사하고 이전 manifest SHA와 비교한다. 재수집 명령으로 읽기 검증을 대신하지 않는다.
Lock이 남아 있으면 writer PID/run 증거를 먼저 확인하고 owner가 처리하며 자동 삭제하지 않는다.
본 로컬 작업에서는 실제 Volume 동시성이나 Pod 재시작 후 가독성을 검증하지 못했다.
