# 고정 split · validation state R² 학습 운영 안내

날짜: 2026-10-09. 상태: **READY_AFTER_INPUTS / Draft — tensor 연결 미검증**.
base: `55b117d`; 작업 branch: `feat/frozen-split-validation-r2`.
정확한 변경 revision은 PR commit과 `git rev-parse HEAD`로 확인한다.
이번 작업에서 공식 GPU 학습은 시작하지 않았다.

## 1. 검증 경계

|구분|근거|적용 범위|
|---|---|---|
|기존 증빙 재사용|2026-10-05 `POST_DELIVERY_LOAD_CHECK.json`, `BOUNDARY_STOP_EVIDENCE.json`, model lock|동일 frozen model/API 입출력·양 branch 갱신·strict model reload·rank 차이 0. 새 split/R² runner 검증으로 확대하지 않음|
|신규 실행|중간 구현의 stdlib synthetic/metadata fixture 32 tests, 최종 AST/CLI help|split 불변성·role mapping·numeric R²/Chan·exact record schedule·best/early/resume 상태. 실제 split/data/tensor 아님. 이후 prior-index 재사용·batch 기본값 보존 수정은 정적 검사만 수행|
|미실행|RunPod 접근·CPU 실행 승인 대기|최종 revision의 stdlib 재검사, 실제 inventory/split SHA/count, 새 index 연결, train-only RMS, 6개 작은 tensor/statistics/metadata tests, 변경된 trainer 통합|
|이번 범위 밖|명시적 미승인|공식 GPU 학습, 수집·J&F·baseline·external benchmark|

기존 모델 revision은 `746ea3e7d84c366c2d7ac06159e90a1f684bca56`,
`base` config다. 모델팀 body 파일 SHA는 `MIGRATION.md`에 있다.
기존 검증은 새 index/selection/checkpoint metadata의 실행 증빙이 아니다.

## 2. 내일 필요한 입력과 승인

|입력/승인|담당|현재 상태|
|---|---|---|
|현재 SSH host/port/user, host-key fingerprint, 기존 개인키 경로|운영자|새 endpoint 미제공. 과거 endpoint 재사용 금지|
|같은 `new` volume 연결, CPU/RAM/가용 디스크, 실행 시간·Pod 전체 USD/hour·CPU 작업 예산|운영자|`j700qtcgrg / EUR-IS-1 / /workspace`는 사용자 제공값. 현재 Pod 실측 대기|
|CPU fixture 검사·소수 cache record 연결·inventory/split/index/statistics 파일 생성 승인|사용자|대기. 노트북 대체 실행 없음|
|MOSE official train **전체 원본 영상 목록**과 해당 RGB 영상 directory root|데이터팀|미확인. case manifest/cache 목록으로 전체 목록을 대신 만들지 않음|
|LVOS official train/valid 전체 영상 목록과 RGB root, actual case manifest 경로|데이터팀|미확인. 원본 extracted root까지만 제공됨|
|원본 clip 그룹 mapping 및 확인 범위|데이터팀|선택 입력. 없으면 `UNKNOWN_SINGLE_VIDEO_GROUP`으로 보고|
|교체 cache 완료 확인, writer 상태, 파일명 pattern, 제외 case+사유|데이터팀|현재 snapshot 미확인. 수정 fit 9개 자동 제외 없음; 정상 검사 통과 시 포함|
|공식 GPU 학습 별도 승인 및 UUID/시간/예산|사용자|이번 요청에 포함하지 않음|

known 원본 root: `/workspace/CMMT/data/MOSEv2/extracted/`,
`/workspace/CMMT/data/LVOSv2/extracted/`.
그 아래 RGB·official list의 실제 구조는 확인 전이다.
`<...>`는 모두 **미입력 placeholder**이며 실행 전에 바꿔야 한다.
host-key 검증을 끄지 않고 키 내용·token을 로그/Git에 기록하지 않는다.

## 3. 데이터 역할과 split

Train = MOSE official train의 고정 90% + LVOS official train.
Validation = MOSE의 고정 5% + LVOS official valid.
Internal-test = MOSE official train의 고정 5%; official MOSE test가 아니다.
Internal-test와 DAVIS·external benchmark는 학습·RMS·scheduler·best에서 차단한다.
기존 `fit/development`는 파일 보관 위치다. 새 role은 영상 membership으로만 정한다.

1. 원본 official 목록과 RGB **directory 이름**을 대조해 original inventory를 만든다.
2. `(dataset, video_id)`로 정렬한다. 같은 original group은 함께 배정한다.
3. `SHA256(seed + NUL + dataset + NUL + group)` 정렬 후 group 수에 90/5/5
   largest remainder를 적용한다. 나머지 동점은 train→validation→internal-test다.
4. 그룹 보존 때문에 video 비율이 정확히 90/5/5가 아닐 수 있다. 실제 둘 다 보고한다.
5. seed=7, 알고리즘 v1, inventory와 membership SHA를 고정한다.

Frozen root는 새로운 `/workspace/CMMT-training-contracts/splits/mosev2_90_05_05_v1`.
`original_inventory.json`, `split_manifest.json`, `split_manifest.sha256`,
`SPLIT_FROZEN.json`, `split_report.json`을 저장한다. 동일 입력은 재사용한다.
변경 seed/inventory/unknown video/불완전 frozen 파일은 중단한다. 자동 확장·덮어쓰기
명령은 제공하지 않는다. 필요하면 기존 valid/test를 보존하는 확장 계약을 별도 승인한다.
cache 누락·제외·교체는 새 index namespace에 반영하며 비율을 다시 채우지 않는다.
확인하지 못한 clip/중복 관계 때문에 완전한 누수 방지를 주장하지 않는다.

## 4. 계약과 R²

Spatial `[B,O,K,64,64,64]` BF16, pointer `[B,O,K,256]` FP32.
single-object cache의 source/target validity·frame·slot·conditioning·object·switch를
동일하게 검사한다. 유효 tensor는 finite여야 한다. padding은 loader가 제외한다.
late prompt는 전체 runtime을 유지한다. MOSE는 sorted RGB runtime,
LVOS는 official frame ID를 실제 RGB runtime으로 매핑한다. conditioning=0 가정은 없다.
metadata SHA/파일 안정성과 tensor 정렬을 검사하되 없는 과거 생성 정보는 만들지 않는다.
original completion marker 부재만으로 재생성을 요구하지 않는다.
CPU `weights_only=True` + `CanonicalState/StateSpec` allowlist만 사용하며 unsafe fallback은 없다.

학습 loss는 기존 normalized spatial MSE + normalized pointer MSE.
Base+ target RMS를 **새 train**에서만 한 번 계산한다. 추론 입력을 RMS로 나누지 않는다.
SAM2 모델은 학습 프로세스에서 생성하지 않는다. cached state를 읽으며 Translator만 갱신한다.
구조/AdamW LR=3e-4, WD=1e-4, no-decay group, FP32, clip=1, augmentation 없음.
마지막 공식 run의 실제 기록(`CURRENT_RUN_REPORT_KO.md`)은
global64 = 2 ranks × microbatch32 × accumulation1이므로 기본 config에 보존했다.
이전 LVOS-only run의 microbatch16×accumulation2와 구분한다.
4 ranks는 별도 config에서 microbatch16×accumulation1로 global64를 유지한다.
case shuffle, 내부 record 순서는 보존한다. 전 record 한 번/epoch, oversampling 없음.
tail은 실제 global valid-record 수로 gradient를 가중하고 padding/복제 없이 검증한다.

R²는 Base+ validation target의 고정 충분통계를 사용한다.
Spatial channel 평균은 records×H×W, pointer coordinate 평균은 records 전체에서 구한다.
float64 Welford/Chan으로 target M2를 streaming merge한다. 전체 tensor RAM 수집 없음.
`variance=M2/count <= 1e-12` 차원은 SSE/SST 분모에서 제외하고 수와 별도 MSE를 기록한다.
전 차원이 제외되면 undefined이며 1로 대체하지 않는다.
branch R²=`1-sum(eligible SSE)/sum(eligible M2)`, 음수 허용.
MOSE spatial/pointer와 LVOS spatial/pointer R² **네 값의 동일 비중 평균**으로 선정한다.
NaN/Inf/누락/부분 coverage는 promotion을 차단한다. batch R² 평균은 하지 않는다.
target 통계는 validation index에 고정되며 train normalization/입력으로 쓰지 않는다.

`best_validation_r2`: 최대 score에서 abs 1e-12 이내인 가장 낮은 epoch.
실제 best 선정과 early의 의미 있는 개선(score > reference + .001)을 구분한다.
Plateau: mode=max, threshold_mode=abs, .001; factor=.5, patience=3, cooldown=1,
minLR=1e-5. update warmup=min(updates/epoch,1000)을 유지한다.
early: min_epochs15 이후 meaningful improvement 없는 8 full-validation checks;
최대60. MSE/train loss/J&F로 best를 선정하지 않는다.

## 5. 승인 후 RunPod CPU 순서

아래는 **구현된 CLI**다. help만 실행 검증됐고 실제 데이터 명령은 아직 미실행이다.
모든 데이터 처리·tensor 검사·정규화는 RunPod bash에서 수행한다.
운영 중 checkout/environment에는 설치하지 않는다. 격리 checkout과 기존 Torch의 별도
환경을 운영자가 준비한 후 프로젝트 root에서 실행한다. 전역 Torch/CUDA 교체 없음.

### 5.1 CPU/RAM metadata 확인

```bash
nproc
```

```bash
free -h
```

```bash
df -h /workspace
```

### 5.2 CLI와 새 CPU 검사

```bash
CUDA_VISIBLE_DEVICES="" python -m vos_memory_inspector.state_training --help
```

```bash
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -p test_state_training_contract.py -v
```

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q tests/test_state_training_statistics.py
```

예상: stdlib 32 tests, 새 작은 tensor/statistics/metadata tests 6개.
후자는 현재 **미실행**이다. 기존 모델 forward/gradient/reload 검사를 반복하지 않는다.
실패하면 최초 traceback·revision·Python/Torch와 해당 fixture만 공유한다.

### 5.3 Original inventories

`official-list`는 영상 ID 한 줄씩인 authoritative txt 또는 dataset/split/videos
JSON이다. cases에서 video 목록을 유추하지 않는다. `--groups`는 확인된 JSON만 선택 제공한다.

```bash
CUDA_VISIBLE_DEVICES="" python -m vos_memory_inspector.state_training inventory --dataset MOSEv2 --official-split train --video-root '<MOSE_OFFICIAL_TRAIN_RGB_VIDEO_ROOT>' --official-list '<MOSE_FULL_OFFICIAL_TRAIN_VIDEO_LIST>' --output '/workspace/CMMT-training-contracts/inventories/<SNAPSHOT>/mose_train.json'
```

```bash
CUDA_VISIBLE_DEVICES="" python -m vos_memory_inspector.state_training inventory --dataset LVOSv2 --official-split train --video-root '<LVOS_TRAIN_RGB_VIDEO_ROOT>' --official-list '<LVOS_FULL_TRAIN_VIDEO_LIST>' --output '/workspace/CMMT-training-contracts/inventories/<SNAPSHOT>/lvos_train.json'
```

```bash
CUDA_VISIBLE_DEVICES="" python -m vos_memory_inspector.state_training inventory --dataset LVOSv2 --official-split valid --video-root '<LVOS_VALID_RGB_VIDEO_ROOT>' --official-list '<LVOS_FULL_VALID_VIDEO_LIST>' --output '/workspace/CMMT-training-contracts/inventories/<SNAPSHOT>/lvos_valid.json'
```

성공: `.json`과 `.json.sha256`; 목록/directory 불일치는 중단한다.
데이터팀 원본 목록·하위 root를 확인하며 임의 제외로 통과시키지 않는다.

### 5.4 Permanent split

```bash
CUDA_VISIBLE_DEVICES="" python -m vos_memory_inspector.state_training freeze --inventory '/workspace/CMMT-training-contracts/inventories/<SNAPSHOT>/mose_train.json' --output /workspace/CMMT-training-contracts/splits/mosev2_90_05_05_v1 --seed 7
```

성공: `SPLIT_FROZEN.json`, 그룹 overlap0, SHA 일치. 기존 frozen 입력이 다르면
새 seed로 재분할하지 않고 중단 근거를 공유한다.

### 5.5 역할 plan / 실제 index

`configs/state_training_datasets.template.json`을 원본 밖 `<DATASETS_JSON>`으로 복사하고
placeholder, 확인된 actual cache filename pattern, case manifest를 채운다.
제외는 `exclusions: [{"case_id":"MOSEv2|...", "reason":"..."}]`에 명시한다.
수정 LVOS fit9를 미리 제외하지 않는다. missing cache는 used/excluded에 남으며 membership 이동 없음.

기존 검증 index를 재사용하려면 확인된 실제 path/SHA로 config에
`"verified_prior_index": {"path":"<OLD_TENSOR_VALIDATED_INDEX>", "sha256":"<VERIFIED_SHA256>"}`를
선택 추가한다. 지원 schema는 `cmmt.official_state_training_index.v1`, `state=tensor_validated`다.
case semantic·file stamp/SHA sidecar·RGB frame map·conditioning mapping이 같을 때만
검사 증빙을 재사용한다. 기존 normalization/role은 재사용하지 않는다.
파일이 바뀌었거나 prior에 없으면 새 tensor 검사가 필요하다. 최종 index의
`inspection_counts`에서 기존 증빙 재사용과 신규 검사를 구분한다.
전체 신규 tensor 검사가 예상되면 운영 승인 범위를 확인한 뒤 진행한다.

```bash
CUDA_VISIBLE_DEVICES="" python -m vos_memory_inspector.state_training plan --datasets '<DATASETS_JSON>' --split /workspace/CMMT-training-contracts/splits/mosev2_90_05_05_v1 --output '/workspace/CMMT-training-contracts/indices/<SNAPSHOT>/plan.json'
```

plan은 tensor PASS가 아니다. source별 역할·경로·unknown IDs를 먼저 확인한다.
실제 cache 연결은 소수 record를 운영 승인으로 먼저 확인한 뒤 아래 index 작업 범위를 확정한다.
소수 검사 helper는 Python `state_training.cache.inspect(case,path,rgb_root)`이며 CLI subset 옵션은 없다.

```bash
CUDA_VISIBLE_DEVICES="" python -m vos_memory_inspector.state_training index --datasets '<DATASETS_JSON>' --split /workspace/CMMT-training-contracts/splits/mosev2_90_05_05_v1 --output '/workspace/CMMT-training-contracts/indices/<SNAPSHOT>' --operator-completed '<ACTUAL_DATA_TEAM_COMPLETION_CONFIRMATION>' --stable-seconds 60
```

성공: `index.json(.sha256)`, `used_excluded.json`, `INSPECTION.json`, per-case journal.
original video/group vs available/used case/valid record 수를 구분한다. 실패 cache는
`INSPECTION.json`에 기록하고 전체 index ready를 만들지 않는다. internal-test tensor는 읽지 않는다.
재실행은 journal과 입력/파일 stamp/SHA sidecar가 같을 때만 재사용한다. stale lock은 자동 삭제하지 않는다.

### 5.6 Train RMS / validation target statistics

```bash
CUDA_VISIBLE_DEVICES="" python -m vos_memory_inspector.state_training statistics --index '/workspace/CMMT-training-contracts/indices/<SNAPSHOT>/index.json' --output '/workspace/CMMT-training-contracts/statistics/<SNAPSHOT>' --config configs/state_training_r2.json
```

성공: `normalization.json`, `validation_targets.json`, `r2_spec.json`, 각 SHA와
`STATISTICS_READY.json`. missing/changed binding은 중단한다. 부분 stats는 덮어쓰지 않고
새 namespace에서 재시도한다. 실제 작업 시간/storage 비용은 내일 측정하며 현재 미측정이다.

## 6. 향후 별도 GPU 승인 후 학습·재개 명령

**아래를 이번 작업에서 실행하지 않는다.** 새 contract의 fresh init만 허용한다.
이전 모델은 MOSE 새 valid/test 영상에 노출됐을 수 있어 warm-start 금지다.
모델팀 body 검증을 반복하려는 명령이 아니라 새 input/DDP 연결의 실제 실행 gate다.
값과 budget을 채운 RunPod bash에서 GPU 가용성·UUID 순서를 operator가 확인한다.

```bash
CUDA_VISIBLE_DEVICES='<GPU_UUID_0>,<GPU_UUID_1>' CUBLAS_WORKSPACE_CONFIG=:4096:8 torchrun --standalone --nproc-per-node=2 -m vos_memory_inspector.state_training train --mode smoke --device cuda --index '<NEW_INDEX_JSON>' --statistics '<NEW_STATISTICS_ROOT>' --config configs/state_training_r2.json --output '<NEW_SMOKE_ROOT>' --max-wall-seconds 600 --execute-approved --gpu-uuid '<GPU_UUID_0>' --gpu-uuid '<GPU_UUID_1>' --approval-start-utc '<APPROVAL_UTC>' --deadline-utc '<DEADLINE_UTC>' --pod-hourly-rate '<WHOLE_POD_USD_PER_HOUR>' --budget-usd '<APPROVED_REMAINING_USD>'
```

smoke subset score는 diagnostic이며 공식 best가 아니다. CPU smoke를 GPU PASS로 쓰지 않는다.
새 index/source/world/normalization binding의 실제 `REAL_R2_DDP_SMOKE_PASS`가 본 학습 조건이다.

```bash
CUDA_VISIBLE_DEVICES='<GPU_UUID_0>,<GPU_UUID_1>' CUBLAS_WORKSPACE_CONFIG=:4096:8 torchrun --standalone --nproc-per-node=2 -m vos_memory_inspector.state_training train --mode train --device cuda --index '<NEW_INDEX_JSON>' --statistics '<NEW_STATISTICS_ROOT>' --config configs/state_training_r2.json --output '<NEW_RUN_ROOT>' --smoke-evidence '<NEW_SMOKE_ROOT>/STATUS.json' --max-wall-seconds '<APPROVED_SECONDS>' --execute-approved --gpu-uuid '<GPU_UUID_0>' --gpu-uuid '<GPU_UUID_1>' --approval-start-utc '<APPROVAL_UTC>' --deadline-utc '<DEADLINE_UTC>' --pod-hourly-rate '<WHOLE_POD_USD_PER_HOUR>' --budget-usd '<APPROVED_REMAINING_USD>'
```

첫 epoch 경계 점검이 별도 승인되면 위 명령에 `--stop-after-epoch 1`을 추가한다.
기본 명령은 최대60 또는 R² early stop까지 이어진다. tmux에서 실행하면 SSH 종료와
무관하게 진행한다. tmux/controller 생성은 현재 실행하지 않았다.
4 ranks는 같은 code, microbatch16을 명시한 새 config와
`--nproc-per-node=4`, GPU UUID 네 개를 사용한다. 다른 batch config로 resume하지 않는다.
GPU당 visible local index는 재매핑되므로 물리 index를 직접 가정하지 않는다.

```bash
CUDA_VISIBLE_DEVICES='<GPU_UUID_0>,<GPU_UUID_1>' CUBLAS_WORKSPACE_CONFIG=:4096:8 torchrun --standalone --nproc-per-node=2 -m vos_memory_inspector.state_training train --mode train --device cuda --resume --index '<SAME_INDEX_JSON>' --statistics '<SAME_STATISTICS_ROOT>' --config configs/state_training_r2.json --output '<SAME_RUN_ROOT>' --smoke-evidence '<BOUND_SMOKE_ROOT>/STATUS.json' --max-wall-seconds '<REMAINING_SECONDS>' --execute-approved --gpu-uuid '<GPU_UUID_0>' --gpu-uuid '<GPU_UUID_1>' --approval-start-utc '<ORIGINAL_APPROVAL_UTC>' --deadline-utc '<ORIGINAL_DEADLINE_UTC>' --pod-hourly-rate '<WHOLE_POD_USD_PER_HOUR>' --budget-usd '<ORIGINAL_APPROVED_USD>'
```

원 approval start/budget을 재개 때 초기화하지 않는다. 준비/idle도 포함한다.
Python 종료는 Pod 과금 종료가 아니다. Pod 자동 Stop 기능은 없다.
budget/deadline은 cooperative batch/window 경계에서 확인하며 OS 수준 즉시 종료 보장은 없다.
보수적 여유를 둔 외부 `timeout`/운영 controller 한도를 operator가 준비한다.
서로 다른 contract/source/환경/world의 resume는 차단한다. world-size 변경 resume와
mid-epoch exact resume는 지원한다고 주장하지 않는다.
마지막 **완료 epoch**만 재개하며 optimizer/scheduler/RNG/history/early 상태를 복원한다.

## 7. 읽기 전용 확인·전달

```bash
python -m vos_memory_inspector.state_training monitor --run '<NEW_RUN_ROOT>'
```

`LIVE_STATUS.json`, `epochs.jsonl`, `history.json`, `best_validation_r2.json`,
`last_complete.json`, `checkpoints/epoch-00001.pt(.complete.json)` 등을 확인한다.
window당 component loss는 epoch summary에 집계되며 gradient/LR/step은 live status에 있다.
checkpoint는 save→checksum→strict model/optimizer/scheduler/output reload 후 marker 생성이다.
완료되지 않은 파일을 epoch/checkpoint ready로 인정하지 않는다.

`docs/state_training_readonly_monitor.ipynb`은 JSON 조회만 한다. cell에 train/kill/resume 없음.
Jupyter root가 `/workspace`라면 `<NEW_RUN_ROOT>`의 해당 상대 폴더를 연다.
이번 작업은 Jupyter 설치/인증/서비스 변경을 하지 않는다.

자동 전달 폴더: `<NEW_RUN_ROOT>/delivery-final-validation-r2`.
파일: `best_validation_r2_weights.pth`, `model_config.json`, `normalization.json`,
`last_complete.pt`, `history.json`, `index.json`, `used_excluded.json`,
split artifacts, `r2_spec.json`, `validation_targets.json`, `binding.json`,
`SHA256.json`, `DELIVERY_READY.json`, `FINAL_REPORT_KO.md`.
archive: `<NEW_RUN_ROOT>/translator_validation_r2_delivery.tar.gz`, `.sha256`,
`DELIVERY_ARCHIVE_READY.json`. 데이터/cache/과거 weights는 포함하지 않는다.

```python
from vos_memory_inspector.state_training.delivery import load_translator
translator = load_translator('/workspace/<NEW_RUN>/delivery-final-validation-r2')
translated_state = translator.translate(source_state)
```

평가팀은 기존 `CanonicalState`를 제공하고 기존 Base+ injection API를 사용한다.
frame/slot/conditioning/validity/object ID는 바꾸지 않는다. inference input normalization 없음.
single-object 학습 cache는 joint multi-object 동등성의 증거가 아니다.

본 모델은 validation state R² 기준으로 선정되었으며,
J&F 및 외부 benchmark 평가는 별도 담당자가 수행한다.
이 문구는 향후 실제 모델 선정 후 전달 보고서에 기록한다. 현재 새 모델은 학습하지 않았다.
