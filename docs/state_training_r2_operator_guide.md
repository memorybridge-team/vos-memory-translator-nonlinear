# 고정 split · validation state R² 학습 운영 안내

날짜: 2026-10-09. 상태: **NOT_READY / Draft — 새 입력·원격 실행기 검증 전**.
base: `55b117d`; 작업 branch: `feat/frozen-split-validation-r2`.
정확한 변경 revision은 PR commit과 `git rev-parse HEAD`로 확인한다.
이번 작업에서 공식 GPU 학습은 시작하지 않았다.

최신 승인: 같은 Pod의 L4 2개, Pod 전체 USD 1.16/hour, 승인 시작부터 12시간·USD30.
기존 USD50과 별도다. 정확한 endpoint·지문·승인 시작/deadline은 Git 밖의
로컬 운영 기록에 보관한다. 이번 승인에는 준비→10분 smoke→fresh 공식 학습이 포함된다.
기존 split은 완료됐으므로 재생성하지 않는다. 원격 trust 확인 전에는 접속하지 않는다.

## 1. 검증 경계

|구분|근거|적용 범위|
|---|---|---|
|기존 증빙 재사용|2026-10-05 `POST_DELIVERY_LOAD_CHECK.json`, `BOUNDARY_STOP_EVIDENCE.json`, model lock|동일 frozen model/API 입출력·양 branch 갱신·strict model reload·rank 차이 0. 새 split/R² runner 검증으로 확대하지 않음|
|기존 split 증빙 재사용|사용자 제공 RunPod 실행 기록: revision4fb807b, seed7, metadata32 PASS, 3300/183/183, 영상 교집합0, 동일 재실행SHA|해당 frozen split의 증빙. 같은 검사를 반복하지 않는다. 원본 계보 전체 확인을 의미하지 않음|
|신규 검사 완료|작은 tensor/statistics tests6 + 신규 운영 안전검사6|승인 Pod CPU에서 12 PASS / exit 0. 노트북 실행 없음|
|미실행 또는 진행 중|새 입력 준비 gate|새 index 연결, train-only RMS·validation 통계, 실제 새 2-GPU smoke/공식 학습; run별 실제 상태 파일 우선|
|이번 범위 밖|명시적 미승인|신규 수집·J&F·baseline·external benchmark·Pod 변경·Jupyter 변경|

기존 모델 revision은 `746ea3e7d84c366c2d7ac06159e90a1f684bca56`,
`base` config다. 모델팀 body 파일 SHA는 `MIGRATION.md`에 있다.
기존 검증은 새 index/selection/checkpoint metadata의 실행 증빙이 아니다.

## 2. 현재 운영 입력과 확인 상태

|입력/승인|담당|현재 상태|
|---|---|---|
|현재 SSH endpoint·GPU UUID|운영자|공개 host-key 독립 대조·SSH 인증·L4 두 UUID 실측 완료. 상세값은 로컬 운영 기록|
|같은 `new` volume 연결, CPU/RAM/가용 디스크, Pod 단가·한도|운영자|볼륨 확인. cgroup CPU quota 30.6개·RAM 약133GiB; host RAM 수치를 전용 할당으로 사용하지 않음|
|CPU6·최소 record 연결·inventory/index/statistics 준비|사용자|승인. 전체 신규 tensor audit 필요 시 범위 별도 승인. 노트북 대체 실행 없음|
|MOSE official train **전체 원본 영상 목록**과 RGB root|데이터팀|제공 완료: 고정 split·목록은 재사용. case/cache 목록으로 전체 목록을 대신 만들지 않음|
|LVOS official train/valid 전체 영상 목록과 RGB root, actual case manifest 경로|데이터팀|원본 meta/RGB 제공 완료. RunPod에서 표준 목록 생성·실제 case manifest 연결 확인 예정|
|원본 clip 그룹 mapping 및 확인 범위|데이터팀|선택 입력. 없으면 `UNKNOWN_SINGLE_VIDEO_GROUP`으로 보고|
|교체 cache 완료 확인, writer 상태, 파일명 pattern, 제외 case+사유|데이터팀|현재 snapshot 미확인. 수정 fit 9개 자동 제외 없음; 정상 검사 통과 시 포함|
|공식 GPU 학습·UUID·시간·예산|사용자|이번 승인에 포함. 필수 trust/CPU/index/statistics/new smoke 통과 후 실행|

known 원본 root: `/workspace/CMMT/data/MOSEv2/extracted/`,
`/workspace/CMMT/data/LVOSv2/extracted/`.
원본 metadata의 전체 영상 key와 RGB directory를 대조해 LVOS train420/valid140 목록을 생성한다.
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
CUDA_VISIBLE_DEVICES="" python -m pytest -q tests/test_state_training_statistics.py
```

기존 stdlib32/split 증빙은 재사용한다. 새 작은 tensor/statistics/metadata tests6은 현재 **미실행**이다.
신규 외부 실행기·검사 상한의 안전6 검사는 8절에 있다. 기존 모델 forward/gradient/reload 검사를 반복하지 않는다.
실패하면 최초 traceback·revision·Python/Torch와 해당 fixture만 공유한다.

### 5.3 Original inventories

`official-list`는 영상 ID 한 줄씩인 authoritative txt 또는 dataset/split/videos
JSON이다. cases에서 video 목록을 유추하지 않는다. `--groups`는 확인된 JSON만 선택 제공한다.

MOSE inventory는 고정 split의 `original_inventory.json(.sha256)`을 재사용한다.
기존 원본 목록·split SHA를 확인하며 이 단계에서 재생성하지 않는다.

```bash
CUDA_VISIBLE_DEVICES="" python -m vos_memory_inspector.state_training inventory --dataset LVOSv2 --official-split train --video-root '<LVOS_TRAIN_RGB_VIDEO_ROOT>' --official-list '<LVOS_FULL_TRAIN_VIDEO_LIST>' --output '/workspace/CMMT-training-contracts/inventories/<SNAPSHOT>/lvos_train.json'
```

```bash
CUDA_VISIBLE_DEVICES="" python -m vos_memory_inspector.state_training inventory --dataset LVOSv2 --official-split valid --video-root '<LVOS_VALID_RGB_VIDEO_ROOT>' --official-list '<LVOS_FULL_VALID_VIDEO_LIST>' --output '/workspace/CMMT-training-contracts/inventories/<SNAPSHOT>/lvos_valid.json'
```

성공: `.json`과 `.json.sha256`; 목록/directory 불일치는 중단한다.
데이터팀 원본 목록·하위 root를 확인하며 임의 제외로 통과시키지 않는다.

### 5.4 기존 Permanent split 읽기 전용 검증

```bash
CUDA_VISIBLE_DEVICES="" python -c "from vos_memory_inspector.state_training.splits import read_split; s=read_split('/workspace/CMMT-training-contracts/splits/mosev2_90_05_05_v1'); print(s['video_counts'])"
```

성공: 기존 frozen marker·SHA 검사가 통과하고 3300/183/183을 출력한다. 기존 frozen 입력이 다르면
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
CUDA_VISIBLE_DEVICES="" python -m vos_memory_inspector.state_training inspection-plan --datasets '<DATASETS_JSON>' --split /workspace/CMMT-training-contracts/splits/mosev2_90_05_05_v1 --output '<NEW_INDEX_ROOT>/inspection_scope.json'
```

이 명령은 tensor를 읽지 않는다. 신규 tensor 검사 후보가 전체 규모라면 보고 후 승인을 받는다.
index의 `--max-new-inspections` 기본값은0이다. 소수 실제 연결을 포함해 승인된 범위만
해당 숫자로 명시한다. 재개에서도 같은 index에 완료된 신규 검사 수를 누적한다.
이 범위를 넘으면 `INSPECTION_SCOPE_BLOCKED.json`을 남기고 중단한다.

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
새 namespace에서 재시도한다. 실제 작업 시간/storage 비용은 원격 실행에서 측정하며 현재 미측정이다.

## 6. 승인 한도 안의 학습·재개 명령

**이번 자율 run은 8절 controller로 실행한다. 아래는 CLI 계약 설명이다.** 새 contract의 fresh init만 허용한다.
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
trainer의 cooperative 검사와 별도로 8절의 외부 guard를 사용한다.
외부 guard는 Network Volume을 읽지 않고 이번 controller와 자식 rank의 process group만 종료한다.
학습 진입 시 index metadata를 한 번 검증한 객체를 내부 run에 전달한다. loader의 실제 파일 SHA·stamp·slot/validity 검사는 유지한다.
커널의 중단 불가능 I/O 상태라면 프로세스 회수 지연이 가능하므로 Pod 청구액 강제 제한으로 해석하지 않는다.
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

## 8. 이번 승인용 원격 독립 실행기 — CPU 안전장치 검증 완료, 실제 학습은 별도 gate

구현: `vos_memory_inspector.state_training.remote_run`.
`configs/state_training_remote_run.template.json`은 **실행 불가 placeholder 템플릿**이다.
검증된 격리 checkout/venv, 새 index/statistics SHA, revision/package SHA로 채운
`OPERATION_CONFIG.input.json`을 원본 밖 새 `r2-<RUN_ID>` 폴더에 저장한다.
준비·idle을 포함한 최초 승인 시작과 deadline은 그대로 유지한다.

1. RunPod CPU에서 신규 통계6과 운영 안전6만 검사한다. 기존 split32/모델 검사는 반복하지 않는다.

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q tests/test_state_training_statistics.py tests/test_state_training_remote_run.py
```

성공 기준: 12 passed, 실패/skip 없음. 2026-10-09 승인 Pod의 격리 Python 3.12.11 / Torch 2.8.0+cu128에서 `12 passed in 23.09s`, exit 0을 확인했다.
CPU 장치에서 작은 fixture와 자기가 생성한 임시 process group만 검사한다.
근거: 실행 폴더 `runs/r2-20261009T043718Z/CPU_CHECKS.json` 및 `cpu_check_0.log`.
CLI help 3개도 exit 0이다. 이는 실제 GPU smoke나 공식 학습의 통과 증거가 아니다.

2. RunPod에서 CLI 계약을 확인한다. 아직 학습을 시작하는 명령이 아니다.

```bash
CUDA_VISIBLE_DEVICES="" python -m vos_memory_inspector.state_training.remote_run --help
```

3. **아래 명령은 미입력 예시다.** 실제 원격 실행 전에 모든 placeholder를 검증된 값으로 대체한다.
`HARD_REMAINING_SECONDS`는 절대 deadline에서 현재 UTC를 뺀 값이다. 기존 12시간을 다시 넣지 않는다.

```bash
tmux new-session -d -s '<RUN_TMUX>' 'exec timeout --signal=TERM --kill-after=35s <HARD_REMAINING_SECONDS>s <ISOLATED_VENV>/bin/python -m vos_memory_inspector.state_training.remote_run watchdog --config <RUN_ROOT>/OPERATION_CONFIG.input.json > <RUN_ROOT>/watchdog.log 2>&1'
```

watchdog→분리 process group의 controller→torchrun/ranks 구조다.
[PyTorch 2.8 구현](https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/distributed/elastic/multiprocessing/subprocess_handler/subprocess_handler.py)은
rank마다 새 session을 만든다. 따라서 controller group 하나만 종료하지 않는다.
watchdog의 child-subreaper와 `/proc` PID 시작 시각·자식 관계 추적으로 이번 run의
rank별 group과 orphan rank까지 식별한다. 인자 전체·환경변수는 읽지 않는다.
별도 no-volume guard가 deadline에 해당 트리를 종료하고 smoke에도 600초 guard를 둔다.
guard는 다른 run의 group을 종료하지 않는다. 자동 retry/새 Pod/Pod Stop은 없다.
정상 경로는 smoke PASS 후 동일 source/config/index/statistics로 본 학습을 한 번 시작하며
첫 epoch에서 멈추지 않는다. trainer는 매 완료 epoch 이후 다음 epoch의 시간 여유를 확인한다.
optimizer·scheduler·RNG를 초기화하는 epoch별 재실행은 하지 않는다.

4. 사용자는 필요할 때 읽기 전용으로 확인한다. 주기적 Codex 조회/automation은 없다.

```bash
cat '<RUN_ROOT>/OPERATION_STATUS.json'
```

```bash
cat '<RUN_ROOT>/training/LIVE_STATUS.json'
```

```bash
cat '<RUN_ROOT>/training/FIRST_EPOCH_ETA.json'
```

첫 전체 epoch ETA는 train+full validation+checkpoint 포함 실측이다.
평균 ETA와 최근 최대 epoch×1.25+60초의 보수적 ETA를 함께 기록한다.
최대60 완료 예측을 초과해도 승인 한도를 늘리지 않고 다음 epoch가 들어갈 때만 계속한다.
마무리 예약은 템플릿 기본600초이며 학습 deadline을 전체 deadline보다 그만큼 앞당긴다.

정상 종료: `OPERATION_STATUS.json.status=TRAINING_COMPLETED`,
`training/DELIVERY_ARCHIVE_READY.json`과 `training_complete=true`를 함께 확인한다.
한도 경계 중단: `LIMIT_STOPPED_AT_COMPLETE_EPOCH`, `training/PARTIAL_ARCHIVE_READY.json`,
`training/translator_validation_r2_partial.tar.gz`; 정상 학습 완료가 아니다.
외부 timeout/오류: `LIMIT_STOPPED`/`ERROR_STOPPED`,
`PARTIAL_EVIDENCE_READY.json`, `partial_evidence.tar.gz`; 모델 export READY와 구분한다.
`WATCHDOG_EXIT.json.cleanup.remaining=[]`는 해당 group 회수 결과다.
중단 불가능 I/O나 강제 종료로 marker가 없으면 완료로 추정하지 않는다.

archive 다운로드 후 `.sha256`을 대조하고, 해당 controller/ranks 종료를 확인한 뒤
사용자가 Dashboard에서 Pod를 수동 Stop한다. Network Volume은 삭제하지 않는다.
프로세스 종료 이후에도 Pod Stop 전까지 idle 과금은 계속된다.
