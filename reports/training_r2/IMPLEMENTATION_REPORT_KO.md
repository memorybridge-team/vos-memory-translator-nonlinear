# 고정 split / validation R² 학습 구현 중간 보고

작성 2026-10-09. **Draft / 원격 검증 대기**.
최신 main `55b117d`에서 만든 `feat/frozen-split-validation-r2`에서 작업한다.
기존 `feat/official-mixed-state-training`의 `604644c` 및 미추적 teammate guide는 보존했다.
main 직접 수정·기존 학습/원본/cache/split 변경·GPU 실행은 하지 않았다.

## 변경

- `state_training/splits.py`: authoritative original inventory→group split 90/5/5,
  stable SHA order/seed7/largest remainder, immutable freeze와 재사용, unknown 차단.
- `index.py`, `cache.py`, `loader.py`: 저장 위치와 새 role 분리, internal-test 차단,
  source/target valid record 대응, late prompt runtime 보존, SHA/stamp 안정성,
  used/excluded와 dataset/role별 원본·available·사용 case/record 수.
- 기존 tensor-validated index는 명시한 SHA/semantic/RGB/file binding이 같은 경우
  검사 증빙만 재사용한다. 새 role/RMS는 기존 값으로 대체하지 않는다.
- `statistics.py`, `r2.py`: 새 train-only RMS, float64 channel/coordinate target
  sufficient statistics, variance-weighted branch R², 네 지표 equal mean,
  constant variance<=1e-12 제외·진단, duplicate/tail/undefined/NaN 차단.
- `policy.py`, `training.py`: MSE loss와 frozen body 유지, max/absolute R² 제어,
  best와 early meaningful improvement 구분, exact DDP valid-record 가중치,
  complete-epoch strict identity resume. 기존 checkpoint warm-start 금지.
- 마지막 공식 run의 microbatch32×2/accumulation1/global64를 기본값으로 보존.
  4 rank는 새 config에서 microbatch16/global64. 서로 다른 world/batch resume는 미지원.
- `checkpoint.py`, `delivery.py`: atomic save 이후 strict reload 검증 후 marker,
  best R² weights와 last complete 분리, SHA/data/source/config/statistics binding,
  전달 archive와 readonly status/monitor.
- 기존 frozen model 두 파일을 byte 그대로 선별 이관. 원 저자/출처는 `MIGRATION.md`.
  최신 main의 runtime/benchmark 분리와 frozen manifests를 되돌리지 않았다.

## 기존 증빙 재사용

모델팀 revision `746ea3e7d84c366c2d7ac06159e90a1f684bca56`, base config,
입력 spatial BF16 `[B,O,K,64,64,64]`, pointer FP32 `[B,O,K,256]`.
model/API 파일 hash가 기존 `model_config.json`과 동일함을 확인했다.
기존 CPU delivered strict load와 실제 epoch39의 양 branch 갱신·strict output/
optimizer/scheduler reload·rank difference0 증빙을 재사용한다.
파일 SHA와 재사용 범위는 `EXISTING_EVIDENCE_REUSE.json`에 기록했다.
이 증빙을 새 split/R²/control/checkpoint metadata의 실행 PASS로 표시하지 않는다.
동일 모델의 gradient/forward 검사를 반복하지 않았다.

## 신규 실행 / 미실행

|항목|상태|근거|
|---|---|---|
|중간 metadata/numeric fixture|32 PASS, exit0|stdlib unittest; mock video IDs/list vectors, 실제 데이터/tensor 아님|
|최종 source/test AST|36 PASS|`STATIC_VERIFICATION.json`|
|최종 CLI --help|8 PASS|root+inventory/freeze/plan/index/statistics/train/monitor|
|config/readonly notebook/model byte equality|정적 PASS|global64, 실제 기존 microbatch32, 명령 제어 없는 조회 cell, 두 모델 파일 SHA 동일|
|최종 stdlib 재검사|미실행|prior-index 경로·batch 보존 수정 이후. 내일 RunPod에서 실행|
|작은 synthetic tensor/statistics/metadata tests|6개 미실행|`tests/test_state_training_statistics.py`; CPU RunPod 승인 대기|
|실제 inventory/split/index/RMS|미실행|현재 endpoint/원격 CPU 승인·full official inventory 미제공|
|실제 manifest SHA/video/case/valid record 수|미확인|기존 run 수치를 새 split의 수치로 대체하지 않음|
|새 runner 통합/실제 GPU 학습|미실행|공식 GPU 학습은 승인 범위 밖|

첫 Windows 기본 temp-directory 테스트는 파일 접근 문제로 종료했다.
작업 폴더의 전용 temp 경로에서 중간 32 tests가 1.406초에 통과했다.
최종 revision에는 정적 검사만 수행했으므로 전체 runtime PASS라고 보고하지 않는다.
새 정책은 저장한 상태를 복원하도록 구현했지만 실제 tensor checkpoint-resume 통합은 대기다.
새 pipeline runtime/storage 비용은 미측정, 이번 원격/GPU 작업은 실행하지 않았다.

## 다음 단계

1. 사용자로부터 현재 SSH/host key와 원격 CPU 실행 범위·시간·예산을 받는다.
2. RunPod에서 최종 stdlib/작은 tensor tests만 수행한다. 동일 모델 body 재검사는 재사용한다.
3. official **전체** video list/RGB root와 그룹 관계를 확인한다. 기존 case 목록으로
   원본 inventory를 대신 만들지 않는다.
4. 변경된 loader/index 경로에 소수 cache record를 연결하고 재사용할 기존 index SHA를 확인한다.
5. 승인된 범위에서 새 실제 split/index와 train-only RMS/validation target statistics 생성.
6. 별도 공식 GPU 승인 후에만 새 계약 학습을 시작한다. 이전 exposed weights는 resume하지 않는다.

준비·학습·재개·model load 명령과 필요한 입력은
[운영 안내](../../docs/state_training_r2_operator_guide.md)에 있다.
최종 학습 완료 시 “본 모델은 validation state R² 기준으로 선정되었으며,
J&F 및 외부 benchmark 평가는 별도 담당자가 수행한다.”를 전달 보고서에 기록한다.
현재 새 모델 선정/전체 연구 gate 완료를 주장하지 않는다.
