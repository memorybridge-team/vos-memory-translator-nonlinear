# 학습 파이프라인 PR 검토 및 실행 증거

검토일: **2026-10-13 KST**. 대상: 기존 PR #13 / `feat/frozen-split-validation-r2`.
base `55b117dcae4a0102b92382ed06cbee4f1fc3956c`.
이번 작업은 GitHub 게시 검토·코드 정리·문서 갱신이다. RunPod 접속/학습/추가 GPU
검사/전체 cache 검사/Pod 변경은 수행하지 않았다. 기존 실행 증거를 재사용했다.

## 1. GitHub에 필요한 코드

- `state_training/`: frozen inventory/split, 기존 cache read-only adapter/index,
  train-only RMS, validation target 충분통계, 단일 모델 DDP MSE 학습,
  full-validation state R² best/early stopping, strict checkpoint/epoch-boundary resume,
  정상·부분 전달 archive, 한도 controller/watchdog.
- 모델팀의 frozen `transformer_translator.py`, `frozen_tensor_api.py`는 위 경로의
  필수 dependency다. 구조/preset/I/O 변경 없이 보존하며 원 저자·SHA는
  [MIGRATION.md](../../MIGRATION.md)를 따른다.
- tests/config/CLI entry point, readonly notebook, 한국어 운영 안내.
- 초기 PR head4fb807b 이후 검증한 ad20da1/e615e55/d9568c6/040ca88/165d81a를 포함한다.

기존 main runtime/export/inject·frozen manifest는 수정하지 않았다. 이전 로컬
collector/LVOS/official 학습·J&F 운영 branch 전체는 이 PR에 가져오지 않았다.
공유 저장소의 기존 API를 사용하지 않는다는 이유만으로 삭제하지 않는다.
과거 보고서는 역사적 상태임을 표시하고 최신 기록으로 연결한다.

## 2. 이번 검토에서 정리한 항목

|발견|조치·검증|
|---|---|
|새 실행기에서 사용하지 않는 `group_members()`|제거. src/tests 전체 AST에서 참조0. 제거 후 모듈 AST가 source165d81a에서 해당 함수만 제외한 AST와 동일|
|PR/운영 문서가 “GPU 미실행”으로 남음|실제 입력·CPU·smoke·학습 시작 증거로 갱신. 최종 완료/weights 미확인은 유지|
|최초 승인 시작과 잔여 예산을 함께 전달하는 예제|`ORIGINAL_APPROVED_USD`로 수정. 시작 이후 준비/idle을 코드가 차감하므로 이중 차감을 피함|
|템플릿에 과거 GPU UUID·승인 시각 고정|명시적 placeholder로 변경. 기존 만료 승인을 새 승인으로 해석하지 않음|
|통신 PASS 설정/evidence를 템플릿에 찾기 어려움|구현된 communication_env/evidence/SHA, smoke_seconds를 명시하고 실제 JSON 계약 설명|
|운영 설정/인증 폴더/source archive 오추적 가능성|`.local.json`, `jupyter-private/`, `.bundle`, `.tar.gz` ignore 추가·check-ignore 확인|

모델/trainer/loss/optimizer/scheduler/split/index/통계 구현을 추가 변경하지 않았다.
현재 package SHA는 `7a79c4c94d726a26d5f8259a2f6c1f8746d47add24223fac33129b121bee9935`이며,
실제 GPU 검증 source165d81a의 package SHA는
`097c7ba52d2fb51f1b04080cd1ff00aa3a4fa3916ca32bea18eaa2f6e23c9724`다.
차이는 위 미참조 helper 제거다. **기존 run은 source165d81a로만 재개한다.** 새 PR HEAD로
실행 중 소스를 덮어쓰거나 strict identity를 우회하지 않는다. source165d81a는 Git 이력에 보존한다.

## 3. 증빙 재사용 / 신규 검사 / 미실행

### 기존 및 2026-10-09 원격 증거 재사용

- frozen split metadata32 PASS, seed7, MOSE3300/183/183, 영상 ID 교집합0,
  재실행 membership/SHA 동일. 원본 clip 계보 전체는 `NOT_FULLY_VERIFIED`.
- 동일 model revision746ea3e의 기존 forward/gradient/reload 증거 재사용.
- RunPod CPU: statistics/운영12 PASS(23.09초), delivery marker1 PASS(11.99초),
  UUID/smoke limit1 PASS(0.31초), NCCL binding1 PASS(0.25초), 각 exit0.
  **네 별도 실행 총15개이며 전체15개를 한 번에 재실행한 결과가 아니다.**
- 최소 실제 cache CPU 연결4 PASS; 기존 tensor 검사22248건을 unchanged SHA/stamp/
  semantic/RGB binding 조건으로 재사용. 신규 전체 tensor audit는 하지 않았다.
- 실제 index1078.67초, 새 train-only RMS/validation target stats1535.37초/exit0.
  완료한 split/index/statistics-v2는 이후 재생성하지 않았다.
- 작은 2-rank CUDA 통신: device binding만 적용한 구성은35초 timeout,
  해당 검사에만 P2P_DISABLE=1을 적용한 한 번의 비교는 양 rank all_reduce3.0/exit0.
  두 검사 모두 소유 프로세스 회수 remaining=[]; 시스템/드라이버 설정 변경 없음.
- source165d81a 실제 2-GPU smoke PASS: 2 diagnostic epochs/2 updates,
  두 branch 실제 갱신, strict reload, rank max difference0. 모델 단계46.73초.
- 공식 학습 epoch1에서 SSH 종료·재접속 전후 step900→1040,
  동일 controller/watchdog/guard/torchrun/두 rank의 PID 및 시작 tick 유지.
- Jupyter 인증200/미인증403·실제 notebook 접근200은 당시 인계 증거다.
  인증값·원격 endpoint·private key·실행 wrapper·원본 로그는 PR에 포함하지 않는다.

선별한 실제 실행 수치·명령·source/data binding은
[RUNTIME_VERIFICATION_2026-10-09.json](RUNTIME_VERIFICATION_2026-10-09.json)에 있다.
원본 실행 증거는 개발방 `outputs/r2_remote_20261009T043718Z`에 보존한다.

### 이번 신규 정적 검사

- src/tests Python AST43개, CLI help10개 exit0. Notebook 셀 문법·읽기 전용 확인.
- 배포 source archive165d81a와 비교: source Python 변경은 미참조 함수 삭제 한 곳뿐.
  실행 함수 AST 동일. 모델팀 두 파일 LF SHA 동일.
- R² spec JSON/소스 SPEC 일치, console entry point/global64, 문서 fence 확인.
- PR 대상 파일에서 대형 tensor/archive·private key/token·SSH endpoint 없음.
  운영 파일 ignore 동작 확인. `git diff --check` 통과.
- 실제 split 생성·정규화·tensor 연산을 노트북에서 실행하지 않았다.

세부 결과: [PR_STATIC_REVIEW_2026-10-13.json](PR_STATIC_REVIEW_2026-10-13.json).

### 미실행·아직 확인하지 않은 항목

- 이번 PR 작업에서 최종 전체 회귀·CPU tensor·GPU 검사를 다시 실행하지 않았다.
- 마지막 관측은 2026-10-09T07:57:42Z 학습 시작이다. 첫 전체 epoch ETA,
  최종 완료 epoch/종료 사유/R² best/정상 전달 archive와 SHA는 아직 수집하지 않았다.
- 새 source의 실제 중단→재개 통합과 정상 학습 종료→실제 최종 delivery까지의
  end-to-end gate는 이 증거로 완료 표시하지 않는다. marker 파일 fixture와 구분한다.
- 실제 검증 환경은 Python3.12.11/Torch2.8.0+cu128이다. core의 최소 의존성 표기가
  더 낮다는 이유로 그 모든 버전에서 이 학습 runner가 검증됐다고 주장하지 않는다.
- controller는 기존2GPU·1.16/hour·12시간·USD30 승인용이다. 다른 승인값용 범용
  controller라고 보고하지 않는다. 범용 통신 진단 CLI도 이 PR에는 없다.
- J&F·test/benchmark·baseline·새 pair 생성은 수행하지 않는다.

## 4. 실제 입력과 의미

|역할|MOSE cases / records|LVOS cases / records|합계 cases / records|
|---|---:|---:|---:|
|train|18738 / 265945|1803 / 28848|20541 / 294793|
|validation|993 / 13887|714 / 11424|1707 / 25311|

수정 LVOS fit9는 모두 포함했다. MOSE train337/validation17개 원본 영상에는
선택된 case가 없고, 선언된 기대 case의 cache 누락은0이다. 원본 membership은
재배정하지 않았으며 internal-test1110cases는 봉인했다. full-validation R²는
validation index의 모든 유효 record 범위다. VOS rollout 성능이나 J&F를 뜻하지 않는다.

## 5. 검토 요청과 인수인계

검토 초점은 role/누수 차단, valid-record DDP tail 가중치, R² 통계·best/early 상태,
source/data/statistics/통신 binding, complete marker와 epoch-boundary resume,
owned-process watchdog 및 정상/부분 전달물 구분이다.

CLI/실행 승인/재개/평가팀 load API는 [운영 안내](../../docs/state_training_r2_operator_guide.md)를
따른다. 선정 모델은 **full-validation state R² best**이며 마지막 완료 checkpoint와
구분한다. J&F/test best나 전체 연구 gate 완료를 주장하지 않는다. 최종 산출물
확인은 별도 실제 기록 수집 gate로 남긴다. 이 PR은 merge를 자동 실행하지 않는다.
