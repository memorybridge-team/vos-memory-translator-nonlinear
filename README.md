# SAM 2.1 Small → Base+ Nonlinear Memory Handoff

이 저장소는 비디오를 처리하던 SAM 2.1 Small의 객체별 메모리를 nonlinear translator로 변환해, SAM 2.1 Base+가 과거 영상을 전부 다시 처리하지 않고 다음 프레임부터 분할을 이어갈 수 있는지 연구합니다.

**모델:** SAM 2.1 Small → Base+
**학습·in-domain 평가:** MOSEv2, LVOS v2

**external 평가:** VOST(primary external zero-shot), PUMaVOS(partial/unusual-mask stress), M³-VOS(material phase-transition stress)
**기간:** 2026-09-18부터 약 4주. 실제 투고 마감일은 확정 후 기록합니다.  
**제안 방법:** 객체별 spatial memory와 object pointer를 변환하는 nonlinear translator. Presence, frame ID, positional encoding과 객체 ID는 별도의 상태 계약에 따라 처리합니다.

한 달은 원고 일정이지 실험량의 상한이 아닙니다. 필요한 전체 데이터·반복·비교군·모델 후보를 완료하고, 장시간 GPU 실험은 재개 가능한 작업으로 운영합니다.

## 현재 상태

이 저장소는 이전 프로젝트에서 검증한 상태 추출·주입, DAVIS 평가, 기존 baseline runner, MLP 학습 코드를 **새 Git 이력으로 선별 이관한 시작점**입니다. 이전 Tiny→Large 실험은 [과거 결과](reports/legacy/)로 보존했습니다. Small/Base+ runtime I/O inventory와 Base+ same-checkpoint round-trip은 완료됐고, Small→Base+ 전체 baseline 비교와 nonlinear 학습 결과는 아직 생성되지 않았습니다.

### 팀원이 현재 연구 상태를 읽는 순서

1. [현재 연구 진행 요약](docs/research_progress_summary.md)에서 확정된 연구 질문, 완료 결과, 미완료 gate와 다음 작업을 먼저 읽습니다.
2. [Project #2 Tasks](https://github.com/orgs/memorybridge-team/projects/2/views/1)에서 현재 `In Progress` task, 담당자, 순서와 blocker를 확인합니다.
3. 해당 카드가 연결한 canonical Issue에서 범위·완료 조건·결정 이유를 확인합니다.
4. [`reports/` Task 인덱스](reports/README.md)에서 Task별 현재 보고서와 최종 통합본을 읽습니다.
5. 연결된 PR과 날짜별 `runs/`에서 실제 diff, 실행 명령, 원본 수치와 한계를 검토합니다.
6. 구현을 재현하거나 이어서 작업할 때만 [State I/O 계약](docs/design/small_base_state_io_contract.md), [Benchmark protocol](docs/design/03_benchmark_protocol.md), [협업 파이프라인](docs/github_task_workflow.md) 원문을 읽습니다.

Project Board는 상세 연구 문서를 복제해 보관하는 곳이 아니라 **상태·담당·완료 기준·근거 링크를 찾는 관제판**입니다. 안정된 현재 상태는 이 README와 진행 요약에, task별 논의는 Issue에, 검토 가능한 변경과 실행 증거는 PR·보고서에 둡니다.

### 연구·협업 진행판

| 단계 | 보드 상태 | 지금의 완료 조건 |
|---|---|---|
| 연구 범위·성공 기준 고정 | `Done` — [Issue #1](https://github.com/memorybridge-team/vos-memory-translator-nonlinear/issues/1) | [Baseline·성공·중단 기준](docs/design/01_scope_baselines_success_stop.md) 동결 완료 |
| Small/Base+ State I/O 계약 | `Done` — [Issue #2](https://github.com/memorybridge-team/vos-memory-translator-nonlinear/issues/2) | [I/O 계약 v1.1](docs/design/small_base_state_io_contract.md), [Task 02 최종 보고서](reports/tasks/02_state_io/FINAL_REPORT.md) |
| dataset·난이도·baseline·metric 동결 | `In Progress` — [Issue #4](https://github.com/memorybridge-team/vos-memory-translator-nonlinear/issues/4) | VOST gate 완료; PUMaVOS·M³-VOS download·manifest·loader·metric contract 검증 대기 |
| 상태 추출·self-injection·target injection | `Done` — [Issue #5](https://github.com/memorybridge-team/vos-memory-translator-nonlinear/issues/5) | [Task 06 최종 보고서](reports/tasks/06_runtime/FINAL_REPORT.md); correction·반복 handoff까지 exact |
| paired Small/Base+ state 수집 | `Todo` | MOSEv2/LVOS v2 fit/dev에서 future GT 없는 state-only pair, video-level split, checksum manifest, compact shard |
| 공통 evaluator·baseline | `Todo` | 같은 manifest에서 강한 재인코딩·replay·native 비교 결과 |
| nonlinear 후보 학습·비교 | `Todo` | MLP/gated MLP/slot-context 후보의 downstream·비용 비교 |

실제 코드·실험 작업은 위 보드 카드를 canonical Issue로 승격한 뒤 `In Progress`로 바꾸고, branch의 commit·실행 로그·보고서를 연결합니다. 완료 기준과 재현 명령이 검증되기 전에는 `Done`으로 옮기지 않습니다. [Project #2 Tasks](https://github.com/orgs/memorybridge-team/projects/2/views/1)에서 전체 현황을 확인할 수 있습니다.

## 브랜치 이름 규칙

팀원이 이름만 보고 목적과 범위를 알 수 있도록 **작업유형/짧은-범위** 형식을 사용합니다.
모든 이름은 소문자 ASCII와 하이픈만 사용하고, 한 브랜치에는 하나의 작업만 담습니다.

- `feature/task-03-external-zero-shot-protocol` — Project task의 기능·프로토콜 구현
- `docs/task-03-benchmark-protocol` — 문서만 수정하는 작업
- `experiment/small-base-direct-copy` — 재현 가능한 실험·비교군 실행
- `fix/memory-shape-mismatch` — 결함 수정
- `chore/update-ci` — 빌드·CI·도구 정리

일반 흐름은 `main → branch → 작은 단위 commit → PR → review/checks → merge`다. PR 제목은
브랜치 범위와 같은 동사를 사용하고, Issue·실행 명령·결과 보고서를 연결한다. 연구 Task 번호가
중요한 경우 `feature/task-번호-범위`처럼 번호를 보존한다.

현재 PR #10은 이미 생성된 Task 03 PR과의 연결을 보존하기 위해
`task/03-external-zero-shot-protocol-v1-3`에서 계속 운영한다. 다음 새 작업부터는 위 규칙을
적용하며, 다음 문서 전용 작업은 `docs/task-번호-범위`를 우선 사용한다.

기존 `feature/task-06-continuation-injection`은 이전 작업의 역사적 branch로 보존합니다. 현재 최소 handoff 계약 전체를 검증하는 변경은 `codex/minimal-handoff-contract`처럼 범위가 넓은 branch로 분리했으며, 앞으로 새 작업은 위 의미 기반 이름을 우선합니다.

기존 코드를 가져온 경위와 제외한 자료는 [MIGRATION.md](MIGRATION.md)에 기록했습니다. 새 저장소의 Git 커밋 작성자와 과거 코드의 실제 작성 기여는 별개의 정보입니다.

## 연구 흐름

```text
Small이 switch 시점 t까지 처리 → 객체별 source state 추출
                                 ↓
                      Nonlinear Translator
                                 ↓
                   Base+ 상태 계약에 맞춰 주입
                                 ↓
              Base+가 t+1부터 같은 객체 추적
```

성공 기준은 state tensor가 비슷한지만이 아닙니다. 전환 후 객체 분할 J&F, 부재 중 오검출, 재등장 후 복구, 전환 지연과 전달량을 함께 봅니다.

실제 데이터 운영은 `translator fit → in-domain development → sealed in-domain final → external frozen benchmark`의 네 역할로 분리합니다. 주 translator는 MOSEv2/LVOS v2 train-fit의 `(Small state, Base+ native state)` 쌍으로 학습하며 future frame GT를 primary loss에 사용하지 않습니다. 같은 train의 video-disjoint dev에서만 dense GT downstream 성능으로 구조와 checkpoint를 선택합니다. 설정 동결 뒤 LVOS v2 validation은 local detailed final, MOSEv2 validation은 Codabench sealed final, VOST는 primary external zero-shot으로 사용합니다. PUMaVOS와 M³-VOS는 각각 partial/unusual-mask와 material phase-transition에 대한 complementary external stress benchmark입니다. M³-VOS 본문 주지표는 공식 `J/J_tr/J_cc`이며, boundary `F/J&F`는 사전 integrity gate를 통과한 경우에만 부록 보조지표로 냅니다. DAVIS는 현재 연구의 학습·평가·주장 범위에서 제외합니다.

## 계획과 비교군

[4주 실험 계획](docs/experimental_plan.md)에 각 주의 산출물·중단 기준·평가 규칙이 있습니다. [GitHub 연구 업무 파이프라인](docs/github_task_workflow.md)은 Issue→Project→commit/PR→증거 검증→Done 순서를 정의하고, [Project #2 감사 기록](docs/project_board_audit_2026-09-19.md)은 `01 Scope → 20 Submission` 연구 뼈대와 카드별 완료 기준을 남깁니다. Issue와 PR은 이 20단계를 수행하는 추적·검토 수단이며 별도 연구 단계로 세지 않습니다. [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)는 새 작업자가 검증된 연구 문맥을 빠르게 파악하는 요약본입니다.

주요 비교군은 Source-only, Base+-native/Full Replay, Direct State Copy, Moment-Matched Copy, 객체별 Original-Prompt, Last-Visible Source Mask, 두 anchor의 결합, Original-Prompt(s)+Replay-4/8/16, Nonlinear Translator입니다. Moment-Matched Copy는 학습 split에서 고정한 평균·표준편차로 memory·pointer의 분포만 보정합니다. 모든 방법에 같은 객체·switch·미래 프레임을 사용하며, 미래 GT는 평가에만 사용합니다.

Nonlinear 후보는 component-wise residual MLP, gated MLP, slot/context attention을 포함합니다. 각 구조를 동일 분할과 공정한 비용 계측으로 비교하고, 결과가 요구하면 추가 구조와 loss를 검토합니다.

## 코드와 결과물

| 경로 | 용도 |
|---|---|
| `src/vos_memory_inspector/` | SAM 2 상태 계약, 주입, 평가, translator 구현 |
| `scripts/` | 데이터 준비, baseline, nonlinear 학습·평가 진입점 |
| `tests/` | 상태 구조와 기존 기능 회귀 검사 |
| `configs/` | 재현용 설정과 manifest 정책 |
| `docs/experimental_plan.md` | 현재 유효한 4주 연구 계획 |
| `docs/design/03_translator_train_dev_final_evaluation_decision.md` | future-GT-free state fit, dense-GT dev, sealed/external final 평가 역할 결정 |
| `docs/github_task_workflow.md` | Issue·Project·commit/PR·완료 증거 운영 순서 |
| `docs/project_board_audit_2026-09-19.md` | Project #2의 01→20 task 감사·수정 기록 |
| `docs/design/small_base_state_io_contract.md` | Small/Base+ memory tensor와 translator 입력·복사·재생성 정책 |
| `docs/design/03_benchmark_protocol.md` | fit/dev·sealed in-domain·external zero-shot 역할, difficulty taxonomy, baseline 입력, metric·통계·누수 방지 계약 |
| `reports/README.md` | Task별 현재 보고서·최종 통합본·날짜별 실행 증거 인덱스 |
| `reports/tasks/` | Project Task 번호와 직접 대응하는 연구 보고서와 재현 증거 |
| `src/vos_memory_inspector/mose.py` | MOSEv2 validation first-frame-only manifest builder |
| `src/vos_memory_inspector/lvos.py` | LVOS v2 공식 split metadata 기반 manifest builder |
| `manifests/mosev2_valid_v1.json` | MOSEv2 validation 고정 switch/object manifest; 원본 데이터는 포함하지 않음 |
| `docs/architecture/` | 상태 조립 구조 설명 |
| `reports/legacy/` | 이전 모델 쌍의 제한적 결과; 새 실험 결과가 아님 |

이전 pilot의 Ridge·Linear 코드 일부는 회귀·과거 보고서 재현용으로 남아 있습니다. 이번 논문의 제안 방식 및 신규 학습 범위는 nonlinear입니다.

## 시작하기

Python 3.10+, PyTorch와 CUDA가 설치된 환경을 사용합니다. 공식 SAM 2는 별도 checkout으로 설치하고 [`SAM2_UPSTREAM_COMMIT`](SAM2_UPSTREAM_COMMIT)에 기록된 revision으로 고정합니다.

```bash
git clone https://github.com/memorybridge-team/vos-memory-translator-nonlinear.git
cd vos-memory-translator-nonlinear
python -m pip install -e ".[dev]"
python -m pytest -q
```

GPU Pod에서는 `bash scripts/runpod_bootstrap.sh "$PWD"`가 공식 SAM 2 및 Small/Base+ checkpoint를 준비합니다. 이어서 `bash scripts/runpod_preflight.sh "$PWD"`로 CUDA, Network Volume 여유 공간, revision과 SHA-256을 확인합니다. 첫 Base+ self-injection 실행과 종료 확인 방법은 [RunPod L4 운영 문서](docs/runpod_l4_operations.md)를 따릅니다. MOSEv2/LVOS v2/VOST/PUMaVOS/M³-VOS 원본, checkpoint, 개인 SSH 키, raw state cache는 Git에 넣지 않습니다. 각 데이터셋의 이용 조건과 저장 경로를 확인한 후 수집합니다.

현재 `prepare_paired_state_dataset.py`와 일부 baseline script의 DAVIS 전용 경로는 역사적 자료로만 남아 있으며 새 연구 실행에는 사용하지 않습니다. Task 03에서 MOSEv2/LVOS v2 manifest loader와 VOST loader/evaluator contract를 검증했다. paired-state 수집기와 모든 anchor baseline을 두 학습 데이터셋에 연결하는 작업은 Task 07·08에 남아 있습니다.
