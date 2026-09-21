# SAM 2.1 Small → Base+ Nonlinear Memory Handoff

이 저장소는 비디오를 처리하던 SAM 2.1 Small의 객체별 메모리를 nonlinear translator로 변환해, SAM 2.1 Base+가 과거 영상을 전부 다시 처리하지 않고 다음 프레임부터 분할을 이어갈 수 있는지 연구합니다.

**모델:** SAM 2.1 Small → Base+
**데이터셋:** DAVIS 2017, MOSEv2, LVOS v2  
**기간:** 2026-09-18부터 약 4주. 실제 투고 마감일은 확정 후 기록합니다.  
**제안 방법:** 객체별 spatial memory와 object pointer를 변환하는 nonlinear translator. Presence, frame ID, positional encoding과 객체 ID는 별도의 상태 계약에 따라 처리합니다.

한 달은 원고 일정이지 실험량의 상한이 아닙니다. 필요한 전체 데이터·반복·비교군·모델 후보를 완료하고, 장시간 GPU 실험은 재개 가능한 작업으로 운영합니다.

## 현재 상태

이 저장소는 이전 프로젝트에서 검증한 상태 추출·주입, DAVIS 평가, 기존 baseline runner, MLP 학습 코드를 **새 Git 이력으로 선별 이관한 시작점**입니다. 이전 Tiny→Large 실험은 [과거 결과](reports/legacy/)로 보존했습니다. Small/Base+ runtime I/O inventory와 Base+ same-checkpoint round-trip은 완료됐고, Small→Base+ 전체 baseline 비교와 nonlinear 학습 결과는 아직 생성되지 않았습니다.

### 연구·협업 진행판

| 단계 | 보드 상태 | 지금의 완료 조건 |
|---|---|---|
| 연구 범위·성공 기준 고정 | `Done` — [Issue #1](https://github.com/memorybridge-team/vos-memory-translator-nonlinear/issues/1) | [Baseline·성공·중단 기준](docs/design/01_scope_baselines_success_stop.md) 동결 완료 |
| Small/Base+ State I/O 계약 | `Done` — [Issue #2](https://github.com/memorybridge-team/vos-memory-translator-nonlinear/issues/2) | [I/O 계약 v1.1](docs/design/small_base_state_io_contract.md), runtime inventory, paired dump, Map 동결 완료 |
| dataset·난이도·baseline·metric 동결 | `In Progress` | [Benchmark protocol v1.0](docs/design/03_benchmark_protocol.md); 이용조건·split manifest·loader 검증이 남음 |
| 상태 추출·self-injection·target injection | `In Progress` — KIMKYUDO | 단일/다객체·late prompt·재등장 exact, Small→Base+ injection 실행 완료; correction gate가 남음 |
| paired Small/Base+ state 수집 | `Todo` | video-level split, checksum manifest, compact state pair |
| 공통 evaluator·baseline | `Todo` | 같은 manifest에서 강한 재인코딩·replay·native 비교 결과 |
| nonlinear 후보 학습·비교 | `Todo` | MLP/gated MLP/slot-context 후보의 downstream·비용 비교 |

실제 코드·실험 작업은 위 보드 카드를 canonical Issue로 승격한 뒤 `In Progress`로 바꾸고, branch의 commit·실행 로그·보고서를 연결합니다. 완료 기준과 재현 명령이 검증되기 전에는 `Done`으로 옮기지 않습니다. [Project #2 Tasks](https://github.com/orgs/memorybridge-team/projects/2/views/1)에서 전체 현황을 확인할 수 있습니다.

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
| `docs/github_task_workflow.md` | Issue·Project·commit/PR·완료 증거 운영 순서 |
| `docs/project_board_audit_2026-09-19.md` | Project #2의 01→20 task 감사·수정 기록 |
| `docs/design/small_base_state_io_contract.md` | Small/Base+ memory tensor와 translator 입력·복사·재생성 정책 |
| `docs/design/03_benchmark_protocol.md` | 세 dataset, difficulty taxonomy, baseline 입력, metric·통계·누수 방지 계약 |
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

GPU Pod에서는 `bash scripts/runpod_bootstrap.sh "$PWD"`가 공식 SAM 2 및 Small/Base+ checkpoint를 준비합니다. 이어서 `bash scripts/runpod_preflight.sh "$PWD"`로 CUDA, Network Volume 여유 공간, revision과 SHA-256을 확인합니다. 첫 Base+ self-injection 실행과 종료 확인 방법은 [RunPod L4 운영 문서](docs/runpod_l4_operations.md)를 따릅니다. DAVIS/MOSEv2/LVOS v2 원본, checkpoint, 개인 SSH 키, raw state cache는 Git에 넣지 않습니다. 각 데이터셋의 이용 조건과 저장 경로를 확인한 후 수집합니다.

현재 `prepare_paired_state_dataset.py`와 일부 baseline script는 DAVIS 전용입니다. MOSEv2/LVOS v2 로더와 새로운 객체별 anchor baseline은 계획에 포함됐지만 아직 구현되지 않았습니다. 과거 Tiny→Large 명령을 재활용할 때 source를 Small로 바꾸더라도 checkpoint·config·cache metadata를 함께 검증해야 합니다.
