# 기존 저장소에서 v2로 옮긴 범위

## 2026-10-03 LVOS 고정 모델 통합 출처

- `transformer_translator.py`와 `frozen_tensor_api.py`는 모델팀 branch
  `feat/transformer-adapted-translator`의 `746ea3e7d84c366c2d7ac06159e90a1f684bca56`에서
  선별 이관했다. 원 commit의 Git author는 `fragile`이다. 이번 pipeline 작업자가 모델 구조를 새로 설계했다고 표시하지 않는다.
- 모든 architecture class와 tensor API class/function body의 AST 동일성을 확인했다.
  기존 `translators.py`를 덮어쓰지 않고 tensor base를 별도 module에 연결했다.
  기존 저장소에 없는 MomentMatched registry import는 해당 registry 선택 분기로 이동했다.
  LVOS runner는 승인된 `base`만 받으며 다른 preset의 통합을 주장하지 않는다.
- 원본/이관 파일 SHA와 비교 pin은 `configs/lvos_reference_lock.json`에 기록한다.
  SAM2 pinned commit과 기존 collector `4a1bb1b`의 출처·변경을 보존한다.
- Benchmark `feature/best_model_selection`의 `bcf0a0f6a36c4129d487e5e58e151468d7ca714b`
  (Git author `RohSeongmin`)에서 실제 `best_model.py` 함수를 읽고, 지정된 별도 checkout에서 호출한다.
  Metric 함수 원저작을 pipeline 기여로 바꾸지 않는다. Baseline generator와 최종 export 수용 계약은 미제공이다.
- LVOS audit/view/runner/gates/orchestration와 CPU 검증은 이번 task branch의 통합 작업이다.
  실제 SAM checkpoint GPU 결과와 배포 성공은 아직 없다. 상세 근거는
  `reports/tasks/07_paired_training/LVOS_PIPELINE_REPORT_2026-10-03.md`에 남긴다.

기준일: 2026-09-18 KST. 원본 저장소: `memorybridge-team/vos-memory-translator-nonlinear`의 당시 `main` 및 로컬의 미커밋 연구 문서.

## 이관한 것

- `src/vos_memory_inspector/`: 실제 SAM 2 state 추출·주입, canonical schema, cache, DAVIS 평가, MLP 코드. 일부 Ridge/Linear 경로는 과거 결과 재현과 회귀 검사용으로 남겨 두었다.
- `tests/`: 기존 동작 검증. 새 Base+/MOSEv2/LVOS v2 검사는 이후 추가한다.
- `scripts/`: paired-state 수집, nonlinear 학습·평가, 기존 baseline, 결과 집계에 쓰는 명령만 선별했다.
- `SAM2_UPSTREAM_COMMIT`, `pyproject.toml`, `.gitignore`, DAVIS manifest 정책.
- 구조 이해에 필요한 조립 지도와 핵심 선행 연구 설계 문서. 기존 Tiny→Large 결과 중 규모가 작은 MLP 보고서와 rare-event 요약은 `reports/legacy/`에 옮겼다.

## 새로 작성하거나 고친 것

- `README.md`, `PROJECT_CONTEXT.md`, `docs/experimental_plan.md`: Tiny→Base+, 세 데이터셋, 4주 논문 일정과 nonlinear 범위로 재작성.
- `scripts/runpod_bootstrap.sh`: 새 기본 target을 Base+로 변경.
- 새로운 checkpoint 기반 Base+ 결과, MOSEv2/LVOS v2 로더와 객체별 anchor baseline은 구현·검증 전이다. 존재하지 않는 실험 결과를 옮겨 쓰지 않는다.

## 이관하지 않은 것

- 과거 Tiny→Large 전용 대량 gallery·이미지, 오래된 날짜별 보고서 대부분, 과거 학습용 대형 tensor cache, 원본 데이터·checkpoint, RunPod SSH 정보.
- `run_tiny_large_probe.py`와 `runpod_direct_smoke.sh`처럼 모델 쌍이 고정된 예전 실행 스크립트.
- 과거 Git commit history. 새 저장소의 첫 commit은 새 루트이며 이전 commit을 parent로 가져오지 않는다.

## 작성 기여와 복구

### 2026-09-29 paired-state ZIP repair의 출처

사용자가 제공한 `paired-state 제작 코드.zip`(SHA256
`312702886b9ae7c231fb1abcaee69e35eec9fa5822c33c9333b88fd24dc193aa`)의
단일 객체 case 순서와 active-memory 선택 방식을 검토하여 기존 training branch 위에 필요한 수집·검증·변환을 반영했다.
ZIP 자체의 작성자와 Git revision은 확인되지 않았다. 이를 이번 commit 작성자의 원저작으로 재표기하지 않는다.
ZIP와 실제 배포 source가 같다는 증거도 아직 없다. 원본 ZIP 및 기존 팀원의 파일/이력은 보존한다.
변경 범위와 검증은 `reports/tasks/07_paired_training/REPAIR_REPORT.md`에 기록한다.

새 저장소의 초기 commit 작성자는 GitHub 계정 `KIMKYUDO`로 설정한다. 이는 **파일 전체를 KIMKYUDO가 처음 작성했다는 뜻이 아니다.** 이전 저장소에는 `KyudoKim`과 `서수빈`의 커밋이 있으며, 후자는 초기 memory probe 등 일부 이관 파일에 기여했다. 과거 저장소의 `git log`와 이 문서가 작성 경위를 설명한다. 원본 Git 이력과 전체 산출물을 복구할 수 있도록 로컬 기존 checkout의 `.git`을 보존한다. 원본 GitHub 저장소를 삭제하면 그 저장소의 Pages URL 및 GitHub 상의 history 접근도 사라진다.

새 GitHub 저장소의 Contributors 표시는 새 commit의 작성 이력에 의해 계산된다. 초기 commit을 단일 GitHub 계정의 검증 가능한 이메일로 작성한 뒤 확인한다. 이후 팀원 기여가 생기면 정당한 작성자를 그대로 기록한다.

## 2026-09-18 이관 실행 결과

- 새 저장소 `main`의 독립된 첫 commit `dc1d667e0a4f42739f1e86d66f2715e06b11f9c2`를 push했다. GitHub Contributors API에서 `KIMKYUDO` 한 명만 확인했다.
- 새 저장소 GitHub Pages를 `main /docs`로 설정했다. 공개 주소 `https://memorybridge-team.github.io/vos-memory-translator-nonlinear-v2/`의 HTTP 200 응답을 확인했다.
- 기존 원격 `main`과 로컬 checkout의 commit이 일치하며, 원격 branch는 `main` 하나이고 tag는 없다. 원본 이력 백업 `.external/archives/vos-memory-translator-nonlinear-before-v2.bundle`을 만들고 `git bundle verify`로 검증했다. 이 bundle은 기존 로컬 작업공간에만 있다.
- **기존 GitHub 저장소 삭제는 미완료**다. GitHub REST 삭제 요청은 `403 Must have admin rights to Repository`로 거절됐다. 브라우저 UI 진입도 자동 승인 검토에서 거절돼 우회하지 않았다. 저장소 관리자 계정으로 기존 저장소의 Settings → General → Danger Zone에서 삭제해야 한다. 그전에는 기존 저장소와 Pages가 남아 있다.
- 기존 로컬 checkout의 `origin`은 해제했다. 코드, 미커밋 변경, Git 이력과 위 bundle은 그대로 보존하며 새 저장소 `v2/`만 새 원격에 연결한다.

## 2026-10-03 LVOS benchmark bridge 출처

- `vos-memory-benchmark`의 `feature/best_model_selection` revision `bcf0a0f6a36c4129d487e5e58e151468d7ca714b` (`best_model.py`, 작성자 RohSeongmin)을 canonical J/F 및 three-fraction `selection_score`의 실행 reference로 사용한다.
- 별도 `feature/baseline` revision `dcd335380dbe62f3299dbc5d4456b5ac8a12b4b4` (`baseline/no_handoff.py`, `main_metrics.py`, 작성자 RohSeongmin)의 Full Replay entrypoint와 legacy pooled reducer를 확인했다. 원본 repository를 수정하지 않는다.
- `tests/fixtures/benchmark_metric.py`, `benchmark_full_replay.py`, `benchmark_legacy_metrics.py`는 위 원본 파일을 CPU regression용으로 보존한 사본이다. Canonical source hashes와 pins는 `configs/lvos_benchmark_reference_lock.json`에 둔다. 이 파일들의 설명에서 Full Replay를 상한으로 부르는 원문은 원본 출처이며 이번 연구의 수학적 상한 주장이 아니다.
- 신규 `lvos_benchmark.py`, `lvos_metrics.py`는 frozen membership/prompt/provenance와 training output 계약에 연결하는 adapter다. 원저작 기여를 이번 commit 작성자의 신규 모델/metric으로 표기하지 않는다. Fixed model revision `746ea3e7d84c366c2d7ac06159e90a1f684bca56`의 architecture body와 frozen split은 이번 repair에서 변경하지 않았다.
