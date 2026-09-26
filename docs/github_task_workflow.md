# GitHub 연구 업무 파이프라인

## 정보가 위치하는 곳

| 위치 | 역할 | 여기에 두지 않는 것 |
|---|---|---|
| `README.md` | 연구 목적과 팀원용 진입점 | 일별 실행 로그 |
| `docs/research_progress_summary.md` | 검증된 현재 상태, 완료 결과, 다음 gate | task별 긴 토론 |
| GitHub Project | 상태·담당·순서·blocker와 원문 링크 | 상세 프로토콜과 결과 전문 |
| canonical Issue | task 범위, 완료 조건, 결정과 진행 논의 | 큰 바이너리·원시 로그 |
| Pull Request | 검토 가능한 변경 diff, 관련 Issue, checks | 아직 합의하지 않은 장기 계획 |
| `reports/` | 실행 명령, 환경, 정량 결과, 한계 | 작업 상태의 유일한 기록 |

새 팀원은 `README → 현재 연구 진행 요약 → Project의 In Progress 카드 → 연결 Issue → PR·report` 순서로 읽습니다. Project 카드에는 문서 전문을 복사하지 않고 canonical 원문 링크를 유지합니다. 프로젝트 전체의 고수준 변화는 GitHub Project status update로 짧게 남기되, 세부 증거는 반드시 Issue·PR·report에 연결합니다.

## 기본 흐름

1. Project `Tasks`에서 작업을 고른다.
2. 코드·실험·문서 작업은 canonical `vos-memory-translator-nonlinear` 저장소의 `Research task` Issue로 만든다. 회의와 날짜형 milestone만 Draft로 유지한다.
3. Issue를 Project에 연결하고 작업 시작 시 `In Progress`로 바꾼다.
4. 로컬 `cmmt-task` 기록에 완료 기준·명령·산출물을 누적한다.
5. 작은 단위로 commit하고 PR에 Issue를 연결한다.
6. 완료 기준, 재현 명령, 결과 링크가 모두 있을 때만 Issue를 닫고 `Done`으로 바꾼다.

코드·실험을 이미 시작했는데 카드가 Draft인 것을 발견하면 작업을 폐기하지 않는다. 즉시
canonical repository Issue로 전환하고, 기존 commit·보고서·실험 명령을 소급 연결한 뒤
다음 작업부터 정상 흐름을 따른다. Draft 카드만 둔 상태에서 새 실험을 계속하지 않는다.

GitHub Project의 기존 workflow는 Issue 자동 추가와 닫힌 Issue의 `Done` 이동에 사용한다. 로컬 도구는 연구 증거가 빠진 채 `Done`이 되는 것을 막는다.

## 명령 예시

```bash
cmmt-task init \
  --task tasks/model-state/task.json \
  --task-id model-state \
  --title "Map Small/Base+ state" \
  --issue-url https://github.com/memorybridge-team/vos-memory-translator-nonlinear/issues/7 \
  --owner KIMKYUDO \
  --criterion "Small/Base+ tensor contract table exists" \
  --criterion "Base+ self-injection command and log exist"

cmmt-task start --task tasks/model-state/task.json

cmmt-task record \
  --task tasks/model-state/task.json \
  --artifact reports/model-state/state-contract.md \
  --command-line "python -m pytest -q tests/test_state_transfer.py"

cmmt-task check \
  --task tasks/model-state/task.json \
  --criterion "Small/Base+ tensor contract table exists"

cmmt-task finish \
  --task tasks/model-state/task.json \
  --summary "State contract and self-injection were verified."
```

`finish`는 다음 조건을 만족하지 않으면 실패한다.

- Issue URL이 `vos-memory-translator-nonlinear`을 가리킨다.
- 모든 완료 기준을 확인했다.
- 실제로 존재하는 산출물을 하나 이상 기록했다.
- 재현 또는 검증 명령을 하나 이상 기록했다.
- 결과 요약을 작성했다.
