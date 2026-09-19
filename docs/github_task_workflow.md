# GitHub 연구 업무 파이프라인

## 기본 흐름

1. Project `Tasks`에서 작업을 고른다.
2. 코드·실험·문서 작업은 canonical `vos-memory-translator-nonlinear` 저장소의 `Research task` Issue로 만든다. 회의와 날짜형 milestone만 Draft로 유지한다.
3. Issue를 Project에 연결하고 작업 시작 시 `In Progress`로 바꾼다.
4. 로컬 `cmmt-task` 기록에 완료 기준·명령·산출물을 누적한다.
5. 작은 단위로 commit하고 PR에 Issue를 연결한다.
6. 완료 기준, 재현 명령, 결과 링크가 모두 있을 때만 Issue를 닫고 `Done`으로 바꾼다.

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
