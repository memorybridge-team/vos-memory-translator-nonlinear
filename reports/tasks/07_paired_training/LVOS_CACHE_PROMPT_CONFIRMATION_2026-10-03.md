# LVOS fit 9개 case — 데이터팀 확인 요청

2026-10-03. 원본 수정/삭제/재생성 없음. GPU 실행 없음.
1,803개를 실제 CPU 로딩·SHA·Small/Base+ 대응 검사했고 tensor 정렬은 모두 통과했다.
다음 9개의 conditioning frame은 실제 저장된 값 `[0]`이다. RGB 파일명으로 매핑한
frozen 최초 prompt와 달라서 과거 prompt 조건 확인이 필요하다.

|video/object|case IDs (official switch)|frozen prompt official/runtime|cache conditioning runtime|
|---|---|---|---|
|`2vt1LoVz` / obj2|`train:2vt1LoVz:obj2:switch966`, `switch1886`, `switch2801`|26 / 5|0|
|`8TrHEtoY` / obj2|`train:8TrHEtoY:obj2:switch1751`, `switch1991`, `switch2231`|1491 / 298|0|
|`XfbRshfV` / obj2|`train:XfbRshfV:obj2:switch546`, `switch1021`, `switch1496`|51 / 10|0|

복사 가능한 전체 case ID:

```text
train:2vt1LoVz:obj2:switch966
train:2vt1LoVz:obj2:switch1886
train:2vt1LoVz:obj2:switch2801
train:8TrHEtoY:obj2:switch1751
train:8TrHEtoY:obj2:switch1991
train:8TrHEtoY:obj2:switch2231
train:XfbRshfV:obj2:switch546
train:XfbRshfV:obj2:switch1021
train:XfbRshfV:obj2:switch1496
```

## 데이터팀에 확인할 내용

1. 저장된 conditioning frame 0은 실제 영상의 runtime frame 0인가, 잘라낸 prefix의 상대 index인가?
2. 그 시점에 제공한 mask는 어떤 파일/객체에서 읽었고 어느 원본 frame의 mask인가?
3. Small/Base+가 동일한 frame·mask·prompt 조건을 사용했는가?

현재 cache의 switch/non-conditioning frames는 전체 RGB runtime index에 대응한다.
이 관측만으로 conditioning 0의 생성 의도를 확정하지 않는다. Mask 원본/생성 당시 code·weights는 미확인이다.
9개 전체 SHA/frames는 로컬 `outputs/lvos_pilot_20261003T095455Z/prompt_details.local.json`에 저장했다.
원격 증거: `/workspace/CMMT-lvos-isolated/lvos-ddp-inspection-20261003T110200Z/prompt_details.json`.
연락할 데이터팀의 채널/수신자가 제공되지 않아 자동 메시지는 보내지 않았다. 이 문서를 전달하면 된다.

## 확인 지연 시 학습안

`configs/lvos_ddp_prompt_exclusions.proposed.json`에 정확한 9개 case ID와
`PENDING_PROMPT_HISTORY_CONFIRMATION`을 기록했다. **미승인 proposal**이며 본 학습에 자동 적용하지 않는다.
승인 시 frozen video membership을 유지하고 fit 1,479 / development 315 cases로 단일 run을 수행한다.
실제 사용 목록과 제외 이유는 run별 index/manifest에 남긴다. Development는 재분할/제외하지 않는다.
데이터팀 확인 결과와 실제 승인자·시각을 운영자가 기록한 후 학습 담당에게 전달한다.
