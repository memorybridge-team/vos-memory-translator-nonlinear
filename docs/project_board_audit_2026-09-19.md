# GitHub Project #2 task 감사·수정 기록

기준일: 2026-09-19 KST. Project의 `Tasks`와 `Roadmap`에 있던 20개 카드의 제목, 본문, 담당자, 상태, 날짜를 직접 확인했다. 감사 후 Infra 카드 1개를 추가해 현재는 21개다.

## 결론

연구 단계의 큰 흐름은 충분하다. 새로운 대규모 단계를 추가하기보다 기존 카드의 비어 있는 완료 기준을 채우고, GPU 실행 환경을 별도 task로 추가하는 것이 적절하다.

감사에서 발견한 즉시 수정 항목은 두 가지였다.

1. `[Scope]` Issue #1 본문의 `Tiny → Base+`를 `Small → Base+`로 수정했다. target-native 필요성 gate, same-checkpoint injection, 필수 baseline, 통계·누수 기준도 함께 명시했다.
2. `[Benchmark] Freeze dataset...` 본문의 `Tiny→Base+`를 `Small→Base+`로 수정하고, switch manifest·사건 표본 수·video-clustered confidence interval 고정 조건을 추가했다.

사용자 확인 결과 `-v2` URL의 redirect는 새 저장소의 이름을 `vos-memory-translator-nonlinear`로 변경해서 생긴 정상 동작이다. 로컬 `origin`, README, Issue 검증 규칙과 Pages 링크를 canonical 이름으로 동기화했다.

## 카드별 판정

| # | Task | 판정 | 보완할 완료 기준 |
|---:|---|---|---|
| 1 | Submit IEIE paper | 수정 완료 | 접수번호, 제출 파일 hash, 최종 원고 보관 위치 |
| 2 | Mentor agendas | 수정 완료 | 회의별 agenda 링크, 결정·action 기록 위치 |
| 3 | Validate paper file/dry run | 보완 완료 | 마감 시간대와 등록/결제 확인 증거 |
| 4 | Verify reproducibility | 수정 완료 | lockfile, 환경 inventory, checkpoint/data hash, clean-run 결과 |
| 5 | Internal review | 수정 완료 | claim-evidence, 인용, 통계, 누수, 한계 검토 checklist |
| 6 | Freeze tables/figures | 수정 완료 | 표 숫자 원본 JSON 매핑, 그림 생성 명령, 수동 전사 금지 |
| 7 | Result freeze | 보완 완료 | 제외 사례와 실패 run을 포함한 freeze manifest |
| 8 | Integrate paper | 수정 완료 | section별 입력·산출물, notation/claim consistency |
| 9 | Full baseline comparison | 수정 완료 | 실행 matrix, retry/failure 규칙, video-clustered CI |
| 10 | Full training | 보완 완료 | checkpoint 선택 규칙과 중단/재개 조건 |
| 11 | Candidate review | 보완 완료 | 구조별 go/no-go rubric과 결정 기록 링크 |
| 12 | Overfit/pilot | 수정 완료 | overfit 통과 수치, held-out smoke, deterministic rerun |
| 13 | Nonlinear candidates | 보완 완료 | 공통 parameter/compute budget과 checkpoint 선택 규칙 |
| 14 | Common evaluator/baselines | 보완 완료 | dataset별 공식 evaluator와 결과 일치 test |
| 15 | Paired memory pipeline | 보완 완료 | strata별 sample count와 storage capacity report |
| 16 | Extraction/injection | 보완 완료 | same-checkpoint 허용 오차와 uninterrupted 비교 규칙 |
| 17 | Paper title/outline/RQ | 수정 완료 | 연구 필요성, related work, 반증 가능한 claim, 과장 금지 |
| 18 | Dataset/taxonomy/baseline freeze | 오류 수정·보완 완료 | Small 표기, switch selection rule과 표본 수 고정 |
| 19 | Map memory tensors | 보완 완료 | component별 source/target 예시 dump 링크 |
| 20 | Scope/success criteria | 수정 완료 | Small 표기, target-native 이득 확인 gate |

## 추가한 task

### `[Infra] Freeze RunPod environment, storage and artifact sync`

- CUDA/PyTorch/SAM 2 revision과 Small/Base+ checkpoint hash를 기록한다.
- Network Volume의 dataset/cache/checkpoint/log 디렉터리 계약을 정한다.
- 중단·재개 smoke test와 로컬/GitHub에 올릴 경량 산출물 동기화 절차를 검증한다.
- 장시간 job의 PID, log heartbeat, GPU utilization, 완료 marker 확인법을 문서화한다.

보드에 21번 카드로 추가했다. 담당자는 `KIMKYUDO`, 상태는 `Todo`, 기간은 2026-09-20∼2026-09-24다.

문헌 필요성 감사와 통계 프로토콜은 새 카드를 늘리기보다 각각 `Paper title/outline/RQ`, `Freeze dataset...`, `Internal review` 카드의 완료 기준으로 넣는 편이 중복이 적다.

## 최종 판정

- 21개 카드로 범위→상태 I/O→데이터·평가→nonlinear 후보→전체 학습·baseline→논문·재현성→제출의 전체 단계가 덮인다.
- 추가 대규모 카드는 현재 필요 없다. 실제 구현을 시작할 때 Draft 카드를 canonical repository Issue로 승격하고 PR·실험 증거를 연결하면 된다.
