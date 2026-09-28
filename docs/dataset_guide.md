# CMMT 데이터셋 안내

> 팀원이 “어떤 데이터셋을 왜 쓰는가?”를 빠르게 확인하기 위한 현재 기준 문서다. 원본 데이터와 checkpoint는 GitHub에 올리지 않으며, 실행 증거와 세부 protocol은 [Benchmark protocol](design/03_benchmark_protocol.md)을 따른다.

## 한눈에 보기

| 데이터셋 | 연구 역할 | 모델 선택에 사용 | 현재 상태 |
|---|---|---|---|
| MOSEv2 | Translator 학습(fit/dev), sealed in-domain final | train의 video-disjoint dev만 사용 | manifest·loader 검증 완료 |
| LVOS v2 | Translator 학습(fit/dev), 공개 GT 기반 local in-domain final | train의 video-disjoint dev만 사용 | manifest·loader 검증 완료 |
| VOST validation | primary external zero-shot | 사용하지 않음 | archive·manifest·loader·official `J/J_last` contract 완료 |
| PUMaVOS 전체 공개 archive | partial/unusual-mask complementary external stress | 사용하지 않음 | checksum·inventory·manifest·local `J/F/J&F` contract 완료 |
| M³-VOS | material phase-transition complementary external stress | 사용하지 않음 | immutable delivery inventory·manifest·loader·official evaluator contract 완료 |
| DAVIS 2017 | 현재 연구 범위 아님 | 사용하지 않음 | 과거 runtime engineering 증거로만 보존 |

## 학습과 평가의 분리

1. **학습과 개발 선택**은 MOSEv2·LVOS v2의 train split에서만 수행한다. 영상 단위로 fit/dev를 나누며, primary Translator는 같은 prefix의 Small/Base+ paired state만으로 학습한다. 미래 frame의 GT는 primary pair loss에 사용하지 않는다.
2. **in-domain final**은 설정·checkpoint·threshold를 동결한 뒤 실행한다. LVOS v2 validation은 공개 GT로 local detailed final을 계산한다. MOSEv2 validation은 이후 prediction을 Codabench에 제출하는 sealed final이다.
3. **external zero-shot**은 학습·정규화 통계·구조·loss·threshold·replay 길이 선택에 쓰지 않는다. VOST validation이 primary external이며, PUMaVOS와 M³-VOS는 서로 다른 실패 조건을 검증하는 보완적 stress benchmark다.

## 데이터셋별 상세 역할

### MOSEv2

- Small/Base+ paired-state 수집과 nonlinear Translator fit/dev의 주 데이터다.
- official validation은 future GT가 local에 공개되지 않은 sealed final 경로다. 세부 switch J&F를 local 주장으로 쓰지 않는다.

### LVOS v2

- MOSEv2와 함께 fit/dev를 구성한다.
- validation 공개 GT로 J/F/J&F, switch shock, recovery, identity continuity를 포함한 local detailed final을 계산한다.

### VOST validation

- 학습에는 넣지 않는 **primary external zero-shot** benchmark다.
- config freeze 뒤 처음 prompt만 주고, official `J/J_last`와 CMMT의 보조 조건별 분석을 분리해 보고한다.

### PUMaVOS

- 공식 train/validation/test split이 없는 공개 archive 전체를 단일 frozen external stress test로 사용한다.
- 객체별 first-nonempty GT 한 장만 prompt로 사용하고, 이후 GT는 평가에만 사용한다.
- partial/unusual-mask 조건에서 handoff가 얼마나 견고한지 본다.

### M³-VOS

- material phase transition 상황을 보는 external stress benchmark다.
- 본문 주지표는 공식 evaluator가 실제로 출력하는 `J`, `J_last`, `J_cc`다. `J_last`는 endpoint를 제외하고 temporal downsampling한 평가 frame의 마지막 25% mean이다. Boundary `F/J&F`는 void 처리·GT-copy·shard merge·morphology 민감도 integrity gate를 통과했을 때만 보조 지표로 쓴다.
- 2026-09-28 immutable HF revision inventory는 471 sequences, 202,577 RGB/GT pairs, 530 object records, core 68, `failure_count=0`으로 통과했다. 논문·project page의 479 videos/205,181 masks 수치와 다르므로 두 수치를 혼용하지 않는다. 1,590-case first-prompt loader와 official `J/J_last/J_cc` GT-copy·shard-merge smoke도 통과했다. 아직 어떤 CMMT 성능 결론에도 사용하지 않는다.

## 누수 방지 규칙

- 외부 데이터셋(VOST·PUMaVOS·M³-VOS)은 hyperparameter, architecture, loss, checkpoint, threshold, replay 길이를 고르는 데 사용하지 않는다.
- 모든 비교군은 같은 등록 객체·prompt timeline·switch 조건을 사용한다.
- 미래 GT는 평가 점수 계산에만 사용하며 handoff 입력 또는 switch 선택에 사용하지 않는다.

## 관련 문서

- [현재 연구 진행 요약](research_progress_summary.md)
- [Benchmark protocol](design/03_benchmark_protocol.md)
- [Task 03 보고서](../reports/tasks/03_benchmark/README.md)
