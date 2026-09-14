# CMMT 정식 실험 계획

> 갱신: 2026-09-10 KST
> 원칙: RunPod 비용 때문에 연구에 필요한 실험 수·모델·데이터를 축소하지 않는다.

## 전체 연구 흐름과 현재 위치

```text
[0. 환경·데이터·고정 manifest]                         완료
                │
                ▼
[1. state contract·주입·평가 파이프라인]               완료
                │  └─ 전체/GT-visible J&F와 temporal metric 완료
                ▼
[2. 강한 baseline과 rare-event 분포]                   완료
                │  └─ Direct/Reset/Last/Replay/Oracle 구현 완료
                │     rare-event 10/10 temporal metric 재집계 완료
                ▼
[3. paired Tiny↔Large state 수집·nonlinear 학습]       ◀ 현재 단계
                │
                ▼
[4. held-out 일반화·통계·양방향·반복 switch]
                │
                ▼
[5. MOSE/LVOS/YouTube-VOS hard·long-term 검증]
                │
                ▼
[6. SAM 2↔XMem/Cutie cross-architecture 검증]
                │
                ▼
[7. 논문용 ablation·시스템 비용·failure boundary 정리]
```

2026-09-10 현재 고정 rare-event 10-case subset을 완료했다. GT-visible J&F 평균은
Last-Mask 0.319945, Replay-2 0.425541, Replay-4 0.623753, Full Replay 0.483535였다.
Replay-4가 평균상 가장 강했지만 `kite-surf` 재등장에서는 Full Replay만 성공했고,
`lab-coat`에서는 반대로 Full Replay가 실패하고 Replay-4가 성공했다. 동일 결과의
원본 `davis.json`에서 switch shock·identity-loss proxy·recovery length를 GPU
추론 없이 backfill했다. 현재 실행은 video-level split을 고정하고 paired state를
수집해 nonlinear Translator를 학습하는 Phase 3이다.

## 최종 연구 질문

Source VOS model의 누적 spatial/object-centric memory를 target-compatible state로
번역하면, full history replay 없이도 target이 다음 프레임부터 identity와
segmentation 품질을 유지할 수 있는가?

성공 여부는 tensor MSE가 아니라 post-switch J&F, identity continuity,
occlusion recovery, latency, FLOPs, VRAM과 전송량으로 판정한다.

## Phase 1 — 평가 기반 완성

- DAVIS 2017 train/val의 여러 sequence, object, switch point를 고정 manifest로 만든다.
- switch point는 일반 구간뿐 아니라 occlusion 직전·중간·직후, reappearance,
  fast motion을 포함한다.
- future frame별 J, F, J&F와 switch+1/5/20 성능을 저장한다.
- 모든 실행은 config, checkpoint hash, upstream commit, seed, runtime과 산출물 경로를 남긴다.
- GT / target-native / candidate 비교 영상, slider gallery, frame별 metric 그래프를 자동 생성한다.

Temporal metric은 다음과 같이 고정한다.

- `switch+1/5/20`: switch에서 정확히 1/5/20 frame 뒤의 candidate J&F와
  target-native 대비 gap. 전체값과 GT-visible값을 분리하며 보고서의 대표값은
  GT-visible이다. 영상 끝을 넘거나 해당 offset에 visible case가 없으면 `n/a`다.
- `switch shock@N`: 처음 N개 GT-visible post-switch observation에서
  `target-native J&F - candidate J&F`의 평균.
- `recovery length`: target-native J&F가 0.5 이상인 구간에서 candidate가
  target-native보다 0.05 이내인 상태를 GT-visible observation 3개 연속 유지하기
  시작한 frame까지의 거리. 끝까지 충족하지 못하면 censored로 기록한다.
- `identity-loss proxy`: target-native J&F는 0.5 이상인데 candidate J&F가 0.1
  이하인 상태가 GT-visible observation 2개 이상 연속인 경우. 단일-object DAVIS의
  추적 상실 proxy이며, multi-object ID switch와 같은 지표로 주장하지 않는다.

## Phase 2 — 필수 baseline 완성

동일한 video/object/switch manifest에서 다음을 비교한다.

1. Target Reset
2. Last-Mask
3. Replay-1/3/5/10 및 필요 시 더 긴 k
4. Full Replay / Target-native Oracle
5. Direct Transfer
6. 작은 component-wise MLP Translator
7. attention/gating을 포함한 nonlinear Translator
8. Nonlinear Translation + Short Replay

정확도와 비용을 따로 순위화하지 않고 accuracy–latency Pareto frontier로 비교한다.

## Phase 3 — paired-state 데이터 확대와 nonlinear 학습

- 학습 split의 여러 sequence와 switch point에서 Tiny state `b_t`와 Large-native
  state `a_t`를 수집한다.
- video 단위로 train/validation/test를 분리해 frame leakage를 막는다.
- spatial memory, object pointer, presence를 component별로 정규화·번역한다.
- 작은 component-wise MLP를 첫 학습 모델로 삼고, 필요할 때 attention/gating과
  rollout/distillation loss를 순차 비교한다.
- 모델 backbone은 동결하고 Translator만 학습한다.

2026-09-14 사용자 결정에 따라 새로운 Linear/Ridge Translator 개발·학습은 연구
범위에서 제외한다. 기존 Ridge pilot은 파이프라인이 실제 state를 주입할 수 있음을
보인 역사적 sanity baseline으로만 보존하며, 제안 방법이나 향후 개발 대상으로
사용하지 않는다.

## Phase 4 — 일반화 및 통계

- 고정 DAVIS val manifest 전체에서 video/object 단위 결과를 유지한다.
- 평균뿐 아니라 분산과 video-clustered bootstrap confidence interval을 보고한다.
- switch 시점별 독립 표본 수를 과장하지 않는다.
- direction을 Tiny→Large와 Large→Tiny로 나누어 분석한다.
- repeated switch에서 drift와 refresh 필요성을 측정한다.

## Phase 5 — hard/long-term benchmark

DAVIS에서 pipeline과 baseline이 안정화되면 MOSE, LVOS, YouTube-VOS로 확장한다.
특히 long occlusion, reappearance, distractor, long absence 구간을 별도 slice로
평가한다. easy sequence의 평균 J&F만으로 memory translation의 필요성을 주장하지 않는다.

## Phase 6 — cross-architecture 검증

SAM 2 same-family 결과는 controlled baseline으로 사용한다. 최종 novelty는
SAM 2↔XMem 또는 SAM 2↔Cutie처럼 state 구조가 다른 모델 사이에서 검증한다.
shape adapter, token/slot resampling, component-specific translator와 hybrid replay를
비교한다.

## Go/No-Go gate

다음 조건을 여러 held-out 영상에서 만족하면 다음 단계로 진행한다.

- Direct Transfer보다 안정적으로 높은 post-switch J&F
- hard slice에서 Last-Mask보다 의미 있는 개선
- replay-k보다 우수한 accuracy–latency trade-off
- target-native 대비 작은 switch shock와 짧은 recovery
- identity/occlusion failure 감소

Last-Mask가 hard slice에서도 동등하거나, target 수준에 도달하려면 긴 replay가
필요하거나, translated state가 새 영상에서 일반화하지 못하면 설계를 수정한다.
이 결과도 숨기지 않고 CMMT의 failure boundary로 기록한다.

## Codex 사용량 보호 정책

- Codex가 제공하는 5시간·주간 사용률을 큰 단계 전후에 확인한다.
- 서비스는 정확한 잔여 token 수를 제공하지 않으므로 percentage를 운영 지표로 쓴다.
- 5시간 창 사용률이 80%에 도달하면 새 장기 작업을 시작하지 않고 현재 실행의
  결과·로그·재현 명령·Git 상태를 먼저 저장한다.
- 90%에 도달하면 안전한 종료와 결과 보존만 수행하고, 다음 reset 뒤 이어간다.
- 주간 사용률이 80%에 도달해도 같은 원칙을 적용한다.
- 사용량 부족 때문에 표본 수를 몰래 줄이거나 결론을 조기에 확정하지 않는다.
- 대량 이미지와 JSON은 프로그램으로 집계하고, 모델에는 요약 통계·대표 실패
  사례만 읽혀 불필요한 token 소비를 줄인다.

## RunPod 운영 원칙

- GPU 비용은 실험 규모를 결정하는 기준으로 삼지 않는다.
- CPU 전처리·시각화·보고서는 GPU 밖에서 수행해 자원 낭비를 피한다.
- GPU는 checkpoint inference, paired-state 수집, Translator 학습에 집중한다.
- 예산과 무관하게 실행 전 smoke test, deterministic seed, artifact validation을 유지한다.
