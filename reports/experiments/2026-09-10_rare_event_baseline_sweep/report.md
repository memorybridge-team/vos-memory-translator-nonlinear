# DAVIS rare-event baseline sweep 보고서

> 실행 완료: 2026-09-10 14:26 KST  
> 범위: DAVIS 2017 val의 3개 영상, 10개 object/switch 사례  
> 주의: 고정 rare-event slice의 부분 영상 결과이며 전체 DAVIS benchmark가 아니다.

## 무엇을 확인한 실험인가

SAM 2.1 Tiny에서 Large로 전환할 때 Translator가 넘어야 할 기준을 먼저 고정했다.
GT는 사례 선택과 평가에만 사용했고 모델 입력에는 넣지 않았다. 각 사례에서 Direct
Copy, 빈 object slot으로 시작하는 Target Reset, Last-Mask, Replay-1/2/4, 전체 과거를
Large가 처리한 Full Replay를 같은 switch 이후 구간에서 비교했다.

- source: SAM 2.1 Hiera Tiny
- target: SAM 2.1 Hiera Large
- seed: 7
- 선택 manifest SHA-256:
  `e6492a58a953790d5a5adcae89ad40cf54f2f3c83c1e07fc32c8c347ce737e93`
- 완료 상태: 10/10, 실패 0
- sanity check: 모든 사례에서 Last-Mask와 Replay-1이 frame별로 동일
- 무결성: 모든 case cache SHA-256 일치, JSON/Markdown의 NaN/Infinity 없음

## 10사례 평균

사례마다 GT에 대상이 실제로 존재하는 후속 frame만 평균한 `GT-visible J&F`를 핵심
지표로 사용한다. `Past backbone`은 switch까지 Large backbone이 과거 frame을 처리한
평균 횟수다.

| 방법 | 전체 J&F | GT-visible J&F | Past backbone | 평균 실행시간(s) |
|---|---:|---:|---:|---:|
| Direct Copy | 0.278243 | 0.000000 | 0.0 | 24.893 |
| Target Reset | 0.274672 | 0.000000 | 1.0 | 24.079 |
| Last-Mask | 0.468188 | 0.319945 | 1.0 | 21.482 |
| Replay-1 | 0.468188 | 0.319945 | 1.0 | 21.729 |
| Replay-2 | 0.538350 | 0.425541 | 2.0 | 22.744 |
| Replay-4 | **0.700364** | **0.623753** | 4.0 | 23.432 |
| Full Replay / Large-native | 0.603838 | 0.483535 | 30.5 | 33.545 |

## 사례별 GT-visible J&F

| 사례 | 이벤트 | Last | Replay-2 | Replay-4 | Full Replay |
|---|---|---:|---:|---:|---:|
| india obj1 switch38 | strong area drop | 0.897178 | 0.894166 | 0.892450 | 0.892390 |
| india obj1 switch40 | strong area growth | 0.906723 | 0.907358 | 0.908753 | 0.906710 |
| india obj2 switch38 | strong area growth | 0.515653 | 0.415948 | 0.793560 | 0.436338 |
| india obj2 switch51 | full occlusion entry | 0.000000 | 0.000000 | 0.002376 | 0.000000 |
| india obj3 switch34 | full occlusion entry | 0.437682 | 0.832665 | 0.835897 | 0.868302 |
| india obj3 switch35 | reappearance | 0.000000 | 0.437682 | 0.834992 | 0.868302 |
| kite-surf obj2 switch15 | full occlusion entry | 0.000000 | 0.000000 | 0.406456 | 0.406723 |
| kite-surf obj2 switch20 | reappearance | 0.000000 | 0.000000 | 0.000000 | 0.406723 |
| lab-coat obj1 switch11 | strong area drop | 0.415952 | 0.757557 | 0.758138 | 0.024175 |
| lab-coat obj1 switch13 | strong area growth | 0.026257 | 0.010034 | 0.804905 | 0.025686 |

## 해석

1. Direct Copy와 Target Reset은 10사례 모두 GT-visible J&F 0으로 실패했다. 단순
   tensor 복사나 빈 상태 시작은 learned handoff의 유효한 대체물이 아니다.
2. Last-Mask는 평균 0.319945로 무시할 수 없는 강한 baseline이지만, occlusion과
   reappearance에서는 자주 0이 된다.
3. Replay-4는 평균 정확도와 비용의 가장 강한 기준이었다. 평균 4번의 과거 backbone
   호출로 Full Replay의 30.5번보다 적으면서 visible J&F는 더 높았다.
4. Full Replay는 고정된 상한이 아니다. SAM 2 tracking은 autoregressive memory에
   영향을 받으므로 오래된 memory가 drift를 만들 수 있다. 특히 lab-coat switch13은
   Replay-4 0.804905, Full Replay 0.025686이었다.
5. 반대로 kite-surf switch20에서는 Replay-4가 0이고 Full Replay가 0.406723이다.
   따라서 최근 정보만 보존하는 설계도 충분하지 않으며 long-term identity 단서가
   필요한 사례가 존재한다.
6. india obj2 switch51은 Full Replay도 0이다. 이 사례는 Translator가 target 자체의
   실패를 넘어설 수 있는지 보는 별도 연구 대상이지, 초기 Translator의 정상 성공
   기준으로 사용하면 안 된다.

## 다음 gate

- switch+1/5/20, switch shock, single-object identity-loss proxy, recovery length의
  정의와 구현은 2026-09-14에 고정했다. 기존 10-case suite의 per-method
  `davis.json`에서 GPU 재추론 없이 backfill해 이 표를 보강한다.
- 영상 단위 train/validation/test split을 고정하고 Tiny/Large paired state를 수집한다.
- component별 작은 MLP를 시작으로 nonlinear Translator를 학습한다. 새로운
  Linear/Ridge 개발은 제외하고 기존 Ridge pilot만 역사적 baseline으로 보존한다.
- Translator를 Direct뿐 아니라 Last-Mask, Replay-2/4, Full Replay의 accuracy-cost
  Pareto frontier와 비교한다.
- `translated state + short replay`를 정식 ablation으로 유지한다.
