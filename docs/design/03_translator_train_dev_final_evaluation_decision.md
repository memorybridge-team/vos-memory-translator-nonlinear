# Translator train–development–final evaluation 결정 문서

> 결정일: 2026-09-25 KST  
> 적용 범위: SAM 2.1 Small → Base+ nonlinear memory translator  
> 관련 Task: 03 Benchmark, 07 Paired-state, 08 Baselines, 09 Nonlinear Training, 13 Full Evaluation  
> 문서 목적: 이 대화가 없어져도 동일한 연구 설계로 GitHub 문서·Issue·Project Board를 수정할 수 있게 하는 독립적인 기준 문서

## 1. 한 문장 결정

Translator는 **MOSEv2·LVOS v2 train의 video-disjoint fit/dev**에서만 학습·선택하고,
설정과 checkpoint를 완전히 동결한 뒤 **LVOS v2 validation은 공개 GT로 상세 평가**,
**MOSEv2 validation은 Codabench의 숨은 GT로 공식 aggregate 평가**,
**VOST validation은 primary translator-level external zero-shot 평가**,
**PUMaVOS 전체 24개 영상은 secondary external zero-shot stress test**에 사용한다.

MOSEv2 validation에서는 숨은 후속 GT가 필요한 CMMT 자체 switch/recovery/identity 수치를
로컬 결과처럼 주장하지 않는다. 대신 GT가 필요 없는 latency·VRAM·handoff bytes·replay량은
동일 실행에서 보고한다.

## 1.1 KV Cache Transfer 논문과의 대응·차이

[`Cross-Model KV Cache Transfer in LLM Families`](https://arxiv.org/html/2608.03893)은
500개 FineWeb-Edu sequence에서 source/target KV pair로 ridge mapper를 맞췄으며 downstream
task label을 mapper fitting에 사용하지 않았다. 이 점은 CMMT의 state-only fit을 직접 지지한다.

다만 그 논문이 엄격한 train/validation/test 분리를 사용했다고 요약하면 부정확하다. Source-layer
수 `k`는 ARC-C·HellaSwag·WinoGrande·MMLU accuracy 평균을 최대화하도록 sweep했고, 이 네
benchmark는 최종 결과표에도 포함했다. GSM8K·CoQA·latency만 selection에서 held out했다.
CMMT는 이보다 보수적으로 다음을 분리한다.

```text
future-GT-free state fit
        → video-disjoint dense-GT dev selection
        → config/checkpoint freeze
        → sealed in-domain + frozen external evaluation
```

VOS는 객체 prompt와 시간적 identity가 필요하므로 label-free LLM text와 완전히 같지 않다.
Future dense GT는 없어도 되지만 객체를 등록할 first-nonempty prompt와 object ID는 반드시 필요하다.

## 2. 용어

### 2.1 Fit

Gradient로 Translator parameter를 업데이트하는 학습 구간이다.

```text
입력: Source state b_T
정답: 동일 prefix를 Target이 직접 처리해 만든 native state a_T
출력: translated state â_T = T(b_T)
```

### 2.2 Development(dev)

학습에는 직접 넣지 않은 영상으로 구조·loss·hyperparameter·checkpoint를 고르는 구간이다.
Fit과 dev는 **video-disjoint**여야 한다. 즉 한 영상의 frame·object·prompt·switch·paired state는
모두 한 split에만 속한다.

### 2.3 Config/checkpoint freeze

최종 평가 전에 코드·설정·checkpoint·평가 규칙을 고정하는 절차다. Freeze 뒤 final/external
결과를 보고 설정을 바꾸면 기존 결과와 같은 confirmatory experiment로 취급하지 않는다.

### 2.4 Sealed evaluation

정답지를 서버가 숨겨 둔 시험이다. 연구자는 영상과 최초 prompt로 예측 mask를 만들고,
Codabench에 답안을 제출한다. 서버가 비공개 GT로 채점한 공식 점수만 받는다.

### 2.5 Translator-level external zero-shot

Frozen SAM 2 backbone이 해당 데이터셋을 전혀 본 적 없다는 뜻이 아니다. Translator의 gradient,
normalization/statistics, early stopping, threshold, 구조·loss·checkpoint 선택에 외부 데이터셋을
사용하지 않았다는 뜻이다. 표준 VOS의 첫-frame GT prompt와 최종 채점용 GT는 허용한다.

## 3. 데이터셋 GT 가용성과 역할

| 데이터셋 split | 공개 입력·GT 상태 | 연구 역할 | 허용되는 평가 |
|---|---|---|---|
| MOSEv2 train | 전체 frame GT 공개 | fit/dev | state loss, 선택적 downstream loss, J/F/J&F, switch/recovery/identity 분석 |
| MOSEv2 validation | 첫 frame GT만 공개, 후속 GT 비공개 | sealed in-domain final | Codabench가 반환한 공식 점수, GT 불필요 시스템 지표 |
| MOSEv2 test | 대회 기간에 제한적으로 제공되고 고정 공개 test가 아님 | 현재 core protocol에서 제외; 접근 가능할 때 optional challenge result | 당시 공식 서버가 허용하는 지표 |
| LVOS v2 train | 전체 GT 공개 | fit/dev | state loss, 선택적 downstream loss, J/F/J&F, switch/recovery/identity 분석 |
| LVOS v2 validation | 전체 GT 공개 | local in-domain final | 공식 J/F/J&F와 CMMT 세부 지표 |
| LVOS v2 test | 첫 frame GT만 공개, 후속 GT 비공개; V2 server 가용성 없음 | 현재 core protocol에서 제외 | 채점 서버가 생기기 전에는 정량 최종 점수 불가 |
| VOST train | 전체 GT 공개 | main 학습·선택에는 미사용 | 선택 사항: freeze 후 supplementary zero-shot stress test 또는 별도 adaptation upper-bound |
| VOST validation | 전체 GT 공개 | primary external zero-shot | 공식 J/J_last와 CMMT 세부 지표 |
| VOST test | 현재 공개 archive에는 sequence 이름만 있고 영상·GT 없음 | 공식 영상·prompt·server가 실제 제공될 때만 optional sealed test | 공식 서버가 제공하는 지표 |
| PUMaVOS 전체 | 24 videos, 21,187 frames의 dense GT 공개; 공식 train/val/test 구분 없음 | secondary external zero-shot stress test | 사전 고정 first-nonempty-mask protocol의 J/F/J&F, switch·recovery·system 지표 |

공식 근거:

- MOSEv2: <https://mose.video/>, <https://www.codabench.org/competitions/10062/>
- LVOS v2: <https://lingyihongfd.github.io/lvos.github.io/dataset.html>
- VOST: <https://www.vostdataset.org/>, <https://github.com/TRI-ML/VOST/tree/main/evaluation>
- PUMaVOS: <https://github.com/mbzuai-metaverse/XMem2>, <https://arxiv.org/abs/2307.15958>

## 4. 확정 실행 순서

### 4.1 Train-fit

MOSEv2 train-fit과 LVOS v2 train-fit만 gradient update에 사용한다.

기본 supervised pair는 `(b_T, a_T)`다. `a_T`는 dataset GT가 아니라 frozen Target이 같은
과거 frame·prompt timeline을 직접 처리해 만든 Target-native state다.

주 방법의 첫 단계 학습은 component별 state reconstruction을 사용한다. 이 단계에서 `T+1`
이후 dataset GT mask는 loss 입력으로 사용하지 않는다.

```text
L_state = L_maskmem_features + λ_ptr L_obj_ptr
```

비교할 loss 후보:

- component별 normalized MSE
- cosine loss
- MSE + cosine
- mask-memory와 object-pointer의 별도 loss weight

`T+1` 이후 전체 dataset GT는 state-only loss에 필수는 아니다. 다만 VOS 객체 등록에는
첫 prompt mask·point·box 또는 검증된 prompt timeline이 필요하다.

주 방법은 state-only 학습으로 고정하고, downstream-aware 학습은 별도 ablation으로 비교한다.

```text
L_total = λ_state L_state + λ_rollout L_rollout + λ_identity L_identity
```

- Target-native future mask/logit distillation은 dataset future GT 없이 수행할 수 있다.
- Dataset GT mask를 이용한 supervised rollout loss는 train-fit의 GT만 사용한다.
- `state-only`, `state + Target-native distillation`, `state + supervised rollout`을 서로 다른
  학습 행으로 분리해 GT 의존성과 효과의 원인을 보고한다.
- Primary paired-state switch는 temporal quantile처럼 GT가 필요 없는 규칙으로 고른다.
  GT로 찾은 occlusion/reappearance event switch는 training sample 선택에 쓰지 않고 dev/final
  condition analysis에만 사용한다.

### 4.2 Train-dev

MOSEv2 train-dev와 LVOS v2 train-dev는 fit과 영상이 겹치지 않는다. 다음 선택은 dev에서만
끝낸다.

- Residual MLP / Gated Residual MLP / 사전 정의한 nonlinear 후보
- depth, hidden width, learning rate, weight decay
- state component별 normalization과 loss weight
- rollout loss 사용 여부와 rollout horizon `H`
- threshold, early stopping, checkpoint
- Task 08 baseline의 Replay-4/8/16 중 사전 정의된 선택 규칙

선택 우선순위:

1. 사전 고정한 latency·bytes·parameter budget을 만족한다.
2. post-switch J&F 또는 Target-native 대비 J&F gap/retention을 우선한다.
3. occlusion·absence·reappearance에서 switch shock과 recovery를 확인한다.
4. 성능 차이가 실질적으로 작으면 더 작고 빠른 방법을 선택한다.

State MSE/cosine은 진단 지표이며 최종 model selection의 유일한 기준으로 사용하지 않는다.
여러 `(object, switch)`가 같은 영상에서 나와도 서로 독립된 영상 표본처럼 세지 않는다.

### 4.3 Freeze

Final/external 실행 전에 다음을 기록하고 변경을 막는다.

- Git commit 및 dirty-worktree 여부
- SAM 2 upstream SHA
- Small/Base+ checkpoint SHA-256
- Translator checkpoint SHA-256
- config와 모든 hyperparameter
- dataset·split manifest checksum
- evaluator commit/version
- prompt·correction·switch 규칙
- random seed
- baseline별 입력·fallback·비용 규칙

MOSE/LVOS dev 결과만 freeze 결정에 사용할 수 있다. MOSE official validation, LVOS official
validation, VOST validation의 결과를 보고 freeze 내용을 바꾸지 않는다.

### 4.4 LVOS v2 validation: local in-domain final

전체 GT가 공개되므로 같은 prediction에서 다음을 계산한다.

- 공식 J, F, J&F
- Target-native 대비 J&F gap/retention
- switch 이후 +1/+5/+20 frame 또는 관측의 J&F
- switch shock
- visible/absent false-positive 분리
- disappearance/reappearance recovery와 no-recovery rate
- 구현된 판정 규칙이 있을 때 identity break/ID switch
- latency, replay frame/call, VRAM, handoff bytes, Translator size

결과를 본 뒤 설정을 수정하면 이 실행은 confirmatory final이 아니라 exploratory run으로
재분류하고 protocol version을 올린다.

### 4.5 MOSEv2 validation: Codabench sealed in-domain final

MOSEv2 validation은 첫 frame GT prompt만 로컬 입력으로 사용하고 후속 prediction PNG를
Codabench에 제출한다.

- 공식 표에는 Codabench가 실제 반환한 지표만 넣는다.
- 가능한 경우 scoring/format smoke를 먼저 하고, freeze된 checkpoint의 **한 번의 유효한
  scored submission**을 final로 지정한다.
- ZIP 구조 오류·전송 실패처럼 점수가 생성되지 않은 기술적 실패는 scored submission으로
  세지 않되 로그를 남긴다.
- 서버 점수를 보고 threshold·checkpoint·구조를 다시 고르지 않는다.

후속 GT가 없으므로 다음을 자체 official 수치처럼 주장하지 않는다.

- local switch +1/+5/+20 J&F
- GT 기반 recovery length
- GT 기반 visible/absent/reappearance slice
- GT 기반 identity break

Target-native와 Translator prediction의 agreement를 계산할 수는 있지만 이는 GT 성능이 아닌
`oracle-agreement diagnostic`으로만 표기한다.

다음 시스템 지표는 GT 없이 함께 보고할 수 있다.

- handoff/continuation latency
- replay frame·backbone call 수
- peak VRAM·host RAM
- handoff bytes
- Translator parameter·checkpoint size

MOSEv2 공식 페이지에 disappearance/reappearance 관련 score가 설명되어 있어도, 실제 최종
보고에는 해당 Codabench phase가 반환한 필드만 사용한다.

### 4.6 VOST validation: primary external zero-shot

MOSE/LVOS에서 freeze한 동일 checkpoint와 설정을 변경 없이 적용한다.

- 공식 J와 official evaluator의 `J_last`
- 25/50/75% temporal switch 결과; primary switch는 사전 고정한 50%
- transformation 전후 성능
- switch shock과 recovery
- identity continuity는 구현된 판정 규칙이 있을 때만 보고
- latency, VRAM, replay량, bytes

VOST validation의 결과로 구조·loss·checkpoint·threshold·normalization을 바꾸지 않는다.
변경하면 변경 후 결과는 zero-shot confirmatory result가 아니라 exploratory result다.

### 4.7 VOST train의 선택적 사용

VOST train에는 validation과 같은 형식의 공개 GT가 있으므로, main Translator가 VOST에 전혀
노출되지 않은 상태에서 **supplementary external zero-shot stress test**로 사용할 수 있다.
이 경우 main text에는 다음처럼 간단히 밝힌다.

> Translator는 MOSEv2/LVOS v2로만 학습·선택했고, VOST train+validation의 공개 GT를
> 외부 zero-shot 평가에만 사용했다.

기존 연구와 같은 공식 protocol 비교를 위해 VOST validation 70개 점수는 별도로 산출한다.
VOST train과 validation을 합친 642개 결과는 대규모 supplementary 결과이며, 기존 논문의
validation-only 숫자와 같은 열에 직접 놓지 않는다.

VOST train 결과를 보고 모델을 수정하면 이후 VOST 결과는 엄격한 translator-level zero-shot이
아니다. VOST train fine-tuning을 수행할 경우 `adaptation upper-bound`로 완전히 분리한다.

### 4.8 PUMaVOS: secondary external zero-shot

PUMaVOS는 공식 train/validation/test split이 없는 benchmark-only 데이터셋이다. 따라서
`validation set`이라고 부르지 않고, MOSE/LVOS dev에서 모든 설정을 동결한 뒤 한 번 실행하는
secondary external zero-shot stress test로 사용한다.

- 24개 영상 전체와 21,187개 dense frame GT를 평가에 사용한다.
- 객체별 첫 non-empty GT mask 한 장만 conditioning prompt로 사용한다. 미래 frame GT는
  입력·correction·switch 선택에 쓰지 않고 채점에만 사용한다.
- 객체가 첫 video frame에 없으면 첫 non-empty frame에서 해당 객체의 평가 구간을 시작하고
  원래 frame offset을 manifest에 기록한다.
- 공식 split이 없으므로 내부 dev subset을 만들거나 PUMaVOS 결과로 threshold·checkpoint를
  바꾸지 않는다.
- J/F/J&F, switch +1/+5/+20, shock, recovery, false positive, latency·VRAM·bytes를 보고한다.
- 영상 수가 24개로 작으므로 per-video 결과와 video-clustered bootstrap confidence interval을
  함께 보고하고, VOST나 in-domain 점수와 단순 평균하지 않는다.

## 5. 지표 coverage matrix

| 지표 | Train-dev | LVOS val | MOSEv2 val | VOST val | PUMaVOS |
|---|---:|---:|---:|---:|---:|
| State MSE/cosine | 가능 | 가능 | 가능 | 가능 | 가능 |
| 공식 J/F/J&F | 가능 | 가능 | Codabench 반환값 | VOST 공식 지표는 J/J_last | 고정 local J/F/J&F |
| Switch +1/+5/+20 GT 성능 | 가능 | 가능 | 불가 | 가능 | 가능 |
| GT 기반 recovery | 가능 | 가능 | 불가 | 가능 | 가능 |
| GT 기반 visible/absent/reappearance | 가능 | 가능 | 불가 | 가능 | 가능 |
| GT 기반 identity break | 판정 구현 시 가능 | 판정 구현 시 가능 | 불가 | 판정 구현 시 가능 | 판정 구현 시 가능 |
| Latency/VRAM/replay/bytes | 가능 | 가능 | 가능 | 가능 | 가능 |

모든 데이터셋이 모든 지표를 제공할 필요는 없다. 다만 연구의 각 핵심 주장은 적어도 하나의
model-selection에 쓰지 않은 held-out 데이터에서 검증되어야 한다. MOSEv2 validation은 숨은
GT 기반 공식 aggregate generalization, LVOS v2 validation은 long-term 상세 분석, VOST
validation은 transformation external zero-shot, PUMaVOS는 partial/unusual-mask external stress를 담당한다.

## 6. Final에서 동일하게 실행할 방법

가능한 데이터셋에서는 같은 manifest·prompt·switch·evaluator로 다음을 실행한다.

- Source-only
- Base+-native / Full Replay
- Direct State Copy
- Moment-Matched Copy
- Original-Prompt(s) Only
- Last-Visible Source Mask
- Original + Last-Visible
- Original-Prompt(s)+Replay-4/8/16
- 최종 Nonlinear Translator
- 필요성이 확인된 경우에만 Translator + Short Replay

MOSEv2 Codabench 제출 비용·횟수가 제한되면 dev와 LVOS validation에서 방법을 줄이고,
사전에 정한 최소 핵심군만 서버에 제출할 수 있다. 어떤 방법을 제출할지는 MOSE 서버 점수를
보기 전에 결정한다.

## 7. 누수 방지 규칙

- Fit/dev는 video-disjoint다.
- 미래 GT를 runtime 입력, replay frame 선택, Last-Visible 선택에 사용하지 않는다.
- Moment-Matched 통계는 MOSE/LVOS fit paired state에서만 계산한다.
- Official validation과 VOST의 Target-native state·GT-derived 통계를 fit/selection에 쓰지 않는다.
- Public GT가 있어도 final prediction을 모두 만든 뒤 채점한다.
- External 결과를 보고 설정을 바꾸면 날짜·이유·영향받는 run을 decision log에 남기고
  confirmatory와 exploratory 결과를 분리한다.
- 원자료는 `(dataset, video, object, switch)` 단위로 보존하고 confidence interval은
  video-clustered bootstrap으로 계산한다.

## 8. 기존 문서·GitHub 동기화 지침

이 문서를 적용할 때 다음을 한 PR에서 일관되게 수정한다.

### 문서

- `docs/design/03_benchmark_protocol.md`
  - MOSEv2 validation을 first-frame-only + Codabench sealed evaluation으로 명시
  - local switch/recovery/identity claim 금지
  - LVOS val의 local detailed final 역할 명시
  - VOST val의 primary external zero-shot 역할과 optional VOST-train stress test 명시
  - PUMaVOS 전체 24개 영상의 secondary external zero-shot·first-nonempty prompt 계약 명시
- `docs/experimental_plan.md`
  - train-fit → train-dev → freeze → LVOS local final → MOSE Codabench → VOST/PUMaVOS zero-shot 순서 반영
- `docs/research_progress_summary.md`, repository `README.md`
  - 최신 데이터 역할과 현재 gate를 짧게 동기화
- `reports/tasks/03_benchmark/README.md`와 Task 03 `FINAL_REPORT.md`
  - dataset GT availability·evaluator contract·제한 기록

### GitHub Project/Issue

- **Task 03:** dataset GT availability, split 역할, metric coverage, 누수 금지 규칙을 canonical
  protocol 완료 기준에 반영
- **Task 07:** MOSE/LVOS train-fit/dev에서만 paired state를 수집하고 final/external state를
  학습 shard에 넣지 않음
- **Task 08:** LVOS local evaluator, MOSE prediction packaging/Codabench contract, VOST official
  J/J_last evaluator와 공통 system profiler 구현
- **Task 09:** state-only와 downstream-aware nonlinear 학습, dev-only model selection, freeze artifact
  생성
- **Task 13:** LVOS detailed final, MOSE official aggregate, VOST primary external zero-shot,
  PUMaVOS secondary external stress, metric coverage와 missing metric 사유를 분리 보고

Project Board는 상세 본문을 복제하지 않고 이 문서·canonical Issue·관련 PR/보고서 링크와
상태·blocker만 표시한다.

## 9. 후속 대화에 바로 입력할 작업 지시문

아래 문장을 이 파일과 함께 후속 대화에 입력하면 된다.

```text
첨부한 `docs/design/03_translator_train_dev_final_evaluation_decision.md`를 이번 변경의
source of truth로 사용해줘. 저장소의 관련 문서, Task 03/07/08/09/13 canonical Issue와
GitHub Project 설명을 전수 대조하고, 충돌하는 이전 표현을 수정해줘. 특히 MOSEv2
validation은 first-frame GT만 공개된 Codabench sealed evaluation이며 local GT 기반
switch/recovery/identity 수치를 주장하면 안 된다. LVOS v2 validation은 공개 GT 기반
상세 in-domain final, VOST validation은 freeze 후 primary external zero-shot, PUMaVOS는
split 없는 secondary external zero-shot stress test다. Primary Translator는 future dataset GT
없이 paired state를 맞추고, dense GT는 dev selection과 final/external scoring부터 사용한다.
수정 전 현재 branch와 미커밋 변경을 보존하고, 변경 사항·검증 결과·남은 미확정 항목을
보고해줘. Issue/Project 상태를 Done으로 바꾸지는 말고 완료 기준과 실제 증거가 일치하는지
먼저 확인해줘.
```

## 10. 아직 자동으로 확정하지 않은 항목

- 현재 MOSE/LVOS train의 고정 split은 seed 7 기반 80/20 fit/dev다. 80/10/10
  fit/dev/internal-holdout으로 변경할지는 별도 결정과 manifest version 변경이 필요하다.
- MOSEv2 Codabench가 실제로 반환하는 세부 필드는 freeze 전 format/scoring contract에서
  확인한다. 공식 페이지에 적힌 지표를 서버 출력 확인 없이 결과 열로 약속하지 않는다.
- VOST train+validation 642개 supplementary zero-shot을 실제 주 결과에 넣을지는 계산 예산과
  논문 지면을 보고 freeze 전에 결정한다. VOST validation 70개 공식 결과는 필수다.
- Identity break는 판정 구현과 검증이 끝난 데이터셋에서만 보고한다.
- PUMaVOS는 공식 split이 없으므로 전체 24개를 단일 frozen external benchmark로 사용하며,
  loader·object-ID·first-nonempty prompt·metric contract 검증이 Task 03의 남은 gate다.

이 미확정 항목은 최종 결과를 본 뒤 유리하게 고르지 않고, 해당 데이터를 실행하기 전에
decision log와 protocol version으로 확정한다.
