# Task 03 — Benchmark protocol v1.1

> 상태: **IN PROGRESS** — v1.0은 2026-09-23 완료, v1.1의 VOST onboarding gate가 남음
> 범위: SAM 2.1 Small → Base+ nonlinear state handoff  
> 목적: 결과를 보기 전에 dataset role, case taxonomy, baseline 입력, metric과 통계 단위를 고정한다.
> 변경일: 2026-09-24 — in-domain held-out와 external cross-dataset zero-shot을 분리했다.
> 협업 추적: [GitHub Issue #4](https://github.com/memorybridge-team/vos-memory-translator-nonlinear/issues/4)

## 1. Task 경계

- Task 03은 **무엇을 어떤 공통 조건으로 평가할지**를 정한다.
- Task 06은 state export·self-injection·target injection의 구현 정확성을 검증한다.
- Task 07은 이 문서의 split/manifest 규칙으로 paired state를 수집한다.
- Task 08은 이 문서에 정의된 evaluator와 baseline을 구현한다.
- Task 13은 고정한 protocol로 전체 비교와 조건별 분석을 실행한다.

따라서 same-checkpoint export→inject는 Task 06의 정확성 검사이며 경쟁 baseline이 아니다.

## 2. 데이터 역할과 split 정책

실제 운영은 다음 네 역할을 섞지 않는다.

1. **Translator fit:** paired state와 loss gradient를 만드는 데이터
2. **In-domain development:** architecture, loss, epoch, threshold, replay-k와 checkpoint를 선택하는 데이터
3. **Sealed in-domain final evaluation:** 선택이 끝난 뒤 한 번 여는 동일 분포의 공식 validation
4. **External frozen benchmark:** translator가 학습·선택·통계 추정에 전혀 쓰지 않은 cross-dataset 평가

| 데이터셋 | 고정 배포·규모 | 연구에서 맡는 역할 | split 사용 규칙 |
|---|---|---|---|
| MOSEv2 | 2025 공개본, train 3,666 / val 433 / test 614 videos | 주 fit/dev 및 sealed in-domain final; 복잡 장면·재등장·distractor | 공식 train의 video-disjoint 80/20 fit/dev만 학습·선택에 쓴다. 공식 valid는 checkpoint 동결 후 최종 평가에만 사용한다. 311 compatibility set은 주 결과에 섞지 않는다. |
| LVOS v2 | 2024 공개본(v2), train 420 / val 140 / test 160 videos | 주 fit/dev 및 sealed in-domain final; 장기 부재·재등장 | 공식 train의 video-disjoint 80/20 fit/dev만 학습·선택에 쓴다. 공식 val은 checkpoint 동결 후 최종 평가에만 사용한다. |
| DAVIS 2017 | 공식 2017 multi-object semi-supervised VOS, train 60 / val 30 videos | 보조 external cross-dataset 평가와 runtime regression | translator fit/dev에서 제외한다. val의 `walking`, `bike-packing`, `india`를 Task 06 개발에 사용했으므로 `engineering-seen external`로 표시하고 untouched zero-shot이라고 부르지 않는다. |
| VOST | 713 videos; train 572 / val 70 / test 71, 51 transformation types, 5 FPS | 주 external cross-dataset zero-shot; 극단적 appearance/identity transformation | main translator는 VOST train/val을 전혀 보지 않는다. val은 config 동결 후 한 번 평가하고, 가능하면 official test server를 최종 외부 평가로 사용한다. VOST-train fine-tuning은 별도 adaptation upper-bound ablation이다. |

공식 자료:

- DAVIS: <https://davischallenge.org/> — 로컬 재배포 자료에는 CC BY-NC 4.0 표기를 유지한다.
- MOSEv2: <https://arxiv.org/abs/2508.05630>, <https://github.com/henghuiding/MOSE-api>
- LVOS v2: <https://arxiv.org/abs/2404.19326>, <https://github.com/LingyiHongfd/LVOS>
- VOST: <https://arxiv.org/abs/2212.06200>, <https://www.vostdataset.org/>, <https://github.com/TRI-ML/VOST/tree/main/evaluation>

GitHub 저장소의 코드 license와 dataset 자체의 이용조건을 같은 것으로 간주하지 않는다.
다운로드 전 각 배포 페이지의 dataset terms를 별도 기록하고, 원본 RGB/GT/checkpoint는
Git에 넣지 않는다. 실제 사용 snapshot에는 download source, archive/file checksum,
추출 날짜와 split manifest checksum을 남긴다.

고정 이용조건은 다음과 같다. DAVIS 재배포는 CC BY-NC 4.0, MOSEv2는
CC BY-NC-SA 4.0 및 비상업 연구 용도, LVOS annotation은 CC BY 4.0이고 원본
영상 데이터는 비상업 연구 용도다. LVOS evaluation toolkit의 BSD-3-Clause는
평가 코드의 license이며 dataset license로 확장하지 않는다. VOST는
CC BY-NC-SA 4.0이다.

### 2.1 Zero-shot의 정확한 의미와 금지 사항

본 연구에서 zero-shot은 **frozen SAM 2 backbone의 pretraining provenance 전체가 아니라,
translator-level cross-dataset zero-shot transfer**를 뜻한다. DAVIS/VOST에서는 다음을
금지한다.

- gradient, early stopping, architecture/loss/checkpoint 선택
- threshold, replay-k, switch rule 또는 normalization 통계 선택
- Moment-Matched Copy의 target moments 계산
- 실패 사례를 본 뒤 외부 benchmark에 유리하게 protocol 수정

표준 semi-supervised VOS의 첫-frame GT mask prompt와 채점용 GT는 허용한다. 외부 결과를
본 뒤 설정을 바꾸면 새 설정의 외부 결과는 zero-shot confirmatory result가 아니라
exploratory result로 구분한다.

### 2.2 결과표의 고정 구조

주 결과는 두 영역으로 나눈다.

- **In-domain held-out:** MOSEv2 valid, LVOS v2 val
- **External cross-dataset zero-shot:** VOST val/test, DAVIS val(`engineering-seen` 표기)

학습 노출 matrix의 행은 `MOSE-only`, `LVOS-only`, `MOSE+LVOS`이고, 열은
MOSE valid, LVOS val, DAVIS val, VOST val이다. `MOSE+LVOS+VOST-train`은 필요한 경우에만
별도 adaptation upper-bound 행으로 보고하며 zero-shot 행과 평균내지 않는다.

## 3. 공통 case manifest

평가 단위는 다음 key를 가진다.

```text
(dataset, release, official_split, video_id, object_id, switch_frame)
```

각 row에는 최소한 다음을 기록한다.

- 원본 frame 수·해상도·가능한 경우 FPS/timestamp
- 객체별 최초 prompt frame과 실제 prompt 종류
- switch 이전 실제 correction timeline
- switch frame과 선택 규칙(`uniform`, `event-relative`, `fixed-smoke`)
- switch 시점의 GT visibility는 **평가용 label**로만 저장
- switch 이후 평가 frame과 annotation availability
- difficulty tag의 출처(`official attribute`, `GT-derived`, `manual-reviewed`)
- code commit, SAM 2 upstream SHA, checkpoint SHA-256, dataset checksum, seed

Source와 모든 Target 방법은 같은 manifest와 같은 prompt/correction timeline을 사용한다.
미래 GT는 metric·사후 stratification에만 쓰며 runtime 입력, baseline 선택, translator
입력 또는 replay frame 선택에는 쓰지 않는다.

Dataset manifest가 제공하는 실제 interaction은 객체별 최초 mask prompt까지다.
공개 데이터셋에 존재하지 않는 사용자 correction을 GT에서 임의로 만들어 주 비교군에
입력하지 않는다. Correction 실험은 별도의 interaction manifest에
`(frame_index, object_id, prompt_type, mask_reference)`를 명시하고, 동일 interaction을
모든 방법에 제공한다. correction이 없는 표준 benchmark case의 timeline은 빈 목록이다.

## 4. Difficulty taxonomy

한 case는 여러 tag를 가질 수 있다. 전체 평균과 tag별 결과를 모두 보고한다.

| 축 | 고정 tag/값 | 판정 원칙 |
|---|---|---|
| Switch visibility | `visible`, `partial/occluded`, `absent` | switch frame의 GT로 사후 분류; 방법 입력에는 미사용 |
| Post-switch event | `continuous-visible`, `disappearance`, `reappearance`, `no-return` | 객체별 GT timeline에서 사후 계산 |
| Gap length | frame 수와 가능한 경우 초 단위; `short/medium/long` 경계는 manifest 생성 전에 dataset별 고정 | dataset sampling rate가 다르므로 frame 수만으로 데이터셋 간 동일 난이도를 주장하지 않음 |
| Prompt history | `initial-only`, `late-prompt`, `corrected` | 실제 사용자 prompt timeline에서 판정 |
| Object interaction | `single`, `multi-object`, `crossing`, `similar-distractor` | 공식 attribute 우선, 없으면 명시된 파생/검토 규칙 사용 |
| Visual challenge | `small-object`, `fast-motion`, `scale-change`, `appearance-change`, `camera-motion`, `background-confusion` | 공식 dataset attribute를 우선 사용하고 임의 육안 tag는 reviewer와 근거를 남김 |
| Switch relation | event `before/during/after`와 offset | event 기준 switch 실험에서만 사용 |

MOSEv2의 공식 disappearance/reappearance 지표와 CMMT의 자체 switch-relative 지표는
이름을 구분한다. LVOS의 long-term attribute도 원 논문의 정의를 만족할 때만 공식 명칭을
사용한다.

## 5. 고정 비교군

### 5.1 주 비교군

| 비교군 | Target에 전달·사용하는 데이터 | 검증 질문 |
|---|---|---|
| Source-only | 전환하지 않은 Small의 원래 state와 이후 예측 | Base+ 전환 자체가 필요한가? |
| Base+-native / Full Replay | switch 이전 RGB와 실제 prompt/correction timeline을 Base+가 처음부터 처리 | Target이 과거를 직접 처리한 정확도·비용은 얼마인가? |
| Direct State Copy | Small `maskmem_features`, `obj_ptr`, discrete history metadata; Target PE 재생성 | 학습·통계 보정 없이도 state가 호환되는가? |
| Moment-Matched Copy | Direct와 같은 state에 train-only paired statistics로 conditioning/type별 평균·표준편차 affine 보정 | 값 분포 보정만으로 충분한가? |
| Original-Prompt(s) Only | 등록 객체별 최초 실제 prompt와 당시 RGB/frame | 최초 객체 지정 정보만으로 충분한가? |
| Last-Visible Source Mask | 객체별 마지막 non-empty Source 예측 mask와 당시 RGB/frame | 객체별 최신 유효 관측 하나로 충분한가? |
| Original + Last-Visible | 위 두 anchor의 합집합 | 신뢰 가능한 최초·최신 관측 조합으로 충분한가? |
| Original-Prompt(s)+Replay-4 | 모든 객체의 original anchor + switch 전 최근 RGB 4장 | 짧은 최신 문맥의 정확도–비용은? |
| Original-Prompt(s)+Replay-8 | 모든 객체의 original anchor + switch 전 최근 RGB 8장 | recent-memory 범위 근방 문맥의 효과는? |
| Original-Prompt(s)+Replay-16 | 모든 객체의 original anchor + switch 전 최근 RGB 16장 | 더 긴 제한 replay의 추가 이득은? |
| Nonlinear Translator | Small memory/pointer를 component별 nonlinear mapper로 변환; 나머지는 Direct와 같은 계약 | 비선형 의미 변환이 강한 대안보다 유리한가? |

`Last-Mask (= Replay-1)`과 original anchor 없는 Recent-Window Replay-k는 객체 coverage가
약해 최종 비교군에서 제외한다. Empty-mask Target Reset은 실패 형태를 보는 진단 proxy일
뿐 정식 reset baseline으로 과장하지 않는다. Translation+Short Replay는 translation-only가
부족할 때의 후속 hybrid 분석이며 최초 주 비교표를 대체하지 않는다.

### 5.2 공정성 규칙

- 모든 방법은 동일한 등록 객체·prompt timeline·switch·future evaluation frame을 쓴다.
- Last-Visible은 GT가 아니라 Source의 switch 이전 예측으로 선택한다.
- 과거 mask를 다른 시점의 RGB에 붙이지 않고 항상 원래 frame ID와 함께 사용한다.
- Moment-Matched statistics는 MOSEv2/LVOS v2 **fit video**의 paired state로만 계산한다.
- development는 MOSEv2/LVOS v2 train에서 동결한 video-disjoint dev만 사용한다.
- MOSEv2/LVOS v2 official validation과 DAVIS/VOST의 Target-native state나 future frame을
  translator fitting·통계·선택에 쓰지 않는다.
- 각 방법의 입력 RGB 수, mask/prompt 수, state bytes, Target backbone call을 함께 기록한다.
- `state bytes`는 실제 handoff API가 옮기는 `maskmem_features`, `obj_ptr`, `frame_indices`, `slot_order`, `is_conditioning`, `validity`, `object_ids`, `switch_frame`만 센다. `presence_logits`, Source mask archive, prompt manifest, 영상 checksum, `num_frames`·높이·너비와 Target-generated PE는 제외한다.
- 영상·checkpoint·split checksum은 paired-data 무결성을 확인하는 실험 manifest 필드이며, handoff payload나 전송량의 일부로 보고하지 않는다.

## 6. 지표와 통계 단위

### 정확도·연속성

- 공식 `J`, `F`, `J&F`
- switch 이후 `+1/+5/+20` 관측 또는 frame의 J&F
- switch shock: 같은 frame 범위의 Base+-native 대비 하락
- GT-visible J&F와 GT-absent false-positive rate/area 분리
- disappearance 이후 reappearance recovery: 첫 재등장 `+1/+5/+20`, recovery length, no-recovery rate
- 다객체 identity break/ID switch는 구현된 판정 규칙이 있을 때만 보고
- case failure rate와 제외 사유

### 공식 평가와 CMMT switch 평가의 분리

- **DAVIS 2017:** 공개 validation GT에서 공식 semi-supervised `J`, `F`, `J&F`를 보고한다.
- **MOSEv2:** 공식 README는 train의 dense annotation과 validation의 first-frame-only annotation을 구분하고, validation/test 제출은 공식 Evaluation Server 경로로 안내한다. 따라서 로컬 MOSEv2 validation의 switch 이후 값은 official J&F라고 부르지 않으며, `first-frame prompt continuation diagnostic`으로만 기록한다. dense train의 switch J&F는 model selection·debug용이다.
- **LVOS v2:** 공식 `lvos-evaluation` toolkit의 `semi-supervised` validation score를 dataset-level 결과로 사용한다. CMMT의 `switch +1/+5/+20`, recovery length, visible/absent slice는 같은 prediction을 추가로 분석하는 자체 지표이며 LVOS 공식 score를 대체하지 않는다. test는 공식 CodaLab server 외의 local score로 주장하지 않는다.
- **VOST:** 공식 `J`와 마지막 25% frame의 transformation score `J_last`를 주 지표로
  보고한다. VOST는 경계가 모호하고 motion blur가 큰 특성을 반영해 `F`를 공식 지표처럼
  주장하지 않는다. CMMT switch는 미래 GT 사건에 맞추지 않고 영상 길이의 25/50/75%
  temporal quantile로 고정하며 primary switch는 50%다.

출처는 MOSEv2 공식 [README](https://github.com/henghuiding/MOSE-api)와 LVOS 공식 [evaluation toolkit](https://github.com/LingyiHongfd/lvos-evaluation)이다.

### 시스템 비용

- handoff wall-clock latency와 continuation latency
- Target 과거 backbone replay frame/call 수
- peak VRAM, host RAM(가능하면), 전송 state bytes
- translator parameter 수·checkpoint/statistics bytes·FLOPs 또는 MACs

### 통계

- 원자료는 `(video, object, switch)` 단위로 보존한다.
- 주 confidence interval은 **video-clustered bootstrap 95% CI**로 계산한다.
- 같은 영상의 여러 object/switch를 독립 video처럼 세지 않는다.
- seed 평균·분산과 유효 video/object/case 수를 함께 보고한다.
- 사전에 정한 `1.0 J&F point` 비열등 margin을 쓸 때는 비용 이득도 함께 제시한다.

## 7. Task 03 v1.1 완료 기준

- [x] 네 데이터 역할과 결과 섹션을 분리했다.
- [x] MOSEv2/LVOS v2 fit·dev와 sealed official validation을 분리했다.
- [x] DAVIS를 `engineering-seen external`, VOST를 primary untouched external로 고정했다.
- [x] translator-level zero-shot의 금지 데이터 사용을 고정했다.
- [x] 주 비교군·제외 비교군·진단군과 각 입력 데이터를 고정했다.
- [x] difficulty taxonomy, metric, clustered 통계 단위를 고정했다.
- [x] DAVIS·MOSEv2·LVOS v2 배포본의 이용조건, snapshot/checksum과 manifest를 기록했다.
- [x] MOSEv2/LVOS v2 fit/development membership과 checksum을 생성했다. (seed 7,
  hash 기반 80/20 video-level split)
- [x] DAVIS·MOSEv2·LVOS v2의 prompt/switch loader를 전수 검증했다. (`failure_count=0`)
- [x] MOSEv2/LVOS v2 공식 metric 구현·명칭과 자체 switch metric의 구분을 검증한다. (MOSEv2 validation은 first-frame annotation·official server 경로로, LVOS v2 validation은 공식 `lvos-evaluation` semi-supervised toolkit으로 분리한다.)
- [x] VOST 공식 배포본의 이용조건·download source snapshot을 기록했다. ([access ledger](../../reports/tasks/03_benchmark/runs/2026-09-24_vost_onboarding/access_ledger.md))
- [x] VOST archive를 내려받아 압축 구조와 checksum을 기록한다.
- [x] VOST train/val annotation·frame inventory를 검증했다 (`failure_count=0`;
  train 59,930쌍, val 7,820쌍). Test archive는 이름 목록만 포함되어 로컬 평가에서 제외한다.
- [x] VOST val/test 역할, 25/50/75% switch manifest와 checksum을 생성한다. (local val 210 case;
  test는 이름 목록만 있어 official server/별도 archive로 분리)
- [x] VOST prompt loader와 공식 `J/J_last` evaluator를 실제 데이터로 검증한다. (actual SAM 2
  state-export 및 official-layout PNG export; evaluator GT-copy contract smoke)
- [x] 외부 benchmark access ledger를 만들고 config freeze commit을 기록한다.

v1.0의 세 데이터셋 gate는 2026-09-23 모두 충족했다. 2026-09-24에 평가 역할을
강화하면서 VOST onboarding gate를 추가했고 2026-09-25에 완료했다. 이후 결과를 본 뒤
taxonomy, k 값, metric 또는 split을 유리하게 바꾸려면 날짜·이유·영향받는 run을 decision
log에 남긴다.
