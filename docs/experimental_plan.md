# 4주 논문 실험 계획

기준일: 2026-09-18 KST. [대한전자공학회 2026 추계학술대회](https://conf.theieie.org/2026f/pages/outlines.vm)의 논문 제출일은 **2026-10-19**다. 4주 연구 구간에 10/15~19 검토·제출 구간을 이어 운영한다. 마감 시각·시간대와 최종 업로드 형식은 제출 화면에서 다시 확인한다.

Baseline의 정확한 입력 데이터, 공정성 규칙, 성공·중단 판정은 [`01 연구 범위 동결`](design/01_scope_baselines_success_stop.md)을 기준으로 한다.

**실험 규모 원칙:** 4주는 일정 관리 기준이며 데이터 크기·case 수·반복·필요 비교군의 상한이 아니다. 세 데이터셋에서 결론을 뒷받침할 만큼 학습·검증·평가를 실행한다. 실험은 checksum과 상태 파일을 가진 재개 가능한 GPU 작업으로 운영한다. 일정 때문에 표본을 줄인 결과를 전체 benchmark라고 주장하지 않는다.

## 고정 조건

| 항목 | 결정 |
|---|---|
| 모델 | SAM 2.1 Small → Base+ |
| 데이터셋 | DAVIS 2017, MOSEv2, LVOS v2 |
| 제안 방식 | Nonlinear memory translator |
| 후속 성능 | J&F, 재등장·부재 오류, switch 직후 성능 |
| 시스템 비용 | handoff 지연, 과거 재처리량, VRAM, 전송 bytes |

Base+를 Small의 메모리로 이어 쓰게 하는 것이 주 실험이다. Tiny→Large 및 Tiny→Base+의 과거 결과는 코드와 위험 분석에 활용하되 새 pair의 결과로 취급하지 않는다. 새 모델 쌍이 추가로 필요하다는 실험적 근거가 생기면 논문 목표와 비교 가능성을 검토해 확장한다.

Source와 Target은 동일 영상·동일 frame 순서·동일 preprocessing을 사용한다고 연구 조건으로 고정한다. 영상 checksum과 frame 수·해상도 검사는 dataset/manifest 생성 오류를 찾는 실험 준비 절차이며, CMMT memory handoff payload나 translator API 입력으로 계산하지 않는다. 분산 서비스에서의 영상 identity 검증은 연구 범위 밖의 별도 handoff protocol로 둔다.

## 1주차: 상태 계약과 자료 준비

- 공식 SAM 2 revision, Small/Base+ checkpoint hash, DAVIS/MOSEv2/LVOS v2의 이용 조건·version·split·해상도·fps를 기록한다.
- Base+ same-checkpoint export→inject가 native continuation을 재현하는지 검증한다. 객체가 안 보이거나 다시 등장하는 사례와 다객체 ID를 포함한다.
- 세 데이터셋의 표준 loader와 `(video, object, switch)` manifest 형식을 맞춘다. Source와 target이 같은 frame/prompt 이력을 사용하는지 검사한다.
- 데이터는 학습·모델 선택·최종 평가 사이에 영상 단위로 분리한다. 기존 DAVIS train 내부 pilot을 untouched test라고 부르지 않는다.

**산출물:** 데이터/체크포인트 inventory, checksum이 있는 manifest, Base+ round-trip 보고서, smoke 테스트와 실행 명령.

## 2주차: 같은 사례에서 비교군 고정

| 비교군 | target에 전달하는 것 | 역할 |
|---|---|---|
| Source-only | 전환하지 않은 Small state | 전환 필요성 참조 |
| Base+-native / Full Replay | 과거 RGB와 실제 prompt timeline | target 직접 처리 참조 및 비용 |
| Direct State Copy | Small memory·pointer·history 조립에 필요한 최소 metadata | 번역 자체 필요성 |
| Moment-Matched Copy | 학습 split의 paired state로 고정한 component·conditioning별 평균·표준편차로 Small memory·pointer를 affine 보정 | 단순 분포 보정으로 충분한지 확인 |
| Original-Prompt(s) Only | 객체별 처음 실제 prompt와 해당 RGB | 최초 지정 정보의 효과 |
| Last-Visible Source Mask | 객체별 마지막 비어 있지 않은 Small 예측 mask와 해당 RGB | 최신 관측의 효과 |
| Original + Last-Visible | 위 두 anchor | 조합의 효과 |
| Original-Prompt(s) + Replay-4/8/16 | 객체별 원래 anchor + 최근 RGB 4/8/16장 | 객체 등록을 보장한 제한된 재처리의 효과 |
| Nonlinear Translator | Small memory·pointer를 component별 nonlinear mapper로 변환 | 단순 복사·정규화보다 의미 변환이 필요한지 확인 |

GT mask를 switch 시점에 새로 주지 않는다. Last-Visible 선택은 source의 과거 예측만 사용한다. Moment-Matched Copy의 통계는 학습 split의 paired state로만 계산하며, test video·future frame·test target-native state는 쓰지 않는다. 각 방법을 동일 manifest 전체에서 실행한 뒤 visible/absent/reappearance 특성별 결과를 나눈다. 같은 checkpoint export→inject는 경쟁군이 아니라 정확성 검사다. Empty-mask Reset은 현재 코드의 proxy임을 표기한다.

전송 bytes에는 `maskmem_features`, `obj_ptr`와 실제로 전달하는 history 조립 metadata만 포함한다. Dataset manifest, 원본 RGB, checkpoint hash, 영상 checksum, Target이 자체 산출한 `num_frames`·높이·너비는 handoff payload에서 제외한다.

**산출물:** 각 비교군의 정보 입력·비용 정의, 동일 사례 결과표, 다객체/빈 mask 테스트, 누락 사례 없는 실행 로그.

## 3주차: Nonlinear 구조와 학습 비교

1. 기존 component-wise residual MLP를 최소 재현 모델로 사용한다. Spatial memory와 object pointer를 분리해 mask에 대한 영향을 측정한다. positional encoding은 Target에서 재생성하고 score logit은 진단 기록으로만 남긴다.
2. Component별 scale·loss normalization과 validation 기반 stopping을 확인한다. State MSE와 후속 J&F를 함께 기록한다.
3. Gated MLP를 두 번째 후보로 비교한다. 원본 source 정보와 변환량을 조절하는 gate가 실제 일반화에 도움이 되는지 본다.
4. Slot/context attention을 세 번째 후보로 비교한다. 각 구조는 같은 train/validation split에서 비교하며 parameter 수·학습량·handoff 연산량을 함께 보고한다.
5. One-step mask distillation 및 짧은 rollout loss를 ablation으로 비교한다. 데이터와 계산량이 추가로 필요하면 실험을 재개 가능한 작업으로 확장한다. Backbone은 동결한다.

모델 구조·하이퍼파라미터 선택은 development split에만 근거한다. 작은 학습 사례 overfit은 파이프라인 검사이며 held-out 성공의 근거가 아니다. 최종 checkpoint와 seed, 설정, 학습곡선, validation 지표를 남긴다.

**산출물:** 구조별 parameter·latency·J&F 표, component ablation, 선정한 한 가지 주 방법과 실패 조건.

## 4주차: 세 데이터셋 결과와 논문

- DAVIS: 전체 및 사건별 J&F, 전환 직후 shock, 객체 유지, 간단한 재현 그림.
- MOSEv2: 가림·사라짐·재등장·유사 객체 조건. 공식 지표 구현과 자체 switch별 지표를 명칭까지 분리한다.
- LVOS v2: 긴 관측 공백, 재등장, 경과 프레임/초와 비용. Sparse annotation 간격을 시간으로 해석한다.
- Source-only와 Base+-native 차이를 확인한 뒤 translator의 정확도–전환 지연–전송량 관계를 강한 mask/prompt/replay 비교군과 함께 제시한다.
- 영상별 표본 수, 실패 사례, video-clustered confidence interval, 한계를 보고한다. 다객체 ID switch는 판정 규칙을 구현한 경우에만 주장한다.
- 실험 수가 늘어도 영상 단위 독립성과 표본 수를 보존한다. 복수 seed, hard-event의 충분한 사례, 데이터셋별 전체 평가를 계획에 포함하고 필요하면 더 실행한다.
- 실험 설정·원자료 경로·코드 revision·논문 표/그림의 숫자가 대응하는지 감사한다.

**산출물:** 재현 표·그림·case gallery, 서론/방법/실험/한계 초안, 제출 형식에 맞춘 원고.

## 날짜별 통합 게이트

- **9/23:** 세 데이터셋의 버전·이용 조건·split과 객체별 `(video, object, switch)` manifest, 강한 비교군·평가 지표를 고정한다.
- **9/27~30:** Base+ same-checkpoint export→inject를 단일·다객체, 부재·재등장에서 검증한 후 paired state 수집을 shard 단위로 확장한다. 실패하면 대규모 학습보다 상태 계약 수정이 우선이다.
- **10/3~11:** overfit/smoke gate를 통과한 shard부터 세 nonlinear 후보의 재개 가능한 전체 학습·반복 평가를 시작한다. 기존 일정의 10/8 일괄 시작을 기다리지 않는다.
- **10/7:** 후보 구조와 평가 프로토콜의 중간 검토만 한다. 최종 방법은 전체 결과를 보기 전까지 확정하지 않는다.
- **10/12~14:** 세 데이터셋·전체 비교군·사전 정의한 hard-event strata의 결과와 시스템 비용을 대조해 주 방법을 선정하고 표·그림을 동결한다. 미완료/실패는 그대로 명시한다.
- **10/15~17:** 재현성·통계·인용·그림·원고를 내부 검토하고 수정한다.
- **10/18:** [공식 제출 안내](https://conf.theieie.org/2026f/pages/conference_paperinfo.vm)의 실제 업로드 파일 형식과 양식을 재확인하고 제출 모의 실행을 한다. PDF만 준비하면 된다고 가정하지 않는다.
- **10/19:** 확인된 마감 시각 이전에 최종 원고를 제출하고 접수 증빙을 보관한다.

일정은 작업의 선후관계와 검증 게이트를 나타낸다. 기간을 맞추려고 DAVIS 2017·MOSEv2·LVOS v2, 필수 비교군, 반복 seed 또는 실패 사례 보고를 조용히 축소하지 않는다.

## 주간 Go/No-Go

- 1주차: 동일 checkpoint round-trip과 대상별 manifest 무결성 실패 시, 신규 학습 전에 상태 주입·데이터 정렬을 고친다.
- 2주차: Base+-native가 Source-only보다 유리한 조건이 없는 경우, 전환의 필요성을 다시 검토한다. 간단한 baseline이 충분히 강하면 그 사실을 남긴다.
- 3주차: tensor loss만 내려가고 미래 mask가 개선되지 않으면 component·loss·state injection을 분리 진단한다.
- 4주차: 세 데이터셋의 실행 범위와 표본 수를 명시한다. 데이터셋별 실패나 미완료를 성공으로 대체하지 않는다.

기존 Tiny→Large pilot의 제한과 baseline 정의를 자세히 설명한 [과거 설계 자료](legacy/2026-09-18_RESEARCH_MOTIVATION_BASELINES_EVALUATION.md)는 참고용이다. 이 문서가 현재 범위의 기준이다.
