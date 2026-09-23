# Task 03 benchmark protocol v1.1 addendum — 2026-09-24

## 결정

Task 03 v1.0의 DAVIS 2017·MOSEv2·LVOS v2 manifest, loader 및 metric 검증 결과는
그대로 보존한다. 다만 translator의 in-domain 성능과 cross-dataset 일반화를 구분하기
위해 데이터 역할을 다음처럼 재동결한다.

| 역할 | 데이터 |
|---|---|
| Translator fit | MOSEv2 train-fit + LVOS v2 train-fit |
| In-domain development | 두 train split의 video-disjoint dev |
| Sealed in-domain final | MOSEv2 official valid + LVOS v2 official val |
| External frozen benchmark | VOST val/test(primary), DAVIS 2017 val(engineering-seen) |

## 변경 이유

학습 데이터셋과 같은 분포의 validation만 보고하면 translator가 새로운 영상 분포에
일반화하는지 알 수 없다. 반대로 official validation이나 external benchmark로 구조·loss·
threshold를 선택하면 최종 평가가 오염된다. 따라서 학습, model selection, in-domain final,
external zero-shot의 네 역할을 운영상 분리한다.

DAVIS val의 `walking`, `bike-packing`, `india`는 Task 06 runtime 개발에 이미 사용했다.
따라서 DAVIS는 보조 external 결과로 유지하되 `untouched`라고 표현하지 않는다. 아직 개발에
사용하지 않은 VOST를 primary external benchmark로 둔다.

## Zero-shot 주장 범위

`zero-shot`은 frozen SAM 2 전체가 해당 dataset을 전혀 보지 않았다는 뜻이 아니라
**translator-level cross-dataset zero-shot transfer**를 뜻한다. DAVIS/VOST의 state, 통계,
label-derived threshold, early stopping 또는 실패 사례는 translator fitting과 선택에 쓰지
않는다. 첫-frame GT prompt와 metric용 GT만 표준 VOS 평가 절차로 허용한다.

## 주 결과 matrix

학습 행은 `MOSE-only`, `LVOS-only`, `MOSE+LVOS`로 고정하고, 평가 열은 MOSE valid,
LVOS val, DAVIS val, VOST val로 고정한다. VOST-train fine-tuning은 선택적으로 수행하되
별도 adaptation upper-bound 행으로만 보고한다.

논문 결과는 두 표로 나눈다.

1. In-domain held-out evaluation: MOSEv2 valid, LVOS v2 val
2. External cross-dataset zero-shot evaluation: VOST val/test, DAVIS val(engineering-seen)

## 남은 v1.1 gate

- VOST 이용조건·download source·archive checksum 기록
- VOST val/test inventory와 25/50/75% switch manifest 생성(primary 50%)
- prompt loader와 공식 `J`, `J_tr` evaluator 검증
- external benchmark access ledger와 config freeze commit 기록

이 네 항목이 끝날 때까지 Task 03은 `In Progress`다. Task 07의 collector 구현은 병렬로
준비할 수 있지만, 수집 대상은 MOSEv2/LVOS v2 fit/dev로 제한한다.

## 후속 Task에 미치는 영향

- **Task 07:** MOSEv2/LVOS v2 fit/dev만 paired-state training shard에 포함한다.
- **Task 08:** Moment-Matched 통계도 fit shard에서만 계산하고 VOST `J/J_tr` evaluator를 추가한다.
- **Task 09–11:** dev에서 모든 선택을 끝낸 뒤 config를 동결한다.
- **Task 13:** sealed in-domain과 external zero-shot을 서로 다른 표·결론으로 보고한다.
