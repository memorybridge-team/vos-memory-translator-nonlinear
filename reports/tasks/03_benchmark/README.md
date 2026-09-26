# Task 03 — Benchmark protocol

> 상태: **In Progress** — VOST onboarding core gate 완료, PUMaVOS onboarding 대기
> Canonical Issue: [#4](https://github.com/memorybridge-team/vos-memory-translator-nonlinear/issues/4)
> 현재 PR: [#9](https://github.com/memorybridge-team/vos-memory-translator-nonlinear/pull/9)

## 한눈에 보기

MOSEv2·LVOS v2의 v1.0 manifest, video-level split과 loader 검증은 2026-09-23 완료했다.
2026-09-24에 in-domain과 external zero-shot을 분리하면서 VOST onboarding gate를 추가했고
2026-09-25에 archive·manifest·loader·evaluator contract 검증까지 완료했다. 다만 사용자의
PUMaVOS external stress 반영 요청으로 Task는 열린 상태를 유지한다. DAVIS v1.0 자료는 역사적
milestone으로만 보존하며 현재 연구 범위에는 포함하지 않는다.

## 현재 데이터 역할

| 역할 | 데이터 |
|---|---|
| Translator fit | MOSEv2/LVOS v2 train-fit |
| In-domain development | 두 train의 video-disjoint dev |
| Sealed in-domain final | MOSEv2 official valid, LVOS v2 official val |
| External frozen benchmark | VOST primary external zero-shot, PUMaVOS secondary external stress |

## 완료된 milestone과 실행 기록

- [2026-09-23 Benchmark v1.0 freeze](milestones/2026-09-23_benchmark_v1_0_freeze.md)
- [2026-09-24 v1.1 addendum](milestones/2026-09-24_v1_1_addendum.md)
- [2026-09-22 manifest validation](runs/2026-09-22_manifest_validation.md)
- [2026-09-23 loader validation JSON](runs/2026-09-23_loader_validation/)
- [2026-09-24 VOST access ledger](runs/2026-09-24_vost_onboarding/access_ledger.md)
- [2026-09-25 VOST inventory and switch manifest](runs/2026-09-25_vost_inventory.md)
- [2026-09-25 official evaluator smoke](runs/2026-09-25_vost_evaluator_smoke.md)
- [2026-09-25 SAM 2 loader and PNG export smoke](runs/2026-09-25_vost_sam2_loader_and_export_smoke.md)

VOST 공식 archive(약 54.0GB)는 RunPod에 다운로드·압축 해제되었다.
Archive SHA-256은 `fb17075ab3afab0fe30f264d8adce2e29ce6249a73cd44d2a9cf4936cc8de978`이며,
공식 split 목록은 train 572, val 70, test 71 sequence이다. 추출본의 1차 파일 수는
Annotations 67,751개, JPEGImages 67,751개, JPEGImages_10fps 15,607개, Videos 642개로
확인했다. 실제 annotation/frame 대응·switch manifest·loader gate는 아직 남아 있다.
annotation/frame 대응 inventory는 train 59,930쌍, val 7,820쌍에서 `failure_count=0`으로
확인했다. test는 sequence 이름 목록만 있고 로컬 frame/annotation은 없어 공식 서버용으로
분리한다. [inventory 보고서](runs/2026-09-25_vost_inventory.md)

## VOST split별 사용 규칙

- train 572개는 main translator의 fit/dev와 model selection에 쓰지 않는다. 별도 VOST adaptation upper-bound만 허용한다.
- 공개 GT가 있는 val 70개는 MOSE/LVOS dev에서 설정을 동결한 뒤 한 번 실행하는 external zero-shot 평가다. first prompt mask는 VOS 입력으로 허용하되 결과로 설정을 바꾸지 않는다.
- test 71개는 공개 archive에 영상·GT가 없고 sequence 이름만 있다. official server가 영상과 initial prompt를 제공하는 접근 계약을 확인한 뒤 PNG를 제출하는 sealed 결과로만 사용한다.

## VOST onboarding core gate

- [x] VOST 이용조건·공식 download source 확인 ([access ledger](runs/2026-09-24_vost_onboarding/access_ledger.md))
- [x] VOST archive 다운로드·SHA-256·압축 해제 기록
- [x] VOST val/test inventory와 25/50/75% switch manifest 생성 (val 완료; test 파일은 archive에 없음)
- [x] VOST prompt loader와 공식 `J`/`J_last` evaluator 검증 (actual VOST SAM 2 export + evaluator contract smoke)
- [x] external benchmark access ledger와 config-freeze commit 기록

VOST onboarding core gate는 모두 통과했다. 추가 검증 데이터셋은 PUMaVOS 전체 24개를 split 없는
secondary external zero-shot으로 확정했다. 아래 onboarding
gate가 남아 있으므로 Task 03은 `In Progress`를 유지한다. 전체 baseline 및 translator score 실행은
Task 08·09·13에서 수행한다.

## PUMaVOS 남은 gate

- [ ] 공식 download source·CC BY 4.0 license·archive checksum 기록
- [ ] 24 videos·21,187 dense frame/mask inventory와 object ID 검증
- [ ] 객체별 first-nonempty GT mask만 conditioning으로 쓰는 loader와 temporal switch manifest 생성
- [ ] local J/F/J&F 및 CMMT switch-relative evaluator contract smoke
- [ ] 24-video per-video 결과·video-clustered bootstrap CI 출력 형식 검증
