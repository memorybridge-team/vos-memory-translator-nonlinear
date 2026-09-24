# Task 03 — Benchmark protocol

> 상태: **In Progress**
> Canonical Issue: [#4](https://github.com/memorybridge-team/vos-memory-translator-nonlinear/issues/4)
> 현재 PR: [#9](https://github.com/memorybridge-team/vos-memory-translator-nonlinear/pull/9)

## 한눈에 보기

DAVIS 2017·MOSEv2·LVOS v2의 v1.0 manifest, video-level split과 loader 검증은
2026-09-23 완료했다. 2026-09-24에 in-domain과 external zero-shot을 분리하면서 VOST
onboarding gate를 추가했으므로 Task 03 v1.1은 진행 중이다.

## 현재 데이터 역할

| 역할 | 데이터 |
|---|---|
| Translator fit | MOSEv2/LVOS v2 train-fit |
| In-domain development | 두 train의 video-disjoint dev |
| Sealed in-domain final | MOSEv2 official valid, LVOS v2 official val |
| External frozen benchmark | VOST primary, DAVIS engineering-seen |

## 완료된 milestone과 실행 기록

- [2026-09-23 Benchmark v1.0 freeze](milestones/2026-09-23_benchmark_v1_0_freeze.md)
- [2026-09-24 v1.1 addendum](milestones/2026-09-24_v1_1_addendum.md)
- [2026-09-22 manifest validation](runs/2026-09-22_manifest_validation.md)
- [2026-09-23 loader validation JSON](runs/2026-09-23_loader_validation/)
- [2026-09-24 VOST access ledger](runs/2026-09-24_vost_onboarding/access_ledger.md)
- [2026-09-25 VOST inventory and switch manifest](runs/2026-09-25_vost_inventory.md)
- [2026-09-25 official evaluator smoke](runs/2026-09-25_vost_evaluator_smoke.md)

VOST 공식 archive(약 54.0GB)는 RunPod에 다운로드·압축 해제되었다.
Archive SHA-256은 `fb17075ab3afab0fe30f264d8adce2e29ce6249a73cd44d2a9cf4936cc8de978`이며,
공식 split 목록은 train 572, val 70, test 71 sequence이다. 추출본의 1차 파일 수는
Annotations 67,751개, JPEGImages 67,751개, JPEGImages_10fps 15,607개, Videos 642개로
확인했다. 실제 annotation/frame 대응·switch manifest·loader gate는 아직 남아 있다.
annotation/frame 대응 inventory는 train 59,930쌍, val 7,820쌍에서 `failure_count=0`으로
확인했다. test는 sequence 이름 목록만 있고 로컬 frame/annotation은 없어 공식 서버용으로
분리한다. [inventory 보고서](runs/2026-09-25_vost_inventory.md)

## 남은 v1.1 gate

- [x] VOST 이용조건·공식 download source 확인 ([access ledger](runs/2026-09-24_vost_onboarding/access_ledger.md))
- [x] VOST archive 다운로드·SHA-256·압축 해제 기록
- [ ] VOST val/test inventory와 25/50/75% switch manifest 생성 (val inventory·manifest 완료; test 파일은 archive에 없음)
- [ ] VOST prompt loader와 공식 `J`/`J_last` evaluator 검증 (evaluator GT-copy smoke 완료;
  CMMT PNG export smoke가 남음)
- [ ] external benchmark access ledger와 config-freeze commit 기록

위 gate를 통과하고 최종 통합본을 작성·병합하기 전까지 Task 03을 Done으로 바꾸지 않는다.
