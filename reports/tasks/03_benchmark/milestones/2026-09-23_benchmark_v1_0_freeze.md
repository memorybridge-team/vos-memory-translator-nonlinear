# Task 03 Benchmark Freeze Report

> 역사적 v1.0 완료 기록이다. 2026-09-24의 데이터 역할 분리와 VOST 추가는
> [v1.1 addendum](2026-09-24_v1_1_addendum.md)을 따른다.

## 범위

SAM 2.1 Small→Base+ memory handoff를 DAVIS 2017, MOSEv2, LVOS v2에서 동일한
case·prompt·switch 조건으로 평가하기 위한 protocol을 결과 확인 전에 고정한다.

## 데이터 snapshot

| Dataset | 확보·검증 상태 | 이용조건·평가 경로 |
|---|---|---|
| DAVIS 2017 | train 60/val 30, JPEG·PNG 각 6,208개 | CC BY-NC 4.0, 공개 val GT의 J/F/J&F |
| MOSEv2 | train 3,666 videos·311,843 dense masks, valid 433 videos·first mask only | CC BY-NC-SA 4.0·비상업 연구, validation/test는 공식 server 경로 |
| LVOS v2 | train 420 videos·160,452 masks, valid 140 videos·66,056 masks | annotation CC BY 4.0·영상 비상업 연구, 공식 `lvos-evaluation` semi-supervised toolkit |

LVOS train archive SHA-256은
`e4c0cfcb400dbd103ddea6eac3b5ffb5cdd3d50cb87013bf457d963a0b248c0b`이다.
LVOS valid archive SHA-256은
`beb488046f74e0cb4154a0cb2bcc2c79cae858da0693e5adabcf966a5712d4e2`이다.

## 고정 case와 split

- DAVIS train: 60 sequences, 599 cases; val: 30 sequences, 249 cases
- MOSEv2 train: 3,666 sequences, 20,841 cases; valid: 433 sequences, 1,720 cases
- LVOS v2 train: 420 sequences, 1,803 cases; valid: 140 sequences, 714 cases
- seed 7과 video ID SHA-256 bucket으로 train의 eligible video를 80/20 fit/development로 분리했다.
- 한 video의 object/switch는 두 split에 동시에 포함되지 않는다.
- case가 생성되지 않은 짧거나 부적격한 video/object는 원본 manifest의 `excluded`에 남긴다.
- split 파일은 case를 복제하지 않고 video membership, case 수, source manifest checksum만 저장한다.

## Prompt와 correction 정책

- 실제 dataset interaction은 객체별 최초 mask prompt다.
- MOSEv2 train 311,843개 dense mask를 스캔해 객체별 최초 등장 frame을 확인했다.
- 공개 데이터셋에 없는 correction을 미래 GT에서 임의 생성하지 않는다.
- correction 실험은 별도 interaction manifest에 frame/object/type/mask reference를 명시하고 모든 방법에 동일하게 제공한다.
- correction이 없는 표준 benchmark case의 correction timeline은 빈 목록이다.

## Metric 명칭

- DAVIS 공개 val에서는 공식 J, F, J&F를 사용한다.
- MOSEv2 valid의 local switch 결과는 official J&F가 아니라 first-frame-prompt continuation diagnostic이다.
- LVOS dataset-level validation score는 공식 toolkit으로 계산한다.
- switch +1/+5/+20, shock, absence false positive, reappearance recovery는 CMMT 자체 분석 지표로 분리한다.

## 구현 검증

- 모든 source manifest와 compact split manifest의 content SHA-256 재계산이 일치했다.
- fit/development video 집합은 서로 겹치지 않고 eligible video 집합을 정확히 분할한다.
- 실제 RGB·prompt mask·object ID·switch frame loader 전수 검증 결과는 같은 날짜의 JSON report로 남겼다.
- DAVIS train 599/val 249, MOSEv2 train 20,841/valid 1,720, LVOS v2 train 1,803/valid 714 cases가 모두 `failure_count=0`이었다.
- 최초 LVOS manifest가 실제로 존재하지 않는 중간 frame ID를 switch로 고르는 오류를 검출했다. 실제 sparse RGB frame ID 목록에서 quantile을 선택하도록 builder를 수정한 뒤 재검증했다.

## Task 경계

Task 03은 입력·분할·비교군·metric 계약을 동결한다. SAM 2 runtime correction continuation은
Task 06, paired state 수집은 Task 07, baseline 실행기는 Task 08, nonlinear 학습은 Task 09가 담당한다.
