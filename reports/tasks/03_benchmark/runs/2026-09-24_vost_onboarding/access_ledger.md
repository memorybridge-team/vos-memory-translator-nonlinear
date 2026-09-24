# 2026-09-24 · VOST onboarding access ledger

## 확인 범위

Task 03 v1.1의 첫 gate로 VOST 공식 배포 경로, 이용조건, 평가 구현 위치를 확인했다.
archive 다운로드·SHA-256·압축 해제는 완료되었고, annotation/frame 대응·manifest·실제
loader 실행은 미완료 상태로 유지한다.

## 공식 자료 snapshot

| 항목 | 확인 내용 | 상태 |
|---|---|---|
| 공식 dataset page | [VOST Data](https://www.vostdataset.org/data.html) | 확인 |
| 공식 code/evaluation | [TRI-ML/VOST](https://github.com/TRI-ML/VOST), `evaluation/evaluation_method.py` | 확인 |
| train/validation archive | `https://tri-ml-public.s3.amazonaws.com/datasets/VOST.zip` | 다운로드·압축 해제 완료 |
| test split | 공식 페이지에서 challenge deadline 전 별도 공개한다고 안내 | 접근 시점 확인 필요 |
| license | Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0) | 확인 |
| official metrics | VOST evaluator의 `J`와 변환 구간 중심 `J_tr`를 보고 | 코드 실행 전 |
| archive SHA-256 | `fb17075ab3afab0fe30f264d8adce2e29ce6249a73cd44d2a9cf4936cc8de978` | 확인 |

2026-09-24 KST에 공식 URL의 HEAD를 확인했다. `Content-Length`는
`54,012,104,924` bytes(약 50.3 GiB)이며, multipart S3 `ETag`는 SHA-256 대체값으로
사용하지 않는다. RunPod `/workspace/datasets/VOST/VOST.zip`에 `wget --continue`로
다운로드를 시작했고, 로그는 `/workspace/logs/vost_download.log`에 둔다. archive가
완전히 닫힌 뒤에 SHA-256과 압축 구조를 검증한다.

SHA-256 계산은 완료됐고, ZIP 중앙 목록에서 총 `153,136`개 파일을 확인했다. 공식
`ImageSets` 파일은 train `572`, val `70`, test `71` sequence이며, 마지막 줄에 개행이
없어 `wc -l`은 각각 하나씩 작게 보일 수 있다. 내부 구조에는 `Annotations`, `Videos`,
`JPEGImages`(5 fps), `JPEGImages_10fps`, `ImageSets`가 포함된다.

2026-09-25 KST에 `/workspace/datasets/VOST/extracted`로 압축 해제를 완료했다. 1차
inventory는 Annotations 67,751, JPEGImages 67,751, JPEGImages_10fps 15,607,
Videos 642이며, split 파일은 train/val/test 572/70/71 sequence다. 다음 단계에서
annotation/video/frame 대응과 실제 25/50/75% switch manifest를 검증한다.

2026-09-25 KST inventory에서 train 59,930 frame/annotation 쌍과 val 7,820 쌍은 모두
stem 대응 `failure_count=0`이었다. test는 이름 목록 71개만 포함되고 frame/annotation은
없어 로컬 test 평가 대상으로 사용하지 않는다. 상세 결과는
[`2026-09-25_vost_inventory.md`](../2026-09-25_vost_inventory.md)에 기록했다.

## 연구상 고정 규칙

- VOST는 `primary external cross-dataset zero-shot` 평가로만 사용한다.
- VOST train/validation은 translator fit, moment statistics, checkpoint·threshold·replay-k
  선택에 사용하지 않는다.
- 25/50/75% switch manifest는 archive의 실제 frame·object metadata를 확인한 뒤 생성한다.
- test archive가 공개되지 않은 경우 val을 고정 external 결과로 보고하고, test는
  접근 가능 시 별도 sealed 결과로 추가한다. 없는 split을 추정해 만들지 않는다.

## 다음 실행 gate

1. 완료: RunPod에 공식 archive를 다운로드하고 파일 목록·압축 크기·SHA-256을 기록한다.
2. train/validation inventory에서 실제 video/object/frame 범위를 산출한다.
3. 25/50/75% switch manifest와 prompt loader를 생성한다.
4. 공식 evaluator의 `J`/`J_tr`를 작은 smoke subset에서 실행한 뒤 전체 검증한다.
5. access ledger와 config-freeze commit을 연결하고 Task 03 완료 여부를 재판정한다.
