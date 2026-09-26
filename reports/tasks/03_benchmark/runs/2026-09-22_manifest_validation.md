# Task 03 manifest validation — 2026-09-22

## 확인한 범위

RunPod에서 세 validation manifest를 같은 schema로 읽고 `sequence_count`,
`case_count`, `case_id` 중복을 확인했다.

| Dataset/split | Sequences | Cases | Duplicate case IDs |
|---|---:|---:|---:|
| DAVIS 2017 val | 30 | 249 | 0 |
| MOSEv2 valid | 433 | 1,720 | 0 |
| LVOS v2 val | 140 | 717 | 0 |

LVOS v2 Eval archive는 공식 `valid.zip`이며 다음을 확인했다.

- Archive SHA-256: `beb488046f74e0cb4154a0cb2bcc2c79cae858da0693e5adabcf966a5712d4e2`
- Extracted JPEG frames: 66,056
- Extracted annotation masks: 66,056
- Extracted video directories: 140
- LVOS manifest content SHA-256: `3a3e3c7610a08e251f109d77d516723392b14484be81ad3eb604aa3a63a1b5c1`

## 검증 명령

RunPod에서 다음 세 JSON을 읽었다.

```text
/workspace/cmmt-mose-manifest/manifests/davis2017_val_v1.json
/workspace/cmmt-mose-manifest/manifests/mosev2_valid_v1.json
/workspace/CMMT/manifests/lvosv2_valid_v1.json
```

각 manifest에 대해 선언된 sequence/case 수와 실제 `cases` 길이를 비교하고,
`case_id`를 set으로 만들어 중복이 없는지 검사했다.

## 아직 남은 Task 03 gate

- MOSEv2/LVOS v2 train에서 fit/development video manifest 생성
- LVOS attribute 및 prompt/correction timeline loader 검증
- 각 배포본 이용조건·snapshot 기록 보강
- MOSEv2/LVOS v2 공식 metric과 CMMT switch-relative metric의 구현·명칭 검증

따라서 이 보고서는 validation manifest 검증 완료를 기록하지만 Task 03을 `Done`으로
전환하는 근거로 사용하지 않는다.
