# 2026-09-25 · VOST SAM 2 loader and PNG-export smoke

## 목적

VOST의 frame 이름·annotation label을 CMMT/SAM 2 runtime에 전달하고, prediction을 공식
VOST evaluator가 요구하는 PNG layout으로 다시 내보내는 end-to-end 입력·출력 계약을 검증한다.
이 run은 translator 성능 비교가 아니다.

## 발견과 처리

VOST RGB frame은 `frame00102.jpg`처럼 이름이 붙어 있지만 SAM 2 video loader는 integer
stem만 허용한다. 원본 dataset을 변경하지 않고
[`prepare_vost_sam2_sequence.py`](../../../../scripts/prepare_vost_sam2_sequence.py)가
`00000.jpg` … numeric symlink view와 원본 stem↔SAM 2 index mapping을 만든다.

## 실제 실행

| 항목 | 값 |
|---|---|
| VOST sequence | val `9671_split_cups` |
| original frames | 42 (`frame00000` … `frame00246`) |
| prompt | `frame00000.png`, object ID 1 |
| source model | SAM 2.1 Small, upstream `2b90b9f` |
| state-export switch | SAM 2 index 20 (원본 stem `frame00120`) |
| state export | frames 0–20 기록, canonical state 생성 성공 |
| PNG export | SAM 2 prediction 42장 |

`export_vost_sam2_predictions.py`는 prediction을
`results/<sequence>/<original-frame-stem>.png`으로 쓴다. GT annotation과 exporter output은
각각 42 PNG였고 `comm -3` 비교에서 missing/extra frame stem이 없었다.

## 해석

- VOST val의 실제 RGB·prompt mask·object label·switch index는 CMMT runtime에 연결 가능하다.
- 현재 export smoke는 **SAM 2 Small source-only prediction**으로 output contract만 검증한다.
  Small→Base+ handoff, baseline, nonlinear translator의 정량 성능 평가는 후속 Task 08/09/13에서
  동일 layout을 사용해 수행한다.
- VOST evaluator의 original-resolution 전체 run은 sequence별 memory budget과 CSV 병합으로
  실행한다. Task 03은 evaluator input contract와 data protocol을 동결하는 단계이며 전체 모델
  score를 산출하는 단계가 아니다.
