# 2026-09-25 · VOST official evaluator smoke

## 목적

공식 TRI-ML/VOST evaluator가 요구하는 dataset root, sequence folder, PNG frame name,
prediction mask label과 metric 출력 계약을 확인한다. 이 run은 CMMT 성능 평가가 아니다.

## 고정 입력

| 항목 | 값 |
|---|---|
| official evaluator | `TRI-ML/VOST` commit `fe274574cb03c8a3ea83e121dd76e20b703781fd` |
| sequence | VOST val `9671_split_cups` (42 frames) |
| prediction | GT annotation을 prediction으로 복사 |
| smoke resize | max side 512 px; 원본을 변경하지 않은 별도 복사본 |
| command helper | `scripts/run_vost_evaluator_smoke.py` |

## 결과

공식 `evaluation/evaluation_method.py`가 성공적으로 실행됐다.

| metric | result |
|---|---:|
| J-Mean | 1.000 |
| J-Recall | 1.000 |
| J-Decay | 0.000 |
| J_last-Mean | 1.000 |
| J_last-Recall | 1.000 |
| J_last-Decay | 0.000 |

공식 구현의 후반 변환 구간 지표 이름은 `J_last`다. 따라서 프로젝트 문서의 과거 표기
`J_tr`를 `J_last`로 정정한다.

## 한계와 후속 실행 규칙

- GT를 prediction으로 복사했으므로 1.0은 **정상 입력 계약 확인값**일 뿐 모델 성능이 아니다.
- 원본 해상도 119-frame smoke는 evaluator가 모든 mask를 `float64` 배열로 적재하면서 약
  60GB까지 사용한 뒤 환경에서 `SIGKILL`됐다. 원본 해상도 최종 평가는 sequence 단위 실행,
  메모리 상한 기록, 결과 CSV 병합으로 운영한다.
- evaluator는 첫 annotation frame의 object ID를 기준으로 삼는다. 이후 새 label이 생기는
  sequence에서는 warning이 발생할 수 있으므로 CMMT prompt/correction loader는 prompt 시점과
  평가 대상 object ID를 명시적으로 선택해야 한다.
- 다음 gate는 CMMT prediction loader의 PNG export smoke다. 이 evaluator smoke만으로 Task 03을
  Done으로 바꾸지 않는다.
