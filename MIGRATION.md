# 이관 출처와 책임

- `transformer_translator.py`, `frozen_tensor_api.py`: 모델팀
  `feat/transformer-adapted-translator`, revision
  `746ea3e7d84c366c2d7ac06159e90a1f684bca56`에서 기존 학습 workspace로
  선별 이관한 파일을 재사용했다. 원 commit author는 `fragile`이다.
- 이번 작업은 로컬 `604644c`에 보존된 두 파일을 byte 그대로 복사했다.
  모델 구조·preset·tensor API를 이번 pipeline 작업자의 모델 설계로 표시하지 않는다.
- LF 기준 SHA256: transformer
  `60d56b6785e83c10d2dc313986189d93e8d223ebc031cc6054706986b63f4421`,
  tensor API `6e647775fa96726b26ca6f56e9ac4ef144ccd617c28d334e3baf664e7ad8e0ff`.
- `state_training/`는 기존 cached-state loss, bounded loader와 valid-record DDP
  가중치 구현을 참고한 새 계약이다. 이전 run·split·checkpoint를 소급 변경하지 않는다.
- 최신 main `55b117d`의 runtime/benchmark 분리와 팀원 manifests를 보존한다.
  이전 학습 branch 전체 diff와 benchmark 실행기를 이 PR에 옮기지 않았다.
