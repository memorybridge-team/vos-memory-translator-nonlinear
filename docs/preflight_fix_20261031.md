# DDP 시작·평가·결정성 수정 검증 — 2026-10-31

대상 기준 commit은 `22dc968`이며 수정 브랜치는 `fix/train-preflight-timeout`이다.
리뷰에 남겨진 `/private/tmp/cmmt-verify/timeout.log`에서 rank 1의 180000ms recv timeout을
확인했다. 원래 trainer가 rank-0 collective 안에서 전체 SHA/tensor/RMS 검사를 반복한
경로를 제거했다. 실제 CUDA 학습은 실행하지 않았다.

## 수정과 인터페이스

- `preflight` 보고서 v2는 index identity, 검증한 파일의 case ID/SHA/stamp와 train RMS를 기록한다.
- `train --preflight-report FILE`이 필수다. 각 process가 group 생성 전에 metadata/sidecar만
  대조한다. 시작·재개에 전체 tensor 검사는 없다. Dataset의 실제 load-time SHA 검사는 유지한다.
- 보고서 digest·index·RMS·stamp 변경, v1 보고서·부분 검사는 거부한다.
- `ridge-fit`도 보고서를 전달하면 중복 전체 검사를 생략한다. 기존 단일 process 호출은 지원한다.
- `evaluate`는 validation 데이터만 읽는다. Index 또는 검증 보고서의 train-only RMS를 사용하고,
  train 파일이 없어도 실행한다. Sample index에 train RMS가 없으면 보고서를 요구한다.
- 결정적 알고리즘, cuDNN benchmark off, DDP `broadcast_buffers=False`를 적용했다.
  CUDA 초기화 전에 `CUBLAS_WORKSPACE_CONFIG=:4096:8`을 설정하고 충돌은 거부한다.
- 실행 profile과 PyTorch 버전은 checkpoint identity에 포함한다. Profile이 없거나 다른 학습
  checkpoint의 재개는 거부한다. 기존 모델 가중치의 inference/state evaluation은 유지한다.

실행 예제와 기존 checkpoint 처리 규칙은 [학습 안내](state_training.md)에 있다.

## 자동 회귀

전체 결과: **201 passed, 1 skipped**. 생략한 항목은 기존 CUDA benchmark다.

```sh
GLOO_SOCKET_IFNAME=lo0 PYTHONPATH=src:/private/tmp/cmmt-module-testdeps \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ../sam2/.venv/bin/python -m pytest -q -rs
```

- 단일 process의 full preflight를 6초 지연시켰다. 이후 timeout 5초의 실제 `torchrun`
  2-rank CLI에서 첫 epoch와 완료 checkpoint 재개가 모두 통과했다. 학습에서 전체 검사를
  다시 호출하면 지연 후 실패하도록 별도 guard를 넣어 회귀를 검출한다.
- CPU Gloo에서 DDP 생성 인자 `broadcast_buffers=False`를 실제로 확인했다.
- 보고서 재사용 시 tensor 읽기가 없음을 검사했고, stamp가 같은 내용 손상도 실제 load-time
  SHA로 거부함을 확인했다. 변경된 stamp·잘못된 index·수정된 RMS·부분 보고서도 거부했다.
- Train 캐시를 제거하고 해당 경로의 stat을 금지한 조건에서 validation 평가가 통과했다.
  Index 정규화와 보고서 정규화 두 경로를 모두 확인했다.
- CUBLAS 환경 설정 순서는 GPU 연산 없이 검사했다. 변경/누락된 실행 profile의 재개를 거부했다.
- 기존 CPU 연속/재개 일치, 1/2-rank 결과, worker, ridge·affine 수치 회귀도 통과했다.

코드 수정 세 커밋마다 해당 회귀를 함께 추가한다. 이미 공유한 이전 5개 커밋은 재작성하지 않는다.

## 실데이터 CPU 추가 확인

리뷰에서 제공한 `small.json`의 8 cases / 109 records를 사용했다.
보고서를 만든 뒤 metadata 대조는 이 표본에서 약 0.000451초였다.
이 값은 전체 데이터나 네트워크 볼륨 속도의 예측값이 아니다.

- 새 보고서를 재사용한 CPU Gloo 2-rank Transformer 학습이 완료됐다.
- 이어서 validation 27 records를 평가했고 train cache 읽기는 0회였다.
- 표본의 개발 영상은 원래 1차 학습의 train에 포함됐다. 이 결과는 독립 validation 또는
  VOS/J&F 성능 증거가 아니다. 원본 cache와 리뷰 스크립트는 변경하지 않았다.
- 원시 결과와 보조 스크립트는 Git 제외 폴더 `runs/preflight-fix-20261031/`에 보관했다.

## 유지한 정책과 남은 검증

Affine skip의 weight decay는 원래 비교 정책대로 유지했다. Ridge를 중심으로 고정하거나
새 정규화 실험을 추가하지 않았다. `broadcast_buffers`의 PyTorch 2.13 deprecation 경고는
이전 버전 및 1차 실행과 같은 초기 동기화 의미를 유지하기 위해 그대로 둔다.

CUDA/NCCL·전체 원본 데이터의 메타데이터 검사 시간·GPU 처리량·GPU별 비트 일치와
후속 predictor/J&F는 아직 검증하지 않았다. 이번 수정의 분산 실행 증거는 CPU Gloo다.
