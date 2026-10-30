# CUDA 학습 준비 검증 결과 — 2026-10-30

코드 구현과 CPU 검증을 완료했다. 실제 CUDA 학습은 실행하지 않았다.
전체 데이터 준비 상태는 `full_data_ready=false`다.

## 구현

- v2 state-only 캐시의 checksum·source/target 정렬 검사, 원본 index 경로 매핑,
  worker당 2-case·64MiB cache의 record loader를 추가했다.
- 공통 FP32 AdamW trainer, train-only RMS, global batch 64, warmup/plateau,
  기록 수로 보정하는 DDP tail, validation 선택, epoch 경계 재개를 추가했다.
- FP64 streaming ridge 선택과 두 개선 모델의 affine skip 초기화를 추가했다.
  기존 payload와 epoch 27 가중치는 계속 읽는다.
- preflight·ridge-fit·cpu-smoke·evaluate·train CLI와 모델별 설정을 제공한다.
- 사용 기록 마스크와 presence/conditioning별 지표를 추가했다. 실제 번역 생략은 비활성이다.

## CPU 검증

환경: Python 3.10.20, PyTorch 2.13.0, SAM2의 기존 가상환경.

전체 테스트는 **191 passed, 1 skipped**였다. 생략한 항목은 기존 CUDA benchmark 테스트다.
검사에는 streaming/direct ridge 비교, ridge 초기 출력, 분기 기울기, 여러 batch 갱신,
완료 epoch 재개와 연속 실행의 정확한 가중치 일치, worker spawn, API·payload 호환성이 포함된다.
CPU Gloo 2-process에서 tail을 `[1,0]`으로 분배한 실행은 단일 process와 허용오차 내에서
일치했고 두 rank의 선택된 가중치는 정확히 같았다. Gloo 검사에는 로컬 loopback을 사용했다.

실행 명령:

```sh
GLOO_SOCKET_IFNAME=lo0 PYTHONPATH=src:/private/tmp/cmmt-module-testdeps \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ../sam2/.venv/bin/python -m pytest -q -rs
```

추가로 200GB 다운로드에서 MOSE/LVOS·fit/development를 포함한 실제 train 샘플을 확인했다.
새 validation 분할을 만들지 않았다.

| 모델 | Pair | 유효 기록 | Forward/backward·유한값 |
|---|---:|---:|---|
| `transformer` | 8 | 110 | 통과 |
| `residual_mlp` | 8 | 110 | 통과 |
| `mlp_context` | 8 | 110 | 통과 |

Affine 모델의 샘플 초기화는 고정 alpha의 CPU smoke 전용 해다. 공식 validation으로
선택된 ridge가 아니며 공식 학습 초기화로 사용할 수 없도록 다른 data identity를 기록했다.
샘플에서 모델을 갱신한 값은 validation 또는 benchmark 점수가 아니다.

Epoch 27 artifact SHA256은 `92802842b0f9c2247f95627784a4919625aaced43633c5203f619472ae0f17da`로 확인했다.
기존 모델 strict load와 `FIG7Tyb9` 실제 캐시 출력을 확인했고 이전 출력이 유지됐다.
공간 MSE는 0.09872704744338989, pointer MSE는 0.05874130129814148다.
이 샘플은 기존 학습 데이터에 속하며 독립 평가가 아니다.

## 전체 데이터 준비 상태

- 원본 index: 23,358 cases.
- 로컬 다운로드: 13,290 cases.
- 누락: train 9,354, official validation 714 cases.
- 공식 index 기준 실제 tensor 검사는 8 cases에서 수행했으며 손상은 발견하지 않았다.
- 전체 train RMS 재계산·전체 ridge 적합·공식 validation 재평가는 수행하지 않았다.

로컬 기계용 원시 증거는 `runs/cpu-readiness-20261030/`에 있다. 해당 폴더와 모델 가중치는
Git 제외 대상이다. `preflight.json`은 누락 case 목록을 포함하고 각 모델 폴더에는
`cpu_smoke_report.json`이 있다. 전체 데이터용 root-map과 실행 명령은
[state_training.md](state_training.md)에 설명했다.

## 남은 실행 검증

전체 원본 train과 LVOS official valid를 연결하고 full preflight·ridge 적합을 해야 한다.
그다음에 CUDA/NCCL, GPU peak memory, 처리량과 실제 학습을 검증한다.
후속 predictor logits 동등성·J&F는 아직 미검증이다. 현재 코드는 spatial 번역 생략을
활성화하지 않는다. FiLM·부재 전용 분기·새 attention 손실은 이번 구현에 포함하지 않았다.
