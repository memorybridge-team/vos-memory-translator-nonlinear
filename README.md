# Cross-Model Memory Translator (CMMT)

비디오를 처리하던 모델을 중간에 다른 모델로 바꿀 때, 새 모델이 과거 프레임을
처음부터 다시 읽지 않도록 **기존 모델의 temporal memory/state를 새 모델용
state로 번역하는 연구 프로젝트**입니다.

```text
SAM 2 Tiny가 frame 1…t 처리
          │
          ▼
   source memory b_t
          │
          ▼
      Translator T
          │
          ▼
 Large-compatible state â_t
          │
          ▼
Large가 frame t+1부터 계속 추론
```

첫 controlled pair는 **SAM 2.1 Tiny → Large**입니다. 최종 목표는 같은 SAM 2
계열을 넘어 **SAM 2 ↔ XMem/Cutie**처럼 memory 구조가 다른 모델 사이의
runtime handoff를 검증하는 것입니다.

## 지금 어디까지 되었나요?

- 공식 SAM 2.1 Tiny/Large checkpoint와 DAVIS 2017을 RunPod A40에서 실행했습니다.
- SAM 2 memory를 continuation 가능한 canonical state로 저장하고 다른 predictor에
  주입하는 경로를 구현했습니다.
- 같은 checkpoint의 export→inject round-trip에서 과거 프레임 backbone replay가
  0회인지 확인했습니다.
- Direct Copy와 component-wise Ridge translator를 실제 Tiny→Large handoff에
  주입했습니다.
- 제한적인 한 개 held-out pilot에서 switch 이후 DAVIS J&F는
  Large-native `0.8630`, Direct `0.6385`, Ridge hybrid `0.8605`였습니다.

이 수치는 **DAVIS 전체 benchmark 결과가 아닙니다.** `bear` 한 영상의 작은 학습
표본과 `bmx-bumps` 한 영상·한 객체·한 switch point만 사용한 pilot입니다.
현재 결론은 “실제 state injection과 평가 pipeline이 동작한다”까지이며,
Translator의 일반화나 우월성은 아직 검증되지 않았습니다.

## 결과 직접 보기

- [연구 결과 홈페이지](https://memorybridge-team.github.io/vos-memory-translator-nonlinear/)
- [83프레임 인터랙티브 갤러리](https://memorybridge-team.github.io/vos-memory-translator-nonlinear/experiments/2026-09-08-davis-handoff/)
- [현재 정식 실험 계획](docs/experimental_plan.md)
- [첫 DAVIS handoff pilot 보고서](reports/experiments/2026-09-08_davis_handoff_pilot/report.md)
- [RunPod 실행 기록](reports/run_logs/2026-09-08_runpod.md)

갤러리는 각 프레임에서 `DAVIS GT / Large-native / Direct / Ridge hybrid`를
나란히 보여주고, 전체 영상과 frame별 J&F 그래프를 제공합니다.

## 앞으로의 핵심 실험

1. 여러 DAVIS train/validation 영상·객체·switch point를 고정 manifest로 구성
2. Target Reset, Last-Mask, Replay-k, Full Replay baseline 완성
3. leakage 없이 Tiny/Large paired-state 학습 데이터를 확대
4. Ridge → component-wise MLP → downstream/rollout loss 순으로 비교
5. switch+1/5/20, identity break, occlusion recovery, latency·VRAM·전송량 평가
6. MOSE/LVOS의 hard·long-term 조건과 반복 switch로 확장
7. SAM 2↔XMem/Cutie cross-architecture handoff 검증

RunPod 비용 때문에 연구에 필요한 데이터나 반복 횟수를 줄이지 않습니다. 다만
GPU는 checkpoint inference/state 수집/학습에 집중하고, 전처리·그래프·갤러리는
CPU에서 수행해 불필요한 자원 낭비를 막습니다.

## 성공 판단 기준

Tensor MSE가 낮은 것만으로 성공이라 부르지 않습니다. 다음을 함께 봅니다.

- post-switch DAVIS J&F와 switch 직후 성능 저하
- 객체 identity 유지와 가림 후 recovery
- Last-Mask 및 Replay-k 대비 정확도–지연시간 Pareto 개선
- full replay 대비 handoff latency, FLOPs, VRAM과 전송 bytes 절감
- 새로운 영상·switch 시점·반대 방향에서의 일반화

## 저장소 구성

| 경로 | 내용 |
|---|---|
| `src/vos_memory_inspector/` | state 추출·검증·주입, translator, metric 핵심 코드 |
| `tests/` | synthetic unit/integration test |
| `scripts/` | RunPod 준비, probe, 결과 갤러리 생성 명령 |
| `configs/` | 저장소가 관리하는 실험 설정 |
| `docs/design/` | state contract와 Translator 설계 근거 |
| `docs/experimental_plan.md` | 현재 유효한 정식 실험 계획 |
| `docs/storage_pipeline.md` | GPU를 기다리게 하지 않는 hot/cold 저장 정책 |
| `reports/experiments/` | 재현 가능한 pilot metric과 대표 시각 자료 |
| `reports/run_logs/` | 실제 실행 시간·명령·성공/실패 기록 |
| `references/` | 논문 링크와 출처 인덱스; 원문 PDF는 Git에서 제외 |
| `data/`, `outputs/`, `checkpoints/` | 로컬/RunPod 전용이며 Git에서 제외 |

## 설치와 실행 준비

공식 SAM 2는 이 저장소 안에 복사하지 않고 별도 checkout으로 설치합니다. 지원
revision은 [`SAM2_UPSTREAM_COMMIT`](SAM2_UPSTREAM_COMMIT)에 고정돼 있습니다.

```bash
git clone https://github.com/facebookresearch/sam2.git
git -C sam2 checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
python -m pip install -e ./sam2

git clone https://github.com/memorybridge-team/vos-memory-translator-nonlinear.git
cd vos-memory-translator-nonlinear
git switch main
python -m pip install -e ".[dev]"
python -m pytest -q
```

RunPod 구성과 실제 checkpoint 명령은 [RunPod 실행 가이드](docs/runpod.md), state
구조와 검증 근거는 [memory tensor inventory](docs/memory_tensor_inventory.md),
실행된 테스트는 [validation 기록](docs/validation.md)을 참고하세요.

## Git에 포함하지 않는 것

- SAM 2 checkpoint와 학습 checkpoint
- DAVIS/MOSE/LVOS 원본 데이터
- raw canonical state와 대량 tensor dump
- 개인 SSH key, 환경변수와 계정 정보
- 제3자 논문 PDF 및 자동 추출한 논문 전문

Git에는 코드, 설정, 작은 metric 보고서, 재현 기록과 공개 검토용으로 선별·압축한
시각화만 저장합니다. DAVIS를 사용한 공개 결과에는
[dataset attribution](docs/DATA_ATTRIBUTION.md)을 표시합니다.
