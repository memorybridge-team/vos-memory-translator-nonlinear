# DAVIS paired-state Direct/Ridge pilot — 2026-09-08

## 목적

실제 DAVIS 2017 영상에서 SAM 2.1 Tiny가 만든 canonical state를 SAM 2.1
Large의 native/oracle state에 맞추는 첫 비합성(실데이터) 점검이다. 이 단계는
translator의 입력·출력 구조와 학습 경로가 실제 checkpoint state에서 동작하는지
검증한다. 아직 번역된 state를 Large predictor에 주입해 미래 마스크를 평가하지
않았으므로 VOS 성능 실험은 아니다.

## 실행 환경과 데이터

- GPU: RunPod NVIDIA A40, 표시 VRAM 46,068 MiB
- 코드: `main`으로 이관, commit `e0f9466`
- SAM 2: 공식 checkout commit `2b90b9f5ceec907a1c18123530e92e794ad901a4`
- Dataset: 공식 DAVIS 2017 trainval 480p, RunPod
  `/workspace/CMMT/data/DAVIS`
- Source/target: SAM 2.1 Hiera Tiny → SAM 2.1 Hiera Large
- Object id: 1, switch frame: 6, seed: 7
- Train pair: DAVIS train `bear` frames 0–6
- Held-out pair: DAVIS train `bmx-bumps` frames 0–6
- Raw canonical `.pt`: RunPod `/workspace/CMMT/outputs/paired_states/`
  아래에만 보관하며 Git에는 포함하지 않음

두 모델의 canonical boundary는 이 pilot에서 모두 spatial memory
`[1, 1, 7, 64, 64, 64]`, object pointer `[1, 1, 7, 256]`, presence logits
`[1, 1, 7, 1]`로 정렬됐다. 이 shape 일치는 semantic equivalence를 보장하지 않는다.

## 결과

| Component | Direct MSE | Ridge MSE | Direct cosine | Ridge cosine |
|---|---:|---:|---:|---:|
| Spatial memory | 3.0086 | 0.8187 | 0.0365 | 0.8093 |
| Object pointer | 0.6973 | 0.4891 | -0.0215 | 0.4662 |
| Presence logits | 1.1686 | 4.0098 | 1.0000 | 1.0000 |

- Direct Copy: 0 parameters, median CPU translation latency 0.100 ms.
- Ridge: 69,954 parameters, median CPU translation latency 1.109 ms.
- 현재 세 component MSE의 단순 평균은 Direct 1.6248, Ridge 1.7726으로 Ridge가
  9.1% 나쁘다.
- 그러나 Ridge는 핵심 continuous state인 spatial memory와 object pointer를 각각
  크게 개선했고, presence logits만 악화했다.

## 해석과 한계

현재 aggregate MSE는 크기가 전혀 다른 세 component를 동일 비중으로 평균한다.
따라서 이 값 하나로 translator 우열을 정하면 안 된다. 이 결과는 component별
mapping 또는 presence의 Direct Copy/별도 보정이 필요할 가능성을 보여 주는
engineering smoke다.

학습 영상 1개와 held-out 영상 1개만 사용했으므로 일반화 성능을 주장할 수 없다.
또한 tensor-level MSE/cosine은 보조 지표일 뿐이다. 최종 판단은 번역 state를 실제
Large predictor에 주입한 뒤 switch 이후 DAVIS J&F, switch shock, identity break,
recovery length, latency와 VRAM으로 내려야 한다.

## 다음 gate

1. 저장된 Ridge를 다시 불러와 component별로 적용할 수 있게 한다.
2. `spatial=Ridge`, `pointer=Ridge`, `presence=Direct` hybrid와 전체 Direct를 실제
   Large continuation에 각각 주입한다.
3. 같은 held-out 영상에서 Large-native oracle, target reset, Last-Mask,
   replay-k를 포함해 미래 마스크 성능을 비교한다.
4. gate가 동작하면 여러 train/val sequence와 여러 switch frame으로 확장한다.
