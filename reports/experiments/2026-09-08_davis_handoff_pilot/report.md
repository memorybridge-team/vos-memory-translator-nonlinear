# DAVIS Tiny→Large Ridge-hybrid handoff pilot — 2026-09-08/09

## 한눈에 보는 결론

DAVIS 2017 train의 `bear` 한 영상에서 맞춘 69,952-parameter hybrid translator를
학습에 사용하지 않은 `bmx-bumps`에 실제 주입했다. Hybrid는 spatial memory와
object pointer에 Ridge를 적용하고 presence logit은 Tiny 값을 그대로 사용한다.

Switch frame 6 이후 frame 7–88의 부분구간에서 hybrid는 Direct Copy보다 DAVIS
정답 J&F가 0.2219 높았고, Large-native와의 차이는 0.0025였다. 이는 매우 유망한
engineering pilot이지만 한 개 train/한 개 test sequence 결과이므로 일반화나
benchmark 성능을 주장하지 않는다.

## 재현 조건

- RunPod GPU: NVIDIA A40, 46,068 MiB 표시 VRAM
- CMMT code: branch `main`으로 이관, commit `18a1ecf`
- Official SAM 2 checkout: `2b90b9f5ceec907a1c18123530e92e794ad901a4`
- Official DAVIS evaluation: [`davisvideochallenge/davis2017-evaluation`](https://github.com/davisvideochallenge/davis2017-evaluation)
  commit `ac7c43fca936f9722837b7fbd337d284ba37004b`
- Source/target: SAM 2.1 Hiera Tiny → SAM 2.1 Hiera Large
- Translator fit: DAVIS train `bear`, object 1, frames 0–6, seed 7
- Held-out handoff: DAVIS train `bmx-bumps`, object 1, switch frame 6, seed 7
- Evaluated future: frames 7–88. DAVIS semi-supervised protocol처럼 video last
  frame 89는 제외했다. 그러나 switch 이전도 제외했으므로 full benchmark가 아닌
  single-object partial-sequence score다.

## 결과 1 — Large-native와의 handoff fidelity

| Candidate | Mean binary IoU to Large-native | Mean logit MSE | First future frame IoU | Wall time | Peak CUDA memory |
|---|---:|---:|---:|---:|---:|
| Direct Copy | 0.5616 | 121,852.47 | 0.2460 | 90.89 s | 1.660 GB |
| Ridge hybrid | 0.9494 | 1.8787 | 0.8582 | 91.91 s | 1.660 GB |

두 방법 모두 injection 이전과 도중의 target backbone 호출은 0회였고, frame
7–89의 future continuation에서만 83회 호출했다. 즉 과거 프레임을 target으로
replay하지 않았다. 표의 wall time은 source prefix, target-native oracle, candidate
continuation과 artifact 생성이 포함된 전체 실험 시간이라 translator latency만을
뜻하지 않는다.

## 결과 2 — DAVIS annotation 기준 부분구간 J&F

| Method | Mean J | Mean F | Mean J&F | Direct 대비 J&F | Large-native 대비 J&F |
|---|---:|---:|---:|---:|---:|
| Large-native | 0.7721 | 0.9539 | 0.8630 | +0.2245 | — |
| Direct Copy | 0.4815 | 0.7956 | 0.6385 | — | -0.2245 |
| Ridge hybrid | 0.7669 | 0.9540 | 0.8605 | +0.2219 | -0.0025 |

`J`는 region IoU, `F`는 boundary F-measure다. 공식 DAVIS metric 함수를 그대로
호출했지만, 이 표는 `bmx-bumps` object 1의 switch 이후 82프레임만 평가한
pilot이므로 DAVIS validation benchmark 점수로 인용하면 안 된다.

## 대표 프레임 육안 확인

현재 저장된 이미지는 왼쪽부터 Input, Large-native(초록), candidate(보라), 겹침을
보여 준다. 이후 생성되는 비교 이미지는 Input 대신 DAVIS GT(빨강)를 원본 위에
표시한다. 이 pilot의 기존 PNG는 재생성 전 산출물이므로 아래 링크에는 아직
Input이 남아 있다.

| Frame | Direct Copy | Ridge hybrid | 관찰 |
|---|---|---|---|
| 7 | [이미지](direct/comparisons/frame_00007.png) | [이미지](ridge_hybrid/comparisons/frame_00007.png) | 전환 직후 native IoU 0.2460→0.8582, GT J&F 0.5527→0.8729 |
| 44 | [이미지](direct/comparisons/frame_00044.png) | [이미지](ridge_hybrid/comparisons/frame_00044.png) | 둘 다 화면 왼쪽의 일부 object만 보이는 어려운 구간 |
| 50 | [이미지](direct/comparisons/frame_00050.png) | [이미지](ridge_hybrid/comparisons/frame_00050.png) | Hybrid는 Large-native와 동일하게 빈 mask지만 세 방법 모두 GT J&F 0 |
| 78 | [이미지](direct/comparisons/frame_00078.png) | [이미지](ridge_hybrid/comparisons/frame_00078.png) | 재등장 구간 native IoU 0.3449→0.8589, GT J&F 0.5402→0.7545 |

Frame 50은 중요한 반례다. Large-native를 모방하는 것만으로 항상 DAVIS 정답에
도달하지 않는다. 따라서 state reconstruction/native agreement는 보조 지표이고
ground-truth J&F가 최종 지표여야 한다.

## 산출물

- 공개 검토 페이지: [전체 83-frame 인터랙티브 갤러리](https://memorybridge-team.github.io/vos-memory-translator-nonlinear/experiments/2026-09-08-davis-handoff/)
- 페이지에는 GT / Large-native / Direct / Ridge-hybrid 비교 영상, frame slider와
  frame별 J&F 그래프가 포함된다.

- `direct/report.json`: frame별 Large-native agreement와 실행 자원
- `ridge_hybrid/report.json`: frame별 hybrid agreement와 실행 자원
- `davis_ground_truth/*.json`: frame별 공식 J/F 함수 결과
- 각 candidate의 `comparisons/`: Git에 넣은 대표 4-panel PNG
- 전체 83-frame masks/comparisons와 11MB canonical states는 RunPod
  `/workspace/CMMT/outputs/`에만 보관

## 아직 말할 수 없는 것

- 다른 sequence/object/switch에서도 재현되는지
- validation split 평균 J&F와 identity/occlusion recovery 통계
- Last-Mask, target reset, replay-k보다 나은지
- 한 개 state pair가 아닌 대규모 학습에서 최적 translator 구조가 무엇인지
- Tiny→Large 이외의 Large→Tiny 또는 SAM 2↔XMem/Cutie 결과

다음 실험 gate는 여러 train sequence/switch pair로 학습 데이터를 늘리고, 고정된
DAVIS val subset에서 Direct, hybrid, reset, Last-Mask, replay-k, Large-native를
동일하게 비교하는 것이다.
