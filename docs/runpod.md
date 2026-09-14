# RunPod GPU 실행 가이드

이 프로젝트는 RunPod 비용 때문에 연구에 필요한 데이터·baseline·반복 수를
축소하지 않습니다. 대신 GPU와 CPU 작업을 분리하고, 작은 smoke에서 실행 오류를
먼저 제거한 다음 동일한 protocol로 정식 실험을 확장합니다.

## 현재 검증된 환경

- GPU: NVIDIA A40 48GB
- Persistent volume: `/workspace`
- 프로젝트: `/workspace/CMMT`
- 공식 SAM 2 checkout: `/workspace/CMMT/.external/sam2`
- SAM 2.1 Tiny/Large checkpoint: `/workspace/CMMT/checkpoints/`
- DAVIS 2017: `/workspace/CMMT/data/DAVIS`
- 원시 실행 결과: `/workspace/CMMT/outputs/`

Checkpoint, dataset, raw state와 SSH key는 Git에 올리지 않습니다.

## 새 Pod 권장 사양 — 2026-09-09

### 기본 권장: GPU를 바꿔도 연구 자산을 유지

- Cloud: **Secure Cloud**
- GPU: **L40S 48GB**
- Container disk: **50GB**
- Network Volume: **500GB**, `/workspace`
- 별도 Volume disk: 사용하지 않음. Network Volume이 `/workspace`를 대체함

500GB는 현재 3.8GB working set뿐 아니라 DAVIS 전체 case 확대, 여러 switch와
baseline의 mask/state, Translator checkpoint, 이후 MOSE/LVOS의 활성 subset을
동시에 둘 여유를 확보하기 위한 크기다. Dataset 원본과 모든 중간 산출물을 장기
누적할 계획이면 1TB를 사용한다. Network Volume은 나중에 늘릴 수 있지만 줄일 수
없다.

Network Volume은 Pod와 독립적이어서 GPU를 바꾸거나 Pod를 삭제해도 유지되지만,
Secure Cloud에서만 사용할 수 있고 해당 데이터센터의 GPU만 선택할 수 있다.
Network I/O가 GPU를 기다리게 하지 않도록 현재 case의 frame shard와 반복 로드할
checkpoint는 실행 전에 50GB container의 local temporary cache로 복사하고, 완료된
bundle을 `/workspace`에 atomic move한다.

### 가격 우선 대안: 같은 Pod를 계속 사용할 때

- Cloud: **Community Cloud**
- GPU: **RTX A6000 48GB** 또는 **A40 48GB**
- Container disk: **50GB**
- Volume disk: **500GB**, `/workspace`

Volume disk는 빠른 local storage라 별도 staging이 필요 없지만 Pod 삭제 시 함께
삭제되고 다른 Pod에 연결할 수 없다. 이 구성을 쓰면 완료 bundle과 checkpoint를
Pod 종료 전에 외부 object storage에 checksum과 함께 복제한다.

### Disk와 Volume의 역할

| 종류 | 권장 크기 | 역할 | 보존 범위 |
|---|---:|---|---|
| Container disk | 50GB | OS, build cache, 현재 case의 임시 hot cache | stop/restart 시 삭제 |
| Volume disk | 500GB 대안 | 단일 Pod의 dataset/checkpoint/output | stop에는 유지, Pod 삭제 시 삭제 |
| Network Volume | 500GB 기본 | GPU 교체 가능한 persistent `/workspace` | Pod와 독립적으로 유지 |
| 외부 object storage | 필요량만 | 종료된 대형 state/result의 장기 cold archive | RunPod와 독립 |

따라서 **Container disk만 크게 잡는 구성은 사용하지 않는다.** Network Volume을
선택하면 Volume disk는 필요 없고, Network Volume을 쓰지 못하는 Community Pod에서만
Volume disk를 사용한다. RunPod 공식 문서 기준 storage 요금은 Container disk와
running Volume disk가 `$0.10/GB/month`, stopped Volume disk가
`$0.20/GB/month`, Network Volume 첫 1TB가 `$0.07/GB/month`다. 즉 500GB
Network Volume은 약 `$35/month`이며 GPU 요금과 별도다.

출처: [RunPod storage types](https://docs.runpod.io/pods/storage/types),
[RunPod network volumes](https://docs.runpod.io/storage/network-volumes)

## GPU 선택 — 2026-09-09 조회

아래는 RunPod 공식 GPU Models 페이지에 표시된 Community Cloud 시작 가격과
RunPod가 2026-08-10 확인했다고 밝힌 Secure Cloud 가격이다. 실제 배포 화면의
가격·가용성이 최종 기준이다.

| GPU | VRAM | Community 시작가 | Secure 가격 | 이 연구에서의 판단 |
|---|---:|---:|---:|---|
| RTX 3090 | 24GB | $0.22/h | $0.50/h | smoke 가능, 정식 학습은 VRAM 여유 부족 |
| RTX 4090 | 24GB | $0.34/h | $0.74/h | 빠르지만 24GB가 확장 시 제약 |
| RTX A6000 | 48GB | $0.33/h | $0.53/h | 저비용 정식 반복 실험 1순위 |
| A40 | 48GB | $0.35/h | $0.49/h | 이미 검증됨; A6000과 함께 비용 효율 선택 |
| L40S | 48GB | $0.79/h | $1.09/h | **기본 권장**; 48GB와 높은 AI/video 처리량 |
| A100 PCIe | 80GB | $1.19/h | $1.59/h | OOM/high-batch 또는 SAM 2 fine-tuning 때 승격 |
| H100 PCIe | 80GB | $1.99/h | $2.89/h | 현재 작은 Translator에는 과투자 |

현재 연구의 무거운 부분은 SAM 2 Tiny/Large forward와 후속 video rollout이고,
학습 대상 Translator 자체는 SAM 2 전체보다 훨씬 작다. 따라서 48GB L40S가
처리량과 여유의 균형점이다. A40 pilot의 peak VRAM은 약 1.66GB였으므로 현재
inference만 보면 A40/A6000도 충분하다. 반면 SAM 2 전체 fine-tuning은 공식 예제가
80GB A100 8장을 전제로 하므로, 그 단계는 Translator 학습과 별도의 multi-GPU
실험으로 계획한다.

출처: [RunPod GPU models](https://www.runpod.io/gpu-models),
[RunPod Secure Cloud price table](https://www.runpod.io/articles/guides/ai-server-cost),
[NVIDIA A40 specs](https://www.nvidia.com/en-us/data-center/a40/),
[NVIDIA L40S specs](https://www.nvidia.com/en-us/data-center/l40s/),
[NVIDIA A100 specs](https://www.nvidia.com/en-au/data-center/a100/),
[SAM 2 training guide](https://github.com/facebookresearch/sam2/blob/main/training/README.md)

## 최초 구성

```bash
cd /workspace
git clone --branch main \
  https://github.com/memorybridge-team/vos-memory-translator-nonlinear.git CMMT
cd /workspace/CMMT
bash scripts/runpod_bootstrap.sh /workspace/CMMT
```

기존 checkout에서는 새 실험 전에 작업 브랜치를 갱신하고 전체 test를 실행합니다.

```bash
cd /workspace/CMMT
git switch main
git pull --ff-only
.venv/bin/python -m pytest -q
```

## 자원 사용 원칙

GPU에서 수행:

- 공식 checkpoint inference
- Tiny/Large paired-state 수집
- MLP/attention Translator 학습
- downstream rollout과 latency/VRAM 측정

CPU에서 수행:

- dataset manifest와 split 검증
- Ridge/OLS fit이 메모리에 맞는 경우
- DAVIS metric 집계
- 그래프, MP4와 HTML gallery 생성
- 보고서·Git 산출물 구성

GPU가 0%여도 Pod가 켜져 있으면 요금이 발생할 수 있으므로 실행 상태는 기록합니다.
그러나 비용을 줄이려고 필요한 실험 case를 제거하지는 않습니다.

학습 중 외부 object storage의 JPEG/PNG를 직접 읽지 않습니다. 현재 case의 dataset과
checkpoint를 local hot tier에 먼저 준비하고 GPU 작업이 끝난 뒤 결과를 persistent
`/workspace`와 외부 저장소로 올리는 hot/cold 구조를 사용합니다. 자세한 기준은
[`storage_pipeline.md`](storage_pipeline.md)를 따릅니다.

## 실행 순서

1. 1개 case smoke로 경로·checkpoint·state contract·artifact 생성을 확인합니다.
2. 고정 manifest 전체에서 Target Reset, Last-Mask, Replay-k, Full Replay,
   Direct Transfer를 같은 evaluator로 실행합니다.
3. train video에서 paired state를 수집하고 validation/test video와 분리합니다.
4. Ridge와 component-wise MLP를 학습합니다.
5. post-switch J&F, identity, recovery, latency·VRAM·bytes를 집계합니다.
6. 전체 영상 gallery와 실패 case를 검수한 뒤 보고서를 Git에 반영합니다.

현재 유효한 전체 계획과 Go/No-Go 기준은
[`experimental_plan.md`](experimental_plan.md)를 따릅니다.

## Git에 가져올 결과

- 실행 config, seed, upstream/checkpoint 식별자
- case별 raw metric JSON과 요약 Markdown
- 대표 실패·성공 PNG
- 공개 검토용으로 압축한 MP4/HTML gallery
- 재현 명령과 알려진 한계

대량 binary mask, raw state tensor, dataset과 checkpoint는 RunPod persistent
volume에 보관하고 Git에는 위치·hash·schema만 기록합니다.

## 현재 공개 Pilot

2026-09-08 Tiny→Large DAVIS pilot의 전체 83-frame 결과는
[GitHub Pages gallery](https://memorybridge-team.github.io/vos-memory-translator-nonlinear/experiments/2026-09-08-davis-handoff/)에서
확인할 수 있습니다. 이는 한 영상·한 객체·한 switch의 제한적 pilot이며 전체
DAVIS benchmark 결과가 아닙니다.
