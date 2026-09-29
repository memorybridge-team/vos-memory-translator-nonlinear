# Weighted-shard scaling — 2026-09-29

동일한 MOSEv2 fit case 24개를 사용했다. `frame_count <= 200`으로 초장기 outlier를
제외하고, `frame_count + switch_frame`와 GPU별 측정 speed factor를 이용해 LPT
weighted assignment를 적용했다. 각 worker는 배정된 case를 순차 처리했다.

| 조건 | worker case-wall 합 | 가장 느린 worker | 4 GPU 대비 |
|---|---:|---:|---:|
| RTX 2000 Ada ×2 + RTX 4000 Ada ×2 | 1,146.9 s | 430.7 s | 1.00× |
| RTX 2000 Ada ×2 + RTX 4000 Ada ×5 + RTX A4000 ×1 | 984.6 s | 169.0 s | 2.55× |
| RTX 2000 Ada ×2 + RTX 4000 Ada ×6 + RTX A4000 ×4 | 992.1 s | 142.4 s | 3.03× |

12 GPU는 8 GPU보다 가장 느린 worker 기준 약 1.19배 빨랐다. 12 GPU의 병목은 두 RTX
2000 Ada worker 중 느린 worker였고, 나머지 고성능 GPU는 먼저 종료했다.

이번 assignment는 영상 길이를 반영했지만, 실제 처리시간은 frame 수 외에도 해상도,
object 복잡도, model/video 초기화 편차의 영향을 받았다. 따라서 production에는
고정 weighted shard보다 완료한 worker가 다음 case를 가져가는 dynamic queue 또는
pilot에서 측정한 case별 실제 wall time을 이용한 재배정이 더 적합하다.

모든 조건에서 state-only cache write/checksum은 inference보다 훨씬 작았으므로,
현재 evidence만으로 dataset별 storage 분리를 적용하지 않는다.

Raw 결과:

- `/workspace/CMMT-task07-artifacts/weighted_4gpu/mosev2/`
- `/workspace/CMMT-task07-artifacts/weighted_8gpu/mosev2/`
- `/workspace/CMMT-task07-artifacts/weighted_12gpu/mosev2/`
