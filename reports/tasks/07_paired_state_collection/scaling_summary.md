# Task 07 GPU scaling pilot — 2026-09-29

동일한 MOSEv2 fit case 8개를 state-only·active-memory collector로 실행했다.
각 조건은 별도 출력 경로와 telemetry를 사용했으며, cache 결과는 기존 pilot과 섞지 않았다.

| GPU 조건 | makespan | 1 GPU 대비 | 비고 |
|---|---:|---:|---|
| RTX 2000 Ada ×1 | 302.1 s | 1.00× | 기준선 |
| RTX 2000 Ada ×2 | 329.8 s | 0.92× | round-robin shard 불균형; GPU 수 비교로 사용하지 않음 |
| RTX 2000 Ada ×2 + RTX 4000 Ada ×2 | 110.9 s | 2.72× | 4 shard |
| 위 4장 + RTX 4000 Ada ×3 + RTX A4000 ×1 | 69.4 s | 4.35× | 8 shard |

8 GPU 조건은 4 GPU 대비 1.59배 빨라졌다. 8개 case의 가장 긴 shard가 69.4초였고,
짧은 shard는 20~26초였다. 따라서 병목은 GPU 수 자체보다 case 길이 불균형과 model/video
초기화 overhead의 영향을 받는다.

1 GPU phase 평균은 target inference 16.93초, source inference 4.07초, 두 video init
합계 7.72초, cache write/checksum 0.24초였다. 8 GPU에서도 cache write/checksum 평균은
0.14초로 작았다. 이 pilot만으로 storage 분리를 정당화할 근거는 없으며, production은
`num_frames`·`switch_frame`·pilot 시간을 이용한 weighted shard를 사용해야 한다.

상세 raw telemetry는 RunPod Network Volume의
`/workspace/CMMT-task07-artifacts/scaling_{1,2,4,8}gpu/mosev2/`에 있다.
