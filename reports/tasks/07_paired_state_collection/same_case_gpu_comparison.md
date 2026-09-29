# Same-case GPU comparison — 2026-09-29

동일한 MOSEv2 case `002b4dce / object 1 / switch 28 / 120 frames`를 모든 GPU에서
독립 실행했다. 동일 Small/Base+ checkpoint, config, prompt mask를 사용했다.

| GPU | 전체 | source inference | target inference | source+target inference |
|---|---:|---:|---:|---:|
| RTX 2000 Ada (59680) | 63.16 s | 6.49 s | 36.18 s | 42.66 s |
| RTX 2000 Ada (19066) | 107.40 s | 8.48 s | 39.01 s | 47.49 s |
| RTX 4000 Ada (22869) | 45.25 s | 3.72 s | 20.24 s | 23.96 s |
| RTX 4000 Ada (23201) | 45.18 s | 3.64 s | 19.97 s | 23.60 s |
| RTX 4000 Ada (23351) | 46.03 s | 3.62 s | 19.91 s | 23.53 s |
| RTX 4000 Ada (23557) | 45.06 s | 3.67 s | 19.99 s | 23.65 s |
| RTX 4000 Ada (25360) | 46.54 s | 3.63 s | 19.75 s | 23.38 s |
| RTX A4000 (21129) | 58.17 s | 5.08 s | 28.06 s | 33.14 s |

추론만 비교하면 RTX 4000 Ada는 RTX 2000 Ada보다 약 1.8배 빠르며, RTX A4000은
RTX 4000 Ada보다 약 1.4배 느리고 RTX 2000 Ada보다 약 1.3배 빠르다. RTX 2000
port 19066의 전체 시간은 model/video initialization이 약 57초로 비정상적으로 커서,
GPU 연산 속도 비교에는 사용하지 않는다.

결과 위치: RunPod `/workspace/CMMT-task07-artifacts/same_case_8gpu/mose_002b4dce_obj1_switch28/`.
