# 문서 인덱스

## Verified implementation documents

- [`experimental_plan.md`](experimental_plan.md): 현재 유효한 정식 실험 계획과 Codex/RunPod 운영 원칙
- [`storage_pipeline.md`](storage_pipeline.md): GPU를 기다리게 하지 않는 hot/warm/cold 저장·입출력 정책
- [공개 83-frame pilot gallery](https://memorybridge-team.github.io/vos-memory-translator-nonlinear/experiments/2026-09-08-davis-handoff/): GT와 세 방법의 영상·frame별 J&F 검토
- [공개 cached baseline suite](https://memorybridge-team.github.io/vos-memory-translator-nonlinear/experiments/2026-09-10-bike-packing-baselines/): Direct, Reset, Last-Mask, Replay-1/2/4, Full Replay의 53-frame 선택형 비교
- [공개 reappearance hard-case gallery](https://memorybridge-team.github.io/vos-memory-translator-nonlinear/experiments/2026-09-10-india-reappearance-baselines/): 물체가 사라진 switch에서 Last-Mask와 Replay-2/4의 회복 차이를 보여 주는 44-frame 비교
- [`memory_tensor_inventory.md`](memory_tensor_inventory.md): pinned SAM 2 upstream에서 확인한 temporal-memory producer, storage, consumer 경로
- [`validation.md`](validation.md): 현재 memory-inspection milestone의 실행·검증 기록

## Design

- [`design/C_TRANSLATOR_METHOD_REPORT.md`](design/C_TRANSLATOR_METHOD_REPORT.md): SAM 2.1 Tiny↔Large 기반 translator 구조, state contract, 학습 단계, ablation과 역할 분담 설계
- [`design/CROSS_MODEL_KV_TO_SAM2_IMPLEMENTATION_REPORT.md`](design/CROSS_MODEL_KV_TO_SAM2_IMPLEMENTATION_REPORT.md): Cross-Model KV Cache Transfer의 정확한 tensor/mapping 분석, SAM 2 adaptation, 구현 파일, synthetic 실행 결과와 실제 checkpoint blocker
- [`design/SAM2_TRANSLATOR_EXPERIMENT_PLAN.md`](design/SAM2_TRANSLATOR_EXPERIMENT_PLAN.md): 2026-09-07 초기 계획 보관본; 현재 계획으로 사용하지 않음

새 설계 문서는 주제별 Markdown 파일로 `docs/design/`에 추가합니다. 실험 결과는 코드·설정·원시 metric과 연결되는 형태로 기록하고, 연구의 확정된 결정은 루트 `PROJECT_CONTEXT.md`에도 반영합니다.
