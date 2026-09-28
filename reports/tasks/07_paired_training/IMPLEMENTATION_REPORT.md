# Paired-state·학습 파이프라인 구현 보고서

날짜: 2026-09-28. 검토용 로컬 branch: `task/training-pipeline-small-baseplus`.
최신 확인한 main: `9128e6d`. 원격 push/PR 게시 없음.

## 단계별 상태

| 단계 | 구현·CPU 검증 | 실제 GPU gate |
|---|---|---|
| 1 수집 | Small/Base+ 모델/config/hash, video/prompt content hash, registry/timeline/discrete alignment; prefix 경계와 controlled 빈 mask 테스트 | weights-backed collection 미실행 |
| 2 영속 storage | 원자적 shard/manifest, checksummed safe load, 한 writer lock, resume 시 완료 shard read 검증 | 실제 KNSW_DATASET restart 검증 미실행 |
| 3 fit/dev loader | 영상별 disjoint, 기존 frozen video manifests 소비, native/controlled 분리, official val/external 거부; LVOS sparse frame 및 MOSE train prompt indexing 검증 | 실제 train shard read 미실행 |
| 4 runner | 외부 factory와 learned Linear, frozen parameter helper, spatial/pointer valid loss, fit-only RMS, JSON metrics, optimizer/RNG/checkpoint resume | real nonlinear factory 미제공, real-GPU training 미실행 |
| 5 smoke | synthetic one-video overfit → video-disjoint dev → resume exact parity → checkpoint의 실제 injector 함수 연결 | real one-video/dev/rollout, GT VOS 평가 미실행 |

기존 전체 테스트 57개에 pipeline 회귀 테스트 22개를 추가한 suite를 사용한다.
최종 test output 및 synthetic 실행 JSON은 검토용 산출물에 포함한다.

## 관측값과 해석

2026-09-28 CPU synthetic `smoke-v2`:

- One-video fit probe normalized loss: `2.8332686424 → 0.0004315791` (80 epochs).
- Video-disjoint dev, 6 epochs: spatial MSE `0.3013922274`, pointer MSE `0.0905203670`, normalized loss `1.0811694145`, valid records `10`.
- 3+3 epoch resume와 uninterrupted 6 epoch weights exact 일치.
- checkpoint 로드 → 실제 `inject_sam2_canonical_state`: 객체 2개, record 10개, switch 5. predictor/PE는 synthetic fixture.
- wall time `46.26 s`, shard 3개 `31,467 bytes`, 모든 smoke checkpoint `2,404,480 bytes`.

synthetic grid는 `C=4,H=W=3,D=6`이다. 이 storage/시간을 실제 SAM 2 상태나 GPU 요금으로 추정하지 않는다.
최종 소스의 clean-commit smoke 수치는 별도 검토용 JSON을 우선한다.
state loss 감소는 VOS 개선의 근거가 아니다. runtime rollout 현재 출력은 Target-native agreement이며 GT benchmark가 아니다.

## 남은 입력

1. 현재 Pod의 실제 host/port와 개인 SSH key 등록 상태. mount·데이터·checkpoint root 경로는 사용자가 제공했다.
2. 원격 checkpoint filenames/hash, dataset inventory 및 pinned SAM 2 checkout의 실재 확인.
3. 모델팀 `module:factory`, constructor kwargs, `forward(CanonicalState) -> CanonicalState`, context batching 및 checkpoint source/version 계약.
4. 실제 dev rollout/GT evaluator 연결 및 GPU storage·VRAM·restart 관측.

실행 명령, factory 계약, 재시작 검증, 출력 경로는 [training pipeline](../../../docs/training_pipeline.md)에 있다.
attention/logit distillation은 위 기본 real rollout gate 뒤에 추가한다.
