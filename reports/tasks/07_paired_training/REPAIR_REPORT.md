# ZIP paired-state collector repair — 로컬 검토본

2026-09-29. 기준 `5be4fba5ec2e7df3c95178639f02b3078388b0f3`, task branch
`fix/paired-state-collection-contract`. 원격 게시·live worker 수정·GPU 실행은 하지 않았다.

## 확인된 계약과 수정

Small→Base+ 한 방향·한 번, Target은 t+1에서 시작하는 기존 runtime 계약을 유지한다.
O=1 independent native-history와 provisional active-memory policy를 명시한다.
`cmmt.small_to_base_plus.io.v1.1`의 spatial/pointer 학습, metadata 보존,
diagnostic presence 및 Target PE regeneration을 유지한다.

`paired_collection_cli`는 ZIP production case 경로의 격리된 replacement다. Late prompt와 switch를
JPEG 목록에서 독립적으로 해석하고 양 모델 propagation을 switch까지 제한한다.
모든 pair와 checksum/생성 조건을 검증한 뒤 skip하며 원본을 덮어쓰지 않는다.
검증된 trusted-team legacy cache만 새 tensor/dict shards와 exact fit/dev manifests로 CPU import한다.
Policy/object semantics/generating producer가 다르면 자동 pooling하지 않는다.

## 검증

```powershell
$env:PYTHONPATH='src;.'
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp 'C:\Users\SAMSUNG\Documents\Codex\rpr-delivery-tests' --tb=short
```

전체 **107 passed in 49.08s**. Sparse mapping, late insertion, prefix boundary, alignment,
tampering/conditions, locks/interruption, legacy import/idempotence, policy mixing,
frozen splits, 8-way union, loader·optimizer·checkpoint 계약을 포함한다.
위 temp 경로는 실행 증거이며 재실행할 때는 새 전용 경로를 지정한다.

실제 synthetic CLI audit/import/verify에는 38.131초, imported shards 2개에는 16,754 bytes,
Linear 1 epoch에는 0.026764초, checkpoint에는 21,424 bytes가 관측되었다.
Fixture 생성 시 명시한 synthetic provenance이며 실제 SAM 모델/배포 provenance가 아니다.
MOSE 20,841·LVOS 1,803 선택을 8개 modulo shards에 중복 없이 전체 포함했다.
Production throughput/storage/요금은 측정하지 못했다. 8 GPUs·3일은 사용자 전달값이다.

## 운영 문서와 남은 gate

[운영 지침](../../../docs/paired_state_repair.md)에 실제 CLI flags, audit/import/verify,
격리 환경 설치, dry-run, new namespace, persistence 및 제한 GPU gate가 있다.
실제 completed cache 검사 수 0이며 production 재사용/재수집 수는 판단 대기다.
Current endpoint/read-only 승인, topology/PID/source hashes/run 기록 및 trusted completed sample이 필요하다.
Actual GPU continuation parity와 checkpoint-backed no-replay rollout은 실행되지 않았다.
State loss를 VOS 성능 개선으로 주장하지 않는다.

기존 training factory는 `factory(source_spec=..., target_spec=..., **kwargs) -> nn.Module`,
`forward(CanonicalState) -> CanonicalState`를 검사한다. Nonlinear 모델팀의 실제 module:factory,
kwargs/shape adapter 및 checkpoint 호환 구현은 미제공이다. 이번 repair는 optimizer/factory를 재설계하지 않았다.
