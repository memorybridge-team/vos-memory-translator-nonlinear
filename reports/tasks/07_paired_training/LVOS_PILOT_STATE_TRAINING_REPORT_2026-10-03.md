# LVOS diagnostic pilot·학습 전용 모드 — 로컬 구현 보고서

> 이 문서는 기존 cache 실제 읽기/DDP 구현 이전 단계의 기록이다. 현재 입력 상태와 실행 절차는
> `LVOS_DDP_TRAINING_REPORT_2026-10-03.md` 및 `docs/runpod_lvos_translator_training_operator_guide.md`를 따른다.

기준: `f2632d73100ad8d5dfebc7a4eb55c9e3e9a4bfd7` → `feat/lvos-pilot-preflight`. 2026-10-03.

## 범위

최신 사용자 지시는 translator 학습, 두 component loss, checkpoint/reload/재개, fit/dev state loss 및 weights/config/normalization 전달입니다. 신규 pair 생성은 하지 않습니다. 다른 팀의 runtime handoff·추론·benchmark 구현은 변경하지 않았습니다. 원본 cache·frozen split·모델 body/API와 기존 미추적 팀원 guide를 보존했습니다. 실제 GPU는 verified snapshot과 최종 실행 시간/예산 입력 전 **BLOCKED**입니다.

## 재현과 수정

- 변경 전 CPU mock에서 G1 `REAL_FULL_AUDIT_REQUIRED` FAIL 뒤 `['G2','G4']` 호출을 실제 재현했습니다. 수정 전 test exit 1. 실제 GPU를 호출한 재현이 아닙니다.
- `run_gates`는 검증에 성공한 full snapshot만 할당하고 공식 G2/G4가 현재 run의 G0/G1 PASS에 명시적으로 의존하도록 수정했습니다.
- `pilot_ready`는 공식 G1 PASS가 아닙니다. G0/G1 FAIL/BLOCKED에서는 GPU 함수가 호출되지 않고 정상 full mock 경로는 보존합니다.
- 첫 재현 시 local `PYTHONPATH` 미설정으로 collection error(exit 4)가 있었습니다. src를 지정한 별도 재현 log/JUnit에서 실제 결함을 확인했습니다. 첫 로그도 보존합니다.

## 학습/평가 분리

- 기본 CLI `train --training-scope state_supervised`: 기존 runner·loss·stats·optimizer·atomic checkpoint를 재사용합니다. monitor/J&F worker와 evaluation wait·J&F early stopping을 사용하지 않습니다.
- state 승인에는 G0/G1/G2/G3/G4/G6과 data/model/code identity 및 artifact checksum이 필요합니다. G2/G4는 실제 GPU 근거여야 합니다. 기존 G0 static freeze 계약은 유지하고 G5 평가 완료만 분리합니다.
- 전체 연구 `authorize_training`은 기존 G0..G6을 모두 요구합니다. `state_training_ready`는 별도 필드이고 diagnostic/synthetic/stale/변조 근거는 본 학습 승인으로 쓸 수 없습니다.
- 고정 한도로 fit/dev raw spatial/pointer MSE, normalized 두 loss와 total을 기록합니다. CSV에도 두 component를 추가했습니다.
- epoch 0과 완성 epoch의 `translator/epoch-NNNNN.pth`는 기존 canonical `from_payload` 형식입니다. config는 payload/model lock, stats는 `translator/normalization.json`, provenance/checkpoint SHA는 epoch sidecar로 전달합니다. 완성 epoch checkpoint를 재개하면 누락 export도 복구합니다.
- `best_state_loss.ckpt.json`과 최종 VOS best는 구분합니다. partial epoch exact resume 또는 VOS 개선을 주장하지 않습니다.

## 독립 diagnostic

`pilot-preflight`는 exact one-fit `pilot_ready` snapshot에 대해 finite shape/dtype/reference/frame/slot/conditioning·validity 대응을 검사합니다. `synthetic=false`가 CLI의 필수 조건입니다. CPU fixture 허용은 테스트용 Python 함수 인자에만 있고 CLI에 없습니다.

FP32/seed7/workers0/microbatch1/no augmentation으로 fit stats를 한 번 계산하고 3 update를 수행합니다. residual alpha로 첫 step 일부 deep gradient가 0일 수 있으므로 최종 deep 전달을 확인합니다. complete optimizer boundary checkpoint를 atomic 저장하고 model/optimizer/scheduler/RNG/stats/config/source/data identity 및 next update를 zero-tolerance 비교합니다. failure difference와 로그를 보존합니다. scope는 `diagnostic_only`, 두 학습 ready flag는 false입니다.

## CPU 근거

검사 환경: local Windows Python 3.14/Torch 2.14+cpu, `CUDA_VISIBLE_DEVICES=''`, `PYTHONPATH=src`. 실제 command/log/JUnit/report는 workspace `outputs/lvos_pilot_20261003T095455Z/`에 있습니다.

- 수정 전 재현: `gate_reproduction_before_fixed_env.log/xml`, 1 FAIL, exit 1.
- 최초 의존성 수정: `gate_dependency_after.log/xml`, 5 PASS, exit 0.
- 최초 pilot/state 분리: `pilot_cpu_implementation.log/xml`, 14 PASS, exit 0.
- 최종 source-bound: `local_source_bound_final/{report.json,pytest.log,junit.xml}`, **96 PASS**, failures/errors/skips 0, exit 0, pytest 309.54초.
- 앞의 14/5 및 중간 91개와 최종 96개는 겹칩니다. 합산하지 않습니다. 최종 전체 회귀와 Python 3.12/Torch 2.8 Pod 결과는 새 배포의 실행 보고서에 별도 기록합니다. f2632d7의 199/77을 수정 후 코드 PASS로 사용하지 않습니다.

최종 package SHA: `175aef224199e4a357eac336033573bf7267b00584d685b790a3faa1434aabbf`.
최종 safety suite SHA: `30428ba7ac9127f80cf23f7e9a1a46b1508c1958722dd188baef78524b7aa830`.

## 실제 입력과 비용

한 case의 읽기 전용 metadata에서 frozen fit `train:0ClBYzYm:obj1:switch1221`, 505 JPEG, official prompt 1/runtime 0, official switch 1221/runtime 244가 맞습니다. 기존 `.pt`, prepare, SHA sidecar는 존재하나 completion/generating binding과 actual tensor는 미검증입니다. 원본 pickle load/copy/hash/import는 수행하지 않았습니다. 다른 Pod writer는 UNKNOWN입니다.

사용자가 현재 Dashboard 단가를 확인했습니다: GPU USD 0.740/h + Container disk USD 0.004/h = **USD 0.744/h**, 기존 Network Volume 별도. 원격 120분 예약은 산술상 USD 1.488입니다. 실제 세션 비용/remote elapsed는 종료 receipt로 계산합니다. 실제 GPU 누적 사용은 현재 0초이고 Pod 과금은 계속됩니다.

남은 입력: 데이터팀 verified pilot/full view 및 provenance/completion; 정식 model lock/static freeze 근거; 실제 GPU 실행 시간/예산 확정. 신규 pair 생성 없이 데이터팀 입력을 기다립니다. 정확한 명령과 전달 파일은 `docs/runpod_lvos_translator_training_operator_guide.md`에 있습니다.
