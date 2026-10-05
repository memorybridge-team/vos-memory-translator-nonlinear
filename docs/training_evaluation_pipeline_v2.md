# CMMT 학습·평가 파이프라인 v2

기준일: **2026-10-05 KST**. 실행 상태 snapshot: **14:50:34 KST**. 날짜가 있는 상태 기록이며 실시간 dashboard가 아니다.

이 문서는 현재 학습, 승인된 평가 목표, 아직 구현·검증되지 않은 부분을 구분한다. 공개 runtime core를 현재 RunPod 실행기라고 오해하지 않도록 정리한 운영 설명이다. 데이터·checkpoint·접속 인증값은 Git에 넣지 않는다.

## 1. 요약과 v1 대비 변경

| 항목 | 10/4 corrected LVOS-only run | 10/5 공식 run 및 v2 목표 |
|---|---|---|
| Train | LVOS train 내부 fit | MOSEv2 + LVOS v2 official train 전체 기존 cache |
| Validation | LVOS train 내부 development | LVOS v2 official valid |
| Epoch | 고정 30, 완료 | 최대 60; 시간·비용 상한 우선 |
| Batch | 16 × 2 GPU × 누적 2 = 64 | 32 × 2 GPU × 누적 1 = 64 |
| 학습 loss | Spatial/pointer normalized MSE | 동일 종류의 MSE; 새 train-only RMS |
| 모델 선택 | 내부 dev state loss 최소 | State best 보존 + 실제 valid J&F 최대인 best_JF 목표 |
| J&F | 해당 run에서 실행하지 않음 | 과거 완료 epoch 소급 + 향후 매 epoch, 준비 중 |
| GPU | RTX PRO 4500 ×2 | 학습 L4 ×2 / 평가 RTX 4000 Ada ×2 |

**v2는 새 학습을 epoch 1부터 다시 시작하라는 지시가 아니다.** 현재 공식 run의 가중치·optimizer·scheduler·RNG·cache/index를 보존하고 평가 및 선택 경로를 추가한다. 10/4 run과 10/5 run의 데이터·RMS가 다르므로 normalized loss 수치만으로 직접 우열을 비교하지 않는다.

## 2. 데이터 역할과 단위

| 역할 | Dataset | 영상 | Cases | Valid memory records |
|---|---|---:|---:|---:|
| Train | MOSEv2 official train | 3,295 | 20,841 | 295,009 |
| Train | LVOS v2 official train | 420 | 1,803 | 28,848 |
| Train 합계 | MOSEv2 + LVOS v2 | 3,715 | 22,644 | 323,857 |
| Validation | LVOS v2 official valid | 140 | 714 | 11,424 |

이는 현재 학습 inventory의 수량이다. Video는 영상, case는 영상·객체·switch 조합, record는 한 memory 시점의 spatial/pointer pair다. Batch 64는 RGB 영상 64개가 아니라 memory records 64개다. 11,424 records의 state validation과 714 cases의 실제 영상 rollout J&F는 작업량이 다르다.

- 기존 `fit/`와 `development/`는 cache가 저장된 역사적 디렉터리 이름이다. 이번 run에서는 양쪽 official **train** membership을 합쳐 train으로 사용한다. 원본 frozen membership을 재작성한 것이 아니다.
- LVOS official valid는 gradient와 normalization 계산에 사용하지 않지만 scheduler·checkpoint 선택에 노출된다. 따라서 최종 미노출 test라고 부르지 않는다.
- MOSE official valid는 이번 run의 선택에 사용하지 않는다. DAVIS는 train/validation 및 현재 연구 주장 범위에서 제외한다.
- VOST valid, PUMaVOS, M³-VOS는 팀이 지정한 external evaluation 후보다. 이들의 결과로 best 모델이나 hyperparameter를 고르지 않는다. 현재 사용자 요청은 새로운 external test/baseline 실행 승인이 아니다.
- Dataset마다 공개 GT 및 공식 evaluator의 범위가 다르다. 예측을 생성했다는 것과 점수 산출을 구분하며, 모든 데이터셋의 J&F를 무조건 산출 가능하다고 표현하지 않는다.

## 3. 학습 모델과 objective

Small/Base+의 대응 memory pair를 저장해 둔 offline **feature/state regression**이다. RGB에서 mask를 직접 예측하는 학습도, 두 SAM 2 backbone을 fine-tune하는 학습도 아니다.

현재 별도 실행 snapshot의 translator는 spatial-context Transformer와 residual pointer MLP다. 이 구조가 공개 core `main`에 통합되었다는 뜻은 아니다.

```text
Small spatial memory ── spatial translator ── Base+ spatial memory와 MSE
Small object pointer ── pointer translator ── Base+ object pointer와 MSE
                             ↓
                 Translator 가중치만 업데이트
```

Frame/object/slot/conditioning/validity/switch metadata는 대응 관계를 보존하며 학습 target이 아니다. Runtime handoff에서는 Target positional encoding을 Target이 재생성한다.

```text
L = mean_valid_record_MSE(spatial) / train_RMS_spatial²
  + mean_valid_record_MSE(pointer) / train_RMS_pointer²
```

- 현재 train RMS: spatial `0.8042111680477584`, pointer `0.6337650544928317`.
- RMS는 loss scale이다. 평가 시 translator 입력을 별도로 다시 정규화하라는 의미가 아니다.
- Padding은 제외하며 실제 absent-object valid records는 유지한다. Tail batch는 실제 valid count로 gradient/loss를 가중한다.
- Spatial/pointer weight는 각 1, cosine objective weight는 0이다. Cosine은 진단 지표다.

| 설정 | 현재 공식 run |
|---|---|
| Optimizer | AdamW, 초기 peak LR 3e-4, weight decay 1e-4 |
| Seed / precision | 7 / FP32 |
| Global batch | 64; GPU당 microbatch 32, accumulation 1 |
| Loader | Rank당 workers 2, prefetch factor 2, pinned memory |
| Gradient clipping | Global norm 1.0 |
| Warmup | 최대 1,000 optimizer updates |
| LR scheduler | Official-valid state loss 기반 Plateau; 기존 상태 보존 |
| Epoch당 updates | 5,061; 마지막 batch는 17 valid records |
| Epoch 상한 | 60; 승인된 시간·비용이 우선 |

낮은 state loss가 높은 mask J&F를 보장하지 않는다. State loss는 표현 복원 오차이고 J&F는 실제 handoff 이후 분할 결과를 GT와 비교하는 지표다.

## 4. 현재부터 최종 전달까지

| 단계 | 내용 | 현재 상태 |
|---|---|---|
| A | Index/RMS 검사, 실제 2-GPU smoke, epoch 1 측정 | 완료된 학습 gate |
| B | 기존 L4 ×2에서 train → state validation → 저장 반복 | 원격 실행 중 |
| C | 평가 Pod SSH·host key·GPU·volume 확인 | 접속 검증 중 |
| D | Official checkpoint adapter + 실제 SAM 2 rollout smoke | 미가동/미검증 |
| E | 전체 validation 1회 실측, coverage·시간·비용 확인 | 미실측 |
| F | 과거 모든 완료 epoch 소급 및 향후 checkpoint 자동 평가 | 승인 목표; 미가동 |
| G | Full-coverage J&F 집계, best_JF 갱신 | 미가동 |
| H | Best_JF 및 state/last 산출물·SHA·보고서 정리 | J&F용 구현/검증 대기 |
| I | 평가팀 다운로드·무결성 확인, GPU Pod 정리 | 사람의 확인/중지 필요 |

평가 Pod의 두 GPU는 서로 다른 영상 shard를 처리하는 독립 worker 구성이다. 학습 DDP에 합쳐 쓰지 않는다. RTX 4000 Ada는 GPU당 20,475 MiB이며 두 장을 하나의 40 GB 메모리로 가정하지 않는다. 실제 SAM 2 rollout의 peak VRAM/RAM은 아직 측정하지 않았다.

### 실제 평가와 best_JF 판정

1. 완료 marker와 checkpoint/weights/config/normalization SHA 및 source/index binding을 검증한다.
2. 고정된 prompt/switch 조건으로 실제 Small prefix를 처리한다.
3. Small state를 해당 epoch translator로 변환하여 Base+에 주입한다.
4. Base+는 switch 다음 frame부터 이어 추론한다. 과거 Base+ frame replay를 새로 실행하는 방법으로 바꾸지 않는다.
5. Sparse GT의 원본 frame ID와 runtime frame을 대조하여 실제 J/F를 계산한다.
6. 모든 frozen case 및 필요한 GT/prediction coverage, disjoint shard union, 실패 없음과 metric binding이 확인된 결과만 공식 best_JF 후보가 된다.

일부 영상이나 짧은 smoke 점수, visible-only proxy, Full-Replay retention은 raw full-validation J&F와 구분한다. 평가 대상 조건이 같지 않은 점수를 서로 비교하지 않는다. LVOS official split에서 switch 평가를 수행해도 표준 전체 영상 leaderboard 점수와 자동으로 같아지는 것은 아니다.

**남은 protocol 결정:** 기존 714-case manifest는 official object frame-range metadata를 이용해 25/50/75% switch를 만든다. 이를 엄격한 미래 GT metadata 독립 조건으로 주장할 수 없다. 현재 요구와의 정합성을 평가 전 확정해야 한다. 새 RGB-only protocol이 필요하면 별도 namespace에 동결하고 기존 cache/index를 수정하지 않는다. 미래 GT mask pixel을 추가 prompt나 유리한 입력 선택에 사용하는 것은 금지한다.

## 5. 어디까지 자동화됐는가

| 기능 | 현재 확인 | 주의점 |
|---|---|---|
| 노트북 종료 후 학습 지속 | 확인 | 원격 tmux/controller; Pod는 계속 실행돼야 함 |
| 매 epoch state validation·checkpoint/export | 확인 | J&F가 아니라 memory MSE |
| State-best 기록·LR 조정 | 확인 | J&F-best와 구분 |
| 학습 자체 시간/비용 제한 | 기존 controller에 있음 | 프로세스 제한은 Pod 과금 중지와 다름 |
| 종료 뒤 기존 state 전달물 정리 | Watcher 구성·실행 기록 있음 | 새 정책 continuation/J&F 최종물까지 보장하지 않음 |
| 모든 epoch J&F queue·두 GPU 평가·best_JF | 목표 승인, 미가동 | Adapter·실제 smoke·full validation·예산 검증 필요 |
| State-loss-only stopping 보류 | 변경 요청, 적용 미확인 | Checkpoint-boundary receipt와 상태 보존 검증 필요 |
| 평가팀 다운로드·로컬 SHA 확인 | 자동 확인 아님 | 수령자가 확인해야 함 |
| RunPod Stop/Terminate | 자동화하지 않음 | 종료 후에도 idle 과금 |
| 5분 알림 automation | 만들지 않음 | 서버 상태 파일과 사람의 조회를 사용 |

프롬프트를 보냈다는 사실은 구현·원격 배포·작동 검증이 끝났다는 증거가 아니다. 특히 현재 controller가 J&F를 기다리거나 J&F로 stopping을 제어한다고 표현하지 않는다.

## 6. 현재 stopping과 복구 조건

14:24:51 KST 감사에서 완료 epoch 33, state stopping bad checks 6/8, LR 3.75e-5를 확인했다. 이후 14:50:34 조회에서 epoch 34 validation loss는 `0.49403918403036456`이고 state best epoch 27 loss는 `0.49264229066064713`다. 기존 stopping 상태가 그대로라면 개선 없이 epoch 35 완료 뒤 조기 종료할 수 있다.

이 경우 저장된 완성 epoch checkpoint는 보존된다. 정책 migration을 구현·검증한 후 해당 checkpoint의 model/optimizer/scheduler/RNG/epoch/step을 이어받을 수 있으며 epoch 1부터 재학습할 필요는 없다. 아직 그러한 migration 또는 재개가 적용됐다는 증거는 없다. 최대 60/time/USD 상한을 늘리지 않으며 LR scheduler는 임의 변경하지 않는다.

학습 run이 끝나도 소급 평가를 계속할 수 있다. 반대로 이전 state-only `FINAL_READY`가 생성되어도 `best_JF`가 준비됐다는 뜻은 아니다.

## 7. 작업량·ETA·비용

현재 metadata 조사 기준 full validation은 140 videos / 238 objects / 714 cases다. Source prefix 148,889 frames, Target future 159,556 frames, fresh processing 합계 proxy 308,445 frames다. 이는 **GPU 처리 실측이 아니라 manifest/RGB filename 계산**이다. 최종 protocol이 바뀌면 작업량도 다시 계산한다.

평가 source를 매 epoch 새로 처리하지 않고 재사용할 수 있다면, 먼저 실제 source weights/prompt/RGB/runtime/policy/SHA가 결합된 평가 전용 immutable source state를 검증해야 한다. 현재 그런 source view가 생성·통과됐다고 기록하지 않는다. 기존 pair의 UNKNOWN generating provenance를 채워 넣지 않는다.

평가 ETA는 다음을 실측한 후 계산한다.

```text
평가 종료 = 현재 시각 + 준비/검증 시간
          + 남은 checkpoint들의 실제 두-GPU 평가 시간
          + 최신 checkpoint 도착 대기 + merge/저장/전달 시간
비용 = 모든 실행 Pod의 전체 단가 × 예약 wall hours 합계
```

순수 평가 시간에 GPU 개수를 곱해 청구하거나 두 GPU가 반드시 두 배 빠르다고 가정하지 않는다. 학습 ETA와 평가/전달 ETA를 따로 보고한다. 10/6 21:00 KST는 iteration 전달 목표이지 현재 보장된 완료 시각이 아니다.

- 사용자가 확인한 전체 Pod 표시 단가: 학습 USD 0.98/h, 평가 USD 0.56/h. 둘 다 실행 중이면 USD 1.54/h. 기존 학습 controller의 보수적 운영 rate는 USD 1.00/h이며 청구 단가와 구분한다.
- 평가 초기 검증 승인: 10/5 14:41부터 최대 2시간/최대 USD 3 중 먼저 도달하는 한도. 이 단계 이후 지속 실행에는 전체 실측과 잔여 예산 확인이 필요하다.
- 합산 승인 상한은 기존 USD 50이다. 이번 USD 3은 추가 총예산이 아니다. 기존 사용분·준비·idle을 차감한다.
- 학습 한도는 10/6 18:00 KST, iteration 목표는 21:00 KST다. 자동 Pod Stop은 하지 않는다.
- 두 Pod를 승인 accounting 시작(학습 10/5 01:22:56, 평가 10/5 14:41)부터 10/6 21:00까지 유지하는 단순 표시 단가 시나리오는 약 **USD 59.72**다. 보수적 학습 USD 1.00/h로는 약 USD 60.60이다. Volume·승인 전 사용분 등은 별도 미확인이다.

따라서 두 Pod를 계속 켜 두는 방식이 USD 50 안에 들어간다고 약속하지 않는다. 학습 산출물 완성 확인 후 사용자가 학습 Pod를 중지하고 평가 Pod에서 같은 volume의 전달물을 제공하는 운영은 검토할 수 있다. 이 문서가 Pod 중지 승인이나 자동 중지 구현은 아니다.

## 8. Snapshot·근거·저장소 경계

### 14:50:34 KST 실제 SSH 읽기 전용 조회

| 항목 | 확인값 |
|---|---|
| 완료 epochs / 현재 epoch | 34 / 35 |
| 현재 optimizer step / LR | 174,000 / 3.75e-5 |
| Epoch 34 train / validation state loss | 0.4525325399 / 0.4940391840 |
| State-best | Epoch 27, validation loss 0.4926422907 |
| Epoch 34 wall | 1,444.91초, 약 24.08분 |
| Controller·torchrun·양 rank | 기존 프로세스 존재 |
| 최종 전체 delivery marker | 미생성 |
| 실제 J&F / best_JF | 없음; 연결 준비 중 |

Process 존재와 step/history 관측은 그 시점의 운영 증거다. 이 문서의 수치를 이후 시점의 실시간 진행률로 사용하지 않는다. `training/STATUS.json`은 첫 epoch 경계의 오래된 완료 기록이므로 단독으로 전체 run 완료를 판단하지 않는다.

```text
Current official run:
/workspace/CMMT-official-isolated/official-20261005T012256KST

Trainer local source revision:
d152ca8ab1ae16d5b9aff978d2fbcd10722d9efa
Deployed package SHA256:
8b2d2c195facae4ddd9d8ccea014412321276cee9799ea70ba1b9f741fa26397
Official index SHA256:
3ae8b00b5cfdf7df624a6c1b476d95ceee7ce259f796338b02975c201556ae9c
Best-state epoch 27 checkpoint SHA256:
9a9001f0bf394ced50722809747a16e6efc741534705767afe86c7886cf7c04d
```

기존 완료 근거:

- 10/4 LVOS-only corrected run: 30 epochs, 11,160 updates, 전달 archive SHA256 `5bd3f630ed032b87892a641cfbc255eef2a289dd91e85432a3e55a1a49e56693`.
- 10/5 공식 run: CPU 전용 최종 검사 9 PASS, 전체 cache 검사 23,358 cases/거부 0, 실제 2-GPU smoke 및 epoch 1 strict reload/rank 일치 확인 기록. 이것은 downstream J&F PASS 증거가 아니다.
- 비공개 운영 기록: `CURRENT_RUN_REPORT_KO.md`, `JF_RESOURCE_BRIEFING_KO.md`(14:24:51), `JF_WORKLOAD.json`(14:21:07), `training/LIVE_STATUS.json`, `training/metrics/history.json`. 접속 키·token·raw tensors를 공개 저장소에 복사하지 않는다.
- 현재 v2 문서가 기준으로 삼은 공개 core `main`은 `55b117d`다. `main`의 core와 배포 trainer revision은 서로 다르며 이 문서 변경으로 자동 동기화되지 않는다.

공개 core·local orchestration·benchmark·evidence archive 경계는 [repository architecture](repository_architecture.md)를 유지한다. 기존 개발방 checkout과 실행 중인 trainer는 변경하지 않는다. 미완료 J&F adapter를 게시된 API로 적지 않는다.

## 9. 평가팀이 확인할 순서

1. 현재 진행은 공식 run의 `training/LIVE_STATUS.json`과 `training/metrics/history.json`으로 확인한다.
2. `training/delivery/epoch-*/READY.json`은 해당 epoch의 **state 학습 후보**가 준비됐다는 의미다. J&F-best나 전체 연구 완료 marker가 아니다.
3. `DELIVERY_FINAL_READY.json`은 기존 state 전달 패키징 조건이다. 정책 continuation의 중간 종료 기록일 수 있는지 함께 확인한다.
4. J&F worker가 배포되면 실제 평가 root, 상태 JSON schema, full-coverage best_JF 및 최종 READY의 정확한 경로를 개발방이 보고해야 한다. 아직 없는 경로를 확정 명령처럼 안내하지 않는다.
5. 최종 다운로드는 weights/config/normalization/binding/평가 protocol/coverage/J&F/checkpoint SHA를 함께 수령하고 archive SHA를 확인한다.
6. Jupyter의 인증값은 별도 안전한 경로로 전달한다. 접속 링크를 안다는 것만으로 모든 팀원이 인증되거나 읽기 전용 권한을 갖는 것은 아니다.
7. 두 Pod의 idle 과금과 중지 여부는 사용자가 확인한다. 학습 종료, 평가 종료, 산출물 준비, 다운로드 확인, Pod 중지는 서로 다른 상태다.
