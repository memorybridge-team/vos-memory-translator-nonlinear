# CMMT 현재 프로젝트 문맥

기준: 2026-10-05 KST. 현재 운영 문서는 [학습·평가 파이프라인 v2](docs/training_evaluation_pipeline_v2.md)다. 이 파일과 v2 문서는 공개 core에 실행기를 다시 넣거나 실행 중인 RunPod 소스를 변경하지 않는다.

## 최신 결정

- 모델 쌍은 SAM 2.1 **Small → Base+**, 한 방향·한 번의 no-replay memory handoff다. 과거 Tiny→Large/Tiny→Base+ pilot을 현재 성능으로 취급하지 않는다.
- 10/4 corrected LVOS-only 30-epoch run은 완료된 과거 run으로 보존한다. 10/5 공식 run은 MOSEv2 + LVOS v2 official train으로 별도 시작했으며, 이 run의 optimizer를 초기화하지 않는다.
- 10/5 공식 run은 기존 train 내부 fit/development membership을 합쳐 train으로 사용한다. LVOS v2 official valid는 모델 선택에 노출되는 validation이며 sealed final test가 아니다. DAVIS는 제외한다.
- 학습 objective는 train-only RMS로 정규화한 spatial/pointer MSE다. SAM 2 backbone은 고정하고 translator만 갱신한다.
- 최신 사용자 요청은 모든 완료 epoch의 실제 LVOS validation downstream J&F를 평가하여 `best_JF`를 선정하는 것이다. `best_state`를 진단용으로 보존한다.
- L4 ×2 학습을 유지하고 RTX 4000 Ada ×2를 평가 전용으로 사용한다. 4-GPU 학습 전환이 아니다. 자동 Pod Stop은 승인되거나 구현된 것으로 간주하지 않는다.

## 구현·운영 상태

- 2026-10-05 14:50:34 KST 읽기 전용 SSH 조회: 34 epochs 완료, epoch 35 / step 174,000, 학습 controller 및 양 rank 존재. State-loss best는 epoch 27이다.
- 매 epoch 실제 J&F 평가, `best_JF` 갱신, J&F 최종 패키징은 아직 가동 증거가 없다. 새 평가 Pod 접속 및 공개 host-key 확인이 진행 중이다.
- State-loss stopping을 보류하는 정책 migration은 아직 적용 증거가 없다. 최대 60 epochs는 무조건 60회를 보장하는 값이 아니다.
- 공식 trainer revision `d152ca8ab1ae16d5b9aff978d2fbcd10722d9efa`는 별도 로컬 실행 snapshot이다. 공개 `main`에 게시된 trainer라고 표현하지 않는다.

## 변경 이력과 출처

2026-10-05 사용자 요청에 따라 official-validation 선택 정책과 평가 전용 GPU 구성을 문서화했다. 과거 실행·승인·작성 기여 기록은 삭제하거나 소급 수정하지 않는다. 저장소 경계와 역사적 증거의 위치는 [repository architecture](docs/repository_architecture.md)를 따른다. 본 문서의 snapshot 근거와 미완료 gate는 v2 문서 8절에 적었다.
