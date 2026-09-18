# 프로젝트 문맥: SAM 2.1 Tiny → Base+ nonlinear state handoff

> 확인된 코드·기존 실험과 현재 계획을 구분한 시작 문서. 기준일: 2026-09-18 KST.

## 1. 연구 질문

비디오 프레임 `t`까지 SAM 2.1 Tiny가 추적한 객체별 state `b_t`를 작은 nonlinear translator `T`로 Base+용 state `â_t=T(b_t)`로 바꾸면, Base+가 과거 영상을 전부 다시 처리하지 않고 `t+1`부터 정확히 추적할 수 있는가? Base+가 같은 과거를 직접 처리한 state는 `a_t`라고 쓴다.

목표는 tensor MSE 자체보다 후속 J&F, 객체 유지·재등장, 부재 중 오검출, 전환 지연·전달량 개선이다. Source-only와 강한 prompt/mask/replay 대안보다 실익이 있어야 연구 필요성이 성립한다.

## 2. 확정 범위

- Model: 공식 SAM 2.1 Tiny → Base+ 한 방향. 같은 revision의 코드와 각각의 checkpoint를 사용한다.
- Dataset: DAVIS 2017로 구현과 초기 검증, MOSEv2로 복잡한 장면·재등장, LVOS v2로 장기 추적을 평가한다. 세 데이터셋은 논문 범위에 포함된다.
- Method: nonlinear. 첫 모델은 component-wise residual MLP. 필요성을 검증하며 gated MLP와 slot/context attention을 비교한다. 새로운 Linear/Ridge 학습은 범위가 아니다.
- Baseline: Source-only, Base+-native/Full Replay, Direct Copy, 객체별 Original-Prompt, Last-Visible Source Mask, Original+Last-Visible, Original+Replay-k. Last-Mask/최근 Replay-k/Empty-reset proxy는 진단군으로 유지한다.
- 기간: [대한전자공학회 2026 추계학술대회](https://conf.theieie.org/2026f/pages/outlines.vm)의 논문 제출일은 2026-10-19로 확인했다. 정확한 마감 시각·시간대와 업로드 형식은 제출 화면에서 재확인한다. 일정은 필요한 데이터셋·사례·반복 횟수를 줄이는 상한이 아니다. 추가 GPU·작업 자원과 재개 가능한 실행으로 규모를 유지한다.

## 3. 확인된 기술 사실

- 공식 SAM 2 predictor는 등록 객체별 conditioning/non-conditioning output을 프레임 번호로 보관한다. 보관된 모든 과거를 매 프레임 읽는 것은 아니다. 객체가 없다고 예측한 프레임도 부재 처리가 반영된 출력이 남는다.
- SAM 2.1 네 크기의 공개 설정에서 memory attention은 256차원, spatial memory 출력은 64채널, 기본 `num_maskmem=7`이다. 같은 mapper 구조면 Tiny→Base+와 Tiny→Large의 translator 파라미터 수 자체는 같다.
- 현재 코드의 hidden=128 residual MLP는 82,498 parameters다. Base+가 더 쉽게 번역될지는 미검증이다.
- 첫 4개 기존 저장소 커밋에는 다른 팀원의 작업이 포함됐다. 이 저장소의 새 커밋 이력과 기존 파일의 작성 기여는 [이관 기록](MIGRATION.md)에서 따로 밝힌다.

출처: [SAM 2 논문](https://arxiv.org/abs/2408.00714), [공식 체크포인트 표](https://github.com/facebookresearch/sam2#model-description), 고정 upstream commit `2b90b9f5ceec907a1c18123530e92e794ad901a4`.

## 4. 기존 pilot과 해석의 한계

기존 Tiny→Large의 DAVIS train 내부 8개 handoff case 중 GT-visible 미래가 있는 5개에서 Direct J&F 0.014222, nonlinear MLP 0.112462, Large-native 0.653399였다. 한 사례에서는 MLP가 0.532671로 회복했지만 다른 사례의 실패가 커서 일반화가 확인되지 않았다. 기존 rare-event 10-case 결과도 3개 영상에 군집된 pilot이다. 이 수치를 Base+ 성능이나 DAVIS 공식 benchmark로 옮겨 쓰지 않는다. [이전 MLP 보고서](reports/legacy/report.md).

## 5. 실험 규칙

- 영상 단위로 train/validation/test를 분리한다. 모델 선택에 test 결과를 쓰지 않는다.
- 모든 비교군은 switch 이전에 등록된 동일 객체 집합과 실제 prompt timeline을 쓴다. 미래 GT로 handoff 입력을 고르지 않는다.
- Original-Prompt는 객체별 최초 지정 frame을 뜻한다. Last-Visible은 source 예측에서 객체별 마지막 비어 있지 않은 mask와 해당 RGB를 쓴다.
- Source-only는 수학적 하한, Base+-native는 수학적 상한이 아니다.
- 기존 runner는 대부분 단일 객체다. 새 다객체 anchor 비교군과 MOSEv2/LVOS v2 로더는 구현·검증 전이다.
- 재등장은 `(video, object, switch)` 단위로 평가한다. GT-visible J&F와 GT-absent false positive를 분리하고, 같은 영상의 여러 사례를 독립 표본처럼 세지 않는다.
- 모든 실험은 commit, upstream SHA, checkpoint SHA256, dataset/split, seed, config, 실행 명령, 로그, 결과 위치를 기록한다.

## 6. 현재 위치와 다음 단계

새 저장소 구성 및 범위 고정 → Base+ checkpoint와 same-checkpoint export→inject 검증 → 고정 baseline → nonlinear 모델 비교 → DAVIS/MOSEv2/LVOS v2 평가 → 4주 논문 작성. 자세한 일정과 성공 판정은 [실험 계획](docs/experimental_plan.md)을 따른다.

GitHub repository를 새로 만들었다는 사실만으로 실험이 이전됐다는 뜻은 아니다. 기존 GPU cache는 pair와 checkpoint가 일치하는지 확인한 뒤 사용하고 Base+-native state는 새로 생성한다.

## 7. 2026-09-18 범위 수정 기록

이전 연구의 Tiny→Large 중심 계획을 사용자의 지시에 따라 Tiny→Base+로 교체했다. DAVIS 2017·MOSEv2·LVOS v2와 nonlinear 여러 방식을 필수 연구 범위로 확정했다. 이후 사용자는 1개월 일정을 이유로 실험 규모·필요 비교군·반복을 축소하지 말라고 명시했다. 이 결정은 기존 pilot의 결과를 변경하지 않으며, 새 모델 쌍의 성능은 미검증이다.

새 저장소의 독립 Git 이력과 `main /docs` GitHub Pages는 공개됐고, 2026-09-18 확인 당시 GitHub Contributors는 `KIMKYUDO` 한 명이었다. 다만 기존 GitHub 저장소 삭제는 권한 부족(HTTP 403)으로 미완료다. 원본 이력은 기존 로컬 작업공간의 `.external/archives/vos-memory-translator-nonlinear-before-v2.bundle`에 검증된 백업으로 남겨 두었다.

## 8. 2026-09-18 일정·GitHub Projects 동기화

GitHub [연구 보드](https://github.com/orgs/memorybridge-team/projects/2/views/1)의 제목과 설명을 Tiny→Base+ 및 새 저장소 기준으로 수정했다. 데이터·상태 주입·강한 비교군·세 nonlinear 후보의 완료 조건을 항목 설명에 기록하고, 10/7의 최종 `Method freeze`를 후보/프로토콜 중간 검토로 바꿨다. 전체 학습의 시작일은 10/8에서 10/3으로 앞당겨 shard별로 확장하게 했으며, 10/18 작업은 PDF 전용이 아닌 공식 업로드 형식 검증으로 바꿨다. 원래 담당자와 완료 상태는 임의로 변경하지 않았다. 10/12~14 최종 방법/결과 동결, 10/15~17 검토, 10/18 모의 제출, 10/19 제출의 상세 게이트는 [실험 계획](docs/experimental_plan.md)에 둔다.

보드에 이 연구와 무관한 외부 저장소의 [`subinidus/cell-msca-odiac-estimation#2`](https://github.com/subinidus/cell-msca-odiac-estimation/issues/2)가 연결돼 있다. 원본 이슈를 삭제하지 않고 보드 연결만 해제하려는 조작이 자동 안전 검토에서 특정 항목 승인 부족으로 거절되었다. 사용자 승인 전까지 그대로 둔다.
