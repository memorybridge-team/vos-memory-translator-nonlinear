# 프로젝트 문맥: SAM 2.1 Small → Base+ nonlinear state handoff

> 확인된 코드·기존 실험과 현재 계획을 구분한 시작 문서. 기준일: 2026-09-24 KST.

## 1. 연구 질문

비디오 프레임 `t`까지 SAM 2.1 Small이 추적한 객체별 state `b_t`를 작은 nonlinear translator `T`로 Base+용 state `â_t=T(b_t)`로 바꾸면, Base+가 과거 영상을 전부 다시 처리하지 않고 `t+1`부터 정확히 추적할 수 있는가? Base+가 같은 과거를 직접 처리한 state는 `a_t`라고 쓴다.

목표는 tensor MSE 자체보다 후속 J&F, 객체 유지·재등장, 부재 중 오검출, 전환 지연·전달량 개선이다. Source-only와 강한 prompt/mask/replay 대안보다 실익이 있어야 연구 필요성이 성립한다.

## 2. 확정 범위

- Model: 공식 SAM 2.1 Small → Base+ 한 방향. 같은 revision의 코드와 각각의 checkpoint를 사용한다.
- Dataset role: MOSEv2/LVOS v2 train-fit의 paired state로 translator를 학습하고 video-disjoint dev에서 dense-GT downstream 성능으로 선택한다. LVOS v2 validation은 local detailed final, MOSEv2 validation은 Codabench sealed final, VOST validation은 primary external zero-shot으로 사용한다. PUMaVOS와 M³-VOS는 각각 partial/unusual-mask와 material phase-transition을 보는 complementary external zero-shot stress benchmark다. DAVIS는 현재 연구의 학습·평가·주장 범위에서 제외한다.
- Method: nonlinear. 첫 모델은 component-wise residual MLP. 필요성을 검증하며 gated MLP와 slot/context attention을 비교한다. 새로운 Linear/Ridge 학습은 범위가 아니다.
- Baseline: Source-only, Base+-native/Full Replay, Direct Copy, Moment-Matched Copy, 객체별 Original-Prompt, Last-Visible Source Mask, Original+Last-Visible, Original-Prompt(s)+Replay-4/8/16, Nonlinear Translator. Last-Mask와 original anchor 없는 Recent-Window Replay-k는 최종 비교군에서 제외한다. Empty-reset proxy는 구현 진단군으로만 유지한다.
- 기간: [대한전자공학회 2026 추계학술대회](https://conf.theieie.org/2026f/pages/outlines.vm)의 논문 제출일은 2026-10-19로 확인했다. 정확한 마감 시각·시간대와 업로드 형식은 제출 화면에서 재확인한다. 일정은 필요한 데이터셋·사례·반복 횟수를 줄이는 상한이 아니다. 추가 GPU·작업 자원과 재개 가능한 실행으로 규모를 유지한다.

## 3. 확인된 기술 사실

- 공식 SAM 2 predictor는 등록 객체별 conditioning/non-conditioning output을 프레임 번호로 보관한다. 보관된 모든 과거를 매 프레임 읽는 것은 아니다. 객체가 없다고 예측한 프레임도 부재 처리가 반영된 출력이 남는다.
- SAM 2.1 네 크기의 공개 설정에서 memory attention은 256차원, spatial memory 출력은 64채널, 기본 `num_maskmem=7`이다. 같은 mapper 구조면 Small→Base+와 Tiny→Large의 translator 파라미터 수 자체는 같다.
- 현재 코드의 hidden=128 residual MLP는 82,498 parameters다. Base+가 Small state를 충분히 활용할 수 있을지는 미검증이다.
- 첫 4개 기존 저장소 커밋에는 다른 팀원의 작업이 포함됐다. 이 저장소의 새 커밋 이력과 기존 파일의 작성 기여는 [이관 기록](MIGRATION.md)에서 따로 밝힌다.

출처: [SAM 2 논문](https://arxiv.org/abs/2408.00714), [공식 체크포인트 표](https://github.com/facebookresearch/sam2#model-description), 고정 upstream commit `2b90b9f5ceec907a1c18123530e92e794ad901a4`.

## 4. 기존 pilot과 해석의 한계

기존 Tiny→Large의 DAVIS train 내부 8개 handoff case 중 GT-visible 미래가 있는 5개에서 Direct J&F 0.014222, nonlinear MLP 0.112462, Large-native 0.653399였다. 한 사례에서는 MLP가 0.532671로 회복했지만 다른 사례의 실패가 커서 일반화가 확인되지 않았다. 기존 rare-event 10-case 결과도 3개 영상에 군집된 pilot이다. 이 수치를 Base+ 성능이나 DAVIS 공식 benchmark로 옮겨 쓰지 않는다. [이전 MLP 보고서](reports/legacy/report.md).

## 5. 실험 규칙

- MOSEv2/LVOS v2 train을 영상 단위 fit/dev로 분리한다. Primary fit에는 future dataset GT를 쓰지 않고 `(b_T,a_T)` state pair를 사용한다. Official validation과 VOST/PUMaVOS/M³-VOS는 config freeze 전까지 열지 않으며 모델 선택에 쓰지 않는다.
- 모든 비교군은 switch 이전에 등록된 동일 객체 집합과 실제 prompt timeline을 쓴다. 미래 GT로 handoff 입력을 고르지 않는다.
- Original-Prompt는 객체별 최초 지정 frame을 뜻한다. Last-Visible은 source 예측에서 객체별 마지막 비어 있지 않은 mask와 해당 RGB를 쓴다.
- Source-only는 수학적 하한, Base+-native는 수학적 상한이 아니다.
- 기존 runner는 대부분 단일 객체다. 새 다객체 anchor 비교군과 MOSEv2/LVOS v2 로더는 구현·검증 전이다.
- 재등장은 `(video, object, switch)` 단위로 평가한다. GT-visible J&F와 GT-absent false positive를 분리하고, 같은 영상의 여러 사례를 독립 표본처럼 세지 않는다.
- 모든 실험은 commit, upstream SHA, checkpoint SHA256, dataset/split, seed, config, 실행 명령, 로그, 결과 위치를 기록한다.

## 6. 현재 위치와 다음 단계

새 저장소 구성 및 범위 고정 → Base+ checkpoint와 same-checkpoint export→inject 검증 → Task 03 VOST/PUMaVOS/M³-VOS onboarding → MOSE/LVOS future-GT-free paired-state·baseline·nonlinear 학습 → sealed in-domain 평가 → VOST/PUMaVOS/M³-VOS external 평가 → 논문 작성. 자세한 일정과 성공 판정은 [실험 계획](docs/experimental_plan.md)을 따른다.

GitHub repository를 새로 만들었다는 사실만으로 실험이 이전됐다는 뜻은 아니다. 기존 GPU cache는 pair와 checkpoint가 일치하는지 확인한 뒤 사용하고 Base+-native state는 새로 생성한다.

## 7. 2026-09-18 범위 수정 기록

이전 연구의 Tiny→Large 중심 계획을 사용자의 지시에 따라 Tiny→Base+로 교체했다. DAVIS 2017·MOSEv2·LVOS v2와 nonlinear 여러 방식을 필수 연구 범위로 확정했다. 이후 사용자는 1개월 일정을 이유로 실험 규모·필요 비교군·반복을 축소하지 말라고 명시했다. 이 결정은 기존 pilot의 결과를 변경하지 않으며, 새 모델 쌍의 성능은 미검증이다.

새 저장소의 독립 Git 이력과 `main /docs` GitHub Pages는 공개됐고, 2026-09-18 확인 당시 GitHub Contributors는 `KIMKYUDO` 한 명이었다. 다만 기존 GitHub 저장소 삭제는 권한 부족(HTTP 403)으로 미완료다. 원본 이력은 기존 로컬 작업공간의 `.external/archives/vos-memory-translator-nonlinear-before-v2.bundle`에 검증된 백업으로 남겨 두었다.

## 8. 2026-09-18 일정·GitHub Projects 동기화

GitHub [연구 보드](https://github.com/orgs/memorybridge-team/projects/2/views/1)의 제목과 설명을 Small→Base+ 및 새 저장소 기준으로 수정했다. 데이터·상태 주입·강한 비교군·세 nonlinear 후보의 완료 조건을 항목 설명에 기록하고, 10/7의 최종 `Method freeze`를 후보/프로토콜 중간 검토로 바꿨다. 전체 학습의 시작일은 10/8에서 10/3으로 앞당겨 shard별로 확장하게 했으며, 10/18 작업은 PDF 전용이 아닌 공식 업로드 형식 검증으로 바꿨다. 원래 담당자와 완료 상태는 임의로 변경하지 않았다. 10/12~14 최종 방법/결과 동결, 10/15~17 검토, 10/18 모의 제출, 10/19 제출의 상세 게이트는 [실험 계획](docs/experimental_plan.md)에 둔다.

## 9. 2026-09-19 모델 쌍 결정: Small → Base+

사용자가 제공한 `Appendix.3`을 검토한 뒤 현재 연구 쌍을 **SAM 2.1 Small → Base+**로 변경했다. 여기서 `Base+`는 공식 checkpoint 명칭이며 `Base++`가 아니다.

- **확인:** 공식 checkpoint 표에서 Small은 46.0M parameters, 84.8 FPS이고 Base+는 80.8M, 64.1 FPS이다. 따라서 전환은 더 빠른 Small 실행에서 약 1.76배 큰 Base+로 품질 우선 모드로 넘어가는 시나리오다.
- **판단:** 이 변경의 장점은 translator의 텐서 크기가 작아진다는 데 있지 않다. 두 모델은 같은 video-memory 경계를 공유하므로 mapper의 입출력 계약은 대체로 같다. 기대하는 이점은 Tiny보다 풍부한 source 표현으로 상태 번역의 성공 가능성을 높이되, Base+의 계산 비용과 품질 선택권을 여전히 남기는 것이다.
- **제한:** 공식 MOSE validation/LVOS v2 평균에서 Small과 Base+의 차이는 작다. 따라서 “항상 Base+가 더 정확하다” 또는 “Small→Base+ 전환이 언제나 필요하다”라고 주장하지 않는다. 가림·재등장·장기 부재 등 사전 정의한 어려운 조건에서 target-native 이득과 handoff 이득이 함께 나타나는지 검증한다.
- **후속 조치:** Tiny→Base+ 전용 paired state와 결과는 재사용하지 않는다. Small/Base+ checkpoint, prompt timeline, preprocessing, export SHA가 일치하는 paired state를 새로 수집한다. 기존 Tiny→Large pilot은 역사적 위험 분석으로만 보존한다.

## 10. 2026-09-19 GitHub Project 감사와 협업 자동화

- Project #2의 기존 20개 카드 본문·담당자·상태·날짜를 전부 확인하고, 비어 있거나 부족한 완료 기준을 보완했다. Scope Issue #1과 Benchmark 카드의 `Tiny→Base+` 오기도 `Small→Base+`로 수정했다.
- 연구 뼈대는 `01 Scope → 20 Submission`으로 확정하고 Tasks 기본 보기를 제목 오름차순으로 저장했다. 별도로 만들었던 Infra 카드의 요구사항은 06·07·11·18의 완료 기준으로 흡수한다. Issue와 PR은 실행·검토 증거이지 21번 이후 연구 단계가 아니다. 상세 판정은 [보드 감사·수정 기록](docs/project_board_audit_2026-09-19.md)에 남겼다.
- 완료 기준·실제 산출물·재현 명령·canonical Issue가 없으면 `Done`으로 끝낼 수 없는 `cmmt-task` 도구와 Issue/PR template를 구현했다. 로컬 CLI smoke와 5개 직접 테스트로 task 생성→진행→증거→체크→완료 흐름을 확인했다.
- 사용자가 저장소 이름을 `vos-memory-translator-nonlinear-v2`에서 `vos-memory-translator-nonlinear`로 변경했다. redirect는 정상이며 로컬 `origin`, package 이름, README, Pages 링크, Issue 검증 규칙을 canonical 이름으로 동기화했다.

## 11. 2026-09-19 연구 재개용 GitHub·실행 상태

- GitHub Project #2를 실제 UI에서 재확인했다. 01 Scope Issue #1과 02 state I/O Issue #2가 `In Progress`다. 2026-09-20 책임 경계를 바로잡아, 02는 무엇을 옮길지에 관한 계약·inventory·State Assembly Map까지만 담당하고, export/self-injection/target injection 구현과 edge-case continuation 검증은 별도 06 task로 추적한다.
- v2 `main`에는 scope·계획 문서가 있으나, L4 preflight, Base+ round-trip script, evidence-gated Issue/PR workflow는 `codex/project-board-workflow` branch의 `9eca421`, `ce9dd33`에 있다. 이 변경은 검증 뒤 PR로 main에 반영한다. main을 직접 변경하지 않는다.
- 2026-09-19 새 RunPod endpoint `157.157.221.29:43471`은 로컬 `~/.ssh/id_ed25519`(public fingerprint `SHA256:QxA6HrPgzjrkcUVMTx7vAIijjDs9wPxlnIRRLAN168E`)로 접속을 시도했으나 `Permission denied (publickey,password)`였다. host key 확인은 성공했지만 GPU·volume 검증과 Base+ round-trip은 실행하지 않았다. Pod의 등록 공개키와 이 fingerprint의 일치를 확인한 뒤 재시도한다.

## 12. 2026-09-19 Small/Base+ translator I/O 정적 계약

- GPU 보류 중 다음 비-GPU task로 pinned official SAM 2 config와 `SAM2Base`, 현재 canonical exporter/injector를 대조했다. Small과 Base+는 backbone 시작 차원은 96/112로 다르지만 memory encoder 64 channels, 기본 64×64 grid, object pointer 256, `num_maskmem=7`, `max_obj_ptrs_in_encoder=16` 경계가 같다.
- 당시 v1.0 계약은 `spatial_memory`, `object_pointer`, `presence_logits`를 연속 입력으로 기록하고 `pred_masks`를 continuation payload로 보존했다. 이는 2026-09-21 v1.1에서 바뀌었으며, 최신 정책은 본 문서 끝의 v1.1 결정과 I/O 계약을 따른다. Frame/slot/conditioning/validity/object ID/switch metadata를 복사·검증하고 Target PE를 생성한다는 원칙은 유지된다.
- Paired-state validator가 기존 frame/object/conditioning 검사에 더해 `schema_version`, `switch_frame`, `slot_order`, `validity`, video 크기 metadata 불일치를 fail closed하도록 강화했다. 실제 Small/Base+ dtype·shape inventory와 Base+ self-injection은 GPU가 생긴 뒤 확인한다.
- 상세 계약과 남은 runtime gate는 `docs/design/small_base_state_io_contract.md`에 기록했다.
- 대화 운영은 고정된 “다음 6개 답변” 카운트가 아니다. 한 프롬프트에 요청이 여러 개이면 각 항목을 `완료 / 진행 중 / Blocker / 미착수`로 나눠 병목을 보고한다.

## 13. 2026-09-19 canonical Issue와 PR

- [Issue #2](https://github.com/memorybridge-team/vos-memory-translator-nonlinear/issues/2)를 생성하고 `KIMKYUDO`에게 할당한 뒤 Project #2에 연결해 `In Progress`로 지정했다. 정적 memory boundary, field policy, fail-closed pair validation, CPU test는 완료됐고 실제 checkpoint runtime inventory와 Base+ same-checkpoint export→inject는 GPU가 확보될 때까지 미완료다.
- `codex/project-board-workflow`의 검증 커밋은 [PR #3](https://github.com/memorybridge-team/vos-memory-translator-nonlinear/pull/3)으로 제출했다. PR은 Issue #2를 `Refs`로만 연결했으므로 미완료 runtime gate를 자동으로 닫지 않는다. Tasks 보드가 20개 연구 단계만 보이도록 PR Project 카드는 보관했지만 PR 자체는 열려 있고 검토 이력은 유지된다.
- PR 생성 시 GitHub가 4 commits, 24 files changed, 1 contributor(`KIMKYUDO`)를 표시했고 merge conflict가 없음을 확인했다. 자동 병합은 하지 않았으며 `main`은 검토 전 상태를 유지한다.

사용자 승인 후 이 연구와 무관한 외부 저장소의 [`subinidus/cell-msca-odiac-estimation#2`](https://github.com/subinidus/cell-msca-odiac-estimation/issues/2)를 원본 변경 없이 연구 보드에서만 제거했다. 대신 이 저장소의 [범위 이슈 #1](https://github.com/memorybridge-team/vos-memory-translator-nonlinear/issues/1)을 생성해 보드에 연결했다. 이슈의 활동 기록에서 `KIMKYUDO`가 보드에 추가한 사실과 `Todo` 상태를 확인했다.

## 14. 2026-09-19 RunPod L4 환경 정보와 재개 gate

- **[확인]** 사용자가 실행에 사용할 Pod를 NVIDIA L4 1장(24 GB VRAM), vCPU 16개, RAM 62 GB, Container disk 20 GB, Network Volume 200 GB(`/workspace`) 구성으로 정했다. 2026-09-19 화면상 Network Volume의 사용량은 24 GB(12%)였다.
- **[구현]** `runpod_bootstrap.sh`를 Small/Base+ checkpoint를 준비하도록 수정하고, `runpod_preflight.sh`와 `runpod_base_plus_roundtrip.sh`를 추가했다. 이들은 Network Volume 여유 공간, revision, checkpoint SHA-256, CUDA를 기록하고 Base+ same-checkpoint export→inject를 검증한다.
- **[미검증]** 새 Pod의 SSH endpoint가 이 대화에 제공되지 않아 로컬 코드만 준비된 상태다. SSH 접속 명령을 받으면 bootstrap→preflight→Base+ round-trip을 실행한다. 현재 gate가 통과하기 전에는 Small→Base+ paired-state 수집이나 nonlinear 학습을 시작하지 않는다.

## 15. 2026-09-20 RunPod runtime gate와 Notion 실행 허브

- **[확인]** 새 endpoint `157.157.221.29:57478`에 로컬 `~/.ssh/id_ed25519`로 접속했다. 실제 장치는 NVIDIA RTX 2000 Ada Generation 16,380 MiB였으며, 이전에 전달된 L4 24 GB 화면과는 다른 현재 Pod다.
- **[확인]** `codex/project-board-workflow` commit `32dc1400b5b3b2cbbbbac11f4780010358a5f83b`, 공식 SAM 2 commit `2b90b9f5ceec907a1c18123530e92e794ad901a4`, Small/Base+ checkpoint SHA-256을 고정해 bootstrap과 preflight를 통과했다. 기본 `pytest`는 `scripts` import 경로 누락으로 collection 실패했지만 `PYTHONPATH=.`로 실행하면 50개 테스트가 모두 통과하므로 코드 실패가 아니라 실행 환경 경로 문제로 판정했다.
- **[확인]** DAVIS `walking`, object 1, switch frame 10에서 Base+→Base+ state export→inject를 실행했다. 이후 61개 frame의 native/injected 결과가 mean binary IoU `1.0`, mean MSE `0.0`, max absolute error `0.0`였고 injection 동안 과거 backbone call은 `0`이었다. wall time은 `132.53 s`, peak CUDA memory는 `974,874,624 bytes`였다. 단일 객체·첫 frame prompt 조건의 self-injection gate는 통과했다.
- **[제한]** 이 결과는 단일 객체·첫 frame prompt에서 state assembly 구현의 정확성을 입증하지만 Small→Base+ translator의 성공 증거는 아니다. Small/Base+ runtime inventory와 paired dump는 task 02 계약 근거다. multi-object, late prompt, absence/reappearance, prompt correction 및 복수 영상/switch 검증은 task 06 완료 조건이다.
- **[결정, 2026-09-20]** Project task 경계를 `01=왜·범위·판정 기준`, `02=무엇을 옮길지에 관한 state I/O 계약`, `06=어떻게 추출·조립·주입하고 continuation을 검증할지`로 고정했다. Base+ 필요성을 묻는 hard-event gate는 01에서 정의하지만 실제 pilot/evaluation 실행은 후속 실험 task가 담당한다.
- **[산출물]** 원본 JSON·상태·로그와 해석은 `reports/tasks/06_runtime/runs/2026-09-20_base_plus_self_injection/`에 보존했다.
- **[확인]** 동일 DAVIS case에서 실제 Small/Base+ canonical state를 각각 수집하고 paired cache를 생성했다. 두 모델 모두 spatial `[1,1,11,64,64,64]` bfloat16, pointer `[1,1,11,256]` float32, presence `[1,1,11,1]` float32와 동일한 discrete timeline을 보였다. 166,882,485-byte paired cache의 SHA-256은 `5e9bca17217d522335acf80a454834cf2beab22d4dd8f6204fcd221d2bf1a5f0`이며 원본은 RunPod volume, lightweight inventory는 `reports/tasks/02_state_io/runs/2026-09-20_runtime_inventory/`에 둔다. 이 결과로 단일 paired example gate는 통과했지만 여러 case와 interaction 조건의 반복 검증은 남아 있다.
- **[Notion]** 전달된 Documents database의 `모델API&실험` 그룹에 `CMMT 연구 실행 허브 — Project #2` 페이지를 만들고 Project·repository·PR·assembly map 링크, 01/02 Done gate, evidence 기록 규칙, 최신 self-injection 결과를 기록했다. Notion connector가 연결된 KNSW workspace에서는 이 guest workspace page를 `NOT_FOUND`로 반환해, 로그인된 Notion UI를 통해 작성 권한과 최종 내용을 검증했다.

## 16. 2026-09-20 Task 01·02 동결과 Notion 자동 기록 권한

- **[결정]** Task 01의 baseline을 전환 참조군, state 번역군, 객체별 anchor 재인코딩군, replay군, 진단/정확성 검사로 분리해 동결했다. 모든 비교군은 같은 등록 객체·prompt timeline·switch manifest를 사용하고 미래 GT를 입력으로 쓰지 않는다. `Original-Prompt(s)+Replay-{4,8,16}`은 모든 객체의 등록 정보를 보장하는 주요 경쟁군이다.
- **[결정]** 성공은 Base+-native의 전환 필요성, Direct/강한 anchor·replay 대비 downstream improvement, Full Replay 대비 비용 절감을 순서대로 판정한다. 무결성 오류는 즉시 수정 gate, Base+ 이득 부재·Direct 충분성·anchor/replay 지배·downstream 개선 부재는 주장 중단 또는 hybrid 전환 조건이다.
- **[완료]** `docs/design/01_scope_baselines_success_stop.md`와 `docs/design/small_base_state_io_contract.md`를 각각 Task 01·02의 frozen v1.0 산출물로 지정했다. Task 06의 edge-case continuation은 02 완료 조건에서 분리한다.
- **[지속 권한]** 사용자는 핵심 연구 산출물을 별도 요청 없이 Notion Documents의 `모델API&실험`에 자동 업로드하도록 승인했다. 단순 로그·cache는 올리지 않고 링크만 남기며, 다른 Notion 영역 수정 권한으로 확장하지 않는다.
- **[결정, 2026-09-20]** `Last-Mask`와 `Recent-Window Replay-1`은 switch 직전 source mask와 동일 RGB/frame을 target에 재인코딩하는 동일 입력 규칙일 때 같은 방법이다. 최종 baseline 표에서는 `Last-Mask (= Replay-1)` 한 행으로만 보고한다. 다른 seed·추가 prompt·target-native state를 쓰는 Replay-1은 별도 방법으로 기록한다. `All Original Prompts`와 `Translation + Short Replay`는 주 비교군이 아니라 interaction/hybrid 선택 분석으로 분리했다.
- **[결정, 2026-09-20 — 최신]** `Last-Mask (= Replay-1)`과 original anchor 없이 최근 RGB만 재처리하는 `Recent-Window Replay-k`는 객체 coverage가 약해 최종 비교군에서 제거했다. 최신 정보의 효과는 모든 등록 객체의 original prompt를 먼저 보장한 `Original-Prompt(s)+Replay-4/8/16`으로만 측정한다. 4·8·16은 각각 짧은, SAM 2 recent-memory 범위 근방의 중간, 더 긴 문맥을 나타내는 사전 고정 k 값이다.
- **[결정, 2026-09-21]** Direct Copy와 Nonlinear Translator 사이에 `Moment-Matched Copy`를 정식 baseline으로 추가했다. 학습 split의 paired state에서 conditioning/non-conditioning별·component별 평균과 표준편차를 고정한 뒤, `maskmem_features`와 `obj_ptr`만 affine 보정한다. test target-native state·future frame·test 통계는 금지한다. 이 비교군은 단순 분포 calibration만으로 충분한지 검증하며, nonlinear 방법의 필요성을 더 엄격하게 판정한다.

- **[결정, 2026-09-21]** State Assembly Map과 Small→Base+ I/O 계약을 v1.1로 통일했다. 번역 대상은 `maskmem_features`와 `obj_ptr`뿐이며, Target PE는 재생성한다. 과거 Source `pred_masks`는 Target `inference_state` 밖의 표시 sidecar archive, `presence_logits`/`object_score_logits`는 진단 기록으로만 남긴다. Target의 새 mask·score·memory는 새 frame 또는 correction replay에서만 생성한다.
- **[구현, 2026-09-21]** 로컬 materializer/injector에서 과거 `pred_masks`와 `object_score_logits` 주입을 제거하고 Target history를 `maskmem_features`·Target-generated `maskmem_pos_enc`·`obj_ptr` 세 필드로 제한했다. Source mask archive와 score diagnostic은 CanonicalState에만 남는다. 정적 compile과 CPU 계약 test를 추가했지만 로컬 Python에는 torch/pytest가 없어 전체 test 실행은 아직 못 했다.
- **[제한, 2026-09-21]** 고정 공식 구현상 `switch_frame + 1` 이후 memory read는 위 세 필드만 쓰지만, 과거 conditioning frame 출력·same-frame refinement는 `pred_masks`를 요구한다. 따라서 전환 이전 correction은 Target replay로 보내야 하며, 실제 checkpoint의 no-replay continuation·다객체·late prompt·부재/재등장·correction 검증 전에는 Task 06 또는 v1.1 runtime 구현 완료라고 주장하지 않는다.

## 17. 2026-09-21 — v1.1 최소 history strict runtime gate

- **[실패 원인 확인]** 과거 `pred_masks`/`object_score_logits`를 제외한 첫 Base+ self-injection 실행은 mean binary IoU `0.9999767`이었지만 max logit error `3.421504`로 strict gate에 실패했다. 주입 직전 비교에서 frame 0–9는 exact였고 최신 frame 10의 CPU-offloaded `maskmem_features`만 MSE `0.6701763`, max error `4.15625`였다.
- **[구현]** Pinned SAM 2가 최신 memory를 `non_blocking=True`로 GPU→CPU offload하므로, `canonicalize_sam2_inference_state`가 export 전에 producer CUDA device를 동기화하도록 수정했다. 이 race-condition fix에는 CPU 회귀 test와 실제 checkpoint history parity 진단 도구를 추가했다.
- **[확인]** 수정 뒤 DAVIS `walking`, object 1, switch 10의 11개 history record에서 `maskmem_features`, Target-generated `maskmem_pos_enc`, `obj_ptr`가 모두 bit-exact였다. 이후 61 frames의 Base+ native/injected logits도 mean MSE `0`, max error `0`, binary IoU `1.0`, injection 중 과거 backbone call `0`으로 strict gate를 통과했다.
- **[해석]** 이 결과는 v1.1 최소 read-state가 단일 객체·첫 frame prompt no-replay continuation에 충분하다는 구현 증거다. Small→Base+ 번역 성능 증거는 아니며, Task 06은 다객체·late prompt·absence/reappearance·correction·cross-model target injection이 남아 `In Progress`다.
- **[산출물]** `reports/tasks/06_runtime/runs/2026-09-21_base_plus_self_injection_after_sync/`에 원인 분석, 재현 명령, report JSON과 history diagnostic을 보존한다.

## 18. 2026-09-21 — Task 06 edge-case와 cross-model Direct Copy gate

- **[구현]** 서로 다른 frame에서 객체를 추가하는 prompt timeline round-trip runner와 CLI를 추가했다. Prompt event는 frame/object/mask 계약을 검증하고, 다음 prompt 직전까지만 propagation하도록 inclusive API 경계를 처리한다. RunPod 전체 test는 `52 passed`였다.
- **[확인]** DAVIS `bike-packing`에서 object 1@frame 0, object 2@frame 10, switch 20의 Base+→Base+ history 32 records를 주입했다. 후속 48 frames의 mean MSE `0`, max error `0`, mean binary IoU `1.0`, injection 중 과거 backbone call `0`이었다.
- **[확인]** DAVIS `india`의 object 3, switch 35 부재·재등장 사례에서도 후속 45 frames가 mean MSE `0`, max error `0`, mean binary IoU `1.0`, injection 중 과거 backbone call `0`이었다.
- **[Pilot]** DAVIS `walking`, object 1, switch 10에서 Small→Base+ Direct Copy는 11 history records를 replay 없이 기계적으로 주입했지만 Base+-native 대비 후속 61 frames의 mean binary IoU가 `0.0`이었다. Spatial-memory cosine은 `0.0211`, object-pointer cosine은 `-0.0220`이었다. 한 case 결과이므로 전체 일반화 결론은 아니지만, 동일 shape의 직접 복사만으로 표현 의미가 정렬되지 않으며 Moment-Matched/Nonlinear Translator 비교가 필요하다는 근거다.
- **[상태]** Task 06은 다객체·late prompt·부재/재등장·cross-model target injection gate까지 통과 또는 실행 증거를 확보했다. Prompt correction과 여러 sequence/switch 반복 검증이 남아 `In Progress`다.
- **[산출물]** `reports/tasks/06_runtime/runs/2026-09-21_edge_case_and_direct_injection/`에 세 원본 JSON과 해석을 보존한다.

## 19. 2026-09-21 — 연구용 최소 handoff payload와 동일 영상 전제

- **[결정]** 연구 단계에서는 Source와 Target이 동일 영상·동일 frame 순서·동일 preprocessing·동일 switch 시점을 사용한다고 manifest에서 고정한다. 연구 질문은 영상 식별 protocol이 아니라 memory translation에 한정한다.
- **[결정]** `num_frames`, `video_height`, `video_width`는 Target runtime이 자신의 영상에서 산출하며 CMMT handoff payload와 전송 bytes에서 제외한다. 로컬 loader/cache에서 쓰는 검사는 dataset 생성 오류를 찾는 assertion으로만 해석한다.
- **[결정]** `video_fingerprint`와 `prefix_fingerprint`는 CanonicalState나 translator API에 추가하지 않는다. 향후 서로 다른 장치·서버 사이의 서비스화에서 필요하면 model state 밖의 선택적 handoff envelope로 구현한다.
- **[고정 payload]** 번역 대상은 `maskmem_features`, `obj_ptr`이고, history 조립에는 `frame_indices`, `slot_order`, `is_conditioning`, `validity`, `object_ids`, `switch_frame` 등 필요한 이산 metadata만 전달한다. Target PE와 video runtime 값은 Target이 생성한다.

## 20. 2026-09-21 — Task 03 protocol과 협업 추적 보정

- **[확인]** Project #2의 Task 03·06은 `In Progress`지만 canonical repository Issue가 아닌 Draft 카드였다. 따라서 기존 Task 06 commit·보고서가 카드 활동과 직접 연결되지 않았다. 실험 결과는 유효하지만 협업 추적은 불완전했다.
- **[결정]** Task 03은 dataset/split 역할, `(dataset, release, split, video, object, switch)` manifest, difficulty taxonomy, baseline 입력, metric·통계·누수 규칙을 고정한다. Task 06은 export/inject 구현과 continuation closure만 담당하며 baseline evaluator 구현은 Task 08이 담당한다.
- **[문서]** `docs/design/03_benchmark_protocol.md`에 DAVIS 2017·MOSEv2·LVOS v2, Source-only/Base+-native/Direct/Moment-Matched/anchor/Replay-4·8·16/Nonlinear 비교군, visible·absence·reappearance·prompt/object/visual taxonomy, video-clustered CI를 정리했다.
- **[상태]** 데이터셋 이용조건/download snapshot, split manifest checksum, MOSEv2/LVOS loader·공식 metric 검증이 남아 Task 03은 `In Progress`다. Task 06은 prompt correction과 반복 switch 검증이 남아 `In Progress`다.
- **[업데이트 2026-09-23]** Task 03의 dataset snapshot, video-level split, 실제 loader 전수 검증, metric 명칭 구분을 완료했다. LVOS sparse frame ID 오류를 수정한 최종 manifest에서 DAVIS·MOSEv2·LVOS v2 train/validation 모두 `failure_count=0`이며 Task 03은 `Done`이다.
- **[협업 규칙]** 코드·실험·문서 task는 Draft로 두지 않고 canonical repository Issue로 추적한다. 진행 중 Draft 누락을 발견하면 기존 결과를 버리지 않고 Issue로 전환해 commit·보고서를 소급 연결한 뒤 다음 작업부터 정상 흐름을 따른다.

## 21. 2026-09-21 — 최소 payload의 코드·지표 반영

- **[구현]** Canonical exporter에서 `num_frames`, `video_height`, `video_width`, Source 내부 object index, prompt 입력과 `frames_tracked_per_obj`를 제거했다. Injector는 Target runtime의 영상 값과 Source metadata를 비교하지 않으며, 외부 `object_ids`로 registry를 만들고 prompt/tracking dictionary를 빈 값으로 시작한다.
- **[구현]** `handoff_bytes()`를 추가해 `maskmem_features`, `obj_ptr`, `frame_indices`, `slot_order`, `is_conditioning`, `validity`, `object_ids`, `switch_frame`만 센다. Presence·Source mask·prompt manifest·PE·영상 metadata/checksum은 제외한다.
- **[구현]** Ridge/Linear/Residual-MLP 학습 loss와 평가 aggregate를 spatial memory와 object pointer 두 component로 제한했다. Presence는 diagnostic metric과 archive에만 남는다. Residual-MLP artifact schema는 v2로 올렸다.
- **[해석]** 기존 Task 06 후속-mask 결과는 실제 Target memory read가 이미 두 translated field와 Target PE만 사용했으므로 유효하다. 기존 Direct JSON의 `translated_bytes`는 legacy metric이며 같은 11-record payload의 새 값은 `5,778,641 bytes`다.
- **[남은 검증]** 로컬 compile은 통과했으나 로컬 Python에 torch/pytest가 없어 전체 runtime test는 RunPod에서 재실행해야 한다. Task 06 Done 전 same-checkpoint smoke 1회를 계약 회귀 검사로 다시 실행한다.

## 22. 2026-09-21 — RunPod 최소 계약 회귀 검증 재개

- **[환경]** RunPod SSH `157.157.221.29:43472` 연결에 성공했고 NVIDIA L4 23,034 MiB와 CUDA 12.8, `torch.cuda.is_available=True`를 확인했다.
- **[확인]** RunPod에서 `PYTHONPATH=. pytest -q`를 실행해 전체 테스트 `52 passed`를 확인했다. 최초 실행의 `scripts` import 오류는 연구 코드 오류가 아니라 실행 경로 설정 문제였으며, 경로를 보정해 재실행했다.
- **[확인]** DAVIS `walking`, object 1, switch frame 10, Base+ same-checkpoint runtime smoke가 통과했다. 11개 history record 주입 후 future 61 frames에서 mean MSE `0`, max error `0`, mean binary IoU `1.0`, injection 중 과거 backbone call `0`이었다. peak CUDA memory는 `976,665,088` bytes, wall time은 약 `45.6 s`였다.
- **[해석]** 최소 handoff 계약의 same-checkpoint 구현 회귀는 통과했지만 Small→Base+ nonlinear translator의 성능을 의미하지 않는다. Task 06은 prompt correction·반복 switch·cross-model nonlinear injection이 남아 `In Progress`다.

## 23. 2026-09-21 — PR branch runtime smoke 재검증

- **[확인]** RunPod의 기존 미커밋 `main` 작업 트리를 보존하고 `/workspace/cmmt-pr` 별도 worktree에서 `origin/codex/minimal-handoff-contract` (`024760f`)를 checkout했다.
- **[확인]** PR branch 코드로 DAVIS `walking`, object 1, switch 10의 Base+ same-checkpoint smoke를 다시 실행했다. future 61 frames에서 mean MSE `0`, max error `0`, mean binary IoU `1.0`, injection 중 과거 backbone call `0`, peak CUDA memory 약 `0.97GB`, wall time 약 `45.4 s`였다.
- **[해석]** PR branch도 최소 handoff strict gate를 통과했다. 이는 구현 계약의 회귀가 없다는 증거이며, Small→Base+ translator 품질이나 Task 06 전체 완료를 의미하지 않는다.

## 24. 2026-09-21 — PR branch prompt correction gate

- **[확인]** PR branch에서 DAVIS `walking`에 object 1을 frame 0과 frame 5에 다시 prompt하고 switch frame 10을 적용한 same-checkpoint prompt-timeline round-trip을 실행했다.
- **[확인]** Target injection 후 future 61 frames에서 mean MSE `0`, max error `0`, mean binary IoU `1.0`, peak CUDA memory 약 `0.98GB`, wall time 약 `46.5 s`로 통과했다.
- **[해석]** 동일 checkpoint의 prompt correction history도 최소 handoff 계약으로 보존된다. 이는 cross-model nonlinear translator 성능이 아니라 Task 06 구현 gate 증거다. 여러 sequence/switch 및 Small→Base+ injection은 여전히 남아 있다.

## 25. 2026-09-22 — Task 06 반복 switch 검증

- **[확인]** 최신 `task/02-06-minimal-handoff-contract` (`94a818b`)를 RunPod 별도 worktree에서 실행했다. DAVIS `walking`, object 1, switch 30과 DAVIS `india`, object 3, switch 35의 Base+ same-checkpoint round-trip을 각각 수행했다.
- **[결과]** 두 실행 모두 injection 중 과거 backbone call `0`, mean MSE `0`, max error `0`, mean binary IoU `1.0`이었다. `walking`은 후속 frame 31–71, wall time `41.39 s`, peak CUDA memory 약 `0.97GB`; `india`는 후속 frame 36–80, wall time `44.73 s`, peak CUDA memory 약 `0.97GB`였다.
- **[해석]** 최소 handoff state의 same-checkpoint continuation이 하나의 switch 위치에만 우연히 맞은 결과는 아니라는 증거가 추가됐다. 다만 cross-model Nonlinear Translator, 더 넓은 video/switch matrix, benchmark dataset manifest·official metric은 별도 gate로 남는다.

## 27. 2026-09-23 — Task 06 correction·반복 handoff 마감

- **[환경]** RunPod RTX A4000 16GB에서 공식 SAM 2.1 Base+ checkpoint(`a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5`)와 upstream `2b90b9f5ceec907a1c18123530e92e794ad901a4`를 사용했다.
- **[전환 후 correction]** DAVIS `walking`, switch 10, correction 20에서 replay 없이 correction을 Target이 처리했다. frame 20–71의 52개 결과가 native와 MSE `0`, max error `0`, binary IoU `1.0`이었고 injection backbone call은 `0`이었다.
- **[전환 전 correction]** switch 20, correction 10은 prompt anchor 0부터 frame 20까지 21 frames를 Target으로 replay했다. frame 21–71의 51개 결과가 native와 exact였다.
- **[반복 handoff]** switch 10→20의 두 injection 모두 backbone call `0`이었고, 첫 전환 뒤 61 frames와 두 번째 전환 뒤 51 frames가 모두 exact였다.
- **[결함 발견·수정]** v1.1 injected history는 진단용 `object_score_logits`를 의도적으로 제외하지만 exporter가 재-export 때 이를 필수로 요구했다. continuation 필수 field를 `maskmem_features`·`obj_ptr`로 바로잡고, 누락 진단 record는 `missing_presence_records` metadata에 기록했다. score는 payload나 Target history에 다시 넣지 않았으므로 v1.1 계약은 바뀌지 않는다.
- **[검증]** 수정 후 RunPod 전체 test `57 passed`. 원본은 `reports/tasks/06_runtime/runs/2026-09-23_correction_and_repeated_switch/`에 보존한다.
- **[상태]** 단일/다객체·late prompt·부재/재등장, 전환 전후 correction, 반복 handoff, Small→Base+ target injection 실행까지 확보해 Task 06을 `Done`으로 판정한다. Direct Copy의 실패는 translator 성능 결론이 아니라 Task 08 baseline pilot 증거다. 다음 단계는 Task 07 paired-state 수집이다.

## 26. 2026-09-22 — Task 03 DAVIS validation manifest 고정

- **[확인]** DAVIS 2017 trainval 480p validation split에서 seed 7, regular quantile switch와 GT 기반 diagnostic event tag 정책으로 deterministic evaluation manifest를 생성했다.
- **[결과]** `manifests/davis2017_val_v1.json`은 30 sequences, 249 video/object/switch cases이며 file SHA-256은 `5b036f173d74e6939c3096fa03e0af64f3dfba8bdaff21a43e1db8f5c1ada2f5`다. JSON에는 원본 pixel이나 dataset root가 포함되지 않는다.
- **[해석]** GT는 case selection·difficulty tagging에만 쓰며 model input, handoff payload, translator 학습은 보지 않는다. MOSEv2/LVOS v2의 official split·loader·metric 검증은 Task 03에 남아 있다.

## 28. 2026-09-24 — Benchmark protocol v1.1 확정

- **[결정]** 데이터 운영을 translator fit, in-domain development, sealed in-domain final,
  external frozen benchmark의 네 역할로 분리한다.
- **[결정]** 주 translator는 MOSEv2/LVOS v2 train-fit으로 학습하고 각 train의
  video-disjoint dev에서만 구조·loss·epoch·threshold·replay-k·checkpoint를 선택한다.
- **[결정]** MOSEv2 official valid와 LVOS v2 official val은 config freeze 뒤 여는
  sealed in-domain final이다.
- **[결정]** VOST val/test는 primary translator-level cross-dataset zero-shot benchmark다.
  DAVIS val은 `walking`, `bike-packing`, `india`의 Task 06 개발 노출 때문에
  `engineering-seen external`로 표시한다.
- **[금지]** DAVIS/VOST는 gradient, train statistics, Moment-Matched target moments,
  early stopping, threshold 또는 후보 선택에 사용하지 않는다. 첫-frame GT prompt와
  metric용 GT만 표준 VOS 평가로 허용한다.
- **[결과 구조]** `MOSE-only`, `LVOS-only`, `MOSE+LVOS` 학습 행과 MOSE/LVOS/DAVIS/VOST
  평가 열을 보고한다. VOST-train fine-tuning은 별도 adaptation upper-bound ablation이다.
- **[상태]** Task 03 v1.0 완료 증거는 보존한다. VOST license/download/checksum,
  25/50/75% switch manifest, loader와 공식 `J/J_last`, external access ledger가 남아 있어
  protocol v1.1의 Project 상태는 다시 `In Progress`다.

## 29. 2026-09-24 — 팀원용 진행 현황 정보 구조와 Project 동기화

- **[결정]** GitHub Project는 상세 연구 문서의 저장소가 아니라 상태·담당·순서·blocker와 canonical evidence 링크를 모으는 관제판으로 사용한다.
- **[읽기 경로]** 새 팀원은 `README → docs/research_progress_summary.md → Project의 In Progress 카드 → 연결 Issue → PR·reports` 순서로 현재 상태를 파악한다.
- **[정보 위치]** 안정된 연구 개요는 README, 검증된 snapshot은 research progress summary, task 범위·완료 조건·논의는 Issue, 변경 검토는 PR, 실행 명령·정량 결과·한계는 `reports/`에 둔다.
- **[Project 반영]** Task 07·08·09·10·11·13의 설명을 protocol v1.1에 맞게 수정했다. MOSE/LVOS fit·dev만 학습/선택에 사용하고, official validation은 config freeze 뒤 sealed final, VOST는 primary external zero-shot, DAVIS는 engineering-seen external로 제한한다.
- **[상태]** Task 03은 VOST manifest·loader·공식 `J/J_last`·access ledger가 남아 `In Progress`; Task 07 이후는 `Todo`를 유지한다.
- **[업데이트 2026-09-24]** VOST 공식 Data page·S3 archive·CC BY-NC-SA 4.0 이용조건과 TRI-ML 평가 코드 위치를 확인해 access ledger를 만들었다. archive 다운로드·SHA-256·inventory·25/50/75% manifest·실제 `J/J_last` 실행은 아직 남아 있다.
- **[진행 2026-09-24]** 공식 VOST archive의 `Content-Length=54,012,104,924` bytes를 확인하고 RunPod `/workspace/datasets/VOST/VOST.zip`에 `wget --continue` 다운로드를 시작했다. 완료 전 checksum·inventory·loader gate는 미완료로 유지한다.
- **[진행 2026-09-25]** VOST archive SHA-256 `fb17075ab3afab0fe30f264d8adce2e29ce6249a73cd44d2a9cf4936cc8de978`를 계산했다. ZIP 목록에서 153,136개 파일과 train/val/test 572/70/71 sequence를 확인했으며, 압축 해제 후 실제 annotation·frame inventory와 loader 검증이 남아 있다.
- **[진행 2026-09-25]** VOST 압축 해제를 `/workspace/datasets/VOST/extracted`에서 완료했다. 1차 inventory는 Annotations 67,751개, JPEGImages 67,751개, JPEGImages_10fps 15,607개, Videos 642개이며, annotation/frame 대응·25/50/75% switch manifest·prompt loader·공식 `J/J_last` smoke가 남아 있다. Task 03은 이 gate들이 끝날 때까지 `In Progress`다.
- **[검증 2026-09-25]** VOST inventory에서 train 59,930쌍과 val 7,820쌍의 frame/annotation stem 대응이 모두 통과(`failure_count=0`)했다. 배포본의 test는 71개 sequence 이름 목록만 있고 로컬 frame/annotation이 없어 공식 서버용으로 분리한다. 다음 gate는 val 기준 switch manifest·prompt loader·공식 `J/J_last` smoke다.
- **[산출물 2026-09-25]** VOST val 70 sequence에 25/50/75% temporal quantile을 적용한 210-case switch manifest를 생성했다. content SHA-256은 `47bab054c87281e7b904831ea8891b81d22f1d8683786f1d5cf84a4f1ba85cbf`다. 이는 prompt/예측 loader와 공식 `J/J_last` 실행 전의 metadata gate이며 Task 03은 아직 `In Progress`다.
- **[검증 2026-09-25]** TRI-ML/VOST official evaluator commit `fe274574`를 GT-copy·512px smoke에서 실행해 `J-Mean=1.000`, `J_last-Mean=1.000`을 확인했다. `J_last`가 공식 코드의 실제 후반 구간 지표 이름이다. 원본 해상도 119-frame smoke는 약 60GB에서 `SIGKILL`돼, 최종 평가는 sequence 단위 메모리 상한·CSV 병합 방식으로 운영한다. CMMT PNG export smoke는 남아 있다.
- **[완료 2026-09-25]** VOST val `9671_split_cups`를 numeric symlink staging으로 SAM 2.1 Small에 실제 입력해 state export(frames 0–20)를 만들고, 42개의 prediction PNG를 official VOST frame stem layout으로 export했다. GT와 prediction PNG stem은 42/42로 exact match다. Task 03 v1.1의 VOST data·manifest·loader·evaluator contract gate는 완료됐으며, full baseline/translator scores는 Task 08/09/13으로 넘긴다.

## 30. 2026-09-24 — Task 중심 보고서 구조 확정

- **[결정]** 연구 보고서는 category-first가 아니라 Project Board와 직접 대응하는
  `reports/tasks/NN_short_name/` 구조를 기준으로 관리한다.
- **[구조]** Task `README.md`는 현재 상태·핵심 결과·남은 gate의 인덱스이고,
  `runs/YYYY-MM-DD_slug/`는 실행 명령·원본 수치·JSON·실패를 보존한다. 중간 동결본은
  `milestones/`, Done 전 통합 결과는 `FINAL_REPORT.md`에 둔다.
- **[완료]** Task 02와 06의 최종 통합본을 만들고 기존 runtime·benchmark 증거를
  Task 02·03·06 디렉터리로 이동했다. 중복을 피하기 위해 이전 category-first 호환 경로는
  제거하고 `reports/tasks/`만 공식 경로로 사용한다.
- **[완료 절차]** 날짜별 증거 → `FINAL_REPORT.md` → Issue 완료 기준별 링크 → PR
  test/review·main 병합 → Issue checklist → Project Done 순서로 닫는다.
- **[읽기 경로]** `README → research_progress_summary → Project → canonical Issue →
  Task 보고서 → PR → 날짜별 run` 순서로 안내한다.

## 31. 2026-09-24 — RunPod Runtime 중간 변경 감사와 정리

- **[원인]** RunPod checkout은 `226790d`에 머문 채 Task 06 완료 과정의 중간 코드가 미커밋 상태로 남아 있었다. GitHub `main`은 이미 PR #8 병합 commit `01889b3`까지 전진해 있었으므로, Board의 Done과 RunPod working tree가 어긋난 것은 GitHub 완료 누락이 아니라 오래된 checkout 문제였다.
- **[판정]** 중간 코드는 `object_score_logits`를 다시 필수로 요구하고 영상 크기·prompt/tracking metadata를 handoff state에 포함해 현재 최소 계약을 되돌렸다. 최신 `main`에는 correction·repeated handoff와 이 회귀를 막는 테스트가 이미 포함되어 있어 별도 병합 가치가 없었다.
- **[처리]** 변경을 임시 snapshot commit `a6ec48c`으로 보존한 뒤 RunPod `main`을 `01889b3`으로 fast-forward했다. 독립 가상환경 `/workspace/.venvs/cmmt-runtime-audit`에서 전체 테스트 `57 passed`를 확인하고 임시 branch를 삭제했다.
- **[현재 상태]** RunPod `/workspace/vos-memory-translator-nonlinear`은 clean `main`이며, Task 03 PR과 후속 실험을 기존 중간 코드와 섞지 않고 진행할 수 있다.

## 32. 2026-09-25 — DAVIS 활성 범위 제외와 Task 03 종료 유보

- **[결정]** DAVIS 2017은 SAM 2 학습 데이터 노출 및 Task 06 engineering 개발 노출이
  결합되어 현재 translator 연구의 학습·평가·표·주장 범위에서 제외한다. 기존 DAVIS runtime
  smoke와 pilot은 구현 통로를 검증한 역사적 증거로만 보존하며, 새 결론에 사용하지 않는다.
- **[결정]** 활성 데이터 역할은 MOSEv2/LVOS v2 train-fit·video-disjoint dev,
  MOSEv2 official valid·LVOS v2 official val sealed final, VOST external zero-shot이다.
- **[상태]** VOST archive·manifest·loader·official evaluator contract는 완료했으나,
  사용자가 추가 검증 데이터셋을 요청했으므로 Task 03은 `In Progress`로 유지한다. 그 데이터셋의
  이름·연구 역할·사용 가능한 GT/공식 evaluator는 아직 미확정이다.

## 33. 2026-09-25 — Translator train/dev/final 평가 역할 정정

- **[결정]** MOSEv2·LVOS v2 train의 video-disjoint fit/dev에서만 Translator를
  학습·선택하고, 코드·config·checkpoint·manifest·evaluator를 동결한 뒤 final/external
  평가를 실행한다.
- **[정정]** MOSEv2 official validation은 첫-frame GT만 공개되고 후속 GT는 숨겨져 있으므로
  local detailed final이 아니다. Frozen prediction을 Codabench에 제출해 서버가 반환한 공식
  aggregate만 최종 표에 사용한다. Local GT 기반 switch J&F·recovery·identity 수치는
  주장하지 않으며, latency·VRAM·replay·bytes처럼 GT가 필요 없는 지표만 추가할 수 있다.
- **[결정]** LVOS v2 validation은 공개 전체 GT로 official J/F/J&F와 CMMT switch/recovery
  세부 지표를 계산하는 local in-domain final이다. VOST validation은 freeze 뒤 한 번 실행하는
  primary translator-level external zero-shot이며 공식 `J/J_last`와 CMMT 세부 지표를 보고한다.
- **[선택 사항]** VOST train은 main fit/dev에 넣지 않는다. Freeze 뒤 공개 train+validation
  642개를 supplementary zero-shot stress test로 평가할 수 있으나, 기존 연구와 직접 비교하는
  공식 숫자는 validation 70개를 별도로 보고한다. VOST-train fine-tuning은 adaptation
  upper-bound로 분리한다.
- **[문서]** 독립적인 결정·후속 동기화 지침은
  `docs/design/03_translator_train_dev_final_evaluation_decision.md`에 기록했다. 기존 protocol,
  실행 계획, Task 03/07/08/09/13 Issue·Project 설명은 이 결정 문서를 기준으로 별도 PR에서
  동기화해야 하며, 이 기록만으로 외부 보드나 기존 문서가 수정됐다고 간주하지 않는다.

## 34. 2026-09-26 — State-only fit과 PUMaVOS external stress 확정

- **[판단]** Cross-Model KV Cache Transfer 논문처럼 paired internal state만으로 mapper를
  맞추는 접근은 CMMT에도 적용 가능하다. 동일 prefix의 Small state `b_T`와 frozen Base+
  native state `a_T`가 supervision이므로 future dataset GT는 primary state-only fit에 필수가 아니다.
- **[차이]** KV 논문은 `k`를 네 downstream benchmark 평균으로 선택하고 그 지표도 결과에
  포함했으므로 엄격한 train/validation/test 분리 사례는 아니다. CMMT는 더 엄격하게
  future-GT-free fit, video-disjoint dense-GT dev selection, sealed/external final로 분리한다.
- **[결정]** Primary nonlinear Translator는 state-only로 학습한다. Target-native future
  logit distillation은 GT-free ablation, dataset GT supervised rollout은 train-fit GT만 쓰는
  별도 ablation으로 보고한다. Primary paired-state switch는 GT 비의존 temporal rule을 쓴다.
- **[결정]** PUMaVOS는 논문·project page 기준 24 videos·21,187 dense frames·30 FPS이며 공식 train/val/test split이
  없으므로 validation split이라 부르지 않는다. 공식 공개 archive 전체를 config freeze 뒤 한 번 실행하는
  secondary external zero-shot stress test로 사용한다. 객체별 first-nonempty GT mask 한 장만
  prompt로 쓰고 미래 GT는 평가에만 사용한다.
- **[상태]** VOST는 primary external을 유지한다. PUMaVOS download/checksum·inventory·loader·
  fixed manifest·J/F/J&F contract가 Task 03의 새 남은 gate이며, 완료 전 Done으로 옮기지 않는다.
- **[출처 불일치]** 현재 XMem2 GitHub README overview에는 PUMaVOS를 23 videos라고 쓴 문장이
  있어 논문·project page의 24 videos와 충돌한다. 다운로드한 공식 archive의 sequence/frame/object
  inventory와 checksum으로 실제 평가 수량을 확정하고, 차이를 Task 03 보고서에 기록한다.
- **[보류]** RunPod `/workspace/CMMT`의 DAVIS 원본·과거 산출물 13개 경로는 사용자의 지시에
  따라 삭제하지 않고 그대로 보존한다.

## 35. 2026-09-26 — M³-VOS external zero-shot과 boundary F 정책 확정

- **[결정]** M³-VOS를 MOSEv2/LVOS v2로 학습·선택한 Translator의 **material phase-transition
  external zero-shot stress benchmark**로 추가한다. VOST는 primary external을 유지하고,
  PUMaVOS와 M³-VOS는 서로 다른 실패 조건을 보는 complementary secondary benchmark다.
- **[공식 규모]** 최신 project page와 arXiv v3 기준 M³-VOS는 479 high-resolution videos와
  205,181 dense masks/frames를 제공한다. 배포본의 split 이름과 실제 video/object 수는
  download checksum·inventory로 다시 고정한다.
- **[누수 금지]** M³-VOS는 gradient, normalization/statistics, architecture/loss/checkpoint,
  threshold 또는 replay-k 선택에 쓰지 않는다. Config/checkpoint freeze 뒤 first prompt만 입력하고
  미래 GT는 채점에만 사용한다.
- **[주 지표]** 논문 본문의 M³-VOS 주지표는 원 논문과 직접 비교 가능한 `J`, `J_tr`(마지막
  25% frame), `J_cc`(connected-component averaged Jaccard)로 사전 고정한다.
- **[F 정책]** Boundary `F`와 `J&F`는 결과를 본 뒤 유불리에 따라 제외하지 않는다. Void mask
  처리, GT-copy identity, shard/monolithic 일치, export/reload 무결성, native-resolution 및
  controlled resize·1-pixel morphology 민감도 검사를 먼저 수행한다. 이 integrity gate를 통과해도
  M³-VOS의 `F/J&F`는 비공식 보조·부록 지표이며, 통과하지 못하면 engineering report에만
  원인과 함께 남긴다. 본문 주결론은 항상 `J/J_tr/J_cc`로 낸다.
- **[상태]** M³-VOS access/licensing ledger, archive checksum·inventory, official full/core split,
  void-aware loader, first-prompt manifest, `J/J_tr/J_cc` evaluator와 F integrity gate가 Task 03의
  새 남은 gate다. Task 03은 PUMaVOS와 M³-VOS onboarding 완료 전까지 `In Progress`다.
- **[근거]** <https://zixuan-chen.github.io/M-cube-VOS.github.io/>,
  <https://arxiv.org/abs/2412.13803>,
  <https://github.com/zixuan-chen/M3VOS_Experiment/blob/main/docs/EVALUATION.md>

## 36. 2026-09-27 — 브랜치 명명 규칙 정비

- **[근거]** 사용자가 공유한 브랜치·PR 협업 글을 검토했다. 공통 원칙은 `main` 직접 작업을
  피하고, 목적이 드러나는 소문자 하이픈 브랜치에서 작은 commit을 만든 뒤 PR review와
  checks를 거쳐 병합하는 것이다. 참고: <https://suhanlim.tistory.com/262>,
  <https://su-devlog.tistory.com/5>.
- **[결정]** 이후 브랜치는 `feature/task-번호-범위`, `docs/task-번호-범위`,
  `experiment/범위`, `fix/범위`, `chore/범위` 중 작업 유형에 맞는 접두어를 사용한다.
  이름은 소문자 ASCII·하이픈만 사용하고 한 브랜치에 한 작업만 담는다.
- **[정정]** 기존 PR #10의 원격 head를 저장소 권한 없이 rename할 수 없으므로
  `task/03-external-zero-shot-protocol-v1-3`을 PR의 canonical 작업 브랜치로 유지한다.
  임시로 만든 `feature/task-03-external-zero-shot-protocol-v1-3`은 삭제하고, 다음 새
  작업부터 `feature/`, `docs/`, `experiment/`, `fix/`, `chore/` 규칙을 적용한다.
