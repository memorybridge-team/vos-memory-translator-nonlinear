# 프로젝트 문맥: SAM 2.1 Small → Base+ nonlinear state handoff

> 확인된 코드·기존 실험과 현재 계획을 구분한 시작 문서. 기준일: 2026-09-21 KST.

## 1. 연구 질문

비디오 프레임 `t`까지 SAM 2.1 Small이 추적한 객체별 state `b_t`를 작은 nonlinear translator `T`로 Base+용 state `â_t=T(b_t)`로 바꾸면, Base+가 과거 영상을 전부 다시 처리하지 않고 `t+1`부터 정확히 추적할 수 있는가? Base+가 같은 과거를 직접 처리한 state는 `a_t`라고 쓴다.

목표는 tensor MSE 자체보다 후속 J&F, 객체 유지·재등장, 부재 중 오검출, 전환 지연·전달량 개선이다. Source-only와 강한 prompt/mask/replay 대안보다 실익이 있어야 연구 필요성이 성립한다.

## 2. 확정 범위

- Model: 공식 SAM 2.1 Small → Base+ 한 방향. 같은 revision의 코드와 각각의 checkpoint를 사용한다.
- Dataset: DAVIS 2017로 구현과 초기 검증, MOSEv2로 복잡한 장면·재등장, LVOS v2로 장기 추적을 평가한다. 세 데이터셋은 논문 범위에 포함된다.
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
- **[산출물]** 원본 JSON·상태·로그와 해석은 `reports/runtime/2026-09-20_base_plus_self_injection/`에 보존했다.
- **[확인]** 동일 DAVIS case에서 실제 Small/Base+ canonical state를 각각 수집하고 paired cache를 생성했다. 두 모델 모두 spatial `[1,1,11,64,64,64]` bfloat16, pointer `[1,1,11,256]` float32, presence `[1,1,11,1]` float32와 동일한 discrete timeline을 보였다. 166,882,485-byte paired cache의 SHA-256은 `5e9bca17217d522335acf80a454834cf2beab22d4dd8f6204fcd221d2bf1a5f0`이며 원본은 RunPod volume, lightweight inventory는 `reports/runtime/2026-09-20_small_base_runtime_inventory/`에 둔다. 이 결과로 단일 paired example gate는 통과했지만 여러 case와 interaction 조건의 반복 검증은 남아 있다.
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
- **[제한, 2026-09-21]** 현재 legacy materializer/injector는 과거 `pred_masks`와 `object_score_logits`를 Target history에 쓰는 v1.0 동작을 아직 포함한다. 이 구현은 v1.1 목표 계약과 다르므로, Task 06에서 해당 주입을 제거하고 다객체·late prompt·부재/재등장·correction continuation 검증을 통과하기 전에는 v1.1 구현 완료라고 주장하지 않는다.
