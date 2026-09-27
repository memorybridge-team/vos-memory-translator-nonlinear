# Task 03 — External onboarding sources and storage policy

작성일: 2026-09-27

## 확인된 공식 접근 경로

| Dataset | 공식 source | 현재 상태 |
|---|---|---|
| PUMaVOS | [XMem2 repository](https://github.com/mbzuai-metaverse/XMem2), [official archive](https://drive.google.com/file/d/1VAClrxhWWiu9Y39QtcoUhp2YBN7R_ZCD/view?usp=sharing), [separate sequences/masks folder](https://drive.google.com/drive/folders/1Q7gSCCgemUyweu-7-Yb9G_W55Muq5-bC) | 파일 ID 확인(2026-09-27); 다운로드 전 archive 크기·checksum 확인 필요 |
| M³-VOS | [Project page](https://zixuan-chen.github.io/M-cube-VOS.github.io/), [Hugging Face data card](https://huggingface.co/datasets/Lijiaxin0111/M3_VOS), [official evaluation instructions](https://github.com/zixuan-chen/M3VOS_Experiment/blob/main/docs/EVALUATION.md) | project page·evaluation 구조 확인; media archive 크기·checksum·inventory 미확정 |

PUMaVOS는 공식 repository 설명과 논문·project page 사이에 video 수 표기 차이가 있으므로,
다운로드 후 archive inventory를 최종 기준으로 삼는다. M³-VOS는 evaluation 문서가
`JPEGImages`, `Annotations`, `Videos`, `ImageSets/val.txt`, `meta` 구조를 요구하므로
이 구조와 실제 배포본을 대조한다.

## 접근 경로 검증 기록

- PUMaVOS 공식 XMem2 README의 `.zip` 링크는 `PUBLIC_PUMaVOS.zip` 파일 ID
  `1VAClrxhWWiu9Y39QtcoUhp2YBN7R_ZCD`로 확인했다(2026-09-27, 웹 페이지 접근 확인).
- 동일 README의 별도 sequences/masks 폴더 ID는
  `1Q7gSCCgemUyweu-7-Yb9G_W55Muq5-bC`다. 두 링크 모두 공개 접근 경로지만, 현재
  RunPod storage에 내려받지는 않았으며 파일 크기와 SHA-256은 다운로드 후 기록한다.
- 2026-09-27에 공개 Drive 파일 페이지의 HTTP HEAD/HTML metadata를 확인했다. 페이지는
  `200 OK`로 접근되지만 `Content-Length`와 archive byte size를 제공하지 않아, 실제
  크기·checksum은 다운로드 또는 Drive UI에서 파일 크기를 확인해야 한다. 따라서 이번
  단계에서는 다운로드를 시작하지 않았다.
- 이후 1-byte range request로 official archive의 정확한 content range
  `0-0/3,008,102,259` bytes를 확인하고 RunPod에 다운로드했다. 다운로드본 SHA-256은
  `ccd062636b0422055d1da7344411726b4fd1d74d490e68b50618c34ca9a087a4`이며,
  `unzip -t`는 42,425 archive entries에 대해 오류 없음으로 통과했다. Archive central
  directory는 `JPEGImages`와 `Annotations`에 각각 21,212개 entry(디렉터리 entry 포함)를
  기록한다. 압축 해제 후 실제 file/sequence 수와 paired-stem validator 결과가 최종
  protocol 수량이다.
- 압축 해제 후 `validate_pumavos_inventory.py`를 RunPod extracted root에 전수 실행했다.
  결과는 **24 sequences, 21,187 RGB frames, 21,187 annotations, `failure_count=0`**이다.
  모든 sequence에서 `.jpg`와 `.png`의 exact stem 대응을 확인했다. 따라서 이 보고서와
  이후 manifest는 실제 official archive 기준의 24 sequences를 사용하며, XMem2 README의
  23-video overview는 historical documentation discrepancy로 기록한다. 결과 JSON은
  RunPod `/workspace/CMMT/reports/task03/pumavos_inventory.json`에 보관했다.
- PUMaVOS metric contract smoke는 local evaluator checkout의
  `davis2017/metrics.py` (SHA-256
  `a71bfb6d2da563ebf50251bda9876cf0b4b6193842543b0a3a286190529e26be`)로 수행했다.
  `billie_hair`, object `1`의 GT-copy 첫 5 frames는 각 frame에서 `J=1.0`, `F=1.0`,
  `J&F=1.0`이었다. 이것은 label/frame/evaluator compatibility 검사일 뿐 CMMT model
  성능 결과가 아니다.
- M³‑VOS project page는 Hugging Face data card를 공식 데이터 링크로 노출한다. 카드의
  현재 viewer는 `test` split 530행의 `video_id`, `obj_id`, phase-transition metadata와
  첫 frame 경로를 보여주지만, 이 화면만으로 479개 video media와 dense mask archive가
  모두 내려받아졌다고 판단할 수 없다. 공식 evaluation 문서가 요구하는
  `JPEGImages/Annotations/Videos/ImageSets/meta` 구조와 실제 media 배포본을 별도로
  확인한다.
- Hugging Face repository API가 공개한 M³‑VOS `usedStorage`는 `56,663,215,406` bytes
  (약 56.7 GB, decimal)이며 commit은
  `5deb15b2baeaaa294ca168b789537729f7fb53a5`다. 이 크기는 500GB volume의 staged
  onboarding 범위에는 들어가지만, PUMaVOS와 full paired-state를 동시에 보존할 여유를
  보장하지 않는다.
- RunPod의 resumable Hugging Face 다운로드는 annotation 파일 1,873개를 받은 뒤, 병렬도를
  16으로 높인 재개 요청에서 공개 API HTTP `429` rate limit을 받았다. 기존 파일은
  보존하며 인증 우회나 반복 요청은 하지 않는다. 기존 project page의 Google Drive folder
  `1qNSvE6dpkCHSs_8eZRo6vruLScCHl7oI`도 2026-09-27에 `404 Not Found`였다. 따라서
  M³‑VOS는 낮은 병렬도의 재개가 허용되거나 저자가 유효한 공식 mirror를 제공할 때까지
  checksum·inventory gate가 막힌 상태다.
- 공식 evaluator repository HEAD `8cf8f9b3cb069d8476ef6c3c0b8f11b8337c3b56`를
  읽어 `J`, 마지막 25%의 `J_last`(CMMT 문서의 `J_tr` 대응), connected-component
  matching `J_cc`를 계산함을 확인했다. `Dataset.get_all_masks()`는 label `255`를
  void로 분리하지만, 현재 `Evaluation.evaluate()`의 호출은 void mask 인자에 `None`을
  넘긴다. 따라서 CMMT의 void-aware 구현은 공식 결과와 GT-copy에서 대조하고 차이가
  있으면 양쪽 규칙을 분리 기록하는 integrity gate가 필요하다.

## 보존·삭제 정책

1. 원본 압축파일은 checksum과 압축 구조 검증이 끝나면 삭제한다.
2. 압축 해제본은 현재 평가에 필요한 split만 보존한다.
3. paired-state 생성 중에는 원본 RGB를 삭제하지 않는다. loader·재수집·split 오류 수정에
   필요하기 때문이다.
4. paired-state shard의 checksum과 manifest가 검증되고 재생성이 가능하다고 확인되면,
   원본은 삭제하지 않고 오래된 중간 cache·중복 shard·raw prediction만 삭제한다.
5. 최종 결과 재현을 위해 원본 데이터 자체보다 `dataset checksum`, split manifest,
   loader version, state-shard checksum을 보존한다. 원본은 외부 archive에서 재취득할 수
   있어야 하며, 이용조건상 재배포하지 않는다.

현재 500GB에서는 MOSEv2/LVOS v2/VOST extracted와 paired-state를 동시에 모두 보존하지
않는다. fit/dev shard를 순차 생성하고 checksum 검증 후 중간 산출물을 정리한다.
