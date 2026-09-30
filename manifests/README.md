# Frozen evaluation manifests

이 디렉터리는 데이터셋의 원본 이미지·마스크를 저장하지 않는다. 같은 dataset/split/video/object/switch 조합을 반복해서 평가하기 위한 작은 JSON manifest만 version control 한다.

Translator fit/development에는 MOSEv2/LVOS v2 train만 사용한다. DAVIS manifest와
DAVIS runtime evidence는 historical archive에만 보존하며 현재 main의 실행·학습
matrix에 포함하지 않는다. VOST/PUMaVOS/M³-VOS는 configuration freeze 뒤의 external
evaluation이다.

## MOSEv2 validation v1

- File: `mosev2_valid_v1.json`
- Dataset: MOSEv2 v2 validation split
- Videos: `433`
- Cases: `1,720` (three fixed temporal switch candidates per first-frame object)
- File SHA-256: `1adccb98aa5a73d48c34eb72ed8990f1cc00925e6cc2242c40dc0ce43263fd00`
- Annotation policy: validation publishes only the first-frame mask. Future visibility,
  disappearance and reappearance tags are unavailable locally and are not inferred.

Manifest builder와 Task 07 collection command는 local workspace에서 관리한다.

## MOSEv2 train v1

- File: `mosev2_train_v1.json`
- Videos: `3,666`
- Cases: `20,841`
- Annotation policy: dense train annotations from `meta_train.json`
- Manifest content SHA-256: `3051f1cb2025ae6a8565544e4311a0cb56efc33d9be77623c9bfc6f98814bd07`

Train은 validation처럼 first-frame-only가 아니므로 `meta_train.json`의 객체 목록·영상 길이와
실제 dense mask를 함께 확인해 객체별 최초 등장 frame 이후에만 3개 temporal switch 후보를 만든다.
전체 311,843개 train mask 스캔 결과, 최종 평가 대상 객체의 최초 prompt frame은 모두 0이었지만
이 값은 더 이상 가정이 아니라 실제 annotation으로 검증된 값이다.


## Video-level fit/development 후보

train manifest의 case를 video ID 기준으로 seed 7, 80/20 비율로 나누며 한 영상의
객체·switch case가 fit과 development에 동시에 들어가지 않게 한다.

최종 split은 MOSEv2 fit 2,680/16,931과 development 615/3,910,
LVOS v2 fit 347/1,488과 development 73/315다. split manifest는 case를 복제하지 않고
video membership과 source manifest checksum만 보존한다.

## LVOS v2 validation

LVOS v2 Eval archive는 공식 `valid.zip`으로 확보했다. RunPod에서 압축 해제한 결과는
140개 video directory, JPEG 66,056개, annotation 66,056개이며 archive SHA-256은
`beb488046f74e0cb4154a0cb2bcc2c79cae858da0693e5adabcf966a5712d4e2`이다.
manifest 파일은 140 sequences·714 cases를 담고, content SHA-256은
`a8afee3094d3326c04f1898108a19348c5c3ad17aa6a1ded653e4051c5d848a5`이다.
LVOS v2 loader는 공식 metadata의 object frame range 안에 실제 존재하는 sparse frame ID만
사용해 switch 후보를 만든다.
현재 archive의 `meta.json`은 loader 입력용 `val_meta.json`으로 파생해 사용했으며,
attribute·prompt/correction loader 검증은 후속 작업이다.


## VOST external benchmark — v1.1

- File: `vost_val_v1.json`
- Validation inventory: 70 sequences, 7,820 RGB frames and 7,820 annotations
- Cases: 210 (25/50/75% temporal quantile per sequence)
- Manifest content SHA-256: `47bab054c87281e7b904831ea8891b81d22f1d8683786f1d5cf84a4f1ba85cbf`
- The official test split is excluded because the delivered archive contains no RGB/GT files.

- Main role: primary translator-level cross-dataset zero-shot evaluation
- Allowed split: validation; 가능하면 official test server
- Main translator에서 금지: VOST train/val gradient, statistics, early stopping, threshold,
  architecture/loss/checkpoint/replay-k 선택
- Switch rule: 25/50/75% temporal quantile, primary 50%; 미래 GT event에 정렬하지 않음
- Official metrics: `J`, `J_tr`(마지막 25% transformation 구간)
- Optional: VOST-train fine-tuning은 별도 adaptation upper-bound ablation

VOST onboarding evidence와 evaluator command는 benchmark archive에서 관리한다.

## PUMaVOS external stress v1

- File: `pumavos_external_v1.json`
- Role: config freeze 뒤 한 번 실행하는 secondary external zero-shot stress benchmark
- Official split: 없음. 공개 archive 전체를 split으로 가장하지 않는다.
- Verified archive: `PUBLIC_PUMaVOS.zip`, `3,008,102,259` bytes,
  SHA-256 `ccd062636b0422055d1da7344411726b4fd1d74d490e68b50618c34ca9a087a4`
- Extracted inventory: 24 sequences, RGB 21,187 frames, GT annotations 21,187,
  paired-stem `failure_count=0`
- Cases: 78 (객체별 actual first-nonempty GT prompt, 25/50/75% frame-index switch)
- Manifest content SHA-256: `97857509580c74c517814dd0cd5ec6e53ebc36991305dc88d7ffdf10a47f3d64`

이 manifest는 prompt frame의 GT 한 장만 표준 VOS 입력으로 사용하며, switch 이후의 GT는
평가에만 사용한다. model fitting, normalization, threshold, replay-k, early stopping 또는
architecture/loss 선택에는 PUMaVOS의 label-derived 정보를 사용하지 않는다.


## M³-VOS external stress v1

- File: `m3vos_external_v1.json`
- Role: config freeze 뒤 한 번 실행하는 material phase-transition secondary external stress benchmark
- Immutable delivery revision: `5deb15b2baeaaa294ca168b789537729f7fb53a5`
- Verified delivery inventory: 471 sequences, RGB/GT 202,577 pairs, 530 metadata object records,
  68 core members, paired-set `failure_count=0`
- Cases: 1,590 (annotation에 실제 존재하는 객체별 first-nonempty GT prompt와 25/50/75% frame-index switch)
- Manifest content SHA-256: `b71c4af8634b53668ed1e74ef51234d815499d48d7a93ab290b4d0884632b612`

문헌·project page의 479 videos/205,181 dense masks 수치와 immutable delivery inventory는 다르므로,
실험 분모는 이 manifest의 received-delivery 수치를 사용하고 문헌 수치는 별도 맥락으로만 인용한다.
`target_object.json` metadata와 annotation label이 다른 4개 sequence는 manifest에 명시적으로
기록된다. metadata-only object에 빈 prompt를 만들지 않으며, annotation에 실제 존재하는 non-void
label만 평가 객체로 쓴다. label `255`는 void이며 prompt object로 사용하지 않는다.


## LVOS v2 train v1

- File: `lvosv2_train_v1.json`
- Dataset: LVOS v2 official train split
- Videos: `420`
- Cases: `1,803` (object frame range 안의 세 temporal switch 후보)
- Archive size: `22,414,546,249` bytes
- Archive SHA-256: `e4c0cfcb400dbd103ddea6eac3b5ffb5cdd3d50cb87013bf457d963a0b248c0b`
- Manifest content SHA-256: `8292b1b4674a2ef34955c908e980826f3d041c518a035b5f5b42ad8fb49df74a`
- `meta.json`의 object별 frame range를 사용하며, train mask는 로컬에서 future 평가에 사용할 수 있다.
