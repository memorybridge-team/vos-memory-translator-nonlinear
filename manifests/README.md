# Frozen evaluation manifests

이 디렉터리는 데이터셋의 원본 이미지·마스크를 저장하지 않는다. 같은 dataset/split/video/object/switch 조합을 반복해서 평가하기 위한 작은 JSON manifest만 version control 한다.

## DAVIS 2017 validation v1

- File: `davis2017_val_v1.json`
- Dataset: DAVIS 2017 trainval, 480p validation split
- Seed: `7`
- Sequences: `30`
- Cases: `249`
- File SHA-256: `5b036f173d74e6939c3096fa03e0af64f3dfba8bdaff21a43e1db8f5c1ada2f5`

Ground truth는 switch case를 고정하고 난이도 tag를 붙이는 데만 쓴다. 모델 입력, handoff payload, translator 학습 target 선택에는 GT pixel을 사용하지 않는다. 원본 dataset root는 runtime argument로만 제공하며 manifest에 기록하지 않는다.

생성 명령은 다음과 같다.

```bash
cmmt-davis-build-manifest \
  --root /path/to/DAVIS \
  --split val --resolution 480p --seed 7 \
  --output manifests/davis2017_val_v1.json
```

## MOSEv2 validation v1

- File: `mosev2_valid_v1.json`
- Dataset: MOSEv2 v2 validation split
- Videos: `433`
- Cases: `1,720` (three fixed temporal switch candidates per first-frame object)
- File SHA-256: `1adccb98aa5a73d48c34eb72ed8990f1cc00925e6cc2242c40dc0ce43263fd00`
- Annotation policy: validation publishes only the first-frame mask. Future visibility,
  disappearance and reappearance tags are unavailable locally and are not inferred.

생성 명령:

```bash
PYTHONPATH=src python -c "from vos_memory_inspector.mose import build_mosev2_evaluation_manifest, write_mosev2_evaluation_manifest; m=build_mosev2_evaluation_manifest('/path/to/MOSEv2/extracted', split='valid'); write_mosev2_evaluation_manifest(m, 'manifests/mosev2_valid_v1.json')"
```

## MOSEv2 train v1

- File: `mosev2_train_v1.json`
- Videos: `3,666`
- Cases: `20,841`
- Annotation policy: dense train annotations from `meta_train.json`
- Manifest content SHA-256: `1734d1cb031a7e7ee3ee3324be7166fd0add726d5cba7289d8a692d08c60c014`

Train은 validation처럼 first-frame-only가 아니므로 `meta_train.json`의 객체 목록과 영상 길이를 사용해
동일한 3개 temporal switch 후보를 만든다.

```bash
cmmt-mose-build-manifest \
  --root /path/to/MOSEv2/extracted \
  --split train --seed 7 \
  --output manifests/mosev2_train_v1.json
```

## Video-level fit/development 후보

train manifest의 case를 video ID 기준으로 seed 7, 80/20 비율로 나누며 한 영상의
객체·switch case가 fit과 development에 동시에 들어가지 않게 한다.

```bash
python scripts/build_video_splits.py \
  --input manifests/mosev2_train_v1.json \
  --output-prefix manifests/mosev2_train_v1
```

현재 생성된 후보는 MOSEv2가 fit 2,680 videos/16,931 cases와 development 615/3,910,
LVOS v2가 fit 347/1,488와 development 73/315이다. 최종 freeze 전 DAVIS train도
같은 스크립트로 생성하고 split checksum을 기록한다.

## LVOS v2 validation

LVOS v2 Eval archive는 공식 `valid.zip`으로 확보했다. RunPod에서 압축 해제한 결과는
140개 video directory, JPEG 66,056개, annotation 66,056개이며 archive SHA-256은
`beb488046f74e0cb4154a0cb2bcc2c79cae858da0693e5adabcf966a5712d4e2`이다.
manifest 파일은 140 sequences·717 cases를 담고, content SHA-256은
`3a3e3c7610a08e251f109d77d516723392b14484be81ad3eb604aa3a63a1b5c1`이다.
LVOS v2 loader는 공식 metadata의 object frame range 안에서만 switch 후보를 만든다.
현재 archive의 `meta.json`은 loader 입력용 `val_meta.json`으로 파생해 사용했으며,
attribute·prompt/correction loader 검증은 후속 작업이다.

```bash
cmmt-lvos-build-manifest \
  --root /path/to/LVOS-v2 \
  --split val --seed 7 \
  --output manifests/lvosv2_val_v1.json
```

## LVOS v2 train v1

- File: `lvosv2_train_v1.json`
- Dataset: LVOS v2 official train split
- Videos: `420`
- Cases: `1,803` (object frame range 안의 세 temporal switch 후보)
- Archive size: `22,414,546,249` bytes
- Manifest content SHA-256: `4c6b663d8af3dfa9c2a5eed0b5d955121710ed9c925fcedcf01837f499a6ed19`
- `meta.json`의 object별 frame range를 사용하며, train mask는 로컬에서 future 평가에 사용할 수 있다.

```bash
cmmt-lvos-build-manifest \
  --root /path/to/LVOS-v2 \
  --split train --seed 7 \
  --output manifests/lvosv2_train_v1.json
```
