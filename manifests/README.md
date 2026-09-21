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

## LVOS v2 validation

LVOS v2 loader는 공식 `val_meta.json`과 선택적 `val_meta_attribute.json`을 읽어
object frame range 안에서만 switch 후보를 만든다. Eval archive 다운로드가 완료되면
다음 명령으로 manifest를 생성한다.

```bash
cmmt-lvos-build-manifest \
  --root /path/to/LVOS-v2 \
  --split val --seed 7 \
  --output manifests/lvosv2_val_v1.json
```
