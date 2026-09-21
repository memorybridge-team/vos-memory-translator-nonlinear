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
