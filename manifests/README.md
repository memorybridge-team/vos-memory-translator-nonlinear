# Frozen evaluation manifests

이 디렉터리는 데이터셋 원본 이미지·마스크를 저장하지 않는다. 현재 남은 파일은 LVOS v2 validation의 객체별 최초 prompt frame이다. `scripts/evaluate_lvos_base_roundtrip.py`가 이 값만 읽고, switch는 manifest case가 아니라 클립 순번 20으로 고정한다.

## LVOS v2 validation

- File: `lvosv2_valid_v1.json`
- Dataset: LVOS v2 validation
- Sequences: `140`
- Cases: `714`
- Content SHA-256: `a8afee3094d3326c04f1898108a19348c5c3ad17aa6a1ded653e4051c5d848a5`

공식 `valid.zip` archive SHA-256은 `beb488046f74e0cb4154a0cb2bcc2c79cae858da0693e5adabcf966a5712d4e2`이다.
