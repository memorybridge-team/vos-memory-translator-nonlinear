# LVOS v2 소규모 Base+ → Base+ 상태 이전 검증 계획

## 목적과 범위

동일한 SAM 2.1 Base+ config/checkpoint로 추론할 때, frame `t`까지의 메모리를 export → 값 변경 없는 identity 전달 → 새 Base+ 세션에 inject한 실행이 중단 없는 Base+ 실행과 동일하게 `t+1`부터 이어지는지 확인한다. 이는 handoff 구현 정확성 검사이며 Small → Base+ 번역 성능이나 MLP 학습 검증이 아니다. 실행 코드는 별도 스크립트로 구현했으며, 현재는 추론을 실행하지 않았다.

## 데이터와 고정 조건

- LVOS **v2 validation**에서 원래 4개 smoke-test 영상에, 이전 로컬 Base+ cold-start J&F가 낮았던 hard 영상 10개를 추가해 총 14개를 사용한다. 원래 영상은 단일 객체 `2VegYEbT`, `2urlAsm8`; 동시 다객체 `0tCWPOrc`; 객체별 최초 mask prompt가 다른 다객체 `9HEh93ef`다. hard 영상은 `x3nD3QQ9`, `MKnlVo6x`, `8lxxCA5h`, `nfcT3owb`, `q1MSEBkh`, `ScFTYisJ`, `aFytsETk`, `dtHbJvYy`, `xpI7xRWN`, `K3OUeINk`이며, 이전 cold-start J&F는 각각 `0.0773, 0.2711, 0.3644, 0.4219, 0.4242, 0.5425, 0.5450, 0.5516, 0.5656, 0.5695`였다. 이 점수는 hard 영상 선택 근거일 뿐 현재 handoff 합격 판정 입력이 아니다.
- 각 영상에서 최초 prompt가 포함된 순서대로 **실제로 존재하는 첫 40개 RGB 프레임**과 같은 원본 frame ID의 annotation만 사용한다. 총 160개 프레임이다. `9HEh93ef`의 늦은 prompt도 switch 이전에 포함되는지 데이터 준비 단계에서 확인하고, 충족되지 않으면 결과를 내지 않고 원인을 기록한다.
- 클립 내부 순번 `0..39`와 LVOS 원본 frame ID를 별도로 보존한다. Source cutoff/switch는 내부 순번 `t=20`; Target continuation은 `21..39`다. 원본 frame ID를 SAM 2의 `frame_idx`로 직접 사용하지 않는다.
- 두 실행 모두 동일한 Base+ config/checkpoint/SAM 2 revision, 영상 순서, preprocessing, 객체 ID, 최초 GT mask prompt, deterministic 설정과 seed `7`을 쓴다. GT는 최초 prompt와 사후 채점에만 사용하며 추가 correction은 넣지 않는다.
- 데이터는 Git 추적 대상이 아닌 `data/lvosv2_subset/val/{JPEGImages,Annotations}/<video_id>/`에 둔다. 공식 archive 출처, 파일명·수·크기·CRC 또는 SHA-256, 선택한 frame ID 목록을 기록한다. 다른 LVOS 배포본(v1)과 혼합하지 않는다.

### 데이터 준비 상태 (2026-09-23)

공식 LVOS v2 `valid.zip`에서 필요한 ZIP entry만 HTTP range로 받아 위 경로에 저장한다. 영상마다 RGB 40장·annotation 40장으로 총 1,120개 파일이다. 선택한 원본 frame ID는 모든 영상에서 `1, 6, ..., 196`이며 switch 내부 순번 20은 원본 ID `101`에 대응한다. `9HEh93ef`의 객체 3/4/5 최초 prompt ID `31/76/81`은 모두 switch 이전이다. 파일별 크기·SHA-256과 14개 영상 목록은 `data/lvosv2_subset/val/download_manifest.json`에 기록한다.

## 비교 실행과 기록

1. **Native:** 객체별 최초 mask prompt 시점부터 39번까지 Base+를 중단 없이 실행한다.
2. **Transferred:** Native와 동일한 prefix를 공유한다. 20번에서 기존 `canonicalize_sam2_inference_state` → `DirectCopyTranslator.translate` → `inject_sam2_canonical_state` API를 차례로 호출하여 별도 Base+ predictor에 주입하고 21번부터 이어간다. translator 구현은 수정하지 않는다. Target 정책으로 positional encoding과 로컬 인덱스를 재구성하며, 주입 중 과거 RGB backbone 호출이 0회인지 확인한다.
3. 모든 객체의 각 관측 프레임에 대해 `video_id`, `object_id`, 원본 frame ID, 내부 순번, prompt 여부, switch 전/후, GT visibility, native/transfer `J`, `F`, `J&F`, `ΔJ&F=transfer−native`를 한 행씩 기록한다. 객체가 아직 prompt되지 않은 프레임은 채점하지 않는다. GT를 제공한 prompt 프레임은 그래프에 표시하되 성능 평균에서 제외한다.
4. 영상별로 객체별 J&F 시계열과 영상 내 활성 객체 평균 시계열을 그린다. 각 객체의 prompt와 switch 위치를 세로선으로 표시하고, `t+1` 첫 변화, `21..39` 평균 변화, 최댓값/최대하락을 표로 함께 남긴다. 단순한 switch 전·후 평균 차이는 영상 난이도 변화와 혼동될 수 있으므로 정상 이전 판정에는 쓰지 않는다.

## 합격 기준과 해석

- 모든 유효 post-switch `(frame, object)`에서 native와 transferred의 객체 ID, binary mask, J/F가 일치하고 logit 최대 절대오차가 `1e-6` 이하여야 한다. 주입 중 backbone 호출은 0회이고, Target continuation 전체의 backbone 호출은 future frame 수(`19`)와 같아야 한다. 주입 API는 원래 모델을 호출하지 않으므로, 과거 RGB를 다시 처리하지 않았다는 근거는 두 번째 조건이다. 일치하지 않으면 첫 불일치 원본 frame ID·객체·state record를 기록한다.
- Base+ checkpoint SHA-256이 `a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5`와 다르면 실행하지 않는다.
- manifest는 객체별 `first_prompt_frame`만 읽는다. switch는 모든 영상에서 클립 순번 20으로 고정하며 manifest의 switch case가 아니다.
- CPU 실행은 CUDA→CPU 비동기 offload 경로를 거치지 않는다. `report.json`의 `cuda_offload_path_exercised=false`인 통과는 GPU gate를 대신하지 않는다.
- 프레임별 J/F는 LVOS 공식 metric 정의와 동일하게 계산하되, 이 결과를 **14개 클립의 자체 진단 지표**로 표기한다. 전체 validation을 평가하지 않으므로 공식 LVOS dataset-level 점수나 일반화 성능으로 주장하지 않는다.
- `scripts/evaluate_lvos_base_roundtrip.py`가 원래 4개와 hard 10개를 포함한 14개 LVOS sparse frame ID 매핑, 객체별 prompt timeline, 프레임별 공식 J/F 계산, 영상별 `frames.csv`·`jf_curve.svg`·`summary.json`, 실행 전체 `report.json`을 생성한다. 이전 전 구간은 두 방식이 동일한 prefix를 공유하므로 CSV의 native/transfer 값은 같다. 프롬프트 프레임은 그래프에 보이되 평균에서는 제외한다.

## 실행 방법과 현재 검증 상태

SAM 2가 설치된 Python 환경에서 저장소 루트 기준으로 실행한다. 별도 공식 평가도구 checkout은 `.external/lvos-evaluation`에 두고, 해당 환경에 `opencv-python-headless`와 `scikit-image`를 설치한다. 현재 작업공간의 SAM 2 가상환경에는 두 패키지와 의존성이 설치되어 있으며 공식 J/F 함수 import·간단 계산이 통과했다. 새 환경에서는 필요하면 `uv pip install --python ../sam2/.venv/bin/python 'opencv-python-headless>=4.8,<5' 'scikit-image>=0.21,<1'`을 실행한다. 평가도구 Git commit과 SAM 2/checkpoint 및 데이터 manifest 해시가 결과에 기록된다.

```bash
PYTHONPATH=src ../sam2/.venv/bin/python scripts/evaluate_lvos_base_roundtrip.py --check-data-only
PYTHONPATH=src ../sam2/.venv/bin/python scripts/evaluate_lvos_base_roundtrip.py --device cpu
```

실행 결과는 기본적으로 `outputs/lvos_base_roundtrip/<UTC timestamp>/`에 쓰며 기존 디렉터리는 덮어쓰지 않는다. CPU에서는 Base+ 영상 추론 시간이 길 수 있다. 기존 4개 smoke-test 실행은 `outputs/lvos_base_roundtrip/20260923T091720Z/`에서 통과했지만, hard 영상 10개를 추가한 14개 전체 재실행은 아직 수행하지 않았다. `status=passed`가 기록되기 전까지 14개 전체의 Base+→Base+ 이전 성공을 주장하지 않는다.

참고: [LVOS v2 공식 배포](https://github.com/LingyiHongfd/LVOS), [LVOS 공식 평가 도구](https://github.com/LingyiHongfd/lvos-evaluation), `manifests/lvosv2_valid_v1.json`.
