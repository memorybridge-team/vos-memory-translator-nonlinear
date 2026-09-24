# 2026-09-25 · VOST extracted inventory

RunPod path: `/workspace/datasets/VOST/extracted/VOST`

검증 스크립트: [`scripts/validate_vost_inventory.py`](../../../../scripts/validate_vost_inventory.py)

| split | sequences | JPEGImages frames | annotations | directory failures |
|---|---:|---:|---:|---:|
| train | 572 | 59,930 | 59,930 | 0 |
| val | 70 | 7,820 | 7,820 | 0 |
| test | 71 (name list only) | 0 | 0 | 71 |

`failure_count=0`은 train/val에서 각 sequence의 frame stem과 annotation stem이
일치한다는 뜻이다. 배포 archive에는 test sequence 이름 목록이 있지만 test JPEG/annotation
파일은 포함되어 있지 않으므로, test 결과를 로컬에서 생성하거나 추정하지 않는다. Test는
공식 평가 서버 또는 별도 공개 archive가 접근 가능할 때만 sealed external 결과로 추가한다.

현재 VOST local benchmark의 실행 대상은 val 70 sequence이며, train/val 모두 translator
학습·선택에는 사용하지 않는다. 25/50/75% switch manifest와 공식 `J/J_tr` 검증은 다음 gate다.
