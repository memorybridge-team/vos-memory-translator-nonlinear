# Task 03 — External onboarding sources and storage policy

작성일: 2026-09-27

## 확인된 공식 접근 경로

| Dataset | 공식 source | 현재 상태 |
|---|---|---|
| PUMaVOS | [XMem2 repository](https://github.com/mbzuai-metaverse/XMem2), PUMaVOS Google Drive/Mirror | 다운로드 전 archive 크기·checksum 확인 필요 |
| M³-VOS | [Project page](https://zixuan-chen.github.io/M-cube-VOS.github.io/), [official evaluation instructions](https://github.com/zixuan-chen/M3VOS_Experiment/blob/main/docs/EVALUATION.md) | Google Drive archive 링크 확인; 크기·checksum·inventory 미확정 |

PUMaVOS는 공식 repository 설명과 논문·project page 사이에 video 수 표기 차이가 있으므로,
다운로드 후 archive inventory를 최종 기준으로 삼는다. M³-VOS는 evaluation 문서가
`JPEGImages`, `Annotations`, `Videos`, `ImageSets/val.txt`, `meta` 구조를 요구하므로
이 구조와 실제 배포본을 대조한다.

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
