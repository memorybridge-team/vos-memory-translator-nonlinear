# Direct Copy handoff preview

이 문서는 SAM 2.1 Tiny의 memory를 별도 학습 없이 Large에 직접 복사한 제한적
Pilot 결과다. 전체 DAVIS benchmark 결과가 아니며, `bmx-bumps` 한 영상·한
객체·switch frame 6 조건만 다룬다.

- Source: `sam2.1-hiera-tiny`
- Target: `sam2.1-hiera-large`
- Translator: `direct_copy`
- Switch frame: `6`
- Mean binary IoU vs target-native: `0.5616458207236444`
- Mean mask-logit MSE vs target-native: `121852.47061858838`
- Wall time: `90.8903582347557` seconds
- Peak CUDA memory: `1660043264` bytes

아래에는 Git 저장 용량을 제한하기 위해 switch 직후, 중간, 어려운 재등장 구간의
대표 4프레임만 남겼다. 모든 83프레임과 GT 비교는
[공개 인터랙티브 갤러리](https://memorybridge-team.github.io/vos-memory-translator-nonlinear/experiments/2026-09-08-davis-handoff/)에서
확인할 수 있다.

![frame 7](comparisons/frame_00007.png)

![frame 44](comparisons/frame_00044.png)

![frame 50](comparisons/frame_00050.png)

![frame 78](comparisons/frame_00078.png)
