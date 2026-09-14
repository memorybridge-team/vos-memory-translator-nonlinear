# Ridge hybrid handoff preview

이 문서는 Tiny의 spatial memory를 component-wise Ridge로 번역하고 pointer와
presence를 직접 전달한 제한적 Pilot 결과다. 전체 DAVIS benchmark 결과가
아니며, `bmx-bumps` 한 영상·한 객체·switch frame 6 조건만 다룬다.

- Source: `sam2.1-hiera-tiny`
- Target: `sam2.1-hiera-large`
- Translator: `ridge_spatial_pointer_direct_presence`
- Switch frame: `6`
- Mean binary IoU vs target-native: `0.9493896016398312`
- Mean mask-logit MSE vs target-native: `1.8786671569189393`
- Wall time: `91.91071961075068` seconds
- Peak CUDA memory: `1660043264` bytes

아래에는 Git 저장 용량을 제한하기 위해 switch 직후, 중간, 어려운 재등장 구간의
대표 4프레임만 남겼다. 모든 83프레임과 GT 비교는
[공개 인터랙티브 갤러리](https://memorybridge-team.github.io/vos-memory-translator-nonlinear/experiments/2026-09-08-davis-handoff/)에서
확인할 수 있다.

![frame 7](comparisons/frame_00007.png)

![frame 44](comparisons/frame_00044.png)

![frame 50](comparisons/frame_00050.png)

![frame 78](comparisons/frame_00078.png)
