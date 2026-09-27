# vos-memory-translator-nonlinear

SAM 2.1 Small이 switch 시점까지 만든 객체별 memory를 translator로 바꾼 뒤 SAM 2.1 Base+ predictor에 주입해 `switch frame + 1`부터 이어 추적하는 코드입니다. MLP를 사용하여 memory를 변환합니다.

This repository exports per-object memory from SAM 2.1 Small up to a switch frame, translates it with a translator (including an MLP), injects it into a SAM 2.1 Base+ predictor, and continues tracking from `switch frame + 1`.

아직 검증되지 않음
