"""[주] 지표: 결과 줄의 J·J&F, 그리고 표에서 쓰는 회복률·격차 회복률·속도 배수.

결과 줄 (evaluate_video.py 가 씀)
  j, jf, n_frames  전환 뒤, 정답에 객체가 보이는 프레임만의 평균.
                   MOSEv2 valid 는 여기서 채점을 못 해서 서버가 준 영상 전체 점수가 들어간다 (4_mosev2_scores.py).

표 (tables/main_tables.py 가 씀)
  회복률(%)      영상마다 방법 ÷ Full Replay × 100 → 영상들의 평균
  속도 배수      영상마다 Full Replay 첫 결과 시간 ÷ 방법 첫 결과 시간 → 영상들의 평균
  격차 회복률(%) (방법 − Source-only) ÷ (Full Replay − Source-only) × 100
                 영상 평균 점수로 한 번만 나눈다 (C2C 논문의 PGR 과 같은 방식).
                 영상마다 나누면 Source-only 와 Full Replay 가 거의 같은 영상에서 분모가 0 에 가까워 값이 튄다.
"""

from __future__ import annotations

from collections import defaultdict


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def score_columns(scores: list) -> dict:
    """전환 뒤 보이는 프레임들의 FrameScore → j, jf, n_frames."""
    if not scores:
        return {"j": None, "jf": None, "n_frames": 0}
    j = mean(s.j for s in scores)
    f = mean(s.f for s in scores)
    return {"j": j, "jf": (j + f) / 2, "n_frames": len(scores)}


def video_means(rows: list[dict], key: str) -> dict[str, float]:
    """영상 → 그 영상 줄들(객체·전환 시점)의 key 평균. 값이 없는 줄은 건너뛴다."""
    values = defaultdict(list)
    for r in rows:
        if r.get(key) is not None:
            values[r["video"]].append(r[key])
    return {video: sum(v) / len(v) for video, v in values.items()}


def ratio_by_video(top_rows: list[dict], bottom_rows: list[dict], key: str) -> dict[str, float]:
    """영상 → top ÷ bottom. bottom 이 0 이거나 없는 영상은 뺀다."""
    top, bottom = video_means(top_rows, key), video_means(bottom_rows, key)
    return {v: top[v] / bottom[v] for v in top if bottom.get(v, 0) > 0}


def retention(rows: list[dict], replay_rows: list[dict], key: str):
    """회복률(%) — key 는 "jf" 또는 "j"."""
    value = mean(ratio_by_video(rows, replay_rows, key).values())
    return None if value is None else value * 100


def speedup(rows: list[dict], replay_rows: list[dict]):
    """속도 배수 — Full Replay 보다 첫 결과가 몇 배 빨리 나오는가."""
    return mean(ratio_by_video(replay_rows, rows, "seconds_to_first_frame").values())


def gap_retention(rows: list[dict], source_rows: list[dict], replay_rows: list[dict], key: str):
    """격차 회복률(%). 세 비교군 모두 점수가 있는 영상만 써서 영상 평균을 낸 뒤 한 번 나눈다."""
    method, source, replay = (video_means(r, key) for r in (rows, source_rows, replay_rows))
    videos = method.keys() & source.keys() & replay.keys()
    if not videos:
        return None
    m, s, r = (mean(d[v] for v in videos) for d in (method, source, replay))
    return None if r == s else (m - s) / (r - s) * 100
