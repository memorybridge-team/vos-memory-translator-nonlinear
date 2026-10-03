"""학습에 쓸 Best model 고르기.

기준: 전환 뒤 J&F 회복률을 25/50/75% 전환에서 각각 구한 뒤 평균한 값이 가장 큰 모델.
회복률(%) = 방법의 점수 ÷ Full Replay의 점수 × 100  (영상마다 비율을 구한 뒤 평균)
Full Replay = Base+ 로 처음부터 끝까지 돌린 결과 (상한선).

점수 줄 하나: {"video": "abc", "fraction": 0.25, "jf": 0.81}
"jf"는 post_switch_jf()로 구한다 (예측 마스크와 정답 마스크 → 전환 뒤 J&F).

학습 루프에서:
    best = BestModel("outputs/train_run1")
    for epoch in range(n_epochs):
        train_one_epoch(model)
        best.update(epoch, model, evaluate_on_dev(model), replay_rows)
"""

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

FRACTIONS = (0.25, 0.50, 0.75)
BOUNDARY_THRESHOLD = 0.008   # F: 테두리 허용 거리 = 이미지 대각선 × 이 값 (DAVIS 기본값)


def j_score(pred, gt):
    """겹친 넓이 ÷ 합친 넓이. 둘 다 비어 있으면 1."""
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return np.logical_and(pred, gt).sum() / union


def boundary(mask):
    """마스크의 테두리 픽셀: 오른쪽·아래·오른쪽아래 이웃과 값이 다른 픽셀 (DAVIS와 같음)."""
    right = np.zeros_like(mask)
    down = np.zeros_like(mask)
    diag = np.zeros_like(mask)
    right[:, :-1] = mask[:, 1:]
    down[:-1, :] = mask[1:, :]
    diag[:-1, :-1] = mask[1:, 1:]
    b = (mask ^ right) | (mask ^ down) | (mask ^ diag)
    b[-1, :] = mask[-1, :] ^ right[-1, :]
    b[:, -1] = mask[:, -1] ^ down[:, -1]
    b[-1, -1] = False
    return b


def f_score(pred, gt):
    """테두리가 서로 허용 거리 안에 있는 비율(정밀도·재현율)의 조화평균. 둘 다 테두리가 없으면 1."""
    pred_b, gt_b = boundary(pred), boundary(gt)
    if not pred_b.any() and not gt_b.any():
        return 1.0
    if not pred_b.any() or not gt_b.any():
        return 0.0

    # 테두리를 허용 거리만큼 부풀려서, 상대 테두리가 그 안에 들어오면 맞은 것으로 본다.
    radius = int(np.ceil(BOUNDARY_THRESHOLD * np.hypot(*pred.shape)))
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    disk = (x * x + y * y <= radius * radius).astype(np.uint8)
    pred_near = cv2.dilate(pred_b.astype(np.uint8), disk).astype(bool)
    gt_near = cv2.dilate(gt_b.astype(np.uint8), disk).astype(bool)

    precision = (pred_b & gt_near).sum() / pred_b.sum()
    recall = (gt_b & pred_near).sum() / gt_b.sum()
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def post_switch_jf(preds, gts):
    """전환 뒤 프레임들의 예측·정답 마스크 → J&F.

    정답에 객체가 보이는 프레임만 채점한다. J&F = (평균 J + 평균 F) ÷ 2
    preds, gts: 프레임 순서대로 같은 길이의 True/False 2차원 배열 목록.
    """
    frames = [(p, g) for p, g in zip(preds, gts) if g.any()]
    j = sum(j_score(p, g) for p, g in frames) / len(frames)
    f = sum(f_score(p, g) for p, g in frames) / len(frames)
    return float((j + f) / 2)


def video_scores(rows, fraction):
    """해당 전환 시점의 줄만 골라 영상별 J&F 평균 (객체가 여러 개면 평균)."""
    scores = defaultdict(list)
    for r in rows:
        if r["fraction"] == fraction:
            scores[r["video"]].append(r["jf"])
    return {video: sum(v) / len(v) for video, v in scores.items()}


def retention(method_rows, replay_rows, fraction):
    """영상마다 방법 ÷ Full Replay × 100 → 영상들의 평균. Full Replay가 0인 영상은 뺌."""
    method = video_scores(method_rows, fraction)
    replay = video_scores(replay_rows, fraction)
    if method.keys() != replay.keys():
        raise ValueError(f"{fraction} 전환: 방법과 Full Replay의 영상 목록이 다름")
    ratios = [method[v] / replay[v] * 100 for v in method if replay[v] > 0]
    return sum(ratios) / len(ratios)


def selection_score(method_rows, replay_rows):
    """시점별 회복률과 그 평균. "score"가 클수록 좋은 모델."""
    result = {f"r{round(f * 100)}": retention(method_rows, replay_rows, f) for f in FRACTIONS}
    result["score"] = sum(result.values()) / len(FRACTIONS)
    return result


class BestModel:
    """epoch마다 점수를 기록하고, 최고 점수가 나오면 모델을 저장.

    out_dir/selection_log.jsonl  epoch마다 한 줄 (r25, r50, r75, score)
    out_dir/best.pt              최고 점수 모델의 state_dict
    out_dir/best.json            최고 점수와 그 epoch
    """

    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.best_score = float("-inf")
        self.best_epoch = None

    def update(self, epoch, model, method_rows, replay_rows):
        """점수를 기록하고, 새 최고면 모델을 저장하고 True."""
        result = {"epoch": epoch, **selection_score(method_rows, replay_rows)}
        with open(self.out_dir / "selection_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")

        if result["score"] <= self.best_score:
            return False
        self.best_score = result["score"]
        self.best_epoch = epoch
        torch.save(model.state_dict(), self.out_dir / "best.pt")
        (self.out_dir / "best.json").write_text(json.dumps(result), encoding="utf-8")
        return True
