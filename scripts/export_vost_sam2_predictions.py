"""Export one SAM 2 VOST prompt run in the official evaluator's PNG layout.

The script is deliberately model-agnostic at the output boundary: a future
CMMT continuation can replace the predictor state before propagation while
retaining the same ``results/<sequence>/<original-frame-stem>.png`` contract.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch


def load_prompt(path: Path, object_id: int) -> np.ndarray:
    labels = np.asarray(Image.open(path))
    if labels.ndim == 3:
        labels = labels[..., 0]
    mask = labels == object_id
    if not mask.any():
        raise ValueError(f"object {object_id} is absent from {path}")
    return mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam2-repo", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--staging-json", type=Path, required=True)
    parser.add_argument("--object-id", type=int, default=1)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    staging = json.loads(args.staging_json.read_text())
    if str(args.sam2_repo.resolve()) not in sys.path:
        sys.path.insert(0, str(args.sam2_repo.resolve()))
    from sam2.build_sam import build_sam2_video_predictor

    predictor = build_sam2_video_predictor(args.config, str(args.checkpoint), device=args.device)
    state = predictor.init_state(
        video_path=staging["sam2_frames_directory"],
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
    )
    predictor.add_new_mask(
        state, frame_idx=0, obj_id=args.object_id,
        mask=load_prompt(Path(staging["prompt_annotation"]), args.object_id),
    )
    output_dir = args.results_root / staging["sequence"]
    output_dir.mkdir(parents=True, exist_ok=True)
    stems = staging["original_stems"]
    frames_written = []
    with torch.inference_mode():
        for frame_idx, object_ids, mask_logits in predictor.propagate_in_video(state, start_frame_idx=0):
            labels = np.zeros(mask_logits.shape[-2:], dtype=np.uint8)
            for index, object_id in enumerate(object_ids):
                binary_mask = mask_logits[index].detach().cpu().numpy().squeeze() > 0.0
                labels[binary_mask] = int(object_id)
            stem = stems[int(frame_idx)]
            Image.fromarray(labels).save(output_dir / f"{stem}.png")
            frames_written.append(stem)
    print(json.dumps({
        "sequence": staging["sequence"],
        "object_id": args.object_id,
        "prediction_dir": str(output_dir.resolve()),
        "frame_count": len(frames_written),
        "first_frame": frames_written[0],
        "last_frame": frames_written[-1],
    }, indent=2))


if __name__ == "__main__":
    main()
