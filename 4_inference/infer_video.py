#!/usr/bin/env python3
"""CLI: SCALAR whole-video segmentation (YOLO -> DINOv3 -> V5 -> masks)."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from scalar_infer import model_loader, pipeline

HERE = Path(__file__).resolve().parent
DEFAULT_TRAIN = HERE.parent / "3_video_model" / "train_video.py"

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SCALAR segmentation-only whole-video inference")
    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument("--video", type=Path, help="One authorized ultrasound video")
    inp.add_argument("--video-list", type=Path, help="Private CSV with video_path column")
    p.add_argument("--train-script", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--checkpoint", type=Path, required=True, help="Final SCALAR V5 checkpoint, e.g. best_anatomy")
    p.add_argument("--dino-repo", type=Path, required=True, help="DINOv3 upstream source tree")
    p.add_argument("--dino-weights", type=Path, required=True, help="Carotid SSL DINOv3 checkpoint")
    p.add_argument("--yolo-code-dir", type=Path, required=True, help="YOLO ROI detector source directory")
    p.add_argument("--training-cache-manifest", type=Path, default=None,
                   help="Private training ROI cache prior for matching crop scales")
    p.add_argument("--output", type=Path, default=HERE / "outputs")
    p.add_argument("--feature-batch-size", type=int, default=8)
    p.add_argument("--fusion-batch-size", type=int, default=24)
    p.add_argument("--no-overlay", action="store_true", help="Skip contour-overlay video")
    p.add_argument("--overwrite", action="store_true", help="Regenerate previous outputs")
    return p.parse_args(argv)


def videos_from_args(args):
    if args.video is not None:
        candidates = [args.video]
    else:
        table = pd.read_csv(args.video_list, low_memory=False)
        if "video_path" not in table.columns:
            raise ValueError("Input CSV must have a video_path column")
        candidates = [Path(x) for x in table["video_path"].dropna().astype(str) if x.strip()]
    videos, seen = [], set()
    for candidate in candidates:
        video = candidate.expanduser().resolve()
        if not video.is_file():
            raise FileNotFoundError(f"Video missing: {video}")
        if video not in seen:
            videos.append(video)
            seen.add(video)
    if not videos:
        raise RuntimeError("No videos supplied")
    return videos


def main(argv=None):
    args = parse_args(argv)
    if args.feature_batch_size < 1 or args.fusion_batch_size < 1:
        raise ValueError("Batch sizes must be positive")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    videos = videos_from_args(args)
    model_lib = model_loader.load_train_module(args.train_script, args.dino_repo, args.dino_weights, args.yolo_code_dir)
    args.output.mkdir(parents=True, exist_ok=True)
    statuses = []
    for video in videos:
        try:
            output = pipeline.run_one(video, args, model_lib)
            statuses.append({"video_path": str(video), "status": "ok", "output_dir": str(output), "error": ""})
        except Exception as exc:
            statuses.append({"video_path": str(video), "status": "error", "output_dir": "",
                             "error": f"{type(exc).__name__}: {exc}"})
            if len(videos) == 1:
                raise
            print(f"[FAILED] {video}: {exc}")
        pd.DataFrame(statuses).to_csv(args.output / "run_status.csv", index=False)
    return 0 if all(s["status"] == "ok" for s in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
