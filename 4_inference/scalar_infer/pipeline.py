"""Single-video orchestration, checkpoint fingerprinting and resume protection."""
from __future__ import annotations
import gc
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from . import io_utils, roi, model_loader, feature_bank, segmentation, overlay
def sha256(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run_one(video, args, model_lib):
    """Process a single video. Output labels: 0 background / 1 wall / 2 lumen."""
    out_dir = io_utils.output_dir_for(video, args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = out_dir / "video_done.json"
    checkpoint_hash = sha256(args.checkpoint)
    artifacts = (out_dir / "frame_predictions.csv", out_dir / "masks_448.npy")
    if done.is_file() and all(f.is_file() for f in artifacts) and not args.overwrite:
        previous = json.loads(done.read_text(encoding="utf-8"))
        if previous.get("checkpoint_sha256") != checkpoint_hash or previous.get("source_video") != str(video):
            raise RuntimeError("Existing output has different video/checkpoint; use --overwrite")
        print(f"[REUSE] {out_dir}")
        return out_dir
    if args.overwrite:
        for name in ("masks_448.npy", "frame_predictions.csv", "video_done.json", "feature_bank.float16.dat"):
            target = out_dir / name
            if target.exists():
                target.unlink()

    print(f"[DETECT] {video}")
    detections, width, height, fps = roi.detect_video(video, model_lib)
    track = roi.build_roi_track(detections, width, height)
    crop_w, crop_h, crop_source = roi.estimate_crop_size(
        track, width, height, args.training_cache_manifest or Path("__NO_CACHE_PRIOR__")
    )
    track = roi.attach_canonical_boxes(track, crop_w, crop_h, width, height)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_loader.load_v5_model(model_lib, args.checkpoint, device)
    bank_path = None
    try:
        bank_path, feature_shape = feature_bank.extract_feature_bank(video, track, model_lib, model, device, out_dir, batch_size=args.feature_batch_size)
        result = segmentation.infer_feature_windows(track, bank_path, feature_shape, model_lib, model, device, out_dir, fusion_batch_size=args.fusion_batch_size)
        frame_table = io_utils.compact_frame_table(result, video, out_dir, width, height, fps)
        frame_table.to_csv(out_dir / "frame_predictions.csv", index=False, encoding="utf-8-sig")
        if not args.no_overlay:
            overlay.write_overlay_video(video, frame_table, out_dir / "masks_448.npy",
                                     out_dir / "overlay.mp4", fps, width, height)
        io_utils.save_json(done, {
            "source_video": str(video), "n_frames": int(len(frame_table)),
            "n_valid": int(frame_table["inference_valid"].sum()),
            "native_width": width, "native_height": height, "fps": fps,
            "checkpoint_sha256": checkpoint_hash, "checkpoint_name": args.checkpoint.name,
            "crop_prior_source": crop_source, "crop_size": [crop_w, crop_h],
            "context_offsets": [-4, -2, 0, 2, 4],
            "mask_labels": {"background": 0, "wall": 1, "lumen": 2},
            "mask_coordinates": "448x448 ROI, native-frame restoration via roi box",
        })
    finally:
        if bank_path is not None and bank_path.exists():
            bank_path.unlink()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out_dir
