"""DINOv3 batched per-frame feature extraction and float16 memmap cache."""
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
from .roi import crop_to_448_rgb
from .settings import FEATURE_BATCH_SIZE, AMP
def extract_feature_bank(
    video_path: Path,
    track: pd.DataFrame,
    train_lib,
    model,
    device: torch.device,
    out_dir: Path,
    batch_size: int = FEATURE_BATCH_SIZE,
    amp: bool = AMP,
) -> Tuple[Path, Tuple[int, int, int]]:
    bank_path = out_dir / "feature_bank.float16.dat"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(video_path)

    batch_tensors, batch_indices = [], []
    bank = None
    feature_shape = None
    n_frames = len(track)

    def flush_batch():
        nonlocal bank, feature_shape, batch_tensors, batch_indices
        if not batch_tensors:
            return
        x = torch.stack(batch_tensors, dim=0).to(device, non_blocking=True)
        with torch.no_grad(), torch.amp.autocast(
            device_type="cuda", dtype=torch.float16,
            enabled=bool(amp and device.type == "cuda"),
        ):
            feat = model.extract_frame_features(x.unsqueeze(1))[:, 0]
        arr = feat.detach().cpu().numpy().astype(np.float16)
        if bank is None:
            feature_shape = tuple(arr.shape[1:])
            bank = np.memmap(bank_path, mode="w+", dtype=np.float16, shape=(n_frames, *feature_shape))
            bank[:] = 0
        bank[np.asarray(batch_indices, dtype=int)] = arr
        batch_tensors, batch_indices = [], []

    idx = 0
    while idx < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        rec = track.iloc[idx]
        if int(rec["track_valid"]) == 1:
            box = tuple(int(rec[k]) for k in ["roi_x1", "roi_y1", "roi_x2", "roi_y2"])
            rgb448 = crop_to_448_rgb(frame, box, train_lib)
            batch_tensors.append(train_lib.image_tensor(rgb448))
            batch_indices.append(idx)
            if len(batch_tensors) >= batch_size:
                flush_batch()
        idx += 1
        if idx % 100 == 0:
            print(f"  Features: {idx}/{n_frames}")

    flush_batch()
    cap.release()
    if bank is None or feature_shape is None:
        raise RuntimeError("No valid ROI frames available for feature extraction.")
    bank.flush()
    del bank
    return bank_path, feature_shape
