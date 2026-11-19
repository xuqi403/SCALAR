"""Video metadata, file names, output tables and audit metadata."""
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

def save_json(path: Path, obj: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def path_key(path: str | Path) -> str:
    """Stable Windows-friendly path comparison key."""
    text = str(path).strip().strip('"').replace("\\", "/")
    try:
        return str(Path(text).resolve()).replace("\\", "/").lower()
    except Exception:
        return text.lower()


def safe_stem(text: str, max_len: int = 110) -> str:
    """Preserve Chinese/readable characters; remove only Windows-invalid filename chars."""
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", str(text)).strip(" ._")
    return (s[:max_len] or "video")


def video_uid(path: Path) -> str:
    digest = hashlib.sha1(path_key(path).encode("utf-8", errors="ignore")).hexdigest()[:12]
    return digest


def output_dir_for(video_path: Path, output_root: Path) -> Path:
    return Path(output_root) / f"{safe_stem(video_path.stem)}__{video_uid(video_path)}"


def video_metadata(path: Path) -> Tuple[int, int, int, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if n <= 0 or w <= 0 or h <= 0:
        raise RuntimeError(f"Invalid video metadata: {path}")
    if not np.isfinite(fps) or fps <= 1e-6:
        fps = 25.0
    return n, w, h, fps


def compact_frame_table(
    frame_df: pd.DataFrame,
    video_path: Path,
    out_dir: Path,
    w: int,
    h: int,
    fps: float,
) -> pd.DataFrame:
    uid = video_uid(video_path)
    x = frame_df.copy()
    x.insert(0, "video_uid", uid)
    x.insert(1, "video_path", str(video_path))
    x.insert(2, "video_name", video_path.name)
    x["video_width"] = int(w)
    x["video_height"] = int(h)
    x["fps"] = float(fps)
    x["mask_npy_path"] = str(out_dir / "masks_448.npy")
    x["mask_index"] = x["frame_idx"].astype(int)

    keep = [
        "video_uid", "video_path", "video_name", "frame_idx", "time_sec", "fps",
        "video_width", "video_height",
        "track_status", "track_valid", "segment_id", "track_outlier", "yolo_confidence",
        "roi_x1", "roi_y1", "roi_x2", "roi_y2",
        "inference_valid", "prediction_confidence",
        "mask_npy_path", "mask_index",
    ]
    return x[[c for c in keep if c in x.columns]].copy()
