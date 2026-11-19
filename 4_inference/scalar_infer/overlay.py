"""Native-frame outline-only segmentation video; no physical measurements."""
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
from .roi import mask_448_to_native
from .settings import (OUTER_RGB, LUMEN_RGB, CONTOUR_THICKNESS, CONTOUR_ALPHA, DRAW_ROI_BOX, DRAW_VIDEO_TEXT, OUTPUT_VIDEO_FPS)
def _rgb_to_bgr(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return int(rgb[2]), int(rgb[1]), int(rgb[0])


def overlay_native_mask(
    frame: np.ndarray,
    native_mask: Optional[np.ndarray],
    rec: Dict,
    *,
    draw_info: bool = True,
) -> np.ndarray:
    """Draw V5 contours only; never fill wall/lumen regions.

    The contour layer is alpha-blended with the raw frame so boundaries remain visible
    without obscuring the vessel wall or lumen.
    """
    out = frame.copy()

    if native_mask is not None:
        contour_layer = out.copy()
        outer = native_mask > 0
        lumen = native_mask == 2
        outer_bgr = _rgb_to_bgr(OUTER_RGB)
        lumen_bgr = _rgb_to_bgr(LUMEN_RGB)

        for binary, color in ((outer, outer_bgr), (lumen, lumen_bgr)):
            cs, _ = cv2.findContours(
                binary.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            if cs:
                cv2.drawContours(
                    contour_layer,
                    cs,
                    -1,
                    color,
                    int(CONTOUR_THICKNESS),
                    lineType=cv2.LINE_AA,
                )

        alpha = float(np.clip(CONTOUR_ALPHA, 0.0, 1.0))
        out = cv2.addWeighted(out, 1.0 - alpha, contour_layer, alpha, 0.0)

    if DRAW_ROI_BOX:
        try:
            if int(rec.get("track_valid", 0)) == 1:
                x1, y1, x2, y2 = [int(rec[k]) for k in ["roi_x1", "roi_y1", "roi_x2", "roi_y2"]]
                cv2.rectangle(out, (x1, y1), (x2, y2), (205, 205, 205), 1, cv2.LINE_AA)
        except Exception:
            pass

    if draw_info and DRAW_VIDEO_TEXT:
        lines = [
            f"Frame {int(rec.get('frame_idx', -1))}",
            f"Track {rec.get('track_status', 'NA')} | infer {int(rec.get('inference_valid', 0))}",
        ]
        conf = rec.get("prediction_confidence", np.nan)
        try:
            if np.isfinite(float(conf)):
                lines.append(f"Confidence {float(conf):.3f}")
        except (TypeError, ValueError):
            pass
        for i, text in enumerate(lines):
            y = 27 + i * 25
            cv2.putText(out, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(out, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)

    return out


def write_overlay_video(
    video_path: Path,
    frame_df: pd.DataFrame,
    masks_path: Path,
    out_path: Path,
    fps: float,
    w: int,
    h: int,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(video_path)
    masks = np.load(masks_path, mmap_mode="r", allow_pickle=False)
    render_fps = float(OUTPUT_VIDEO_FPS)
    if not np.isfinite(render_fps) or render_fps <= 0:
        cap.release()
        raise ValueError("OUTPUT_VIDEO_FPS must be a positive finite number")
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), render_fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create overlay video: {out_path}")

    idx = 0
    while idx < len(frame_df):
        ok, frame = cap.read()
        if not ok:
            break
        rec = frame_df.iloc[idx].to_dict()
        native_mask = None
        if int(rec.get("inference_valid", 0)) == 1:
            box = tuple(int(rec[k]) for k in ["roi_x1", "roi_y1", "roi_x2", "roi_y2"])
            native_mask = mask_448_to_native(np.asarray(masks[idx]), box, (h, w))
        writer.write(overlay_native_mask(frame, native_mask, rec))
        idx += 1
        if idx % 200 == 0:
            print(f"  Overlay: {idx}/{len(frame_df)}")

    cap.release()
    writer.release()
