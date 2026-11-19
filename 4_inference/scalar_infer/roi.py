"""YOLO ROI detection, tracking, fixed crop geometry and native mask restore."""
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
from .settings import (IMAGE_SIZE, TRACK_INTERP_MAX_GAP, TRACK_SMOOTH_WINDOW, TRACK_OUTLIER_WINDOW, TRACK_OUTLIER_MAD_MULT, VIDEO_YOLO_BOX_PERCENTILE, YOLO_TO_CROP_SCALE, TRAINING_ROI_CAP_PERCENTILE, MAX_ROI_SCALE_FROM_TRAIN_MEDIAN)
def choose_detection(
    boxes: List[Dict], prev_center: Optional[Tuple[float, float]], w: int, h: int
) -> Optional[Dict]:
    if not boxes:
        return None
    if prev_center is None:
        return max(boxes, key=lambda b: float(b.get("score", 0.0)))
    diag = max(math.hypot(w, h), 1.0)
    px, py = prev_center

    def score(b):
        cx = (float(b["x1"]) + float(b["x2"])) / 2
        cy = (float(b["y1"]) + float(b["y2"])) / 2
        dist = math.hypot(cx - px, cy - py) / diag
        return float(b.get("score", 0.0)) - 0.75 * dist

    return max(boxes, key=score)


def detect_video(video_path: Path, train_lib) -> Tuple[pd.DataFrame, int, int, float]:
    YOLO, cvtColor, preprocess_input, resize_image = train_lib.import_yolo_components()
    yolo = YOLO()
    yolo.confidence = train_lib.YOLO_CONF_THRESHOLD

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 1e-6:
        fps = 25.0

    rows = []
    prev_center = None
    missing_run = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes = train_lib.yolo_detect_boxes(yolo, pil, cvtColor, preprocess_input, resize_image)
        best = choose_detection(boxes, prev_center, w, h)
        rec = {"frame_idx": idx, "time_sec": idx / fps}
        if best is None:
            missing_run += 1
            if missing_run > TRACK_INTERP_MAX_GAP:
                prev_center = None
            rec.update({
                "yolo_confidence": np.nan,
                "raw_box_w": np.nan,
                "raw_box_h": np.nan,
                "track_x_raw": np.nan,
                "track_y_raw": np.nan,
            })
        else:
            missing_run = 0
            cx = (float(best["x1"]) + float(best["x2"])) / 2
            cy = (float(best["y1"]) + float(best["y2"])) / 2
            prev_center = (cx, cy)
            rec.update({
                "yolo_confidence": float(best.get("score", np.nan)),
                "raw_box_w": float(best["x2"] - best["x1"]),
                "raw_box_h": float(best["y2"] - best["y1"]),
                "track_x_raw": cx,
                "track_y_raw": cy,
            })
        rows.append(rec)
        idx += 1
        if idx % 100 == 0:
            print(f"  YOLO: {idx} frames")

    cap.release()
    del yolo
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not rows:
        raise RuntimeError(f"No readable frames: {video_path}")
    return pd.DataFrame(rows), w, h, fps


def _fill_short_gaps(values: np.ndarray, valid: np.ndarray, max_gap: int) -> Tuple[np.ndarray, np.ndarray]:
    out = values.astype(float).copy()
    filled = np.zeros(len(out), dtype=bool)
    i = 0
    while i < len(out):
        if valid[i]:
            i += 1
            continue
        start = i
        while i < len(out) and not valid[i]:
            i += 1
        end = i - 1
        gap = end - start + 1
        left, right = start - 1, i
        if gap <= max_gap and left >= 0 and right < len(out) and valid[left] and valid[right]:
            out[start:right] = np.linspace(out[left], out[right], gap + 2)[1:-1]
            filled[start:right] = True
    return out, filled


def build_roi_track(det: pd.DataFrame, w: int, h: int) -> pd.DataFrame:
    out = det.copy()
    x = pd.to_numeric(out["track_x_raw"], errors="coerce")
    y = pd.to_numeric(out["track_y_raw"], errors="coerce")
    raw_valid = x.notna() & y.notna()

    med_x = x.rolling(TRACK_OUTLIER_WINDOW, center=True, min_periods=1).median()
    med_y = y.rolling(TRACK_OUTLIER_WINDOW, center=True, min_periods=1).median()
    residual = np.sqrt((x - med_x) ** 2 + (y - med_y) ** 2) / max(math.hypot(w, h), 1.0)
    rv = residual[raw_valid].dropna().to_numpy(dtype=float)
    if len(rv):
        med = float(np.median(rv))
        mad = float(np.median(np.abs(rv - med)))
        threshold = max(0.08, med + TRACK_OUTLIER_MAD_MULT * 1.4826 * mad)
    else:
        threshold = np.inf
    keep = raw_valid & (residual <= threshold)

    x0 = x.to_numpy(dtype=float)
    y0 = y.to_numpy(dtype=float)
    x0[~keep.to_numpy()] = np.nan
    y0[~keep.to_numpy()] = np.nan
    valid0 = np.isfinite(x0) & np.isfinite(y0)

    xi, fill_x = _fill_short_gaps(x0, valid0, TRACK_INTERP_MAX_GAP)
    yi, fill_y = _fill_short_gaps(y0, valid0, TRACK_INTERP_MAX_GAP)
    interpolated = fill_x & fill_y
    valid = np.isfinite(xi) & np.isfinite(yi)

    sx = pd.Series(xi).rolling(TRACK_SMOOTH_WINDOW, center=True, min_periods=1).median().to_numpy()
    sy = pd.Series(yi).rolling(TRACK_SMOOTH_WINDOW, center=True, min_periods=1).median().to_numpy()
    sx[~valid] = np.nan
    sy[~valid] = np.nan

    status = np.full(len(out), "missing", dtype=object)
    status[keep.to_numpy()] = "detected"
    status[interpolated] = "interpolated"

    segment = np.full(len(out), -1, dtype=int)
    seg = -1
    prev_valid = False
    for i, ok in enumerate(valid):
        if ok and not prev_valid:
            seg += 1
        if ok:
            segment[i] = seg
        prev_valid = bool(ok)

    out["track_status"] = status
    out["track_x"] = sx
    out["track_y"] = sy
    out["segment_id"] = segment
    out["track_outlier"] = (raw_valid & ~keep).astype(int)
    return out


def canonical_box(cx: float, cy: float, crop_w: int, crop_h: int, w: int, h: int) -> Tuple[int, int, int, int]:
    crop_w = min(max(8, int(crop_w)), w)
    crop_h = min(max(8, int(crop_h)), h)
    x1 = int(round(cx - crop_w / 2))
    y1 = int(round(cy - crop_h / 2))
    x1 = max(0, min(max(w - crop_w, 0), x1))
    y1 = max(0, min(max(h - crop_h, 0), y1))
    return x1, y1, x1 + crop_w, y1 + crop_h


def estimate_crop_size(track: pd.DataFrame, w: int, h: int, cache_manifest: Path) -> Tuple[int, int, str]:
    """Choose one fixed crop size for the whole video, staying close to training scale.

    Base rule is the ORIGINAL pipeline: use the median training-cache anchor size.
    The only change is a conservative, bounded enlargement when this video's larger
    YOLO detections need a little more room. Enlargement is capped both by the
    training-cache upper ROI distribution and by MAX_ROI_SCALE_FROM_TRAIN_MEDIAN.

    Crop width/height remain fixed throughout the video; only the tracked centre moves.
    """
    ratios_w, ratios_h = [], []
    cache_manifest = Path(cache_manifest)
    if cache_manifest.is_file():
        try:
            cache_df = pd.read_csv(cache_manifest, low_memory=False)
            if "split" in cache_df.columns:
                cache_df = cache_df[cache_df["split"].astype(str).str.lower().eq("train")]
            if len(cache_df) > 600:
                cache_df = cache_df.sample(600, random_state=42)
            for row in cache_df.itertuples(index=False):
                path = Path(str(getattr(row, "cache_path", "")))
                vw = float(getattr(row, "video_width", np.nan))
                vh = float(getattr(row, "video_height", np.nan))
                if not path.is_file() or not (np.isfinite(vw) and np.isfinite(vh) and vw > 0 and vh > 0):
                    continue
                try:
                    with np.load(path, allow_pickle=False) as z:
                        box = z["anchor_box"].astype(float)
                    ratios_w.append((box[2] - box[0]) / vw)
                    ratios_h.append((box[3] - box[1]) / vh)
                except Exception:
                    continue
        except Exception:
            pass

    if ratios_w and ratios_h:
        # ORIGINAL training-scale ROI: this remains the base crop.
        base_rw = float(np.median(ratios_w))
        base_rh = float(np.median(ratios_h))
        base_w = max(1.0, base_rw * w)
        base_h = max(1.0, base_rh * h)

        # Upper crop scale that was still represented in the training cache.
        train_upper_rw = float(np.percentile(ratios_w, TRAINING_ROI_CAP_PERCENTILE))
        train_upper_rh = float(np.percentile(ratios_h, TRAINING_ROI_CAP_PERCENTILE))
        train_cap_scale = min(
            train_upper_rw / max(base_rw, 1e-8),
            train_upper_rh / max(base_rh, 1e-8),
        )
        train_cap_scale = max(1.0, float(train_cap_scale))
        allowed_scale = min(float(MAX_ROI_SCALE_FROM_TRAIN_MEDIAN), train_cap_scale)
        allowed_scale = max(1.0, allowed_scale)

        # Use only robustly retained YOLO detections when possible.
        if "track_status" in track.columns:
            detected = track["track_status"].astype(str).str.lower().eq("detected")
        else:
            detected = pd.Series(True, index=track.index)

        bw = pd.to_numeric(track.loc[detected, "raw_box_w"], errors="coerce")
        bh = pd.to_numeric(track.loc[detected, "raw_box_h"], errors="coerce")
        bw = bw[np.isfinite(bw) & (bw > 0)].to_numpy(dtype=float)
        bh = bh[np.isfinite(bh) & (bh > 0)].to_numpy(dtype=float)

        requested_scale = 1.0
        if len(bw) and len(bh):
            yolo_w = float(np.percentile(bw, VIDEO_YOLO_BOX_PERCENTILE))
            yolo_h = float(np.percentile(bh, VIDEO_YOLO_BOX_PERCENTILE))

            # Same geometric convention as the original YOLO fallback (about 2x box size),
            # but now it can only produce a SMALL enlargement of the training-scale crop.
            requested_scale = max(
                1.0,
                yolo_w * float(YOLO_TO_CROP_SCALE) / base_w,
                yolo_h * float(YOLO_TO_CROP_SCALE) / base_h,
            )

        applied_scale = min(requested_scale, allowed_scale)
        rw = base_rw * applied_scale
        rh = base_rh * applied_scale

        if applied_scale > 1.0001:
            source = "training_cache_anchor_prior_conservative_enlargement"
        else:
            source = "training_cache_anchor_prior"
    else:
        # Preserve the ORIGINAL fallback exactly when the training cache is unavailable.
        bw = pd.to_numeric(track["raw_box_w"], errors="coerce").dropna().to_numpy()
        bh = pd.to_numeric(track["raw_box_h"], errors="coerce").dropna().to_numpy()
        if not len(bw) or not len(bh):
            raise RuntimeError("Cannot estimate crop size: no training-cache prior or YOLO boxes.")
        rw = float(np.median(bw) * 2.0 / w)
        rh = float(np.median(bh) * 2.0 / h)
        source = "yolo_box_fallback"

    rw = float(np.clip(rw, 0.08, 1.0))
    rh = float(np.clip(rh, 0.08, 1.0))
    crop_w = min(w, max(32, int(round(rw * w))))
    crop_h = min(h, max(32, int(round(rh * h))))
    return crop_w, crop_h, source


def attach_canonical_boxes(track: pd.DataFrame, crop_w: int, crop_h: int, w: int, h: int) -> pd.DataFrame:
    out = track.copy()
    x1s, y1s, x2s, y2s = [], [], [], []
    for r in out.itertuples(index=False):
        if not np.isfinite(r.track_x) or not np.isfinite(r.track_y):
            x1s.append(np.nan); y1s.append(np.nan); x2s.append(np.nan); y2s.append(np.nan)
            continue
        x1, y1, x2, y2 = canonical_box(r.track_x, r.track_y, crop_w, crop_h, w, h)
        x1s.append(x1); y1s.append(y1); x2s.append(x2); y2s.append(y2)
    out["roi_x1"], out["roi_y1"], out["roi_x2"], out["roi_y2"] = x1s, y1s, x2s, y2s
    out["track_valid"] = np.isfinite(out["roi_x1"]).astype(int)
    return out


def crop_to_448_rgb(frame_bgr: np.ndarray, box: Tuple[int, int, int, int], train_lib) -> np.ndarray:
    x1, y1, x2, y2 = box
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        raise RuntimeError(f"Empty crop: {box}")
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    rgb448, _ = train_lib.letterbox_image_and_mask(rgb, None, IMAGE_SIZE)
    return rgb448


def mask_448_to_native(mask448: np.ndarray, box: Tuple[int, int, int, int], native_hw: Tuple[int, int]) -> np.ndarray:
    h_native, w_native = native_hw
    x1, y1, x2, y2 = box
    crop_w, crop_h = x2 - x1, y2 - y1
    scale = min(IMAGE_SIZE / max(crop_w, 1), IMAGE_SIZE / max(crop_h, 1))
    nw, nh = max(1, round(crop_w * scale)), max(1, round(crop_h * scale))
    left, top = (IMAGE_SIZE - nw) // 2, (IMAGE_SIZE - nh) // 2
    inner = np.asarray(mask448)[top:top + nh, left:left + nw]
    native_crop = cv2.resize(inner, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
    out = np.zeros((h_native, w_native), np.uint8)
    out[y1:y2, x1:x2] = native_crop
    return out
