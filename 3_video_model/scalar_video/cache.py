"""Cache operations for SCALAR V4/V5 video segmentation."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
import warnings
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, Iterable, List, Optional, Tuple
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None
try:
    from scipy.ndimage import binary_erosion, distance_transform_edt
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False
from . import settings
from .dataset import annotation_box_from_row
from .roi import crop_exact_box, import_yolo_components, laplacian_blur_score, make_tracked_anchor_box, reconstruction_similarity, select_context_box, yolo_detect_boxes
from .utils import clean_text, letterbox_image_and_mask, load_labelme_mask, progress, progress_postfix, read_csv_compatible, save_json


def _path_signature(raw_path: str, hash_content: bool = False) -> str:
    """Stable cache signature for a referenced source file without hashing large videos."""
    raw = clean_text(raw_path)
    if not raw:
        return ""
    try:
        p = Path(raw)
        if not p.is_file():
            return raw
        st = p.stat()
        parts = [str(p), str(int(st.st_size)), str(int(st.st_mtime_ns))]
        if hash_content:
            h = hashlib.sha1()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            parts.append(h.hexdigest())
        return "|".join(parts)
    except Exception:
        return raw


def _yolo_code_signature() -> str:
    """Track YOLO code and, when discoverable, the configured weight file metadata."""
    if not settings.YOLO_CODE_DIR:
        return ""
    source_path = Path(settings.YOLO_CODE_DIR) / "yolo_roi.py"
    parts = [_path_signature(str(source_path), hash_content=True)]
    try:
        source = source_path.read_text(encoding="utf-8", errors="ignore")
        # Common YOLO implementations keep a literal model_path in a defaults dict.
        m = re.search(r"[\"\']model_path[\"\']\s*:\s*[rRuUbBfF]*[\"\']([^\"\']+)[\"\']", source)
        if m:
            model_path = Path(m.group(1))
            if not model_path.is_absolute():
                model_path = source_path.parent / model_path
            parts.append("MODEL=" + _path_signature(str(model_path), hash_content=False))
    except Exception:
        pass
    return "|".join(parts)


def cache_fingerprint(df: pd.DataFrame) -> str:
    rows = []
    for _, r in df.iterrows():
        crop = []
        for c in ["crop_x1", "crop_y1", "crop_x2", "crop_y2"]:
            try:
                v = float(r.get(c, np.nan))
                crop.append("nan" if not np.isfinite(v) else f"{v:.6f}")
            except Exception:
                crop.append("nan")

        rows.append("\t".join([
            clean_text(r.get("sample_id", "")),
            clean_text(r.get("patient_id", r.get("ID", ""))),
            clean_text(r.get("split", "")).lower(),
            _path_signature(clean_text(r.get("video_path", "")), hash_content=False),
            str(int(r.get("frame_idx", -1))),
            *crop,
            _path_signature(clean_text(r.get("annotation_json_path", "")), hash_content=True),
            _path_signature(clean_text(r.get("annotation_image_path", "")), hash_content=False),
        ]))

    fingerprint_settings = {
        "cache_schema": settings.CACHE_SCHEMA_VERSION,
        "clip_offsets": tuple(settings.CLIP_OFFSETS),
        "image_size": settings.IMAGE_SIZE,
        "reconstruction_qc": settings.RECONSTRUCTION_QC,
        "recon_min_correlation": settings.RECON_MIN_CORRELATION,
        "recon_max_mean_abs_diff": settings.RECON_MAX_MEAN_ABS_DIFF,
        "context_roi_mode": settings.CONTEXT_ROI_MODE,
        "yolo_conf_threshold": settings.YOLO_CONF_THRESHOLD,
        "yolo_code": _yolo_code_signature(),
        "context_weight_good_min": settings.CONTEXT_WEIGHT_GOOD_MIN,
        "context_weight_missing": settings.CONTEXT_WEIGHT_MISSING,
        "context_weight_unstable": settings.CONTEXT_WEIGHT_UNSTABLE,
        "context_weight_out_of_range": settings.CONTEXT_WEIGHT_OUT_OF_RANGE,
        "context_min_iou": settings.CONTEXT_MIN_IOU,
        "context_max_center_shift_image_diag": settings.CONTEXT_MAX_CENTER_SHIFT_IMAGE_DIAG,
        "context_min_size_ratio": settings.CONTEXT_MIN_SIZE_RATIO,
        "context_max_size_ratio": settings.CONTEXT_MAX_SIZE_RATIO,
    }
    payload = "\n".join(sorted(rows)) + "\nSETTINGS=" + json.dumps(fingerprint_settings, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()[:16]


def read_needed_video_frames(video_path: str, needed_indices: List[int]) -> Tuple[Dict[int, np.ndarray], int, int, int, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}, 0, 0, 0, np.nan
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 1e-6:
        fps = 25.0

    valid_needed = sorted({i for i in needed_indices if 0 <= i < total})
    frames = {}
    if valid_needed:
        min_i, max_i = valid_needed[0], valid_needed[-1]
        cap.set(cv2.CAP_PROP_POS_FRAMES, min_i)
        wanted = set(valid_needed)
        for idx in range(min_i, max_i + 1):
            ret, frame = cap.read()
            if not ret:
                break
            if idx in wanted:
                frames[idx] = frame.copy()
                if len(frames) == len(wanted):
                    break
    cap.release()
    return frames, total, width, height, fps


def prepare_clip_cache(split_df: pd.DataFrame, out_dir: Path, force: bool = False) -> pd.DataFrame:
    settings.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_manifest_path = settings.CACHE_DIR / "video_clip_manifest.csv"
    fingerprint = cache_fingerprint(split_df)

    if cache_manifest_path.is_file() and not force:
        old = read_csv_compatible(cache_manifest_path, low_memory=False)
        old_fp = clean_text(old.get("cache_fingerprint", pd.Series([""])).iloc[0]) if len(old) else ""
        if old_fp == fingerprint:
            missing_files = ~old["cache_path"].astype(str).map(lambda x: Path(x).is_file())
            if not missing_files.any():
                print(f"Using existing compatible clip cache: {cache_manifest_path}")
                old.to_csv(out_dir / "video_clip_manifest_used.csv", index=False, encoding="utf-8-sig")
                return old
        print("Clip dataset changed or cache incomplete -> rebuilding cache.")

    YOLO, cvtColor, preprocess_input, resize_image = import_yolo_components()
    print("Loading YOLO ROI detector for one-time clip preparation...")
    yolo = YOLO()
    yolo.confidence = settings.YOLO_CONF_THRESHOLD

    cache_rows = []
    cache_failures = []
    used_cache_paths = set()

    # Grouping by video avoids reopening and rerunning YOLO on the same context frames.
    groups = list(split_df.groupby("video_path", sort=False))
    n_videos = len(groups)
    n_labels = len(split_df)
    print(f"Preparing clip cache from {n_videos} videos / {n_labels} labeled center frames ...")

    # Keep ONE persistent progress bar instead of opening hundreds of nested bars.
    # Per-video YOLO/sample progress is shown in the postfix.
    outer_bar = progress(
        groups,
        "Prepare clips",
        total=n_videos,
        unit="video",
    )

    for video_no, (video_path, g) in enumerate(outer_bar, start=1):
        video_path = str(video_path)
        progress_postfix(
            outer_bar,
            f"video={video_no}/{n_videos} | labels={len(g)} | cached={len(cache_rows)} | failed={len(cache_failures)}",
            refresh=True,
        )
        needed = []
        for _, row in g.iterrows():
            c = int(row["frame_idx"])
            needed.extend([c + off for off in settings.CLIP_OFFSETS])

        frame_map, total_frames, width, height, fps = read_needed_video_frames(video_path, needed)
        if not frame_map:
            for _, row in g.iterrows():
                cache_failures.append({
                    "sample_id": clean_text(row.get("sample_id", "")),
                    "video_path": video_path,
                    "frame_idx": int(row["frame_idx"]),
                    "reason": "cannot_read_required_video_frames",
                })
            continue

        # Run YOLO once for each unique requested frame.
        # This is the slowest part of first-run cache preparation, so expose its
        # within-video progress through the outer tqdm postfix.
        yolo_map: Dict[int, List[Dict[str, Any]]] = {}
        blur_map: Dict[int, float] = {}
        frame_items = list(frame_map.items())
        n_yolo_frames = len(frame_items)
        for yolo_no, (idx, frame) in enumerate(frame_items, start=1):
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yolo_map[idx] = yolo_detect_boxes(
                    yolo, Image.fromarray(rgb), cvtColor, preprocess_input, resize_image
                )
            except Exception:
                yolo_map[idx] = []
            blur_map[idx] = laplacian_blur_score(frame)

            if yolo_no == 1 or yolo_no % 5 == 0 or yolo_no == n_yolo_frames:
                progress_postfix(
                    outer_bar,
                    f"video={video_no}/{n_videos} | YOLO={yolo_no}/{n_yolo_frames} | "
                    f"labels={len(g)} | cached={len(cache_rows)} | failed={len(cache_failures)}",
                )

        n_group_labels = len(g)
        for sample_no, (_, row) in enumerate(g.iterrows(), start=1):
            if sample_no == 1 or sample_no % 5 == 0 or sample_no == n_group_labels:
                progress_postfix(
                    outer_bar,
                    f"video={video_no}/{n_videos} | save={sample_no}/{n_group_labels} | "
                    f"cached={len(cache_rows)} | failed={len(cache_failures)}",
                )
            sample_id = clean_text(row.get("sample_id", "")) or f"row_{int(row.name):07d}"
            center_frame_idx = int(row["frame_idx"])
            anchor = annotation_box_from_row(row)
            if anchor is None:
                cache_failures.append({"sample_id": sample_id, "reason": "invalid_anchor"})
                continue

            center_frame = frame_map.get(center_frame_idx)
            if center_frame is None:
                cache_failures.append({"sample_id": sample_id, "reason": "center_frame_not_read"})
                continue

            # Clamp anchor to actual decoded frame dimensions.
            anchor = dict(anchor)
            anchor["x1"] = max(0, min(width - 1, anchor["x1"]))
            anchor["y1"] = max(0, min(height - 1, anchor["y1"]))
            anchor["x2"] = max(anchor["x1"] + 1, min(width, anchor["x2"]))
            anchor["y2"] = max(anchor["y1"] + 1, min(height, anchor["y2"]))

            center_crop = crop_exact_box(center_frame, anchor)
            if center_crop is None:
                cache_failures.append({"sample_id": sample_id, "reason": "center_crop_failed"})
                continue

            corr, mad = reconstruction_similarity(
                center_crop, clean_text(row.get("annotation_image_path", ""))
            )
            recon_ok = True
            if not settings.RECONSTRUCTION_QC:
                recon_qc_status = "disabled"
            elif np.isfinite(corr) and np.isfinite(mad):
                recon_ok = corr >= settings.RECON_MIN_CORRELATION and mad <= settings.RECON_MAX_MEAN_ABS_DIFF
                recon_qc_status = "verified_pass" if recon_ok else "verified_fail"
            else:
                # Preserve the original inclusion behavior, but do not call missing QC evidence a pass.
                recon_qc_status = "not_verifiable"
            if not recon_ok:
                cache_failures.append({
                    "sample_id": sample_id,
                    "video_path": video_path,
                    "frame_idx": center_frame_idx,
                    "reason": "center_reconstruction_mismatch",
                    "reconstruction_corr": corr,
                    "reconstruction_mad": mad,
                })
                continue

            # Center mask is defined on the saved annotation crop, whose dimensions match
            # crop_w/crop_h in the merged manifest.
            try:
                mask_native = load_labelme_mask(
                    str(row["annotation_json_path"]), center_crop.shape[0], center_crop.shape[1]
                )
            except Exception as e:
                cache_failures.append({
                    "sample_id": sample_id, "reason": "mask_load_failed", "error": str(e)
                })
                continue

            clip_rgb = []
            context_weights = []
            context_status = []
            yolo_confidences = []
            yolo_ious = []
            context_blur = []
            context_boxes = []
            actual_indices = []

            for ti, off in enumerate(settings.CLIP_OFFSETS):
                # Always initialize metadata ROI to the annotation anchor. This prevents
                # boundary/unreadable frames from inheriting a previous frame's box.
                current_box = dict(anchor)
                frame_idx = center_frame_idx + off
                actual_indices.append(frame_idx)
                frame = frame_map.get(frame_idx)
                if frame is None:
                    # Duplicate center pixels only as tensor padding; weight=0 ensures no
                    # temporal evidence is taken from this artificial boundary frame.
                    crop = center_crop.copy()
                    weight = settings.CONTEXT_WEIGHT_OUT_OF_RANGE
                    status = "out_of_range_or_unreadable"
                    conf = np.nan
                    iou = np.nan
                    blur = np.nan
                else:
                    current_box = anchor

                    if ti == settings.CENTER_INDEX:
                        weight = 1.0
                        status = "center_annotated_anchor"
                        conf = 1.0
                        iou = 1.0
                    else:
                        best, status0, weight, metrics = select_context_box(
                            yolo_map.get(frame_idx, []), anchor, width, height
                        )

                        if settings.CONTEXT_ROI_MODE == "yolo_track":
                            current_box = make_tracked_anchor_box(
                                best, anchor, width, height
                            )
                        elif settings.CONTEXT_ROI_MODE == "hybrid" and status0 == "stable":
                            current_box = make_tracked_anchor_box(
                                best, anchor, width, height
                            )

                        status = f"context_{status0}"
                        conf = float(best.get("score", np.nan)) if best is not None else np.nan
                        iou = float(metrics.get("iou", np.nan))

                    crop = crop_exact_box(frame, current_box)

                    if crop is None:
                        crop = center_crop.copy()
                        weight = settings.CONTEXT_WEIGHT_OUT_OF_RANGE
                        status = "crop_failed"
                        conf = np.nan
                        iou = np.nan

                    blur = laplacian_blur_score(crop)

                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                rgb_448, _ = letterbox_image_and_mask(rgb, None, settings.IMAGE_SIZE)
                clip_rgb.append(rgb_448)
                context_weights.append(float(weight))
                context_status.append(status)
                yolo_confidences.append(conf)
                yolo_ious.append(iou)
                context_blur.append(blur)
                context_boxes.append([
                    int(current_box["x1"]),
                    int(current_box["y1"]),
                    int(current_box["x2"]),
                    int(current_box["y2"]),
                ])

            center_rgb = cv2.cvtColor(center_crop, cv2.COLOR_BGR2RGB)
            _, mask_448 = letterbox_image_and_mask(center_rgb, mask_native, settings.IMAGE_SIZE)
            clip_arr = np.stack(clip_rgb).astype(np.uint8)
            weights_arr = np.asarray(context_weights, np.float32)

            safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id)[:120]
            cache_key_payload = "|".join([
                sample_id,
                video_path,
                str(center_frame_idx),
                str(anchor["x1"]), str(anchor["y1"]), str(anchor["x2"]), str(anchor["y2"]),
            ])
            cache_key = hashlib.sha1(cache_key_payload.encode("utf-8", errors="ignore")).hexdigest()[:12]
            subdir = settings.CACHE_DIR / str(row.get("split", "unknown"))
            subdir.mkdir(parents=True, exist_ok=True)
            cache_path = subdir / f"{safe_id}__{cache_key}.npz"
            cache_path_str = str(cache_path).lower()
            if cache_path_str in used_cache_paths:
                raise RuntimeError(f"Cache filename collision detected: {cache_path}")
            used_cache_paths.add(cache_path_str)
            np.savez_compressed(
                cache_path,
                clip=clip_arr,
                mask=mask_448.astype(np.uint8),
                context_weights=weights_arr,
                frame_indices=np.asarray(actual_indices, np.int32),
                yolo_confidences=np.asarray(yolo_confidences, np.float32),
                yolo_ious=np.asarray(yolo_ious, np.float32),
                blur_scores=np.asarray(context_blur, np.float32),
                context_status=np.asarray(context_status, dtype="U40"),
                anchor_box=np.asarray([anchor["x1"], anchor["y1"], anchor["x2"], anchor["y2"]], np.int32),
                context_boxes=np.asarray(context_boxes, np.int32),
            )

            record = row.to_dict()
            record.update({
                "cache_path": str(cache_path),
                "cache_fingerprint": fingerprint,
                "video_total_frames": total_frames,
                "video_width": width,
                "video_height": height,
                "video_fps": fps,
                "reconstruction_corr": corr,
                "reconstruction_mad": mad,
                "reconstruction_qc_status": recon_qc_status,
                "mean_context_weight": float(np.mean(weights_arr)),
                "min_context_weight": float(np.min(weights_arr)),
                "n_good_context": int(np.sum(weights_arr >= settings.CONTEXT_WEIGHT_GOOD_MIN)),
                "n_nonzero_context": int(np.sum(weights_arr > 0)),
                "context_statuses": "|".join(context_status),
                "context_weights": "|".join(f"{x:.3f}" for x in context_weights),
                "context_yolo_confidences": "|".join("nan" if not np.isfinite(x) else f"{x:.3f}" for x in yolo_confidences),
                "context_yolo_ious": "|".join("nan" if not np.isfinite(x) else f"{x:.3f}" for x in yolo_ious),
                "context_roi_mode": settings.CONTEXT_ROI_MODE,
            })
            cache_rows.append(record)

        progress_postfix(
            outer_bar,
            f"video={video_no}/{n_videos} done | cached={len(cache_rows)} | failed={len(cache_failures)}",
            refresh=True,
        )

    cache_df = pd.DataFrame(cache_rows)
    if cache_df.empty:
        raise RuntimeError("No video clips were successfully cached.")
    cache_df.to_csv(cache_manifest_path, index=False, encoding="utf-8-sig")
    cache_df.to_csv(out_dir / "video_clip_manifest_used.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cache_failures).to_csv(
        out_dir / "video_clip_cache_failures.csv", index=False, encoding="utf-8-sig"
    )

    summary = {
        "cache_fingerprint": fingerprint,
        "n_input_sparse_annotations": int(len(split_df)),
        "n_cached": int(len(cache_df)),
        "n_failed": int(len(cache_failures)),
        "clip_offsets": list(settings.CLIP_OFFSETS),
        "image_size": settings.IMAGE_SIZE,
        "mean_reconstruction_corr": float(pd.to_numeric(cache_df["reconstruction_corr"], errors="coerce").mean()),
        "mean_reconstruction_mad": float(pd.to_numeric(cache_df["reconstruction_mad"], errors="coerce").mean()),
        "mean_context_weight": float(cache_df["mean_context_weight"].mean()),
    }
    save_json(out_dir / "video_clip_cache_summary.json", summary)
    print(f"Cached {len(cache_df)}/{len(split_df)} labeled video clips -> {settings.CACHE_DIR}")
    return cache_df

