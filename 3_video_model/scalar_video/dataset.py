"""Dataset operations for SCALAR V4/V5 video segmentation."""
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
from .utils import clean_text, image_tensor, inspect_labelme, progress, resolve_file_path, resolve_video_path


def annotation_box_from_row(row: pd.Series) -> Optional[Dict[str, Any]]:
    """Return native-video annotation ROI for both old and new labels.

    Old annotations:
        crop_x1, crop_y1, crop_x2, crop_y2

    New targeted hard-case editor:
        roi_x1, roi_y1, roi_x2, roi_y2
    """
    candidates = [
        ("crop_x1", "crop_y1", "crop_x2", "crop_y2", "crop"),
        ("roi_x1", "roi_y1", "roi_x2", "roi_y2", "roi"),
    ]
    for x1c, y1c, x2c, y2c, source in candidates:
        try:
            vals = [float(row[c]) for c in [x1c, y1c, x2c, y2c]]
        except Exception:
            continue
        if not all(np.isfinite(vals)):
            continue

        x1, y1, x2, y2 = vals
        x1, x2 = sorted([int(round(x1)), int(round(x2))])
        y1, y2 = sorted([int(round(y1)), int(round(y2))])
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue

        return {
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "score": 1.0,
            "class": "annotation_anchor",
            "class_id": -1,
            "anchor_source": source,
        }
    return None


def audit_sparse_video_annotations(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    usable = []
    for idx, row in progress(
        df.iterrows(),
        "Audit sparse video labels",
        total=len(df),
        unit="row",
    ):
        json_path = resolve_file_path(row.get("annotation_json_path", ""))
        image_path = resolve_file_path(row.get("annotation_image_path", ""))
        video_path = resolve_video_path(row.get("video_path", ""))
        box = annotation_box_from_row(row)
        try:
            frame_idx = int(row.get("frame_idx", -1))
        except Exception:
            frame_idx = -1

        has_wall = has_lumen = False
        json_h = json_w = 0
        if json_path:
            has_wall, has_lumen, json_h, json_w = inspect_labelme(json_path)

        reason = "included"
        include = True
        if not json_path:
            include, reason = False, "json_missing"
        elif not (has_wall and has_lumen):
            include, reason = False, "missing_wall_or_lumen"
        elif not video_path:
            include, reason = False, "video_missing"
        elif frame_idx < 0:
            include, reason = False, "invalid_frame_idx"
        elif box is None:
            include, reason = False, "invalid_annotation_crop_box"

        rec = {
            "manifest_row": idx,
            "sample_id": clean_text(row.get("sample_id", "")),
            "patient_id": clean_text(row.get("patient_id", row.get("ID", ""))),
            "stored_video_path": clean_text(row.get("video_path", "")),
            "resolved_video_path": video_path,
            "frame_idx": frame_idx,
            "resolved_json_path": json_path,
            "resolved_annotation_image_path": image_path,
            "json_h": json_h,
            "json_w": json_w,
            "annotation_anchor_source": (
                str(box.get("anchor_source", "")) if box is not None else ""
            ),
            "include": int(include),
            "reason": reason,
        }
        rows.append(rec)

        if include:
            r = row.to_dict()
            r["video_path"] = video_path
            r["annotation_json_path"] = json_path
            r["annotation_image_path"] = image_path
            r["frame_idx"] = frame_idx

            # Canonical form used by the existing reconstruction/cache pipeline.
            r["crop_x1"] = int(box["x1"])
            r["crop_y1"] = int(box["y1"])
            r["crop_x2"] = int(box["x2"])
            r["crop_y2"] = int(box["y2"])
            r["annotation_anchor_source"] = str(box.get("anchor_source", ""))
            usable.append(r)

    audit = pd.DataFrame(rows)
    audit.to_csv(out_dir / "video_annotation_audit.csv", index=False, encoding="utf-8-sig")
    audit[audit["include"] != 1].to_csv(
        out_dir / "video_annotation_excluded.csv", index=False, encoding="utf-8-sig"
    )
    usable_df = pd.DataFrame(usable)
    if usable_df.empty:
        raise RuntimeError("No usable sparse video annotations remain after audit.")
    return usable_df


def attach_image_split(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Use a supplied, frozen split. Do not re-split study participants.

    The private manifest is prepared externally by the data custodian. No
    participant identifiers or clinical linkage tables ship with this repo.
    """
    if "split" not in df.columns:
        raise ValueError("Manifest must contain a preassigned 'split' column (train/val/test).")
    out = df.copy()
    out["split"] = out["split"].fillna("").astype(str).str.lower().str.strip()
    allowed = {"train", "val", "test"}
    if not out["split"].isin(allowed).all():
        raise ValueError("Every valid annotation must have a train/val/test split.")
    # Where available, enforce participant-level and video-level exclusivity.
    for key in ("patient_id", "video_path"):
        if key not in out.columns:
            continue
        nonempty = out[key].fillna("").astype(str).str.strip().ne("")
        if not nonempty.any():
            continue
        cross = out.loc[nonempty].groupby(key)["split"].nunique()
        if (cross > 1).any():
            raise RuntimeError(f"Leakage: {int((cross>1).sum())} {key} group(s) cross splits.")
    out["image_split_match"] = "provided_frozen_split"
    # Private local output, excluded by .gitignore.
    out.to_csv(out_dir / "video_training_split_used.csv", index=False, encoding="utf-8-sig")
    return out


def augment_video_clip(
    clip_rgb: np.ndarray,
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    clip = clip_rgb.copy()
    m = mask.copy()
    T, H, W, _ = clip.shape

    # Same spatial transform for every frame and the center mask.
    if random.random() < 0.5:
        clip = np.ascontiguousarray(clip[:, :, ::-1])
        m = np.ascontiguousarray(m[:, ::-1])

    if random.random() < 0.65:
        angle = random.uniform(-6.0, 6.0)
        scale = random.uniform(0.96, 1.04)
        M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, scale)
        clip2 = []
        for t in range(T):
            clip2.append(cv2.warpAffine(
                clip[t], M, (W, H), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            ))
        clip = np.stack(clip2)
        m = cv2.warpAffine(m, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)

    # Global gain/gamma change shared across the clip.
    x = clip.astype(np.float32) / 255.0
    if random.random() < 0.5:
        gamma = random.uniform(0.88, 1.12)
        x = np.power(np.clip(x, 0, 1), gamma)
    if random.random() < 0.5:
        x = x * random.uniform(0.90, 1.10) + random.uniform(-0.03, 0.03)

    # Small frame-specific speckle variation is realistic and prevents the temporal
    # module from depending on identical noise patterns.
    if random.random() < 0.35:
        for t in range(T):
            noise = np.random.normal(0, random.uniform(0.008, 0.025), size=(H, W, 1)).astype(np.float32)
            x[t] = x[t] * (1.0 + noise)
    clip = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    return clip, m


def _corrupt_adaptive_center(frame_rgb: np.ndarray) -> np.ndarray:
    """Create a realistic local information deficit on the labeled center frame.

    This is used only while training V5. The GT remains unchanged, forcing the
    learned gate to discover when neighboring sweep frames are more informative.
    It does not impose any fixed inference-time center/context ratio.
    """
    x = frame_rgb.astype(np.float32) / 255.0
    H, W = x.shape[:2]
    op = random.choice(["blur", "low_contrast", "shadow", "local_dropout", "speckle"])

    if op == "blur":
        k = random.choice([9, 13, 17, 21])
        x = cv2.GaussianBlur(x, (k, k), random.uniform(1.5, 4.0))

    elif op == "low_contrast":
        mean = x.mean(axis=(0, 1), keepdims=True)
        x = mean + random.uniform(0.25, 0.60) * (x - mean)
        x *= random.uniform(0.75, 1.05)

    elif op == "shadow":
        mask = np.zeros((H, W), np.float32)
        cx = random.randint(W // 4, max(W // 4, 3 * W // 4))
        cy = random.randint(H // 4, max(H // 4, 3 * H // 4))
        ax = random.randint(max(12, W // 10), max(13, W // 3))
        ay = random.randint(max(12, H // 10), max(13, H // 3))
        cv2.ellipse(mask, (cx, cy), (ax, ay), random.uniform(0, 180), 0, 360, 1.0, -1)
        sigma = max(5.0, min(H, W) * random.uniform(0.025, 0.060))
        mask = cv2.GaussianBlur(mask, (0, 0), sigma)
        attenuation = random.uniform(0.12, 0.45)
        x *= (1.0 - mask[..., None] * (1.0 - attenuation))

    elif op == "local_dropout":
        rw = random.randint(max(16, W // 8), max(17, W // 3))
        rh = random.randint(max(16, H // 8), max(17, H // 3))
        x1 = random.randint(0, max(0, W - rw))
        y1 = random.randint(0, max(0, H - rh))
        patch_mean = x[max(0, y1-rh//4):min(H, y1+rh+rh//4),
                       max(0, x1-rw//4):min(W, x1+rw+rw//4)].mean(axis=(0, 1), keepdims=True)
        x[y1:y1+rh, x1:x1+rw] = patch_mean

    else:  # stronger frame-specific speckle
        noise = np.random.normal(0, random.uniform(0.06, 0.14), size=(H, W, 1)).astype(np.float32)
        x = x * (1.0 + noise)

    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def augment_adaptive_rescue_clip(
    clip_rgb: np.ndarray,
    context_weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """V5-only center corruption + context dropout.

    The original numeric QC weights are preserved for audit, but V5 uses them only
    as an availability mask (>0). Context dropout temporarily sets selected usable
    neighbors to zero during training so the model cannot depend on a fixed frame.
    """
    clip = clip_rgb.copy()
    weights = context_weights.astype(np.float32).copy()
    weights[settings.CENTER_INDEX] = 1.0

    if random.random() < settings.V5_CENTER_CORRUPT_PROB:
        clip[settings.CENTER_INDEX] = _corrupt_adaptive_center(clip[settings.CENTER_INDEX])

    if random.random() < settings.V5_CONTEXT_DROPOUT_PROB:
        candidates = [
            i for i in range(len(weights))
            if i != settings.CENTER_INDEX and weights[i] > 0
        ]
        # Keep at least one real neighbor whenever possible.
        if len(candidates) > 1:
            max_drop = min(2, len(candidates) - 1)
            n_drop = random.randint(1, max_drop)
            for i in random.sample(candidates, n_drop):
                weights[i] = 0.0

    return clip, weights


class VideoClipDataset(Dataset):
    def __init__(self, df: pd.DataFrame, training: bool, adaptive_rescue: bool = False):
        self.df = df.reset_index(drop=True).copy()
        self.training = training
        self.adaptive_rescue = bool(adaptive_rescue)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        with np.load(str(row["cache_path"]), allow_pickle=False) as z:
            clip = z["clip"].astype(np.uint8)
            mask = z["mask"].astype(np.uint8)
            weights = z["context_weights"].astype(np.float32)
            frame_indices = z["frame_indices"].astype(np.int32)

        if self.training:
            clip, mask = augment_video_clip(clip, mask)
            if self.adaptive_rescue:
                clip, weights = augment_adaptive_rescue_clip(clip, weights)

        tensors = torch.stack([image_tensor(clip[t]) for t in range(len(clip))], dim=0)
        return {
            "clip": tensors,                                  # [T,3,H,W]
            "mask": torch.from_numpy(mask.astype(np.int64)),
            "context_weights": torch.from_numpy(weights),     # [T]
            "frame_indices": torch.from_numpy(frame_indices),
            "sample_name": clean_text(row.get("sample_id", "")) or Path(str(row["cache_path"])).stem,
            "patient_id": clean_text(row.get("patient_id", "")),
            "video_path": clean_text(row.get("video_path", "")),
            "center_frame_idx": int(row.get("frame_idx", -1)),
            "cache_path": str(row["cache_path"]),
        }

