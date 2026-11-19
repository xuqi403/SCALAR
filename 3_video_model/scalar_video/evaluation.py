"""Evaluation operations for SCALAR V4/V5 video segmentation."""
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
from .metrics import dice_np, one_metrics
from .postprocess import decode_dual_logits, topology_cleanup
from .utils import base_model, denormalize_tensor, progress


def binary_entropy_confidence(p: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    p = p.clamp(eps, 1 - eps)
    ent = -(p * torch.log2(p) + (1 - p) * torch.log2(1 - p))
    return 1.0 - ent


@torch.no_grad()
def image_model_all_frames(image_model: nn.Module, clip: torch.Tensor) -> Dict[str, torch.Tensor]:
    B, T, C, H, W = clip.shape
    out = image_model(clip.view(B * T, C, H, W))
    return {
        "outer_logits": out["outer_logits"].view(B, T, 1, H, W),
        "lumen_logits": out["lumen_logits"].view(B, T, 1, H, W),
    }


def clean_sequence_from_logits(seq_out: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, List[List[Dict[str, float]]]]:
    B, T, _, H, W = seq_out["outer_logits"].shape
    po = torch.sigmoid(seq_out["outer_logits"][:, :, 0]) >= settings.PRED_THRESHOLD
    pl = torch.sigmoid(seq_out["lumen_logits"][:, :, 0]) >= settings.PRED_THRESHOLD
    all_masks = np.zeros((B, T, H, W), np.uint8)
    stats_all = []
    for b in range(B):
        bstats = []
        for t in range(T):
            m, s = topology_cleanup(po[b, t].cpu().numpy(), pl[b, t].cpu().numpy())
            all_masks[b, t] = m
            bstats.append(s)
        stats_all.append(bstats)
    return all_masks, stats_all


def area_jump_against_neighbors(masks: np.ndarray, center_index: int, class_kind: str, weights: np.ndarray) -> float:
    if class_kind == "outer":
        areas = np.sum(masks > 0, axis=(-2, -1)).astype(float)
    else:
        areas = np.sum(masks == 2, axis=(-2, -1)).astype(float)
    valid = np.arange(len(areas)) != center_index
    valid &= weights > 0
    neigh = areas[valid]
    if len(neigh) == 0:
        return 0.0
    ref = float(np.median(neigh))
    center = float(areas[center_index])
    return abs(center - ref) / max((center + ref) * 0.5, 1.0)


@torch.no_grad()
def evaluate_paired(
    video_model: Optional[nn.Module],
    image_model: nn.Module,
    loader: DataLoader,
    device,
    use_amp: bool,
    phase: str,
    plot_module=None,
    qualitative_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    image_model.eval()
    if video_model is not None:
        video_model.eval()

    rows = []
    qualitative_cases = []

    for batch in progress(loader, phase, total=len(loader), unit="batch"):
        clip = batch["clip"].to(device, non_blocking=True)
        gt_t = batch["mask"].to(device, non_blocking=True)
        weights_t = batch["context_weights"].to(device, non_blocking=True)
        weights_np = weights_t.cpu().numpy()

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            seq_image_out = image_model_all_frames(image_model, clip)
            center_image_out = {
                "outer_logits": seq_image_out["outer_logits"][:, settings.CENTER_INDEX],
                "lumen_logits": seq_image_out["lumen_logits"][:, settings.CENTER_INDEX],
            }
            if video_model is not None:
                video_out = video_model(clip, weights_t)
            else:
                video_out = center_image_out

            context_out = None
            if video_model is not None and "context_feature" in video_out:
                # Evaluation only: decode the context-only branch to verify that the
                # learned gate is choosing between two genuinely meaningful sources.
                context_out = base_model(video_model).decoder(
                    video_out["context_feature"], (clip.shape[-2], clip.shape[-1])
                )

        image_raw, image_post, image_top, pio, pil = decode_dual_logits(center_image_out)
        _, video_post, _, _, _ = decode_dual_logits(video_out)
        context_post = decode_dual_logits(context_out)[1] if context_out is not None else None
        image_seq_masks, image_seq_stats = clean_sequence_from_logits(seq_image_out)

        # Center-frame segmentation confidence, focused on pixels with at least some
        # foreground probability so the large black background does not dominate.
        pio = pio.detach()
        pil = pil.detach()
        region = (pio > 0.10) | (pil > 0.10)
        conf_o = binary_entropy_confidence(pio)
        conf_l = binary_entropy_confidence(pil)
        conf_map = 0.5 * (conf_o + conf_l)
        conf_num = (conf_map * region.float()).sum(dim=(-2, -1))
        conf_den = region.float().sum(dim=(-2, -1)).clamp_min(1.0)
        confidence = (conf_num / conf_den).cpu().numpy()

        gt = gt_t.cpu().numpy()
        image_np = image_post.cpu().numpy()
        video_np = video_post.cpu().numpy()
        context_np = context_post.cpu().numpy() if context_post is not None else None
        raw_np = image_raw.cpu().numpy()

        gate_np = video_out.get("gate") if isinstance(video_out, dict) else None
        alpha_np = video_out.get("neighbor_alpha") if isinstance(video_out, dict) else None
        offsets_np = video_out.get("alignment_offsets") if isinstance(video_out, dict) else None
        if gate_np is not None:
            gate_np = gate_np.detach().cpu().numpy()[:, 0]
            alpha_np = alpha_np.detach().cpu().numpy()
            offsets_np = offsets_np.detach().cpu().numpy()
            gt_low = F.interpolate(
                (gt_t > 0).float().unsqueeze(1),
                size=gate_np.shape[-2:],
                mode="nearest",
            )[:, 0].cpu().numpy().astype(bool)
        else:
            gt_low = None

        for i in range(len(gt)):
            jump_o = area_jump_against_neighbors(image_seq_masks[i], settings.CENTER_INDEX, "outer", weights_np[i])
            jump_l = area_jump_against_neighbors(image_seq_masks[i], settings.CENTER_INDEX, "lumen", weights_np[i])
            top = image_top[i]
            reasons = []
            if confidence[i] < settings.QC_CONFIDENCE_MIN:
                reasons.append("low_confidence")
            if top.get("raw_outer_components", 1) > 1 or top.get("raw_lumen_components", 1) > 1:
                reasons.append("multi_component")
            if top.get("raw_lumen_outside_outer_px", 0) > settings.QC_RAW_LUMEN_OUTSIDE_MAX_PX:
                reasons.append("lumen_outside_outer")
            if jump_o > settings.QC_OUTER_AREA_JUMP_MAX:
                reasons.append("outer_area_jump")
            if jump_l > settings.QC_LUMEN_AREA_JUMP_MAX:
                reasons.append("lumen_area_jump")
            trigger = len(reasons) > 0 and video_model is not None
            conditional = video_np[i] if trigger else image_np[i]

            rec = {
                "sample_name": batch["sample_name"][i],
                "patient_id": batch["patient_id"][i],
                "video_path": batch["video_path"][i],
                "frame_idx": int(batch["center_frame_idx"][i]),
                "qc_trigger": int(trigger),
                "qc_reasons": ";".join(reasons),
                "image_confidence": float(confidence[i]),
                "image_outer_area_jump": float(jump_o),
                "image_lumen_area_jump": float(jump_l),
            }
            rec.update(one_metrics(image_np[i], gt[i], prefix="image_"))
            rec.update(one_metrics(video_np[i], gt[i], prefix="video_"))
            rec.update(one_metrics(conditional, gt[i], prefix="conditional_"))
            if context_np is not None:
                rec.update(one_metrics(context_np[i], gt[i], prefix="context_"))
            rec["delta_wall_dice_video_minus_image"] = rec["video_wall_dice"] - rec["image_wall_dice"]
            rec["delta_wall_dice_conditional_minus_image"] = rec["conditional_wall_dice"] - rec["image_wall_dice"]

            if gate_np is not None:
                region = gt_low[i]
                if not region.any():
                    region = np.ones_like(region, dtype=bool)
                gate_vals = gate_np[i][region]
                rec["gate_mean"] = float(np.mean(gate_vals))
                rec["gate_p10"] = float(np.percentile(gate_vals, 10))
                rec["gate_p90"] = float(np.percentile(gate_vals, 90))
                rec["context_use_mean"] = 1.0 - rec["gate_mean"]
                for t, off in enumerate(settings.CLIP_OFFSETS):
                    rec[f"alpha_{off:+d}"] = float(np.mean(alpha_np[i, t][region]))
                mag = np.sqrt(np.sum(offsets_np[i] ** 2, axis=1))
                valid_neighbor = np.array([off != 0 for off in settings.CLIP_OFFSETS], dtype=bool)
                rec["alignment_offset_feature_px_mean"] = float(np.mean(mag[valid_neighbor][:, region]))

            if context_np is not None:
                rec["context_wall_advantage_vs_image"] = rec["context_wall_dice"] - rec["image_wall_dice"]
            rows.append(rec)

            if qualitative_path is not None and len(qualitative_cases) < settings.FIXED_PREVIEW_N:
                # Save the already calculated Dice metrics together with each
                # qualitative example. The plotting module can directly display
                # O/W/L Dice without repeating prediction or metric calculation.
                image_metrics = {
                    "outer_dice": float(rec["image_outer_dice"]),
                    "wall_dice": float(rec["image_wall_dice"]),
                    "lumen_dice": float(rec["image_lumen_dice"]),
                }
                video_metrics = {
                    "outer_dice": float(rec["video_outer_dice"]),
                    "wall_dice": float(rec["video_wall_dice"]),
                    "lumen_dice": float(rec["video_lumen_dice"]),
                }
                conditional_metrics = {
                    "outer_dice": float(rec["conditional_outer_dice"]),
                    "wall_dice": float(rec["conditional_wall_dice"]),
                    "lumen_dice": float(rec["conditional_lumen_dice"]),
                }

                qualitative_cases.append({
                    "clip_rgb": [denormalize_tensor(clip[i, t]) for t in range(clip.shape[1])],
                    "gt": gt[i],
                    "image_pred": image_np[i],
                    "video_pred": video_np[i],
                    "conditional_pred": conditional,
                    "weights": weights_np[i],
                    "name": batch["sample_name"][i],
                    "trigger": trigger,
                    "reasons": ";".join(reasons),
                    "context_pred": context_np[i] if context_np is not None else None,
                    "gate_map": gate_np[i] if gate_np is not None else None,
                    "alpha_means": [float(rec.get(f"alpha_{off:+d}", np.nan)) for off in settings.CLIP_OFFSETS],
                    "context_metrics": ({
                        "outer_dice": float(rec.get("context_outer_dice", np.nan)),
                        "wall_dice": float(rec.get("context_wall_dice", np.nan)),
                        "lumen_dice": float(rec.get("context_lumen_dice", np.nan)),
                    } if context_np is not None else None),

                    # Added for HD qualitative visualization
                    "image_metrics": image_metrics,
                    "video_metrics": video_metrics,
                    "conditional_metrics": conditional_metrics,
                })

    paired_df = pd.DataFrame(rows)
    summary_rows = []
    summary_methods = ["image", "video", "conditional"]
    if "context_wall_dice" in paired_df.columns:
        summary_methods.append("context")
    for method in summary_methods:
        summary_rows.append({
            "method": method,
            "n": len(paired_df),
            "outer_dice": paired_df[f"{method}_outer_dice"].mean(),
            "wall_dice": paired_df[f"{method}_wall_dice"].mean(),
            "lumen_dice": paired_df[f"{method}_lumen_dice"].mean(),
            "outer_hd95_px": paired_df[f"{method}_outer_hd95_px"].mean(),
            "lumen_hd95_px": paired_df[f"{method}_lumen_hd95_px"].mean(),
            "outer_hd95_evaluable_rate": paired_df[f"{method}_outer_hd95_evaluable"].mean(),
            "lumen_hd95_evaluable_rate": paired_df[f"{method}_lumen_hd95_evaluable"].mean(),
            "outer_empty_prediction_rate": paired_df[f"{method}_outer_empty_prediction"].mean(),
            "lumen_empty_prediction_rate": paired_df[f"{method}_lumen_empty_prediction"].mean(),
            "wall_area_rel_error": paired_df[f"{method}_wall_area_rel_error"].mean(),
        })
    summary_df = pd.DataFrame(summary_rows)

    if qualitative_path is not None and plot_module is not None and qualitative_cases:
        plot_module.save_clip_case_grid(qualitative_cases, qualitative_path)
        if any(c.get("gate_map") is not None for c in qualitative_cases) and hasattr(plot_module, "save_adaptive_case_grid"):
            adaptive_path = qualitative_path.with_name(
                qualitative_path.stem + "_adaptive" + qualitative_path.suffix
            )
            plot_module.save_adaptive_case_grid(qualitative_cases, adaptive_path)
    return paired_df, summary_df


def _safe_nanmean(values) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else np.nan


def _mask_area_sequence(masks: np.ndarray, class_kind: str) -> np.ndarray:
    if class_kind == "outer":
        return np.sum(masks > 0, axis=(-2, -1)).astype(np.float64)
    if class_kind == "wall":
        return np.sum(masks == 1, axis=(-2, -1)).astype(np.float64)
    if class_kind == "lumen":
        return np.sum(masks == 2, axis=(-2, -1)).astype(np.float64)
    raise ValueError(class_kind)


def _relative_first_difference(values: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> float:
    """Mean adjacent relative change on real adjacent targets only."""
    a = np.asarray(values, dtype=np.float64)
    if len(a) < 2:
        return np.nan
    valid = np.ones(len(a), dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    pair_valid = valid[1:] & valid[:-1] & np.isfinite(a[1:]) & np.isfinite(a[:-1])
    if not pair_valid.any():
        return np.nan
    denom = 0.5 * (np.abs(a[1:]) + np.abs(a[:-1])) + 1.0
    diff = np.abs(a[1:] - a[:-1]) / denom
    return float(np.mean(diff[pair_valid]))


def _relative_second_difference(values: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> float:
    """Normalized second difference on real contiguous valid triples only."""
    a = np.asarray(values, dtype=np.float64)
    if len(a) < 3:
        return np.nan
    valid = np.ones(len(a), dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    triple_valid = (
        valid[2:] & valid[1:-1] & valid[:-2]
        & np.isfinite(a[2:]) & np.isfinite(a[1:-1]) & np.isfinite(a[:-2])
    )
    if not triple_valid.any():
        return np.nan
    d2 = np.abs(a[2:] - 2.0 * a[1:-1] + a[:-2])
    local_scale = (np.abs(a[2:]) + np.abs(a[1:-1]) + np.abs(a[:-2])) / 3.0 + 1.0
    return float(np.mean((d2 / local_scale)[triple_valid]))


def _centroid_sequence(masks: np.ndarray, class_kind: str) -> np.ndarray:
    pts = []
    for m in masks:
        if class_kind == "outer":
            binary = m > 0
        elif class_kind == "lumen":
            binary = m == 2
        else:
            raise ValueError(class_kind)

        yy, xx = np.nonzero(binary)
        if len(xx) == 0:
            pts.append((np.nan, np.nan))
        else:
            pts.append((float(xx.mean()), float(yy.mean())))
    return np.asarray(pts, dtype=np.float64)


def _centroid_first_difference(
    centroids: np.ndarray, valid_mask: Optional[np.ndarray] = None
) -> float:
    c = np.asarray(centroids, dtype=np.float64)
    if len(c) < 2:
        return np.nan
    target_valid = np.ones(len(c), dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    d = c[1:] - c[:-1]
    valid = target_valid[1:] & target_valid[:-1] & np.isfinite(d).all(axis=1)
    if not valid.any():
        return np.nan
    return float(np.mean(np.sqrt(np.sum(d[valid] ** 2, axis=1))))


def _centroid_second_difference(
    centroids: np.ndarray, valid_mask: Optional[np.ndarray] = None
) -> float:
    c = np.asarray(centroids, dtype=np.float64)
    if len(c) < 3:
        return np.nan
    target_valid = np.ones(len(c), dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    d2 = c[2:] - 2.0 * c[1:-1] + c[:-2]
    valid = (
        target_valid[2:] & target_valid[1:-1] & target_valid[:-2]
        & np.isfinite(d2).all(axis=1)
    )
    if not valid.any():
        return np.nan
    return float(np.mean(np.sqrt(np.sum(d2[valid] ** 2, axis=1))))


def _adjacent_sequence_dice(
    masks: np.ndarray, class_kind: str, valid_mask: Optional[np.ndarray] = None
) -> float:
    """Audit metric only. Higher is not automatically better in a spatial sweep."""
    valid = np.ones(len(masks), dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    vals = []
    for t in range(len(masks) - 1):
        if not (valid[t] and valid[t + 1]):
            continue
        if class_kind == "outer":
            a, b = masks[t] > 0, masks[t + 1] > 0
        elif class_kind == "wall":
            a, b = masks[t] == 1, masks[t + 1] == 1
        elif class_kind == "lumen":
            a, b = masks[t] == 2, masks[t + 1] == 2
        else:
            raise ValueError(class_kind)
        vals.append(dice_np(a, b))
    return _safe_nanmean(vals)


def temporal_sequence_metrics(
    masks: np.ndarray,
    topology_stats: List[Dict[str, float]],
    prefix: str = "",
    valid_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Metrics on a T-frame prediction sequence using only real/usable target frames.

    Primary interpretation:
      - first-difference: amount of frame-to-frame change;
      - second-difference: abrupt trajectory jitter (preferred stability metric);
      - centroid metrics: spatial trajectory stability;
      - raw topology failure rates: anatomical plausibility before cleanup.

    Padded/unreadable targets are excluded, and differences never bridge an invalid
    position. Empty-prediction rates are reported because an all-empty sequence can
    otherwise look artificially stable. Adjacent Dice remains an audit metric only.
    """
    out: Dict[str, float] = {}
    T = len(masks)
    valid = np.ones(T, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool).copy()
    if len(valid) != T:
        raise ValueError(f"valid_mask length {len(valid)} does not match sequence length {T}")

    out[f"{prefix}n_valid_frames"] = float(valid.sum())
    out[f"{prefix}valid_frame_fraction"] = float(valid.mean()) if T else np.nan

    for kind in ["outer", "wall", "lumen"]:
        areas = _mask_area_sequence(masks, kind)
        out[f"{prefix}{kind}_area_rel_step"] = _relative_first_difference(areas, valid)
        out[f"{prefix}{kind}_area_rel_jitter"] = _relative_second_difference(areas, valid)
        out[f"{prefix}{kind}_adjacent_dice"] = _adjacent_sequence_dice(masks, kind, valid)
        if valid.any():
            if kind == "outer":
                present = np.asarray([(m > 0).any() for m in masks], dtype=bool)
            elif kind == "wall":
                present = np.asarray([(m == 1).any() for m in masks], dtype=bool)
            else:
                present = np.asarray([(m == 2).any() for m in masks], dtype=bool)
            out[f"{prefix}{kind}_empty_prediction_rate"] = float(np.mean(~present[valid]))
        else:
            out[f"{prefix}{kind}_empty_prediction_rate"] = np.nan

    for kind in ["outer", "lumen"]:
        c = _centroid_sequence(masks, kind)
        out[f"{prefix}{kind}_centroid_step_px"] = _centroid_first_difference(c, valid)
        out[f"{prefix}{kind}_centroid_jitter_px"] = _centroid_second_difference(c, valid)

    valid_stats = [s for j, s in enumerate(topology_stats) if j < len(valid) and valid[j]]
    if valid_stats:
        out[f"{prefix}raw_outer_multi_component_rate"] = float(
            np.mean([float(s.get("raw_outer_components", 1) > 1) for s in valid_stats])
        )
        out[f"{prefix}raw_lumen_multi_component_rate"] = float(
            np.mean([float(s.get("raw_lumen_components", 1) > 1) for s in valid_stats])
        )
        out[f"{prefix}raw_lumen_outside_outer_rate"] = float(
            np.mean([
                float(s.get("raw_lumen_outside_outer_px", 0) > settings.QC_RAW_LUMEN_OUTSIDE_MAX_PX)
                for s in valid_stats
            ])
        )
        out[f"{prefix}topology_changed_fraction"] = float(
            np.mean([float(s.get("topology_changed_fraction", 0.0)) for s in valid_stats])
        )
    else:
        out[f"{prefix}raw_outer_multi_component_rate"] = np.nan
        out[f"{prefix}raw_lumen_multi_component_rate"] = np.nan
        out[f"{prefix}raw_lumen_outside_outer_rate"] = np.nan
        out[f"{prefix}topology_changed_fraction"] = np.nan

    out[f"{prefix}mean_area_rel_jitter"] = _safe_nanmean([
        out[f"{prefix}outer_area_rel_jitter"],
        out[f"{prefix}wall_area_rel_jitter"],
        out[f"{prefix}lumen_area_rel_jitter"],
    ])
    out[f"{prefix}mean_centroid_jitter_px"] = _safe_nanmean([
        out[f"{prefix}outer_centroid_jitter_px"],
        out[f"{prefix}lumen_centroid_jitter_px"],
    ])
    return out


def summarize_temporal_stability(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_names = [
        "outer_area_rel_step",
        "wall_area_rel_step",
        "lumen_area_rel_step",
        "outer_area_rel_jitter",
        "wall_area_rel_jitter",
        "lumen_area_rel_jitter",
        "outer_centroid_step_px",
        "lumen_centroid_step_px",
        "outer_centroid_jitter_px",
        "lumen_centroid_jitter_px",
        "raw_outer_multi_component_rate",
        "raw_lumen_multi_component_rate",
        "raw_lumen_outside_outer_rate",
        "topology_changed_fraction",
        "outer_adjacent_dice",
        "wall_adjacent_dice",
        "lumen_adjacent_dice",
        "mean_area_rel_jitter",
        "mean_centroid_jitter_px",
        "n_valid_frames",
        "valid_frame_fraction",
        "outer_empty_prediction_rate",
        "wall_empty_prediction_rate",
        "lumen_empty_prediction_rate",
    ]

    for method in ["image", "video"]:
        row = {"method": method, "n": int(len(df))}
        for metric in metric_names:
            col = f"{method}_{metric}"
            if col in df.columns:
                x = pd.to_numeric(df[col], errors="coerce")
                row[metric] = float(x.mean())
                row[f"{metric}_std"] = float(x.std())
                row[f"{metric}_median"] = float(x.median())
        rows.append(row)
    return pd.DataFrame(rows)


@torch.no_grad()
def evaluate_temporal_stability(
    video_model: nn.Module,
    image_model: nn.Module,
    loader: DataLoader,
    device,
    use_amp: bool,
    phase: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compare image-only and video-model stability on the SAME local 5-frame clips.

    No extra manual labels are required. The frozen image model segments every frame
    independently. The trained video model predicts every frame by re-centering the
    available 5-frame context window on that target frame.
    """
    image_model.eval()
    video_model.eval()
    rows = []

    for batch in progress(loader, phase, total=len(loader), unit="batch"):
        clip = batch["clip"].to(device, non_blocking=True)
        weights = batch["context_weights"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            image_seq_out = image_model_all_frames(image_model, clip)
            # Same DataParallel safeguard as the qualitative preview. The last
            # validation/test batch can contain a single sample, so sequence-mode
            # inference is run on the underlying primary-GPU module.
            video_seq_out = base_model(video_model)(
                clip,
                weights,
                return_sequence=True,
            )

        image_masks, image_stats = clean_sequence_from_logits(image_seq_out)
        video_masks, video_stats = clean_sequence_from_logits(video_seq_out)

        frame_indices = batch["frame_indices"].cpu().numpy()
        weights_np = weights.detach().cpu().numpy()

        for i in range(len(image_masks)):
            valid_target_mask = weights_np[i] > 0
            valid_target_mask[settings.CENTER_INDEX] = True
            rec = {
                "sample_name": batch["sample_name"][i],
                "patient_id": batch["patient_id"][i],
                "video_path": batch["video_path"][i],
                "center_frame_idx": int(batch["center_frame_idx"][i]),
                "frame_indices": "|".join(str(int(x)) for x in frame_indices[i]),
                "valid_target_mask": "|".join("1" if x else "0" for x in valid_target_mask),
            }
            rec.update(temporal_sequence_metrics(
                image_masks[i], image_stats[i], prefix="image_", valid_mask=valid_target_mask
            ))
            rec.update(temporal_sequence_metrics(
                video_masks[i], video_stats[i], prefix="video_", valid_mask=valid_target_mask
            ))

            # Negative delta = less jitter / fewer failures with the video model.
            for metric in [
                "outer_area_rel_jitter",
                "wall_area_rel_jitter",
                "lumen_area_rel_jitter",
                "outer_centroid_jitter_px",
                "lumen_centroid_jitter_px",
                "raw_outer_multi_component_rate",
                "raw_lumen_multi_component_rate",
                "raw_lumen_outside_outer_rate",
                "mean_area_rel_jitter",
                "mean_centroid_jitter_px",
            ]:
                rec[f"delta_{metric}_video_minus_image"] = (
                    rec.get(f"video_{metric}", np.nan) - rec.get(f"image_{metric}", np.nan)
                )
            rows.append(rec)

    metrics_df = pd.DataFrame(rows)
    summary_df = summarize_temporal_stability(metrics_df)
    return metrics_df, summary_df

