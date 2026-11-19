"""Trainer operations for SCALAR V4/V5 video segmentation."""
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
from .dataset import VideoClipDataset
from .evaluation import clean_sequence_from_logits
from .losses import adaptive_rescue_loss, temporal_structure_loss
from .metrics import one_metrics, one_raw_head_metrics, summarize_metrics, torch_dice
from .model import apply_video_train_mode, set_video_trainability
from .postprocess import decode_dual_logits
from .utils import base_model, denormalize_tensor, progress


def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    m = base_model(model)
    backbone_params = list(m.backbone.parameters())
    temporal_params = list(m.temporal.parameters()) + list(m.aux_outer_head.parameters()) + list(m.aux_lumen_head.parameters())
    temporal_ids = {id(p) for p in temporal_params}
    backbone_ids = {id(p) for p in backbone_params}
    decoder_params = [p for p in m.parameters() if id(p) not in backbone_ids and id(p) not in temporal_ids]
    return torch.optim.AdamW([
        {"params": backbone_params, "lr": settings.LR_BACKBONE},
        {"params": temporal_params, "lr": settings.LR_TEMPORAL},
        {"params": decoder_params, "lr": settings.LR_DECODER},
    ], weight_decay=settings.WEIGHT_DECAY)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    anatomy_loss,
    scaler,
    device,
    use_amp: bool,
    cfg: Dict[str, Any],
    epoch: int,
) -> Dict[str, float]:
    stage = set_video_trainability(model, cfg, epoch)
    apply_video_train_mode(model, cfg, epoch)

    total_losses = []
    anatomy_losses = []
    regularized_anatomy_losses = []
    outer_spur_losses = []
    outer_curvature_losses = []
    temporal_losses = []
    adaptive_losses = []
    adaptive_gate_losses = []
    adaptive_context_losses = []
    adaptive_context_use = []
    bar = progress(
        loader,
        f"Train {epoch:02d}/{settings.EPOCHS} [{stage}]",
        total=len(loader),
        unit="batch",
    )

    for batch in bar:
        clip = batch["clip"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)
        weights = batch["context_weights"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            out = model(clip, weights)
            loss_anatomy_total, anatomy_parts = anatomy_loss(out, target, return_parts=True)
            loss_anatomy_base = anatomy_parts["base_anatomy"]
            loss_outer_spur = anatomy_parts["outer_spur"]
            loss_outer_curvature = anatomy_parts["outer_curvature"]
            loss_temporal, temp_parts = temporal_structure_loss(out, target, weights) if cfg["temporal_loss"] else (
                loss_anatomy_total.new_tensor(0.0), {"aux_center": 0.0, "temporal_area": 0.0, "temporal_centroid": 0.0}
            )
            loss_adaptive, adaptive_parts = adaptive_rescue_loss(out, target) if cfg.get("adaptive_rescue", False) else (
                loss_anatomy_total.new_tensor(0.0), {"center_aux": 0.0, "context_aux": 0.0, "gate": 0.0, "context_use": 0.0}
            )
            loss = loss_anatomy_total + loss_temporal + loss_adaptive

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        total_losses.append(float(loss.detach().cpu()))
        anatomy_losses.append(float(loss_anatomy_base.detach().cpu()))
        regularized_anatomy_losses.append(float(loss_anatomy_total.detach().cpu()))
        outer_spur_losses.append(float(loss_outer_spur.detach().cpu()))
        outer_curvature_losses.append(float(loss_outer_curvature.detach().cpu()))
        temporal_losses.append(float(loss_temporal.detach().cpu()))
        adaptive_losses.append(float(loss_adaptive.detach().cpu()))
        adaptive_gate_losses.append(float(adaptive_parts.get("gate", 0.0)))
        adaptive_context_losses.append(float(adaptive_parts.get("context_aux", 0.0)))
        adaptive_context_use.append(float(adaptive_parts.get("context_use", 0.0)))

        if tqdm is not None:
            with torch.no_grad():
                _, pred, _, _, _ = decode_dual_logits(out)
                postfix = dict(
                    loss=f"{np.mean(total_losses[-30:]):.4f}",
                    O=f"{torch_dice(pred > 0, target > 0):.3f}",
                    W=f"{torch_dice(pred == 1, target == 1):.3f}",
                    L=f"{torch_dice(pred == 2, target == 2):.3f}",
                )
                if cfg.get("adaptive_rescue", False):
                    postfix["adapt"] = f"{np.mean(adaptive_losses[-30:]):.3f}"
                    postfix["ctx"] = f"{np.mean(adaptive_context_use[-30:]):.2f}"
                    if cfg.get("outer_spur_loss", False):
                        postfix["spur"] = f"{np.mean(outer_spur_losses[-30:]):.4f}"
                    if cfg.get("outer_curvature_loss", False):
                        postfix["curv"] = f"{np.mean(outer_curvature_losses[-30:]):.4f}"
                else:
                    postfix["temp"] = f"{np.mean(temporal_losses[-30:]):.3f}"
                bar.set_postfix(**postfix)

    return {
        "train_loss": float(np.mean(total_losses)),
        # Baseline anatomy objective, kept comparable with original V5.
        "train_anatomy_loss": float(np.mean(anatomy_losses)),
        "train_regularized_anatomy_loss": float(np.mean(regularized_anatomy_losses)),
        "train_outer_spur_loss": float(np.mean(outer_spur_losses)),
        "train_outer_curvature_loss": float(np.mean(outer_curvature_losses)),
        "train_outer_spur_weighted": float(settings.LAMBDA_OUTER_SPUR * np.mean(outer_spur_losses)),
        "train_outer_curvature_weighted": float(settings.LAMBDA_OUTER_CURV * np.mean(outer_curvature_losses)),
        "train_temporal_loss": float(np.mean(temporal_losses)),
        "train_adaptive_loss": float(np.mean(adaptive_losses)),
        "train_gate_loss": float(np.mean(adaptive_gate_losses)),
        "train_context_aux_loss": float(np.mean(adaptive_context_losses)),
        "train_context_use": float(np.mean(adaptive_context_use)),
        "stage": stage,
    }


@torch.no_grad()
def evaluate_video(
    model: nn.Module,
    loader: DataLoader,
    anatomy_loss,
    device,
    use_amp: bool,
    phase: str,
    cfg: Dict[str, Any],
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    model.eval()
    records = []
    losses = []
    anatomy_losses = []
    regularized_anatomy_losses = []
    outer_spur_losses = []
    outer_curvature_losses = []
    temporal_losses = []
    adaptive_losses = []
    adaptive_gate_losses = []
    adaptive_context_losses = []

    for batch in progress(loader, phase, total=len(loader), unit="batch"):
        clip = batch["clip"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)
        weights = batch["context_weights"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            out = model(clip, weights)
            la_total, anatomy_parts = anatomy_loss(out, target, return_parts=True)
            la_base = anatomy_parts["base_anatomy"]
            lspur = anatomy_parts["outer_spur"]
            spur_per_sample = anatomy_parts["outer_spur_per_sample"]
            lcurv = anatomy_parts["outer_curvature"]
            curvature_per_sample = anatomy_parts["outer_curvature_per_sample"]
            lt, _ = temporal_structure_loss(out, target, weights) if cfg["temporal_loss"] else (la_total.new_tensor(0.0), {})
            lad, adaptive_parts = adaptive_rescue_loss(out, target) if cfg.get("adaptive_rescue", False) else (
                la_total.new_tensor(0.0), {"center_aux": 0.0, "context_aux": 0.0, "gate": 0.0, "context_use": 0.0}
            )
            loss = la_total + lt + lad
        losses.append(float(loss.cpu()))
        anatomy_losses.append(float(la_base.cpu()))
        regularized_anatomy_losses.append(float(la_total.cpu()))
        outer_spur_losses.append(float(lspur.cpu()))
        outer_curvature_losses.append(float(lcurv.cpu()))
        temporal_losses.append(float(lt.cpu()))
        adaptive_losses.append(float(lad.cpu()))
        adaptive_gate_losses.append(float(adaptive_parts.get("gate", 0.0)))
        adaptive_context_losses.append(float(adaptive_parts.get("context_aux", 0.0)))

        raw, post, top_stats, p_outer, p_lumen = decode_dual_logits(out)
        raw = raw.cpu().numpy()
        post = post.cpu().numpy()
        raw_outer = (p_outer >= settings.PRED_THRESHOLD).detach().cpu().numpy()
        raw_lumen = (p_lumen >= settings.PRED_THRESHOLD).detach().cpu().numpy()
        gt = target.cpu().numpy()
        for i in range(len(gt)):
            rec = one_metrics(post[i], gt[i])
            rec.update(one_raw_head_metrics(raw_outer[i], raw_lumen[i], gt[i], prefix="raw_"))
            rec.update(top_stats[i])
            rec.update({
                "sample_name": batch["sample_name"][i],
                "patient_id": batch["patient_id"][i],
                "video_path": batch["video_path"][i],
                "frame_idx": int(batch["center_frame_idx"][i]),
                "mean_context_weight": float(weights[i].mean().cpu()),
                "min_context_weight": float(weights[i].min().cpu()),
                "outer_spur_penalty": float(spur_per_sample[i].detach().cpu()),
                "outer_curvature_penalty": float(curvature_per_sample[i].detach().cpu()),
            })
            if "gate" in out:
                gate_i = out["gate"][i, 0].detach()
                rec["gate_mean"] = float(gate_i.mean().cpu())
                rec["context_use_mean"] = float((1.0 - gate_i).mean().cpu())
                alpha_i = out["neighbor_alpha"][i].detach()
                for t, off in enumerate(settings.CLIP_OFFSETS):
                    rec[f"alpha_{off:+d}"] = float(alpha_i[t].mean().cpu())
                delta_i = out["alignment_offsets"][i].detach()
                mag = torch.sqrt((delta_i ** 2).sum(dim=1) + 1e-8)
                rec["alignment_offset_feature_px_mean"] = float(mag.mean().cpu())
                rec["alignment_offset_feature_px_p95"] = float(torch.quantile(mag.flatten().float(), 0.95).cpu())
            records.append(rec)

    summary = summarize_metrics(records)
    summary["loss"] = float(np.mean(losses))
    # ORIGINAL fused anatomy objective; this remains the primary checkpoint
    # criterion so baseline V5 and V5+spur use the same validation selector.
    summary["anatomy_loss"] = float(np.mean(anatomy_losses))
    summary["regularized_anatomy_loss"] = float(np.mean(regularized_anatomy_losses))
    summary["outer_spur_loss"] = float(np.mean(outer_spur_losses))
    summary["outer_curvature_loss"] = float(np.mean(outer_curvature_losses))
    summary["outer_spur_weighted"] = float(settings.LAMBDA_OUTER_SPUR * np.mean(outer_spur_losses))
    summary["outer_curvature_weighted"] = float(settings.LAMBDA_OUTER_CURV * np.mean(outer_curvature_losses))
    summary["temporal_loss"] = float(np.mean(temporal_losses))
    summary["adaptive_loss"] = float(np.mean(adaptive_losses))
    summary["gate_loss"] = float(np.mean(adaptive_gate_losses))
    summary["context_aux_loss"] = float(np.mean(adaptive_context_losses))
    return summary, records


@torch.no_grad()
def save_temporal_epoch_preview(
    model: nn.Module,
    dataset: VideoClipDataset,
    indices: List[int],
    device,
    use_amp: bool,
    plot_module,
    out_path: Path,
    epoch: int,
) -> None:
    """Save high-resolution qualitative previews for ALL frames in each local clip.

    Training supervision is still only on the annotated center frame. For monitoring,
    however, the trained video model is re-centered on every frame in the 5-frame
    clip so we can visually inspect whether Outer/Wall/Lumen predictions change
    smoothly or show abrupt frame-to-frame jumps.

    The center frame additionally shows the manual GT and O/W/L Dice.
    This is evaluation/visualization only and does not change training.
    """
    model.eval()
    cases = []

    for idx in indices:
        item = dataset[int(idx)]
        clip = item["clip"].unsqueeze(0).to(device)
        weights = item["context_weights"].unsqueeze(0).to(device)

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp,
        ):
            # `model` may be nn.DataParallel. This preview uses one case at a time
            # (batch size = 1). Passing a non-tensor keyword such as
            # return_sequence=True through DataParallel can create a kwargs-only
            # replica on the second GPU, causing forward() to receive no clip/weights.
            # Sequence preview is evaluation-only, so run the underlying module on
            # the primary GPU directly.
            seq_out = base_model(model)(
                clip,
                weights,
                return_sequence=True,
            )

        seq_masks, _ = clean_sequence_from_logits(seq_out)
        pred_sequence = seq_masks[0]
        gt = item["mask"].numpy()
        center_pred = pred_sequence[settings.CENTER_INDEX]

        center_metrics = one_metrics(
            center_pred,
            gt,
            prefix="",
        )

        cases.append({
            "clip_rgb": [
                denormalize_tensor(item["clip"][t])
                for t in range(item["clip"].shape[0])
            ],
            "gt": gt,
            "video_pred_sequence": pred_sequence,
            # Retain the old center prediction key for backward compatibility.
            "video_pred": center_pred,
            "center_metrics": {
                "outer_dice": float(center_metrics["outer_dice"]),
                "wall_dice": float(center_metrics["wall_dice"]),
                "lumen_dice": float(center_metrics["lumen_dice"]),
            },
            "weights": item["context_weights"].numpy(),
            "frame_indices": item["frame_indices"].numpy(),
            "center_frame_idx": int(item["center_frame_idx"]),
            "name": item["sample_name"],
            "epoch": epoch,
            "adaptive_mode": bool(getattr(base_model(model), "adaptive_rescue", False)),
        })

    if cases:
        plot_module.save_temporal_epoch_grid(
            cases,
            out_path,
        )

