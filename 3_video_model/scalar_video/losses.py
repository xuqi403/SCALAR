"""Losses operations for SCALAR V4/V5 video segmentation."""
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


def soft_dice(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dims = tuple(range(1, prob.ndim))
    inter = (prob * target).sum(dims)
    den = prob.sum(dims) + target.sum(dims)
    return 1 - ((2 * inter + eps) / (den + eps)).mean()


def boundary_map(x: torch.Tensor) -> torch.Tensor:
    dil = F.max_pool2d(x, 3, 1, 1)
    ero = -F.max_pool2d(-x, 3, 1, 1)
    return torch.clamp(dil - ero, 0, 1)


def _dilate_binary_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Binary dilation with max-pooling; mask shape is [B,1,H,W]."""
    radius = int(max(radius, 0))
    if radius == 0:
        return mask
    k = 2 * radius + 1
    return F.max_pool2d(mask, kernel_size=k, stride=1, padding=radius)


def outer_protrusion_penalty_per_sample(
    p_outer: torch.Tensor,
    gt_outer: torch.Tensor,
) -> torch.Tensor:
    """Weak GT-tolerant penalty for confident OUTER predictions far outside GT.

    This is not a circularity/ellipse/convexity prior. It only asks the predicted
    outer mask not to send thin confident extensions far away from the annotated
    vessel boundary.

    Approximate distance bands:
      <= ALLOW px            : no penalty
      ALLOW..MID px          : weak
      MID..FAR px            : moderate
      > FAR px               : strongest
    """
    # Use fp32 here because the large pooling kernels are more stable in fp32.
    p = p_outer.float().clamp(0.0, 1.0)
    g = gt_outer.float().clamp(0.0, 1.0)

    allow = _dilate_binary_mask(g, settings.OUTER_SPUR_ALLOW_PX)
    mid = _dilate_binary_mask(g, settings.OUTER_SPUR_MID_PX)
    far = _dilate_binary_mask(g, settings.OUTER_SPUR_FAR_PX)

    band_weak = torch.clamp(mid - allow, 0.0, 1.0)
    band_mid = torch.clamp(far - mid, 0.0, 1.0)
    band_far = torch.clamp(1.0 - far, 0.0, 1.0)
    distance_weight = 0.25 * band_weak + 0.50 * band_mid + 1.00 * band_far

    # Squaring p focuses the penalty on confident protrusions and largely ignores
    # the low-probability background.
    p_focus = p.pow(float(settings.OUTER_SPUR_PROB_POWER))
    dims = tuple(range(1, p_focus.ndim))
    numerator = (p_focus * distance_weight).sum(dims)
    denominator = p_focus.sum(dims).clamp_min(1e-6)
    return numerator / denominator


def outer_local_curvature_penalty_per_sample(
    p_outer: torch.Tensor,
) -> torch.Tensor:
    """Weak local regularity loss for the predicted OUTER contour.

    The loss measures rapid changes in the normal direction of the soft outer
    boundary. It is intentionally prediction-only: manual GT is not used here.

    Important properties:
      1) very light 3x3 smoothing suppresses pixel-level probability noise only;
      2) a dead zone allows ordinary slow anatomical curvature without penalty;
      3) weighting is concentrated near a confident probability transition;
      4) no circularity, ellipticity, convexity, or fixed wall-thickness prior.

    Returns one value per sample.
    """
    p = p_outer.float().clamp(0.0, 1.0)

    k = int(settings.OUTER_CURV_SMOOTH_KERNEL)
    if k > 1:
        if k % 2 == 0:
            raise ValueError("OUTER_CURV_SMOOTH_KERNEL must be odd.")
        p = F.avg_pool2d(p, kernel_size=k, stride=1, padding=k // 2)

    # Central finite differences, with replicated padding back to H x W.
    gx = 0.5 * (p[:, :, :, 2:] - p[:, :, :, :-2])
    gx = F.pad(gx, (1, 1, 0, 0), mode="replicate")

    gy = 0.5 * (p[:, :, 2:, :] - p[:, :, :-2, :])
    gy = F.pad(gy, (0, 0, 1, 1), mode="replicate")

    mag = torch.sqrt(gx * gx + gy * gy + 1e-8)
    nx = gx / (mag + 1e-4)
    ny = gy / (mag + 1e-4)

    # Concentrate regularization near the soft contour. The gradient gate prevents
    # uncertain flat p~0.5 regions from receiving a large curvature weight.
    boundary_prob = 4.0 * p * (1.0 - p)
    grad_gate = mag / (mag + float(settings.OUTER_CURV_GRAD_SCALE))
    boundary_weight = (boundary_prob * grad_gate).detach()

    # Change in unit normal between neighboring pixels. A smooth curved vessel
    # changes normal gradually; a spike/foot/corner changes it abruptly.
    dnx_x = nx[:, :, :, 1:] - nx[:, :, :, :-1]
    dny_x = ny[:, :, :, 1:] - ny[:, :, :, :-1]
    bend_x = torch.sqrt(dnx_x * dnx_x + dny_x * dny_x + 1e-8)

    dnx_y = nx[:, :, 1:, :] - nx[:, :, :-1, :]
    dny_y = ny[:, :, 1:, :] - ny[:, :, :-1, :]
    bend_y = torch.sqrt(dnx_y * dnx_y + dny_y * dny_y + 1e-8)

    wx = 0.5 * (boundary_weight[:, :, :, 1:] + boundary_weight[:, :, :, :-1])
    wy = 0.5 * (boundary_weight[:, :, 1:, :] + boundary_weight[:, :, :-1, :])

    tau = float(settings.OUTER_CURV_DEADZONE)
    excess_x = F.relu(bend_x - tau).pow(2)
    excess_y = F.relu(bend_y - tau).pow(2)

    dims = tuple(range(1, excess_x.ndim))
    loss_x = (wx * excess_x).sum(dims) / wx.sum(dims).clamp_min(1e-6)

    dims_y = tuple(range(1, excess_y.ndim))
    loss_y = (wy * excess_y).sum(dims_y) / wy.sum(dims_y).clamp_min(1e-6)

    return 0.5 * (loss_x + loss_y)


class DualAnatomyLoss(nn.Module):
    def __init__(
        self,
        enable_outer_spur: bool = False,
        enable_outer_curvature: bool = False,
    ):
        super().__init__()
        self.enable_outer_spur = bool(enable_outer_spur)
        self.enable_outer_curvature = bool(enable_outer_curvature)

    def forward(
        self,
        out: Dict[str, torch.Tensor],
        target: torch.Tensor,
        return_parts: bool = False,
    ):
        outer_logit = out["outer_logits"]
        lumen_logit = out["lumen_logits"]
        gt_outer = (target > 0).float().unsqueeze(1)
        gt_lumen = (target == 2).float().unsqueeze(1)
        gt_wall = (target == 1).float().unsqueeze(1)
        p_outer = torch.sigmoid(outer_logit)
        p_lumen = torch.sigmoid(lumen_logit)
        p_wall = p_outer * (1.0 - p_lumen)

        outer_loss = F.binary_cross_entropy_with_logits(outer_logit, gt_outer) + soft_dice(p_outer, gt_outer)
        lumen_loss = F.binary_cross_entropy_with_logits(lumen_logit, gt_lumen) + soft_dice(p_lumen, gt_lumen)

        # ORIGINAL anatomy objective: unchanged from baseline V5.
        base_loss = 0.5 * (outer_loss + lumen_loss)
        base_loss += settings.LAMBDA_WALL * soft_dice(p_wall, gt_wall)
        base_loss += settings.LAMBDA_CONTAINMENT * torch.mean(p_lumen * (1.0 - p_outer))
        base_loss += settings.LAMBDA_BOUNDARY * 0.5 * (
            soft_dice(boundary_map(p_outer), boundary_map(gt_outer))
            + soft_dice(boundary_map(p_lumen), boundary_map(gt_lumen))
        )

        if self.enable_outer_spur:
            spur_per_sample = outer_protrusion_penalty_per_sample(p_outer, gt_outer)
            spur_loss = spur_per_sample.mean()
        else:
            spur_per_sample = p_outer.new_zeros((p_outer.shape[0],), dtype=torch.float32)
            spur_loss = p_outer.new_tensor(0.0, dtype=torch.float32)

        if self.enable_outer_curvature:
            curvature_per_sample = outer_local_curvature_penalty_per_sample(p_outer)
            curvature_loss = curvature_per_sample.mean()
        else:
            curvature_per_sample = p_outer.new_zeros((p_outer.shape[0],), dtype=torch.float32)
            curvature_loss = p_outer.new_tensor(0.0, dtype=torch.float32)

        # Regularizers are intentionally weak and complementary:
        #   spur      -> prevents confident long outward extensions;
        #   curvature -> suppresses local spikes/corners/jagged contour changes.
        total_loss = (
            base_loss
            + float(settings.LAMBDA_OUTER_SPUR) * spur_loss
            + float(settings.LAMBDA_OUTER_CURV) * curvature_loss
        )

        if return_parts:
            return total_loss, {
                "base_anatomy": base_loss,
                "outer_spur": spur_loss,
                "outer_spur_per_sample": spur_per_sample,
                "outer_curvature": curvature_loss,
                "outer_curvature_per_sample": curvature_per_sample,
            }
        return total_loss


def soft_mask_centroid(prob: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # prob [B,T,H,W]
    B, T, H, W = prob.shape
    yy = torch.linspace(0, 1, H, device=prob.device).view(1, 1, H, 1)
    xx = torch.linspace(0, 1, W, device=prob.device).view(1, 1, 1, W)
    mass = prob.sum(dim=(-2, -1)).clamp_min(1e-6)
    cx = (prob * xx).sum(dim=(-2, -1)) / mass
    cy = (prob * yy).sum(dim=(-2, -1)) / mass
    return cx, cy


def temporal_structure_loss(
    out: Dict[str, torch.Tensor],
    target: torch.Tensor,
    context_weights: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if "aux_outer_logits" not in out:
        zero = out["outer_logits"].new_tensor(0.0)
        return zero, {"aux_center": 0.0, "temporal_area": 0.0, "temporal_centroid": 0.0}

    aux_o = out["aux_outer_logits"][:, :, 0]  # [B,T,h,w]
    aux_l = out["aux_lumen_logits"][:, :, 0]
    p_o = torch.sigmoid(aux_o)
    p_l = torch.sigmoid(aux_l)
    h, w = p_o.shape[-2:]

    gt_o = F.interpolate((target > 0).float().unsqueeze(1), (h, w), mode="nearest")[:, 0]
    gt_l = F.interpolate((target == 2).float().unsqueeze(1), (h, w), mode="nearest")[:, 0]
    center_o = p_o[:, settings.CENTER_INDEX]
    center_l = p_l[:, settings.CENTER_INDEX]

    # AMP-safe auxiliary center supervision:
    # BCE is computed from logits; Dice is computed from sigmoid probabilities.
    center_o_logit = aux_o[:, settings.CENTER_INDEX]
    center_l_logit = aux_l[:, settings.CENTER_INDEX]

    aux_center = 0.5 * (
        F.binary_cross_entropy_with_logits(center_o_logit, gt_o) + soft_dice(center_o, gt_o)
        + F.binary_cross_entropy_with_logits(center_l_logit, gt_l) + soft_dice(center_l, gt_l)
    )

    # Anatomical continuity is gated by both context reliability and DINO semantic
    # similarity. We regularize only low-dimensional structure (area/centroid), not
    # pixel-wise equality, so true longitudinal sweep changes remain possible.
    feats = out["frame_features"]
    pooled = feats.mean(dim=(-2, -1))
    pooled = F.normalize(pooled, dim=-1)
    sim = (pooled[:, 1:] * pooled[:, :-1]).sum(dim=-1).detach().clamp(0, 1)
    pair_w = (
        context_weights[:, 1:].clamp(0, 1)
        * context_weights[:, :-1].clamp(0, 1)
        * sim
    )
    denom = pair_w.sum().clamp_min(1e-6)

    def area_loss(p):
        a = p.mean(dim=(-2, -1))
        rel = (a[:, 1:] - a[:, :-1]).abs() / ((a[:, 1:] + a[:, :-1]) * 0.5 + 1e-3)
        return (rel * pair_w).sum() / denom

    area = 0.5 * (area_loss(p_o) + area_loss(p_l))

    ocx, ocy = soft_mask_centroid(p_o)
    lcx, lcy = soft_mask_centroid(p_l)
    od = torch.sqrt((ocx[:, 1:] - ocx[:, :-1]) ** 2 + (ocy[:, 1:] - ocy[:, :-1]) ** 2 + 1e-8)
    ld = torch.sqrt((lcx[:, 1:] - lcx[:, :-1]) ** 2 + (lcy[:, 1:] - lcy[:, :-1]) ** 2 + 1e-8)
    centroid = (((od + ld) * 0.5) * pair_w).sum() / denom

    total = (
        settings.LAMBDA_AUX_CENTER * aux_center
        + settings.LAMBDA_TEMPORAL_AREA * area
        + settings.LAMBDA_TEMPORAL_CENTROID * centroid
    )
    return total, {
        "aux_center": float(aux_center.detach().cpu()),
        "temporal_area": float(area.detach().cpu()),
        "temporal_centroid": float(centroid.detach().cpu()),
    }


def _lowres_dual_loss_per_sample(
    outer_logit: torch.Tensor,
    lumen_logit: torch.Tensor,
    gt_outer: torch.Tensor,
    gt_lumen: torch.Tensor,
    gt_wall: torch.Tensor,
) -> torch.Tensor:
    """Lightweight low-resolution auxiliary loss, returned per sample."""
    po = torch.sigmoid(outer_logit)
    pl = torch.sigmoid(lumen_logit)
    pw = po * (1.0 - pl)

    bce_o = F.binary_cross_entropy_with_logits(outer_logit, gt_outer, reduction="none").mean(dim=(-2, -1))
    bce_l = F.binary_cross_entropy_with_logits(lumen_logit, gt_lumen, reduction="none").mean(dim=(-2, -1))

    def dice_per_sample(prob, gt):
        inter = (prob * gt).sum(dim=(-2, -1))
        den = prob.sum(dim=(-2, -1)) + gt.sum(dim=(-2, -1))
        return 1.0 - (2.0 * inter + 1e-6) / (den + 1e-6)

    return 0.5 * (
        bce_o + dice_per_sample(po, gt_outer)
        + bce_l + dice_per_sample(pl, gt_lumen)
    ) + 0.5 * dice_per_sample(pw, gt_wall)


def adaptive_rescue_loss(
    out: Dict[str, torch.Tensor],
    target: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """V5 auxiliary objective: center skill + context skill + learned rescue gate."""
    required = {
        "adaptive_center_outer_logits", "adaptive_center_lumen_logits",
        "adaptive_context_outer_logits", "adaptive_context_lumen_logits",
        "gate_logits", "has_context",
    }
    if not required.issubset(out):
        zero = out["outer_logits"].new_tensor(0.0)
        return zero, {"center_aux": 0.0, "context_aux": 0.0, "gate": 0.0, "context_use": 0.0}

    co = out["adaptive_center_outer_logits"][:, 0]
    cl = out["adaptive_center_lumen_logits"][:, 0]
    xo = out["adaptive_context_outer_logits"][:, 0]
    xl = out["adaptive_context_lumen_logits"][:, 0]
    h, w = co.shape[-2:]

    gt_outer = F.interpolate((target > 0).float().unsqueeze(1), (h, w), mode="nearest")[:, 0]
    gt_lumen = F.interpolate((target == 2).float().unsqueeze(1), (h, w), mode="nearest")[:, 0]
    gt_wall = (gt_outer - gt_lumen).clamp(0, 1)
    has_context = out["has_context"].float()

    center_per = _lowres_dual_loss_per_sample(co, cl, gt_outer, gt_lumen, gt_wall)
    context_per = _lowres_dual_loss_per_sample(xo, xl, gt_outer, gt_lumen, gt_wall)
    center_aux = center_per.mean()
    context_aux = (context_per * has_context).sum() / has_context.sum().clamp_min(1.0)

    # Counterfactual soft oracle: whichever branch is locally closer to GT should
    # receive more weight. BCE-from-logits is AMP safe. The target is detached so
    # branches cannot game the gate objective by changing their error maps.
    center_err = 0.5 * (
        F.binary_cross_entropy_with_logits(co, gt_outer, reduction="none")
        + F.binary_cross_entropy_with_logits(cl, gt_lumen, reduction="none")
    )
    context_err = 0.5 * (
        F.binary_cross_entropy_with_logits(xo, gt_outer, reduction="none")
        + F.binary_cross_entropy_with_logits(xl, gt_lumen, reduction="none")
    )
    gate_target = torch.sigmoid((context_err - center_err) / settings.V5_GATE_TEMPERATURE).detach()

    # Gate supervision is concentrated on the vessel and its immediate boundary,
    # rather than letting the large ultrasound background dominate.
    region = F.max_pool2d(gt_outer.unsqueeze(1), 3, 1, 1)[:, 0]
    gate_mask = region * has_context[:, None, None]
    gate_map = F.binary_cross_entropy_with_logits(
        out["gate_logits"][:, 0], gate_target, reduction="none"
    )
    gate_loss = (gate_map * gate_mask).sum() / gate_mask.sum().clamp_min(1.0)

    total = (
        settings.LAMBDA_V5_CENTER_AUX * center_aux
        + settings.LAMBDA_V5_CONTEXT_AUX * context_aux
        + settings.LAMBDA_V5_GATE * gate_loss
    )
    gate = out["gate"].detach()
    context_use = float((1.0 - gate).mean().cpu())
    return total, {
        "center_aux": float(center_aux.detach().cpu()),
        "context_aux": float(context_aux.detach().cpu()),
        "gate": float(gate_loss.detach().cpu()),
        "context_use": context_use,
    }

