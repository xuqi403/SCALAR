"""Metrics operations for SCALAR V4/V5 video segmentation."""
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


def dice_np(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not pred.any() and not gt.any():
        return 1.0
    return float((2 * np.logical_and(pred, gt).sum() + eps) / (pred.sum() + gt.sum() + eps))


def iou_np(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not pred.any() and not gt.any():
        return 1.0
    return float((np.logical_and(pred, gt).sum() + eps) / (np.logical_or(pred, gt).sum() + eps))


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    if not SCIPY_AVAILABLE:
        return np.nan
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not pred.any() or not gt.any():
        return np.nan
    ps = pred ^ binary_erosion(pred)
    gs = gt ^ binary_erosion(gt)
    if not ps.any() or not gs.any():
        return np.nan
    d1 = distance_transform_edt(~gs)[ps]
    d2 = distance_transform_edt(~ps)[gs]
    return float(np.percentile(np.concatenate([d1, d2]), 95))


def area_error(pred: np.ndarray, gt: np.ndarray) -> float:
    g = gt.astype(bool).sum()
    return np.nan if g == 0 else float(abs(pred.astype(bool).sum() - g) / g)


def _binary_anatomy_metrics(
    po: np.ndarray,
    pw: np.ndarray,
    pl: np.ndarray,
    gt: np.ndarray,
    prefix: str = "",
) -> Dict[str, float]:
    po, pw, pl = po.astype(bool), pw.astype(bool), pl.astype(bool)
    go, gw, gl = gt > 0, gt == 1, gt == 2
    outer_hd95_ok = bool(po.any() and go.any())
    lumen_hd95_ok = bool(pl.any() and gl.any())
    return {
        f"{prefix}outer_dice": dice_np(po, go),
        f"{prefix}wall_dice": dice_np(pw, gw),
        f"{prefix}lumen_dice": dice_np(pl, gl),
        f"{prefix}outer_iou": iou_np(po, go),
        f"{prefix}wall_iou": iou_np(pw, gw),
        f"{prefix}lumen_iou": iou_np(pl, gl),
        f"{prefix}outer_hd95_px": hd95(po, go),
        f"{prefix}lumen_hd95_px": hd95(pl, gl),
        f"{prefix}outer_hd95_evaluable": float(outer_hd95_ok),
        f"{prefix}lumen_hd95_evaluable": float(lumen_hd95_ok),
        f"{prefix}outer_empty_prediction": float(not po.any()),
        f"{prefix}wall_empty_prediction": float(not pw.any()),
        f"{prefix}lumen_empty_prediction": float(not pl.any()),
        f"{prefix}outer_area_rel_error": area_error(po, go),
        f"{prefix}wall_area_rel_error": area_error(pw, gw),
        f"{prefix}lumen_area_rel_error": area_error(pl, gl),
        f"{prefix}gt_outer_area_px": float(go.sum()),
        f"{prefix}pred_outer_area_px": float(po.sum()),
        f"{prefix}gt_wall_area_px": float(gw.sum()),
        f"{prefix}pred_wall_area_px": float(pw.sum()),
        f"{prefix}gt_lumen_area_px": float(gl.sum()),
        f"{prefix}pred_lumen_area_px": float(pl.sum()),
    }


def one_metrics(pred: np.ndarray, gt: np.ndarray, prefix: str = "") -> Dict[str, float]:
    return _binary_anatomy_metrics(pred > 0, pred == 1, pred == 2, gt, prefix)


def one_raw_head_metrics(
    raw_outer: np.ndarray,
    raw_lumen: np.ndarray,
    gt: np.ndarray,
    prefix: str = "raw_",
) -> Dict[str, float]:
    """Metrics for the two independent raw heads before topology cleanup.

    Outer is measured from the outer head itself; lumen from the lumen head itself.
    Wall follows the model definition outer AND NOT lumen. This avoids counting
    raw lumen pixels outside the outer head as raw outer prediction.
    """
    po = raw_outer.astype(bool)
    pl = raw_lumen.astype(bool)
    pw = po & ~pl
    return _binary_anatomy_metrics(po, pw, pl, gt, prefix)


def summarize_metrics(records: List[Dict[str, Any]]) -> Dict[str, float]:
    df = pd.DataFrame(records)
    out = {}
    for c in df.select_dtypes(include=[np.number]).columns:
        out[f"{c}_mean"] = float(df[c].mean())
        out[f"{c}_std"] = float(df[c].std())
    out["mean_dice"] = float(np.nanmean([
        out.get("outer_dice_mean", np.nan),
        out.get("wall_dice_mean", np.nan),
        out.get("lumen_dice_mean", np.nan),
    ]))
    return out


def torch_dice(pred: torch.Tensor, gt: torch.Tensor) -> float:
    pred = pred.bool()
    gt = gt.bool()
    inter = (pred & gt).sum().float()
    return float((2 * inter + 1e-6) / (pred.sum() + gt.sum() + 1e-6))

