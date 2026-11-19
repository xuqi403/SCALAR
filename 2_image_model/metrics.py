"""Original image-level segmentation metrics (pixel units for HD95)."""
from __future__ import annotations
from typing import Dict, List
import numpy as np
import pandas as pd
try:
    from scipy.ndimage import binary_erosion, distance_transform_edt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
def dice_np(pred: np.ndarray, gt: np.ndarray, eps=1e-6) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() and not gt.any():
        return 1.0
    return float((2 * np.logical_and(pred, gt).sum() + eps) / (pred.sum() + gt.sum() + eps))



def iou_np(pred: np.ndarray, gt: np.ndarray, eps=1e-6) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() and not gt.any():
        return 1.0
    return float((np.logical_and(pred, gt).sum() + eps) / (np.logical_or(pred, gt).sum() + eps))



def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    if not SCIPY_AVAILABLE:
        return np.nan
    pred, gt = pred.astype(bool), gt.astype(bool)
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



def one_metrics(pred: np.ndarray, gt: np.ndarray, prefix: str = "") -> Dict[str, float]:
    po, pw, pl = pred > 0, pred == 1, pred == 2
    go, gw, gl = gt > 0, gt == 1, gt == 2
    return {
        f"{prefix}outer_dice": dice_np(po, go),
        f"{prefix}wall_dice": dice_np(pw, gw),
        f"{prefix}lumen_dice": dice_np(pl, gl),
        f"{prefix}outer_iou": iou_np(po, go),
        f"{prefix}wall_iou": iou_np(pw, gw),
        f"{prefix}lumen_iou": iou_np(pl, gl),
        f"{prefix}outer_hd95_px": hd95(po, go),
        f"{prefix}lumen_hd95_px": hd95(pl, gl),
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



def summarize(records: List[Dict]) -> Dict[str, float]:
    df = pd.DataFrame(records)
    out = {}
    for c in df.select_dtypes(include=[np.number]).columns:
        out[f"{c}_mean"] = float(df[c].mean())
        out[f"{c}_std"] = float(df[c].std())
    out["mean_dice"] = float(np.nanmean([
        out.get("outer_dice_mean", np.nan), out.get("wall_dice_mean", np.nan), out.get("lumen_dice_mean", np.nan)
    ]))
    return out
