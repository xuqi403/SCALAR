"""Original topology cleanup and outer/lumen output decoding."""
from __future__ import annotations
from typing import Dict, Tuple
import cv2
import numpy as np
import torch

TOPOLOGY_MIN_COMPONENT_PX = 24
TOPOLOGY_LUMEN_OUTER_OVERLAP_MIN = 0.75
TOPOLOGY_OUTER_SMOOTH = True
TOPOLOGY_SMOOTH_SIGMA = 0.8
TOPOLOGY_SMOOTH_MAX_AREA_CHANGE = 0.02
TOPOLOGY_SMOOTH_MIN_IOU = 0.97
from config import PRED_THRESHOLD

def component_info(mask: np.ndarray):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    comps = [(i, int(stats[i, cv2.CC_STAT_AREA])) for i in range(1, n)]
    comps.sort(key=lambda x: x[1], reverse=True)
    return labels, comps



def largest_component(mask: np.ndarray, min_px: int = TOPOLOGY_MIN_COMPONENT_PX) -> np.ndarray:
    labels, comps = component_info(mask)
    if not comps or comps[0][1] < min_px:
        return np.zeros_like(mask, dtype=bool)
    return labels == comps[0][0]



def fill_holes(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if not mask.any():
        return mask
    padded = np.pad(mask.astype(np.uint8), 1)
    flood = padded.copy()
    ff = np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, ff, (0, 0), 1)
    return ((padded > 0) | (flood == 0))[1:-1, 1:-1]



def smooth_outer(mask: np.ndarray, lumen: np.ndarray) -> np.ndarray:
    if not TOPOLOGY_OUTER_SMOOTH or mask.sum() < 16:
        return mask
    sigma = TOPOLOGY_SMOOTH_SIGMA
    k = max(3, int(round(sigma * 6 + 1)))
    if k % 2 == 0:
        k += 1
    candidate = cv2.GaussianBlur(mask.astype(np.float32), (k, k), sigma) >= 0.5
    candidate = fill_holes(candidate)
    if not np.all(candidate[lumen]):
        return mask
    area_change = abs(candidate.sum() - mask.sum()) / max(mask.sum(), 1)
    iou = np.logical_and(candidate, mask).sum() / max(np.logical_or(candidate, mask).sum(), 1)
    return candidate if area_change <= TOPOLOGY_SMOOTH_MAX_AREA_CHANGE and iou >= TOPOLOGY_SMOOTH_MIN_IOU else mask



def topology_cleanup(raw_outer: np.ndarray, raw_lumen: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    raw_outer = raw_outer.astype(bool)
    raw_lumen = raw_lumen.astype(bool)
    raw_outer_n = len(component_info(raw_outer)[1])
    raw_lumen_n = len(component_info(raw_lumen)[1])

    outer = largest_component(fill_holes(raw_outer))
    lumen_labels, lumen_comps = component_info(fill_holes(raw_lumen))
    lumen = np.zeros_like(raw_lumen, dtype=bool)
    best_score = -1
    for label, area in lumen_comps:
        if area < TOPOLOGY_MIN_COMPONENT_PX:
            continue
        comp = lumen_labels == label
        overlap = np.logical_and(comp, outer).sum()
        ratio = overlap / max(area, 1)
        if ratio >= TOPOLOGY_LUMEN_OUTER_OVERLAP_MIN and overlap > best_score:
            best_score = overlap
            lumen = comp
    lumen &= outer  # strict anatomical containment
    outer = smooth_outer(outer, lumen)
    lumen &= outer

    post = np.zeros(raw_outer.shape, dtype=np.uint8)
    post[outer] = 1
    post[lumen] = 2
    raw_class = np.zeros_like(post)
    raw_class[raw_outer] = 1
    raw_class[raw_lumen] = 2
    stats = {
        "raw_outer_components": raw_outer_n,
        "raw_lumen_components": raw_lumen_n,
        "post_outer_components": len(component_info(post > 0)[1]),
        "post_lumen_components": len(component_info(post == 2)[1]),
        "raw_lumen_outside_outer_px": int(np.logical_and(raw_lumen, ~raw_outer).sum()),
        "topology_changed_fraction": float(np.mean(post != raw_class)),
        "topology_removed_outer_px": int(max(0, raw_outer.sum() - (post > 0).sum())),
        "topology_removed_lumen_px": int(max(0, raw_lumen.sum() - (post == 2).sum())),
    }
    return post, stats



def decode_outputs(outputs, threshold: float = PRED_THRESHOLD, topology: bool = True):
    raw_outer = torch.sigmoid(outputs["outer_logits"])[:, 0] >= threshold
    raw_lumen = torch.sigmoid(outputs["lumen_logits"])[:, 0] >= threshold
    raw_class = torch.zeros_like(raw_outer, dtype=torch.long)
    raw_class[raw_outer] = 1
    raw_class[raw_lumen] = 2
    if not topology:
        return raw_class, raw_class, [{} for _ in range(len(raw_class))]

    post, stats = [], []
    for o, l in zip(raw_outer.detach().cpu().numpy(), raw_lumen.detach().cpu().numpy()):
        p, s = topology_cleanup(o, l)
        post.append(p)
        stats.append(s)
    return raw_class, torch.from_numpy(np.stack(post)).long().to(raw_class.device), stats
