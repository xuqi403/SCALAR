"""Postprocess operations for SCALAR V4/V5 video segmentation."""
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


def component_info(mask: np.ndarray):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    comps = [(i, int(stats[i, cv2.CC_STAT_AREA])) for i in range(1, n)]
    comps.sort(key=lambda x: x[1], reverse=True)
    return labels, comps


def largest_component(mask: np.ndarray, min_px: int = settings.TOPOLOGY_MIN_COMPONENT_PX) -> np.ndarray:
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
    if not settings.TOPOLOGY_OUTER_SMOOTH or mask.sum() < 16:
        return mask
    sigma = settings.TOPOLOGY_SMOOTH_SIGMA
    k = max(3, int(round(sigma * 6 + 1)))
    if k % 2 == 0:
        k += 1
    candidate = cv2.GaussianBlur(mask.astype(np.float32), (k, k), sigma) >= 0.5
    candidate = fill_holes(candidate)
    if not np.all(candidate[lumen]):
        return mask
    area_change = abs(candidate.sum() - mask.sum()) / max(mask.sum(), 1)
    iou = np.logical_and(candidate, mask).sum() / max(np.logical_or(candidate, mask).sum(), 1)
    return candidate if area_change <= settings.TOPOLOGY_SMOOTH_MAX_AREA_CHANGE and iou >= settings.TOPOLOGY_SMOOTH_MIN_IOU else mask


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
        if area < settings.TOPOLOGY_MIN_COMPONENT_PX:
            continue
        comp = lumen_labels == label
        overlap = np.logical_and(comp, outer).sum()
        ratio = overlap / max(area, 1)
        if ratio >= settings.TOPOLOGY_LUMEN_OUTER_OVERLAP_MIN and overlap > best_score:
            best_score = overlap
            lumen = comp
    lumen &= outer
    outer = smooth_outer(outer, lumen)
    lumen &= outer

    post = np.zeros(raw_outer.shape, np.uint8)
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
    }
    return post, stats


def decode_dual_logits(out: Dict[str, torch.Tensor], threshold: float = settings.PRED_THRESHOLD):
    p_o = torch.sigmoid(out["outer_logits"])[:, 0]
    p_l = torch.sigmoid(out["lumen_logits"])[:, 0]
    raw_o = p_o >= threshold
    raw_l = p_l >= threshold
    raw_class = torch.zeros_like(raw_o, dtype=torch.long)
    raw_class[raw_o] = 1
    raw_class[raw_l] = 2

    post, stats = [], []
    for o, l in zip(raw_o.detach().cpu().numpy(), raw_l.detach().cpu().numpy()):
        p, s = topology_cleanup(o, l)
        post.append(p)
        stats.append(s)
    post_t = torch.from_numpy(np.stack(post)).long().to(raw_class.device)
    return raw_class, post_t, stats, p_o, p_l

