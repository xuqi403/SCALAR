"""Roi operations for SCALAR V4/V5 video segmentation."""
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
from .utils import read_image_unicode


def import_yolo_components():
    if settings.YOLO_CODE_DIR:
        p = Path(settings.YOLO_CODE_DIR)
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    try:
        from yolo_roi import YOLO
        from utils.utils import cvtColor, preprocess_input, resize_image
        return YOLO, cvtColor, preprocess_input, resize_image
    except Exception as e:
        raise ImportError(
            "Cannot import yolo_roi.py / utils.utils. Set YOLO_CODE_DIR at the top of this script "
            "to the directory used by your original VVS inference code. Original error: " + str(e)
        ) from e


@torch.no_grad()
def yolo_detect_boxes(yolo, pil_image: Image.Image, cvtColor, preprocess_input, resize_image) -> List[Dict[str, Any]]:
    image_shape = np.array(np.shape(pil_image)[0:2])
    image = cvtColor(pil_image)
    image_data = resize_image(
        image,
        (yolo.input_shape[1], yolo.input_shape[0]),
        yolo.letterbox_image,
    )
    image_data = np.expand_dims(
        np.transpose(preprocess_input(np.array(image_data, dtype="float32")), (2, 0, 1)),
        0,
    )
    images = torch.from_numpy(image_data)
    if yolo.cuda:
        images = images.cuda()

    outputs = yolo.net(images)
    outputs = yolo.bbox_util.decode_box(outputs)
    results = yolo.bbox_util.non_max_suppression(
        outputs,
        yolo.num_classes,
        yolo.input_shape,
        image_shape,
        yolo.letterbox_image,
        conf_thres=yolo.confidence,
        nms_thres=yolo.nms_iou,
    )
    if results[0] is None:
        return []

    top_label = np.array(results[0][:, 5], dtype="int32")
    top_conf = results[0][:, 4]
    top_boxes = results[0][:, :4]
    W, H = image.size[0], image.size[1]

    boxes = []
    for i, c in enumerate(top_label):
        top, left, bottom, right = top_boxes[i]
        score = float(top_conf[i])
        if score < settings.YOLO_CONF_THRESHOLD:
            continue
        top = max(0, int(np.floor(top)))
        left = max(0, int(np.floor(left)))
        bottom = min(H, int(np.floor(bottom)))
        right = min(W, int(np.floor(right)))
        if right <= left or bottom <= top:
            continue
        cls_name = yolo.class_names[int(c)] if hasattr(yolo, "class_names") else str(int(c))
        boxes.append({
            "x1": left, "y1": top, "x2": right, "y2": bottom,
            "score": score, "class": cls_name, "class_id": int(c),
        })
    return boxes


def box_area(box: Dict[str, Any]) -> float:
    return float(max(0, box["x2"] - box["x1"]) * max(0, box["y2"] - box["y1"]))


def box_center(box: Dict[str, Any]) -> Tuple[float, float]:
    return ((box["x1"] + box["x2"]) / 2.0, (box["y1"] + box["y2"]) / 2.0)


def box_iou(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> float:
    if a is None or b is None:
        return 0.0
    x1 = max(float(a["x1"]), float(b["x1"]))
    y1 = max(float(a["y1"]), float(b["y1"]))
    x2 = min(float(a["x2"]), float(b["x2"]))
    y2 = min(float(a["y2"]), float(b["y2"]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 1e-8 else 0.0


def box_anchor_metrics(candidate: Dict[str, Any], anchor: Dict[str, Any], image_w: int, image_h: int) -> Dict[str, float]:
    cx, cy = box_center(candidate)
    ax, ay = box_center(anchor)
    dist = float(math.hypot(cx - ax, cy - ay))
    image_diag = float(math.hypot(max(image_w, 1), max(image_h, 1)))
    ratio = box_area(candidate) / max(box_area(anchor), 1.0)
    return {
        "iou": box_iou(candidate, anchor),
        "center_shift_image_diag": dist / max(image_diag, 1.0),
        "size_ratio": ratio,
    }


def select_context_box(
    boxes: List[Dict[str, Any]],
    anchor: Dict[str, Any],
    image_w: int,
    image_h: int,
) -> Tuple[Optional[Dict[str, Any]], str, float, Dict[str, float]]:
    if not boxes:
        return None, "missing", settings.CONTEXT_WEIGHT_MISSING, {
            "iou": np.nan, "center_shift_image_diag": np.nan, "size_ratio": np.nan,
        }

    scored = []
    rejected = []
    for b in boxes:
        m = box_anchor_metrics(b, anchor, image_w, image_h)
        stable = (
            settings.CONTEXT_MIN_SIZE_RATIO <= m["size_ratio"] <= settings.CONTEXT_MAX_SIZE_RATIO
            and (
                m["iou"] >= settings.CONTEXT_MIN_IOU
                or m["center_shift_image_diag"] <= settings.CONTEXT_MAX_CENTER_SHIFT_IMAGE_DIAG
            )
        )
        score = float(b.get("score", 0.0)) + 0.35 * m["iou"] - 0.20 * m["center_shift_image_diag"]
        if stable:
            scored.append((score, b, m))
        else:
            rejected.append((score, b, m))

    if scored:
        _, best, m = max(scored, key=lambda x: x[0])
        conf = float(best.get("score", settings.YOLO_CONF_THRESHOLD))
        weight = max(settings.CONTEXT_WEIGHT_GOOD_MIN, min(1.0, 0.50 + 0.50 * conf))
        return best, "stable", weight, m

    _, best, m = max(rejected, key=lambda x: x[0])
    return best, "unstable", settings.CONTEXT_WEIGHT_UNSTABLE, m


def make_tracked_anchor_box(
    candidate: Optional[Dict[str, Any]],
    anchor: Dict[str, Any],
    image_w: int,
    image_h: int,
) -> Dict[str, Any]:
    """Move the ROI center using YOLO while preserving annotation crop size.

    This avoids changing spatial scale between frames. The center-frame mask
    remains valid because only context frames are shifted.
    """
    if candidate is None:
        return dict(anchor)

    crop_w = int(anchor["x2"] - anchor["x1"])
    crop_h = int(anchor["y2"] - anchor["y1"])

    cx, cy = box_center(candidate)

    x1 = int(round(cx - crop_w / 2))
    y1 = int(round(cy - crop_h / 2))

    x1 = max(0, min(max(image_w - crop_w, 0), x1))
    y1 = max(0, min(max(image_h - crop_h, 0), y1))

    return {
        "x1": x1,
        "y1": y1,
        "x2": x1 + crop_w,
        "y2": y1 + crop_h,
        "score": float(candidate.get("score", 0.0)),
        "class": "tracked_context",
        "class_id": int(candidate.get("class_id", -1)),
    }


def crop_exact_box(frame_bgr: np.ndarray, box: Dict[str, Any]) -> Optional[np.ndarray]:
    h, w = frame_bgr.shape[:2]
    x1 = max(0, min(w, int(box["x1"])))
    x2 = max(0, min(w, int(box["x2"])))
    y1 = max(0, min(h, int(box["y1"])))
    y2 = max(0, min(h, int(box["y2"])))
    if x2 <= x1 or y2 <= y1:
        return None
    roi = frame_bgr[y1:y2, x1:x2].copy()
    return roi if roi.size else None


def reconstruction_similarity(reconstructed_bgr: np.ndarray, annotation_image_path: str) -> Tuple[float, float]:
    if not annotation_image_path:
        return np.nan, np.nan
    try:
        ann = read_image_unicode(annotation_image_path)
    except Exception:
        return np.nan, np.nan
    rec = reconstructed_bgr
    if rec.shape[:2] != ann.shape[:2]:
        rec = cv2.resize(rec, (ann.shape[1], ann.shape[0]), interpolation=cv2.INTER_LINEAR)
    g1 = cv2.cvtColor(rec, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g2 = cv2.cvtColor(ann, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mad = float(np.mean(np.abs(g1 - g2)))
    a = g1.ravel() - float(g1.mean())
    b = g2.ravel() - float(g2.mean())
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    corr = float(np.dot(a, b) / denom) if denom > 1e-8 else np.nan
    return corr, mad


def laplacian_blur_score(bgr: np.ndarray) -> float:
    if bgr is None or bgr.size == 0:
        return np.nan
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

