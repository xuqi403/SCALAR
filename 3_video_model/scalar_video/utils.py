"""Utils operations for SCALAR V4/V5 video segmentation."""
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


def seed_everything(seed: int = settings.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip().strip('"').strip("'")
    return "" if s.lower() in {"", "nan", "none", "null"} else s


def read_csv_compatible(path, **kwargs) -> pd.DataFrame:
    """Read a CSV strictly using safe, common Windows/Chinese encodings.

    Preserve caller arguments (low_memory, dtype etc.). A real malformed CSV
    still raises Pandas ParserError; an undecodable CSV is never silently
    decoded with errors='ignore' or replacement characters.
    """
    import codecs

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    if "encoding" in kwargs:
        raise TypeError("Pass CSV encoding via read_csv_compatible's auto-detection, not encoding=")

    with path.open("rb") as f:
        first = f.read(4)

    # Longest BOM patterns first (UTF-32 shares a prefix with UTF-16).
    if first.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
        candidates = ["utf-32"]
    elif first.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        candidates = ["utf-16"]
    elif first.startswith(codecs.BOM_UTF8):
        candidates = ["utf-8-sig"]
    elif b"\x00" in first:
        # BOM-less UTF-16 files sometimes come from Excel exports.
        candidates = ["utf-16-le", "utf-16-be", "utf-8-sig", "gb18030", "big5"]
    else:
        candidates = ["utf-8-sig", "gb18030", "big5"]

    candidates += list(settings.CSV_EXTRA_ENCODINGS)
    errors = []
    for encoding in dict.fromkeys(candidates):
        try:
            df = pd.read_csv(path, encoding=encoding, encoding_errors="strict", **kwargs)
            # A wrong UTF-16 guess can decode nonsense without an error;
            # an actual CSV must have readable, non-NUL column names.
            if any("\x00" in str(c) for c in df.columns):
                raise UnicodeError("NUL byte found in parsed CSV headers")
            print(f"[CSV encoding] {path.name}: {encoding} | {len(df)} rows")
            return df
        except UnicodeError as exc:
            errors.append(f"{encoding}: {str(exc)[:160]}")

    raise UnicodeError(
        f"Failed to decode CSV: {path}\n"
        + "\n".join(errors)
        + "\nSave the CSV as UTF-8 with BOM in Excel/WPS, or explicitly "
          "add its verified encoding to CSV_EXTRA_ENCODINGS."
    )


def progress(
    iterable: Iterable,
    desc: str,
    total: Optional[int] = None,
    unit: str = "it",
    leave: bool = True,
):
    """Unified tqdm wrapper with explicit totals when available.

    Some iterators used in this project (pandas.iterrows, enumerate) do not expose
    __len__(), so tqdm cannot infer the denominator unless total is provided.
    Supplying total keeps long video-cache preparation visibly alive instead of
    looking stalled.
    """
    if tqdm is None:
        return iterable
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        unit=unit,
        dynamic_ncols=True,
        mininterval=0.5,
        smoothing=0.1,
        leave=leave,
    )


def progress_postfix(bar, text: str, refresh: bool = False) -> None:
    """Safely update tqdm postfix without breaking the no-tqdm fallback."""
    if tqdm is not None and hasattr(bar, "set_postfix_str"):
        bar.set_postfix_str(text, refresh=refresh)


def base_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_image_unicode(path: str) -> np.ndarray:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)
    data = np.fromfile(str(p), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot decode image: {path}")
    return img


def resolve_file_path(raw_path: str) -> str:
    raw = clean_text(raw_path)
    if not raw:
        return ""
    candidates = [raw]
    low = raw.lower()
    for old, new in settings.PATH_REPLACEMENTS:
        if low.startswith(old.lower()):
            candidates.append(new + raw[len(old):])
    for c in candidates:
        try:
            if Path(c).is_file():
                return str(Path(c))
        except OSError:
            pass
    return ""


def resolve_video_path(raw_path: str) -> str:
    raw = clean_text(raw_path)
    if not raw:
        return ""
    direct = Path(raw)
    if direct.is_file():
        return str(direct)

    normalized = raw.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    normalized = normalized.lstrip("/")

    for root in settings.VIDEO_ROOTS:
        candidate = root / Path(PureWindowsPath(normalized))
        if candidate.is_file():
            return str(candidate)
        # PureWindowsPath on a slash path can sometimes keep it as one part on POSIX;
        # the split fallback is harmless on Windows and useful for testing elsewhere.
        candidate2 = root.joinpath(*[x for x in normalized.split("/") if x])
        if candidate2.is_file():
            return str(candidate2)
    return ""


def polygon_mask(points, h: int, w: int) -> np.ndarray:
    out = np.zeros((h, w), dtype=np.uint8)
    if points is None or len(points) < 3:
        return out
    pts = np.asarray(points, dtype=np.float32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    cv2.fillPoly(out, [np.round(pts).astype(np.int32)], 1)
    return out


def load_labelme_mask(json_path: str, h: int, w: int) -> np.ndarray:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    outer = np.zeros((h, w), np.uint8)
    lumen = np.zeros((h, w), np.uint8)
    for shape in data.get("shapes", []):
        label = str(shape.get("label", "")).strip().lower()
        pts = shape.get("points", [])
        shape_type = str(shape.get("shape_type", "polygon")).strip().lower()
        if shape_type not in {"", "polygon"} or pts is None or len(pts) < 3:
            continue
        poly = polygon_mask(pts, h, w)
        if label == "wall":
            outer = np.maximum(outer, poly)
        elif label == "lumen":
            lumen = np.maximum(lumen, poly)
    if outer.sum() == 0 or lumen.sum() == 0:
        raise ValueError(f"Missing wall/lumen polygon: {json_path}")
    lumen &= outer
    target = np.zeros((h, w), np.uint8)
    target[outer > 0] = 1
    target[lumen > 0] = 2
    return target


def inspect_labelme(json_path: str) -> Tuple[bool, bool, int, int]:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        has_wall = False
        has_lumen = False
        for s in data.get("shapes", []):
            label = str(s.get("label", "")).strip().lower()
            pts = s.get("points", [])
            if pts is not None and len(pts) >= 3:
                has_wall |= label == "wall"
                has_lumen |= label == "lumen"
        return has_wall, has_lumen, int(data.get("imageHeight", 0) or 0), int(data.get("imageWidth", 0) or 0)
    except Exception:
        return False, False, 0, 0


def letterbox_image_and_mask(
    image_rgb: np.ndarray,
    mask: Optional[np.ndarray],
    out_size: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    h, w = image_rgb.shape[:2]
    scale = min(out_size / max(w, 1), out_size / max(h, 1))
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    image = cv2.resize(image_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out_img = np.zeros((out_size, out_size, 3), np.uint8)
    left = (out_size - nw) // 2
    top = (out_size - nh) // 2
    out_img[top:top + nh, left:left + nw] = image

    out_mask = None
    if mask is not None:
        m = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
        out_mask = np.zeros((out_size, out_size), np.uint8)
        out_mask[top:top + nh, left:left + nw] = m
    return out_img, out_mask


def image_tensor(rgb: np.ndarray) -> torch.Tensor:
    x = rgb.astype(np.float32) / 255.0
    x = (x - settings.IMAGENET_MEAN) / settings.IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1)).float()


def denormalize_tensor(x: torch.Tensor) -> np.ndarray:
    a = x.detach().cpu().numpy().transpose(1, 2, 0)
    return np.clip((a * settings.IMAGENET_STD + settings.IMAGENET_MEAN) * 255, 0, 255).astype(np.uint8)

