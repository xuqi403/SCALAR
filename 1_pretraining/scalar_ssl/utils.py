"""Video/image IO, shared reproducibility and utility functions (methodology preserved)."""
from __future__ import annotations
import hashlib
import random
import re
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable, List, Optional
import cv2
import numpy as np
import pandas as pd
import torch
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None
from . import settings
from .settings import *


def seed_everything(seed: int=SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    s = torch.initial_seed() % 2 ** 32
    random.seed(s)
    np.random.seed(s)


def progress(iterable: Iterable, desc: str):
    return tqdm(iterable, desc=desc, dynamic_ncols=True) if tqdm is not None else iterable


def clean_text(value: Any) -> str:
    if value is None:
        return ''
    try:
        if pd.isna(value):
            return ''
    except Exception:
        pass
    s = str(value).strip().strip('"').strip("'")
    return '' if s.lower() in {'', 'nan', 'none', 'null'} else s


def safe_stem(text: str, max_len: int=90) -> str:
    s = re.sub('[<>:"/\\\\|?*\\x00-\\x1f]+', '_', str(text)).strip(' ._')
    return s[:max_len] or 'video'


def video_uid(path: str | Path) -> str:
    p = str(path).replace('\\', '/').lower()
    stem = safe_stem(Path(str(path)).stem, 55)
    return f"{stem}__{hashlib.sha1(p.encode('utf-8', errors='ignore')).hexdigest()[:12]}"


def write_image(path: Path, image_bgr: np.ndarray, quality: int=JPEG_QUALITY) -> bool:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or '.jpg'
    params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)] if ext in {'.jpg', '.jpeg'} else []
    ok, buf = cv2.imencode(ext, image_bgr, params)
    if ok:
        buf.tofile(str(path))
        return True
    return False


def read_image(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f'Cannot decode image: {path}')
    return img


def resolve_video_path(raw_path: Any) -> Optional[Path]:
    raw = clean_text(raw_path)
    if not raw:
        return None
    candidates: List[Path] = []
    try:
        candidates.append(Path(raw))
    except Exception:
        pass
    low = raw.lower()
    for old, new in PATH_REPLACEMENTS:
        if low.startswith(old.lower()):
            candidates.append(Path(new + raw[len(old):]))
    pure = PureWindowsPath(raw)
    rel_parts = [x for x in pure.parts if x not in {'.', '\\', '/'}]
    if pure.drive:
        rel_parts = rel_parts[1:]
    for root in settings.VIDEO_ROOTS:
        if rel_parts:
            candidates.append(root.joinpath(*rel_parts))
        candidates.append(root / pure.name)
    seen = set()
    for c in candidates:
        key = str(c).lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            if c.is_file() and c.suffix.lower() in VIDEO_EXTENSIONS:
                return c.resolve()
        except OSError:
            continue
    return None


def image_tensor(rgb: np.ndarray) -> torch.Tensor:
    x = rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1)).float()


def denormalize_tensor(x: torch.Tensor) -> np.ndarray:
    a = x.detach().cpu().numpy().transpose(1, 2, 0)
    return np.clip((a * IMAGENET_STD + IMAGENET_MEAN) * 255.0, 0, 255).astype(np.uint8)


def laplacian_blur_score(bgr: np.ndarray) -> float:
    if bgr is None or bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def letterbox_bgr(image: np.ndarray, out_size: int=IMAGE_SIZE) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(out_size / max(w, 1), out_size / max(h, 1))
    nw, nh = (max(1, round(w * scale)), max(1, round(h * scale)))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    left = (out_size - nw) // 2
    right = out_size - nw - left
    top = (out_size - nh) // 2
    bottom = out_size - nh - top
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def load_module(path: Path, name: str):
    import importlib.util
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
