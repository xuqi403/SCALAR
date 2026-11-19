"""Ultrasound spatial/appearance augmentation and SSL clip dataset (methodology preserved)."""
from __future__ import annotations
import random
from typing import List, Sequence, Tuple
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from . import settings
from .settings import *
from . import cache as _ssl_mod_cache
from . import utils as _ssl_mod_utils


def _sample_spatial_params(h: int, w: int) -> Tuple[bool, np.ndarray]:
    flip = random.random() < HFLIP_PROB
    angle = random.uniform(-SPATIAL_ROTATE_DEG, SPATIAL_ROTATE_DEG)
    scale = random.uniform(SPATIAL_SCALE_MIN, SPATIAL_SCALE_MAX)
    tx = random.uniform(-SPATIAL_TRANSLATE_FRACTION, SPATIAL_TRANSLATE_FRACTION) * w
    ty = random.uniform(-SPATIAL_TRANSLATE_FRACTION, SPATIAL_TRANSLATE_FRACTION) * h
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    return (flip, M.astype(np.float32))


def _apply_spatial(rgb: np.ndarray, flip: bool, M: np.ndarray) -> np.ndarray:
    out = np.ascontiguousarray(rgb[:, ::-1]) if flip else rgb.copy()
    h, w = out.shape[:2]
    return cv2.warpAffine(out, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def _photo_aug(rgb: np.ndarray, strong: bool) -> np.ndarray:
    x = rgb.astype(np.float32) / 255.0
    if strong:
        gamma = random.uniform(*STRONG_GAMMA)
        gain = random.uniform(*STRONG_GAIN)
        bias = random.uniform(*STRONG_BIAS)
        speckle = random.uniform(*STRONG_SPECKLE_STD)
        blur_prob = STRONG_BLUR_PROB
    else:
        gamma = random.uniform(*WEAK_GAMMA)
        gain = random.uniform(*WEAK_GAIN)
        bias = random.uniform(*WEAK_BIAS)
        speckle = random.uniform(*WEAK_SPECKLE_STD)
        blur_prob = WEAK_BLUR_PROB
    x = np.power(np.clip(x, 0, 1), gamma)
    x = x * gain + bias
    if speckle > 0:
        noise = np.random.normal(0, speckle, size=x.shape[:2]).astype(np.float32)
        x = x * (1.0 + noise[..., None])
    x = np.clip(x, 0, 1)
    out = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    if random.random() < blur_prob:
        out = cv2.GaussianBlur(out, (3, 3), sigmaX=random.uniform(0.35, 0.9))
    if strong and random.random() < STRONG_ERASE_PROB:
        h, w = out.shape[:2]
        ew = random.randint(max(8, w // 20), max(10, w // 7))
        eh = random.randint(max(8, h // 20), max(10, h // 7))
        x1 = random.randint(0, max(0, w - ew))
        y1 = random.randint(0, max(0, h - eh))
        local = out[max(0, y1 - 10):min(h, y1 + eh + 10), max(0, x1 - 10):min(w, x1 + ew + 10)]
        fill = int(np.median(local)) if local.size else int(np.median(out))
        out[y1:y1 + eh, x1:x1 + ew] = fill
    return out


def make_ssl_clip_views(clip_bgr: Sequence[np.ndarray], training: bool) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    rgbs = [cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in clip_bgr]
    if training:
        h, w = rgbs[CENTER_INDEX].shape[:2]
        flip, M = _sample_spatial_params(h, w)
        shared = [_apply_spatial(x, flip, M) for x in rgbs]
        weak = [_photo_aug(x, strong=False) for x in shared]
        strong = [_photo_aug(x, strong=True) for x in shared]
    else:
        weak = [x.copy() for x in rgbs]
        strong = [x.copy() for x in rgbs]
    return (weak, strong)


def sample_context_mask(base_valid: np.ndarray, training: bool) -> Tuple[np.ndarray, float]:
    valid = np.asarray(base_valid, dtype=np.float32).copy()
    n_ctx = len(CONTEXT_OFFSETS) - 1
    if valid.shape != (n_ctx,):
        raise ValueError(f'Expected context-valid shape {(n_ctx,)}, got {valid.shape}')
    valid = (valid > 0.5).astype(np.float32)
    if not training:
        eligible = float(valid.sum() >= MIN_TRACKED_CONTEXT_FOR_CONTEXT_LOSS)
        return (valid, eligible)
    keep = valid.copy()
    for i in range(n_ctx):
        if keep[i] > 0.5 and random.random() < CONTEXT_FRAME_DROPOUT_PROB:
            keep[i] = 0.0
    if int(valid.sum()) >= MIN_TRACKED_CONTEXT_FOR_CONTEXT_LOSS:
        need = max(MIN_TRACKED_CONTEXT_FOR_CONTEXT_LOSS, CONTEXT_MIN_KEEP)
        if int(keep.sum()) < need:
            available = np.flatnonzero(valid > 0.5).tolist()
            random.shuffle(available)
            for j in available:
                keep[j] = 1.0
                if int(keep.sum()) >= need:
                    break
    eligible = float(keep.sum() >= MIN_TRACKED_CONTEXT_FOR_CONTEXT_LOSS)
    return (keep.astype(np.float32), eligible)


class CarotidSSLDataset(Dataset):

    def __init__(self, df: pd.DataFrame, training: bool):
        self.df = df.reset_index(drop=True).copy()
        self.training = bool(training)
        self.path_cols = _ssl_mod_cache.clip_path_columns()
        self.idx_cols = _ssl_mod_cache.clip_index_columns()
        self.valid_cols = [f'frame_{off:+d}_tracking_valid' for off in CONTEXT_OFFSETS]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        clip_bgr = [_ssl_mod_utils.read_image(str(row[c])) for c in self.path_cols]
        weak, strong = make_ssl_clip_views(clip_bgr, training=self.training)
        frame_valid = np.asarray([float(row.get(c, 1.0 if i == CENTER_INDEX else 0.0)) for i, c in enumerate(self.valid_cols)], dtype=np.float32)
        frame_valid[CENTER_INDEX] = 1.0
        ctx_valid = np.delete(frame_valid, CENTER_INDEX)
        context_mask, context_eligible = sample_context_mask(ctx_valid, self.training)
        return {'weak': torch.stack([_ssl_mod_utils.image_tensor(x) for x in weak], dim=0), 'strong': torch.stack([_ssl_mod_utils.image_tensor(x) for x in strong], dim=0), 'frame_valid_mask': torch.from_numpy(frame_valid), 'context_mask': torch.from_numpy(context_mask), 'context_loss_enabled': torch.tensor(context_eligible, dtype=torch.float32), 'frame_indices': torch.tensor([int(row[c]) for c in self.idx_cols], dtype=torch.long), 'cache_path': str(row['cache_path']), 'clip_uid': str(row['clip_uid']), 'video_uid': str(row['video_uid']), 'center_frame_idx': int(row['center_frame_idx'])}
