"""Five-frame adaptive fusion, topology-cleaned masks, and confidence."""
from __future__ import annotations
import gc
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from .settings import IMAGE_SIZE, FUSION_BATCH_SIZE, AMP
def _prediction_confidence(po: np.ndarray, pl: np.ndarray, region: np.ndarray) -> float:
    """One compact uncertainty field retained because it cannot be reconstructed from hard masks."""
    certainty = 1.0 - 0.5 * (4.0 * po * (1.0 - po) + 4.0 * pl * (1.0 - pl))
    if region.any():
        return float(np.mean(certainty[region]))
    return float(np.mean(certainty))


def infer_feature_windows(
    track: pd.DataFrame,
    feature_bank_path: Path,
    feature_shape: Tuple[int, int, int],
    train_lib,
    model,
    device: torch.device,
    out_dir: Path,
    fusion_batch_size: int = FUSION_BATCH_SIZE,
    amp: bool = AMP,
) -> pd.DataFrame:
    """Locked V5 fusion producing segmentation masks and essential status fields."""
    n = len(track)
    bank = np.memmap(feature_bank_path, mode="r", dtype=np.float16, shape=(n, *feature_shape))
    masks_path = out_dir / "masks_448.npy"
    masks = np.lib.format.open_memmap(
        masks_path, mode="w+", dtype=np.uint8, shape=(n, IMAGE_SIZE, IMAGE_SIZE)
    )
    masks[:] = 0

    metrics = track.copy()
    metrics["inference_valid"] = 0
    metrics["prediction_confidence"] = np.nan

    valid_targets = metrics.index[metrics["track_valid"].astype(int).eq(1)].to_numpy(dtype=int)

    for start in range(0, len(valid_targets), fusion_batch_size):
        targets = valid_targets[start:start + fusion_batch_size]
        feat_batch, avail_batch = [], []

        for t in targets:
            seg = int(metrics.at[t, "segment_id"])
            indices, avail = [], []
            for off in train_lib.CLIP_OFFSETS:
                j = t + int(off)
                ok = (
                    0 <= j < n
                    and int(metrics.at[j, "track_valid"]) == 1
                    and int(metrics.at[j, "segment_id"]) == seg
                )
                indices.append(j if ok else t)
                avail.append(1.0 if ok else 0.0)
            avail[train_lib.CENTER_INDEX] = 1.0
            feat_batch.append(np.stack([bank[j] for j in indices], axis=0))
            avail_batch.append(avail)

        feats = torch.from_numpy(np.stack(feat_batch)).to(device=device, dtype=(torch.float16 if device.type == "cuda" else torch.float32))
        availability = torch.tensor(avail_batch, device=device, dtype=torch.float32)

        with torch.no_grad(), torch.amp.autocast(
            device_type="cuda", dtype=torch.float16,
            enabled=bool(amp and device.type == "cuda"),
        ):
            fused, _diag = model.fuse_once(feats, availability, train_lib.CENTER_INDEX)
            out = model.decoder(fused, (IMAGE_SIZE, IMAGE_SIZE))
            po_t = torch.sigmoid(out["outer_logits"][:, 0])
            pl_t = torch.sigmoid(out["lumen_logits"][:, 0])

        po_all = po_t.float().cpu().numpy()
        pl_all = pl_t.float().cpu().numpy()

        for b, t in enumerate(targets):
            raw_outer = po_all[b] >= train_lib.PRED_THRESHOLD
            raw_lumen = pl_all[b] >= train_lib.PRED_THRESHOLD
            mask, _ = train_lib.topology_cleanup(raw_outer, raw_lumen)
            masks[t] = mask

            outer = mask > 0
            region = outer | (po_all[b] > 0.10) | (pl_all[b] > 0.10)

            metrics.at[t, "inference_valid"] = 1
            metrics.at[t, "prediction_confidence"] = _prediction_confidence(po_all[b], pl_all[b], region)

        done = min(start + len(targets), len(valid_targets))
        print(f"  V5 fusion: {done}/{len(valid_targets)}")

    masks.flush()
    del masks
    del bank
    return metrics
