"""Artifacts operations for SCALAR V4/V5 video segmentation."""
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
from .utils import read_csv_compatible


def load_plot_module():
    import plot as p
    return p


def update_registry(experiment: str, row: Dict[str, Any]) -> None:
    path = settings.EXPERIMENT_ROOT / "video_experiment_comparison.csv"
    new = pd.DataFrame([{**row, "experiment": experiment}])
    if path.is_file():
        old = read_csv_compatible(path, low_memory=False)
        if "experiment" in old.columns:
            old = old[old["experiment"].astype(str) != experiment]
        new = pd.concat([old, new], ignore_index=True, sort=False)
    new.to_csv(path, index=False, encoding="utf-8-sig")


def fixed_preview_indices(df: pd.DataFrame, n: int, seed: int) -> List[int]:
    if len(df) <= n:
        return list(range(len(df)))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(len(df), n, replace=False).tolist())


def atomic_torch_save(obj: Dict[str, Any], path: Path) -> None:
    """Write one checkpoint atomically.

    Only ONE full model serialization is performed for an epoch. Writing to a
    temporary file first prevents a partially written .pth from being mistaken
    for a valid checkpoint after an interrupted run.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        if tmp.exists():
            tmp.unlink()
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        # Best effort cleanup after a failed/interrupted save.
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def refresh_checkpoint_alias(source: Path, target: Path, allow_copy: bool = False) -> str:
    """Point *target* at an already serialized epoch checkpoint.

    On the normal Windows/NTFS project layout, os.link() creates a hard link:
    the alias is a real .pth path but causes no second model serialization and
    consumes essentially no additional model-file disk space. If hard linking
    is unavailable, copying is deliberately deferred during training; an
    optional sequential copy fallback is allowed only after training finishes.
    """
    source = Path(source)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not source.is_file():
        raise FileNotFoundError(source)

    try:
        if target.exists() or target.is_symlink():
            target.unlink()
    except OSError:
        pass

    try:
        os.link(str(source), str(target))
        return "hardlink"
    except OSError as e:
        if allow_copy:
            shutil.copy2(source, target)
            return "copy"
        warnings.warn(
            f"Could not create checkpoint hard link {target.name} -> {source.name}: {e}. "
            "The alias will be finalized after training; the epoch checkpoint itself is safe."
        )
        return "deferred"


def save_candidate_outputs(
    source_checkpoint: Path,
    checkpoint_path: Path,
    criterion_name: str,
    criterion_value: float,
    dice_score: float,
    val_summary: Dict[str, float],
    val_records: List[Dict[str, Any]],
    metrics_csv_path: Path,
    summary_csv_path: Path,
) -> str:
    """Update one selected candidate without serializing the model again."""
    link_mode = refresh_checkpoint_alias(
        source_checkpoint, checkpoint_path, allow_copy=False
    )

    pd.DataFrame(val_records).to_csv(
        metrics_csv_path, index=False, encoding="utf-8-sig"
    )
    pd.DataFrame([{
        **val_summary,
        "epoch": int(val_summary.get("epoch", -1)),
        "criterion_name": str(criterion_name),
        "criterion_value": float(criterion_value),
        "dice_selection_score": float(dice_score),
        "checkpoint_storage": link_mode,
        "source_epoch_checkpoint": str(source_checkpoint),
    }]).to_csv(
        summary_csv_path, index=False, encoding="utf-8-sig"
    )
    return link_mode

