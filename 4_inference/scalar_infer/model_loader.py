"""Load the paired V5 training architecture and its validated weights."""
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
import sys
import importlib.util
def load_train_module(path: Path, dino_repo: Path, dino_weights: Path, yolo_code_dir: Path):
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Video training library not found: {path}")
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("scalar_video_model_for_inference", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.DINO_REPO = str(dino_repo.expanduser().resolve())
    module.DINO_WEIGHTS = str(dino_weights.expanduser().resolve())
    module.YOLO_CODE_DIR = str(yolo_code_dir.expanduser().resolve())
    if hasattr(module, "configure_runtime"):
        module.configure_runtime(
            dinov3_repo=module.DINO_REPO,
            dinov3_weights=module.DINO_WEIGHTS,
            yolo_code_dir=module.YOLO_CODE_DIR,
        )
    required = (
        "VideoDINOv3Segmenter", "load_dinov3_backbone", "topology_cleanup",
        "import_yolo_components", "yolo_detect_boxes", "letterbox_image_and_mask",
        "image_tensor", "load_checkpoint_state", "CLIP_OFFSETS", "CENTER_INDEX",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(f"Training module is missing: {missing}")
    if tuple(module.CLIP_OFFSETS) != (-4, -2, 0, 2, 4):
        raise ValueError(f"Unexpected context offsets: {module.CLIP_OFFSETS}")
    return module

def load_v5_model(train_lib, checkpoint: Path, device: torch.device):
    if not hasattr(train_lib, "AdaptiveRescueFusion"):
        raise RuntimeError("The selected train file is not the V5 adaptive-rescue version.")
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(checkpoint)

    model = train_lib.VideoDINOv3Segmenter(
        backbone=train_lib.load_dinov3_backbone(),
        fusion="adaptive",
        temporal_loss=False,
    )
    state = train_lib.load_checkpoint_state(str(checkpoint))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint/model mismatch. missing={len(missing)}, unexpected={len(unexpected)}"
        )
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model
