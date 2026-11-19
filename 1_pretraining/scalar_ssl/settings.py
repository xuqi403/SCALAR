#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Carotid DINOv3 sweep-context SSL with conservative stable ROI tracking.
Exports a drop-in DINOv3 backbone for the existing image and V5 pipelines.
"""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import json
import math
import random
import re
import time
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
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
# Paths are injected by CLI in main(); no institution-specific absolute paths are embedded.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
VIDEO_TABLES: List[Path] = []
VIDEO_PATH_COLUMN = "video_path"
VIDEO_TRAIN_FILE = Path("__missing_roi_detector_adapter__.py")
V5_CACHE_MANIFEST = Path("__optional_roi_prior_not_supplied__.csv")
DINO_REPO = Path("__missing_dinov3_source__")
ORIGINAL_DINO_WEIGHTS = Path("__missing_dinov3_weights__.pth")
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "pretraining"
ROI_CACHE_ROOT = OUTPUT_ROOT / "roi_cache"
CACHE_MANIFEST_OUT = OUTPUT_ROOT / "pretrain_roi_manifest.csv"
BACKBONE_OUT = OUTPUT_ROOT / "dinov3_vitb16_carotid_ssl.pth"
BACKBONE_RAW_OUT = OUTPUT_ROOT / "dinov3_vitb16_carotid_ssl_raw_state_dict.pth"
FULL_CHECKPOINT_OUT = OUTPUT_ROOT / "ssl_resume_checkpoint.pth"
VIDEO_ROOTS: List[Path] = []
PATH_REPLACEMENTS: List[Tuple[str, str]] = []
EXPECTED_INTERNAL_VIDEO_COUNT = 5288
APPROVED_VIDEO_UIDS: set = set()

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
CACHE_SCHEMA_VERSION = 3
DETECTION_CANDIDATES_PER_VIDEO = 16
CACHE_CLIPS_PER_VIDEO = 3
CLIP_CANDIDATES_TO_VALIDATE = 5
CONTEXT_OFFSETS = (-4, -2, 0, 2, 4)
CENTER_INDEX = CONTEXT_OFFSETS.index(0)
TRACK_RADIUS_FRAMES = max(abs(int(x)) for x in CONTEXT_OFFSETS)   # run YOLO on every frame t-4...t+4
EDGE_EXCLUDE_FRACTION = 0.03
JPEG_QUALITY = 92
MIN_BLUR_SCORE = 3.0            # intentionally permissive; retain difficult ultrasound
MIN_YOLO_CONFIDENCE = 0.40
ROI_SMOOTH_ALPHA = 0.65
CENTER_CONFIRM_MIN_NEIGHBORS = 1        # among t-1 and t+1
MAX_TRACK_GAP_FRAMES = 1                # permit one missing/rejected intermediate frame
MIN_TEMPORAL_IOU = 0.05
MAX_CENTER_SHIFT_BOX_DIAG_RATIO = 1.75
MAX_CENTER_SHIFT_IMAGE_DIAG_RATIO = 0.20
MIN_BOX_SIZE_RATIO = 0.45
MAX_BOX_SIZE_RATIO = 2.25
TRACK_IOU_WEIGHT = 0.45
TRACK_CENTER_PENALTY = 0.20
TRACK_TRAJECTORY_PENALTY = 0.30
TRACK_SIZE_PENALTY = 0.10
CENTER_COARSE_PRIOR_PENALTY = 0.35    # keep local center identity near the coarse video track
MAX_TRAJECTORY_ERROR_BOX_DIAG_RATIO = 2.00
MIN_TRACKED_CONTEXT_FOR_CONTEXT_LOSS = 2
TRACKING_QC_PREVIEW_CLIPS = 20
TRACKING_QC_THUMB_SIZE = 192
VIDEO_YOLO_BOX_PERCENTILE = 90.0
YOLO_TO_CROP_SCALE = 2.0
TRAINING_ROI_CAP_PERCENTILE = 90.0
MAX_ROI_SCALE_FROM_TRAIN_MEDIAN = 1.10
IMAGE_SIZE = 448
SSL_LAYERS = [8, 9, 10, 11]    # only late DINO features are domain-adapted
TRAINABLE_LAST_N_BLOCKS = 4
PROJECTION_DIM = 256
PROJECTOR_HIDDEN = 512
CONTEXT_HEAD_HIDDEN = 512
BATCH_SIZE = 1
NUM_WORKERS = 4
EPOCHS = 3
SEED = 42
WEIGHT_DECAY = 0.04
LR_BACKBONE = 2e-6
LR_HEAD = 1e-4
WARMUP_FRACTION = 0.05
AMP = True
GRAD_CLIP = 3.0
EMA_MOMENTUM_START = 0.996
EMA_MOMENTUM_END = 0.9998
LAMBDA_DENSE = 1.0
LAMBDA_GLOBAL = 0.20
LAMBDA_CONTEXT_MAX = 0.10
CONTEXT_DELAY_FRACTION = 0.10   # first 10% of optimization: no context pressure
CONTEXT_RAMP_FRACTION = 0.15    # then linearly ramp to LAMBDA_CONTEXT_MAX
CONTEXT_FRAME_DROPOUT_PROB = 0.25
CONTEXT_MIN_KEEP = 1
SSL_VAL_VIDEO_FRACTION = 0.05
SPATIAL_ROTATE_DEG = 6.0
SPATIAL_SCALE_MIN = 0.92
SPATIAL_SCALE_MAX = 1.08
SPATIAL_TRANSLATE_FRACTION = 0.025
HFLIP_PROB = 0.50
WEAK_GAMMA = (0.92, 1.08)
WEAK_GAIN = (0.95, 1.05)
WEAK_BIAS = (-0.015, 0.015)
WEAK_SPECKLE_STD = (0.0, 0.012)
WEAK_BLUR_PROB = 0.08
STRONG_GAMMA = (0.78, 1.22)
STRONG_GAIN = (0.88, 1.12)
STRONG_BIAS = (-0.035, 0.035)
STRONG_SPECKLE_STD = (0.008, 0.035)
STRONG_BLUR_PROB = 0.25
STRONG_ERASE_PROB = 0.12
PROBE_CASES = 12
NN_GALLERY_CASES = 72
NN_QUERY_CASES = 6
NN_TOPK = 3
CONTEXT_VIS_CASES = 8
VIS_THUMB_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
