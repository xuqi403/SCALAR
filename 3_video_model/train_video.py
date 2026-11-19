#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SCALAR video training entry point and backward-compatible inference interface.

Training: python train_video.py --experiment V5_adaptive_rescue_spur_curv --mode train ...
Implementation is in scalar_video/; no networks or losses are defined here.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scalar_video import settings
from scalar_video.runner import main, parse_args
from scalar_video import utils, dataset, roi, cache, model, losses, postprocess, metrics, trainer, evaluation, artifacts

_MODULES = (utils, dataset, roi, cache, model, losses, postprocess, metrics, trainer, evaluation, artifacts)

def configure_runtime(*, dinov3_repo=None, dinov3_weights=None, yolo_code_dir=None):
    """Configure shared model paths when the inference module loads this file by path."""
    if dinov3_repo is not None:
        settings.DINO_REPO = str(dinov3_repo)
    if dinov3_weights is not None:
        settings.DINO_WEIGHTS = str(dinov3_weights)
    if yolo_code_dir is not None:
        settings.YOLO_CODE_DIR = str(yolo_code_dir)

def load_dinov3_backbone():
    # Support the original 4_inference loader, which assigns module.DINO_REPO.
    if 'DINO_REPO' in globals(): settings.DINO_REPO = globals()['DINO_REPO']
    if 'DINO_WEIGHTS' in globals(): settings.DINO_WEIGHTS = globals()['DINO_WEIGHTS']
    return model.load_dinov3_backbone()

def import_yolo_components():
    if 'YOLO_CODE_DIR' in globals(): settings.YOLO_CODE_DIR = globals()['YOLO_CODE_DIR']
    return roi.import_yolo_components()

def __getattr__(name: str):
    # Preserve the source module API used by external evaluation and inference.
    if hasattr(settings,name):
        return getattr(settings,name)
    for module in _MODULES:
        if hasattr(module,name):
            return getattr(module,name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

if __name__ == '__main__':
    main()
