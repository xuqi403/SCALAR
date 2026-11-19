"""Runner operations for SCALAR V4/V5 video segmentation."""
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
from .artifacts import atomic_torch_save, fixed_preview_indices, load_plot_module, refresh_checkpoint_alias, save_candidate_outputs, update_registry
from .cache import cache_fingerprint, prepare_clip_cache
from .dataset import VideoClipDataset, attach_image_split, audit_sparse_video_annotations
from .evaluation import evaluate_paired, evaluate_temporal_stability
from .losses import DualAnatomyLoss
from .model import VideoDINOv3Segmenter, initialize_adaptive_from_previous_checkpoint, initialize_video_from_image_checkpoint, load_dinov3_backbone, load_image_model
from .trainer import build_optimizer, evaluate_video, save_temporal_epoch_preview, train_one_epoch
from .utils import base_model, clean_text, read_csv_compatible, save_json, seed_everything, seed_worker


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SCALAR video segmentation (V4 initialization / final V5)")
    p.add_argument("--experiment", choices=list(settings.VIDEO_EXPERIMENTS), default=settings.DEFAULT_VIDEO_EXPERIMENT)
    p.add_argument("--mode", choices=["all", "prepare", "train"], default=settings.MODE,
                   help="all=raw-video clip preparation + training, prepare=clip preparation, train=NPZ cache-only training")
    p.add_argument("--manifest", default=settings.MANIFEST_PATH, help="Private frozen image-level split CSV; required for prepare/all")
    p.add_argument("--image_checkpoint", default=settings.IMAGE_CHECKPOINT)
    p.add_argument("--dinov3_repo", default=settings.DINO_REPO)
    p.add_argument("--dinov3_weights", default=settings.DINO_WEIGHTS)
    p.add_argument("--v4_checkpoint", default=None, help="Required V4 initialization checkpoint for V5")
    p.add_argument("--yolo_code_dir", default=settings.YOLO_CODE_DIR, help="Path containing original yolo_roi.py and utils/ for clip preparation")
    p.add_argument("--video_root", action="append", default=[], help="Optional private video root for relative paths; repeatable")
    p.add_argument("--output_root", default=str(settings.PROJECT_ROOT))
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--run_tag", default=settings.VIDEO_RUN_TAG)
    p.add_argument("--epochs", type=int, default=settings.EPOCHS)
    p.add_argument("--batch_size", type=int, default=settings.BATCH_SIZE)
    p.add_argument("--num_workers", type=int, default=settings.NUM_WORKERS)
    p.add_argument("--run_test", action="store_true", default=settings.RUN_TEST)
    p.add_argument("--force_rebuild_cache", action="store_true", default=settings.FORCE_REBUILD_CACHE)
    return p.parse_args(argv)


def main():
    args = parse_args()
    settings.MANIFEST_PATH = args.manifest
    settings.IMAGE_SPLIT_PATH = args.manifest
    settings.IMAGE_CHECKPOINT = args.image_checkpoint
    settings.DINO_REPO = args.dinov3_repo
    settings.DINO_WEIGHTS = args.dinov3_weights
    settings.YOLO_CODE_DIR = args.yolo_code_dir
    settings.VIDEO_ROOTS = [Path(x) for x in args.video_root]
    settings.PROJECT_ROOT = Path(args.output_root)
    settings.EXPERIMENT_ROOT = settings.PROJECT_ROOT / "experiments"
    settings.CACHE_DIR = Path(args.cache_dir) if args.cache_dir else settings.PROJECT_ROOT / "clip_cache"
    settings.VIDEO_RUN_TAG = args.run_tag
    settings.V5_INIT_CHECKPOINT = (Path(args.v4_checkpoint) if args.v4_checkpoint else
                          settings.EXPERIMENT_ROOT / f"V4_attn_last4__{settings.VIDEO_RUN_TAG}" / "best_video_checkpoint.pth")
    settings.EPOCHS, settings.BATCH_SIZE, settings.NUM_WORKERS = args.epochs, args.batch_size, args.num_workers
    if min(settings.EPOCHS, settings.BATCH_SIZE) < 1 or settings.NUM_WORKERS < 0:
        raise ValueError("Invalid epochs/batch_size/num_workers")
    seed_everything(settings.SEED)
    cfg = dict(settings.VIDEO_EXPERIMENTS[args.experiment])
    if cfg.get("outer_spur_loss", False):
        print(
            f"[OUTER-SPUR] enabled | lambda={settings.LAMBDA_OUTER_SPUR} | "
            f"0-{settings.OUTER_SPUR_ALLOW_PX}px free; "
            f"{settings.OUTER_SPUR_ALLOW_PX}-{settings.OUTER_SPUR_MID_PX}px weak; "
            f"{settings.OUTER_SPUR_MID_PX}-{settings.OUTER_SPUR_FAR_PX}px moderate; "
            f">{settings.OUTER_SPUR_FAR_PX}px strong"
        )
    if cfg.get("outer_curvature_loss", False):
        print(
            f"[OUTER-CURV] enabled | lambda={settings.LAMBDA_OUTER_CURV} | "
            f"deadzone={settings.OUTER_CURV_DEADZONE} | "
            f"smooth_kernel={settings.OUTER_CURV_SMOOTH_KERNEL}"
        )

    run_name = f"{args.experiment}__{settings.VIDEO_RUN_TAG}"
    out_dir = settings.EXPERIMENT_ROOT / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    settings.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    settings.EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("Carotid DINOv3 VIDEO segmentation experiment")
    print("=" * 88)
    print(f"Experiment:    {args.experiment}")
    print(f"Run name:      {run_name}")
    print(f"Description:   {cfg['description']}")
    print(f"V4 init ckpt:  {settings.V5_INIT_CHECKPOINT}")
    print(f"Mode:          {args.mode}")
    print(f"Manifest:      {settings.MANIFEST_PATH}")
    print(f"Image split:   {settings.IMAGE_SPLIT_PATH}")
    print(f"Image ckpt:    {settings.IMAGE_CHECKPOINT}")
    print(f"Clip offsets:  {settings.CLIP_OFFSETS}")
    print(f"Run test:      {args.run_test}")

    config = {
        "experiment": args.experiment,
        "run_name": run_name,
        "video_run_tag": settings.VIDEO_RUN_TAG,
        **cfg,
        "manifest": settings.MANIFEST_PATH,
        "image_split": settings.IMAGE_SPLIT_PATH,
        "image_checkpoint": settings.IMAGE_CHECKPOINT,
        "clip_offsets": list(settings.CLIP_OFFSETS),
        "image_size": settings.IMAGE_SIZE,
        "batch_size": settings.BATCH_SIZE,
        "epochs": settings.EPOCHS,
        "seed": settings.SEED,
        "run_test": bool(args.run_test),
        "checkpoint_storage": {
            "save_every_epoch": bool(settings.SAVE_EVERY_EPOCH_CHECKPOINT),
            "full_model_serializations_per_epoch": 1,
            "candidate_alias_strategy": "hardlink_when_supported; final_copy_fallback_only",
        },
        "loss": {
            "wall": settings.LAMBDA_WALL,
            "containment": settings.LAMBDA_CONTAINMENT,
            "boundary": settings.LAMBDA_BOUNDARY,
            "outer_spur_enabled": bool(cfg.get("outer_spur_loss", False)),
            "outer_spur_lambda": settings.LAMBDA_OUTER_SPUR,
            "outer_spur_allow_px": settings.OUTER_SPUR_ALLOW_PX,
            "outer_spur_mid_px": settings.OUTER_SPUR_MID_PX,
            "outer_spur_far_px": settings.OUTER_SPUR_FAR_PX,
            "outer_spur_prob_power": settings.OUTER_SPUR_PROB_POWER,
            "outer_curvature_enabled": bool(cfg.get("outer_curvature_loss", False)),
            "outer_curvature_lambda": settings.LAMBDA_OUTER_CURV,
            "outer_curvature_deadzone": settings.OUTER_CURV_DEADZONE,
            "outer_curvature_smooth_kernel": settings.OUTER_CURV_SMOOTH_KERNEL,
            "outer_curvature_grad_scale": settings.OUTER_CURV_GRAD_SCALE,
            "aux_center": settings.LAMBDA_AUX_CENTER,
            "temporal_area": settings.LAMBDA_TEMPORAL_AREA,
            "temporal_centroid": settings.LAMBDA_TEMPORAL_CENTROID,
            "v5_center_aux": settings.LAMBDA_V5_CENTER_AUX,
            "v5_context_aux": settings.LAMBDA_V5_CONTEXT_AUX,
            "v5_gate": settings.LAMBDA_V5_GATE,
        },
        "v5": {
            "init_checkpoint": str(settings.V5_INIT_CHECKPOINT),
            "max_align_offset_feature_px": settings.V5_MAX_ALIGN_OFFSET_FEATURE_PX,
            "center_corrupt_prob": settings.V5_CENTER_CORRUPT_PROB,
            "context_dropout_prob": settings.V5_CONTEXT_DROPOUT_PROB,
            "gate_temperature": settings.V5_GATE_TEMPERATURE,
        },
    }
    save_json(out_dir / "experiment_config.json", config)

    # Preparing clips requires private annotations, raw videos and the YOLO ROI implementation.
    # Training from previously constructed .npz clips does NOT require clinical tables or raw video access.
    if args.mode in {"all", "prepare"}:
        if not Path(settings.MANIFEST_PATH).is_file():
            raise FileNotFoundError(f"Private preassigned split manifest not found: {settings.MANIFEST_PATH}")
        manifest = read_csv_compatible(settings.MANIFEST_PATH, low_memory=False)
        sparse = audit_sparse_video_annotations(manifest, out_dir)
        split_df = attach_image_split(sparse, out_dir)
        print("Split frames -> " + " | ".join(
            f"{s} {int((split_df['split'] == s).sum())}" for s in ("train", "val", "test")
        ))
        cache_df = prepare_clip_cache(split_df, out_dir, force=args.force_rebuild_cache)
    else:
        cache_manifest = settings.CACHE_DIR / "video_clip_manifest.csv"
        if not cache_manifest.is_file():
            raise FileNotFoundError(
                f"Prepared cache manifest not found: {cache_manifest}. Run --mode prepare first."
            )
        cache_df = read_csv_compatible(cache_manifest, low_memory=False)
        required_columns = {"cache_path", "split"}
        if not required_columns.issubset(cache_df.columns):
            raise ValueError(f"Cache manifest lacks columns: {required_columns - set(cache_df.columns)}")
        if not cache_df["split"].astype(str).str.lower().isin({"train", "val", "test"}).all():
            raise ValueError("Cache manifest contains split labels outside train/val/test")
        if not cache_df["cache_path"].astype(str).map(lambda p: Path(p).is_file()).all():
            raise FileNotFoundError("A referenced .npz clip is missing; correct private cache paths")
        for key in ("patient_id", "video_path"):
            if key in cache_df.columns:
                x = cache_df[key].fillna("").astype(str).str.strip()
                g = cache_df[x.ne("")].groupby(key)["split"].nunique()
                if (g > 1).any():
                    raise RuntimeError(f"Leakage: {key} crosses train/val/test in the cache")
        if Path(settings.MANIFEST_PATH).is_file():
            manifest = read_csv_compatible(settings.MANIFEST_PATH, low_memory=False)
            split_df = attach_image_split(audit_sparse_video_annotations(manifest, out_dir), out_dir)
            expected_fp = cache_fingerprint(split_df)
            got_fp = clean_text(cache_df.get("cache_fingerprint", pd.Series([""])).iloc[0]) if len(cache_df) else ""
            if got_fp != expected_fp:
                raise RuntimeError("Prepared cache does not match supplied private manifest or settings")
        else:
            print("[WARNING] Cache-only mode: no private manifest supplied; original cache fingerprint not revalidated")
        cache_df.to_csv(out_dir / "video_clip_manifest_used.csv", index=False, encoding="utf-8-sig")

    if args.mode != "prepare":
        # Image model is required by the final paired evaluation. V5 additionally requires V4 init.
        if not Path(settings.IMAGE_CHECKPOINT).is_file():
            raise FileNotFoundError(f"Image D checkpoint not found: {settings.IMAGE_CHECKPOINT}")
        if bool(cfg.get("adaptive_rescue", False)) and not Path(settings.V5_INIT_CHECKPOINT).is_file():
            raise FileNotFoundError(f"Final V5 requires an existing V4 checkpoint: {settings.V5_INIT_CHECKPOINT}")

    if args.mode == "prepare":
        print("Preparation completed. Inspect video_clip_cache_summary.json and reconstruction QC before training.")
        return

    train_df = cache_df[cache_df["split"].astype(str) == "train"].copy()
    val_df = cache_df[cache_df["split"].astype(str) == "val"].copy()
    test_df = cache_df[cache_df["split"].astype(str) == "test"].copy()
    if min(len(train_df), len(val_df), len(test_df)) == 0:
        raise RuntimeError("One of train/val/test cache subsets is empty.")

    print(f"Cached samples -> train {len(train_df)} | val {len(val_df)} | test {len(test_df)}")

    adaptive_mode = bool(cfg.get("adaptive_rescue", False))
    train_ds = VideoClipDataset(train_df, training=True, adaptive_rescue=adaptive_mode)
    val_ds = VideoClipDataset(val_df, training=False, adaptive_rescue=adaptive_mode)
    test_ds = VideoClipDataset(test_df, training=False, adaptive_rescue=adaptive_mode)
    gen = torch.Generator().manual_seed(settings.SEED)
    loader_kwargs = dict(
        batch_size=settings.BATCH_SIZE,
        num_workers=settings.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
    )
    train_loader = DataLoader(train_ds, shuffle=True, generator=gen, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(settings.AMP and device.type == "cuda")
    plot_module = load_plot_module()

    # -------------------------------------------------------------------------
    # V4: initialize from image D; V5: initialize from the resulting V4 checkpoint
    # -------------------------------------------------------------------------
    backbone = load_dinov3_backbone()
    video_model = VideoDINOv3Segmenter(
        backbone=backbone,
        fusion=cfg["fusion"],
        temporal_loss=cfg["temporal_loss"],
    )
    if cfg.get("adaptive_rescue", False):
        if not initialize_adaptive_from_previous_checkpoint(video_model, settings.V5_INIT_CHECKPOINT):
            raise FileNotFoundError(
                f"V5 requires the predefined V4 initialization checkpoint, but it was not found: "
                f"{settings.V5_INIT_CHECKPOINT}. Run/restore V4 first; do not silently change V5 to image-D initialization."
            )
    else:
        initialize_video_from_image_checkpoint(video_model, settings.IMAGE_CHECKPOINT)
    video_model = video_model.to(device)
    if torch.cuda.device_count() > 1:
        video_model = nn.DataParallel(video_model)
        print("Using GPUs: " + ", ".join(
            f"cuda:{i}={torch.cuda.get_device_name(i)}" for i in range(torch.cuda.device_count())
        ))

    optimizer = build_optimizer(video_model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings.EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    anatomy_loss = DualAnatomyLoss(
        enable_outer_spur=cfg.get("outer_spur_loss", False),
        enable_outer_curvature=cfg.get("outer_curvature_loss", False),
    )

    history = []

    # Independent checkpoint candidates.
    # 1) hard-mask weighted Dice (legacy comparability)
    best_dice_score = -np.inf
    best_dice_epoch = 0

    # 2) final fused anatomy/segmentation objective (PRIMARY)
    best_anatomy_loss = np.inf
    best_anatomy_epoch = 0

    # 3) V5 adaptive mechanism objective (diagnostic candidate)
    best_adaptive_loss = np.inf
    best_adaptive_epoch = 0

    best_dice_ckpt = out_dir / "best_dice_checkpoint.pth"
    best_anatomy_ckpt = out_dir / "best_anatomy_checkpoint.pth"
    best_adaptive_ckpt = out_dir / "best_adaptive_checkpoint.pth"

    # Historical filename retained for downstream inference compatibility.
    # After training this file is an exact copy/alias of best_anatomy_checkpoint.pth.
    best_ckpt = out_dir / "best_video_checkpoint.pth"

    # If early stopping is ever enabled, monitor the primary anatomy criterion.
    no_improve = 0

    fixed_idx = fixed_preview_indices(val_df, settings.FIXED_PREVIEW_N, settings.SEED + 100)
    pd.DataFrame({"dataset_index": fixed_idx}).to_csv(
        out_dir / "fixed_validation_cases.csv", index=False, encoding="utf-8-sig"
    )

    epoch_ckpt_dir = out_dir / "epoch_checkpoints"
    if settings.SAVE_EVERY_EPOCH_CHECKPOINT:
        epoch_ckpt_dir.mkdir(parents=True, exist_ok=True)
    current_epoch_ckpt = out_dir / "_current_epoch_checkpoint.pth"

    for epoch in range(1, settings.EPOCHS + 1):
        train_stats = train_one_epoch(
            video_model, train_loader, optimizer, anatomy_loss, scaler,
            device, use_amp, cfg, epoch,
        )
        val_summary, val_records = evaluate_video(
            video_model, val_loader, anatomy_loss, device, use_amp,
            f"Val {epoch:02d}/{settings.EPOCHS}", cfg,
        )

        dice_score = (
            settings.CHECKPOINT_OUTER_WEIGHT * val_summary["outer_dice_mean"]
            + settings.CHECKPOINT_WALL_WEIGHT * val_summary["wall_dice_mean"]
            + settings.CHECKPOINT_LUMEN_WEIGHT * val_summary["lumen_dice_mean"]
        )
        val_anatomy_loss = float(val_summary["anatomy_loss"])
        val_adaptive_loss = float(val_summary.get("adaptive_loss", np.nan))
        row = {
            "epoch": epoch,
            **train_stats,
            "val_loss": val_summary["loss"],
            "val_anatomy_loss": val_anatomy_loss,
            "val_regularized_anatomy_loss": val_summary.get("regularized_anatomy_loss", val_anatomy_loss),
            "val_outer_spur_loss": val_summary.get("outer_spur_loss", 0.0),
            "val_outer_curvature_loss": val_summary.get("outer_curvature_loss", 0.0),
            "val_outer_spur_weighted": val_summary.get("outer_spur_weighted", 0.0),
            "val_outer_curvature_weighted": val_summary.get("outer_curvature_weighted", 0.0),
            "val_temporal_loss": val_summary.get("temporal_loss", 0.0),
            "val_adaptive_loss": val_summary.get("adaptive_loss", 0.0),
            "val_gate_loss": val_summary.get("gate_loss", 0.0),
            "val_context_aux_loss": val_summary.get("context_aux_loss", 0.0),
            "val_outer_dice": val_summary["outer_dice_mean"],
            "val_wall_dice": val_summary["wall_dice_mean"],
            "val_lumen_dice": val_summary["lumen_dice_mean"],
            "val_outer_hd95_px": val_summary.get("outer_hd95_px_mean", np.nan),
            "val_lumen_hd95_px": val_summary.get("lumen_hd95_px_mean", np.nan),
            # Keep the legacy column for existing plotting code.
            "selection_score": dice_score,
            "dice_selection_score": dice_score,
            "lr_backbone": optimizer.param_groups[0]["lr"],
            "lr_temporal": optimizer.param_groups[1]["lr"],
            "lr_decoder": optimizer.param_groups[2]["lr"],
        }
        history.append(row)
        hist_df = pd.DataFrame(history)
        hist_df.to_csv(out_dir / "history.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(val_records).to_csv(
            out_dir / f"validation_epoch_{epoch:03d}_metrics.csv",
            index=False, encoding="utf-8-sig",
        )
        pd.DataFrame(val_records).to_csv(
            out_dir / "validation_latest_metrics.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame([{
            **val_summary,
            "epoch": epoch,
            "selection_score": dice_score,
            "dice_selection_score": dice_score,
            "val_anatomy_loss": val_anatomy_loss,
        }]).to_csv(
            out_dir / "validation_latest_summary.csv", index=False, encoding="utf-8-sig"
        )
        plot_module.update_training_monitor(out_dir)

        if epoch == 1 or (settings.PREVIEW_EVERY > 0 and epoch % settings.PREVIEW_EVERY == 0):
            save_temporal_epoch_preview(
                video_model, val_ds, fixed_idx, device, use_amp, plot_module,
                out_dir / f"validation_fixed_epoch_{epoch:03d}.png", epoch,
            )
            rng = np.random.default_rng(settings.SEED + epoch)
            n_random = min(settings.RANDOM_PREVIEW_N, len(val_ds))
            random_idx = sorted(rng.choice(len(val_ds), n_random, replace=False).tolist()) if n_random > 0 else []
            save_temporal_epoch_preview(
                video_model, val_ds, random_idx, device, use_amp, plot_module,
                out_dir / f"validation_random_epoch_{epoch:03d}.png", epoch,
            )
            pd.DataFrame({"dataset_index": random_idx}).to_csv(
                out_dir / f"validation_random_epoch_{epoch:03d}_cases.csv",
                index=False, encoding="utf-8-sig",
            )

        print(
            f"Epoch {epoch:02d}/{settings.EPOCHS} | {train_stats['stage']:<16} "
            f"| train {train_stats['train_loss']:.4f} "
            f"| val {val_summary['loss']:.4f} "
            f"| anatomy {val_anatomy_loss:.4f} "
            f"| adaptive {val_adaptive_loss:.4f} "
            f"| O {val_summary['outer_dice_mean']:.4f} "
            f"| W {val_summary['wall_dice_mean']:.4f} "
            f"| L {val_summary['lumen_dice_mean']:.4f} "
            f"| spur {val_summary.get('outer_spur_loss', 0.0):.4f} "
            f"| curv {val_summary.get('outer_curvature_loss', 0.0):.4f} "
            f"| dice-score {dice_score:.4f}"
        )

        candidate_summary = {
            **val_summary,
            "anatomy_loss": val_anatomy_loss,
            "dice_selection_score": dice_score,
        }

        # Serialize the full model EXACTLY ONCE for this epoch.
        # Candidate checkpoints below are hard-link aliases to this file, so a
        # best-Dice / best-anatomy / best-adaptive update does not write the model
        # a second, third, or fourth time.
        epoch_ckpt_path = (
            epoch_ckpt_dir / f"epoch_{epoch:03d}.pth"
            if settings.SAVE_EVERY_EPOCH_CHECKPOINT
            else current_epoch_ckpt
        )
        epoch_payload = {
            "model_state_dict": base_model(video_model).state_dict(),
            "experiment": args.experiment,
            "run_name": run_name,
            "video_run_tag": settings.VIDEO_RUN_TAG,
            "v4_init_checkpoint": str(settings.V5_INIT_CHECKPOINT),
            "image_checkpoint": str(settings.IMAGE_CHECKPOINT),
            "config": cfg,
            "epoch": int(epoch),
            "checkpoint_role": "epoch_archive",
            "dice_selection_score": float(dice_score),
            "val_loss": float(val_summary.get("loss", np.nan)),
            "val_anatomy_loss": float(val_anatomy_loss),
            "val_regularized_anatomy_loss": float(val_summary.get("regularized_anatomy_loss", val_anatomy_loss)),
            "val_outer_spur_loss": float(val_summary.get("outer_spur_loss", 0.0)),
            "val_outer_curvature_loss": float(val_summary.get("outer_curvature_loss", 0.0)),
            "val_outer_spur_weighted": float(val_summary.get("outer_spur_weighted", 0.0)),
            "val_outer_curvature_weighted": float(val_summary.get("outer_curvature_weighted", 0.0)),
            "val_adaptive_loss": float(val_adaptive_loss),
            "val_outer_dice": float(val_summary.get("outer_dice_mean", np.nan)),
            "val_wall_dice": float(val_summary.get("wall_dice_mean", np.nan)),
            "val_lumen_dice": float(val_summary.get("lumen_dice_mean", np.nan)),
            "cache_fingerprint": clean_text(cache_df["cache_fingerprint"].iloc[0]),
        }
        atomic_torch_save(epoch_payload, epoch_ckpt_path)
        # Drop the temporary Python container immediately; model tensors remain
        # owned by the live model and are not duplicated for candidate aliases.
        del epoch_payload

        # A. Best legacy hard-Dice candidate.
        if dice_score > best_dice_score:
            best_dice_score = float(dice_score)
            best_dice_epoch = int(epoch)
            candidate_summary["epoch"] = int(epoch)
            save_candidate_outputs(
                epoch_ckpt_path, best_dice_ckpt,
                "dice_score", dice_score, dice_score,
                candidate_summary, val_records,
                out_dir / "validation_best_dice_metrics.csv",
                out_dir / "validation_best_dice_summary.csv",
            )

        # B. Best final fused anatomy candidate -- PRIMARY.
        if val_anatomy_loss < best_anatomy_loss:
            best_anatomy_loss = float(val_anatomy_loss)
            best_anatomy_epoch = int(epoch)
            no_improve = 0
            candidate_summary["epoch"] = int(epoch)
            save_candidate_outputs(
                epoch_ckpt_path, best_anatomy_ckpt,
                "anatomy_loss", val_anatomy_loss, dice_score,
                candidate_summary, val_records,
                out_dir / "validation_best_anatomy_metrics.csv",
                out_dir / "validation_best_anatomy_summary.csv",
            )

            # Backward-compatible "best" validation outputs now follow the
            # primary best-anatomy checkpoint.
            pd.DataFrame(val_records).to_csv(
                out_dir / "validation_best_metrics.csv",
                index=False, encoding="utf-8-sig",
            )
            pd.DataFrame([{
                **candidate_summary,
                "best_epoch": epoch,
                "primary_criterion": "anatomy_loss",
                "primary_criterion_value": val_anatomy_loss,
                "best_score": dice_score,  # legacy field kept for compatibility
            }]).to_csv(
                out_dir / "validation_best_summary.csv",
                index=False, encoding="utf-8-sig",
            )
        else:
            no_improve += 1

        # C. Best adaptive-mechanism candidate -- diagnostic, not auto-primary.
        if (
            cfg.get("adaptive_rescue", False)
            and np.isfinite(val_adaptive_loss)
            and val_adaptive_loss < best_adaptive_loss
        ):
            best_adaptive_loss = float(val_adaptive_loss)
            best_adaptive_epoch = int(epoch)
            candidate_summary["epoch"] = int(epoch)
            save_candidate_outputs(
                epoch_ckpt_path, best_adaptive_ckpt,
                "adaptive_loss", val_adaptive_loss, dice_score,
                candidate_summary, val_records,
                out_dir / "validation_best_adaptive_metrics.csv",
                out_dir / "validation_best_adaptive_summary.csv",
            )

        scheduler.step()
        if settings.EARLY_STOP_PATIENCE > 0 and no_improve >= settings.EARLY_STOP_PATIENCE:
            print(
                f"Early stopping at epoch {epoch}; "
                f"best anatomy epoch={best_anatomy_epoch}."
            )
            break

    # -------------------------------------------------------------------------
    # Summarize independent checkpoint candidates.
    # -------------------------------------------------------------------------
    print()
    print("=" * 88)
    print("Checkpoint candidates")
    print("=" * 88)
    print(
        f"Best Dice:     epoch {best_dice_epoch:02d} "
        f"| score {best_dice_score:.6f}"
    )
    print(
        f"Best anatomy:  epoch {best_anatomy_epoch:02d} "
        f"| loss {best_anatomy_loss:.6f}  <-- primary"
    )
    if cfg.get("adaptive_rescue", False) and np.isfinite(best_adaptive_loss):
        print(
            f"Best adaptive: epoch {best_adaptive_epoch:02d} "
            f"| loss {best_adaptive_loss:.6f}"
        )

    candidate_rows = [
        {
            "criterion": "dice_score",
            "epoch": best_dice_epoch,
            "value": best_dice_score,
            "direction": "higher_is_better",
            "checkpoint": str(best_dice_ckpt),
            "primary": 0,
        },
        {
            "criterion": "anatomy_loss",
            "epoch": best_anatomy_epoch,
            "value": best_anatomy_loss,
            "direction": "lower_is_better",
            "checkpoint": str(best_anatomy_ckpt),
            "primary": 1,
        },
    ]
    if cfg.get("adaptive_rescue", False) and np.isfinite(best_adaptive_loss):
        candidate_rows.append({
            "criterion": "adaptive_loss",
            "epoch": best_adaptive_epoch,
            "value": best_adaptive_loss,
            "direction": "lower_is_better",
            "checkpoint": str(best_adaptive_ckpt),
            "primary": 0,
        })
    pd.DataFrame(candidate_rows).to_csv(
        out_dir / "checkpoint_candidates.csv",
        index=False, encoding="utf-8-sig",
    )

    # Finalize the four familiar .pth names WITHOUT reserializing the model.
    # On NTFS this should be metadata-only hard-link creation. Copy fallback is
    # allowed here only because training has already finished.
    if settings.SAVE_EVERY_EPOCH_CHECKPOINT:
        dice_source = epoch_ckpt_dir / f"epoch_{best_dice_epoch:03d}.pth"
        anatomy_source = epoch_ckpt_dir / f"epoch_{best_anatomy_epoch:03d}.pth"
        adaptive_source = (
            epoch_ckpt_dir / f"epoch_{best_adaptive_epoch:03d}.pth"
            if best_adaptive_epoch > 0 else None
        )

        dice_mode = refresh_checkpoint_alias(dice_source, best_dice_ckpt, allow_copy=True)
        anatomy_mode = refresh_checkpoint_alias(anatomy_source, best_anatomy_ckpt, allow_copy=True)
        if adaptive_source is not None:
            adaptive_mode = refresh_checkpoint_alias(adaptive_source, best_adaptive_ckpt, allow_copy=True)
        else:
            adaptive_mode = "not_applicable"
        video_mode = refresh_checkpoint_alias(anatomy_source, best_ckpt, allow_copy=True)
    else:
        # With archival disabled, the in-training hard links already preserve the
        # inode of each selected epoch even though _current_epoch_checkpoint.pth
        # is atomically replaced on subsequent epochs.
        dice_mode = "existing_alias"
        anatomy_mode = "existing_alias"
        adaptive_mode = "existing_alias" if best_adaptive_epoch > 0 else "not_applicable"
        video_mode = refresh_checkpoint_alias(best_anatomy_ckpt, best_ckpt, allow_copy=True)

    print(
        "Checkpoint storage -> "
        f"dice={dice_mode} | anatomy={anatomy_mode} | "
        f"adaptive={adaptive_mode} | video={video_mode}"
    )

    # Standard paired/temporal validation uses the best final-anatomy model.
    primary_ckpt = best_anatomy_ckpt
    ckpt = torch.load(primary_ckpt, map_location=device)
    base_model(video_model).load_state_dict(ckpt["model_state_dict"], strict=True)

    print(
        f"Primary video checkpoint: epoch {best_anatomy_epoch} "
        f"| criterion=anatomy_loss "
        f"| value={best_anatomy_loss:.6f}"
    )

    # Load a SECOND frozen image model only for final paired evaluation, not training.
    image_model = load_image_model(device)

    paired_val, paired_val_summary = evaluate_paired(
        video_model, image_model, val_loader, device, use_amp,
        "Paired validation: image vs temporal vs QC rescue",
        plot_module,
        out_dir / "validation_best_paired_qualitative.png",
    )
    paired_val.to_csv(out_dir / "validation_paired_metrics.csv", index=False, encoding="utf-8-sig")
    paired_val_summary.to_csv(out_dir / "validation_paired_summary.csv", index=False, encoding="utf-8-sig")
    paired_val[paired_val["qc_trigger"] == 1].to_csv(
        out_dir / "validation_qc_triggered_cases.csv", index=False, encoding="utf-8-sig"
    )

    # -------------------------------------------------------------------------
    # Evaluation-only temporal stability analysis on the SAME validation clips.
    # This does not change training or checkpoint selection.
    # -------------------------------------------------------------------------
    temporal_stability_val = None
    temporal_stability_val_summary = None
    if settings.RUN_TEMPORAL_STABILITY_EVAL:
        temporal_stability_val, temporal_stability_val_summary = evaluate_temporal_stability(
            video_model,
            image_model,
            val_loader,
            device,
            use_amp,
            "Temporal stability validation: image vs video",
        )
        temporal_stability_val.to_csv(
            out_dir / "validation_temporal_stability_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        temporal_stability_val_summary.to_csv(
            out_dir / "validation_temporal_stability_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

    test_summary_df = None
    if args.run_test:
        paired_test, test_summary_df = evaluate_paired(
            video_model, image_model, test_loader, device, use_amp,
            "PAIRED TEST: image vs temporal vs QC rescue",
            plot_module,
            out_dir / "test_paired_qualitative.png",
        )
        paired_test.to_csv(out_dir / "test_paired_metrics.csv", index=False, encoding="utf-8-sig")
        test_summary_df.to_csv(out_dir / "test_paired_summary.csv", index=False, encoding="utf-8-sig")

        if settings.RUN_TEMPORAL_STABILITY_EVAL:
            temporal_stability_test, temporal_stability_test_summary = evaluate_temporal_stability(
                video_model,
                image_model,
                test_loader,
                device,
                use_amp,
                "Temporal stability TEST: image vs video",
            )
            temporal_stability_test.to_csv(
                out_dir / "test_temporal_stability_metrics.csv",
                index=False,
                encoding="utf-8-sig",
            )
            temporal_stability_test_summary.to_csv(
                out_dir / "test_temporal_stability_summary.csv",
                index=False,
                encoding="utf-8-sig",
            )

    plot_module.make_all_figures(out_dir)

    def get_method(summary_df, method, metric):
        x = summary_df.loc[summary_df["method"] == method, metric]
        return float(x.iloc[0]) if len(x) else np.nan

    registry = {
        "run_name": run_name,
        "video_run_tag": settings.VIDEO_RUN_TAG,
        "v4_init_checkpoint": str(settings.V5_INIT_CHECKPOINT),
        "image_checkpoint": str(settings.IMAGE_CHECKPOINT),
        "cache_n": len(cache_df),
        # Legacy fields now refer to the primary best-anatomy model.
        "best_epoch": best_anatomy_epoch,
        "best_score": best_dice_score,
        "primary_checkpoint": str(best_anatomy_ckpt),
        "primary_checkpoint_criterion": "anatomy_loss",
        "primary_checkpoint_epoch": best_anatomy_epoch,
        "primary_checkpoint_anatomy_loss": best_anatomy_loss,
        "best_dice_checkpoint": str(best_dice_ckpt),
        "best_dice_epoch": best_dice_epoch,
        "best_dice_score": best_dice_score,
        "best_adaptive_checkpoint": str(best_adaptive_ckpt) if best_adaptive_epoch > 0 else "",
        "best_adaptive_epoch": best_adaptive_epoch,
        "best_adaptive_loss": best_adaptive_loss if np.isfinite(best_adaptive_loss) else np.nan,
        "val_image_wall_dice": get_method(paired_val_summary, "image", "wall_dice"),
        "val_video_wall_dice": get_method(paired_val_summary, "video", "wall_dice"),
        "val_conditional_wall_dice": get_method(paired_val_summary, "conditional", "wall_dice"),
        "val_image_outer_dice": get_method(paired_val_summary, "image", "outer_dice"),
        "val_video_outer_dice": get_method(paired_val_summary, "video", "outer_dice"),
        "val_image_lumen_dice": get_method(paired_val_summary, "image", "lumen_dice"),
        "val_video_lumen_dice": get_method(paired_val_summary, "video", "lumen_dice"),
        "qc_trigger_rate": float(paired_val["qc_trigger"].mean()),
        "test_run": bool(args.run_test),
    }
    if "context_use_mean" in paired_val.columns:
        registry.update({
            "val_adaptive_context_use_mean": float(paired_val["context_use_mean"].mean()),
            "val_adaptive_gate_mean": float(paired_val["gate_mean"].mean()),
            "val_adaptive_alignment_offset_feature_px_mean": float(paired_val["alignment_offset_feature_px_mean"].mean()),
            "val_context_wall_dice": float(paired_val["context_wall_dice"].mean()) if "context_wall_dice" in paired_val.columns else np.nan,
        })

    if temporal_stability_val_summary is not None and len(temporal_stability_val_summary):
        def _stab(method, metric):
            x = temporal_stability_val_summary.loc[
                temporal_stability_val_summary["method"] == method, metric
            ]
            return float(x.iloc[0]) if len(x) else np.nan

        registry.update({
            "val_image_mean_area_rel_jitter": _stab("image", "mean_area_rel_jitter"),
            "val_video_mean_area_rel_jitter": _stab("video", "mean_area_rel_jitter"),
            "val_image_mean_centroid_jitter_px": _stab("image", "mean_centroid_jitter_px"),
            "val_video_mean_centroid_jitter_px": _stab("video", "mean_centroid_jitter_px"),
            "val_image_lumen_outside_rate": _stab("image", "raw_lumen_outside_outer_rate"),
            "val_video_lumen_outside_rate": _stab("video", "raw_lumen_outside_outer_rate"),
        })
    if test_summary_df is not None:
        registry.update({
            "test_image_wall_dice": get_method(test_summary_df, "image", "wall_dice"),
            "test_video_wall_dice": get_method(test_summary_df, "video", "wall_dice"),
            "test_conditional_wall_dice": get_method(test_summary_df, "conditional", "wall_dice"),
        })
    pd.DataFrame([registry]).to_csv(out_dir / "experiment_summary.csv", index=False, encoding="utf-8-sig")
    update_registry(run_name, registry)
    print("Done.")

