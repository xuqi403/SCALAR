#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Nature-style visualization for DINOv3 carotid video segmentation v1.

Designed to be imported by train_video.py. PNG only, flat experiment directory.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def set_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "savefig.dpi": 300,
        "figure.dpi": 140,
    })


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")


def save_fig(fig, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(dtype=float)


# =============================================================================
# 1. Training monitor
# =============================================================================

def update_training_monitor(output_dir):
    output_dir = Path(output_dir)
    path = output_dir / "history.csv"
    if not path.is_file():
        return
    df = pd.read_csv(path)
    if df.empty:
        return

    set_style()
    x = num(df, "epoch")
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6))

    ax = axes[0, 0]
    ax.plot(x, num(df, "train_loss"), marker="o", markersize=2.5, label="Train")
    ax.plot(x, num(df, "val_loss"), marker="o", markersize=2.5, label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and validation loss")
    ax.legend(frameon=False)
    clean_axes(ax)

    ax = axes[0, 1]
    for c, label in [
        ("val_outer_dice", "Outer"),
        ("val_wall_dice", "Wall"),
        ("val_lumen_dice", "Lumen"),
        ("selection_score", "Selection score"),
    ]:
        if c in df.columns:
            ax.plot(x, num(df, c), marker="o", markersize=2.5, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice / score")
    ax.set_ylim(0, 1.02)
    ax.set_title("Validation segmentation")
    ax.legend(frameon=False)
    clean_axes(ax)

    ax = axes[1, 0]
    if "train_adaptive_loss" in df.columns and num(df, "train_adaptive_loss").abs().max() > 0:
        ax.plot(x, num(df, "train_adaptive_loss"), marker="o", markersize=2.5, label="Train adaptive")
        ax.plot(x, num(df, "val_adaptive_loss"), marker="o", markersize=2.5, label="Validation adaptive")
        ax.set_ylabel("Adaptive loss")
        ax.set_title("Adaptive rescue learning")
    else:
        if "train_temporal_loss" in df.columns:
            ax.plot(x, num(df, "train_temporal_loss"), marker="o", markersize=2.5, label="Train temporal")
        if "val_temporal_loss" in df.columns:
            ax.plot(x, num(df, "val_temporal_loss"), marker="o", markersize=2.5, label="Validation temporal")
        ax.set_ylabel("Temporal loss")
        ax.set_title("Temporal regularization")
    ax.set_xlabel("Epoch")
    ax.legend(frameon=False)
    clean_axes(ax)

    ax = axes[1, 1]
    for c, label in [
        ("lr_decoder", "Decoder"),
        ("lr_temporal", "Temporal"),
        ("lr_backbone", "Backbone"),
    ]:
        if c in df.columns:
            ax.plot(x, num(df, c), label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_yscale("log")
    ax.set_title("Learning rate")
    ax.legend(frameon=False)
    clean_axes(ax)

    save_fig(fig, output_dir / "00_training_monitor.png")


# =============================================================================
# 2. Paired image vs video vs conditional-QC plots
# =============================================================================

def plot_paired_dice_summary(summary: pd.DataFrame, out_path: Path):
    if summary.empty:
        return
    set_style()
    methods = [m for m in ["image", "video", "conditional"] if m in set(summary["method"].astype(str))]
    structures = ["outer", "wall", "lumen"]
    x = np.arange(len(structures))
    width = 0.24

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for j, method in enumerate(methods):
        row = summary[summary["method"].astype(str) == method].iloc[0]
        vals = [float(row[f"{s}_dice"]) for s in structures]
        offset = (j - (len(methods) - 1) / 2) * width
        ax.bar(x + offset, vals, width=width, label={
            "image": "Image baseline",
            "video": "Temporal",
            "conditional": "QC rescue",
        }.get(method, method))
    ax.set_xticks(x)
    ax.set_xticklabels(["Outer", "Wall", "Lumen"])
    ax.set_ylabel("Dice coefficient")
    ax.set_ylim(0, 1.02)
    ax.set_title("Paired segmentation performance")
    ax.legend(frameon=False)
    clean_axes(ax)
    save_fig(fig, out_path)


def plot_wall_delta_distribution(df: pd.DataFrame, out_path: Path):
    if df.empty or "delta_wall_dice_video_minus_image" not in df.columns:
        return
    set_style()
    vals1 = num(df, "delta_wall_dice_video_minus_image").dropna().to_numpy()
    vals2 = num(df, "delta_wall_dice_conditional_minus_image").dropna().to_numpy()
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    ax.boxplot([vals1, vals2], labels=["Temporal − image", "QC rescue − image"], showfliers=False)
    rng = np.random.default_rng(42)
    for i, vals in enumerate([vals1, vals2], start=1):
        if len(vals):
            take = vals if len(vals) <= 300 else rng.choice(vals, 300, replace=False)
            ax.scatter(i + rng.normal(0, 0.045, len(take)), take, s=7, alpha=0.22)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_ylabel("Change in Wall Dice")
    ax.set_title("Per-case benefit of temporal context")
    clean_axes(ax)
    save_fig(fig, out_path)


def plot_image_vs_video_wall(df: pd.DataFrame, out_path: Path):
    needed = {"image_wall_dice", "video_wall_dice"}
    if df.empty or not needed.issubset(df.columns):
        return
    x = num(df, "image_wall_dice")
    y = num(df, "video_wall_dice")
    keep = x.notna() & y.notna()
    x = x[keep].to_numpy()
    y = y[keep].to_numpy()
    if len(x) == 0:
        return
    set_style()
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.scatter(x, y, s=9, alpha=0.35)
    ax.plot([0, 1], [0, 1], "--", linewidth=1)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Image-model Wall Dice")
    ax.set_ylabel("Temporal-model Wall Dice")
    ax.set_title("Case-level Wall Dice")
    clean_axes(ax)
    save_fig(fig, out_path)


def plot_qc_subgroups(df: pd.DataFrame, out_path: Path):
    if df.empty or "qc_trigger" not in df.columns:
        return
    set_style()
    rows = []
    for trigger, g in df.groupby("qc_trigger"):
        rows.append({
            "trigger": int(trigger),
            "n": len(g),
            "image": num(g, "image_wall_dice").mean(),
            "video": num(g, "video_wall_dice").mean(),
            "conditional": num(g, "conditional_wall_dice").mean(),
        })
    sm = pd.DataFrame(rows).sort_values("trigger")
    if sm.empty:
        return

    groups = ["QC not triggered" if x == 0 else "QC triggered" for x in sm["trigger"]]
    x = np.arange(len(groups))
    width = 0.24
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for j, (c, label) in enumerate([
        ("image", "Image baseline"),
        ("video", "Temporal"),
        ("conditional", "QC rescue"),
    ]):
        ax.bar(x + (j - 1) * width, sm[c].to_numpy(), width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Mean Wall Dice")
    ax.set_ylim(0, 1.02)
    ax.set_title("Performance in QC-triggered frames")
    ax.legend(frameon=False)
    clean_axes(ax)
    save_fig(fig, out_path)


def plot_qc_reason_counts(df: pd.DataFrame, out_path: Path):
    if df.empty or "qc_reasons" not in df.columns:
        return
    counts = {}
    for text in df.loc[df.get("qc_trigger", 0) == 1, "qc_reasons"].fillna("").astype(str):
        for reason in [x for x in text.split(";") if x]:
            counts[reason] = counts.get(reason, 0) + 1
    if not counts:
        return
    items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    labels = [x[0] for x in items]
    vals = [x[1] for x in items]
    set_style()
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    y = np.arange(len(labels))
    ax.barh(y, vals)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Triggered cases")
    ax.set_title("QC trigger reasons")
    clean_axes(ax)
    save_fig(fig, out_path)


def plot_context_qc(cache_df: pd.DataFrame, out_path: Path):
    if cache_df.empty or "mean_context_weight" not in cache_df.columns:
        return
    vals = num(cache_df, "mean_context_weight").dropna().to_numpy()
    if len(vals) == 0:
        return
    set_style()
    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    ax.hist(vals, bins=20)
    ax.set_xlabel("Mean context reliability weight")
    ax.set_ylabel("Clips")
    ax.set_title("YOLO context reliability")
    clean_axes(ax)
    save_fig(fig, out_path)


# =============================================================================
# 3. Qualitative clip grid
# =============================================================================

def contour_overlay(rgb: np.ndarray, gt: np.ndarray = None, pred: np.ndarray = None) -> np.ndarray:
    out = rgb.copy()
    if gt is not None:
        for mask, color, thickness in [
            (gt > 0, (40, 220, 40), 2),
            (gt == 2, (40, 180, 40), 1),
        ]:
            cs, _ = cv2.findContours((mask.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cs, -1, color, thickness)
    if pred is not None:
        for mask, color, thickness in [
            (pred > 0, (255, 70, 70), 2),
            (pred == 2, (220, 40, 40), 1),
        ]:
            cs, _ = cv2.findContours((mask.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cs, -1, color, thickness)
    return out


def add_label(rgb: np.ndarray, text: str, y: int = 24) -> np.ndarray:
    out = rgb.copy()
    cv2.putText(out, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def add_multiline_label(rgb: np.ndarray, lines, start_y: int = 28, line_gap: int = 26) -> np.ndarray:
    out = rgb.copy()
    if isinstance(lines, str):
        lines = [lines]
    for i, line in enumerate(lines):
        cv2.putText(
            out,
            str(line),
            (8, start_y + i * line_gap),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return out


def format_dice_triplet(metrics):
    if not metrics:
        return "Dice: NA"
    return (
        f"O {metrics.get('outer_dice', np.nan):.3f} | "
        f"W {metrics.get('wall_dice', np.nan):.3f} | "
        f"L {metrics.get('lumen_dice', np.nan):.3f}"
    )


def resize_tile(img: np.ndarray, w: int, h: int) -> np.ndarray:
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def save_clip_case_grid(cases: List[Dict], out_path: Path):
    """
    High-resolution qualitative clip grid.

    Top:
        temporal context frames + reliability weights

    Bottom:
        GT | image baseline | temporal prediction | QC conditional

    The train script should optionally provide:
        image_metrics
        video_metrics
        conditional_metrics
    for displaying case-level Dice values.
    """
    if not cases:
        return

    # High-resolution tiles
    thumb_w, thumb_h = 260, 260
    panel_w, panel_h = 420, 420
    gap = 14
    label_h = 60

    T = len(cases[0]["clip_rgb"])
    case_w = max(T * thumb_w, 4 * panel_w)
    case_h = label_h + thumb_h + gap + panel_h + gap

    canvas = np.full(
        (case_h * len(cases), case_w, 3),
        245,
        np.uint8,
    )

    for ci, case in enumerate(cases):

        y0 = ci * case_h

        title = (
            f"{case.get('name', f'Case {ci+1}')} | "
            f"QC={'YES' if case.get('trigger') else 'NO'}"
        )

        if case.get("reasons"):
            title += f" | {case['reasons']}"

        cv2.putText(
            canvas,
            title[:180],
            (8, y0 + 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )

        # context frames
        for t, rgb in enumerate(case["clip_rgb"]):

            tile = resize_tile(rgb, thumb_w, thumb_h)

            weight = (
                float(case["weights"][t])
                if t < len(case["weights"])
                else np.nan
            )

            tile = add_multiline_label(
                tile,
                [
                    f"t={t}",
                    f"w={weight:.2f}",
                ],
            )

            x = t * thumb_w

            canvas[
                y0 + label_h:y0 + label_h + thumb_h,
                x:x + thumb_w,
            ] = tile

        center = case["clip_rgb"][len(case["clip_rgb"]) // 2]

        gt = case["gt"]
        image_pred = case["image_pred"]
        video_pred = case["video_pred"]
        conditional = case["conditional_pred"]

        panels = [
            (
                contour_overlay(center, gt=gt),
                ["GT", "green contour"],
            ),
            (
                contour_overlay(center, gt=gt, pred=image_pred),
                [
                    "Image prediction",
                    format_dice_triplet(case.get("image_metrics")),
                ],
            ),
            (
                contour_overlay(center, gt=gt, pred=video_pred),
                [
                    "Temporal prediction",
                    format_dice_triplet(case.get("video_metrics")),
                ],
            ),
            (
                contour_overlay(center, gt=gt, pred=conditional),
                [
                    "QC conditional",
                    format_dice_triplet(case.get("conditional_metrics")),
                ],
            ),
        ]

        yy = y0 + label_h + thumb_h + gap

        for j, (panel, labels) in enumerate(panels):

            tile = resize_tile(panel, panel_w, panel_h)
            tile = add_multiline_label(tile, labels)

            x = j * panel_w

            canvas[
                yy:yy + panel_h,
                x:x + panel_w,
            ] = tile

    out_bgr = cv2.cvtColor(
        canvas,
        cv2.COLOR_RGB2BGR,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ok, buf = cv2.imencode(".png", out_bgr)

    if ok:
        buf.tofile(str(out_path))


def save_temporal_epoch_grid(cases: List[Dict], out_path: Path):
    """Native-resolution, paged epoch preview with prediction on EVERY frame.

    Why paging is necessary:
      - Model input is 448 x 448, so each tile is kept at native 448 x 448 resolution.
      - FIXED_PREVIEW_N / RANDOM_PREVIEW_N can be 20. Stacking all 20 cases into one
        ultra-tall PNG makes image viewers shrink the whole figure, so every panel
        looks tiny even though the underlying PNG has many pixels.
      - Therefore only a few cases are placed on each page.

    For each case:
      Row 1: raw context frames with actual frame index and context reliability.
      Row 2: temporal-model prediction for each corresponding frame.
             Neighbor frames show AI contours in red.
             The annotated center frame additionally shows GT in green and
             O/W/L Dice.

    Output:
      - page 1 keeps the original requested filename for compatibility.
      - additional pages are saved as *_page02.png, *_page03.png, ...
    """
    if not cases:
        return

    # Keep each panel at the model's native input resolution.
    tile_w, tile_h = 448, 448

    # A 5-column x 2-row case is already wide. Three cases/page keeps the page
    # close enough to a normal screen aspect ratio that Windows/image viewers
    # do not aggressively shrink it.
    cases_per_page = 3

    tile_gap = 14
    row_gap = 16
    title_h = 58
    case_gap = 28

    T = len(cases[0]["clip_rgb"])
    case_w = T * tile_w + (T - 1) * tile_gap
    case_h = title_h + tile_h + row_gap + tile_h + case_gap

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_pages = int(np.ceil(len(cases) / cases_per_page))

    for page_idx in range(n_pages):
        page_cases = cases[
            page_idx * cases_per_page:
            min((page_idx + 1) * cases_per_page, len(cases))
        ]

        canvas = np.full(
            (case_h * len(page_cases), case_w, 3),
            245,
            np.uint8,
        )

        for local_ci, case in enumerate(page_cases):
            global_ci = page_idx * cases_per_page + local_ci
            y0 = local_ci * case_h
            center_index = T // 2

            frame_indices = np.asarray(
                case.get("frame_indices", np.arange(T)),
                dtype=int,
            )
            center_frame_idx = int(
                case.get(
                    "center_frame_idx",
                    frame_indices[center_index] if len(frame_indices) > center_index else -1,
                )
            )

            title = (
                f"{case.get('name', f'Case {global_ci + 1}')} | "
                f"epoch {case.get('epoch', '')} | "
                f"center frame {center_frame_idx} | "
                f"case {global_ci + 1}/{len(cases)} | page {page_idx + 1}/{n_pages}"
            )
            cv2.putText(
                canvas,
                title[:220],
                (10, y0 + 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.82,
                (20, 20, 20),
                2,
                cv2.LINE_AA,
            )

            # ---------------------------------------------------------
            # Row 1: raw 5-frame clip
            # ---------------------------------------------------------
            raw_y = y0 + title_h

            for t, rgb in enumerate(case["clip_rgb"]):
                tile = resize_tile(rgb, tile_w, tile_h)
                weight = float(case["weights"][t])

                actual_idx = int(frame_indices[t]) if t < len(frame_indices) else t
                rel = actual_idx - center_frame_idx
                rel_text = f"{rel:+d}" if rel != 0 else "0"

                raw_lines = [
                    f"Raw | frame {actual_idx} ({rel_text})",
                    (f"available={'YES' if weight > 0 else 'NO'}"
                     if case.get("adaptive_mode", False) else f"context w={weight:.2f}"),
                ]
                if t == center_index:
                    raw_lines.append("ANNOTATED CENTER")

                tile = add_multiline_label(
                    tile,
                    raw_lines,
                    start_y=34,
                    line_gap=31,
                )

                x = t * (tile_w + tile_gap)
                canvas[
                    raw_y:raw_y + tile_h,
                    x:x + tile_w,
                ] = tile

            # ---------------------------------------------------------
            # Row 2: prediction for every corresponding frame
            # ---------------------------------------------------------
            pred_y = raw_y + tile_h + row_gap
            pred_sequence = case.get("video_pred_sequence")

            if pred_sequence is None:
                pred_sequence = [None] * T
                pred_sequence[center_index] = case.get("video_pred")

            for t, rgb in enumerate(case["clip_rgb"]):
                pred = pred_sequence[t]

                if pred is None:
                    panel = rgb.copy()
                    lines = ["Prediction unavailable"]
                else:
                    if t == center_index:
                        panel = contour_overlay(
                            rgb,
                            gt=case.get("gt"),
                            pred=pred,
                        )
                    else:
                        panel = contour_overlay(
                            rgb,
                            pred=pred,
                        )

                    outer_area = int(np.sum(pred > 0))
                    wall_area = int(np.sum(pred == 1))
                    lumen_area = int(np.sum(pred == 2))

                    actual_idx = int(frame_indices[t]) if t < len(frame_indices) else t
                    lines = [
                        f"AI | frame {actual_idx}",
                        f"Area O/W/L {outer_area/1000:.1f}k/{wall_area/1000:.1f}k/{lumen_area/1000:.1f}k",
                    ]

                    if t == center_index:
                        lines[0] = "CENTER: AI red | GT green"
                        lines.append(
                            format_dice_triplet(
                                case.get("center_metrics")
                            )
                        )

                tile = resize_tile(
                    panel,
                    tile_w,
                    tile_h,
                )
                tile = add_multiline_label(
                    tile,
                    lines,
                    start_y=34,
                    line_gap=31,
                )

                x = t * (tile_w + tile_gap)
                canvas[
                    pred_y:pred_y + tile_h,
                    x:x + tile_w,
                ] = tile

            sep_y = y0 + case_h - max(case_gap // 2, 1)
            cv2.line(
                canvas,
                (0, sep_y),
                (case_w - 1, sep_y),
                (215, 215, 215),
                1,
            )

        out_bgr = cv2.cvtColor(
            canvas,
            cv2.COLOR_RGB2BGR,
        )

        # Keep page 1 at the original filename; later pages get explicit suffixes.
        if page_idx == 0:
            page_path = out_path
        else:
            page_path = out_path.with_name(
                f"{out_path.stem}_page{page_idx + 1:02d}{out_path.suffix}"
            )

        ok, buf = cv2.imencode(
            ".png",
            out_bgr,
        )
        if ok:
            buf.tofile(str(page_path))


def _gate_heatmap_panel(rgb: np.ndarray, gate_map: np.ndarray) -> np.ndarray:
    gate = np.clip(gate_map.astype(np.float32), 0, 1)
    gate = cv2.resize(gate, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    # Show CONTEXT use = 1-G, because this is the clinically intuitive rescue map.
    heat = np.uint8(np.clip((1.0 - gate) * 255.0, 0, 255))
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb, 0.45, heat, 0.55, 0)


def save_adaptive_case_grid(cases: List[Dict], out_path: Path):
    """V5-only diagnostic pages: image vs context vs adaptive + learned gate."""
    cases = [c for c in cases if c.get("gate_map") is not None]
    if not cases:
        return

    panel = 448
    gap = 12
    title_h = 58
    case_gap = 26
    cases_per_page = 2
    cols = 6
    case_w = cols * panel + (cols - 1) * gap
    case_h = title_h + panel + case_gap
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_pages = int(np.ceil(len(cases) / cases_per_page))

    for page_idx in range(n_pages):
        page_cases = cases[page_idx * cases_per_page:(page_idx + 1) * cases_per_page]
        canvas = np.full((case_h * len(page_cases), case_w, 3), 245, np.uint8)

        for ci, case in enumerate(page_cases):
            y0 = ci * case_h
            center = case["clip_rgb"][len(case["clip_rgb"]) // 2]
            gt = case["gt"]
            title = f"{case.get('name', 'case')} | learned adaptive rescue | page {page_idx+1}/{n_pages}"
            cv2.putText(canvas, title[:220], (10, y0 + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (20,20,20), 2, cv2.LINE_AA)

            alpha = case.get("alpha_means") or [np.nan] * len(case["clip_rgb"])
            alpha_text = " | ".join(
                f"{off:+d}:{a:.2f}" if np.isfinite(a) else f"{off:+d}:NA"
                for off, a in zip([-4, -2, 0, 2, 4], alpha)
            )

            panels = [
                (contour_overlay(center, gt=gt), ["GT", "green contour"]),
                (contour_overlay(center, gt=gt, pred=case["image_pred"]), ["Image baseline", format_dice_triplet(case.get("image_metrics"))]),
                (contour_overlay(center, gt=gt, pred=case.get("context_pred")), ["Context only", format_dice_triplet(case.get("context_metrics"))]),
                (contour_overlay(center, gt=gt, pred=case["video_pred"]), ["Adaptive fused", format_dice_triplet(case.get("video_metrics"))]),
                (_gate_heatmap_panel(center, case["gate_map"]), ["Learned context-use map", "hotter = more context"]),
                (center.copy(), ["Mean neighbor alpha", alpha_text]),
            ]

            yy = y0 + title_h
            for j, (img, labels) in enumerate(panels):
                tile = resize_tile(img, panel, panel)
                tile = add_multiline_label(tile, labels, start_y=34, line_gap=31)
                x = j * (panel + gap)
                canvas[yy:yy+panel, x:x+panel] = tile

        out_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        page_path = out_path if page_idx == 0 else out_path.with_name(
            f"{out_path.stem}_page{page_idx+1:02d}{out_path.suffix}"
        )
        ok, buf = cv2.imencode(".png", out_bgr)
        if ok:
            buf.tofile(str(page_path))


def plot_training_curves(output_dir: Path):
    history_path = output_dir / "history.csv"
    if not history_path.is_file():
        return
    df = pd.read_csv(history_path)
    if df.empty:
        return
    set_style()
    x = num(df, "epoch")

    specs = [
        ("07_loss_curve.png", [("train_loss", "Train"), ("val_loss", "Validation")], "Loss", None),
        ("08_dice_curve.png", [("val_outer_dice", "Outer"), ("val_wall_dice", "Wall"), ("val_lumen_dice", "Lumen"), ("selection_score", "Selection score")], "Dice / score", (0, 1.02)),
        ("09_temporal_loss_curve.png", [("train_temporal_loss", "Train temporal"), ("val_temporal_loss", "Validation temporal")], "Temporal loss", None),
        ("19_adaptive_loss_curve.png", [("train_adaptive_loss", "Train adaptive"), ("val_adaptive_loss", "Validation adaptive")], "Adaptive loss", None),
    ]
    for filename, series, ylabel, ylim in specs:
        if filename == "19_adaptive_loss_curve.png":
            present = [num(df, col).abs().max() for col, _ in series if col in df.columns]
            if not present or max(present) <= 0:
                continue
        fig, ax = plt.subplots(figsize=(4.8, 3.5))
        has = False
        for col, label in series:
            if col in df.columns:
                ax.plot(x, num(df, col), marker="o", markersize=2.5, label=label)
                has = True
        if not has:
            plt.close(fig)
            continue
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        if ylim is not None: ax.set_ylim(*ylim)
        ax.legend(frameon=False); clean_axes(ax)
        save_fig(fig, output_dir / filename)

    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    has = False
    for col, label in [("lr_decoder", "Decoder"), ("lr_temporal", "Temporal"), ("lr_backbone", "Backbone")]:
        if col in df.columns:
            ax.plot(x, num(df, col), label=label); has = True
    if has:
        ax.set_xlabel("Epoch"); ax.set_ylabel("Learning rate"); ax.set_yscale("log")
        ax.legend(frameon=False); clean_axes(ax)
        save_fig(fig, output_dir / "10_learning_rate_curve.png")
    else:
        plt.close(fig)


def plot_reconstruction_qc(cache_df: pd.DataFrame, out_path: Path):
    if cache_df.empty or "reconstruction_corr" not in cache_df.columns:
        return
    vals = num(cache_df, "reconstruction_corr").dropna().to_numpy()
    if len(vals) == 0:
        return
    set_style(); fig, ax = plt.subplots(figsize=(4.8, 3.5))
    ax.hist(vals, bins=24)
    ax.set_xlabel("Raw-video vs annotation ROI correlation")
    ax.set_ylabel("Clips")
    ax.set_title("Center-frame reconstruction QC")
    clean_axes(ax); save_fig(fig, out_path)



# =============================================================================
# 3B. Temporal stability figures
# =============================================================================

def plot_temporal_area_jitter(summary: pd.DataFrame, out_path: Path):
    if summary.empty or "method" not in summary.columns:
        return
    metrics = [
        ("outer_area_rel_jitter", "Outer"),
        ("wall_area_rel_jitter", "Wall"),
        ("lumen_area_rel_jitter", "Lumen"),
    ]
    methods = [m for m in ["image", "video"] if m in set(summary["method"].astype(str))]
    if not methods:
        return

    values = []
    for method in methods:
        row = summary.loc[summary["method"].astype(str) == method].iloc[0]
        values.append([float(row.get(col, np.nan)) for col, _ in metrics])

    set_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    x = np.arange(len(metrics))
    width = 0.36
    offsets = np.linspace(-width / 2, width / 2, len(methods))

    for off, method, vals in zip(offsets, methods, values):
        ax.bar(x + off, vals, width=width, label=method.capitalize())

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylabel("Normalized second-difference")
    ax.set_title("Temporal area jitter")
    ax.legend(frameon=False)
    clean_axes(ax)
    save_fig(fig, out_path)


def plot_temporal_centroid_jitter(summary: pd.DataFrame, out_path: Path):
    if summary.empty or "method" not in summary.columns:
        return
    metrics = [
        ("outer_centroid_jitter_px", "Outer"),
        ("lumen_centroid_jitter_px", "Lumen"),
    ]
    methods = [m for m in ["image", "video"] if m in set(summary["method"].astype(str))]
    if not methods:
        return

    values = []
    for method in methods:
        row = summary.loc[summary["method"].astype(str) == method].iloc[0]
        values.append([float(row.get(col, np.nan)) for col, _ in metrics])

    set_style()
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    x = np.arange(len(metrics))
    width = 0.36
    offsets = np.linspace(-width / 2, width / 2, len(methods))

    for off, method, vals in zip(offsets, methods, values):
        ax.bar(x + off, vals, width=width, label=method.capitalize())

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylabel("Second-difference (px)")
    ax.set_title("Temporal centroid jitter")
    ax.legend(frameon=False)
    clean_axes(ax)
    save_fig(fig, out_path)


def plot_temporal_topology(summary: pd.DataFrame, out_path: Path):
    if summary.empty or "method" not in summary.columns:
        return
    metrics = [
        ("raw_outer_multi_component_rate", "Outer multi-component"),
        ("raw_lumen_multi_component_rate", "Lumen multi-component"),
        ("raw_lumen_outside_outer_rate", "Lumen outside outer"),
    ]
    methods = [m for m in ["image", "video"] if m in set(summary["method"].astype(str))]
    if not methods:
        return

    values = []
    for method in methods:
        row = summary.loc[summary["method"].astype(str) == method].iloc[0]
        values.append([float(row.get(col, np.nan)) for col, _ in metrics])

    set_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(metrics))
    width = 0.36
    offsets = np.linspace(-width / 2, width / 2, len(methods))

    for off, method, vals in zip(offsets, methods, values):
        ax.bar(x + off, vals, width=width, label=method.capitalize())

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics], rotation=15, ha="right")
    ax.set_ylabel("Frame-level rate")
    ax.set_title("Raw topology failure rate")
    ax.legend(frameon=False)
    clean_axes(ax)
    save_fig(fig, out_path)


def plot_temporal_jitter_delta(metrics_df: pd.DataFrame, out_path: Path):
    col = "delta_mean_area_rel_jitter_video_minus_image"
    if metrics_df.empty or col not in metrics_df.columns:
        return
    vals = pd.to_numeric(metrics_df[col], errors="coerce").dropna().to_numpy()
    if len(vals) == 0:
        return

    set_style()
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.hist(vals, bins=28, alpha=0.75)
    ax.axvline(0, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Video - image mean area jitter")
    ax.set_ylabel("Cases")
    ax.set_title("Per-case change in temporal jitter")
    clean_axes(ax)
    save_fig(fig, out_path)


def plot_adaptive_gain_vs_image_quality(df: pd.DataFrame, out_path: Path):
    needed = {"image_wall_dice", "delta_wall_dice_video_minus_image"}
    if df.empty or not needed.issubset(df.columns):
        return
    x = num(df, "image_wall_dice")
    y = num(df, "delta_wall_dice_video_minus_image")
    keep = x.notna() & y.notna()
    x, y = x[keep].to_numpy(), y[keep].to_numpy()
    if len(x) == 0:
        return
    set_style()
    fig, ax = plt.subplots(figsize=(5.2, 4.1))
    ax.scatter(x, y, s=9, alpha=0.30)
    ax.axhline(0, linestyle="--", linewidth=1)
    # Quantile-bin mean is descriptive only and avoids assuming linearity.
    if len(x) >= 20:
        q = np.quantile(x, np.linspace(0, 1, 6))
        centers, means = [], []
        for a, b in zip(q[:-1], q[1:]):
            sel = (x >= a) & (x <= b if b == q[-1] else x < b)
            if sel.any():
                centers.append(float(np.mean(x[sel])))
                means.append(float(np.mean(y[sel])))
        ax.plot(centers, means, marker="o", linewidth=1.5, label="Quintile mean")
        ax.legend(frameon=False)
    ax.set_xlabel("Image-model Wall Dice")
    ax.set_ylabel("Adaptive - image Wall Dice")
    ax.set_title("Where adaptive context adds value")
    clean_axes(ax)
    save_fig(fig, out_path)


def plot_adaptive_gate_response(df: pd.DataFrame, out_path: Path):
    needed = {"context_wall_advantage_vs_image", "context_use_mean"}
    if df.empty or not needed.issubset(df.columns):
        return
    x = num(df, "context_wall_advantage_vs_image")
    y = num(df, "context_use_mean")
    keep = x.notna() & y.notna()
    x, y = x[keep].to_numpy(), y[keep].to_numpy()
    if len(x) == 0:
        return
    set_style()
    fig, ax = plt.subplots(figsize=(5.2, 4.1))
    ax.scatter(x, y, s=9, alpha=0.30)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Context-only - image Wall Dice")
    ax.set_ylabel("Learned context use (1 - gate)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Gate response to context advantage")
    clean_axes(ax)
    save_fig(fig, out_path)


def plot_adaptive_gain_by_quartile(df: pd.DataFrame, out_path: Path):
    needed = {"image_wall_dice", "video_wall_dice"}
    if df.empty or not needed.issubset(df.columns) or len(df) < 8:
        return
    d = df[["image_wall_dice", "video_wall_dice"]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) < 8:
        return
    try:
        d["group"] = pd.qcut(d["image_wall_dice"], 4, labels=["Worst 25%", "Q2", "Q3", "Best 25%"], duplicates="drop")
    except Exception:
        return
    sm = d.groupby("group", observed=True)[["image_wall_dice", "video_wall_dice"]].mean()
    set_style()
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    x = np.arange(len(sm))
    width = 0.36
    ax.bar(x - width/2, sm["image_wall_dice"].to_numpy(), width=width, label="Image")
    ax.bar(x + width/2, sm["video_wall_dice"].to_numpy(), width=width, label="Adaptive")
    ax.set_xticks(x)
    ax.set_xticklabels(sm.index.astype(str))
    ax.set_ylabel("Mean Wall Dice")
    ax.set_ylim(0, 1.02)
    ax.set_title("Adaptive rescue by image difficulty")
    ax.legend(frameon=False)
    clean_axes(ax)
    save_fig(fig, out_path)


# =============================================================================
# 4. Public entry point
# =============================================================================

def make_all_figures(output_dir):
    output_dir = Path(output_dir)
    set_style()
    update_training_monitor(output_dir)

    paired_summary_path = output_dir / "validation_paired_summary.csv"
    paired_metrics_path = output_dir / "validation_paired_metrics.csv"
    if paired_summary_path.is_file():
        summary = pd.read_csv(paired_summary_path)
        plot_paired_dice_summary(summary, output_dir / "01_paired_dice_summary.png")
    if paired_metrics_path.is_file():
        df = pd.read_csv(paired_metrics_path)
        plot_wall_delta_distribution(df, output_dir / "02_wall_dice_improvement_distribution.png")
        plot_image_vs_video_wall(df, output_dir / "03_image_vs_video_wall_dice.png")
        plot_qc_subgroups(df, output_dir / "04_qc_subgroup_wall_dice.png")
        plot_qc_reason_counts(df, output_dir / "05_qc_trigger_reasons.png")
        if "context_use_mean" in df.columns:
            plot_adaptive_gain_vs_image_quality(df, output_dir / "16_adaptive_gain_vs_image_quality.png")
            plot_adaptive_gate_response(df, output_dir / "17_adaptive_gate_response.png")
            plot_adaptive_gain_by_quartile(df, output_dir / "18_adaptive_rescue_by_difficulty.png")

    cache_path = output_dir / "video_clip_manifest_used.csv"
    if cache_path.is_file():
        cache_df = pd.read_csv(cache_path, low_memory=False)
        plot_context_qc(cache_df, output_dir / "06_context_reliability_distribution.png")
        plot_reconstruction_qc(cache_df, output_dir / "11_center_reconstruction_correlation.png")

    stability_summary_path = output_dir / "validation_temporal_stability_summary.csv"
    stability_metrics_path = output_dir / "validation_temporal_stability_metrics.csv"
    if stability_summary_path.is_file():
        stability_summary = pd.read_csv(stability_summary_path)
        plot_temporal_area_jitter(
            stability_summary,
            output_dir / "12_temporal_area_jitter.png",
        )
        plot_temporal_centroid_jitter(
            stability_summary,
            output_dir / "13_temporal_centroid_jitter.png",
        )
        plot_temporal_topology(
            stability_summary,
            output_dir / "14_temporal_topology_failures.png",
        )
    if stability_metrics_path.is_file():
        stability_metrics = pd.read_csv(stability_metrics_path, low_memory=False)
        plot_temporal_jitter_delta(
            stability_metrics,
            output_dir / "15_temporal_jitter_delta_distribution.png",
        )

    plot_training_curves(output_dir)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("output_dir")
    args = p.parse_args()
    make_all_figures(args.output_dir)
