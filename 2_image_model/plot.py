#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compact plotting helpers for carotid_dinov3_train_v9_balanced.py.
PNG only; every figure is saved directly in the experiment root.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "figure.dpi": 150,
        "savefig.dpi": 220,
    })


def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def numeric(df, col):
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(dtype=float)


def update_training_monitor(output_dir):
    output_dir = Path(output_dir)
    path = output_dir / "history.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    style()
    x = numeric(df, "epoch")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    ax = axes[0, 0]
    ax.plot(x, numeric(df, "train_loss"), label="Train")
    ax.plot(x, numeric(df, "val_loss"), label="Validation")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend(frameon=False); clean(ax)

    ax = axes[0, 1]
    for c, label in [("val_outer_dice", "Outer"), ("val_wall_dice", "Wall"), ("val_lumen_dice", "Lumen")]:
        ax.plot(x, numeric(df, c), label=label)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Dice"); ax.set_ylim(0, 1.02); ax.legend(frameon=False); clean(ax)

    ax = axes[1, 0]
    for c, label in [("val_outer_hd95_px", "Outer"), ("val_lumen_hd95_px", "Lumen")]:
        if c in df.columns:
            ax.plot(x, numeric(df, c), label=label)
    ax.set_xlabel("Epoch"); ax.set_ylabel("HD95 (px)"); ax.legend(frameon=False); clean(ax)

    ax = axes[1, 1]
    ax.plot(x, numeric(df, "lr_decoder"), label="Decoder")
    ax.plot(x, numeric(df, "lr_backbone"), label="Backbone")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning rate"); ax.set_yscale("log"); ax.legend(frameon=False); clean(ax)

    save(fig, output_dir / "00_training_monitor.png")


def simple_curve(df, columns, labels, ylabel, out_path, ylim=None, logy=False):
    style()
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    x = numeric(df, "epoch")
    for c, label in zip(columns, labels):
        if c in df.columns:
            ax.plot(x, numeric(df, c), label=label)
    ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
    if ylim: ax.set_ylim(*ylim)
    if logy: ax.set_yscale("log")
    ax.legend(frameon=False); clean(ax)
    save(fig, out_path)


def make_training_figures(output_dir):
    output_dir = Path(output_dir)
    path = output_dir / "history.csv"
    if not path.exists(): return
    df = pd.read_csv(path)
    simple_curve(df, ["train_loss", "val_loss"], ["Train", "Validation"], "Loss", output_dir / "01_loss_curve.png")
    dice_cols = ["val_outer_dice", "val_wall_dice", "val_lumen_dice"]
    dice_labels = ["Outer", "Wall", "Lumen"]
    if "selection_score" in df.columns:
        dice_cols.append("selection_score"); dice_labels.append("Selection score")
    simple_curve(df, dice_cols, dice_labels, "Dice", output_dir / "02_dice_curve.png", (0, 1.02))
    simple_curve(df, ["val_outer_hd95_px", "val_lumen_hd95_px"], ["Outer", "Lumen"], "HD95 (px)", output_dir / "03_hd95_curve.png")
    simple_curve(df, ["lr_decoder", "lr_backbone"], ["Decoder", "Backbone"], "Learning rate", output_dir / "04_learning_rate_curve.png", logy=True)


def boxplot_three(df, cols, labels, ylabel, out_path, ylim=None):
    style()
    data = [numeric(df, c).dropna().to_numpy() for c in cols]
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    bp = ax.boxplot(data, labels=labels, widths=0.55, showfliers=False, patch_artist=False)
    rng = np.random.default_rng(42)
    for i, vals in enumerate(data, start=1):
        if len(vals):
            take = vals if len(vals) <= 250 else rng.choice(vals, 250, replace=False)
            ax.scatter(i + rng.normal(0, 0.045, len(take)), take, s=7, alpha=0.25)
    ax.set_ylabel(ylabel)
    if ylim: ax.set_ylim(*ylim)
    clean(ax); save(fig, out_path)


def area_agreement(df, structure, out_path):
    gcol, pcol = f"gt_{structure}_area_px", f"pred_{structure}_area_px"
    if gcol not in df.columns or pcol not in df.columns: return
    gt, pred = numeric(df, gcol), numeric(df, pcol)
    keep = gt.notna() & pred.notna()
    gt, pred = gt[keep].to_numpy(), pred[keep].to_numpy()
    if len(gt) < 2: return
    style(); fig, ax = plt.subplots(figsize=(4.5, 4.2))
    ax.scatter(gt, pred, s=10, alpha=0.35)
    lo, hi = min(gt.min(), pred.min()), max(gt.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], "--", linewidth=1)
    r = np.corrcoef(gt, pred)[0, 1]
    ax.text(0.04, 0.95, f"r = {r:.3f}\nn = {len(gt)}", transform=ax.transAxes, va="top")
    ax.set_xlabel("Ground truth area (px)"); ax.set_ylabel("Predicted area (px)"); clean(ax)
    save(fig, out_path)


def bland_altman(df, structure, out_path):
    gcol, pcol = f"gt_{structure}_area_px", f"pred_{structure}_area_px"
    if gcol not in df.columns or pcol not in df.columns: return
    gt, pred = numeric(df, gcol), numeric(df, pcol)
    keep = gt.notna() & pred.notna()
    gt, pred = gt[keep].to_numpy(), pred[keep].to_numpy()
    if len(gt) < 2: return
    mean = (gt + pred) / 2
    diff = pred - gt
    bias, sd = diff.mean(), diff.std(ddof=1)
    style(); fig, ax = plt.subplots(figsize=(5.0, 3.8))
    ax.scatter(mean, diff, s=10, alpha=0.35)
    ax.axhline(bias, linewidth=1)
    ax.axhline(bias + 1.96 * sd, linestyle="--", linewidth=1)
    ax.axhline(bias - 1.96 * sd, linestyle="--", linewidth=1)
    ax.set_xlabel("Mean area (px)"); ax.set_ylabel("Prediction − GT (px)"); clean(ax)
    save(fig, out_path)


def raw_post_dice(df, out_path):
    needed = ["outer_dice", "wall_dice", "lumen_dice", "raw_outer_dice", "raw_wall_dice", "raw_lumen_dice"]
    if not all(c in df.columns for c in needed): return
    style(); fig, ax = plt.subplots(figsize=(5.4, 3.8))
    x = np.arange(3)
    raw = [df[f"raw_{s}_dice"].mean() for s in ["outer", "wall", "lumen"]]
    post = [df[f"{s}_dice"].mean() for s in ["outer", "wall", "lumen"]]
    w = 0.34
    ax.bar(x - w/2, raw, w, label="Raw")
    ax.bar(x + w/2, post, w, label="Topology-cleaned")
    ax.set_xticks(x, ["Outer", "Wall", "Lumen"]); ax.set_ylabel("Mean Dice"); ax.set_ylim(0, 1.02)
    ax.legend(frameon=False); clean(ax); save(fig, out_path)


def topology_fraction(df, out_path):
    if "topology_changed_fraction" not in df.columns: return
    vals = numeric(df, "topology_changed_fraction").dropna().to_numpy() * 100
    style(); fig, ax = plt.subplots(figsize=(5.0, 3.8))
    ax.hist(vals, bins=30)
    ax.set_xlabel("Pixels changed by topology cleanup (%)"); ax.set_ylabel("Frames")
    if len(vals): ax.axvline(np.median(vals), linestyle="--", linewidth=1)
    clean(ax); save(fig, out_path)


def make_best_validation_figures(output_dir):
    output_dir = Path(output_dir)
    path = output_dir / "validation_best_metrics.csv"
    if not path.exists(): return
    df = pd.read_csv(path)
    boxplot_three(df, ["outer_dice", "wall_dice", "lumen_dice"], ["Outer", "Wall", "Lumen"], "Dice", output_dir / "V01_validation_best_dice_distribution.png", (0, 1.02))
    boxplot_three(df, ["outer_hd95_px", "lumen_hd95_px"], ["Outer", "Lumen"], "HD95 (px)", output_dir / "V02_validation_best_hd95_distribution.png")
    boxplot_three(df, ["outer_area_rel_error", "wall_area_rel_error", "lumen_area_rel_error"], ["Outer", "Wall", "Lumen"], "Relative area error", output_dir / "V03_validation_best_area_error_distribution.png")
    area_agreement(df, "outer", output_dir / "V04_validation_outer_area_agreement.png")
    area_agreement(df, "lumen", output_dir / "V05_validation_lumen_area_agreement.png")
    area_agreement(df, "wall", output_dir / "V06_validation_wall_area_agreement.png")
    bland_altman(df, "outer", output_dir / "V07_validation_outer_area_bland_altman.png")
    bland_altman(df, "lumen", output_dir / "V08_validation_lumen_area_bland_altman.png")
    bland_altman(df, "wall", output_dir / "V09_validation_wall_area_bland_altman.png")
    raw_post_dice(df, output_dir / "V10_validation_topology_raw_vs_post_dice.png")
    topology_fraction(df, output_dir / "V11_validation_topology_changed_fraction.png")


def test_dice_summary(df, out_path):
    style(); fig, ax = plt.subplots(figsize=(5.2, 3.8))
    labels = ["Outer", "Wall", "Lumen"]
    means = [numeric(df, f"{x}_dice").mean() for x in ["outer", "wall", "lumen"]]
    stds = [numeric(df, f"{x}_dice").std() for x in ["outer", "wall", "lumen"]]
    x = np.arange(3)
    ax.bar(x, means, yerr=stds, capsize=3)
    ax.set_xticks(x, labels); ax.set_ylabel("Dice"); ax.set_ylim(0, 1.02); clean(ax)
    save(fig, out_path)


def make_test_figures(output_dir):
    output_dir = Path(output_dir)
    path = output_dir / "test_metrics.csv"
    if not path.exists(): return
    df = pd.read_csv(path)
    boxplot_three(df, ["outer_dice", "wall_dice", "lumen_dice"], ["Outer", "Wall", "Lumen"], "Dice", output_dir / "05_test_dice_distribution.png", (0, 1.02))
    test_dice_summary(df, output_dir / "06_test_dice_summary.png")
    boxplot_three(df, ["outer_hd95_px", "lumen_hd95_px"], ["Outer", "Lumen"], "HD95 (px)", output_dir / "07_test_hd95_distribution.png")
    boxplot_three(df, ["outer_area_rel_error", "wall_area_rel_error", "lumen_area_rel_error"], ["Outer", "Wall", "Lumen"], "Relative area error", output_dir / "08_test_area_error_distribution.png")
    area_agreement(df, "outer", output_dir / "09_outer_area_agreement.png")
    area_agreement(df, "lumen", output_dir / "10_lumen_area_agreement.png")
    area_agreement(df, "wall", output_dir / "11_wall_area_agreement.png")
    bland_altman(df, "outer", output_dir / "12_outer_area_bland_altman.png")
    bland_altman(df, "lumen", output_dir / "13_lumen_area_bland_altman.png")
    bland_altman(df, "wall", output_dir / "14_wall_area_bland_altman.png")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("output_dir")
    a = p.parse_args()
    update_training_monitor(a.output_dir)
    make_training_figures(a.output_dir)
    make_best_validation_figures(a.output_dir)
    make_test_figures(a.output_dir)
