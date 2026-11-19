"""Faithful image model optimization + private-manifest interface."""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None
from config import (MODEL_CONFIG, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS,
                    EPOCHS, SEED, WEIGHT_DECAY, PRED_THRESHOLD,
                    CHECKPOINT_SCORE_WEIGHTS)
from dataset import CarotidDataset, read_manifest
from model import (load_dinov3, build_model, set_backbone_policy,
                   apply_backbone_train_mode)
from losses import build_loss
from postprocess import decode_outputs
from metrics import one_metrics, summarize


def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def progress(iterable, desc: str):
    return tqdm(iterable, desc=desc, dynamic_ncols=True) if tqdm is not None else iterable
def torch_dice(pred: torch.Tensor, gt: torch.Tensor) -> float:
    pred, gt = pred.bool(), gt.bool()
    inter = (pred & gt).sum().float()
    return float((2 * inter + 1e-6) / (pred.sum() + gt.sum() + 1e-6))



def train_epoch(model, loader, optimizer, criterion, device, scaler, use_amp, epoch, total_epochs, stage, backbone_policy, backbone_trainable):
    model.train()
    apply_backbone_train_mode(model, backbone_policy, backbone_trainable)
    losses = []
    bar = progress(loader, f"Train {epoch:02d}/{total_epochs} [{stage}]")
    for batch in bar:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            out = model(x)
            loss = criterion(out, y)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if tqdm is not None:
            with torch.no_grad():
                _, pred, _ = decode_outputs(out, topology=False)
                bar.set_postfix(
                    loss=f"{np.mean(losses[-30:]):.4f}",
                    outer=f"{torch_dice(pred > 0, y > 0):.3f}",
                    wall=f"{torch_dice(pred == 1, y == 1):.3f}",
                    lumen=f"{torch_dice(pred == 2, y == 2):.3f}",
                    lr=f"{optimizer.param_groups[1]['lr']:.1e}",
                )
    return float(np.mean(losses))



@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp, phase="Val", threshold=PRED_THRESHOLD):
    model.eval()
    losses, records = [], []
    for batch in progress(loader, phase):
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            out = model(x)
            loss = criterion(out, y)
        losses.append(float(loss.detach().cpu()))
        raw, post, top_stats = decode_outputs(out, threshold=threshold, topology=True)
        raw = raw.cpu().numpy()
        post = post.cpu().numpy()
        gt = y.cpu().numpy()
        for i in range(len(gt)):
            rec = one_metrics(post[i], gt[i])
            rec.update(one_metrics(raw[i], gt[i], prefix="raw_"))
            rec.update(top_stats[i])
            # No patient, file path or video identifier is exported.
            rec["sample_index"] = len(records)
            records.append(rec)
    summary = summarize(records)
    summary["loss"] = float(np.mean(losses))
    return summary, records



def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="SCALAR single-frame DINOv3 segmentation training")
    parser.add_argument("--manifest", required=True, help="Private CSV with fixed train/val/test assignments")
    parser.add_argument("--dino-repo", required=True, help="Local official DINOv3 repository")
    parser.add_argument("--dino-weights", required=True, help="Carotid SSL DINOv3 backbone weights")
    parser.add_argument("--output-dir", required=True, help="Private output directory outside Git")
    parser.add_argument("--init-checkpoint", default="", help="Optional compatible image-model warm-start; default clean retraining")
    parser.add_argument("--run-test", action="store_true", help="Evaluate independent test split once, if authorized")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--prediction-threshold", type=float, default=PRED_THRESHOLD)
    parser.add_argument("--lr-decoder", type=float, default=None)
    parser.add_argument("--lr-backbone", type=float, default=None)
    parser.add_argument("--disable-group-split-check", action="store_true",
                        help="Only for data without trustworthy group_id; not recommended")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.epochs < 1 or args.image_size < 32 or args.batch_size < 1:
        raise ValueError("epochs>=1, image-size>=32, and batch-size>=1 required")
    seed_everything(args.seed)
    cfg = dict(MODEL_CONFIG)
    if args.lr_decoder is not None:
        cfg["lr_decoder"] = args.lr_decoder
    if args.lr_backbone is not None:
        cfg["lr_backbone"] = args.lr_backbone
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_table = read_manifest(args.manifest, check_group_split=not args.disable_group_split_check)
    training = frame_table[frame_table["split"].eq("train")].copy()
    validation = frame_table[frame_table["split"].eq("val")].copy()
    heldout = frame_table[frame_table["split"].eq("test")].copy()
    # Explicitly remove private path/ID fields before reporting aggregate info.
    print(f"Split sizes (frames): train={len(training)}, val={len(validation)}, test={len(heldout)}")
    if args.run_test and heldout.empty:
        raise ValueError("--run-test requested but no test examples in manifest")
    train_ds = CarotidDataset(training, True, cfg["augmentation"], args.image_size)
    val_ds = CarotidDataset(validation, False, "none", args.image_size)
    test_ds = CarotidDataset(heldout, False, "none", args.image_size) if args.run_test else None
    generator = torch.Generator().manual_seed(args.seed)
    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                         pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker)
    train_loader = DataLoader(train_ds, shuffle=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs) if test_ds else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    backbone = load_dinov3(args.dino_repo, args.dino_weights)
    bm = build_model(backbone, cfg).to(device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        state = checkpoint.get("model_state_dict", checkpoint)
        bm.load_state_dict(state, strict=True)
        print("Loaded compatible segmentation warm-start checkpoint")
    else:
        print("Clean image-model retraining from Carotid SSL backbone")

    set_backbone_policy(bm, cfg["unfreeze"], trainable=False)
    model = nn.DataParallel(bm) if device.type == "cuda" and torch.cuda.device_count() > 1 else bm
    bb_params = list(bm.backbone.parameters())
    dec_params = [p for name, p in bm.named_parameters() if not name.startswith("backbone.")]
    optimizer = torch.optim.AdamW([
        {"params": bb_params, "lr": cfg["lr_backbone"]},
        {"params": dec_params, "lr": cfg["lr_decoder"]},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = build_loss(cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Save no patient IDs, image paths, video names, or clinical spreadsheets.
    config_public = {
        "experiment": "D_robust",
        "config": cfg,
        "split_frames": {"train": len(training), "val": len(validation), "test": len(heldout)},
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "seed": args.seed,
        "weight_decay": args.weight_decay,
        "prediction_threshold": args.prediction_threshold,
        "checkpoint_score_weights": CHECKPOINT_SCORE_WEIGHTS,
        "input_split_policy": "preassigned_private_manifest",
        "torch_version": torch.__version__,
    }
    (out_dir / "run_config.json").write_text(json.dumps(config_public, indent=2, default=str), encoding="utf-8")

    history, best_score, no_improve = [], -float("inf"), 0
    best_file = out_dir / "best_checkpoint.pth"
    for epoch in range(1, args.epochs + 1):
        trainable = epoch > cfg["freeze_epochs"]
        set_backbone_policy(model, cfg["unfreeze"], trainable=trainable)
        stage = "decoder_warmup" if not trainable else ("full_finetune" if cfg["unfreeze"] == "all" else "last4_finetune")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler,
                                 use_amp, epoch, args.epochs, stage, cfg["unfreeze"], trainable)
        val_summary, val_records = evaluate(model, val_loader, criterion, device,
                                            use_amp, phase=f"Val {epoch:02d}", threshold=args.prediction_threshold)
        score = (CHECKPOINT_SCORE_WEIGHTS["outer"] * val_summary["outer_dice_mean"]
                 + CHECKPOINT_SCORE_WEIGHTS["wall"] * val_summary["wall_dice_mean"]
                 + CHECKPOINT_SCORE_WEIGHTS["lumen"] * val_summary["lumen_dice_mean"])
        row = {
            "epoch": epoch, "stage": stage, "train_loss": train_loss,
            "val_loss": val_summary["loss"],
            "val_outer_dice": val_summary["outer_dice_mean"],
            "val_wall_dice": val_summary["wall_dice_mean"],
            "val_lumen_dice": val_summary["lumen_dice_mean"],
            "val_outer_hd95_px": val_summary.get("outer_hd95_px_mean", float("nan")),
            "val_lumen_hd95_px": val_summary.get("lumen_hd95_px_mean", float("nan")),
            "selection_score": score,
            "lr_backbone": optimizer.param_groups[0]["lr"],
            "lr_decoder": optimizer.param_groups[1]["lr"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        print(f"Epoch {epoch:02d}/{args.epochs} | {stage} | val Outer/Wall/Lumen "
              f"{row['val_outer_dice']:.4f}/{row['val_wall_dice']:.4f}/{row['val_lumen_dice']:.4f} "
              f"| score {score:.4f}")
        if score > best_score:
            best_score, no_improve = score, 0
            torch.save({
                "model_state_dict": bm.state_dict(),
                "experiment": "D_robust",
                "config": cfg,
                "epoch": epoch,
                "best_score": float(best_score),
                "pretraining": "carotid_ssl_dinov3_vitb16",
            }, best_file)
            pd.DataFrame([val_summary]).to_csv(out_dir / "validation_best_summary.csv", index=False)
            pd.DataFrame(val_records).to_csv(out_dir / "validation_best_metrics.csv", index=False)
        else:
            no_improve += 1
        scheduler.step()
        if cfg["early_stop"] and no_improve >= cfg["early_stop"]:
            print(f"Early stopping at epoch {epoch}")
            break
    if not best_file.is_file():
        raise RuntimeError("No checkpoint saved")
    print(f"Best validation score: {best_score:.6f}")
    if args.run_test:
        # Only run with an explicitly approved held-out test set.
        ckpt = torch.load(best_file, map_location=device, weights_only=False)
        bm.load_state_dict(ckpt["model_state_dict"], strict=True)
        test_summary, test_records = evaluate(model, test_loader, criterion, device,
                                              use_amp, phase="Test", threshold=args.prediction_threshold)
        pd.DataFrame([test_summary]).to_csv(out_dir / "test_summary.csv", index=False)
        pd.DataFrame(test_records).to_csv(out_dir / "test_metrics.csv", index=False)
    try:
        from plot import update_training_monitor, make_training_figures
        update_training_monitor(out_dir)
        make_training_figures(out_dir)
    except ImportError:
        print("Optional plotting dependencies missing; training logs remain available")
    print(f"Outputs: {out_dir}")

if __name__ == "__main__":
    main()
