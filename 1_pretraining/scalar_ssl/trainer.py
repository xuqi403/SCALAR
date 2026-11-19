"""SSL optimizer, schedules and one-epoch training/evaluation (methodology preserved)."""
from __future__ import annotations
import math
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None
from . import settings
from .settings import *
from . import losses as _ssl_mod_losses
from . import model as _ssl_mod_model
from . import utils as _ssl_mod_utils


def lr_multiplier(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return max(0.001, (step + 1) / max(warmup_steps, 1))
    t = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(t, 0.0), 1.0)))


def split_ssl_manifest(df: pd.DataFrame, seed: int=SEED) -> Tuple[pd.DataFrame, pd.DataFrame]:
    vids = np.asarray(sorted(df['video_uid'].astype(str).unique().tolist()))
    rng = np.random.default_rng(seed)
    rng.shuffle(vids)
    n_val = max(1, int(round(len(vids) * SSL_VAL_VIDEO_FRACTION))) if len(vids) > 20 else max(1, len(vids) // 10)
    val_vids = set(vids[:n_val])
    val = df[df['video_uid'].astype(str).isin(val_vids)].copy()
    train = df[~df['video_uid'].astype(str).isin(val_vids)].copy()
    return (train.reset_index(drop=True), val.reset_index(drop=True))


def optimizer_groups(student: _ssl_mod_model.SSLStudent, context_head: _ssl_mod_model.CrossSectionContextHead):
    backbone_params = [p for p in student.backbone.parameters() if p.requires_grad]
    head_params = list(student.projector.parameters()) + list(student.predictor.parameters()) + list(context_head.parameters())
    return [{'params': backbone_params, 'lr': LR_BACKBONE}, {'params': head_params, 'lr': LR_HEAD}]


def run_ssl_epoch(student: _ssl_mod_model.SSLStudent, teacher: _ssl_mod_model.SSLTeacher, context_head: _ssl_mod_model.CrossSectionContextHead, loader: DataLoader, optimizer: Optional[torch.optim.Optimizer], scaler: torch.amp.GradScaler, device: torch.device, epoch: int, total_epochs: int, global_step: int, total_steps: int, train: bool) -> Tuple[Dict[str, float], int]:
    if train:
        _ssl_mod_model.apply_student_train_mode(student)
        context_head.train()
    else:
        student.eval()
        context_head.eval()
    teacher.eval()
    ctx_offsets = [o for o in CONTEXT_OFFSETS if o != 0]
    vals = {'total_loss': [], 'frame_loss': [], 'dense_loss': [], 'global_loss': [], 'context_loss': [], 'context_lambda': [], 'ema_momentum': [], 'valid_frame_fraction': [], 'context_enabled_rate': []}
    for off in ctx_offsets:
        vals[f'alpha_{off:+d}'] = []
    it = _ssl_mod_utils.progress(loader, f"{('Train' if train else 'Monitor')} SSL {epoch:02d}/{total_epochs}")
    for batch in it:
        weak = batch['weak'].to(device, non_blocking=True)
        strong = batch['strong'].to(device, non_blocking=True)
        frame_valid_mask = batch['frame_valid_mask'].to(device, non_blocking=True)
        context_mask = batch['context_mask'].to(device, non_blocking=True)
        context_loss_enabled = batch['context_loss_enabled'].to(device, non_blocking=True)
        B, T = weak.shape[:2]
        weak_flat = weak.flatten(0, 1)
        strong_flat = strong.flatten(0, 1)
        if train:
            warmup_steps = max(1, int(round(total_steps * WARMUP_FRACTION)))
            mult = lr_multiplier(global_step, total_steps, warmup_steps)
            optimizer.param_groups[0]['lr'] = LR_BACKBONE * mult
            optimizer.param_groups[1]['lr'] = LR_HEAD * mult
            optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=AMP and device.type == 'cuda'):
                teacher_proj = teacher(weak_flat)
        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=AMP and device.type == 'cuda'):
                student_proj, student_pred = student(strong_flat)
                frame_loss, frame_parts = _ssl_mod_losses.frame_ssl_loss(student_pred, teacher_proj, frame_valid_mask, B, T)
                ctx_loss, alpha = _ssl_mod_losses.context_prediction_loss(student_proj, teacher_proj, context_head, context_mask, context_loss_enabled, B, T)
                lam = _ssl_mod_losses.context_lambda(global_step, total_steps, train=train)
                loss = frame_loss + lam * ctx_loss
            if train:
                trainable = [p for p in student.parameters() if p.requires_grad] + list(context_head.parameters())
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
                    optimizer.step()
                global_step += 1
                mom = _ssl_mod_losses.ema_momentum(global_step, total_steps)
                _ssl_mod_losses.update_teacher(student, teacher, mom)
            else:
                mom = np.nan
        vals['total_loss'].append(float(loss.detach().cpu()))
        vals['frame_loss'].append(frame_parts['frame_loss'])
        vals['dense_loss'].append(frame_parts['dense_loss'])
        vals['global_loss'].append(frame_parts['global_loss'])
        vals['context_loss'].append(float(ctx_loss.detach().cpu()))
        vals['context_lambda'].append(float(lam))
        vals['valid_frame_fraction'].append(float(frame_parts['valid_frame_fraction']))
        vals['context_enabled_rate'].append(float(context_loss_enabled.detach().float().mean().cpu()))
        enabled_bool = context_loss_enabled.detach().view(-1) > 0.5
        if bool(enabled_bool.any()):
            alpha_mean = alpha.detach().float()[enabled_bool].mean(dim=0).cpu().numpy()
            for off, a in zip(ctx_offsets, alpha_mean):
                vals[f'alpha_{off:+d}'].append(float(a))
        if np.isfinite(mom):
            vals['ema_momentum'].append(float(mom))
        if tqdm is not None:
            it.set_postfix(loss=f"{np.mean(vals['total_loss'][-30:]):.4f}", frame=f"{np.mean(vals['frame_loss'][-30:]):.4f}", ctx=f"{np.mean(vals['context_loss'][-30:]):.4f}", lam=f"{np.mean(vals['context_lambda'][-30:]):.3f}", lr=f"{optimizer.param_groups[0]['lr']:.1e}" if train else '-')
    out = {k: float(np.mean(v)) if len(v) else np.nan for k, v in vals.items()}
    return (out, global_step)


def cpu_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
