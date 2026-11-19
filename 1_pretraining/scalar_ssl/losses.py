"""Dense/global student–teacher consistency and context supervision (methodology preserved)."""
from __future__ import annotations
import math
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from . import settings
from .settings import *
from . import model as _ssl_mod_model


def negative_cosine_dense_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = F.normalize(pred, dim=1, eps=1e-06)
    z = F.normalize(target.detach(), dim=1, eps=1e-06)
    return 2.0 - 2.0 * (p * z).sum(dim=1).mean(dim=(1, 2))


def negative_cosine_vector_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = F.normalize(pred, dim=-1, eps=1e-06)
    z = F.normalize(target.detach(), dim=-1, eps=1e-06)
    return 2.0 - 2.0 * (p * z).sum(dim=-1)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denom


def frame_ssl_loss(student_pred: List[torch.Tensor], teacher_proj: List[torch.Tensor], frame_valid_mask: torch.Tensor, batch_size: int, clip_len: int) -> Tuple[torch.Tensor, Dict[str, float]]:
    valid = frame_valid_mask.float().view(batch_size, clip_len)
    dense_losses = []
    global_losses = []
    for p, z in zip(student_pred, teacher_proj):
        dense_per = negative_cosine_dense_per_sample(p, z).view(batch_size, clip_len)
        dense_losses.append(masked_mean(dense_per, valid))
        pg = p.mean(dim=(2, 3))
        zg = z.mean(dim=(2, 3))
        global_per = negative_cosine_vector_per_sample(pg, zg).view(batch_size, clip_len)
        global_losses.append(masked_mean(global_per, valid))
    dense = torch.stack(dense_losses).mean()
    global_loss = torch.stack(global_losses).mean()
    total = LAMBDA_DENSE * dense + LAMBDA_GLOBAL * global_loss
    return (total, {'dense_loss': float(dense.detach().cpu()), 'global_loss': float(global_loss.detach().cpu()), 'frame_loss': float(total.detach().cpu()), 'valid_frame_fraction': float(valid.detach().mean().cpu())})


def context_prediction_loss(student_proj: List[torch.Tensor], teacher_proj: List[torch.Tensor], context_head: _ssl_mod_model.CrossSectionContextHead, context_mask: torch.Tensor, context_loss_enabled: torch.Tensor, batch_size: int, clip_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    ctx_indices = [i for i in range(clip_len) if i != CENTER_INDEX]
    enabled = context_loss_enabled.float().view(batch_size)
    losses = []
    alphas = []
    for s, t in zip(student_proj, teacher_proj):
        sg = s.mean(dim=(2, 3)).view(batch_size, clip_len, -1)
        tg = t.mean(dim=(2, 3)).view(batch_size, clip_len, -1)
        context_tokens = sg[:, ctx_indices]
        target_center = tg[:, CENTER_INDEX]
        pred_center, alpha = context_head(context_tokens, context_mask)
        per = negative_cosine_vector_per_sample(pred_center, target_center)
        losses.append(masked_mean(per, enabled))
        alphas.append(alpha)
    if float(enabled.detach().sum().cpu()) <= 0:
        zero = sum((x.sum() * 0.0 for x in student_proj))
        loss = zero
    else:
        loss = torch.stack(losses).mean()
    return (loss, torch.stack(alphas, dim=0).mean(dim=0))


def context_lambda(step: int, total_steps: int, train: bool) -> float:
    if not train:
        return float(LAMBDA_CONTEXT_MAX)
    total_steps = max(int(total_steps), 1)
    frac = float(step) / float(total_steps)
    if frac < CONTEXT_DELAY_FRACTION:
        return 0.0
    ramp_end = CONTEXT_DELAY_FRACTION + max(CONTEXT_RAMP_FRACTION, 1e-06)
    if frac >= ramp_end:
        return float(LAMBDA_CONTEXT_MAX)
    u = (frac - CONTEXT_DELAY_FRACTION) / max(CONTEXT_RAMP_FRACTION, 1e-06)
    return float(LAMBDA_CONTEXT_MAX * np.clip(u, 0.0, 1.0))


def ema_momentum(step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return EMA_MOMENTUM_END
    t = step / (total_steps - 1)
    return EMA_MOMENTUM_END - (EMA_MOMENTUM_END - EMA_MOMENTUM_START) * (math.cos(math.pi * t) + 1.0) / 2.0


@torch.no_grad()
def update_teacher(student: _ssl_mod_model.SSLStudent, teacher: _ssl_mod_model.SSLTeacher, momentum: float) -> None:
    for ps, pt in zip(student.backbone.parameters(), teacher.backbone.parameters()):
        pt.data.mul_(momentum).add_(ps.data, alpha=1.0 - momentum)
    for ps, pt in zip(student.projector.parameters(), teacher.projector.parameters()):
        pt.data.mul_(momentum).add_(ps.data, alpha=1.0 - momentum)
