"""Final D_robust image-segmentation objective from the source."""
from __future__ import annotations
from typing import Dict
import torch
from torch import nn
import torch.nn.functional as F
def soft_dice(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dims = tuple(range(1, prob.ndim))
    inter = (prob * target).sum(dims)
    den = prob.sum(dims) + target.sum(dims)
    return 1 - ((2 * inter + eps) / (den + eps)).mean()



def boundary_map(x: torch.Tensor) -> torch.Tensor:
    dil = F.max_pool2d(x, 3, 1, 1)
    ero = -F.max_pool2d(-x, 3, 1, 1)
    return torch.clamp(dil - ero, 0, 1)



class DualLoss(nn.Module):
    def __init__(self, wall: float, containment: float, boundary: float):
        super().__init__()
        self.wall = wall
        self.containment = containment
        self.boundary = boundary

    def forward(self, out, target):
        outer_logit, lumen_logit = out["outer_logits"], out["lumen_logits"]
        gt_outer = (target > 0).float().unsqueeze(1)
        gt_lumen = (target == 2).float().unsqueeze(1)
        gt_wall = (target == 1).float().unsqueeze(1)
        p_outer = torch.sigmoid(outer_logit)
        p_lumen = torch.sigmoid(lumen_logit)
        p_wall = p_outer * (1 - p_lumen)
        outer_loss = F.binary_cross_entropy_with_logits(outer_logit, gt_outer) + soft_dice(p_outer, gt_outer)
        lumen_loss = F.binary_cross_entropy_with_logits(lumen_logit, gt_lumen) + soft_dice(p_lumen, gt_lumen)
        loss = 0.5 * (outer_loss + lumen_loss)
        loss += self.wall * soft_dice(p_wall, gt_wall)
        loss += self.containment * torch.mean(p_lumen * (1 - p_outer))
        if self.boundary > 0:
            loss += self.boundary * 0.5 * (
                soft_dice(boundary_map(p_outer), boundary_map(gt_outer))
                + soft_dice(boundary_map(p_lumen), boundary_map(gt_lumen))
            )
        return loss



def build_loss(cfg: Dict) -> nn.Module:
    """Final D_robust objective: dual masks, wall Dice, containment, boundary."""
    return DualLoss(cfg["lambda_wall"], cfg["lambda_containment"], cfg["lambda_boundary"])
