"""DINOv3 backbone, student/EMA teacher and context heads (methodology preserved)."""
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import torch
import torch.nn as nn
from . import settings
from .settings import *


def load_dinov3(weights_path: Path) -> nn.Module:
    repo = Path(settings.DINO_REPO)
    weight = Path(weights_path)
    if not (repo / 'hubconf.py').is_file():
        raise FileNotFoundError(f'DINOv3 repo not found: {repo}')
    if not weight.is_file():
        raise FileNotFoundError(f'DINOv3 weights not found: {weight}')
    print(f'[DINOv3] repo:    {repo}')
    print(f'[DINOv3] weights: {weight}')
    return torch.hub.load(str(repo), 'dinov3_vitb16', source='local', pretrained=True, weights=str(weight))


def group_norm(ch: int) -> nn.GroupNorm:
    for g in [32, 16, 8, 4, 2, 1]:
        if ch % g == 0:
            return nn.GroupNorm(g, ch)
    return nn.GroupNorm(1, ch)


class DenseProjector(nn.Module):

    def __init__(self, in_ch: int=768, hidden: int=PROJECTOR_HIDDEN, out_ch: int=PROJECTION_DIM):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch, hidden, 1, bias=False), group_norm(hidden), nn.GELU(), nn.Conv2d(hidden, out_ch, 1, bias=False))

    def forward(self, x):
        return self.net(x)


class DensePredictor(nn.Module):

    def __init__(self, ch: int=PROJECTION_DIM, hidden: int=PROJECTOR_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(ch, hidden, 1, bias=False), group_norm(hidden), nn.GELU(), nn.Conv2d(hidden, ch, 1, bias=True))

    def forward(self, x):
        return self.net(x)


class CrossSectionContextHead(nn.Module):

    def __init__(self, ch: int=PROJECTION_DIM, hidden: int=CONTEXT_HEAD_HIDDEN):
        super().__init__()
        n_ctx = len(CONTEXT_OFFSETS) - 1
        self.offset_embed = nn.Parameter(torch.zeros(n_ctx, ch))
        nn.init.normal_(self.offset_embed, std=0.02)
        self.score = nn.Sequential(nn.Linear(ch, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.predict = nn.Sequential(nn.Linear(ch, hidden), nn.GELU(), nn.Linear(hidden, ch))

    def forward(self, context_tokens: torch.Tensor, context_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = context_tokens + self.offset_embed[None]
        scores = self.score(x).squeeze(-1)
        mask = context_mask > 0.5
        scores = scores.masked_fill(~mask, -10000.0)
        alpha = torch.softmax(scores, dim=1)
        alpha = alpha * mask.float()
        alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-06)
        context = torch.sum(alpha[..., None] * x, dim=1)
        return (self.predict(context), alpha)


class SSLStudent(nn.Module):

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.projector = DenseProjector()
        self.predictor = DensePredictor()

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        feats = self.backbone.get_intermediate_layers(x, n=SSL_LAYERS, reshape=True)
        projected = [self.projector(f) for f in feats]
        predicted = [self.predictor(z) for z in projected]
        return (projected, predicted)


class SSLTeacher(nn.Module):

    def __init__(self, backbone: nn.Module, projector: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.projector = projector
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = self.backbone.get_intermediate_layers(x, n=SSL_LAYERS, reshape=True)
        return [self.projector(f) for f in feats]


def set_student_trainable(student: SSLStudent) -> None:
    for p in student.backbone.parameters():
        p.requires_grad = False
    blocks = getattr(student.backbone, 'blocks', None)
    if blocks is None:
        raise AttributeError('DINOv3 backbone has no .blocks; cannot fine-tune last blocks.')
    for block in list(blocks)[-TRAINABLE_LAST_N_BLOCKS:]:
        for p in block.parameters():
            p.requires_grad = True
    norm = getattr(student.backbone, 'norm', None)
    if norm is not None:
        for p in norm.parameters():
            p.requires_grad = True
    for p in student.projector.parameters():
        p.requires_grad = True
    for p in student.predictor.parameters():
        p.requires_grad = True


def apply_student_train_mode(student: SSLStudent) -> None:
    student.train()
    student.backbone.eval()
    blocks = list(getattr(student.backbone, 'blocks', []))
    for block in blocks[-TRAINABLE_LAST_N_BLOCKS:]:
        block.train()
    norm = getattr(student.backbone, 'norm', None)
    if norm is not None:
        norm.train()
    student.projector.train()
    student.predictor.train()
