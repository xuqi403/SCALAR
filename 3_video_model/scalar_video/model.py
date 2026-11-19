"""Model operations for SCALAR V4/V5 video segmentation."""
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
from .utils import base_model


def load_dinov3_backbone() -> nn.Module:
    repo = Path(settings.DINO_REPO)
    weight = Path(settings.DINO_WEIGHTS)
    if not (repo / "hubconf.py").is_file():
        raise FileNotFoundError(f"DINOv3 repo not found: {repo}")
    if not weight.is_file():
        raise FileNotFoundError(f"DINOv3 weights not found: {weight}")
    print(f"[DINOv3] repo:    {repo}")
    print(f"[DINOv3] weights: {weight} ({weight.stat().st_size / 1024**2:.1f} MB)")
    return torch.hub.load(
        str(repo), "dinov3_vitb16", source="local", pretrained=True, weights=str(weight)
    )


def group_norm(ch: int) -> nn.GroupNorm:
    for g in [32, 16, 8, 4, 2, 1]:
        if ch % g == 0:
            return nn.GroupNorm(g, ch)
    return nn.GroupNorm(1, ch)


class UpBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), group_norm(cout), nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), group_norm(cout), nn.GELU(),
        )

    def forward(self, x):
        return self.block(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))


class DualProgressiveSegmenter(nn.Module):
    """Image D model used for image reference model and QC comparison."""
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.layers = [2, 5, 8, 11]
        self.reduce = nn.Sequential(
            nn.Conv2d(768 * 4, 256, 1, bias=False), group_norm(256), nn.GELU()
        )
        self.up1 = UpBlock(256, 192)
        self.up2 = UpBlock(192, 128)
        self.up3 = UpBlock(128, 96)
        self.up4 = UpBlock(96, 64)
        self.outer_head = nn.Conv2d(64, 1, 1)
        self.lumen_head = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        hw = x.shape[-2:]
        feats = self.backbone.get_intermediate_layers(x, n=self.layers, reshape=True)
        z = self.reduce(torch.cat(feats, 1))
        for block in [self.up1, self.up2, self.up3, self.up4]:
            z = block(z)
        outer = self.outer_head(z)
        lumen = self.lumen_head(z)
        if outer.shape[-2:] != hw:
            outer = F.interpolate(outer, hw, mode="bilinear", align_corners=False)
            lumen = F.interpolate(lumen, hw, mode="bilinear", align_corners=False)
        return {"outer_logits": outer, "lumen_logits": lumen}


class ProgressiveDecoder(nn.Module):
    """Same high-resolution decoder as the image D model, isolated for video fusion."""
    def __init__(self):
        super().__init__()
        self.up1 = UpBlock(256, 192)
        self.up2 = UpBlock(192, 128)
        self.up3 = UpBlock(128, 96)
        self.up4 = UpBlock(96, 64)
        self.outer_head = nn.Conv2d(64, 1, 1)
        self.lumen_head = nn.Conv2d(64, 1, 1)

    def forward(self, z: torch.Tensor, out_hw: Tuple[int, int]) -> Dict[str, torch.Tensor]:
        for block in [self.up1, self.up2, self.up3, self.up4]:
            z = block(z)
        outer = self.outer_head(z)
        lumen = self.lumen_head(z)
        if outer.shape[-2:] != out_hw:
            outer = F.interpolate(outer, out_hw, mode="bilinear", align_corners=False)
            lumen = F.interpolate(lumen, out_hw, mode="bilinear", align_corners=False)
        return {"outer_logits": outer, "lumen_logits": lumen}


def load_checkpoint_state(path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    out = {}
    for k, v in state.items():
        k2 = k[7:] if k.startswith("module.") else k
        out[k2] = v
    return out


def load_image_model(device: torch.device) -> DualProgressiveSegmenter:
    model = DualProgressiveSegmenter(load_dinov3_backbone())
    state = load_checkpoint_state(settings.IMAGE_CHECKPOINT)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[Image checkpoint] missing keys: {len(missing)}")
    if unexpected:
        print(f"[Image checkpoint] unexpected keys: {len(unexpected)}")
    model = model.to(device)
    model.eval()
    return model


class SpatialTemporalAttention(nn.Module):
    """
    Attend over TIME independently at every 28x28 spatial location.

    DINO already supplies strong spatial/semantic features. We do not mix spatial
    positions here; the new module only asks: which neighboring time points help
    explain the same local vascular position in the center frame?
    """
    def __init__(self, dim: int = 256, heads: int = 8, max_frames: int = 9, dropout: float = 0.05):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, max_frames, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, feats: torch.Tensor, weights: torch.Tensor, center_index: int) -> torch.Tensor:
        B, T, C, H, W = feats.shape
        if T > self.pos.shape[1]:
            raise ValueError(f"T={T} exceeds max temporal positions={self.pos.shape[1]}")

        # [B,H,W,T,C] -> [BHW,T,C]
        seq = feats.permute(0, 3, 4, 1, 2).contiguous().view(B * H * W, T, C)
        seq = seq + self.pos[:, :T].expand(B * H * W, -1, -1)

        w = weights.clamp(min=0, max=1)
        w_expand = w[:, None, None, :, None].expand(B, H, W, T, 1).reshape(B * H * W, T, 1)
        key_value = seq * w_expand

        query = seq[:, center_index:center_index + 1]
        key_padding = (w <= 1e-6)[:, None, None, :].expand(B, H, W, T).reshape(B * H * W, T).clone()  # Expanded view is overlapping; PyTorch requires a writable copy.
        # Center is always a real labeled frame; guarantee one valid key.
        key_padding[:, center_index] = False

        attn_out, _ = self.attn(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding,
            need_weights=False,
        )
        x = self.norm1(query + attn_out)
        x = self.norm2(x + self.ffn(x))
        x = x[:, 0].view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x


class LocalFeatureAligner(nn.Module):
    """Learn a bounded local sampling offset that aligns one neighbor to center.

    YOLO already performs coarse ROI recentering. This module only learns the small
    residual displacement that remains at the 28x28 DINO feature level.
    """
    def __init__(self, dim: int = 256, hidden: int = 96, max_offset: float = 2.0):
        super().__init__()
        self.max_offset = float(max_offset)
        self.offset_net = nn.Sequential(
            nn.Conv2d(dim * 3, hidden, 1, bias=False), group_norm(hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False), group_norm(hidden), nn.GELU(),
            nn.Conv2d(hidden, 2, 3, padding=1),
        )
        # Start from identity alignment; offsets are learned only when useful.
        nn.init.zeros_(self.offset_net[-1].weight)
        nn.init.zeros_(self.offset_net[-1].bias)

    def forward(self, center: torch.Tensor, neighbor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, _, H, W = center.shape
        delta = torch.tanh(self.offset_net(torch.cat([center, neighbor, (center - neighbor).abs()], dim=1)))
        delta = delta * self.max_offset  # feature-pixel units, [B,2,H,W]

        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=center.device, dtype=center.dtype),
            torch.linspace(-1, 1, W, device=center.device, dtype=center.dtype),
            indexing="ij",
        )
        grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1).clone()
        if W > 1:
            grid[..., 0] += delta[:, 0] * (2.0 / (W - 1))
        if H > 1:
            grid[..., 1] += delta[:, 1] * (2.0 / (H - 1))

        aligned = F.grid_sample(
            neighbor,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return aligned, delta


class AdaptiveRescueFusion(nn.Module):
    """Learned neighbor alignment, reliability weighting, and spatial rescue gate.

    Center is deliberately excluded from the context competition. The module first
    builds a context-only feature C from aligned neighbors, then learns G(x,y):
        fused = G * center + (1-G) * context.
    Thus there is no hand-written inference threshold deciding when to switch.
    """
    def __init__(self, dim: int = 256, max_frames: int = 9):
        super().__init__()
        self.aligner = LocalFeatureAligner(
            dim=dim,
            hidden=96,
            max_offset=settings.V5_MAX_ALIGN_OFFSET_FEATURE_PX,
        )
        self.reliability = nn.Sequential(
            nn.Conv2d(dim * 3, 128, 1, bias=False), group_norm(128), nn.GELU(),
            nn.Conv2d(128, 64, 3, padding=1, bias=False), group_norm(64), nn.GELU(),
            nn.Conv2d(64, 1, 1),
        )
        self.temporal_bias = nn.Parameter(torch.zeros(max_frames))
        self.gate_head = nn.Sequential(
            nn.Conv2d(dim * 3 + 1, 128, 1, bias=False), group_norm(128), nn.GELU(),
            nn.Conv2d(128, 64, 3, padding=1, bias=False), group_norm(64), nn.GELU(),
            nn.Conv2d(64, 1, 1),
        )
        # Stable initialization: begin close to the strong image solution, but do not
        # hard-code any final center/context proportion.
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, 1.5)

    def forward(
        self,
        feats: torch.Tensor,
        availability: torch.Tensor,
        center_index: int,
        center_outer_logit: torch.Tensor,
        center_lumen_logit: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        B, T, C, H, W = feats.shape
        center = feats[:, center_index]
        neighbor_indices = [i for i in range(T) if i != center_index]

        aligned_list = []
        offset_list = []
        score_list = []
        valid_list = []

        for i in neighbor_indices:
            aligned, delta = self.aligner(center, feats[:, i])
            score = self.reliability(torch.cat([center, aligned, (center - aligned).abs()], dim=1))[:, 0]
            score = score + self.temporal_bias[i]
            valid = (availability[:, i] > 0).view(B, 1, 1)
            score = score.masked_fill(~valid, -1e4)
            aligned_list.append(aligned)
            offset_list.append(delta)
            score_list.append(score)
            valid_list.append(valid)

        aligned = torch.stack(aligned_list, dim=1)       # [B,N,C,H,W]
        offsets = torch.stack(offset_list, dim=1)        # [B,N,2,H,W]
        scores = torch.stack(score_list, dim=1)          # [B,N,H,W]
        valid = torch.stack(valid_list, dim=1).float()    # [B,N,1,1]

        alpha = torch.softmax(scores, dim=1) * valid
        alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-6)
        context = (aligned * alpha.unsqueeze(2)).sum(dim=1)

        has_context = (valid.sum(dim=1) > 0).view(B, 1, 1, 1)
        context = torch.where(has_context, context, center)

        with torch.no_grad():
            po = torch.sigmoid(center_outer_logit)
            pl = torch.sigmoid(center_lumen_logit)
            uncertainty = 0.5 * (4.0 * po * (1.0 - po) + 4.0 * pl * (1.0 - pl))

        gate_logits = self.gate_head(torch.cat([center, context, (center - context).abs(), uncertainty], dim=1))
        gate = torch.sigmoid(gate_logits)
        gate = torch.where(has_context, gate, torch.ones_like(gate))
        fused = gate * center + (1.0 - gate) * context

        alpha_full = []
        offset_full = []
        n = 0
        for i in range(T):
            if i == center_index:
                alpha_full.append(torch.zeros(B, H, W, device=feats.device, dtype=feats.dtype))
                offset_full.append(torch.zeros(B, 2, H, W, device=feats.device, dtype=feats.dtype))
            else:
                alpha_full.append(alpha[:, n])
                offset_full.append(offsets[:, n])
                n += 1

        return {
            "fused": fused,
            "center_feature": center,
            "context_feature": context,
            "gate": gate,
            "gate_logits": gate_logits,
            "neighbor_alpha": torch.stack(alpha_full, dim=1),      # [B,T,H,W]
            "alignment_offsets": torch.stack(offset_full, dim=1), # [B,T,2,H,W]
            "has_context": has_context[:, 0, 0, 0],               # [B]
        }


class VideoDINOv3Segmenter(nn.Module):
    def __init__(self, backbone: nn.Module, fusion: str, temporal_loss: bool):
        super().__init__()
        self.backbone = backbone
        self.layers = [2, 5, 8, 11]
        self.fusion_name = fusion
        self.adaptive_rescue = fusion == "adaptive"
        self.use_temporal_aux = bool(temporal_loss) or self.adaptive_rescue

        # Same reduce layer as image D model for maximal checkpoint reuse.
        self.reduce = nn.Sequential(
            nn.Conv2d(768 * 4, 256, 1, bias=False), group_norm(256), nn.GELU()
        )
        self.decoder = ProgressiveDecoder()

        if fusion == "attention":
            self.temporal = SpatialTemporalAttention(dim=256, heads=8, max_frames=max(9, len(settings.CLIP_OFFSETS)))
        elif fusion == "adaptive":
            self.temporal = AdaptiveRescueFusion(dim=256, max_frames=max(9, len(settings.CLIP_OFFSETS)))
        else:
            raise ValueError(f"Unsupported video fusion: {fusion}")

        # Lightweight low-resolution auxiliary heads. V4 uses these for temporal regularization; V5 uses them for rescue.
        self.aux_outer_head = nn.Conv2d(256, 1, 1)
        self.aux_lumen_head = nn.Conv2d(256, 1, 1)

    def extract_frame_features(self, clip: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = clip.shape
        flat = clip.view(B * T, C, H, W)

        # Saving activation memory is important: V4 warm-up freezes DINO completely.
        backbone_trainable = any(p.requires_grad for p in self.backbone.parameters())
        if backbone_trainable:
            feats = self.backbone.get_intermediate_layers(flat, n=self.layers, reshape=True)
        else:
            with torch.no_grad():
                feats = self.backbone.get_intermediate_layers(flat, n=self.layers, reshape=True)
        z = self.reduce(torch.cat(feats, dim=1))
        _, C2, h, w = z.shape
        return z.view(B, T, C2, h, w)

    def fuse_once(
        self,
        frame_feats: torch.Tensor,
        context_weights: torch.Tensor,
        center_index: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not self.adaptive_rescue:
            return self.temporal(frame_feats, context_weights, center_index), {}

        center = frame_feats[:, center_index]
        center_outer = self.aux_outer_head(center)
        center_lumen = self.aux_lumen_head(center)
        # V5 deliberately ignores the handcrafted nonzero weight magnitudes. Only
        # physical availability (>0) is used as a hard mask.
        availability = (context_weights > 0).float()
        availability[:, center_index] = 1.0
        diag = self.temporal(
            frame_feats, availability, center_index, center_outer, center_lumen
        )
        diag["adaptive_center_outer_logits"] = center_outer
        diag["adaptive_center_lumen_logits"] = center_lumen
        diag["adaptive_context_outer_logits"] = self.aux_outer_head(diag["context_feature"])
        diag["adaptive_context_lumen_logits"] = self.aux_lumen_head(diag["context_feature"])
        return diag["fused"], diag

    def forward(
        self,
        clip: torch.Tensor,
        context_weights: torch.Tensor,
        return_sequence: bool = False,
    ) -> Dict[str, torch.Tensor]:
        B, T, _, H, W = clip.shape
        frame_feats = self.extract_frame_features(clip)

        # Evaluation-only dense local sequence inference.
        #
        # The model was trained to segment the CENTER frame of a local clip.
        # To assess temporal stability without changing training, each of the T
        # cached frames is temporarily moved to CENTER_INDEX and predicted with
        # its available local neighbors. Out-of-range positions at the two clip
        # edges are zero-weighted rather than duplicated as temporal evidence.
        if return_sequence:
            outer_seq = []
            lumen_seq = []

            for target_index in range(T):
                src_indices = []
                valid_positions = []

                for k in range(T):
                    raw_index = target_index + (k - settings.CENTER_INDEX)
                    valid = 0 <= raw_index < T
                    src_indices.append(min(max(raw_index, 0), T - 1))
                    valid_positions.append(valid)

                idx = torch.as_tensor(src_indices, device=clip.device, dtype=torch.long)
                local_feats = frame_feats.index_select(1, idx)
                local_weights = context_weights.index_select(1, idx).clone()

                valid_mask = torch.as_tensor(
                    valid_positions,
                    device=clip.device,
                    dtype=local_weights.dtype,
                ).unsqueeze(0)
                local_weights = local_weights * valid_mask

                # The frame being segmented is always a real frame and must remain
                # available to the temporal module even if its original YOLO
                # reliability as a context frame was low.
                local_weights[:, settings.CENTER_INDEX] = 1.0

                fused, _ = self.fuse_once(local_feats, local_weights, settings.CENTER_INDEX)
                dec = self.decoder(fused, (H, W))
                outer_seq.append(dec["outer_logits"])
                lumen_seq.append(dec["lumen_logits"])

            return {
                "outer_logits": torch.stack(outer_seq, dim=1),   # [B,T,1,H,W]
                "lumen_logits": torch.stack(lumen_seq, dim=1),   # [B,T,1,H,W]
            }

        # Original center-frame output contract is preserved for every experiment.
        fused, adaptive_diag = self.fuse_once(frame_feats, context_weights, settings.CENTER_INDEX)
        out = self.decoder(fused, (H, W))
        out["frame_features"] = frame_feats

        if self.adaptive_rescue:
            out.update(adaptive_diag)
        elif self.use_temporal_aux:
            B, T, C, h, w = frame_feats.shape
            flat = frame_feats.view(B * T, C, h, w)
            aux_outer = self.aux_outer_head(flat).view(B, T, 1, h, w)
            aux_lumen = self.aux_lumen_head(flat).view(B, T, 1, h, w)
            out["aux_outer_logits"] = aux_outer
            out["aux_lumen_logits"] = aux_lumen
        return out


def initialize_video_from_image_checkpoint(model: VideoDINOv3Segmenter, checkpoint_path: str) -> None:
    state = load_checkpoint_state(checkpoint_path)
    target = model.state_dict()
    mapped = {}

    # Backbone + reduce names match directly.
    for k, v in state.items():
        if k.startswith("backbone.") or k.startswith("reduce."):
            if k in target and target[k].shape == v.shape:
                mapped[k] = v

    # Image decoder keys up1/up2/... become decoder.up1/...
    for k, v in state.items():
        if k.startswith(("up1.", "up2.", "up3.", "up4.", "outer_head.", "lumen_head.")):
            k2 = "decoder." + k
            if k2 in target and target[k2].shape == v.shape:
                mapped[k2] = v

    missing, unexpected = model.load_state_dict(mapped, strict=False)
    print(
        f"Initialized video model from image checkpoint: matched {len(mapped)}/{len(target)} tensors; "
        f"new temporal/aux tensors remain randomly initialized."
    )


def initialize_adaptive_from_previous_checkpoint(
    model: VideoDINOv3Segmenter,
    checkpoint_path: Path,
) -> bool:
    """Load every shape-compatible V4 tensor; new V5 modules stay initialized."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        return False
    state = load_checkpoint_state(str(checkpoint_path))
    target = model.state_dict()
    mapped = {
        k: v for k, v in state.items()
        if k in target and target[k].shape == v.shape
    }
    model.load_state_dict(mapped, strict=False)
    print(
        f"Initialized V5 from previous video checkpoint: {checkpoint_path} | "
        f"matched {len(mapped)}/{len(target)} tensors; adaptive modules start new."
    )
    return True


def set_video_trainability(model: nn.Module, cfg: Dict[str, Any], epoch: int) -> str:
    m = base_model(model)

    if cfg.get("adaptive_rescue", False):
        # Stage 1: learn only how to align/select/rescue using the established V4 feature space.
        for p in m.parameters():
            p.requires_grad = False
        for p in m.temporal.parameters():
            p.requires_grad = True
        for p in list(m.aux_outer_head.parameters()) + list(m.aux_lumen_head.parameters()):
            p.requires_grad = True

        if epoch <= settings.V5_ADAPTIVE_WARMUP_EPOCHS:
            return "adaptive_only"

        # Stage 2: let reduce + existing decoder adapt to the learned fusion.
        for p in m.reduce.parameters():
            p.requires_grad = True
        for p in m.decoder.parameters():
            p.requires_grad = True
        if epoch <= settings.V5_BACKBONE_WARMUP_EPOCHS:
            return "adaptive_decoder"

        # Stage 3: conservative video-domain tuning of only the last four DINO blocks.
        blocks = list(getattr(m.backbone, "blocks", []))
        for block in blocks[-4:]:
            for p in block.parameters():
                p.requires_grad = True
        norm = getattr(m.backbone, "norm", None)
        if norm is not None:
            for p in norm.parameters():
                p.requires_grad = True
        return "adaptive_last4"

    for p in m.parameters():
        p.requires_grad = True

    # First train adaptation layers with the pretrained DINO backbone frozen.
    for p in m.backbone.parameters():
        p.requires_grad = False

    if cfg["fine_tune_last4"] and epoch > settings.V4_BACKBONE_WARMUP_EPOCHS:
        blocks = list(getattr(m.backbone, "blocks", []))
        for block in blocks[-4:]:
            for p in block.parameters():
                p.requires_grad = True
        norm = getattr(m.backbone, "norm", None)
        if norm is not None:
            for p in norm.parameters():
                p.requires_grad = True
        return "last4_finetune"
    return "frozen_dino"


def apply_video_train_mode(model: nn.Module, cfg: Dict[str, Any], epoch: int) -> None:
    m = base_model(model)
    model.train()
    m.backbone.eval()
    backbone_warmup = settings.V5_BACKBONE_WARMUP_EPOCHS if cfg.get("adaptive_rescue", False) else settings.V4_BACKBONE_WARMUP_EPOCHS
    if cfg["fine_tune_last4"] and epoch > backbone_warmup:
        blocks = list(getattr(m.backbone, "blocks", []))
        for block in blocks[-4:]:
            block.train()
        norm = getattr(m.backbone, "norm", None)
        if norm is not None:
            norm.train()

