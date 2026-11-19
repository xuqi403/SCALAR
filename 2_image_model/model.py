"""Final D_robust DINOv3 ViT-B/16 image segmentation architecture from source.

DINOv3 upstream code and licensed weights are not redistributed here.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict
import torch
from torch import nn
import torch.nn.functional as F

def load_dinov3(repo_path: str, weights_path: str) -> nn.Module:
    repo, weights = Path(repo_path).expanduser().resolve(), Path(weights_path).expanduser().resolve()
    if not (repo / "hubconf.py").is_file():
        raise FileNotFoundError(f"DINOv3 repository with hubconf.py not found: {repo}")
    if not weights.is_file():
        raise FileNotFoundError(f"Carotid SSL DINOv3 weights not found: {weights}")
    return torch.hub.load(str(repo), "dinov3_vitb16", source="local", pretrained=True, weights=str(weights))

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
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.layers = [2, 5, 8, 11]
        self.reduce = nn.Sequential(nn.Conv2d(768 * 4, 256, 1, bias=False), group_norm(256), nn.GELU())
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
        outer, lumen = self.outer_head(z), self.lumen_head(z)
        if outer.shape[-2:] != hw:
            outer = F.interpolate(outer, hw, mode="bilinear", align_corners=False)
            lumen = F.interpolate(lumen, hw, mode="bilinear", align_corners=False)
        return {"outer_logits": outer, "lumen_logits": lumen}



def build_model(backbone: nn.Module, cfg: Dict | None = None) -> nn.Module:
    """Build the final D_robust network (outer/lumen progressive decoder)."""
    return DualProgressiveSegmenter(backbone)


def base_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model



def set_backbone_policy(model, policy: str, trainable: bool) -> None:
    """Warm-up frozen; then train the final 4 DINO blocks and layer norm."""
    bb = base_model(model).backbone
    for p in bb.parameters():
        p.requires_grad = False
    if not trainable:
        return
    if policy == "last4":
        blocks = getattr(bb, "blocks", None)
        if blocks is None:
            raise AttributeError("DINOv3 backbone has no .blocks")
        for block in list(blocks)[-4:]:
            for p in block.parameters():
                p.requires_grad = True
        norm = getattr(bb, "norm", None)
        if norm is not None:
            for p in norm.parameters():
                p.requires_grad = True
        return
    raise ValueError(f"D_robust requires backbone policy last4; got {policy}")



def apply_backbone_train_mode(model, policy: str, trainable: bool) -> None:
    """Keep frozen DINO blocks deterministic; train only the intended tail in D."""
    bb = base_model(model).backbone
    if not trainable:
        bb.eval()
        return
    if policy == "last4":
        bb.eval()
        blocks = list(getattr(bb, "blocks", []))
        for block in blocks[-4:]:
            block.train()
        norm = getattr(bb, "norm", None)
        if norm is not None:
            norm.train()
        return
    raise ValueError(f"D_robust requires backbone policy last4; got {policy}")
