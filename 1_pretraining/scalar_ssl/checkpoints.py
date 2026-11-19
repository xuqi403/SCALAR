"""DINOv3-compatible checkpoint export and verification (methodology preserved)."""
from __future__ import annotations
import copy
import gc
from pathlib import Path
from typing import Any, Tuple
import torch
import torch.nn as nn
from . import settings
from .settings import *
from . import model as _ssl_mod_model
from . import trainer as _ssl_mod_trainer


def safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def state_dict_overlap(candidate: Any, model_keys: set) -> float:
    if not isinstance(candidate, dict) or not candidate:
        return 0.0
    tensor_keys = {str(k) for k, v in candidate.items() if torch.is_tensor(v)}
    if not tensor_keys:
        return 0.0
    return len(tensor_keys & model_keys) / max(len(model_keys), 1)


def save_backbone_in_source_format(backbone: nn.Module, source_weights: Path, out_path: Path) -> str:
    state = _ssl_mod_trainer.cpu_state_dict(backbone)
    torch.save(state, settings.BACKBONE_RAW_OUT)
    model_keys = set(state.keys())
    fmt = 'raw_state_dict'
    try:
        src = safe_torch_load(source_weights)
        if state_dict_overlap(src, model_keys) > 0.9:
            torch.save(state, out_path)
            fmt = 'raw_state_dict'
        elif isinstance(src, dict):
            replaced = False
            for key in ['model', 'state_dict', 'model_state_dict', 'teacher', 'student']:
                if key in src and state_dict_overlap(src[key], model_keys) > 0.9:
                    obj = copy.deepcopy(src)
                    obj[key] = state
                    torch.save(obj, out_path)
                    fmt = f'wrapped:{key}'
                    replaced = True
                    break
            if not replaced:
                torch.save(state, out_path)
                fmt = 'raw_state_dict_fallback'
        else:
            torch.save(state, out_path)
    except Exception as e:
        print(f'Could not inspect original DINO checkpoint wrapper ({e}); saving raw state_dict.')
        torch.save(state, out_path)
        fmt = 'raw_state_dict_fallback'
    return fmt


def verify_hub_weight_file(path: Path) -> Tuple[bool, str]:
    try:
        model = _ssl_mod_model.load_dinov3(path)
        n = sum((p.numel() for p in model.parameters()))
        del model
        gc.collect()
        return (True, f'torch.hub.load succeeded; params={n:,}')
    except Exception as e:
        return (False, f'{type(e).__name__}: {e}')
