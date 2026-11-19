"""Modularization and cross-module compatibility checks."""
from pathlib import Path
import importlib.util
import sys
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scalar_video import settings, cache, model, losses, roi, dataset, postprocess


def _training_lib():
    spec = importlib.util.spec_from_file_location('video_modular', ROOT / 'train_video.py')
    lib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lib)
    return lib


def test_core_exports_for_whole_video_inference():
    lib = _training_lib()
    names = [
        'VideoDINOv3Segmenter', 'AdaptiveRescueFusion', 'load_dinov3_backbone',
        'topology_cleanup', 'import_yolo_components', 'yolo_detect_boxes',
        'letterbox_image_and_mask', 'image_tensor', 'load_checkpoint_state',
        'CLIP_OFFSETS', 'CENTER_INDEX'
    ]
    assert all(hasattr(lib, name) for name in names)
    assert lib.VideoDINOv3Segmenter is model.VideoDINOv3Segmenter
    assert lib.CLIP_OFFSETS == (-4, -2, 0, 2, 4)


def test_runtime_path_settings_are_shared():
    lib = _training_lib()
    old = (settings.DINO_REPO, settings.DINO_WEIGHTS, settings.YOLO_CODE_DIR)
    try:
        lib.configure_runtime(dinov3_repo='/synthetic/repo', dinov3_weights='/synthetic/weights.pth', yolo_code_dir='/synthetic/yolo')
        assert settings.DINO_REPO == '/synthetic/repo'
        assert settings.DINO_WEIGHTS == '/synthetic/weights.pth'
        assert settings.YOLO_CODE_DIR == '/synthetic/yolo'
    finally:
        settings.DINO_REPO, settings.DINO_WEIGHTS, settings.YOLO_CODE_DIR = old


def test_cache_fingerprint_runs_on_synthetic_manifest():
    df = pd.DataFrame([{
        'sample_id':'sim', 'patient_id':'sim_p', 'split':'train',
        'video_path':'/tmp/sim.mp4', 'frame_idx':10,
        'crop_x1':0,'crop_y1':0,'crop_x2':100,'crop_y2':100,
        'annotation_image_path':'/tmp/sim.png',
        'annotation_json_path':'/tmp/sim.json'
    }])
    fp = cache.cache_fingerprint(df)
    assert isinstance(fp,str) and len(fp)==16
    assert fp == cache.cache_fingerprint(df)


def test_model_checkpoint_keys_and_loss_finite():
    class B(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks=torch.nn.ModuleList([torch.nn.Linear(1,1) for _ in range(12)])
            self.norm=torch.nn.LayerNorm(1)
        def get_intermediate_layers(self,x,n,reshape=True):
            a=x.mean((1,2,3),keepdim=True)
            return [a.expand(-1,768,2,2).contiguous() for _ in n]
    torch.manual_seed(21)
    net=model.VideoDINOv3Segmenter(B(),'adaptive',False)
    assert 'decoder.outer_head.weight' in net.state_dict()
    assert any(k.startswith('temporal.') for k in net.state_dict())
    with torch.no_grad():
        out=net(torch.rand((1,5,3,32,32)),torch.ones((1,5)))
    target=torch.zeros((1,32,32),dtype=torch.long)
    target[:,5:27,5:27]=1
    target[:,12:20,12:20]=2
    l,_=losses.DualAnatomyLoss(True,True)(out,target,return_parts=True)
    g,_=losses.adaptive_rescue_loss(out,target)
    assert torch.isfinite(l+g)
