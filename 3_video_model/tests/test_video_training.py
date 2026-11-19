"""Offline smoke tests (no clinical data and no GPU required)."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location('scalar_video_train', HERE/'train_video.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_final_configs_and_offsets():
    assert set(m.VIDEO_EXPERIMENTS) == {'V4_attn_last4','V5_adaptive_rescue_spur_curv'}
    assert m.CLIP_OFFSETS == (-4,-2,0,2,4)
    assert m.CENTER_INDEX == 2
    assert m.V5_GATE_TEMPERATURE == 0.35
    assert m.VIDEO_EXPERIMENTS['V5_adaptive_rescue_spur_curv']['adaptive_rescue']


def test_cli_defaults():
    a=m.parse_args([])
    assert a.mode == 'all'
    assert a.experiment == 'V5_adaptive_rescue_spur_curv'


def test_strict_split_and_leakage(tmp_path):
    rows=pd.DataFrame([
      {'patient_id':'a','video_path':'v1','split':'train'},
      {'patient_id':'b','video_path':'v2','split':'val'},
      {'patient_id':'c','video_path':'v3','split':'test'},
    ])
    out=m.attach_image_split(rows,tmp_path)
    assert list(out.split)==['train','val','test']
    rows.loc[2,'patient_id']='a'
    with pytest.raises(RuntimeError): m.attach_image_split(rows,tmp_path)
    with pytest.raises(ValueError): m.attach_image_split(pd.DataFrame({'patient_id':['a']}),tmp_path)


def test_annotation_polygon_mask(tmp_path):
    import json
    j=tmp_path/'mask.json'
    j.write_text(json.dumps({'shapes':[
      {'label':'wall','shape_type':'polygon','points':[[3,3],[27,3],[27,27],[3,27]]},
      {'label':'lumen','shape_type':'polygon','points':[[10,10],[20,10],[20,20],[10,20]]}
    ]}),encoding='utf-8')
    out=m.load_labelme_mask(str(j),32,32)
    assert out.shape==(32,32) and (out==1).any() and (out==2).any()
    assert np.all(out[10:20,10:20]==2)


def test_npz_clip_loader(tmp_path):
    cache=tmp_path/'synthetic.npz'
    np.savez_compressed(cache,
       clip=np.zeros((5,448,448,3),dtype=np.uint8),
       mask=np.zeros((448,448),dtype=np.uint8),
       context_weights=np.ones(5,dtype=np.float32),
       frame_indices=np.array([6,8,10,12,14],dtype=np.int32))
    ds=m.VideoClipDataset(pd.DataFrame([{'cache_path':str(cache),'sample_id':'synthetic','frame_idx':10}]),training=False)
    item=ds[0]
    assert item['clip'].shape==(5,3,448,448)
    assert item['mask'].shape==(448,448)
    assert item['context_weights'].shape==(5,)


def test_final_model_modules_with_mock_backbone():
    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks=torch.nn.ModuleList([torch.nn.Linear(1,1) for _ in range(12)])
        def get_intermediate_layers(self, x, n, reshape):
            # 16x16 input -> fake 1x1 DINO feature to avoid GPU / upstream download
            return [torch.zeros(x.shape[0],768,1,1,dtype=x.dtype,device=x.device) for _ in n]
    backbone=Backbone()
    v4=m.VideoDINOv3Segmenter(backbone,'attention',True)
    v5=m.VideoDINOv3Segmenter(Backbone(),'adaptive',False)
    assert hasattr(v4,'decoder') and hasattr(v5.temporal,'forward')
    assert v5.adaptive_rescue is True and v4.adaptive_rescue is False


def test_plot_module_present():
    import plot
    assert callable(plot.update_training_monitor)
    assert callable(plot.make_all_figures)


def test_v4_v5_real_forward_and_loss_with_mock_backbone():
    """Exercise the actual fusion, decoder and supervision (without DINO downloads)."""
    torch.set_num_threads(min(torch.get_num_threads(), 2))
    class MiniBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks=torch.nn.ModuleList([torch.nn.Linear(1,1) for _ in range(12)])
            self.norm=torch.nn.LayerNorm(1)
        def get_intermediate_layers(self,x,n,reshape=True):
            a=x.mean(dim=(1,2,3),keepdim=True)
            return [a.expand(-1,768,2,2).contiguous() for _ in n]
    x=torch.rand((1,5,3,32,32))
    w=torch.ones((1,5))
    target=torch.zeros((1,32,32),dtype=torch.long)
    target[:,4:28,4:28]=1
    target[:,12:20,12:20]=2
    for cfg in m.VIDEO_EXPERIMENTS.values():
        net=m.VideoDINOv3Segmenter(MiniBackbone(),cfg['fusion'],cfg['temporal_loss']).eval()
        with torch.no_grad():
            out=net(x,w)
            assert out['outer_logits'].shape == (1,1,32,32)
            assert out['lumen_logits'].shape == (1,1,32,32)
            anatomy=m.DualAnatomyLoss(cfg['outer_spur_loss'],cfg['outer_curvature_loss'])
            l,_=anatomy(out,target,return_parts=True)
            extra,_=(m.adaptive_rescue_loss(out,target) if cfg['adaptive_rescue']
                     else m.temporal_structure_loss(out,target,w))
            assert torch.isfinite(l+extra)
