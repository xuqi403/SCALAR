import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import MODEL_CONFIG
from model import build_model, set_backbone_policy
from losses import build_loss
from postprocess import topology_cleanup, decode_outputs
from metrics import one_metrics

class TinyDINO(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks=nn.ModuleList([nn.Linear(1,1) for _ in range(12)])
        self.norm=nn.LayerNorm(1)
    def get_intermediate_layers(self,x,n,reshape=True):
        batch,h,w=x.shape[0],x.shape[-2],x.shape[-1]
        return [torch.ones(batch,768,h//16,w//16,device=x.device) for _ in n]


def test_d_robust_forward():
    with torch.no_grad():
        model = build_model(TinyDINO(), MODEL_CONFIG)
        out = model(torch.zeros(1, 3, 32, 32))
        assert out['outer_logits'].shape == (1, 1, 32, 32)
        assert out['lumen_logits'].shape == (1, 1, 32, 32)


def test_original_loss_and_backbone_policy():
    model=build_model(TinyDINO(), MODEL_CONFIG)
    set_backbone_policy(model,'last4',True)
    assert not any(p.requires_grad for blk in model.backbone.blocks[:8] for p in blk.parameters())
    assert all(p.requires_grad for blk in model.backbone.blocks[8:] for p in blk.parameters())
    with torch.no_grad():
        out=model(torch.zeros(1,3,32,32))
        label=torch.zeros(1,32,32,dtype=torch.long)
        label[:,5:27,5:27]=1;label[:,10:22,10:22]=2
        loss=build_loss(MODEL_CONFIG)(out,label)
        assert torch.isfinite(loss)


def test_postprocess_and_metrics():
    outer=np.zeros((40,40),dtype=bool)
    outer[5:35,5:35]=True
    lumen=np.zeros_like(outer);lumen[10:30,10:30]=True
    lumen[0:2,0:2]=True
    pred,info=topology_cleanup(outer,lumen)
    assert not np.any((pred==2)&(pred==0))
    assert np.all((pred==2) <= (pred>0))
    assert set(('raw_outer_components','post_outer_components')) <= set(info)
    got=one_metrics(pred,pred)
    assert got['outer_dice']==1.0
    assert got['lumen_dice']==1.0
