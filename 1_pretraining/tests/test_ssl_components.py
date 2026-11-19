from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pretrain_ssl import (context_lambda, ema_momentum, uniform_candidate_indices,
                          sample_context_mask, CONTEXT_OFFSETS, CrossSectionContextHead)


def test_context_indexing():
    assert CONTEXT_OFFSETS == (-4, -2, 0, 2, 4)
    assert uniform_candidate_indices(20, 4)
    assert not uniform_candidate_indices(5, 4)


def test_untracked_context_never_used():
    m, eligible=sample_context_mask(np.zeros(4,dtype=np.float32),training=True)
    assert not m.any() and eligible==0


def test_context_head_shapes():
    head=CrossSectionContextHead(ch=8,hidden=16)
    tokens=torch.randn(2,4,8)
    masks=torch.tensor([[1,1,0,0],[0,1,0,0]],dtype=torch.float32)
    out,alpha=head(tokens,masks)
    assert out.shape==(2,8)
    assert alpha.shape==(2,4)
    assert (alpha[masks==0]==0).all()
