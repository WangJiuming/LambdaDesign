from pathlib import Path

import pytest
import torch

from lambdadesign.model import LatentDesignModel
from lambdadesign.molclr import MolCLREncoder


@pytest.fixture
def checkpoints(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    cfg = dict(node_dim=8, d_model=8, nhead=2, enc_layers=1, dec_layers=1,
               dim_feedforward=16, latent_dim=4, hidden_dim=8, encoder_pool="mean")
    data = dict(max_atoms=20, max_seq_len=12)
    vocab = {s:i for i,s in enumerate(["<pad>", "<bos>", "<eos>", "<unk>", "[C]", "[N]", "[O]"])}
    model = LatentDesignModel(vocab_size=len(vocab), **cfg, **data)
    design = tmp_path / "design.pt"
    torch.save(dict(model_state=model.state_dict(), cvae_cfg=cfg, data_cfg=data,
                    vocab_stoi=vocab, act_p75=0.4), design)
    encoder = tmp_path / "encoder.pth"
    torch.save(MolCLREncoder(8, 2).state_dict(), encoder)
    return design, encoder
