"""Shared latent encoder, activity head and SELFIES decoder.

Parameter names are retained for compatibility with the original LambdaDesign
checkpoints. Training objectives and training loops are intentionally absent.
"""

import math

import torch
from torch import nn


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, width, max_length):
        super().__init__()
        position = torch.arange(max_length).float().unsqueeze(1)
        scale = torch.exp(torch.arange(0, width, 2).float() * (-math.log(10000.0) / width))
        encoding = torch.zeros(max_length, width)
        encoding[:, 0::2] = torch.sin(position * scale)
        encoding[:, 1::2] = torch.cos(position * scale)
        self.register_buffer("pe", encoding.unsqueeze(0))

    def forward(self, values):
        if values.size(1) > self.pe.size(1):
            raise ValueError("Sequence exceeds the checkpoint's positional encoding length")
        return values + self.pe[:, :values.size(1)]


class NodeEncoder(nn.Module):
    def __init__(self, node_dim, d_model, nhead, enc_layers, dim_feedforward,
                 latent_dim, hidden_dim, max_atoms, encoder_pool):
        super().__init__()
        self.pool = encoder_pool
        self.proj = nn.Linear(node_dim, d_model)
        self.pos_enc = SinusoidalPositionEncoding(d_model, max_atoms + 1)
        if self.pool == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout=0.0)
        self.encoder = nn.TransformerEncoder(layer, enc_layers, enable_nested_tensor=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim), nn.ReLU(), nn.Dropout(0.0),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.0),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, node_embeddings, padding_mask):
        values = self.proj(node_embeddings)
        mask = padding_mask
        if self.pool == "cls":
            values = torch.cat([self.cls_token.expand(values.size(0), 1, -1), values], dim=1)
            prefix = torch.zeros(values.size(0), 1, dtype=torch.bool, device=mask.device)
            mask = torch.cat([prefix, mask], dim=1)
        values = self.pos_enc(values).transpose(0, 1)
        values = self.encoder(values, src_key_padding_mask=mask).transpose(0, 1)
        if self.pool == "cls":
            pooled = values[:, 0]
        else:
            valid = (~padding_mask).float().unsqueeze(-1)
            pooled = (values * valid).sum(dim=1) / (valid.sum(dim=1) + 1e-8)
        hidden = self.mlp(pooled)
        return self.fc_mu(hidden), self.fc_logvar(hidden)


class ActivityHead(nn.Module):
    def __init__(self, latent_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(), nn.Dropout(0.0),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1),
        )

    def forward(self, latent):
        return self.net(latent).squeeze(-1)


class LatentDesignModel(nn.Module):
    def __init__(self, vocab_size, node_dim, d_model, nhead, enc_layers, dec_layers,
                 dim_feedforward, latent_dim, hidden_dim, max_atoms, max_seq_len,
                 encoder_pool="mean"):
        super().__init__()
        self.latent_dim = latent_dim
        self.node_dim = node_dim
        self.max_atoms = max_atoms
        self.max_seq_len = max_seq_len
        self.encoder = NodeEncoder(node_dim, d_model, nhead, enc_layers,
                                   dim_feedforward, latent_dim, hidden_dim,
                                   max_atoms, encoder_pool)
        self.z_to_mem = nn.Sequential(nn.Linear(latent_dim, d_model), nn.ReLU(), nn.Dropout(0.0))
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = SinusoidalPositionEncoding(d_model, max_seq_len)
        layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout=0.0)
        self.decoder = nn.TransformerDecoder(layer, dec_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.activity_predictor = ActivityHead(latent_dim)

    def decode_logits(self, latent, tokens):
        memory = self.z_to_mem(latent).unsqueeze(1).transpose(0, 1)
        values = self.pos_enc(self.token_emb(tokens)).transpose(0, 1)
        length = tokens.size(1)
        causal_mask = torch.triu(torch.ones(length, length, device=tokens.device), diagonal=1).bool()
        hidden = self.decoder(tgt=values, memory=memory, tgt_mask=causal_mask)
        return self.lm_head(hidden.transpose(0, 1))
