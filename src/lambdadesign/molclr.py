"""Frozen MolCLR node encoder, adapted from the MIT-licensed GINet source.

Copyright (c) 2021 Yuyang Wang. See licenses/MOLCLR.txt.
Message aggregation is expressed with torch.index_add_; PyG is not required.
Graph pooling and contrastive projection are unused by LambdaDesign inference.
"""

import torch
from torch import nn
from torch.nn import functional as F


class MolecularMessageLayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(width, 2 * width), nn.ReLU(), nn.Linear(2 * width, width))
        self.edge_embedding1 = nn.Embedding(5, width)
        self.edge_embedding2 = nn.Embedding(3, width)

    def forward(self, nodes, edges, bonds):
        indices = torch.arange(len(nodes), device=nodes.device)
        edges = torch.cat([edges, torch.stack([indices, indices])], dim=1)
        self_bonds = torch.zeros(len(nodes), 2, dtype=torch.long, device=nodes.device)
        self_bonds[:, 0] = 4
        bonds = torch.cat([bonds, self_bonds], dim=0)
        edge_features = self.edge_embedding1(bonds[:, 0]) + self.edge_embedding2(bonds[:, 1])
        messages = nodes[edges[0]] + edge_features
        aggregate = torch.zeros_like(nodes).index_add_(0, edges[1], messages)
        return self.mlp(aggregate)


class MolCLREncoder(nn.Module):
    def __init__(self, width, layers):
        super().__init__()
        self.width = width
        self.x_embedding1 = nn.Embedding(119, width)
        self.x_embedding2 = nn.Embedding(3, width)
        self.gnns = nn.ModuleList(MolecularMessageLayer(width) for _ in range(layers))
        self.batch_norms = nn.ModuleList(nn.BatchNorm1d(width) for _ in range(layers))

    def forward(self, atoms, edges, bonds):
        hidden = self.x_embedding1(atoms[:, 0]) + self.x_embedding2(atoms[:, 1])
        for index, (message, norm) in enumerate(zip(self.gnns, self.batch_norms)):
            hidden = norm(message(hidden, edges, bonds))
            if index < len(self.gnns) - 1:
                hidden = F.relu(hidden)
        return hidden
