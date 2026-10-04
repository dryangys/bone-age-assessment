"""Embeddings：Sex Embedding+ Chronological Age。"""
from __future__ import annotations

import torch
import torch.nn as nn


class SexEmbedding(nn.Module):
    """s ∈ {0,1} → E_sex(s) ∈ R^d_s，默认 d_s=16。"""

    def __init__(self, dim: int = 16):
        super().__init__()
        self.dim = dim
        self.embed = nn.Embedding(2, dim)

    def forward(self, sex: torch.Tensor) -> torch.Tensor:
        assert sex.shape[-1] >= 1
        return self.embed(sex.long().view(-1))


class AgeEmbedding(nn.Module):
    """chronological age → MLP(CA) ∈ R^768（可选，标准 RSNA 无 CA 时不用）。"""

    def __init__(self, out_dim: int = 768, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(1),
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, ca: torch.Tensor) -> torch.Tensor:
        return self.mlp(ca.view(-1, 1).float())
