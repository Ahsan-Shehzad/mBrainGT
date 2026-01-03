from __future__ import annotations

import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    """Module 5: Input Embeddings."""

    def __init__(self, in_dim: int, d_model: int, dropout: float = 0.1, eps: float = 1e-6) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.d_model = int(d_model)
        self.eps = float(eps)

        self.proj = nn.Linear(self.in_dim, self.d_model, bias=True)
        self.dropout = nn.Dropout(p=float(dropout))

    @staticmethod
    def zscore_per_sample_across_nodes(x: torch.Tensor, eps: float) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        std = torch.sqrt(var + eps)
        return (x - mean) / std

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        if x_raw.dim() != 3:
            raise ValueError(f"x_raw must be 3D (B,R,F). Got shape: {tuple(x_raw.shape)}")
        if x_raw.size(-1) != self.in_dim:
            raise ValueError(f"x_raw last dim must be {self.in_dim}. Got {x_raw.size(-1)}")

        x = self.zscore_per_sample_across_nodes(x_raw, self.eps)
        h = self.proj(x)
        h = self.dropout(h)
        return h
