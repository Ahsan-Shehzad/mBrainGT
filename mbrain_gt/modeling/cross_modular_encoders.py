from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CrossOutputs:
    h_cross: torch.Tensor
    cls: torch.Tensor


class BiasedMultiheadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.dropout = float(dropout)

        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=True)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=True)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=True)
        self.out = nn.Linear(self.d_model, self.d_model, bias=True)

        self.bias_scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        q_in: torch.Tensor,
        kv_in: torch.Tensor,
        attn_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B, Nq, D = q_in.shape
        B2, Nk, D2 = kv_in.shape
        if B2 != B or D2 != D:
            raise ValueError("Batch/dim mismatch between q_in and kv_in.")
        if D != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got {D}")

        q = self.q_proj(q_in).view(B, Nq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_in).view(B, Nk, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_in).view(B, Nk, self.n_heads, self.head_dim).transpose(1, 2)

        bias = None
        if attn_bias is not None:
            if attn_bias.shape != (B, 1, Nq, Nk):
                raise ValueError(f"attn_bias must be (B,1,Nq,Nk). Got {tuple(attn_bias.shape)}")
            bias = self.bias_scale * attn_bias

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(B, Nq, D)
        y = self.out(y)
        return y


class CrossBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_hidden: int, dropout: float) -> None:
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.attn = BiasedMultiheadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.drop_attn = nn.Dropout(dropout)

        self.ln_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, d_model),
        )
        self.drop_ffn = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        cls: torch.Tensor,
        fc_bias: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, R, D = h.shape
        if cls.shape != (B, 1, D):
            raise ValueError("cls must be (B,1,d).")
        if fc_bias.shape != (B, R, R):
            raise ValueError("fc_bias must be (B,R,R).")

        q = self.ln_q(h)
        kv = self.ln_kv(h)
        attn_bias_nodes = fc_bias.unsqueeze(1).to(dtype=h.dtype)

        h_attn = self.attn(q_in=q, kv_in=kv, attn_bias=attn_bias_nodes)
        h = h + self.drop_attn(h_attn)

        q_cls = self.ln_q(cls)
        kv_cls = self.ln_kv(h)
        cls_attn = self.attn(q_in=q_cls, kv_in=kv_cls, attn_bias=None)
        cls = cls + self.drop_attn(cls_attn)

        h2 = self.ln_ffn(h)
        h = h + self.drop_ffn(self.ffn(h2))

        cls2 = self.ln_ffn(cls)
        cls = cls + self.drop_ffn(self.ffn(cls2))

        return h, cls


class CrossModularEncoders(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float, mlp_hidden: int) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.dropout = float(dropout)
        self.mlp_hidden = int(mlp_hidden)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList(
            [CrossBlock(d_model=d_model, n_heads=n_heads, mlp_hidden=mlp_hidden, dropout=dropout)
             for _ in range(n_layers)]
        )
        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, h_mod: torch.Tensor, global_fc: torch.Tensor) -> CrossOutputs:
        if h_mod.dim() != 3:
            raise ValueError(f"h_mod must be (B,R,d). Got {tuple(h_mod.shape)}")
        B, R, D = h_mod.shape
        if D != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got {D}")
        if global_fc.shape != (B, R, R):
            raise ValueError(f"global_fc must be (B,R,R). Got {tuple(global_fc.shape)}")

        cls = self.cls_token.expand(B, -1, -1)
        h = h_mod
        for blk in self.blocks:
            h, cls = blk(h, cls, global_fc)

        h = self.final_ln(h)
        cls = self.final_ln(cls).squeeze(1)
        return CrossOutputs(h_cross=h, cls=cls)
