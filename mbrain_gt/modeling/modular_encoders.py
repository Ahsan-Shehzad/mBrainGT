from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AttnOutputs:
    h: torch.Tensor
    attn_weights_mean: Optional[torch.Tensor] = None


class BiasedMultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.dropout = float(dropout)

        self.qkv = nn.Linear(self.d_model, 3 * self.d_model, bias=True)
        self.out = nn.Linear(self.d_model, self.d_model, bias=True)
        self.edge_bias_scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor],
        return_attn: bool = False,
    ) -> AttnOutputs:
        B, N, D = x.shape
        if D != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got {D}")

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        combined = None
        if attn_bias is not None:
            if attn_bias.shape != (B, 1, N, N):
                raise ValueError(f"attn_bias must be (B,1,N,N). Got {tuple(attn_bias.shape)}")
            combined = self.edge_bias_scale * attn_bias
        if attn_mask is not None:
            if attn_mask.shape != (B, 1, N, N):
                raise ValueError(f"attn_mask must be (B,1,N,N). Got {tuple(attn_mask.shape)}")
            combined = attn_mask if combined is None else (combined + attn_mask)

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=combined,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )

        y = y.transpose(1, 2).contiguous().view(B, N, D)
        y = self.out(y)

        attn_mean = None
        if return_attn:
            logits = (q @ k.transpose(-1, -2)) / (self.head_dim ** 0.5)
            if attn_bias is not None:
                logits = logits + (self.edge_bias_scale * attn_bias)
            if attn_mask is not None:
                logits = logits + attn_mask
            w = torch.softmax(logits, dim=-1)
            attn_mean = w.mean(dim=1)

        return AttnOutputs(h=y, attn_weights_mean=attn_mean)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_hidden: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = BiasedMultiheadSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor],
        return_attn: bool = False,
    ) -> AttnOutputs:
        h1 = self.ln1(x)
        attn_out = self.attn(h1, attn_bias=attn_bias, attn_mask=attn_mask, return_attn=return_attn)
        x = x + self.drop1(attn_out.h)

        h2 = self.ln2(x)
        x = x + self.drop2(self.ffn(h2))
        return AttnOutputs(h=x, attn_weights_mean=attn_out.attn_weights_mean)


def build_attention_mask_from_pvals(
    pvals: torch.Tensor,
    p_threshold: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    B, N, N2 = pvals.shape
    if N != N2:
        raise ValueError("pvals must be square.")
    allowed = pvals < p_threshold
    eye = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
    allowed = allowed | eye
    neg_inf = torch.tensor(-1e9, device=device, dtype=dtype)
    mask = torch.where(allowed, torch.zeros((), device=device, dtype=dtype), neg_inf)
    return mask.view(B, 1, N, N)


class ModularEncoders(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        mlp_hidden: int,
        edge_p_threshold: float = 0.01,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.dropout = float(dropout)
        self.mlp_hidden = int(mlp_hidden)
        self.edge_p_threshold = float(edge_p_threshold)

        self._built_for_M: Optional[int] = None
        self.module_stacks: nn.ModuleList = nn.ModuleList()

    def _build(self, M: int) -> None:
        if self._built_for_M is not None:
            if self._built_for_M != M:
                raise ValueError(f"ModularEncoders already built for M={self._built_for_M}, got M={M}")
            return
        stacks = []
        for _m in range(M):
            layers = nn.ModuleList(
                [
                    TransformerBlock(
                        d_model=self.d_model,
                        n_heads=self.n_heads,
                        mlp_hidden=self.mlp_hidden,
                        dropout=self.dropout,
                    )
                    for _ in range(self.n_layers)
                ]
            )
            stacks.append(layers)
        self.module_stacks = nn.ModuleList(stacks)
        self._built_for_M = M

    def forward(
        self,
        h_pos: torch.Tensor,
        module_slices: List[torch.Tensor],
        adj_module: List[torch.Tensor],
        pval_module: List[torch.Tensor],
    ) -> torch.Tensor:
        if h_pos.dim() != 3:
            raise ValueError(f"h_pos must be (B,R,d). Got {tuple(h_pos.shape)}")
        if len(module_slices) != len(adj_module) or len(module_slices) != len(pval_module):
            raise ValueError("module_slices, adj_module, pval_module must have same length (M).")

        B, R, D = h_pos.shape
        if D != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got {D}")

        M = len(module_slices)
        self._build(M)

        h_out = h_pos.clone()

        for m in range(M):
            idx = module_slices[m]
            if idx.numel() == 0:
                continue
            hm = h_pos.index_select(dim=1, index=idx)

            A = adj_module[m]
            P = pval_module[m]
            if A.shape[0] != B or P.shape[0] != B:
                raise ValueError(f"Batch mismatch in module {m}.")
            Rm = idx.numel()
            if A.shape[1:] != (Rm, Rm) or P.shape[1:] != (Rm, Rm):
                raise ValueError(f"Shape mismatch for module {m}: adj/pval must be (B,Rm,Rm).")

            attn_bias = A.unsqueeze(1).to(dtype=hm.dtype)
            attn_mask = build_attention_mask_from_pvals(
                P.to(device=hm.device),
                p_threshold=self.edge_p_threshold,
                device=hm.device,
                dtype=hm.dtype,
            )

            for layer in self.module_stacks[m]:
                out = layer(hm, attn_bias=attn_bias, attn_mask=attn_mask, return_attn=False)
                hm = out.h

            h_out[:, idx, :] = hm

        return h_out
