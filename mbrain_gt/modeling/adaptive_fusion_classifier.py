from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ClassifierOutputs:
    logits: torch.Tensor
    probs: torch.Tensor
    h_global: torch.Tensor
    h_modules: torch.Tensor
    alpha_modules: torch.Tensor


class AdaptiveFusionClassifier(nn.Module):
    def __init__(self, d_model: int, n_classes: int, mlp_hidden: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.n_classes = int(n_classes)
        self.mlp_hidden = int(mlp_hidden)
        self.dropout = float(dropout)

        self.pool_q = nn.Parameter(torch.zeros(d_model))
        nn.init.trunc_normal_(self.pool_q, std=0.02)

        self.module_score = nn.Linear(d_model, 1, bias=True)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, n_classes),
        )

    def _pool_module(self, h_cross: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        B, R, D = h_cross.shape
        if idx.numel() == 0:
            return torch.zeros((B, D), device=h_cross.device, dtype=h_cross.dtype)

        hm_nodes = h_cross.index_select(dim=1, index=idx)
        scores = torch.einsum("bnd,d->bn", hm_nodes, self.pool_q)
        alpha = torch.softmax(scores, dim=1)
        h_m = torch.einsum("bn,bnd->bd", alpha, hm_nodes)
        return h_m

    def forward(self, h_cross: torch.Tensor, module_slices: List[torch.Tensor]) -> ClassifierOutputs:
        if h_cross.dim() != 3:
            raise ValueError(f"h_cross must be (B,R,d). Got {tuple(h_cross.shape)}")
        B, R, D = h_cross.shape
        if D != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got {D}")
        M = len(module_slices)
        if M == 0:
            raise ValueError("module_slices must be non-empty.")

        h_m_list = [self._pool_module(h_cross, idx) for idx in module_slices]
        h_modules = torch.stack(h_m_list, dim=1)

        score_m = self.module_score(h_modules).squeeze(-1)
        alpha_m = torch.softmax(score_m, dim=1)

        h_global = torch.einsum("bm,bmd->bd", alpha_m, h_modules)

        logits = self.classifier(h_global)
        probs = torch.softmax(logits, dim=-1)

        return ClassifierOutputs(
            logits=logits,
            probs=probs,
            h_global=h_global,
            h_modules=h_modules,
            alpha_modules=alpha_m,
        )


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None, reduction: str = "mean") -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = alpha
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be mean/sum/none")
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.dim() != 2:
            raise ValueError("logits must be (B,C)")
        if targets.dim() != 1:
            raise ValueError("targets must be (B,)")

        B, C = logits.shape
        targets = targets.long()

        logp = F.log_softmax(logits, dim=-1)
        p = torch.exp(logp)

        idx = targets.view(-1, 1)
        logp_t = logp.gather(1, idx).squeeze(1)
        p_t = p.gather(1, idx).squeeze(1)

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device).gather(0, targets)
        else:
            alpha_t = torch.ones_like(p_t)

        loss = -alpha_t * (1.0 - p_t).pow(self.gamma) * logp_t

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
