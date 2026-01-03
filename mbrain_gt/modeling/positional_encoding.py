from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Module 6: Laplacian PE + MNI-coordinate MLP."""

    def __init__(
        self,
        d_model: int,
        k_lap: int = 8,
        mlp_hidden: int = 256,
        dropout: float = 0.1,
        lap_eps: float = 1e-6,
        abs_adj_for_laplacian: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.k_lap = int(k_lap)
        self.mlp_hidden = int(mlp_hidden)
        self.lap_eps = float(lap_eps)
        self.abs_adj_for_laplacian = bool(abs_adj_for_laplacian)

        self.coord_mlp = nn.Sequential(
            nn.Linear(3, self.mlp_hidden),
            nn.GELU(),
            nn.Linear(self.mlp_hidden, self.d_model),
        )

        self.lap_proj = nn.Linear(self.k_lap, self.d_model, bias=False)
        self.dropout = nn.Dropout(p=float(dropout))
        self.ln = nn.LayerNorm(self.d_model)

    @torch.no_grad()
    def _laplacian_pe_batch(self, A: torch.Tensor) -> torch.Tensor:
        if A.dim() != 3:
            raise ValueError(f"A must be (B,N,N). Got {tuple(A.shape)}")

        B, N, N2 = A.shape
        if N != N2:
            raise ValueError("Adjacency must be square.")

        if N <= 1 or self.k_lap <= 0:
            return torch.zeros((B, N, self.k_lap), device=A.device, dtype=A.dtype)

        W = A.abs() if self.abs_adj_for_laplacian else A
        W = 0.5 * (W + W.transpose(-1, -2))

        deg = W.sum(dim=-1)
        D = torch.diag_embed(deg)

        L = D - W

        evals, evecs = torch.linalg.eigh(L)

        pe = torch.zeros((B, N, self.k_lap), device=A.device, dtype=A.dtype)
        for b in range(B):
            ev = evals[b]
            U = evecs[b]
            nonzero = torch.where(ev > self.lap_eps)[0]
            if nonzero.numel() == 0:
                continue
            take = nonzero[: self.k_lap]
            U_take = U[:, take]
            for j in range(U_take.shape[1]):
                col = U_take[:, j]
                idx = torch.argmax(col.abs())
                if col[idx] < 0:
                    U_take[:, j] = -col
            pe[b, :, : U_take.shape[1]] = U_take
        return pe

    def forward(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        module_slices: List[torch.Tensor],
        adj_module: List[torch.Tensor],
    ) -> torch.Tensor:
        if h.dim() != 3:
            raise ValueError(f"h must be (B,R,d). Got {tuple(h.shape)}")
        if pos.dim() != 3 or pos.size(-1) != 3:
            raise ValueError(f"pos must be (B,R,3). Got {tuple(pos.shape)}")
        if h.shape[0] != pos.shape[0] or h.shape[1] != pos.shape[1]:
            raise ValueError("h and pos must share batch and node dimensions.")

        B, R, d = h.shape
        if d != self.d_model:
            raise ValueError(f"h last dim must be d_model={self.d_model}. Got {d}")
        if len(module_slices) != len(adj_module):
            raise ValueError("module_slices and adj_module must have same length (M).")

        s_coord = self.coord_mlp(pos)

        lap_full = torch.zeros((B, R, self.k_lap), device=h.device, dtype=h.dtype)
        for m, idx in enumerate(module_slices):
            if idx.numel() == 0:
                continue
            A = adj_module[m]
            if A.dim() != 3:
                raise ValueError(f"adj_module[{m}] must be (B,Rm,Rm). Got {tuple(A.shape)}")
            if A.shape[0] != B:
                raise ValueError(f"adj_module[{m}] batch mismatch. Expected {B}, got {A.shape[0]}")
            Rm = idx.numel()
            if A.shape[1] != Rm or A.shape[2] != Rm:
                raise ValueError(f"adj_module[{m}] shape mismatch with module_slices[{m}].")
            lap_m = self._laplacian_pe_batch(A)
            lap_full[:, idx, :] = lap_m

        s_lap = self.lap_proj(lap_full)
        s = s_coord + s_lap

        h_pos = h + s
        h_pos = self.ln(h_pos)
        h_pos = self.dropout(h_pos)
        return h_pos
