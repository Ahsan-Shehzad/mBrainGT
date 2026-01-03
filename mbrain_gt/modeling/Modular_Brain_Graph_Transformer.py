from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from mbrain_gt.modeling.input_embeddings import InputEmbedding
from mbrain_gt.modeling.positional_encoding import PositionalEncoding
from mbrain_gt.modeling.modular_encoders import ModularEncoders
from mbrain_gt.modeling.cross_modular_encoders import CrossModularEncoders
from mbrain_gt.modeling.adaptive_fusion_classifier import AdaptiveFusionClassifier


@dataclass
class MBGTOutputs:
    logits: torch.Tensor
    probs: torch.Tensor
    h: Optional[torch.Tensor] = None
    h_pos: Optional[torch.Tensor] = None
    h_mod: Optional[torch.Tensor] = None
    h_cross: Optional[torch.Tensor] = None
    cls: Optional[torch.Tensor] = None
    h_global: Optional[torch.Tensor] = None
    alpha_modules: Optional[torch.Tensor] = None


class ModularBrainGraphTransformer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        d_model: int,
        n_heads: int,
        n_classes: int,
        dropout: float,
        mlp_hidden: int,
        k_lap_pe: int,
        n_layers_mod: int,
        n_layers_cross: int,
        edge_p_threshold: float = 0.01,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_classes = int(n_classes)
        self.dropout = float(dropout)
        self.mlp_hidden = int(mlp_hidden)
        self.k_lap_pe = int(k_lap_pe)
        self.n_layers_mod = int(n_layers_mod)
        self.n_layers_cross = int(n_layers_cross)
        self.edge_p_threshold = float(edge_p_threshold)

        self.input_emb = InputEmbedding(in_dim=self.in_dim, d_model=self.d_model, dropout=self.dropout)
        self.pos_enc = PositionalEncoding(
            d_model=self.d_model,
            k_lap=self.k_lap_pe,
            mlp_hidden=self.mlp_hidden,
            dropout=self.dropout,
        )
        self.mod_enc = ModularEncoders(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers_mod,
            dropout=self.dropout,
            mlp_hidden=self.mlp_hidden,
            edge_p_threshold=self.edge_p_threshold,
        )
        self.cross_enc = CrossModularEncoders(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers_cross,
            dropout=self.dropout,
            mlp_hidden=self.mlp_hidden,
        )
        self.classifier = AdaptiveFusionClassifier(
            d_model=self.d_model,
            n_classes=self.n_classes,
            mlp_hidden=self.mlp_hidden,
            dropout=self.dropout,
        )

    @classmethod
    def from_config(cls, cfg: Any, in_dim: int) -> "ModularBrainGraphTransformer":
        return cls(
            in_dim=in_dim,
            d_model=cfg.model.d_model,
            n_heads=cfg.model.n_heads,
            n_classes=cfg.model.n_classes,
            dropout=cfg.model.dropout,
            mlp_hidden=cfg.model.mlp_hidden,
            k_lap_pe=cfg.model.k_lap_pe,
            n_layers_mod=cfg.model.n_layers_mod,
            n_layers_cross=cfg.model.n_layers_cross,
            edge_p_threshold=cfg.connectivity.edge_p_threshold,
        )

    def forward(self, batch: Dict[str, Any], return_intermediates: bool = False) -> MBGTOutputs:
        x_raw: torch.Tensor = batch["x_raw"]
        pos: torch.Tensor = batch["pos"]
        global_fc: torch.Tensor = batch["global_fc"]
        module_slices: List[torch.Tensor] = batch["module_slices"]
        adj_module: List[torch.Tensor] = batch["adj_module"]
        pval_module: List[torch.Tensor] = batch["pval_module"]

        if x_raw.dim() != 3:
            raise ValueError(f"x_raw must be (B,R,F). Got {tuple(x_raw.shape)}")
        if pos.dim() != 3 or pos.size(-1) != 3:
            raise ValueError(f"pos must be (B,R,3). Got {tuple(pos.shape)}")
        if global_fc.dim() != 3:
            raise ValueError(f"global_fc must be (B,R,R). Got {tuple(global_fc.shape)}")

        B, R, F = x_raw.shape
        if F != self.in_dim:
            raise ValueError(f"Expected in_dim={self.in_dim}, got {F}")
        if global_fc.shape != (B, R, R):
            raise ValueError(f"global_fc must be (B,R,R). Got {tuple(global_fc.shape)}")

        h = self.input_emb(x_raw)
        h_pos = self.pos_enc(h, pos, module_slices, adj_module)
        h_mod = self.mod_enc(h_pos, module_slices, adj_module, pval_module)
        cross_out = self.cross_enc(h_mod, global_fc)
        h_cross = cross_out.h_cross
        cls = cross_out.cls
        clf_out = self.classifier(h_cross, module_slices)

        out = MBGTOutputs(logits=clf_out.logits, probs=clf_out.probs)
        if return_intermediates:
            out.h = h
            out.h_pos = h_pos
            out.h_mod = h_mod
            out.h_cross = h_cross
            out.cls = cls
            out.h_global = clf_out.h_global
            out.alpha_modules = clf_out.alpha_modules
        return out
