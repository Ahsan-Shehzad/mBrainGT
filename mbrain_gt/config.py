from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml


@dataclass
class PathsConfig:
    project_root: Path = Path(".")
    data_root: Path = Path("data")
    raw_root: Path = Path("data/raw")
    derived_root: Path = Path("data/derived")

    def resolve_all(self) -> "PathsConfig":
        self.project_root = Path(self.project_root).expanduser().resolve()
        self.data_root = Path(self.data_root).expanduser().resolve()
        self.raw_root = Path(self.raw_root).expanduser().resolve()
        self.derived_root = Path(self.derived_root).expanduser().resolve()
        return self


@dataclass
class AtlasConfig:
    m_modules: int = 7
    r_rois: int = 246


@dataclass
class ConnectivityConfig:
    edge_p_threshold: float = 0.01
    combat_covariates: List[str] = field(default_factory=lambda: ["age", "sex", "mean_fd"])


@dataclass
class ModelConfig:
    d_model: int = 256
    dropout: float = 0.1
    mlp_hidden: int = 512
    n_heads: int = 8
    k_lap_pe: int = 8
    n_layers_mod: int = 2
    n_layers_cross: int = 2
    n_classes: int = 2


@dataclass
class TrainConfig:
    seed: int = 42
    n_folds: int = 5
    epochs: int = 200
    batch_size: int = 8
    num_workers: int = 4
    lr: float = 3e-4
    lr_min: float = 1e-6
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    amp: bool = True
    use_cosine: bool = True
    focal_gamma: float = 2.0
    use_class_alpha: bool = False
    early_stopping_patience: int = 30


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    atlas: AtlasConfig = field(default_factory=AtlasConfig)
    connectivity: ConnectivityConfig = field(default_factory=ConnectivityConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def resolve_paths(self) -> "Config":
        self.paths.resolve_all()
        return self


def _merge_dataclass(dc_obj: Any, updates: Dict[str, Any]) -> Any:
    for k, v in updates.items():
        if not hasattr(dc_obj, k):
            continue
        cur = getattr(dc_obj, k)
        if hasattr(cur, "__dataclass_fields__") and isinstance(v, dict):
            _merge_dataclass(cur, v)
        else:
            setattr(dc_obj, k, v)
    return dc_obj


def load_config(path: Union[str, Path]) -> Config:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = Config()

    if "paths" in raw and isinstance(raw["paths"], dict):
        p = raw["paths"]
        for key in ["project_root", "data_root", "raw_root", "derived_root"]:
            if key in p and p[key] is not None:
                p[key] = Path(p[key])

    _merge_dataclass(cfg, raw)
    cfg.resolve_paths()
    return cfg


def save_config(cfg: Config, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _as_dict(obj: Any) -> Any:
        if hasattr(obj, "__dataclass_fields__"):
            out = {}
            for k in obj.__dataclass_fields__.keys():  # type: ignore
                out[k] = _as_dict(getattr(obj, k))
            return out
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, list):
            return [_as_dict(x) for x in obj]
        return obj

    d = _as_dict(cfg)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(d, f, sort_keys=False)
