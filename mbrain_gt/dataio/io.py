from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import torch


def save_graph_package(path: Union[str, Path], payload: Dict[str, Any]) -> None:
    """Save a subject graph package (.pt) using torch.save."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_graph_package(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a subject graph package (.pt) saved by save_graph_package."""
    path = Path(path)
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict in package, got {type(obj)}")
    return obj
