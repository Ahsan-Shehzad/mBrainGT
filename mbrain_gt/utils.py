from __future__ import annotations

import json
import logging
import random
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence, Union, List

import numpy as np


_LOGGERS_CONFIGURED: set[str] = set()


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create/retrieve a logger with consistent formatting and a single StreamHandler."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    global _LOGGERS_CONFIGURED
    if name in _LOGGERS_CONFIGURED:
        return logger

    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    _LOGGERS_CONFIGURED.add(name)
    return logger


class _NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def save_json(path: Union[str, Path], obj: Any, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, sort_keys=False, cls=_NumpyJSONEncoder)
    tmp.replace(path)


def load_json(path: Union[str, Path]) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def seed_everything(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


@contextmanager
def timer(name: str, logger: Optional[logging.Logger] = None) -> Iterator[None]:
    t0 = time.time()
    try:
        yield
    finally:
        dt = time.time() - t0
        if logger is not None:
            logger.info(f"{name} took {dt:.3f}s")


def zscore(x: np.ndarray, axis: int = 0, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mu = x.mean(axis=axis, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=axis, keepdims=True)
    std = np.sqrt(var + eps)
    return (x - mu) / std


def pearson_corr(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be 2D (T,R).")
    T, R = X.shape
    if T < 2:
        raise ValueError("Need T>=2 to compute correlation.")

    Xc = X - X.mean(axis=0, keepdims=True)
    denom = np.sqrt((Xc ** 2).sum(axis=0, keepdims=True)) + eps
    Xn = Xc / denom
    C = (Xn.T @ Xn)
    C = np.clip(C, -1.0, 1.0)
    np.fill_diagonal(C, 1.0)
    return C.astype(np.float32)


def flatten_list(xs: Sequence[Sequence[Any]]) -> List[Any]:
    return [y for x in xs for y in x]
