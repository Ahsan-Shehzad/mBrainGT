from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from mbrain_gt.config import Config
from mbrain_gt.modeling.Modular_Brain_Graph_Transformer import ModularBrainGraphTransformer
from mbrain_gt.training import FoldPackageDataset, collate_graph_packages
from mbrain_gt.utils import get_logger, load_json, save_json


@dataclass
class InferenceMetrics:
    accuracy: float
    f1_macro: float
    f1_weighted: float
    auc: Optional[float]
    auc_per_class: Optional[List[float]] = None


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, n_classes: int) -> InferenceMetrics:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score  # type: ignore
    y_pred = probs.argmax(axis=1)
    acc = float(accuracy_score(y_true, y_pred))
    f1m = float(f1_score(y_true, y_pred, average="macro"))
    f1w = float(f1_score(y_true, y_pred, average="weighted"))

    auc: Optional[float] = None
    auc_per_class: Optional[List[float]] = None

    if len(np.unique(y_true)) >= 2:
        if n_classes == 2:
            auc = float(roc_auc_score(y_true, probs[:, 1]))
        else:
            try:
                auc = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
                per = []
                for c in range(n_classes):
                    y_bin = (y_true == c).astype(int)
                    if len(np.unique(y_bin)) < 2:
                        per.append(float("nan"))
                    else:
                        per.append(float(roc_auc_score(y_bin, probs[:, c])))
                auc_per_class = per
            except Exception:
                auc = None
                auc_per_class = None

    return InferenceMetrics(accuracy=acc, f1_macro=f1m, f1_weighted=f1w, auc=auc, auc_per_class=auc_per_class)


@torch.no_grad()
def predict_split(
    model: ModularBrainGraphTransformer,
    loader: DataLoader,
    device: torch.device,
    n_classes: int,
    amp: bool,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_ids: List[str] = []
    all_true: List[int] = []
    all_probs: List[np.ndarray] = []

    for batch in loader:
        batch["x_raw"] = batch["x_raw"].to(device, non_blocking=True)
        batch["pos"] = batch["pos"].to(device, non_blocking=True)
        batch["global_fc"] = batch["global_fc"].to(device, non_blocking=True)
        batch["module_slices"] = [t.to(device, non_blocking=True) for t in batch["module_slices"]]
        batch["adj_module"] = [t.to(device, non_blocking=True) for t in batch["adj_module"]]
        batch["pval_module"] = [t.to(device, non_blocking=True) for t in batch["pval_module"]]
        y = batch["y"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            out = model(batch, return_intermediates=False)

        probs = out.probs.detach().cpu().numpy().astype(np.float32)
        all_ids.extend(batch["subject_id"])
        all_true.extend(y.detach().cpu().numpy().tolist())
        all_probs.append(probs)

    probs_all = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, n_classes), dtype=np.float32)
    y_true = np.asarray(all_true, dtype=np.int64)
    y_pred = probs_all.argmax(axis=1) if probs_all.shape[0] else np.zeros((0,), dtype=np.int64)
    return all_ids, y_true, y_pred, probs_all


def load_model_from_checkpoint(cfg: Config, ckpt_path: Path, in_dim: int, device: torch.device) -> ModularBrainGraphTransformer:
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    model = ModularBrainGraphTransformer.from_config(cfg, in_dim=in_dim).to(device)
    obj = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(obj["model"], strict=True)
    model.eval()
    return model


def run_inference_on_split_dir(
    cfg: Config,
    ckpt_path: Path,
    split_dir: Path,
    out_dir: Path,
    device: torch.device,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    amp: Optional[bool] = None,
) -> Dict[str, Any]:
    logger = get_logger("mbrain_gt.inference")
    split_dir = Path(split_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = FoldPackageDataset(split_dir)
    bs = int(batch_size) if batch_size is not None else int(cfg.train.batch_size)
    nw = int(num_workers) if num_workers is not None else int(cfg.train.num_workers)
    use_amp = bool(amp) if amp is not None else bool(getattr(cfg.train, "amp", device.type == "cuda"))

    loader = DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_graph_packages,
        drop_last=False,
    )

    sample0 = ds[0]
    in_dim = int(np.asarray(sample0["x_raw"]).shape[-1])
    model = load_model_from_checkpoint(cfg, ckpt_path=ckpt_path, in_dim=in_dim, device=device)

    ids, y_true, y_pred, probs = predict_split(model, loader, device, cfg.model.n_classes, use_amp)
    metrics = compute_metrics(y_true, probs, cfg.model.n_classes)

    pred_csv = out_dir / "predictions.csv"
    import pandas as pd  # local import
    df = pd.DataFrame({"subject_id": ids, "y_true": y_true.tolist(), "y_pred": y_pred.tolist()})
    for c in range(cfg.model.n_classes):
        df[f"prob_{c}"] = probs[:, c] if probs.shape[0] else []
    df.to_csv(pred_csv, index=False)

    metrics_json = out_dir / "metrics.json"
    metrics_dict = {
        "split_dir": str(split_dir),
        "ckpt_path": str(Path(ckpt_path).resolve()),
        "n_samples": int(len(ds)),
        "accuracy": metrics.accuracy,
        "f1_macro": metrics.f1_macro,
        "f1_weighted": metrics.f1_weighted,
        "auc": metrics.auc,
        "auc_per_class": metrics.auc_per_class,
        "predictions_csv": str(pred_csv),
    }
    save_json(metrics_json, metrics_dict)

    logger.info(
        f"Inference on {split_dir} | N={len(ds)} | acc={metrics.accuracy:.4f} "
        f"f1_macro={metrics.f1_macro:.4f} auc={(metrics.auc if metrics.auc is not None else float('nan')):.4f}"
    )
    return metrics_dict


def paired_one_tailed_ttest_greater(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    from scipy.stats import ttest_rel  # type: ignore
    if a.shape != b.shape:
        raise ValueError("AUC vectors must have same shape for paired test.")
    t_stat, p_two = ttest_rel(a, b, nan_policy="omit")
    if np.isnan(t_stat) or np.isnan(p_two):
        return float("nan"), float("nan")
    if t_stat > 0:
        p_one = p_two / 2.0
    else:
        p_one = 1.0 - (p_two / 2.0)
    return float(t_stat), float(p_one)


def benjamini_hochberg(pvals: List[float], q: float = 0.05) -> Tuple[List[bool], List[float]]:
    p = np.asarray(pvals, dtype=np.float64)
    m = len(p)
    if m == 0:
        return [], []
    order = np.argsort(p)
    p_sorted = p[order]
    ranks = np.arange(1, m + 1, dtype=np.float64)
    crit = (ranks / m) * float(q)
    below = p_sorted <= crit
    k = np.max(np.where(below)[0]) if np.any(below) else -1
    rejected_sorted = np.zeros((m,), dtype=bool)
    if k >= 0:
        rejected_sorted[: k + 1] = True
    p_adj_sorted = (m / ranks) * p_sorted
    p_adj_sorted = np.minimum.accumulate(p_adj_sorted[::-1])[::-1]
    p_adj_sorted = np.clip(p_adj_sorted, 0.0, 1.0)
    rejected = np.zeros((m,), dtype=bool)
    p_adj = np.zeros((m,), dtype=np.float64)
    rejected[order] = rejected_sorted
    p_adj[order] = p_adj_sorted
    return rejected.tolist(), p_adj.tolist()


def load_fold_auc_vector_from_run_dir(run_dir: Path) -> np.ndarray:
    run_dir = Path(run_dir)
    fold_dirs = sorted([p for p in run_dir.glob("fold_*") if p.is_dir()])
    if not fold_dirs:
        raise FileNotFoundError(f"No fold_* dirs in {run_dir}")
    aucs: List[float] = []
    for fd in fold_dirs:
        summary = fd / "summary.json"
        if summary.exists():
            s = load_json(summary)
            auc = s.get("test_auc", None)
            aucs.append(float("nan") if auc is None else float(auc))
            continue
        pred_csv = fd / "test_predictions.csv"
        if not pred_csv.exists():
            raise FileNotFoundError(f"Missing summary.json and test_predictions.csv in {fd}")
        import pandas as pd  # local import
        from sklearn.metrics import roc_auc_score  # type: ignore
        df = pd.read_csv(pred_csv)
        y_true = df["y_true"].values.astype(int)
        prob_cols = [c for c in df.columns if c.startswith("prob_")]
        probs = df[prob_cols].values.astype(np.float32)
        if probs.shape[1] == 2 and len(np.unique(y_true)) >= 2:
            aucs.append(float(roc_auc_score(y_true, probs[:, 1])))
        elif probs.shape[1] > 2 and len(np.unique(y_true)) >= 2:
            aucs.append(float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro")))
        else:
            aucs.append(float("nan"))
    return np.asarray(aucs, dtype=np.float64)


def compare_methods_auc_with_fdr(method_to_run_dir: Dict[str, Path], primary_method: str, q: float = 0.05) -> Dict[str, Any]:
    if primary_method not in method_to_run_dir:
        raise ValueError("primary_method must be one of method_to_run_dir keys.")
    names = list(method_to_run_dir.keys())
    aucs_by_method = {n: load_fold_auc_vector_from_run_dir(method_to_run_dir[n]) for n in names}
    base = aucs_by_method[primary_method]
    results = []
    pvals = []
    for name in names:
        if name == primary_method:
            continue
        other = aucs_by_method[name]
        if base.shape != other.shape:
            raise ValueError(f"Fold count mismatch: {primary_method} vs {name}")
        t, p = paired_one_tailed_ttest_greater(base, other)
        pvals.append(p)
        results.append(
            {
                "baseline": name,
                "t_stat": t,
                "p_one_tailed": p,
                "mean_auc_primary": float(np.nanmean(base)),
                "mean_auc_baseline": float(np.nanmean(other)),
                "delta_mean_auc": float(np.nanmean(base - other)),
            }
        )
    rejected, p_adj = benjamini_hochberg(pvals, q=q)
    for i in range(len(results)):
        results[i]["p_fdr"] = p_adj[i]
        results[i]["significant_fdr_q"] = bool(rejected[i])
    return {
        "primary_method": primary_method,
        "q_fdr": float(q),
        "aucs_by_method": {k: aucs_by_method[k].tolist() for k in names},
        "comparisons": results,
    }
