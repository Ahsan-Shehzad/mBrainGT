from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from mbrain_gt.config import Config
from mbrain_gt.dataio import load_graph_package
from mbrain_gt.modeling.Modular_Brain_Graph_Transformer import ModularBrainGraphTransformer
from mbrain_gt.modeling.adaptive_fusion_classifier import FocalLoss
from mbrain_gt.utils import get_logger, save_json


class FoldPackageDataset(Dataset):
    def __init__(self, split_dir: Path) -> None:
        self.split_dir = Path(split_dir)
        if not self.split_dir.exists():
            raise FileNotFoundError(self.split_dir)
        files = sorted(self.split_dir.glob("subject_*.pt"))
        if len(files) == 0:
            raise RuntimeError(f"No subject_*.pt files found in: {self.split_dir}")
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.files[idx]
        payload = load_graph_package(path)
        sid = payload.get("subject_id", None)
        if sid is None:
            sid = path.stem.replace("subject_", "")
        payload["subject_id"] = str(sid)
        return payload


def _stack_module_mats(batch_list: List[Dict[str, Any]], key: str) -> List[torch.Tensor]:
    M = len(batch_list[0][key])
    out: List[torch.Tensor] = []
    for m in range(M):
        mats = [torch.tensor(s[key][m], dtype=torch.float32) for s in batch_list]
        out.append(torch.stack(mats, dim=0))
    return out


def collate_graph_packages(batch_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    x_raw = torch.stack([torch.tensor(s["x_raw"], dtype=torch.float32) for s in batch_list], dim=0)
    pos = torch.stack([torch.tensor(s["pos"], dtype=torch.float32) for s in batch_list], dim=0)
    global_fc = torch.stack([torch.tensor(s["global_fc"], dtype=torch.float32) for s in batch_list], dim=0)

    module_slices = [torch.tensor(ix, dtype=torch.long) for ix in batch_list[0]["module_slices"]]

    adj_module = _stack_module_mats(batch_list, "adj_module")
    pval_module = _stack_module_mats(batch_list, "pval_module")

    y = torch.tensor([int(s["y"]) for s in batch_list], dtype=torch.long)
    subject_id = [str(s.get("subject_id", "")) for s in batch_list]

    return {
        "x_raw": x_raw,
        "pos": pos,
        "global_fc": global_fc,
        "module_slices": module_slices,
        "adj_module": adj_module,
        "pval_module": pval_module,
        "y": y,
        "subject_id": subject_id,
    }


@dataclass
class MetricPack:
    loss: float
    acc: float
    bal_acc: float
    auc: Optional[float]


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    recalls = []
    for c in range(n_classes):
        mask = (y_true == c)
        if mask.sum() == 0:
            continue
        recalls.append(float((y_pred[mask] == c).mean()))
    return float(np.mean(recalls)) if len(recalls) else 0.0


def set_seed(seed: int) -> None:
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(batch)
    out["x_raw"] = batch["x_raw"].to(device, non_blocking=True)
    out["pos"] = batch["pos"].to(device, non_blocking=True)
    out["global_fc"] = batch["global_fc"].to(device, non_blocking=True)
    out["module_slices"] = [t.to(device, non_blocking=True) for t in batch["module_slices"]]
    out["adj_module"] = [t.to(device, non_blocking=True) for t in batch["adj_module"]]
    out["pval_module"] = [t.to(device, non_blocking=True) for t in batch["pval_module"]]
    out["y"] = batch["y"].to(device, non_blocking=True)
    return out


def compute_class_alpha_from_train(train_dir: Path, n_classes: int) -> torch.Tensor:
    ds = FoldPackageDataset(train_dir)
    counts = np.zeros((n_classes,), dtype=np.int64)
    for i in range(len(ds)):
        y = int(ds[i]["y"])
        if 0 <= y < n_classes:
            counts[y] += 1
    counts = counts.astype(np.float32)
    med = float(np.median(counts[counts > 0])) if (counts > 0).any() else 1.0
    alpha = med / (counts + 1e-6)
    alpha = alpha / (alpha.mean() + 1e-12)
    return torch.tensor(alpha, dtype=torch.float32)


@torch.no_grad()
def evaluate(
    model: ModularBrainGraphTransformer,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    n_classes: int,
    amp: bool,
) -> Tuple[MetricPack, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    model.eval()
    losses: List[float] = []
    all_true: List[int] = []
    all_pred: List[int] = []
    all_probs: List[np.ndarray] = []
    all_ids: List[str] = []

    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        y = batch["y"]
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            out = model(batch, return_intermediates=False)
            loss = criterion(out.logits, y)

        probs = out.probs.detach().cpu().numpy()
        pred = probs.argmax(axis=1)

        losses.append(float(loss.item()))
        all_true.extend(y.detach().cpu().numpy().tolist())
        all_pred.extend(pred.tolist())
        all_probs.append(probs)
        all_ids.extend(batch["subject_id"])

    y_true = np.asarray(all_true, dtype=np.int64)
    y_pred = np.asarray(all_pred, dtype=np.int64)
    probs = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, n_classes), dtype=np.float32)

    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    bal = _balanced_accuracy(y_true, y_pred, n_classes=n_classes)

    auc = None
    if n_classes == 2 and probs.shape[0] > 0:
        try:
            from sklearn.metrics import roc_auc_score  # type: ignore
            if len(np.unique(y_true)) >= 2:
                auc = float(roc_auc_score(y_true, probs[:, 1]))
        except Exception:
            auc = None

    pack = MetricPack(
        loss=float(np.mean(losses)) if losses else 0.0,
        acc=acc,
        bal_acc=bal,
        auc=auc,
    )
    return pack, y_true, y_pred, probs, all_ids


def train_one_epoch(
    model: ModularBrainGraphTransformer,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    criterion: nn.Module,
    n_classes: int,
    amp: bool,
    grad_clip: float,
) -> MetricPack:
    model.train()
    losses: List[float] = []
    all_true: List[int] = []
    all_pred: List[int] = []

    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        y = batch["y"]

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            out = model(batch, return_intermediates=False)
            loss = criterion(out.logits, y)

        if amp and scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        probs = out.probs.detach().cpu().numpy()
        pred = probs.argmax(axis=1)
        losses.append(float(loss.item()))
        all_true.extend(y.detach().cpu().numpy().tolist())
        all_pred.extend(pred.tolist())

    y_true = np.asarray(all_true, dtype=np.int64)
    y_pred = np.asarray(all_pred, dtype=np.int64)
    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    bal = _balanced_accuracy(y_true, y_pred, n_classes=n_classes)
    return MetricPack(loss=float(np.mean(losses)) if losses else 0.0, acc=acc, bal_acc=bal, auc=None)


def save_checkpoint(
    ckpt_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_bal_acc: float,
    cfg: Config,
) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "best_val_bal_acc": float(best_val_bal_acc),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cfg": cfg.__dict__,
        },
        ckpt_path,
    )


def load_checkpoint(
    ckpt_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[int, float]:
    obj = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(obj["model"], strict=True)
    if optimizer is not None and "optimizer" in obj:
        optimizer.load_state_dict(obj["optimizer"])
    epoch = int(obj.get("epoch", 0))
    best = float(obj.get("best_val_bal_acc", 0.0))
    return epoch, best


def train_fold(
    cfg: Config,
    fold_id: int,
    device: torch.device,
    experiment_dir: Path,
    resume: bool = False,
) -> Dict[str, Any]:
    logger = get_logger("mbrain_gt.training")
    fold_root = (cfg.paths.derived_root / "fold_packages" / f"fold_{fold_id}").resolve()
    if not fold_root.exists():
        raise FileNotFoundError(fold_root)

    train_dir = fold_root / "train"
    val_dir = fold_root / "val"
    test_dir = fold_root / "test"

    train_ds = FoldPackageDataset(train_dir)
    val_ds = FoldPackageDataset(val_dir)
    test_ds = FoldPackageDataset(test_dir)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_graph_packages,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_graph_packages,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_graph_packages,
        drop_last=False,
    )

    sample0 = train_ds[0]
    in_dim = int(np.asarray(sample0["x_raw"]).shape[-1])
    model = ModularBrainGraphTransformer.from_config(cfg, in_dim=in_dim).to(device)

    alpha = None
    if getattr(cfg.train, "use_class_alpha", False):
        alpha = compute_class_alpha_from_train(train_dir, cfg.model.n_classes)
    criterion = FocalLoss(
        gamma=float(getattr(cfg.train, "focal_gamma", 2.0)),
        alpha=alpha,
        reduction="mean",
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )

    scheduler = None
    if getattr(cfg.train, "use_cosine", True):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(cfg.train.epochs),
            eta_min=float(getattr(cfg.train, "lr_min", 1e-6)),
        )

    amp = bool(getattr(cfg.train, "amp", (device.type == "cuda")))
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    fold_exp_dir = (experiment_dir / f"fold_{fold_id}").resolve()
    fold_exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_best = fold_exp_dir / "best.pt"
    ckpt_last = fold_exp_dir / "last.pt"
    metrics_path = fold_exp_dir / "metrics.json"

    start_epoch = 0
    best_val_bal = 0.0
    if resume and ckpt_last.exists():
        start_epoch, best_val_bal = load_checkpoint(ckpt_last, model, optimizer)
        logger.info(f"[fold {fold_id}] Resumed epoch={start_epoch}, best_val_bal={best_val_bal:.4f}")

    patience = int(getattr(cfg.train, "early_stopping_patience", 30))
    patience_ctr = 0

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "train_bal_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_bal_acc": [],
        "val_auc": [],
    }

    for epoch in range(start_epoch, int(cfg.train.epochs)):
        t0 = time.time()

        tr = train_one_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            criterion=criterion,
            n_classes=cfg.model.n_classes,
            amp=amp,
            grad_clip=float(getattr(cfg.train, "grad_clip", 1.0)),
        )

        va, _, _, _, _ = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            criterion=criterion,
            n_classes=cfg.model.n_classes,
            amp=amp,
        )

        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(tr.loss)
        history["train_acc"].append(tr.acc)
        history["train_bal_acc"].append(tr.bal_acc)
        history["val_loss"].append(va.loss)
        history["val_acc"].append(va.acc)
        history["val_bal_acc"].append(va.bal_acc)
        history["val_auc"].append(va.auc if va.auc is not None else float("nan"))

        save_checkpoint(ckpt_last, model, optimizer, epoch=epoch + 1, best_val_bal_acc=best_val_bal, cfg=cfg)

        improved = va.bal_acc > best_val_bal + 1e-6
        if improved:
            best_val_bal = va.bal_acc
            patience_ctr = 0
            save_checkpoint(ckpt_best, model, optimizer, epoch=epoch + 1, best_val_bal_acc=best_val_bal, cfg=cfg)
        else:
            patience_ctr += 1

        dt = time.time() - t0
        logger.info(
            f"[fold {fold_id}] epoch {epoch+1:03d}/{cfg.train.epochs} | "
            f"train loss {tr.loss:.4f} acc {tr.acc:.4f} bal {tr.bal_acc:.4f} | "
            f"val loss {va.loss:.4f} acc {va.acc:.4f} bal {va.bal_acc:.4f} "
            f"auc {va.auc if va.auc is not None else float('nan'):.4f} | "
            f"best bal {best_val_bal:.4f} | {dt:.1f}s"
        )
        save_json(metrics_path, {"best_val_bal_acc": best_val_bal, "history": history})

        if patience_ctr >= patience:
            logger.info(f"[fold {fold_id}] Early stopping at epoch {epoch+1}.")
            break

    if ckpt_best.exists():
        load_checkpoint(ckpt_best, model, optimizer=None)

    te, y_true, y_pred, probs, ids = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        criterion=criterion,
        n_classes=cfg.model.n_classes,
        amp=amp,
    )

    pred_csv = fold_exp_dir / "test_predictions.csv"
    import pandas as pd  # local import
    df = pd.DataFrame({"subject_id": ids, "y_true": y_true.tolist(), "y_pred": y_pred.tolist()})
    for c in range(cfg.model.n_classes):
        df[f"prob_{c}"] = probs[:, c] if probs.shape[0] else []
    df.to_csv(pred_csv, index=False)

    summary = {
        "fold_id": fold_id,
        "best_val_bal_acc": best_val_bal,
        "test_loss": te.loss,
        "test_acc": te.acc,
        "test_bal_acc": te.bal_acc,
        "test_auc": te.auc,
        "paths": {
            "fold_root": str(fold_root),
            "experiment_fold_dir": str(fold_exp_dir),
            "best_ckpt": str(ckpt_best),
            "last_ckpt": str(ckpt_last),
            "metrics": str(metrics_path),
            "test_predictions": str(pred_csv),
        },
    }
    save_json(fold_exp_dir / "summary.json", summary)
    logger.info(
        f"[fold {fold_id}] TEST | loss {te.loss:.4f} acc {te.acc:.4f} bal {te.bal_acc:.4f} "
        f"auc {te.auc if te.auc is not None else float('nan'):.4f}"
    )
    return summary


def train_all_folds(cfg: Config, device: torch.device, experiment_dir: Path, resume: bool = False) -> Dict[str, Any]:
    logger = get_logger("mbrain_gt.training")
    results = []
    for fold_id in range(int(cfg.train.n_folds)):
        results.append(train_fold(cfg, fold_id=fold_id, device=device, experiment_dir=experiment_dir, resume=resume))

    test_bal = [r["test_bal_acc"] for r in results]
    test_acc = [r["test_acc"] for r in results]
    out = {
        "n_folds": int(cfg.train.n_folds),
        "mean_test_acc": float(np.mean(test_acc)),
        "std_test_acc": float(np.std(test_acc)),
        "mean_test_bal_acc": float(np.mean(test_bal)),
        "std_test_bal_acc": float(np.std(test_bal)),
        "folds": results,
    }
    save_json(experiment_dir / "cv_summary.json", out)
    logger.info(
        f"CV summary | acc {out['mean_test_acc']:.4f}±{out['std_test_acc']:.4f} | "
        f"bal {out['mean_test_bal_acc']:.4f}±{out['std_test_bal_acc']:.4f}"
    )
    return out
