from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from mbrain_gt.config import load_config
from mbrain_gt.dataio import save_graph_package
from mbrain_gt.utils import get_logger, load_json, save_json


def _load_npz(path: Path) -> Dict[str, Any]:
    obj = np.load(path, allow_pickle=True)
    return {k: obj[k] for k in obj.files}


def _combat_harmonize_train_apply(train_X: np.ndarray, test_X: np.ndarray, train_covars: pd.DataFrame, test_covars: pd.DataFrame, batch_col: str) -> Tuple[np.ndarray, np.ndarray]:
    """Fold-safe ComBat using neuroHarmonize. Fits on train, applies to train+test."""
    try:
        from neuroHarmonize import harmonizationLearn, harmonizationApply  # type: ignore
    except Exception as e:
        raise RuntimeError("neuroHarmonize is required for ComBat. Install it or disable harmonization.") from e

    model = harmonizationLearn(train_X, train_covars, batch_col)
    train_X_h = harmonizationApply(train_X, train_covars, model)
    test_X_h = harmonizationApply(test_X, test_covars, model)
    return train_X_h, test_X_h


def main() -> None:
    """Module 4: Modular brain network construction and fold package writing.

    Inputs:
      --metadata: TSV/CSV with at least:
        subject_id, y, site_id
        + optional covariates (age, sex, mean_fd)
      --connectivity_manifest: CSV with subject_id, connectivity_path (from Module 3)

    Output:
      data/derived/fold_packages/fold_k/{train,val,test}/subject_<id>.pt
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--connectivity_manifest", type=str, default="", help="Default: data/derived/connectivity/manifest.csv")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Fraction of train used as val within each fold.")
    parser.add_argument("--no_combat", action="store_true", help="Disable ComBat harmonization.")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    logger = get_logger("scripts.modular_brain_network_construction")

    meta_path = Path(args.metadata)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    if meta_path.suffix.lower() in [".tsv"]:
        meta = pd.read_csv(meta_path, sep="\t")
    else:
        meta = pd.read_csv(meta_path)

    required_cols = {"subject_id", "y", "site_id"}
    missing = required_cols - set(meta.columns)
    if missing:
        raise ValueError(f"metadata missing required columns: {sorted(list(missing))}")

    conn_manifest = Path(args.connectivity_manifest) if args.connectivity_manifest else (cfg.paths.derived_root / "connectivity" / "manifest.csv")
    if not conn_manifest.exists():
        raise FileNotFoundError(conn_manifest)
    conn = pd.read_csv(conn_manifest)
    if "subject_id" not in conn.columns or "connectivity_path" not in conn.columns:
        raise ValueError("connectivity manifest must contain subject_id and connectivity_path.")

    df = meta.merge(conn, on="subject_id", how="inner")
    if len(df) == 0:
        raise RuntimeError("No subjects after joining metadata and connectivity manifest.")

    # Prepare fold split (stratified on y)
    y = df["y"].astype(int).values
    skf = StratifiedKFold(n_splits=int(cfg.train.n_folds), shuffle=True, random_state=int(cfg.train.seed))

    out_root = cfg.paths.derived_root / "fold_packages"
    out_root.mkdir(parents=True, exist_ok=True)

    # Preload feature matrix for ComBat: concatenate all module upper-tri z values (per module) into one vector
    # This keeps the example simple and fold-safe.
    def fc_vectorize(npz: Dict[str, Any]) -> np.ndarray:
        adj_module = list(npz["adj_module"])  # object array of matrices
        feats = []
        for A in adj_module:
            A = np.asarray(A, dtype=np.float32)
            iu = np.triu_indices(A.shape[0], k=1)
            feats.append(A[iu])
        return np.concatenate(feats, axis=0).astype(np.float32)

    # Build vectors for all subjects
    vectors = []
    for p in df["connectivity_path"].astype(str).tolist():
        npz = _load_npz(Path(p))
        vectors.append(fc_vectorize(npz))
    X = np.stack(vectors, axis=0)  # (N, p)

    # Covariates for ComBat
    cov_cols = [c for c in cfg.connectivity.combat_covariates if c in df.columns]
    covars = df[["site_id"] + cov_cols].copy()
    covars.rename(columns={"site_id": "batch"}, inplace=True)

    # fold loop
    for fold_id, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        fold_dir = out_root / f"fold_{fold_id}"
        (fold_dir / "train").mkdir(parents=True, exist_ok=True)
        (fold_dir / "val").mkdir(parents=True, exist_ok=True)
        (fold_dir / "test").mkdir(parents=True, exist_ok=True)

        # split train into train/val
        train_idx = np.asarray(train_idx)
        n_train = len(train_idx)
        n_val = max(1, int(round(n_train * float(args.val_ratio))))
        # deterministic split: last n_val for val
        perm = np.random.RandomState(int(cfg.train.seed) + fold_id).permutation(n_train)
        val_sel = perm[:n_val]
        tr_sel = perm[n_val:]
        tr_idx = train_idx[tr_sel]
        va_idx = train_idx[val_sel]

        # ComBat harmonization (fit on train only)
        X_h = X.copy()
        if not args.no_combat:
            X_tr = X[tr_idx]
            X_va = X[va_idx]
            X_te = X[test_idx]

            cov_tr = covars.iloc[tr_idx].reset_index(drop=True)
            cov_va = covars.iloc[va_idx].reset_index(drop=True)
            cov_te = covars.iloc[test_idx].reset_index(drop=True)

            X_tr_h, X_va_h = _combat_harmonize_train_apply(X_tr, X_va, cov_tr, cov_va, batch_col="batch")
            _, X_te_h = _combat_harmonize_train_apply(X_tr, X_te, cov_tr, cov_te, batch_col="batch")

            X_h[tr_idx] = X_tr_h
            X_h[va_idx] = X_va_h
            X_h[test_idx] = X_te_h

        # Write packages
        def write_split(indices: np.ndarray, split_name: str) -> None:
            for i in indices:
                row = df.iloc[int(i)]
                sid = str(row["subject_id"])
                label = int(row["y"])
                npz = _load_npz(Path(row["connectivity_path"]))

                roi_ts = np.asarray(npz["roi_ts"], dtype=np.float32)  # (T,R)
                degree = np.asarray(npz["degree"], dtype=np.float32)
                betw = np.asarray(npz["betweenness"], dtype=np.float32)
                close = np.asarray(npz["closeness"], dtype=np.float32)
                roi_mni = np.asarray(npz["roi_mni"], dtype=np.float32)

                # Node raw feature = [z_v ; d_v ; b_v ; c_v]
                # z_v is the ROI time series vector (length T) for ROI v.
                # Build x_raw: (R, T+3)
                # roi_ts is (T,R) -> transpose to (R,T)
                z = roi_ts.T  # (R,T)
                x_raw = np.concatenate([z, degree[:, None], betw[:, None], close[:, None]], axis=1).astype(np.float32)

                module_slices = [np.asarray(s, dtype=np.int64) for s in npz["module_slices"]]
                adj_module = [np.asarray(a, dtype=np.float32) for a in npz["adj_module"]]
                pval_module = [np.asarray(p, dtype=np.float32) for p in npz["pval_module"]]
                global_fc = np.asarray(npz["global_fc"], dtype=np.float32)

                payload = {
                    "subject_id": sid,
                    "y": label,
                    "x_raw": x_raw,
                    "pos": roi_mni,
                    "module_slices": [s.tolist() for s in module_slices],
                    "adj_module": [a for a in adj_module],
                    "pval_module": [p for p in pval_module],
                    "global_fc": global_fc,
                    # Optional: store harmonized FC vector if you want to debug
                    "combat_fc_vector": X_h[int(i)].astype(np.float32),
                }

                out_path = fold_dir / split_name / f"subject_{sid}.pt"
                save_graph_package(out_path, payload)

        write_split(tr_idx, "train")
        write_split(va_idx, "val")
        write_split(np.asarray(test_idx), "test")

        logger.info(f"[fold {fold_id}] wrote train={len(tr_idx)} val={len(va_idx)} test={len(test_idx)} packages to {fold_dir}")

    logger.info(f"All folds written to: {out_root}")


if __name__ == "__main__":
    main()
