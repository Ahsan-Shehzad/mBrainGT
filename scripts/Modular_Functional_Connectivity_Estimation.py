from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import nibabel as nib
import networkx as nx
from nilearn.input_data import NiftiLabelsMasker
from scipy import stats

from mbrain_gt.config import load_config
from mbrain_gt.utils import get_logger, pearson_corr, save_json, load_json


def fisher_z(C: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    C = np.clip(C, -1 + eps, 1 - eps)
    return np.arctanh(C).astype(np.float32)


def compute_pvals_from_corr(C: np.ndarray, T: int) -> np.ndarray:
    """Approximate two-sided p-values for Pearson correlation via t-statistic."""
    # avoid divide by zero
    r = np.clip(C, -0.999999, 0.999999)
    df = max(T - 2, 1)
    t = r * np.sqrt(df / (1 - r**2))
    p = 2 * (1 - stats.t.cdf(np.abs(t), df=df))
    return p.astype(np.float32)


def centralities_from_adj(adj: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute degree, betweenness, closeness on weighted graph."""
    W = adj.copy()
    np.fill_diagonal(W, 0.0)
    G = nx.from_numpy_array(W)

    degree = np.array([val for _, val in G.degree(weight="weight")], dtype=np.float32)
    # for betweenness/closeness, need edge distance; use inverse weight as distance
    for u, v, d in G.edges(data=True):
        w = float(d.get("weight", 0.0))
        d["distance"] = 1.0 / (abs(w) + 1e-6)

    bet = nx.betweenness_centrality(G, weight="distance", normalized=True)
    clo = nx.closeness_centrality(G, distance="distance")
    bet = np.array([bet[i] for i in range(G.number_of_nodes())], dtype=np.float32)
    clo = np.array([clo[i] for i in range(G.number_of_nodes())], dtype=np.float32)
    return degree, bet, clo


def main() -> None:
    """Module 3: Modular Functional Connectivity Estimation.

    Inputs:
      - data/derived/preproc/manifest.csv with columns:
          subject_id, fmri_preproc_path
      - data/derived/mapping/mapping.json from Module 2

    Outputs:
      - data/derived/connectivity/manifest.csv with columns:
          subject_id, connectivity_path
      Each connectivity_path is a .npz containing:
        - roi_ts: (T,R) ROI time series
        - global_fc: (R,R) Pearson corr
        - roi_mni: (R,3)
        - roi_to_module: (R,)
        - module_slices: list of arrays
        - adj_module: list of (Rm,Rm) thresholded Fisher-z
        - pval_module: list of (Rm,Rm) p-values
        - degree/betweenness/closeness: (R,) node centralities
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--preproc_manifest", type=str, default="", help="Default: data/derived/preproc/manifest.csv")
    parser.add_argument("--mapping_json", type=str, default="", help="Default: data/derived/mapping/mapping.json")
    parser.add_argument("--out_manifest", type=str, default="", help="Default: data/derived/connectivity/manifest.csv")
    parser.add_argument("--t_target", type=int, default=200, help="Truncate/pad ROI time series length for downstream embedding.")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    logger = get_logger("scripts.Modular_Functional_Connectivity_Estimation")

    preproc_manifest = Path(args.preproc_manifest) if args.preproc_manifest else (cfg.paths.derived_root / "preproc" / "manifest.csv")
    mapping_json = Path(args.mapping_json) if args.mapping_json else (cfg.paths.derived_root / "mapping" / "mapping.json")
    out_manifest = Path(args.out_manifest) if args.out_manifest else (cfg.paths.derived_root / "connectivity" / "manifest.csv")

    if not preproc_manifest.exists():
        raise FileNotFoundError(preproc_manifest)
    if not mapping_json.exists():
        raise FileNotFoundError(mapping_json)

    mapping = load_json(mapping_json)
    roi_to_module = np.asarray(mapping["roi_to_module"], dtype=np.int64)
    roi_mni = np.asarray(mapping["roi_mni"], dtype=np.float32)
    module_slices = [np.asarray(s, dtype=np.int64) for s in mapping["module_slices"]]

    df = pd.read_csv(preproc_manifest)
    if "subject_id" not in df.columns or "fmri_preproc_path" not in df.columns:
        raise ValueError("preproc manifest must contain subject_id and fmri_preproc_path.")

    out_dir = cfg.paths.derived_root / "connectivity" / "subjects"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for _, row in df.iterrows():
        sid = str(row["subject_id"])
        fmri_path = Path(str(row["fmri_preproc_path"]))
        if not fmri_path.exists():
            logger.warning(f"Missing fMRI file for {sid}: {fmri_path}")
            continue

        # Extract ROI time series using ROI atlas labels image (Brainnetome).
        # User must ensure atlas path matches Module 2 roi_atlas used.
        roi_atlas_path = Path(mapping["roi_atlas"])
        if not roi_atlas_path.exists():
            raise FileNotFoundError(roi_atlas_path)

        masker = NiftiLabelsMasker(labels_img=str(roi_atlas_path), standardize=True, detrend=True)
        roi_ts = masker.fit_transform(str(fmri_path))  # (T,R)

        # enforce fixed length
        T, R = roi_ts.shape
        Tt = int(args.t_target)
        if T >= Tt:
            roi_ts_fixed = roi_ts[:Tt, :]
        else:
            pad = np.zeros((Tt - T, R), dtype=roi_ts.dtype)
            roi_ts_fixed = np.vstack([roi_ts, pad])

        # global FC
        global_fc = pearson_corr(roi_ts_fixed)

        # modular FC within each module
        adj_module: List[np.ndarray] = []
        pval_module: List[np.ndarray] = []
        for idx in module_slices:
            ts_m = roi_ts_fixed[:, idx]  # (Tt, Rm)
            C = pearson_corr(ts_m)
            P = compute_pvals_from_corr(C, T=Tt)
            Z = fisher_z(C)

            # threshold at p<0.05 (two-sided); keep sign
            Z_thr = Z.copy()
            Z_thr[P >= 0.05] = 0.0
            np.fill_diagonal(Z_thr, 0.0)

            adj_module.append(Z_thr.astype(np.float32))
            pval_module.append(P.astype(np.float32))

        # centralities from thresholded modular graphs aggregated back to ROI level
        degree = np.zeros((len(roi_to_module),), dtype=np.float32)
        betw = np.zeros((len(roi_to_module),), dtype=np.float32)
        close = np.zeros((len(roi_to_module),), dtype=np.float32)
        for m, idx in enumerate(module_slices):
            deg_m, bet_m, clo_m = centralities_from_adj(adj_module[m])
            degree[idx] = deg_m
            betw[idx] = bet_m
            close[idx] = clo_m

        out_path = out_dir / f"{sid}.npz"
        np.savez_compressed(
            out_path,
            subject_id=sid,
            roi_ts=roi_ts_fixed.astype(np.float32),
            global_fc=global_fc.astype(np.float32),
            roi_mni=roi_mni.astype(np.float32),
            roi_to_module=roi_to_module.astype(np.int64),
            degree=degree,
            betweenness=betw,
            closeness=close,
            # store modules as object arrays
            module_slices=np.array(module_slices, dtype=object),
            adj_module=np.array(adj_module, dtype=object),
            pval_module=np.array(pval_module, dtype=object),
        )

        records.append({"subject_id": sid, "connectivity_path": str(out_path)})

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out_manifest, index=False)
    logger.info(f"Wrote connectivity manifest: {out_manifest} (N={len(records)})")


if __name__ == "__main__":
    main()
