from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import nibabel as nib

from mbrain_gt.config import load_config
from mbrain_gt.utils import get_logger, save_json


def _load_nifti(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    data = img.get_fdata()
    return data


def _roi_centroid_mni(roi_mask: np.ndarray, affine: np.ndarray) -> Tuple[float, float, float]:
    idx = np.argwhere(roi_mask)
    if idx.size == 0:
        return (0.0, 0.0, 0.0)
    ijk = idx.mean(axis=0)
    xyz = nib.affines.apply_affine(affine, ijk)
    return (float(xyz[0]), float(xyz[1]), float(xyz[2]))


def main() -> None:
    """Module 2: Modular Brain Mapping.

    Produces:
      - roi_to_module (R,) mapping each ROI label -> module id in [0, M-1]
      - roi_mni (R,3) centroid coords in MNI space
      - module_slices: list of index arrays for each module

    Inputs:
      - Brainnetome ROI atlas NIfTI (label image with ROI ids 1..R)
      - Yeo7 network atlas NIfTI (label image with network ids 1..M)

    Default expected paths (override via CLI):
      data/raw/atlases/brainnetome_246.nii.gz
      data/raw/atlases/yeo2011_7networks.nii.gz

    Output:
      data/derived/mapping/mapping.json
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--roi_atlas", type=str, default="data/raw/atlases/brainnetome_246.nii.gz")
    parser.add_argument("--module_atlas", type=str, default="data/raw/atlases/yeo2011_7networks.nii.gz")
    parser.add_argument("--out", type=str, default="", help="Output JSON path (default: data/derived/mapping/mapping.json)")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    logger = get_logger("scripts.Modular_Brain_Mapping")

    roi_atlas_path = Path(args.roi_atlas)
    module_atlas_path = Path(args.module_atlas)
    if not roi_atlas_path.exists():
        raise FileNotFoundError(roi_atlas_path)
    if not module_atlas_path.exists():
        raise FileNotFoundError(module_atlas_path)

    roi_img = nib.load(str(roi_atlas_path))
    roi_lab = roi_img.get_fdata().astype(np.int32)
    mod_img = nib.load(str(module_atlas_path))
    mod_lab = mod_img.get_fdata().astype(np.int32)

    # Assumption: same space/grid. If not, user should resample beforehand.
    if roi_lab.shape != mod_lab.shape:
        raise ValueError("ROI atlas and module atlas must be in same voxel grid (shape mismatch).")

    R = int(cfg.atlas.r_rois)
    M = int(cfg.atlas.m_modules)

    roi_to_module = np.zeros((R,), dtype=np.int64)
    roi_mni = np.zeros((R, 3), dtype=np.float32)

    for r in range(1, R + 1):
        mask = (roi_lab == r)
        # centroid in MNI
        roi_mni[r - 1, :] = np.array(_roi_centroid_mni(mask, roi_img.affine), dtype=np.float32)

        # overlap with modules (labels expected 1..M); assign by max overlap
        overlaps = np.zeros((M,), dtype=np.int64)
        for m in range(1, M + 1):
            overlaps[m - 1] = int(np.logical_and(mask, mod_lab == m).sum())
        roi_to_module[r - 1] = int(np.argmax(overlaps))

    module_slices: List[np.ndarray] = []
    for m in range(M):
        module_slices.append(np.where(roi_to_module == m)[0].astype(np.int64))

    out_path = Path(args.out) if args.out else (cfg.paths.derived_root / "mapping" / "mapping.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "R": R,
        "M": M,
        "roi_atlas": str(roi_atlas_path),
        "module_atlas": str(module_atlas_path),
        "roi_to_module": roi_to_module.tolist(),
        "roi_mni": roi_mni.tolist(),
        "module_slices": [s.tolist() for s in module_slices],
    }
    save_json(out_path, payload)
    logger.info(f"Wrote mapping to: {out_path}")


if __name__ == "__main__":
    main()
