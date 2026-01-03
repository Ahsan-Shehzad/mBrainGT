from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from mbrain_gt.config import load_config
from mbrain_gt.utils import get_logger, save_json


def main() -> None:
    """Module 1: fMRI_Preprocessing (practical wrapper).

    This script is provided as a *working scaffold*:
      - If you already have preprocessed fMRI (recommended), create a manifest CSV pointing to files.
      - If you want full preprocessing with Nipype+FSL+SPM, integrate your site-specific pipeline here.

    Output:
      data/derived/preproc/manifest.csv with columns:
        subject_id, fmri_preproc_path, mean_fd(optional)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input_manifest", type=str, default="", help="Optional existing manifest with subject_id,fmri_path.")
    parser.add_argument("--out_manifest", type=str, default="", help="Output manifest path (default: data/derived/preproc/manifest.csv)")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    logger = get_logger("scripts.fMRI_Preprocessing")

    out_manifest = Path(args.out_manifest) if args.out_manifest else (cfg.paths.derived_root / "preproc" / "manifest.csv")
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    if args.input_manifest:
        inp = pd.read_csv(args.input_manifest)
        if "subject_id" not in inp.columns:
            raise ValueError("input_manifest must contain subject_id column.")
        fmri_col = "fmri_preproc_path" if "fmri_preproc_path" in inp.columns else ("fmri_path" if "fmri_path" in inp.columns else None)
        if fmri_col is None:
            raise ValueError("input_manifest must contain fmri_preproc_path or fmri_path column.")
        df = pd.DataFrame({
            "subject_id": inp["subject_id"].astype(str),
            "fmri_preproc_path": inp[fmri_col].astype(str),
        })
        if "mean_fd" in inp.columns:
            df["mean_fd"] = inp["mean_fd"].astype(float)
        df.to_csv(out_manifest, index=False)
        logger.info(f"Wrote preprocessing manifest to: {out_manifest}")
        return

    raise RuntimeError(
        "No preprocessing was executed. Provide --input_manifest pointing to preprocessed fMRI paths "
        "or implement Nipype+FSL+SPM preprocessing in this script."
    )


if __name__ == "__main__":
    main()
