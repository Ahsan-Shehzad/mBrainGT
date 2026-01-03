from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mbrain_gt.config import load_config
from mbrain_gt.inference import compare_methods_auc_with_fdr, run_inference_on_split_dir
from mbrain_gt.utils import get_logger, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=False)
    parser.add_argument("--split_dir", type=str, required=False)
    parser.add_argument("--out_dir", type=str, required=False)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--auc_stats_out", type=str, default=None)
    parser.add_argument("--primary_method", type=str, default="mBrainGT")
    parser.add_argument("--method", action="append", default=[], help="name=run_dir (repeatable)")
    parser.add_argument("--fdr_q", type=float, default=0.05)

    args = parser.parse_args()
    logger = get_logger("scripts.inference")

    cfg = load_config(Path(args.config))
    device = torch.device(args.device)

    if args.split_dir and args.ckpt:
        out_dir = Path(args.out_dir) if args.out_dir else (Path(args.split_dir) / "inference_out")
        run_inference_on_split_dir(
            cfg=cfg,
            ckpt_path=Path(args.ckpt),
            split_dir=Path(args.split_dir),
            out_dir=out_dir,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            amp=args.amp,
        )
    else:
        logger.info("Skipping direct inference (provide --ckpt and --split_dir).")

    if args.auc_stats_out and args.method:
        method_to_dir = {}
        for spec in args.method:
            if "=" not in spec:
                raise ValueError(f"--method must be name=run_dir, got: {spec}")
            name, run_dir = spec.split("=", 1)
            method_to_dir[name.strip()] = Path(run_dir.strip())

        stats = compare_methods_auc_with_fdr(method_to_run_dir=method_to_dir, primary_method=args.primary_method, q=float(args.fdr_q))
        save_json(Path(args.auc_stats_out), stats)
        logger.info(f"Wrote AUC stats to: {args.auc_stats_out}")


if __name__ == "__main__":
    main()
