from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mbrain_gt.config import load_config
from mbrain_gt.training import set_seed, train_all_folds, train_fold
from mbrain_gt.utils import get_logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--experiment", type=str, default="mBrainGT", help="Output folder name under data/derived/runs/")
    parser.add_argument("--fold", type=int, default=-1, help="Fold id to train. -1 trains all folds.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true", help="Resume from last.pt if available.")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    logger = get_logger("scripts.training")

    set_seed(int(cfg.train.seed))
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    exp_dir = (cfg.paths.derived_root / "runs" / args.experiment).resolve()
    exp_dir.mkdir(parents=True, exist_ok=True)

    if args.fold >= 0:
        train_fold(cfg, fold_id=int(args.fold), device=device, experiment_dir=exp_dir, resume=args.resume)
    else:
        train_all_folds(cfg, device=device, experiment_dir=exp_dir, resume=args.resume)


if __name__ == "__main__":
    main()
