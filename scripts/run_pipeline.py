#!/usr/bin/env python3
"""Run the paper's training, comparison, analysis, and figure stages."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


REPOSITORY = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pet", type=Path, default=REPOSITORY / "data/raw/adni_pet.csv")
    parser.add_argument("--cognition", type=Path, default=REPOSITORY / "data/raw/adni_cognition.csv")
    parser.add_argument("--work-dir", type=Path, default=REPOSITORY / "runs/paper")
    parser.add_argument("--config", type=Path, default=REPOSITORY / "configs/training.json")
    parser.add_argument("--folds", type=int, help="Override the number of cross-validation folds")
    parser.add_argument("--parallel", type=int,
                        help="Maximum simultaneous folds; capped by available CPUs and fold count")
    parser.add_argument("--mode", choices=["full", "refit"], default="full",
                        help="Run the configured search budget, or refit the published configuration")
    parser.add_argument("--stage", choices=["all", "training", "comparison", "analysis", "figures"], default="all")
    parser.add_argument("--rscript", default="Rscript", help="Rscript executable")
    args = parser.parse_args()
    work = args.work_dir.resolve()
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    commands = {
        "training": ["train.py", "--raw-input", str(args.pet.resolve()), "--work-dir", str(work),
                     "--config", str(args.config.resolve()), "--stage", "all" if args.mode == "full" else "refit-selected"],
        "comparison": ["compare.py", "--work-dir", str(work)],
        "analysis": ["analyze.py", "--work-dir", str(work), "--cognition", str(args.cognition.resolve()),
                     "--rscript", args.rscript],
        "figures": ["figures.py", "--work-dir", str(work), "--rscript", args.rscript],
    }
    for option in ("folds", "parallel"):
        value = getattr(args, option)
        if value is not None:
            commands["training"].extend([f"--{option}", str(value)])
    stages = list(commands) if args.stage == "all" else [args.stage]
    for stage in stages:
        script, *arguments = commands[stage]
        print(f"Running {stage}", flush=True)
        subprocess.run([sys.executable, "-B", str(REPOSITORY / "scripts" / script), *arguments],
                       cwd=REPOSITORY, env=env, check=True)


if __name__ == "__main__":
    main()
