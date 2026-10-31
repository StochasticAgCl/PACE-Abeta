#!/usr/bin/env python3
"""Calibrate scores and run downstream manuscript analyses."""
from pathlib import Path
import argparse
import json
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--cognition", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--centiloid-max", type=float, default=30.)
    parser.add_argument("--rscript", default="Rscript")
    args = parser.parse_args()
    cache = args.work_dir.resolve() / "tmp" / "matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    from pace_abeta.analysis import run_analysis
    result = run_analysis(args.work_dir, args.cognition, args.seed, args.centiloid_max, args.rscript)
    print(json.dumps({"cohort": result["cohort"], "temporal": result["temporal"],
                      "memory": result["memory"]}, indent=2))


if __name__ == "__main__":
    main()
