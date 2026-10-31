#!/usr/bin/env python3
"""Render result figures from a completed downstream analysis."""
from pathlib import Path
import argparse
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--rscript", default="Rscript")
    args = parser.parse_args()
    cache = args.work_dir.resolve() / "tmp" / "matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    from pace_abeta.analysis.figures import render_figures
    print(render_figures(args.work_dir, rscript=args.rscript))


if __name__ == "__main__":
    main()
