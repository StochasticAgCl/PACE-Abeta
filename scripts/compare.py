#!/usr/bin/env python3
"""Fit the fixed comparators and write aligned four-method score tables."""

import argparse

from pace_abeta.comparators import run_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True,
                        help="Prepared run directory, for example runs/paper, or an external data directory")
    args = parser.parse_args()
    result = run_comparison(args.work_dir)
    print(f"Comparison complete: {result['inventory']}")


if __name__ == "__main__":
    main()
