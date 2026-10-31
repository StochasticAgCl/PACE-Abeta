#!/usr/bin/env python3
"""Prepare regional PET data, tune PACE-Aβ, and evaluate a frozen model."""
import os
import sys
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

import argparse
from pathlib import Path
REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / 'src'))
from pl_staging.runner import Experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=REPOSITORY / 'configs/training.json')
    parser.add_argument('--raw-input', required=True, type=Path,
                        help='Joined, unadjusted ADNI regional PET CSV')
    parser.add_argument('--work-dir', required=True, type=Path,
                        help='Run directory for prepared inputs, trials, models, and reports')
    parser.add_argument('--folds', type=int, help='Override the number of cross-validation folds')
    parser.add_argument('--parallel', type=int,
                        help='Maximum simultaneous folds; capped by available CPUs and fold count')
    parser.add_argument('--stage', required=True, choices=[
        'prepare', 'smoke', 'search', 'freeze', 'select-fixed', 'refit', 'test', 'audit',
        'all', 'refit-selected'])
    args = parser.parse_args()
    experiment = Experiment(args.config, args.work_dir, args.raw_input,
                            folds=args.folds, parallel=args.parallel)
    actions = {'all': ('prepare', 'search', 'freeze', 'refit', 'test', 'audit'),
               'refit-selected': ('prepare', 'select_fixed', 'refit', 'test', 'audit')}
    for action in actions.get(args.stage, (args.stage.replace('-', '_'),)):
        getattr(experiment, action)()


if __name__ == '__main__':
    main()
