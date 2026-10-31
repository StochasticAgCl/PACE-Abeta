#!/usr/bin/env python3
"""Score amyloid PET scans using a pretrained model and a training reference."""
from pathlib import Path
import argparse
import json
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / 'src'))

import numpy as np
import pandas as pd
import torch

from pl_staging.model import ProbabilisticAE, score_model
from pl_staging.preprocessing import apply_preadjust_params_simple
from pace_abeta.analysis.calibration import fit_rank_map, apply_rank_map

COVARIATES = ['PHC_Sex', 'PHC_Education']
OUTPUT_METADATA = ['RID', 'PHC_Age_PET', 'PHC_SCANDATE']


def load_bundle(model_path, preprocessing_path):
    """Load neural-network weights and their fitted regional adjustment."""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    if set(checkpoint) != {'model_state_dict', 'config', 'input_dim'}:
        raise ValueError('Checkpoint must contain model_state_dict, config, and input_dim')
    prep = json.loads(Path(preprocessing_path).read_text())
    if set(prep) != {'features', 'betas', 'means', 'imputation'}:
        raise ValueError('Preprocessing must contain features, betas, means, and imputation')
    features = prep['features']
    if len(features) != 68 or len(set(features)) != 68 or checkpoint['input_dim'] != 68:
        raise ValueError('The model and preprocessing must specify the same 68 features')
    arrays = {name: np.asarray(prep[name], dtype=np.float64) for name in ('betas', 'means', 'imputation')}
    for name, shape in [('betas', (68, 3)), ('means', (68, 2)), ('imputation', (2,))]:
        if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
            raise ValueError(f'Invalid preprocessing array: {name}')
    if not all(isinstance(v, torch.Tensor) and bool(torch.isfinite(v).all())
               for v in checkpoint['model_state_dict'].values()):
        raise ValueError('Model weights must be finite tensors')
    model = ProbabilisticAE(checkpoint['input_dim'], checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, dict(features=features, **arrays)


def preprocess(frame, preparation):
    """Adjust supplied raw SUVRs using the bundle's fixed training coefficients."""
    features = preparation['features']
    required = features + COVARIATES
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f'Missing PET input columns: {missing}')
    if not len(frame):
        raise ValueError('The PET input table is empty')
    values = frame[features].apply(pd.to_numeric, errors='raise').to_numpy(dtype=np.float64)
    covariates = frame[COVARIATES].apply(pd.to_numeric, errors='raise').to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('All regional SUVRs must be observed and finite')
    if np.isinf(covariates).any():
        raise ValueError('Sex and education must be finite or missing')
    return apply_preadjust_params_simple(values, covariates,
        preparation['betas'], preparation['means'], cov_impute_means=preparation['imputation'])


def predict(input_frame, reference_frame, model, preparation):
    """Calibrate posterior means with participant-weighted reference midranks."""
    if 'RID' not in reference_frame or reference_frame.RID.isna().any():
        raise ValueError('The reference table requires a nonmissing RID for every visit')
    subjects = reference_frame.RID.astype(str).to_numpy()
    if any(not identifier.strip() for identifier in subjects):
        raise ValueError('Reference participant identifiers cannot be empty')
    reference_scores = score_model(model, preprocess(reference_frame, preparation))
    mapping = fit_rank_map(reference_scores['stage_mean'], subjects)
    scores = score_model(model, preprocess(input_frame, preparation))
    output = input_frame[[name for name in OUTPUT_METADATA if name in input_frame]].copy()
    output['PACE_Abeta_score'] = apply_rank_map(scores['stage_mean'], mapping)
    output['posterior_mean'] = scores['stage_mean']
    output['posterior_variance'] = scores['stage_variance']
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='Raw regional PET visits to score')
    parser.add_argument('--reference', required=True, type=Path,
                        help='Authorized training-reference visits for participant-weighted rank calibration')
    parser.add_argument('--output', required=True, type=Path, help='Destination score CSV')
    parser.add_argument('--model', type=Path, default=REPOSITORY / 'models/pace_abeta.pt')
    parser.add_argument('--preprocessing', type=Path, default=REPOSITORY / 'models/preprocessing.json')
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination in {args.input.resolve(), args.reference.resolve(), args.model.resolve(), args.preprocessing.resolve()}:
        parser.error('--output must differ from each input file')
    torch.set_num_threads(1)
    model, prep = load_bundle(args.model, args.preprocessing)
    result = predict(pd.read_csv(args.input), pd.read_csv(args.reference), model, prep)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f'Saved {len(result)} visit scores to {args.output}')


if __name__ == '__main__':
    main()
