"""Pretrained preprocessing, score calibration, and input validation."""
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location('pace_predict', ROOT / 'scripts/predict.py')
predict_cli = module_from_spec(SPEC)
SPEC.loader.exec_module(predict_cli)


@pytest.fixture
def bundle():
    torch.set_num_threads(1)
    return predict_cli.load_bundle(ROOT / 'models/pace_abeta.pt', ROOT / 'models/preprocessing.json')


def frame(preparation, seed=42):
    rng = np.random.default_rng(seed)
    result = pd.DataFrame(rng.normal(1.2, .12, (6, 68)), columns=preparation['features'])
    result['PHC_Sex'] = [1., 2., np.nan, 2., 1., 2.]
    result['PHC_Education'] = [12., 16., 18., np.nan, 13., 16.]
    result['RID'] = ['person1', 'person1', 'person2', 'person3', 'person3', 'person3']
    return result


def test_bundle_contains_only_model_and_aggregate_preprocessing():
    checkpoint = torch.load(ROOT / 'models/pace_abeta.pt', map_location='cpu', weights_only=True)
    assert set(checkpoint) == {'model_state_dict', 'config', 'input_dim'}
    assert checkpoint['input_dim'] == 68
    configured = json.loads((ROOT / 'configs/training.json').read_text())['selected_model']
    assert checkpoint['config'] == configured
    assert all(isinstance(value, torch.Tensor) for value in checkpoint['model_state_dict'].values())
    prep = json.loads((ROOT / 'models/preprocessing.json').read_text())
    assert set(prep) == {'features', 'betas', 'means', 'imputation'}
    assert prep['features'] == json.loads((ROOT / 'configs/features.json').read_text())
    assert np.asarray(prep['betas']).shape == (68, 3)
    assert np.asarray(prep['means']).shape == (68, 2)
    assert np.asarray(prep['imputation']).shape == (2,)


def test_fixed_preprocessing_is_independent_of_other_target_rows(bundle):
    _, prep = bundle
    visits = frame(prep)
    separate = predict_cli.preprocess(visits.iloc[[2]], prep)
    full = predict_cli.preprocess(visits, prep)
    np.testing.assert_array_equal(separate[0], full[2])
    imputed = visits.copy()
    for j, name in enumerate(predict_cli.COVARIATES):
        imputed[name] = imputed[name].fillna(prep['imputation'][j])
    np.testing.assert_array_equal(full, predict_cli.preprocess(imputed, prep))


def test_prediction_uses_fixed_participant_weighted_reference(bundle):
    model, prep = bundle
    reference = frame(prep)
    target = frame(prep, seed=13)
    result = predict_cli.predict(target, reference, model, prep)
    ref = predict_cli.score_model(model, predict_cli.preprocess(reference, prep))['stage_mean']
    mean = predict_cli.score_model(model, predict_cli.preprocess(target, prep))['stage_mean']
    mapping = predict_cli.fit_rank_map(ref, reference.RID.to_numpy())
    np.testing.assert_array_equal(result.PACE_Abeta_score, predict_cli.apply_rank_map(mean, mapping))
    np.testing.assert_array_equal(result.posterior_mean, mean)
    assert result.posterior_variance.gt(0).all()
    assert set(result) == {'RID', 'PACE_Abeta_score', 'posterior_mean', 'posterior_variance'}
    # Repeating every reference visit preserves each participant's total mass.
    repeated = pd.concat([reference, reference], ignore_index=True)
    duplicate_result = predict_cli.predict(target, repeated, model, prep)
    np.testing.assert_allclose(result.PACE_Abeta_score, duplicate_result.PACE_Abeta_score, rtol=0, atol=1e-14)


def test_missing_reference_identifier_and_incomplete_suvr_are_rejected(bundle):
    model, prep = bundle
    visits = frame(prep)
    with pytest.raises(ValueError, match='RID'):
        predict_cli.predict(visits, visits.drop(columns='RID'), model, prep)
    missing = visits.copy()
    missing.loc[0, prep['features'][0]] = np.nan
    with pytest.raises(ValueError, match='observed and finite'):
        predict_cli.preprocess(missing, prep)
    with pytest.raises(ValueError, match='Missing PET input columns'):
        predict_cli.preprocess(visits.drop(columns='PHC_Sex'), prep)


def test_scoring_does_not_require_target_ids_or_temporal_information(bundle):
    model, prep = bundle
    reference = frame(prep)
    target = frame(prep, seed=91).drop(columns='RID')
    result = predict_cli.predict(target, reference, model, prep)
    assert len(result) == len(target)
    assert set(result) == {'PACE_Abeta_score', 'posterior_mean', 'posterior_variance'}
