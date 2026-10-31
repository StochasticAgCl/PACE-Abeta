"""Raw-input eligibility, preprocessing leakage, and subject-level splits."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from pl_staging import data
from pl_staging.io import read

ROOT = Path(__file__).resolve().parents[1]


def synthetic_visits():
    features = read(ROOT / 'configs/features.json')
    rng = np.random.default_rng(42)
    rows = []
    for rid in range(60):
        for visit in range(2 if rid < 52 else 1):
            row = dict(RID=rid, PHC_Age_PET=65+rid/10+visit,
                       PHC_Diagnosis=1, PHC_Sex=1+rid%2, PHC_Education=12+rid%7,
                       PHC_TRACER='FBP', PHC_Race=5, PHC_Ethnicity=2,
                       PHC_SCANDATE=f'202{visit}-01-01', PHC_CENTILOIDS=float(rid))
            row.update(zip(features, rng.normal(1.2, .1, len(features))))
            rows.append(row)
    return pd.DataFrame(rows), features


def specification(features):
    return dict(seed=42, split_seed=42, fold_seed=42, folds=12,
                cohort_ready=False, features=features)


@pytest.mark.parametrize('folds', [10, 12, 20])
def test_raw_preparation_is_subject_separated_and_fold_local(tmp_path, folds):
    frame, features = synthetic_visits()
    raw = tmp_path / 'visits.csv'
    frame.to_csv(raw, index=False)
    spec = specification(features)
    spec['folds'] = folds
    first = data.prepare(tmp_path / 'one', raw, spec)
    prep = tmp_path / 'one/data/prepared'
    train = pd.read_csv(prep / 'train_input.csv')
    test = pd.read_csv(prep / 'test_input.csv')
    index = pd.read_csv(prep / 'scan_index.csv')
    assert set(train.RID).isdisjoint(test.RID)
    assert set(range(52, 60)).issubset(train.RID)
    assert index.loc[index.RID.ge(52), 'fold'].eq(-1).all()
    assert index.groupby('RID').fold.nunique().eq(1).all()
    assert set(index.loc[index.fold.ge(0), 'fold']) == set(range(folds))
    assert first['folds'] == folds
    assert len(first['fold_inventory']) == folds
    assert first['pooled_cv_pairs'] == 26
    assert first['test_pairs'] == 26
    assert 'PHC_SCANDATE' in train and 'PHC_CENTILOIDS' in train
    assert data.prepare(tmp_path / 'one', raw, spec) == first
    with pytest.raises(RuntimeError, match='settings have changed'):
        data.prepare(tmp_path / 'one', raw, dict(spec, folds=folds + 1))
    for invalid_fold in (-1, folds):
        with pytest.raises(ValueError, match='outside the prepared range'):
            data.load_fold(tmp_path / 'one', invalid_fold)
    x1, _, held = data.load_fold(tmp_path / 'one', 0)
    altered = frame.copy()
    held_ids = set(index.loc[held, 'RID'])
    altered.loc[altered.RID.isin(held_ids), features] += 100.
    altered.to_csv(tmp_path / 'altered.csv', index=False)
    data.prepare(tmp_path / 'two', tmp_path / 'altered.csv', spec)
    x2, _, held2 = data.load_fold(tmp_path / 'two', 0)
    np.testing.assert_array_equal(held, held2)
    np.testing.assert_array_equal(x1[~held], x2[~held])
    with np.load(prep / 'folds/fold0_preprocessing.npz') as a, np.load(tmp_path / 'two/data/prepared/folds/fold0_preprocessing.npz') as b:
        for key in ('betas', 'means', 'imputation'):
            np.testing.assert_array_equal(a[key], b[key])
    with pytest.raises(RuntimeError, match='changed'):
        data.prepare(tmp_path / 'one', tmp_path / 'altered.csv', spec)


def test_eligibility_and_explicit_cohort_ready_schema(tmp_path):
    frame, features = synthetic_visits()
    frame.loc[0, 'PHC_TRACER'] = 'FBB'
    frame.loc[1, 'PHC_Race'] = 4
    frame.loc[2, 'PHC_Ethnicity'] = 1
    frame.loc[3, features[0]] = np.nan
    frame.to_csv(tmp_path / 'raw.csv', index=False)
    eligible, summary = data.read_visits(tmp_path / 'raw.csv', features)
    assert summary['excluded_rows'] == 4
    assert len(eligible) == len(frame)-4
    ready = frame.drop(columns=['PHC_TRACER', 'PHC_Race', 'PHC_Ethnicity'])
    ready.to_csv(tmp_path / 'ready.csv', index=False)
    with pytest.raises(ValueError, match='Missing input columns'):
        data.read_visits(tmp_path / 'ready.csv', features)
    parsed, _ = data.read_visits(tmp_path / 'ready.csv', features, cohort_ready=True)
    assert len(parsed) == len(frame)-1


def test_duplicate_visits_are_rejected(tmp_path):
    frame, features = synthetic_visits()
    pd.concat([frame, frame.iloc[[0]]]).to_csv(tmp_path / 'duplicate.csv', index=False)
    with pytest.raises(ValueError, match='Duplicate'):
        data.read_visits(tmp_path / 'duplicate.csv', features)


@pytest.mark.parametrize('cohort_ready', [False, True])
def test_baseline_diagnosis_excludes_whole_subjects_before_split(tmp_path, cohort_ready):
    frame, features = synthetic_visits()
    baseline = frame.groupby('RID', sort=True).head(1).index
    frame.loc[baseline[0], 'PHC_Diagnosis'] = np.nan
    frame.loc[baseline[1], 'PHC_Diagnosis'] = 4
    # Valid baseline diagnoses retain later visits with missing or other diagnoses.
    frame.loc[baseline[2] + 1, 'PHC_Diagnosis'] = np.nan
    frame.loc[baseline[3], 'PHC_Diagnosis'] = 3
    frame.loc[baseline[3] + 1, 'PHC_Diagnosis'] = 4
    # A PET row excluded by feature screening does not define baseline.
    frame.loc[baseline[4], 'PHC_Diagnosis'] = 4
    frame.loc[baseline[4], features[0]] = np.nan
    raw = tmp_path / 'visits.csv'
    frame.sample(frac=1, random_state=7).to_csv(raw, index=False)
    eligible, summary = data.read_visits(raw, features, cohort_ready=cohort_ready)
    assert set(eligible.RID) == set(range(2, 60))
    assert len(eligible.loc[eligible.RID.eq(2)]) == 2
    assert len(eligible.loc[eligible.RID.eq(3)]) == 2
    assert len(eligible.loc[eligible.RID.eq(4)]) == 1
    assert summary['baseline_diagnosis_excluded_subjects'] == 2
    assert summary['baseline_diagnosis_missing_subjects'] == 1
    assert summary['baseline_diagnosis_other_subjects'] == 1
    assert summary['baseline_diagnosis_excluded_rows'] == 4
    assert summary['excluded_rows'] == 5

    root = tmp_path / 'run'
    spec = dict(specification(features), cohort_ready=cohort_ready)
    metadata = data.prepare(root, raw, spec)
    prepared = root / 'data/prepared'
    for name in ('train_input', 'test_input', 'subject_partition', 'subject_folds', 'scan_index'):
        assert {0, 1}.isdisjoint(pd.read_csv(prepared / f'{name}.csv').RID)
    sizes = eligible.groupby('RID', sort=True).size()
    expected_test = sizes[sizes.gt(1)].sample(frac=.5, random_state=42).index
    actual_test = pd.read_csv(prepared / 'test_input.csv').RID.unique()
    assert set(actual_test) == set(expected_test)
    assert metadata['eligibility'] == summary
    assert metadata['preparation_spec']['baseline_diagnosis_codes'] == [1, 2, 3]


def test_all_invalid_baseline_diagnoses_are_rejected(tmp_path):
    frame, features = synthetic_visits()
    frame.loc[frame.groupby('RID').head(1).index, 'PHC_Diagnosis'] = np.nan
    frame.to_csv(tmp_path / 'visits.csv', index=False)
    with pytest.raises(ValueError, match='No eligible participants have baseline'):
        data.read_visits(tmp_path / 'visits.csv', features)
