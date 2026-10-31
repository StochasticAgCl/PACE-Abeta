"""Literal VR-times-conditional-gap selection and frozen evaluation."""
from pathlib import Path

import optuna

import numpy as np
import pandas as pd
import pytest

from pace_abeta.analysis.calibration import apply_rank_map, fit_rank_map
from pace_abeta.analysis.metrics import adjacent_pairs, summarize_declines
from pl_staging import metrics, runner
from pl_staging.io import digest, dump, read
from test_training_data import synthetic_visits
from test_training_protocol import controlled_executor, make_experiment


PRODUCT = 'violation_ratio_times_conditional_gap'

def pair_frame(declines):
    declines = np.asarray(declines, dtype=float)
    frame = pd.DataFrame({'RID': np.repeat(np.arange(len(declines)), 2),
                          'PHC_Age_PET': np.tile([70., 71.], len(declines))})
    return frame, np.column_stack([declines, np.zeros(len(declines))]).ravel()


@pytest.mark.parametrize('declines', [[.4, .2, 0., -.1], [-.2, 0.], [.4], [0., 0.]])
def test_all_gap_identity_strict_ties_and_no_violations(declines):
    frame, scores = pair_frame(declines)
    result = metrics.gaps(frame, scores)
    expected = summarize_declines(declines)
    assert result == expected
    if result['n_violations']:
        assert result['all_violation_gap'] == pytest.approx(
            result['violation_ratio'] * result['conditional_violation_gap'])
    else:
        assert result['all_violation_gap'] == 0
        assert result['conditional_violation_gap'] is None
    assert result['n_violations'] == sum(value > 0 for value in declines)
    assert metrics.counts(frame, scores)['violations'] == sum(value >= 0 for value in declines)


def test_cv_pools_pair_totals_instead_of_equal_fold_weights():
    small_frame, small_scores = pair_frame([.6])
    large_frame, large_scores = pair_frame([.2, -.1, -.1])
    first, second = metrics.gaps(small_frame, small_scores), metrics.gaps(large_frame, large_scores)
    result = metrics.pooled_gaps([first, second])
    assert result['all_violation_gap'] == pytest.approx(.8 / 4)
    assert result['all_violation_gap'] != pytest.approx((first['all_violation_gap'] + second['all_violation_gap']) / 2)
    assert result['violation_ratio'] == .5
    assert result['conditional_violation_gap'] == pytest.approx(.4)


def test_training_rank_calibration_is_affine_invariant_and_heldout_only_applied():
    train = np.array([-2., -1., .1, .1, 2.])
    subjects = np.array(['a', 'a', 'b', 'c', 'c'])
    heldout = np.array([-4., -.3, .7, 5.])
    mapping = fit_rank_map(train, subjects)
    calibrated = apply_rank_map(heldout, mapping)
    assert calibrated[0] < 0 and calibrated[-1] > 1
    np.testing.assert_allclose(calibrated, apply_rank_map(1e-6 * heldout + .3,
                               fit_rank_map(1e-6 * train + .3, subjects)), atol=1e-9)
    extended = apply_rank_map(np.r_[heldout, -1e6, 1e6], mapping)
    np.testing.assert_array_equal(extended[:len(heldout)], calibrated)
    assert mapping == fit_rank_map(train, subjects)


def test_fold_mapping_uses_training_participants_only_and_is_hashed(tmp_path, monkeypatch):
    frame = pd.DataFrame({'RID': ['a', 'a', 'b', 'c', 'c'], 'PHC_Age_PET': [70, 71, 80, 60, 61]})
    valid = np.array([False, False, False, True, True])
    primary = np.array([0., 1., 2., 100., 90.])
    monkeypatch.setattr(runner.data, 'load_fold', lambda *args: (np.zeros((5, 2)), frame, valid))
    def fake_fit(x, training_frame, cfg, folder):
        assert training_frame.RID.tolist() == ['a', 'a', 'b']
        (folder / 'model.pt').write_bytes(b'test model')
        dump(folder / 'history.json', [])
        return object(), 0.
    monkeypatch.setattr(runner, 'fit_model', fake_fit)
    monkeypatch.setattr(runner, 'score_model', lambda *args: dict(stage_mean=primary.copy(), diagnostics={}))
    job = (str(tmp_path), 'trial0000', {'seed': 42}, 0)
    result = runner.fit_fold(job)
    folder = tmp_path / 'data/runs/trial0000/fold0'
    mapping = read(folder / 'rank_map.json')
    assert mapping == fit_rank_map(primary[~valid], frame.RID.to_numpy()[~valid])
    assert mapping != fit_rank_map(primary, frame.RID.to_numpy())
    with np.load(folder / 'scores.npz') as saved:
        np.testing.assert_array_equal(saved['rank_stage'], apply_rank_map(primary, mapping))
    assert result['validation_rank']['all_violation_gap'] > 1
    record = read(folder / 'complete.json')
    assert record['hashes'][str((folder / 'rank_map.json').relative_to(tmp_path))] == digest(folder / 'rank_map.json')
    assert runner.fit_fold(job) == result
    dump(folder / 'rank_map.json', dict(mapping, minimum=.123))
    with pytest.raises(RuntimeError, match='Frozen file changed'):
        runner.fit_fold(job)


def test_gap_objective_selects_different_trial_from_violation_ratio_and_resumes(tmp_path, monkeypatch):
    experiment = make_experiment(tmp_path, folds=2, parallel=2)
    experiment.spec.update(trials=2, anchors=[], selection_metric=PRODUCT)
    experiment.selection_metric = PRODUCT
    dump(experiment.config_path, {key: value for key, value in experiment.spec.items() if key != 'features'})
    monkeypatch.setattr(experiment, 'lock_protocol', lambda: None)
    monkeypatch.setattr(experiment, 'verify_protocol', lambda: None)
    monkeypatch.setattr(experiment, 'expected_pairs', lambda split: 4)
    dump(experiment.search_root / 'protocol.json', {})
    controlled_executor(monkeypatch, 2)
    calls = []
    def synthetic(job):
        trial = int(job[1].removeprefix('trial'))
        fold = job[-1]
        calls.append((trial, fold))
        declines = ([.8, -.1] if fold == 0 else [-.1, -.2]) if trial == 0 else [.05, -.1]
        frame, scores = pair_frame(declines)
        return dict(fold=fold, validation=metrics.counts(frame, scores), validation_rank=metrics.gaps(frame, scores))
    monkeypatch.setattr(runner, 'fit_fold', synthetic)
    experiment.search()
    trials = experiment.study().trials
    assert trials[0].user_attrs['violations'] < trials[1].user_attrs['violations']
    assert trials[0].value == pytest.approx(.2)
    assert trials[1].value == pytest.approx(.025)
    experiment.freeze()
    selected = read(experiment.search_root / 'selection.json')
    assert selected['selected_trial'] == 1
    assert selected['selection_metric'] == PRODUCT
    assert selected['tie_break'] == ['objective', 'epochs', 'trial']
    assert selected['cv']['objective'] == selected['cv'][PRODUCT]
    before = list(calls)
    experiment.search()
    assert calls == before


def test_small_synthetic_full_run_audits_calibration_and_detects_protocol_change(tmp_path, monkeypatch):
    frame, features = synthetic_visits()
    for rid in range(0, 52, 2):
        mask = frame.RID.eq(rid)
        frame.loc[mask, features] = frame.loc[mask, features].iloc[0].to_numpy()
    config = tmp_path / 'configs/training.json'
    spec = read(Path(__file__).resolve().parents[1] / 'configs/training.json')
    spec.update(folds=2, fold_workers=2, trials=1, anchors=[], startup_trials=1,
                study_name='test_calibrated_gap', selection_metric=PRODUCT)
    spec['fixed_model'].update(hidden_dims=[4], rank_samples=2)
    spec['search_space'] = {'beta': {'type': 'float', 'low': .01, 'high': .01},
                            'tau': {'type': 'float', 'low': .1, 'high': .1},
                            'rho': {'type': 'categorical', 'choices': [.5]},
                            'lr': {'type': 'float', 'low': .001, 'high': .001},
                            'epochs': {'type': 'int', 'low': 1, 'high': 1}}
    dump(config, spec); dump(config.parent / 'features.json', features)
    raw = tmp_path / 'raw.csv'
    frame.to_csv(raw, index=False)
    experiment = runner.Experiment(config, tmp_path / 'work', raw)
    controlled_executor(monkeypatch, 2)
    for action in ('prepare', 'search', 'freeze', 'refit', 'test', 'audit'):
        getattr(experiment, action)()
    report = read(experiment.root / 'reports/test_summary.json')
    assert report['selection_metric'] == PRODUCT
    audit = read(experiment.root / 'reports/audit.json')
    assert audit['passed'] and audit['training_only_rank_calibration_recomputed']
    assert audit['product_metrics_recomputed']
    assert report['cv']['ties'] > 0
    assert report['cv']['objective'] == report['cv']['violation_ratio'] * report['cv']['rank']['conditional_violation_gap']
    assert report['cv']['objective'] != pytest.approx(report['cv']['rank']['all_violation_gap'])
    product_function = metrics.violation_ratio_times_conditional_gap
    monkeypatch.setattr(metrics, PRODUCT, lambda raw, rank: rank['all_violation_gap'])
    with pytest.raises(AssertionError):
        experiment.audit()
    monkeypatch.setattr(metrics, PRODUCT, product_function)
    protocol = read(experiment.search_root / 'protocol.json')
    assert 'src/pace_abeta/analysis/calibration.py' in protocol['source_hashes']
    for split in ('train', 'test'):
        _, scans = runner.data.load_training(experiment.root) if split == 'train' else runner.data.load_test(experiment.root)
        with np.load(experiment.root / 'data' / ('refit' if split == 'train' else 'test') / 'scores.npz') as saved:
            a, b = adjacent_pairs(scans)
            independent = summarize_declines(saved['rank_stage'][a] - saved['rank_stage'][b])
        for key in ('all_violation_gap', 'conditional_violation_gap', 'total_violation_gap', 'violation_ratio'):
            assert report[f'{split}_rank'][key] == pytest.approx(independent[key])
        assert report[f'{split}_{PRODUCT}'] == report[split]['violation_ratio'] * independent['conditional_violation_gap']
    experiment.test()
    spec['selection_metric'] = 'violation_ratio'
    dump(config, spec)
    changed = runner.Experiment(config, experiment.root, raw)
    with pytest.raises(RuntimeError, match='configuration changed'):
        changed.verify_protocol()


def test_new_configuration_and_invalid_selection_metric(tmp_path):
    spec = read(Path(__file__).resolve().parents[1] / 'configs/training.json')
    assert (spec['folds'], spec['trials'], spec['selection_metric']) == (10, 128, PRODUCT)
    assert spec['search_space']['rho'] == dict(type='categorical', choices=[0., .1, .2, .3, .4, .5, .6, .7, .8, .9])
    assert [anchor['rho'] for anchor in spec['anchors']] == [.2, .4, .6, .8]
    assert spec['selected_model']['rho'] in spec['search_space']['rho']['choices']
    experiment = make_experiment(tmp_path)
    saved = read(experiment.config_path)
    saved['selection_metric'] = 'not_a_metric'
    dump(experiment.config_path, saved)
    with pytest.raises(ValueError, match='selection_metric'):
        runner.Experiment(experiment.config_path, experiment.root, experiment.raw_input)


def test_discrete_rho_sampler_uses_only_configured_categories(tmp_path):
    experiment = make_experiment(tmp_path)
    study = experiment.study()
    values = []
    for index in range(24):
        trial = study.ask()
        config = experiment.suggest(trial)
        value = config['rho']
        assert value in (0., .1, .2, .3, .4, .5, .6, .7, .8, .9)
        distribution = trial.distributions['rho']
        assert distribution.choices == (0., .1, .2, .3, .4, .5, .6, .7, .8, .9)
        values.append(value)
        study.tell(trial, (value - .57) ** 2 + index / 1e6)
        experiment.save_sampler(study)
    assert values[:4] == [.2, .4, .6, .8]
    assert any(value not in values[:4] for value in values[4:])


@pytest.mark.parametrize('declines,expected', [
    ([.4, .2, 0., -.1], .225),
    ([.4, -.1], .2),
    ([-.2, -.1], 0.),
    ([0., -.1], None),
    ([0., 0.], None),
])
def test_literal_product_preserves_legacy_ties_and_undefined_conditional_mean(declines, expected):
    frame, scores = pair_frame(declines)
    raw, rank = metrics.counts(frame, scores), metrics.gaps(frame, scores)
    product = metrics.violation_ratio_times_conditional_gap(raw, rank)
    assert product == pytest.approx(expected) if expected is not None else product is None
    if rank['conditional_violation_gap'] is not None:
        assert product == raw['violation_ratio'] * rank['conditional_violation_gap']
    if raw['ties'] and rank['n_violations']:
        assert product > rank['all_violation_gap']
        assert rank['total_violation_gap'] == sum(value for value in declines if value > 0)


def test_product_search_counts_ties_and_does_not_select_mean_positive_gap_winner(tmp_path, monkeypatch):
    experiment = make_experiment(tmp_path, folds=2, parallel=2)
    experiment.spec.update(trials=2, anchors=[], selection_metric=PRODUCT)
    experiment.selection_metric = PRODUCT
    monkeypatch.setattr(experiment, 'lock_protocol', lambda: None)
    monkeypatch.setattr(experiment, 'verify_protocol', lambda: None)
    monkeypatch.setattr(experiment, 'expected_pairs', lambda split: 4)
    dump(experiment.search_root / 'protocol.json', {})
    controlled_executor(monkeypatch, 2)
    def synthetic(job):
        trial = int(job[1].removeprefix('trial'))
        fold = job[-1]
        declines = ([.4, 0.] if fold == 0 else [0., -.1]) if trial == 0 else ([.5, -.1] if fold == 0 else [-.2, -.1])
        frame, scores = pair_frame(declines)
        return dict(fold=fold, validation=metrics.counts(frame, scores), validation_rank=metrics.gaps(frame, scores))
    monkeypatch.setattr(runner, 'fit_fold', synthetic)
    experiment.search()
    first, second = experiment.study().trials
    assert first.value == .75 * .4
    assert first.user_attrs['rank_metrics']['all_violation_gap'] == .1
    assert second.value == .25 * .5
    assert first.user_attrs['rank_metrics']['all_violation_gap'] < second.user_attrs['rank_metrics']['all_violation_gap']
    assert first.value > second.value
    experiment.freeze()
    selected = read(experiment.search_root / 'selection.json')
    assert selected['selected_trial'] == 1
    assert selected['cv']['objective'] == selected['cv']['violation_ratio'] * selected['cv']['rank']['conditional_violation_gap']


def test_undefined_product_is_nonrankable_but_zero_positive_single_fold_is_allowed(tmp_path, monkeypatch):
    experiment = make_experiment(tmp_path, folds=2, parallel=2)
    experiment.spec.update(trials=2, anchors=[], selection_metric=PRODUCT)
    experiment.selection_metric = PRODUCT
    monkeypatch.setattr(experiment, 'lock_protocol', lambda: None)
    monkeypatch.setattr(experiment, 'verify_protocol', lambda: None)
    monkeypatch.setattr(experiment, 'expected_pairs', lambda split: 4)
    dump(experiment.search_root / 'protocol.json', {})
    controlled_executor(monkeypatch, 2)
    def synthetic(job):
        trial = int(job[1].removeprefix('trial'))
        fold = job[-1]
        declines = [0., -.1] if trial == 0 or fold == 1 else [.5, -.1]
        frame, scores = pair_frame(declines)
        return dict(fold=fold, validation=metrics.counts(frame, scores), validation_rank=metrics.gaps(frame, scores))
    monkeypatch.setattr(runner, 'fit_fold', synthetic)
    experiment.search()
    first, second = experiment.study().trials
    assert first.state == optuna.trial.TrialState.FAIL and first.value is None
    assert 'objective_undefined' in first.user_attrs
    assert 'numerical_failure' not in first.user_attrs
    failed = read(experiment.search_root / 'trial_results/trial0000.json')
    assert failed['objective_status'] == 'undefined'
    assert failed['objective'] is None and failed[PRODUCT] is None
    assert len(failed['folds']) == 2
    assert second.state == optuna.trial.TrialState.COMPLETE
    assert second.value == .5 * .5
    experiment.freeze()
    selected = read(experiment.search_root / 'selection.json')
    assert selected['selected_trial'] == 1 and selected['total_attempts'] == 2
    assert selected['failed_trials'] == selected['undefined_objective_trials'] == 1
    assert selected['successful_trials'] == 1
