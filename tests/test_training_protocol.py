"""Search scheduling, checkpoint scoring, and deterministic resume tests."""
from concurrent.futures import Future
import json
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np
import optuna
import pandas as pd
import pytest
import torch

from pl_staging import runner
from pl_staging.io import digest, dump, read


SPEC = Path(__file__).resolve().parents[1] / 'configs/training.json'


def make_experiment(root, *, folds=None, parallel=None):
    config = root / 'configs/training.json'
    spec = read(SPEC)
    spec.update(folds=12, trials=64, selection_metric='violation_ratio')
    dump(config, spec)
    dump(config.parent / 'features.json', read(SPEC.parent / 'features.json'))
    return runner.Experiment(config, root, root / 'raw.csv', folds=folds, parallel=parallel)


def isolate_protocol(monkeypatch, experiment):
    def lock():
        path = experiment.search_root / 'protocol.json'
        if not path.exists():
            dump(path, dict(hashes={}))
    monkeypatch.setattr(experiment, 'lock_protocol', lock)
    monkeypatch.setattr(experiment, 'verify_protocol', lambda: None)
    monkeypatch.setattr(experiment, 'expected_pairs', lambda split: 874)
    lock()
    def forbidden(*args, **kwargs):
        pytest.fail('A protocol unit test attempted to load actual holdout features')
    monkeypatch.setattr(runner.data, 'load_test', forbidden)


def controlled_executor(monkeypatch, workers):
    """Leave jobs pending until collection, exposing premature batch submission."""
    pools = []
    class ControlledPool:
        def __init__(self, **kwargs):
            assert kwargs['max_workers'] == workers
            assert kwargs['mp_context'].get_start_method() == 'spawn'
            self.pending, self.jobs, self.trace = [], {}, []
            self.collecting = False
            self.consumed = 0
            pools.append(self)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def submit(self, fn, job):
            assert not self.collecting, 'Next batch submitted before collection finished'
            assert len(self.pending) < workers, 'More than one batch queued'
            future = Future()
            self.pending.append(future)
            self.jobs[future] = (fn, job)
            self.trace.append(('submitted', job[1], job[-1]))
            return future
    def collect(futures):
        pool = pools[-1]
        assert list(futures) == pool.pending
        pool.collecting = True
        # Deliberately finish out of fold order to exercise deterministic aggregation.
        for future in reversed(futures):
            fn, job = pool.jobs[future]
            try:
                future.set_result(fn(job))
            except BaseException as error:
                future.set_exception(error)
            pool.trace.append(('finished', job[1], job[-1]))
            yield future
            pool.consumed += 1
        pool.pending = []
        pool.collecting = False
    monkeypatch.setattr(runner, 'ProcessPoolExecutor', ControlledPool)
    monkeypatch.setattr(runner, 'as_completed', collect)
    return pools


@pytest.mark.parametrize('folds', [10, 12, 20])
def test_search_consumes_four_waiting_anchors_within_64_attempts_and_freezes_cv_only(
        tmp_path, monkeypatch, capsys, folds):
    monkeypatch.setattr(runner, 'available_cpu_count', lambda: 4)
    experiment = make_experiment(tmp_path / 'search', folds=folds)
    isolate_protocol(monkeypatch, experiment)
    calls = []
    def synthetic_fold(job):
        root, name, config, fold = job
        number = int(name.removeprefix('trial'))
        calls.append((number, fold))
        if number == 17 and fold == 0:
            raise FloatingPointError('Injected synthetic numerical failure')
        pairs = 874 // folds + int(fold < 874 % folds)
        return dict(fold=fold, validation=dict(pairs=pairs, violations=1, ties=0))
    pools = controlled_executor(monkeypatch, 4)
    monkeypatch.setattr(runner, 'fit_fold', synthetic_fold)
    initial = experiment.study()
    assert len(initial.trials) == 4
    assert all(t.state == optuna.trial.TrialState.WAITING for t in initial.trials)
    experiment.search()
    study = experiment.study()
    assert len(study.trials) == 64
    assert all(t.state.is_finished() for t in study.trials)
    assert study.trials[17].state == optuna.trial.TrialState.FAIL
    assert sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials) == 63
    assert len(calls) == 64 * folds
    assert len(set(calls)) == 64 * folds
    assert [number for number, _ in calls] == [number for number in range(64) for _ in range(folds)]
    assert pools[0].consumed == 64 * folds
    assert set(fold for _, fold in calls) == set(range(folds))
    for trial, anchor in zip(study.trials, experiment.spec['anchors']):
        assert trial.params == anchor
    experiment.freeze()
    selection = read(experiment.search_root / 'selection.json')
    expected = min((t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE),
                   key=lambda t: (t.user_attrs['violations'], t.params['epochs'], t.number))
    assert selection['selected_trial'] == expected.number
    assert selection['total_attempts'] == 64
    assert selection['successful_trials'] == 63
    assert selection['failed_trials'] == 1
    assert selection['cv']['pairs'] == 874
    experiment.search()
    assert len(calls) == 64 * folds
    capsys.readouterr()


def test_defaults_and_overrides_leave_configuration_bytes_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'available_cpu_count', lambda: 24)
    source_before = SPEC.read_bytes()
    default = make_experiment(tmp_path)
    local_before = default.config_path.read_bytes()
    assert (default.spec['folds'], default.spec['fold_workers'], default.spec['trials']) == (12, 12, 64)
    assert default.fold_workers == 12
    changed = runner.Experiment(default.config_path, default.root, default.raw_input,
                                folds=20, parallel=10)
    expected = dict(default.spec, folds=20, fold_workers=10)
    assert changed.spec == expected and changed.fold_workers == 10
    assert default.config_path.read_bytes() == local_before
    assert SPEC.read_bytes() == source_before


@pytest.mark.parametrize('folds,parallel,cpus,expected', [
    (10, 12, 24, 10), (12, 12, 4, 4), (20, 10, 24, 10),
    (20, 10, 3, 3), (20, 3, 24, 3), (20, 10, 1, 1), (2, 1, 1, 1),
])
def test_workers_respect_request_folds_and_cpu_limit(tmp_path, monkeypatch, folds, parallel, cpus, expected):
    monkeypatch.setattr(runner, 'available_cpu_count', lambda: cpus)
    experiment = make_experiment(tmp_path, folds=folds, parallel=parallel)
    assert experiment.fold_workers == expected
    assert experiment.spec['fold_workers'] == parallel


def test_cpu_count_uses_affinity_then_process_then_machine(monkeypatch):
    monkeypatch.setattr(runner.os, 'sched_getaffinity', lambda pid: {2, 3}, raising=False)
    monkeypatch.setattr(runner.os, 'process_cpu_count', lambda: 5, raising=False)
    monkeypatch.setattr(runner.os, 'cpu_count', lambda: 12)
    assert runner.available_cpu_count() == 2
    def inaccessible(pid):
        raise OSError('Affinity unavailable')
    monkeypatch.setattr(runner.os, 'sched_getaffinity', inaccessible)
    assert runner.available_cpu_count() == 5
    monkeypatch.setattr(runner.os, 'sched_getaffinity', lambda pid: set())
    assert runner.available_cpu_count() == 5
    monkeypatch.delattr(runner.os, 'sched_getaffinity')
    monkeypatch.setattr(runner.os, 'process_cpu_count', lambda: None)
    assert runner.available_cpu_count() == 12
    monkeypatch.delattr(runner.os, 'process_cpu_count')
    monkeypatch.setattr(runner.os, 'cpu_count', lambda: None)
    assert runner.available_cpu_count() == 1


@pytest.mark.parametrize('option,value', [
    ('folds', 1), ('folds', 0), ('folds', 2.5), ('folds', True), ('folds', '20'),
    ('parallel', 0), ('parallel', -1), ('parallel', 1.5), ('parallel', False), ('parallel', '10'),
])
@pytest.mark.parametrize('location', ['override', 'config'])
def test_invalid_fold_or_worker_values_are_rejected(tmp_path, option, value, location):
    experiment = make_experiment(tmp_path)
    kwargs = {option: value}
    if location == 'config':
        spec = read(experiment.config_path)
        spec['fold_workers' if option == 'parallel' else option] = value
        dump(experiment.config_path, spec)
        kwargs = {}
    with pytest.raises(ValueError, match='must be an integer'):
        runner.Experiment(experiment.config_path, experiment.root, experiment.raw_input, **kwargs)


@pytest.mark.parametrize('changed', [{'folds': 12, 'parallel': 3}, {'folds': 20, 'parallel': 4}])
def test_overrides_are_locked_and_changed_settings_cannot_resume(tmp_path, monkeypatch, changed):
    original = make_experiment(tmp_path)
    experiment = runner.Experiment(original.config_path, original.root, original.raw_input,
                                   folds=20, parallel=3)
    monkeypatch.setattr(runner.data, 'prepare', lambda *args: {})
    dump(experiment.root / 'data/prepared/metadata.json', {})
    experiment.lock_protocol()
    protocol = read(experiment.search_root / 'protocol.json')
    assert protocol['spec']['folds'] == 20 and protocol['spec']['fold_workers'] == 3
    experiment.verify_protocol()
    different = runner.Experiment(original.config_path, original.root, original.raw_input, **changed)
    with pytest.raises(RuntimeError, match='configuration changed'):
        different.verify_protocol()


def one_trial(experiment):
    experiment.spec.update(trials=1, anchors=[])
    dump(experiment.config_path, {k: v for k, v in experiment.spec.items() if k != 'features'})


def synthetic_metrics(fold, folds=20):
    return dict(fold=fold, validation=dict(pairs=874 // folds + int(fold < 874 % folds),
                                         violations=1, ties=0))


@pytest.mark.parametrize('cpus,failed_fold', [(10, None), (10, 0), (10, 13), (3, None), (1, None)])
def test_strict_batches_finish_all_folds_and_report_progress(tmp_path, monkeypatch, cpus, failed_fold):
    monkeypatch.setattr(runner, 'available_cpu_count', lambda: cpus)
    experiment = make_experiment(tmp_path, folds=20, parallel=10)
    one_trial(experiment)
    isolate_protocol(monkeypatch, experiment)
    pools = controlled_executor(monkeypatch, min(10, cpus))
    calls, events = [], []
    def fit(job):
        fold = job[-1]
        calls.append(fold)
        if fold == failed_fold:
            raise FloatingPointError(f'Injected failure in fold {fold}')
        return synthetic_metrics(fold)
    monkeypatch.setattr(runner, 'fit_fold', fit)
    monkeypatch.setattr(runner, 'event', lambda **values: events.append(values))
    experiment.search()
    assert sorted(calls) == list(range(20)) and pools[0].consumed == 20
    batches = [list(range(start, min(start + experiment.fold_workers, 20)))
               for start in range(0, 20, experiment.fold_workers)]
    batch_events = [e for e in events if e['event'].startswith('fold_batch_')]
    assert len(batch_events) == 2 * len(batches)
    for index, members in enumerate(batches):
        started, completed = batch_events[index * 2:index * 2 + 2]
        assert started['event'] == 'fold_batch_started' and started['batch_status'] == 'running'
        assert completed['event'] == 'fold_batch_complete'
        assert started['batch_folds'] == completed['batch_folds'] == members
        assert started['trial'] == completed['trial'] == 0
        assert started['batch'] == completed['batch'] == index + 1
        assert completed['total_batches'] == len(batches)
        assert completed['batch_folds_finished'] == len(members)
        assert completed['batch_numerical_failures'] == int(failed_fold in members)
        assert completed['batch_status'] == ('numerical_failure' if failed_fold in members else 'complete')
        assert started['folds_finished'] == members[0]
        assert completed['folds_finished'] == members[-1] + 1
        if index:
            first_submit = pools[0].trace.index(('submitted', 'trial0000', members[0]))
            assert all(pools[0].trace.index(('finished', 'trial0000', fold)) < first_submit
                       for fold in batches[index - 1])
    status = read(experiment.search_root / 'status.json')
    assert status['folds_finished'] == 20
    assert status['numerical_failures'] == int(failed_fold is not None)
    trials = experiment.study().trials
    assert len(trials) == 1
    result = read(experiment.search_root / 'trial_results/trial0000.json')
    if failed_fold is None:
        assert trials[0].state == optuna.trial.TrialState.COMPLETE
        assert [r['fold'] for r in result['folds']] == list(range(20))
        assert result['pairs'] == 874 and result['violations'] == 20
    else:
        assert trials[0].state == optuna.trial.TrialState.FAIL
        assert [r['fold'] for r in result['completed_folds']] == [f for f in range(20) if f != failed_fold]


def test_interrupted_batch_resumes_same_trial_and_cached_folds(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'available_cpu_count', lambda: 10)
    experiment = make_experiment(tmp_path, folds=20, parallel=10)
    one_trial(experiment)
    isolate_protocol(monkeypatch, experiment)
    pools = controlled_executor(monkeypatch, 10)
    original_fit_fold = runner.fit_fold
    trained, cached = [], []
    interrupted = False
    def fit(job):
        nonlocal interrupted
        root, name, config, fold = job
        folder = Path(root) / 'data/runs' / name / f'fold{fold}'
        if (folder / 'complete.json').exists():
            cached.append(fold)
            return original_fit_fold(job)
        if fold == 3 and not interrupted:
            interrupted = True
            raise RuntimeError('Injected interruption')
        trained.append(fold)
        result = synthetic_metrics(fold)
        artifact = folder / 'synthetic.json'
        dump(artifact, result)
        dump(folder / 'complete.json', dict(config=config, metrics=result,
             hashes={str(artifact.relative_to(root)): digest(artifact)}))
        return result
    monkeypatch.setattr(runner, 'fit_fold', fit)
    with pytest.raises(RuntimeError, match='Injected interruption'):
        experiment.search()
    first = experiment.study().trials
    assert len(first) == 1 and first[0].state == optuna.trial.TrialState.RUNNING
    completed_before_resume = set(trained)
    assert completed_before_resume == set(range(4, 10))
    assert [fold for action, _, fold in pools[0].trace if action == 'submitted'] == list(range(10))
    resumed = runner.Experiment(experiment.config_path, experiment.root, experiment.raw_input)
    isolate_protocol(monkeypatch, resumed)
    resumed.search()
    trials = resumed.study().trials
    assert len(trials) == 1 and trials[0].state == optuna.trial.TrialState.COMPLETE
    assert trials[0].params == first[0].params
    assert len(pools) == 2 and pools[1].consumed == 20
    assert sorted(trained) == list(range(20))
    assert set(cached) == completed_before_resume
    assert read(resumed.search_root / 'trial_results/trial0000.json')['pairs'] == 874


def test_smoke_uses_strict_batches_with_a_short_final_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'available_cpu_count', lambda: 3)
    experiment = make_experiment(tmp_path, folds=20, parallel=10)
    monkeypatch.setattr(runner.data, 'prepare', lambda *args: {})
    monkeypatch.setattr(experiment, 'expected_pairs', lambda split: 874)
    pools = controlled_executor(monkeypatch, 3)
    monkeypatch.setattr(runner, 'fit_fold', lambda job: synthetic_metrics(job[-1]))
    experiment.smoke()
    assert pools[0].consumed == 20


@pytest.mark.parametrize('overrides', [[], ['--folds', '20', '--parallel', '10']])
def test_train_cli_forwards_optional_overrides(tmp_path, monkeypatch, overrides):
    calls = []
    class StubExperiment:
        def __init__(self, config, work_dir, raw_input, **kwargs):
            calls.append((config, work_dir, raw_input, kwargs))
        def prepare(self):
            calls.append('prepare')
    monkeypatch.setattr(runner, 'Experiment', StubExperiment)
    # The entry point deliberately sets thread limits; let monkeypatch restore them.
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'PYTHONDONTWRITEBYTECODE'):
        monkeypatch.setenv(name, runner.os.environ.get(name, ''))
    monkeypatch.setattr(sys, 'dont_write_bytecode', True)
    monkeypatch.setattr(sys, 'path', list(sys.path))
    script = SPEC.parents[1] / 'scripts/train.py'
    monkeypatch.setattr(sys, 'argv', [str(script), '--raw-input', str(tmp_path / 'raw.csv'),
                        '--work-dir', str(tmp_path), '--stage', 'prepare', *overrides])
    runpy.run_path(str(script), run_name='__main__')
    assert calls[0][:3] == (SPEC, tmp_path, tmp_path / 'raw.csv')
    assert calls[0][3].get('folds') == (20 if overrides else None)
    assert calls[0][3].get('parallel') == (10 if overrides else None)
    assert calls[1:] == ['prepare']


@pytest.mark.parametrize('overrides', [[], ['--folds', '20', '--parallel', '10']])
def test_pipeline_forwards_fold_options_only_to_training(tmp_path, monkeypatch, overrides):
    calls = []
    def capture(command, **kwargs):
        calls.append((command, kwargs))
    monkeypatch.setattr(subprocess, 'run', capture)
    script = SPEC.parents[1] / 'scripts/run_pipeline.py'
    monkeypatch.setattr(sys, 'argv', [str(script), '--work-dir', str(tmp_path), *overrides])
    runpy.run_path(str(script), run_name='__main__')
    assert len(calls) == 4
    assert [Path(command[2]).name for command, _ in calls] == ['train.py', 'compare.py', 'analyze.py', 'figures.py']
    training = calls[0][0]
    for option, expected in [('--folds', '20'), ('--parallel', '10')]:
        if overrides:
            assert training[training.index(option) + 1] == expected
        else:
            assert option not in training
        assert all(option not in command for command, _ in calls[1:])
    assert all(kwargs['check'] is True for _, kwargs in calls)


def test_seed42_sampler_and_inflight_resume_reproduce_parameter_sequence(tmp_path):
    def sequence(root, resume_at=None):
        experiment = make_experiment(root)
        study = experiment.study()
        configurations = []
        for index in range(22):
            trial = study.ask()
            config = experiment.suggest(trial)
            experiment.save_sampler(study)
            if index == resume_at:
                experiment = runner.Experiment(experiment.config_path, experiment.root, experiment.raw_input)
                study = experiment.study()
                running = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]
                assert len(running) == 1
                trial = optuna.trial.Trial(study, running[0]._trial_id)
                assert experiment.suggest(trial) == config
            configurations.append(config)
            value = float((index % 7) + np.log(config['beta']) / 100)
            study.tell(trial, value)
            experiment.save_sampler(study)
        return configurations
    uninterrupted = sequence(tmp_path / 'uninterrupted')
    resumed = sequence(tmp_path / 'resumed', resume_at=18)
    assert uninterrupted == resumed


@pytest.mark.parametrize('missing', ['selection', 'refit'])
def test_holdout_load_is_blocked_until_selection_and_refit_exist(tmp_path, monkeypatch, missing):
    experiment = make_experiment(tmp_path / missing)
    isolate_protocol(monkeypatch, experiment)
    if missing == 'refit':
        dump(experiment.search_root / 'selection.json', dict(hashes={}))
    with pytest.raises(FileNotFoundError):
        experiment.test()


def create_frozen_stub(experiment):
    selection = experiment.search_root / 'selection.json'
    dump(selection, dict(hashes={}, selected_trial=3))
    checkpoint = experiment.root / 'data/refit/model.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'synthetic checkpoint; not a torch model')
    dump(experiment.root / 'data/refit/complete.json',
         dict(hashes={str(selection.relative_to(experiment.root)): digest(selection),
                      str(checkpoint.relative_to(experiment.root)): digest(checkpoint)}))
    return selection, checkpoint


def test_changed_frozen_selection_blocks_test_before_loading_features(tmp_path, monkeypatch):
    experiment = make_experiment(tmp_path / 'changed')
    isolate_protocol(monkeypatch, experiment)
    selection, _ = create_frozen_stub(experiment)
    dump(selection, dict(hashes={}, selected_trial=4))
    with pytest.raises(RuntimeError, match='Frozen file changed'):
        experiment.test()


def test_preexisting_holdout_access_rejects_a_different_checkpoint(tmp_path, monkeypatch):
    experiment = make_experiment(tmp_path / 'access')
    isolate_protocol(monkeypatch, experiment)
    selection, _ = create_frozen_stub(experiment)
    dump(experiment.root / 'data/test/access.json',
         dict(selection_sha256=digest(selection), checkpoint_sha256='0' * 64))
    with pytest.raises(RuntimeError, match='different model'):
        experiment.test()


def test_completed_holdout_is_returned_without_reloading_features(tmp_path, monkeypatch):
    experiment = make_experiment(tmp_path / 'completed')
    isolate_protocol(monkeypatch, experiment)
    create_frozen_stub(experiment)
    dump(experiment.root / 'data/test/complete.json', dict(hashes={}, test={'pairs': 1}))
    experiment.test()


def test_optimizer_uses_common_beta_from_first_epoch_and_no_weight_decay(tmp_path, monkeypatch):
    gradients = []
    captured = {}
    class ToyModel(torch.nn.Module):
        def __init__(self, dimension, config):
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor(2.0))
    class RecordingOptimizer:
        def __init__(self, parameters, lr, weight_decay):
            self.parameters = list(parameters)
            captured.update(lr=lr, weight_decay=weight_decay)
        def zero_grad(self, set_to_none):
            assert set_to_none
            for parameter in self.parameters:
                parameter.grad = None
        def step(self):
            gradients.append(float(self.parameters[0].grad))
    def terms(model, x, padded, mask, normalizer):
        value = model.w.square()
        return {'rec': value, 'rank': 3 * value, 'kl': 5 * value}
    monkeypatch.setattr(runner, 'ProbabilisticAE', ToyModel)
    monkeypatch.setattr(runner, 'loss_terms', terms)
    monkeypatch.setattr(runner.torch.optim, 'AdamW', RecordingOptimizer)
    folder = tmp_path / 'fit'
    folder.mkdir()
    frame = pd.DataFrame({'RID': [1, 1, 2, 3], 'PHC_Age_PET': [70, 71, 80, 90]})
    config = dict(seed=42, lr=.001, beta=.25, epochs=1, batch_subjects=2)
    runner.fit_model(np.zeros((4, 3)), frame, config, folder)
    assert captured == {'lr': .001, 'weight_decay': 0.}
    # d/dw [w² + .25*(3w²+5w²)] at w=2 is 12, in every batch from epoch 1.
    assert gradients == [12., 12.]
    history = read(folder / 'history.json')
    assert history[0]['loss'] == pytest.approx(12.)
    assert history[0]['rec'] == pytest.approx(4.)
    assert history[0]['rank'] == pytest.approx(12.)
    assert history[0]['kl'] == pytest.approx(20.)
