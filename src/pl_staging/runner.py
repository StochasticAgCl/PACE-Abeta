"""Independent, resumable cross-validation with a locked external holdout."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import datetime
import json
import multiprocessing
import os
import pickle
import random
import time

import numpy as np
import optuna
import pandas as pd
import torch

from pace_abeta.analysis.calibration import apply_rank_map, fit_rank_map

from . import data, metrics
from .io import read, dump, digest, hashes, verify
from .model import ProbabilisticAE, loss_terms, score_model


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def event(**values):
    print(json.dumps(dict(time=utc(), **values), allow_nan=False), flush=True)


def available_cpu_count():
    """Count logical CPUs available to this process, respecting CPU affinity."""
    if hasattr(os, 'sched_getaffinity'):
        try:
            count = len(os.sched_getaffinity(0))
            if count:
                return count
        except OSError:
            pass
    process_count = getattr(os, 'process_cpu_count', lambda: None)()
    return max(1, process_count or os.cpu_count() or 1)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def fit_model(x, frame, cfg, folder):
    """Fit specified ELBO from epoch one; no validation-dependent updates."""
    seed_everything(cfg['seed'])
    model = ProbabilisticAE(x.shape[1], cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=0.)
    inputs = torch.as_tensor(x, dtype=torch.float32)
    batcher = data.SubjectBatcher(frame.RID.to_numpy(), frame.PHC_Age_PET.to_numpy(), cfg['batch_subjects'])
    history = []
    started = time.monotonic()
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        totals = np.zeros(4, dtype=np.float64)
        people = 0
        for indices, padded, mask, normalizer in batcher.batches():
            xb = inputs[indices]
            optimizer.zero_grad(set_to_none=True)
            terms = loss_terms(model, xb, padded, mask, normalizer)
            loss = terms['rec'] + cfg['beta'] * (terms['rank'] + terms['kl'])
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at epoch {epoch}')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError(f'Nonfinite gradient at epoch {epoch}')
            optimizer.step()
            b = len(padded)
            totals += b * np.asarray([float(loss.detach()), float(terms['rec'].detach()),
                                      float(terms['rank'].detach()), float(terms['kl'].detach())])
            people += b
        values = totals / people
        if not np.isfinite(values).all():
            raise FloatingPointError(f'Nonfinite training summary at epoch {epoch}')
        if epoch == 1 or epoch % 50 == 0 or epoch == cfg['epochs']:
            row = dict(epoch=epoch, loss=float(values[0]), rec=float(values[1]),
                       rank=float(values[2]), kl=float(values[3]),
                       seconds=time.monotonic() - started)
            history.append(row)
            dump(folder / 'progress.json', row)
    model.eval()
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError('Nonfinite trained parameters')
    dump(folder / 'history.json', history)
    torch.save(dict(model_state_dict=model.state_dict(), config=cfg, input_dim=x.shape[1]), folder / 'model.pt')
    return model, time.monotonic() - started


def save_scores(folder, scores):
    arrays = {k: v for k, v in scores.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(folder / 'scores.npz', **arrays)
    dump(folder / 'score_diagnostics.json', scores['diagnostics'])


def verify_saved_q_from_checkpoint(folder, x, checkpoint_path=None):
    """Independently verify every stored q parameter without the scoring API."""
    folder = Path(folder)
    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else folder / 'model.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model = ProbabilisticAE(checkpoint['input_dim'], checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.double().eval()
    with torch.no_grad():
        mu, logvar = model.posterior(torch.from_numpy(np.asarray(x, dtype=np.float64)))
        expected = {'stage_mean': mu.numpy(), 'stage_logvar': logvar.numpy(),
                    'stage_variance': logvar.exp().numpy()}
    with np.load(folder / 'scores.npz', allow_pickle=False) as saved:
        for name, value in expected.items():
            actual = saved[name]
            assert np.isfinite(value).all(), f'Nonfinite checkpoint q parameter: {folder}/{name}'
            # Include dtype and raw bytes, so even signed-zero differences fail.
            assert (actual.shape == value.shape and actual.dtype == value.dtype
                    and actual.tobytes() == value.tobytes()), f'Saved q parameter differs from checkpoint: {folder}/{name}'


def fit_fold(job):
    root, run_name, cfg, fold = job
    root = Path(root)
    folder = root / 'data/runs' / run_name / f'fold{fold}'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'complete.json').exists():
        saved = read(folder / 'complete.json')
        if saved['config'] != cfg:
            raise RuntimeError('Attempted to resume a fold with a different configuration')
        verify(saved, root)
        return saved['metrics']
    if (folder / 'numerical_failure.json').exists():
        failed = read(folder / 'numerical_failure.json')
        if failed['config'] != cfg:
            raise RuntimeError('Numerical-failure configuration changed')
        raise FloatingPointError(failed['error'])
    started = time.monotonic()
    try:
        x, frame, valid = data.load_fold(root, fold)
        train = ~valid
        model, fit_seconds = fit_model(x[train], frame.loc[train].reset_index(drop=True), cfg, folder)
        scores = score_model(model, x)
        primary = scores['stage_mean']
        if not np.isfinite(primary).all() or np.ptp(primary[train]) <= 0:
            raise FloatingPointError('Nonfinite or constant training q-mean stage')
        mapping = fit_rank_map(primary[train], frame.loc[train, 'RID'].to_numpy())
        scores['rank_stage'] = apply_rank_map(primary, mapping)
        dump(folder / 'rank_map.json', mapping)
        result = dict(fold=fold, validation_rank=metrics.gaps(frame.loc[valid], scores['rank_stage'][valid]),
                      train_rank=metrics.gaps(frame.loc[train], scores['rank_stage'][train]),
                      validation=metrics.counts(frame.loc[valid], primary[valid]),
                      train=metrics.counts(frame.loc[train], primary[train]),
                      fit_seconds=fit_seconds, seconds=time.monotonic() - started,
                      score_diagnostics=scores['diagnostics'])
        save_scores(folder, scores)
        record = dict(config=cfg, metrics=result, completed_at=utc(),
                      hashes=hashes([folder / 'model.pt', folder / 'scores.npz',
                                     folder / 'score_diagnostics.json', folder / 'history.json', folder / 'rank_map.json'], root))
        dump(folder / 'complete.json', record)
        return result
    except FloatingPointError as error:
        dump(folder / 'numerical_failure.json', dict(config=cfg, error=str(error), time=utc()))
        raise


class Experiment:
    def __init__(self, config_path, work_dir, raw_input, *, folds=None, parallel=None):
        self.config_path = Path(config_path).resolve()
        self.root = Path(work_dir).resolve()
        self.raw_input = Path(raw_input).resolve()
        self.source_root = Path(__file__).resolve().parents[2]
        self.spec = read(self.config_path)
        if folds is not None:
            self.spec['folds'] = folds
        if parallel is not None:
            self.spec['fold_workers'] = parallel
        self.spec['features'] = read(self.config_path.parent / self.spec['features_file'])
        self.search_root = self.root / 'data/search'
        self.search_root.mkdir(parents=True, exist_ok=True)
        if not self.spec['seed'] == self.spec['split_seed'] == self.spec['fold_seed'] == 42:
            raise ValueError('This experiment is locked to seed42')
        for name, minimum in (('folds', 2), ('fold_workers', 1)):
            value = self.spec[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f'{name} must be an integer of at least {minimum}')
        self.selection_metric = self.spec.get('selection_metric', 'violation_ratio')
        if self.selection_metric not in ('violation_ratio', 'violation_ratio_times_conditional_gap'):
            raise ValueError('selection_metric must be violation_ratio or violation_ratio_times_conditional_gap')
        self.available_cpus = available_cpu_count()
        self.fold_workers = min(self.spec['fold_workers'], self.spec['folds'], self.available_cpus)
        if optuna.__version__ != self.spec['optuna_version']:
            raise RuntimeError('Optuna version differs from the frozen protocol')
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def source_paths(self):
        return [self.source_root / 'scripts/train.py',
                *sorted((self.source_root / 'src/pl_staging').glob('*.py')),
                *sorted((self.source_root / 'src/pace_abeta/analysis').glob('*.py'))]

    def aggregate_folds(self, records):
        """Compute one pooled CV objective from independently held-out folds."""
        pairs = sum(r['validation']['pairs'] for r in records)
        violations = sum(r['validation']['violations'] for r in records)
        if pairs != self.expected_pairs('cv'):
            raise ValueError('CV pair count differs from the locked protocol')
        result = dict(pairs=pairs, violations=violations,
                      ties=sum(r['validation']['ties'] for r in records),
                      violation_ratio=violations / pairs)
        if all('validation_rank' in record for record in records):
            result['rank'] = metrics.pooled_gaps(r['validation_rank'] for r in records)
            if result['rank']['n_pairs'] != pairs:
                raise ValueError('Raw and calibrated pair denominators differ')
            result['violation_ratio_times_conditional_gap'] = metrics.violation_ratio_times_conditional_gap(
                result, result['rank'])
        if self.selection_metric == 'violation_ratio_times_conditional_gap':
            value = result['violation_ratio_times_conditional_gap']
        else:
            value = result['violation_ratio']
        if value is not None and not np.isfinite(value):
            raise ValueError('A defined CV objective must be finite')
        result.update(selection_metric=self.selection_metric, objective=value,
                      objective_status='defined' if value is not None else 'undefined')
        if value is None:
            result['objective_reason'] = (
                'The pooled nonincrease violation ratio is positive, but there are no '
                'positive rank-score declines; the conditional gap is undefined.')
        return result

    def prepare(self):
        result = data.prepare(self.root, self.raw_input, self.spec)
        event(event='data_prepared', result={k: result[k] for k in
              ('train_subjects', 'train_scans', 'test_subjects', 'test_scans', 'pooled_cv_pairs')})
        return result

    def verify_protocol(self):
        record = read(self.search_root / 'protocol.json')
        verify(record, self.root)
        if record['spec'] != self.spec:
            raise RuntimeError('Training configuration changed after protocol lock')
        for relative, expected in record['source_hashes'].items():
            if digest(self.source_root / relative) != expected:
                raise RuntimeError(f'Training source changed after protocol lock: {relative}')
        data.prepare(self.root, self.raw_input, self.spec)

    def lock_protocol(self):
        path = self.search_root / 'protocol.json'
        if path.exists():
            self.verify_protocol()
            return
        metadata = data.prepare(self.root, self.raw_input, self.spec)
        dump(path, dict(spec=self.spec, locked_at=utc(),
            hashes=hashes([self.root / 'data/prepared/metadata.json'], self.root),
            source_hashes=hashes(self.source_paths(), self.source_root),
            torch_version=torch.__version__, numpy_version=np.__version__,
            optuna_version=optuna.__version__, enqueued_candidates=len(self.spec.get('anchors', []))))

    def expected_pairs(self, split):
        metadata = read(self.root / 'data/prepared/metadata.json')
        return metadata['pooled_cv_pairs' if split == 'cv' else 'test_pairs']

    def select_fixed(self):
        """Freeze the published configuration without running a new search."""
        self.lock_protocol()
        selection_path = self.search_root / 'selection.json'
        cfg = dict(self.spec['selected_model'])
        if selection_path.exists():
            saved = read(selection_path)
            verify(saved, self.root)
            if saved.get('mode') != 'fixed' or saved['config'] != cfg:
                raise RuntimeError('A different model selection already exists')
            return
        if (self.search_root / 'study.sqlite3').exists():
            raise RuntimeError('Use a separate work directory for fixed-configuration refitting')
        dump(selection_path, dict(mode='fixed', selected_trial=None, config=cfg,
            cv=None, total_attempts=0, successful_trials=0, failed_trials=0,
            frozen_at=utc(), hashes=hashes([self.search_root / 'protocol.json'], self.root)))
        event(event='fixed_configuration_selected', config=cfg)

    def suggest(self, trial):
        cfg = dict(seed=42, **self.spec['fixed_model'])
        for name, domain in self.spec['search_space'].items():
            kwargs = {k: v for k, v in domain.items() if k != 'type'}
            cfg[name] = getattr(trial, 'suggest_' + domain['type'])(name, **kwargs)
        return cfg

    def study(self):
        sampler_path = self.search_root / 'sampler.pkl'
        if sampler_path.exists():
            # Only our own locally written sampler is loaded.
            with sampler_path.open('rb') as stream:
                sampler = pickle.load(stream)
        else:
            sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=self.spec['startup_trials'],
                                                constant_liar=False)
        study = optuna.create_study(study_name=self.spec['study_name'],
                                  storage=f'sqlite:///{self.search_root / "study.sqlite3"}',
                                  sampler=sampler, pruner=optuna.pruners.NopPruner(),
                                  direction='minimize', load_if_exists=True)
        if not study.trials:
            for params in self.spec.get('anchors', []):
                study.enqueue_trial(params)
        return study

    def save_sampler(self, study):
        temporary = self.search_root / 'sampler.pkl.tmp'
        with temporary.open('wb') as stream:
            pickle.dump(study.sampler, stream)
        temporary.replace(self.search_root / 'sampler.pkl')

    def run_fold_batches(self, pool, run_name, cfg, status=None):
        """Finish each contiguous worker-sized batch before submitting the next."""
        records, errors = [], []
        total_batches = (self.spec['folds'] + self.fold_workers - 1) // self.fold_workers
        for batch, start in enumerate(range(0, self.spec['folds'], self.fold_workers), 1):
            members = list(range(start, min(start + self.fold_workers, self.spec['folds'])))
            context = dict(run_name=run_name, batch=batch, total_batches=total_batches,
                           batch_folds=members)
            if status is not None:
                context['trial'] = status['trial']
            finished, failures = 0, 0

            def progress(batch_status, event_name=None):
                values = dict(context, batch_status=batch_status,
                              batch_folds_finished=finished, batch_numerical_failures=failures,
                              folds_finished=len(records) + len(errors), numerical_failures=len(errors))
                if status is not None:
                    dump(self.search_root / 'status.json', dict(status, **values))
                if event_name is not None:
                    event(event=event_name, **values)

            progress('running', 'fold_batch_started')
            futures = [pool.submit(fit_fold, (str(self.root), run_name, cfg, fold))
                       for fold in members]
            for future in as_completed(futures):
                try:
                    records.append(future.result())
                except FloatingPointError as error:
                    errors.append(str(error))
                    failures += 1
                finished += 1
                progress('running')
            progress('numerical_failure' if failures else 'complete', 'fold_batch_complete')
        records.sort(key=lambda item: item['fold'])
        return records, errors

    def smoke(self):
        data.prepare(self.root, self.raw_input, self.spec)
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42))
        cfg = self.suggest(study.ask())
        cfg.update(epochs=20, beta=0.01, tau=0.1, rho=0.5)
        run_name = 'smoke_' + digest(self.source_root / 'src/pl_staging/model.py')[:12]
        started = time.monotonic()
        event(event='smoke_scheduling', folds=self.spec['folds'],
              requested_workers=self.spec['fold_workers'], effective_workers=self.fold_workers,
              available_cpus=self.available_cpus, scheduling='strict_batches')
        with ProcessPoolExecutor(max_workers=self.fold_workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            records, errors = self.run_fold_batches(pool, run_name, cfg)
        if errors:
            raise FloatingPointError('; '.join(errors))
        assert sum(r['validation']['pairs'] for r in records) == self.expected_pairs('cv')
        event(event='smoke_complete', seconds=time.monotonic() - started,
              folds=self.spec['folds'], effective_workers=self.fold_workers)

    def search(self):
        self.lock_protocol()
        if (self.search_root / 'selection.json').exists():
            event(event='search_already_frozen')
            return
        study = self.study()
        event(event='search_scheduling', folds=self.spec['folds'],
              requested_workers=self.spec['fold_workers'], effective_workers=self.fold_workers,
              available_cpus=self.available_cpus, scheduling='strict_batches')
        with ProcessPoolExecutor(max_workers=self.fold_workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            while True:
                running = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]
                if running:
                    assert len(running) == 1
                    trial = optuna.trial.Trial(study, running[0]._trial_id)
                elif any(t.state == optuna.trial.TrialState.WAITING for t in study.trials) or len(study.trials) < self.spec['trials']:
                    trial = study.ask()
                else:
                    break
                cfg = self.suggest(trial)
                trial.set_user_attr('configuration', cfg)
                name = f'trial{trial.number:04d}'
                dump(self.search_root / 'trial_configs' / f'{name}.json', cfg)
                self.save_sampler(study)
                status = dict(event='trial_started', trial=trial.number, total=self.spec['trials'],
                              config=cfg, time=utc())
                dump(self.search_root / 'status.json', status)
                event(**{k: v for k, v in status.items() if k != 'time'})
                records, errors = self.run_fold_batches(pool, name, cfg, status)
                if errors:
                    trial.set_user_attr('numerical_failure', errors)
                    dump(self.search_root / 'trial_results' / f'{name}.json',
                         dict(config=cfg, errors=errors, completed_folds=records))
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    event(event='trial_failed', trial=trial.number, errors=errors)
                else:
                    pooled = self.aggregate_folds(records)
                    result = dict(config=cfg, folds=records, **pooled)
                    dump(self.search_root / 'trial_results' / f'{name}.json', result)
                    for key in ('violations', 'pairs', 'objective', 'selection_metric', 'objective_status'):
                        trial.set_user_attr(key, result[key])
                    if 'rank' in result:
                        trial.set_user_attr('rank_metrics', result['rank'])
                        trial.set_user_attr('violation_ratio_times_conditional_gap',
                                            result['violation_ratio_times_conditional_gap'])
                    if result['objective'] is None:
                        trial.set_user_attr('objective_undefined', result['objective_reason'])
                        study.tell(trial, state=optuna.trial.TrialState.FAIL)
                        event(event='trial_objective_undefined', trial=trial.number, **pooled)
                    else:
                        study.tell(trial, result['objective'])
                        event(event='trial_complete', trial=trial.number,
                              **pooled, best_value=study.best_value)
                self.save_sampler(study)
                study.trials_dataframe().to_csv(self.search_root / 'trials.csv', index=False)
        self.verify_protocol()
        study.trials_dataframe().to_csv(self.search_root / 'trials.csv', index=False)
        event(event='search_complete', trials=len(study.trials))

    def freeze(self):
        self.lock_protocol()
        selection_path = self.search_root / 'selection.json'
        if selection_path.exists():
            verify(read(selection_path), self.root)
            return
        study = self.study()
        assert len(study.trials) == self.spec['trials']
        assert all(t.state.is_finished() for t in study.trials)
        rows = []
        paths = [self.search_root / 'protocol.json', self.search_root / 'trials.csv']
        for trial in study.trials:
            result_path = self.search_root / 'trial_results' / f'trial{trial.number:04d}.json'
            result = read(result_path)
            paths.append(result_path)
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            assert result['selection_metric'] == self.selection_metric
            assert result['objective'] == trial.value
            pooled = self.aggregate_folds(result['folds'])
            assert all(result[key] == value for key, value in pooled.items())
            assert result['pairs'] == self.expected_pairs('cv') and len(result['folds']) == self.spec['folds']
            rows.append(dict(trial=trial.number, epochs=result['config']['epochs'],
                             violations=result['violations'], pairs=result['pairs'], ties=result['ties'],
                             violation_ratio=result['violation_ratio'], objective=result['objective'],
                             violation_ratio_times_conditional_gap=result.get('violation_ratio_times_conditional_gap'),
                             all_violation_gap=result.get('rank', {}).get('all_violation_gap'),
                             rank_violation_ratio=result.get('rank', {}).get('violation_ratio'),
                             conditional_violation_gap=result.get('rank', {}).get('conditional_violation_gap')))
        if not rows:
            raise RuntimeError('No complete trial with a defined objective; external test remains unopened')
        ranking = pd.DataFrame(rows).sort_values(['objective', 'epochs', 'trial'])
        ranking.to_csv(self.search_root / 'cv_ranking.csv', index=False)
        winner = int(ranking.iloc[0].trial)
        result = read(self.search_root / 'trial_results' / f'trial{winner:04d}.json')
        paths.append(self.search_root / 'cv_ranking.csv')
        dump(selection_path, dict(selected_trial=winner, config=result['config'], cv=result,
                                  total_attempts=len(study.trials), successful_trials=len(rows),
                                  failed_trials=len(study.trials) - len(rows),
                                  undefined_objective_trials=sum('objective_undefined' in t.user_attrs for t in study.trials),
                                  frozen_at=utc(),
                                  selection_metric=self.selection_metric,
                                  tie_break=['objective', 'epochs', 'trial'], hashes=hashes(paths, self.root)))
        event(event='selection_frozen', trial=winner, selection_metric=self.selection_metric, cv=result['objective'])

    def refit(self):
        self.lock_protocol()
        selection = read(self.search_root / 'selection.json')
        verify(selection, self.root)
        self.verify_protocol()
        folder = self.root / 'data/refit'
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / 'complete.json').exists():
            verify(read(folder / 'complete.json'), self.root)
            return
        cfg = selection['config']
        x, frame = data.load_training(self.root)
        model, seconds = fit_model(x, frame, cfg, folder)
        scores = score_model(model, x)
        if np.ptp(scores['stage_mean']) <= 0:
            raise FloatingPointError('Full-training refit has constant q-mean stages')
        mapping = fit_rank_map(scores['stage_mean'], frame.RID.to_numpy())
        scores['rank_stage'] = apply_rank_map(scores['stage_mean'], mapping)
        dump(folder / 'rank_map.json', mapping)
        save_scores(folder, scores)
        record = dict(config=cfg, selected_trial=selection['selected_trial'], fit_seconds=seconds,
                      completed_at=utc(), train=metrics.counts(frame, scores['stage_mean']),
                      train_rank=metrics.gaps(frame, scores['rank_stage']),
                      hashes=hashes([folder / 'model.pt', folder / 'scores.npz', folder / 'history.json',
                                     folder / 'score_diagnostics.json', folder / 'rank_map.json', self.search_root / 'selection.json'], self.root))
        record['train_violation_ratio_times_conditional_gap'] = metrics.violation_ratio_times_conditional_gap(
            record['train'], record['train_rank'])
        dump(folder / 'complete.json', record)
        event(event='refit_complete', train=record['train'], seconds=seconds)

    def test(self):
        self.lock_protocol()
        selection = read(self.search_root / 'selection.json')
        refit = read(self.root / 'data/refit/complete.json')
        verify(selection, self.root)
        verify(refit, self.root)
        self.verify_protocol()
        folder = self.root / 'data/test'
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / 'complete.json').exists():
            verify(read(folder / 'complete.json'), self.root)
            event(event='test_already_complete', result=read(folder / 'complete.json')['test'])
            return
        # This is the first stage permitted to open the held-out feature matrix.
        access_keys = dict(selection_sha256=digest(self.search_root / 'selection.json'),
                           checkpoint_sha256=digest(self.root / 'data/refit/model.pt'))
        if (folder / 'access.json').exists():
            access = read(folder / 'access.json')
            if any(access[key] != value for key, value in access_keys.items()):
                raise RuntimeError('Held-out evaluation was already started with a different model')
        else:
            dump(folder / 'access.json', dict(first_test_load_at=utc(), **access_keys))
        x, frame = data.load_test(self.root)
        checkpoint = torch.load(self.root / 'data/refit/model.pt', map_location='cpu', weights_only=True)
        model = ProbabilisticAE(checkpoint['input_dim'], checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        scores = score_model(model, x)
        mapping = read(self.root / 'data/refit/rank_map.json')
        scores['rank_stage'] = apply_rank_map(scores['stage_mean'], mapping)
        save_scores(folder, scores)
        result = metrics.counts(frame, scores['stage_mean'])
        assert result['pairs'] == self.expected_pairs('test')
        rank_result = metrics.gaps(frame, scores['rank_stage'])
        product = metrics.violation_ratio_times_conditional_gap(result, rank_result)
        record = dict(test=result, test_rank=rank_result,
                      test_violation_ratio_times_conditional_gap=product,
                      selected_trial=selection['selected_trial'], scored_at=utc(), score_diagnostics=scores['diagnostics'],
                      hashes=hashes([folder / 'scores.npz', folder / 'score_diagnostics.json', folder / 'access.json',
                                     self.root / 'data/refit/model.pt', self.root / 'data/refit/rank_map.json',
                                     self.search_root / 'selection.json'], self.root))
        summary = dict(selected_trial=selection['selected_trial'], config=selection['config'],
                       total_attempts=selection['total_attempts'], successful_trials=selection['successful_trials'],
                       failed_trials=selection['failed_trials'],
                       undefined_objective_trials=selection.get('undefined_objective_trials', 0),
                       cv=({key: selection['cv'][key] for key in ('violations', 'pairs', 'ties', 'violation_ratio',
                                                                                'rank', 'selection_metric', 'objective',
                                                                                'violation_ratio_times_conditional_gap')}
                           if selection['cv'] is not None else None),
                       mode=selection.get('mode', 'search'),
                       train=refit['train'], test=result, train_rank=refit['train_rank'], test_rank=rank_result,
                       train_violation_ratio_times_conditional_gap=refit['train_violation_ratio_times_conditional_gap'],
                       test_violation_ratio_times_conditional_gap=product,
                       selection_metric=self.selection_metric, score=self.spec['score'], completed_at=utc())
        dump(self.root / 'reports/test_summary.json', summary)
        dump(folder / 'complete.json', record)
        event(event='test_complete', test=result, test_rank=rank_result,
              test_violation_ratio_times_conditional_gap=product)

    def audit(self):
        self.lock_protocol()
        selection = read(self.search_root / 'selection.json')
        refit = read(self.root / 'data/refit/complete.json')
        test = read(self.root / 'data/test/complete.json')
        for record in (selection, refit, test):
            verify(record, self.root)
        assert selection['frozen_at'] < refit['completed_at'] < read(self.root / 'data/test/access.json')['first_test_load_at']
        selected_folds = 0
        if selection.get('mode', 'search') == 'search':
            study = self.study()
            assert len(study.trials) == self.spec['trials'] and all(t.state.is_finished() for t in study.trials)
            winner = min((t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE),
                         key=lambda t: (t.value, t.user_attrs['configuration']['epochs'], t.number))
            assert winner.number == selection['selected_trial']
            assert selection['selection_metric'] == self.selection_metric
            for trial in study.trials:
                if trial.state != optuna.trial.TrialState.COMPLETE and 'objective_undefined' not in trial.user_attrs:
                    continue
                result = read(self.search_root / 'trial_results' / f'trial{trial.number:04d}.json')
                pooled = self.aggregate_folds(result['folds'])
                assert result['objective'] == trial.value == pooled['objective']
                assert all(result[key] == value for key, value in pooled.items())
                if trial.state != optuna.trial.TrialState.COMPLETE:
                    assert trial.state == optuna.trial.TrialState.FAIL
                    assert pooled['objective_status'] == 'undefined'
                    assert trial.user_attrs['objective_undefined'] == pooled['objective_reason']
            for fold in range(self.spec['folds']):
                folder = self.root / 'data/runs' / f'trial{winner.number:04d}' / f'fold{fold}'
                record = read(folder / 'complete.json')
                verify(record, self.root)
                x, frame, valid = data.load_fold(self.root, fold)
                verify_saved_q_from_checkpoint(folder, x)
                with np.load(folder / 'scores.npz', allow_pickle=False) as saved:
                    assert metrics.counts(frame.loc[valid], saved['stage_mean'][valid]) == record['metrics']['validation']
                    mapping = fit_rank_map(saved['stage_mean'][~valid], frame.loc[~valid, 'RID'].to_numpy())
                    assert mapping == read(folder / 'rank_map.json')
                    calibrated = apply_rank_map(saved['stage_mean'], mapping)
                    np.testing.assert_array_equal(saved['rank_stage'], calibrated)
                    assert metrics.gaps(frame.loc[valid], calibrated[valid]) == record['metrics']['validation_rank']
                    assert metrics.gaps(frame.loc[~valid], calibrated[~valid]) == record['metrics']['train_rank']
                selected_folds += 1
        mapping = read(self.root / 'data/refit/rank_map.json')
        for split, record in (('refit', refit), ('test', test)):
            x, frame = data.load_training(self.root) if split == 'refit' else data.load_test(self.root)
            folder = self.root / 'data' / split
            verify_saved_q_from_checkpoint(folder, x, self.root / 'data/refit/model.pt')
            with np.load(folder / 'scores.npz', allow_pickle=False) as saved:
                assert metrics.counts(frame, saved['stage_mean']) == record['train' if split == 'refit' else 'test']
                if split == 'refit':
                    assert mapping == fit_rank_map(saved['stage_mean'], frame.RID.to_numpy())
                calibrated = apply_rank_map(saved['stage_mean'], mapping)
                np.testing.assert_array_equal(saved['rank_stage'], calibrated)
                rank_result = metrics.gaps(frame, calibrated)
                prefix = 'train' if split == 'refit' else 'test'
                assert rank_result == record[f'{prefix}_rank']
                assert metrics.violation_ratio_times_conditional_gap(
                    record[prefix], rank_result) == record[f'{prefix}_violation_ratio_times_conditional_gap']
        record = dict(passed=True, audited_at=utc(), mode=selection.get('mode', 'search'),
            trials=selection['total_attempts'], selected_trial=selection['selected_trial'],
            selected_folds_checked=selected_folds, q_parameter_checkpoint_checks=selected_folds + 2,
            q_parameter_comparison='bitwise float64', refit_counts_recomputed=True,
            test_counts_recomputed=True, test_only_after_selection_and_refit=True,
            source_hashes_unchanged=True, selection_metric=self.selection_metric,
            training_only_rank_calibration_recomputed=True, ranked_gap_metrics_recomputed=True,
            product_metrics_recomputed=True)
        dump(self.root / 'reports/audit.json', record)
        event(event='audit_complete', result=record)
