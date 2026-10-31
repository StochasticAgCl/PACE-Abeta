"""ADNI visit-table preparation with subject splits and CN-only adjustment."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .io import digest, dump, hashes, read, verify
from .preprocessing import fit_preadjust_params_simple, apply_preadjust_params_simple

INDEX_COLUMNS = ['RID', 'PHC_Age_PET']
COVARIATES = ['PHC_Sex', 'PHC_Education']
BASELINE_DIAGNOSIS_CODES = (1, 2, 3)


def read_visits(raw_path, features, cohort_ready=False):
    """Read an unadjusted, joined PHC visit table and apply cohort eligibility."""
    frame = pd.read_csv(raw_path)
    required = INDEX_COLUMNS + COVARIATES + ['PHC_Diagnosis'] + list(features)
    if not cohort_ready:
        required += ['PHC_TRACER', 'PHC_Race', 'PHC_Ethnicity']
    missing = sorted(set(required) - set(frame))
    if missing:
        raise ValueError(f'Missing input columns: {missing}')
    if len(features) != 68 or len(set(features)) != 68:
        raise ValueError('Exactly 68 distinct regional features are required')
    for name in set(required) - {'PHC_TRACER'}:
        frame[name] = pd.to_numeric(frame[name], errors='raise')
    complete = frame[list(features) + ['PHC_Age_PET']].notna().all(axis=1)
    eligible = np.ones(len(frame), dtype=bool)
    if not cohort_ready:
        eligible = frame.PHC_TRACER.eq('FBP') & frame.PHC_Race.eq(5) & frame.PHC_Ethnicity.eq(2)
    total_rows = len(frame)
    frame = frame.loc[eligible & complete].copy()
    if not len(frame) or frame.RID.isna().any():
        raise ValueError('Eligible visits require observed numeric subject identifiers')
    if not np.isfinite(frame[list(features) + INDEX_COLUMNS].to_numpy(float)).all():
        raise ValueError('Eligible features, ages, and identifiers must be finite')
    cov = frame[COVARIATES].to_numpy(float)
    if np.isinf(cov).any():
        raise ValueError('Covariates must be finite or missing')
    if frame.duplicated(INDEX_COLUMNS).any():
        raise ValueError('Duplicate subject/scan-age observations')
    frame = frame.drop(columns=['partition'], errors='ignore')
    frame = frame.sort_values(INDEX_COLUMNS).reset_index(drop=True)
    # Keep the earliest eligible row intact; later diagnoses cannot fill baseline gaps.
    baseline = frame.drop_duplicates('RID', keep='first')
    excluded = baseline.loc[~baseline.PHC_Diagnosis.isin(BASELINE_DIAGNOSIS_CODES)]
    excluded_subjects = frame.RID.isin(excluded.RID)
    excluded_diagnosis_rows = int(excluded_subjects.sum())
    frame = frame.loc[~excluded_subjects].reset_index(drop=True)
    if frame.empty:
        raise ValueError('No eligible participants have baseline CN, MCI, or AD diagnosis')
    return frame, {'input_rows': total_rows, 'eligible_rows': len(frame),
                   'excluded_rows': total_rows - len(frame), 'cohort_ready': bool(cohort_ready),
                   'baseline_diagnosis_excluded_subjects': len(excluded),
                   'baseline_diagnosis_missing_subjects': int(excluded.PHC_Diagnosis.isna().sum()),
                   'baseline_diagnosis_other_subjects': int(excluded.PHC_Diagnosis.notna().sum()),
                   'baseline_diagnosis_excluded_rows': excluded_diagnosis_rows}


def prepare(root, raw_path, spec):
    """Fit full-training and fold-local preprocessing from raw regional SUVRs."""
    root, raw_path = Path(root).resolve(), Path(raw_path).resolve()
    destination = root / 'data/prepared'
    metadata_path = destination / 'metadata.json'
    preparation_spec = {k: spec[k] for k in ('seed', 'split_seed', 'fold_seed', 'folds', 'features')}
    preparation_spec['cohort_ready'] = bool(spec.get('cohort_ready', False))
    preparation_spec['baseline_diagnosis_codes'] = list(BASELINE_DIAGNOSIS_CODES)
    source_hash = digest(raw_path)
    if metadata_path.exists():
        saved = read(metadata_path)
        if saved['raw_sha256'] != source_hash or saved['preparation_spec'] != preparation_spec:
            raise RuntimeError('Prepared inputs or preparation settings have changed; use a new work directory')
        verify(saved, root)
        return saved
    features = list(spec['features'])
    frame, eligibility = read_visits(raw_path, features, preparation_spec['cohort_ready'])
    sizes = frame.groupby('RID', sort=True).size().rename('n_scans')
    test_ids = sizes[sizes.gt(1)].sample(frac=.5, random_state=spec['split_seed']).index
    partition = sizes.reset_index()
    partition['partition'] = np.where(partition.RID.isin(test_ids), 'test', 'train')
    frame = frame.merge(partition[['RID', 'partition']], on='RID', validate='many_to_one')
    frames = {s: frame[frame.partition.eq(s)].reset_index(drop=True) for s in ('train', 'test')}
    train = frames['train']
    if set(train.RID) & set(frames['test'].RID):
        raise ValueError('Training and test participants overlap')
    people = train.groupby('RID', sort=True).size().rename('visits').reset_index()
    people['fold'] = -1
    longitudinal = people.index[people.visits.gt(1)].to_numpy()
    if len(longitudinal) < spec['folds']:
        raise ValueError('Too few longitudinal training participants for the requested folds')
    splitter = KFold(n_splits=spec['folds'], shuffle=True, random_state=spec['fold_seed'])
    for fold, (_, valid) in enumerate(splitter.split(longitudinal)):
        people.loc[longitudinal[valid], 'fold'] = fold
    index = train[INDEX_COLUMNS].copy()
    index['fold'] = index.RID.map(people.set_index('RID').fold)
    destination.mkdir(parents=True, exist_ok=True)
    fold_dir = destination / 'folds'
    fold_dir.mkdir(exist_ok=True)
    xt, betas, means = fit_preadjust_params_simple(train[features].to_numpy(),
        train[COVARIATES].to_numpy(), train[['PHC_Diagnosis']].to_numpy(), dx_code=1)
    imputation = train.loc[train.PHC_Diagnosis.eq(1), COVARIATES].mean().to_numpy()
    xv = apply_preadjust_params_simple(frames['test'][features].to_numpy(),
        frames['test'][COVARIATES].to_numpy(), betas, means, cov_impute_means=imputation)
    if not np.isfinite(xt).all() or not np.isfinite(xv).all():
        raise ValueError('CN covariate adjustment did not produce finite features')
    for name, visits in frames.items():
        visits.to_csv(destination / f'{name}_input.csv', index=False)
        visits[INDEX_COLUMNS].to_csv(destination / f'{name}_index.csv', index=False)
    partition.to_csv(destination / 'subject_partition.csv', index=False)
    people.to_csv(destination / 'subject_folds.csv', index=False)
    index.to_csv(destination / 'scan_index.csv', index=False)
    np.savez(destination / 'features.npz', train=xt, test=xv, betas=betas,
             means=means, imputation=imputation)
    np.savez(destination / 'preprocessing.npz', betas=betas, means=means, imputation=imputation,
             features=np.asarray(features))
    dump(destination / 'features.json', features)
    inventory = []
    for fold in range(spec['folds']):
        held = index.fold.eq(fold).to_numpy()
        fitting = train.loc[~held]
        local, b, m = fit_preadjust_params_simple(fitting[features].to_numpy(),
            fitting[COVARIATES].to_numpy(), fitting[['PHC_Diagnosis']].to_numpy(), dx_code=1)
        imp = fitting.loc[fitting.PHC_Diagnosis.eq(1), COVARIATES].mean().to_numpy()
        x = apply_preadjust_params_simple(train[features].to_numpy(), train[COVARIATES].to_numpy(),
                                         b, m, cov_impute_means=imp)
        np.testing.assert_allclose(local, x[~held], rtol=0, atol=1e-12)
        if not np.isfinite(x).all():
            raise ValueError(f'Nonfinite preprocessed features in fold {fold}')
        np.savez(fold_dir / f'fold{fold}_preprocessing.npz', x=x, validation=held,
                 betas=b, means=m, imputation=imp)
        inventory.append(dict(fold=fold, train_people=int(fitting.RID.nunique()),
            validation_people=int(train.loc[held, 'RID'].nunique()), train_scans=len(fitting),
            validation_scans=int(held.sum()), validation_pairs=int(held.sum())-int(train.loc[held, 'RID'].nunique()),
            adjustment_CN_scans=int(fitting.PHC_Diagnosis.eq(1).sum())))
    pd.DataFrame(inventory).to_csv(fold_dir / 'fold_inventory.csv', index=False)
    metadata = dict(seed=spec['seed'], split_seed=spec['split_seed'], fold_seed=spec['fold_seed'],
        folds=spec['folds'], features=features, preparation_spec=preparation_spec,
        raw_sha256=source_hash, eligibility=eligibility, train_subjects=int(train.RID.nunique()),
        train_scans=len(train), test_subjects=int(frames['test'].RID.nunique()),
        test_scans=len(frames['test']), test_pairs=len(frames['test'])-int(frames['test'].RID.nunique()),
        pooled_cv_pairs=int(sum(v['validation_pairs'] for v in inventory)),
        single_visit_people_always_training=int(people.visits.eq(1).sum()),
        longitudinal_training_people=len(longitudinal), fold_inventory=inventory,
        preprocessing='Training-CN sex/education adjustment; CN-mean covariate imputation; no age adjustment or feature standardization.',
        hashes=hashes(sorted(p for p in destination.rglob('*') if p.is_file()), root))
    dump(metadata_path, metadata)
    return metadata


def load_fold(root, fold):
    prepared = Path(root) / 'data/prepared'
    metadata = read(prepared / 'metadata.json')
    if fold not in range(metadata['folds']):
        raise ValueError('Fold outside the prepared range')
    frame = pd.read_csv(prepared / 'scan_index.csv')
    with np.load(prepared / 'folds' / f'fold{fold}_preprocessing.npz', allow_pickle=False) as archive:
        x, valid = archive['x'].copy(), archive['validation'].astype(bool, copy=True)
    np.testing.assert_array_equal(valid, frame.fold.eq(fold).to_numpy())
    if len(frame) != len(x):
        raise ValueError('Fold features/index misalignment')
    return x, frame, valid


def _load_split(root, split):
    prepared = Path(root) / 'data/prepared'
    frame = pd.read_csv(prepared / f'{split}_index.csv')
    with np.load(prepared / 'features.npz', allow_pickle=False) as archive:
        x = archive[split].copy()
    if len(frame) != len(x):
        raise ValueError('Features/index misalignment')
    return x, frame


def load_training(root):
    return _load_split(root, 'train')


def load_test(root):
    return _load_split(root, 'test')


class SubjectBatcher:
    """Uniform whole-subject batches with unbiased visit-normalized sums.

    For b sampled subjects out of M and N total visits, divide all summed
    subject contributions by b*N/M, not the random batch visit count.
    Each subject occurs exactly once in a randomly reshuffled epoch.
    """
    def __init__(self, subjects, ages, batch_subjects):
        import torch
        subjects, ages = np.asarray(subjects), np.asarray(ages, dtype=float)
        if len(subjects) != len(ages) or not len(subjects):
            raise ValueError('Subjects and ages must be nonempty and aligned')
        if pd.isna(subjects).any() or not np.isfinite(ages).all():
            raise ValueError('Subject and time must be observed')
        if int(batch_subjects) < 1:
            raise ValueError('Batch subject count must be positive')
        self.groups=[]
        for sid in np.unique(subjects):
            g=np.flatnonzero(subjects==sid)
            g=g[np.argsort(ages[g],kind='stable')]
            if np.any(np.diff(ages[g]) <= 0):
                raise ValueError('Full-order likelihood requires strictly ordered visit times')
            self.groups.append(g)
        self.batch_subjects=int(batch_subjects)
        self.n_subjects=len(self.groups)
        self.n_visits=len(subjects)

    def _pack(self, keys):
        import torch
        groups=[self.groups[int(k)] for k in keys]
        ids=np.concatenate(groups)
        padded=np.zeros((len(groups),max(map(len,groups))),dtype=np.int64)
        mask=np.zeros_like(padded,dtype=bool)
        offset=0
        for row,g in enumerate(groups):
            padded[row,:len(g)]=np.arange(offset,offset+len(g))
            mask[row,:len(g)]=True
            offset+=len(g)
        normalizer=len(groups)*self.n_visits/self.n_subjects
        return (torch.from_numpy(ids),torch.from_numpy(padded),torch.from_numpy(mask),normalizer)

    def batches(self):
        import torch
        keys=torch.randperm(self.n_subjects).tolist()
        for start in range(0,self.n_subjects,self.batch_subjects):
            yield self._pack(keys[start:start+self.batch_subjects])
