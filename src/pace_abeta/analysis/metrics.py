"""Participant-clustered inference for diagnosis scores and adjacent-visit reversals."""

import numpy as np
from scipy.stats import beta

METHODS = ("PL_DAPS", "Centiloids", "SLOPE", "dEBM_upstream")
LABELS = {"PL_DAPS": "PACE-Abeta", "Centiloids": "Centiloids", "SLOPE": "SLOPE", "dEBM_upstream": "dEBM"}
THRESHOLDS = (0., .025, .05, .075, .1)


def adjacent_pairs(frame):
    """Return positional row indices, with visits sorted within participant."""
    if frame.duplicated(["RID", "PHC_Age_PET"]).any():
        raise ValueError("Duplicate participant/age rows make visit order ambiguous.")
    earlier, later = [], []
    for positions in frame.groupby("RID", sort=False).indices.values():
        order = positions[np.argsort(frame.PHC_Age_PET.to_numpy()[positions], kind="stable")]
        earlier.extend(order[:-1]); later.extend(order[1:])
    return np.asarray(earlier, dtype=int), np.asarray(later, dtype=int)


def summarize_declines(decline, threshold=0.):
    """A qualifying gap is the full decline, without subtracting the threshold."""
    decline = np.asarray(decline, dtype=float)
    if not len(decline) or not np.isfinite(decline).all():
        raise ValueError("At least one finite adjacent-pair decline is required.")
    gaps = decline[decline > threshold]
    return {
        "n_pairs": len(decline), "n_violations": len(gaps),
        "violation_ratio": float(len(gaps) / len(decline)),
        "conditional_violation_gap": float(gaps.mean()) if len(gaps) else None,
        "total_violation_gap": float(gaps.sum()),
        "all_violation_gap": float(gaps.sum() / len(decline)),
        "ties": int((decline == 0).sum()),
    }


def _cluster_ratio_contrasts(gap_sums, violation_counts, contrasts=None):
    """Compute pooled-mean contrasts and participant influence-function errors.

    The final two axes are participant and group. Each group mean is a ratio
    of pooled sums to counts, not a mean of participant means. By default, the
    first group is compared with every remaining group. Explicit index pairs
    select other contrasts while retaining covariance between groups.
    """
    n = gap_sums.shape[-2]
    totals = violation_counts.sum(axis=-2)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        means = gap_sums.sum(axis=-2) / totals
        influence = (gap_sums - means[..., None, :] * violation_counts) / (
            totals[..., None, :] / n
        )
        if contrasts is None:
            differences = means[..., :1] - means[..., 1:]
            contrast_influence = influence[..., :1] - influence[..., 1:]
        else:
            left, right = np.asarray(contrasts, dtype=int).T
            differences = means[..., left] - means[..., right]
            contrast_influence = influence[..., left] - influence[..., right]
        variances = np.sum(contrast_influence**2, axis=-2) / (n * (n - 1))
        standard_errors = np.sqrt(variances)
    valid = (totals > 0).all(axis=-1)
    valid = valid & np.isfinite(means).all(axis=-1)
    valid = valid & np.isfinite(standard_errors).all(axis=-1)
    valid = valid & (standard_errors > 0).all(axis=-1)
    return means, differences, standard_errors, valid


def _bootstrap_settings(n_bootstrap, seed, batch_size):
    for name, value, minimum in (("n_bootstrap", n_bootstrap, 1),
                                 ("batch_size", batch_size, 1), ("seed", seed, 0)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer of at least {minimum}.")
        if value < minimum:
            raise ValueError(f"{name} must be an integer of at least {minimum}.")
    n_bootstrap, batch_size, seed = int(n_bootstrap), int(batch_size), int(seed)
    return n_bootstrap, seed, batch_size


def _participant_indices(values):
    """Keep participant draw order consistent for integer and string IDs.

    Integer-like strings are ordered numerically, with the original string as
    a tie breaker. Distinct labels such as "01" and "1" remain distinct IDs.
    """
    try:
        ids, positions = np.unique(values, return_inverse=True)
    except TypeError as error:
        raise ValueError("RID values must have a consistent sortable type.") from error
    if len(ids) and all(isinstance(value, str) for value in ids):
        try:
            numeric_ids = [int(value) for value in ids]
        except ValueError:
            return ids, positions
        order = np.asarray(sorted(range(len(ids)), key=lambda i: (numeric_ids[i], ids[i])))
        inverse_order = np.empty(len(ids), dtype=int)
        inverse_order[order] = np.arange(len(ids))
        return ids[order], inverse_order[positions]
    return ids, positions


def _cluster_bootstrap_statistics(gap_sums, violation_counts, differences, *,
                                  n_bootstrap, seed, batch_size,
                                  empty_group_message, contrasts=None):
    """Draw participants jointly and center each studentized contrast."""
    n_subjects = len(gap_sums)
    rng = np.random.default_rng(seed)
    statistics = np.empty((n_bootstrap, len(differences)), dtype=float)
    for start in range(0, n_bootstrap, batch_size):
        stop = min(start + batch_size, n_bootstrap)
        indices = rng.integers(0, n_subjects, size=(stop - start, n_subjects), dtype=np.int64)
        _, sampled_difference, sampled_se, good = _cluster_ratio_contrasts(
            gap_sums[indices], violation_counts[indices], contrasts
        )
        if not good.all():
            failed = int(start + np.flatnonzero(~good)[0])
            raise ValueError(
                f"Bootstrap replicate {failed} is undefined: {empty_group_message} "
                "or a contrast has a nonfinite or zero standard error. "
                "Inference is withheld; no resamples are dropped or redrawn."
            )
        statistics[start:stop] = (sampled_difference - differences) / sampled_se
    if not np.isfinite(statistics).all():
        raise ValueError("Bootstrap t statistics are nonfinite; inference is withheld.")

    return statistics


def _bootstrap_comparison(difference, standard_error, t_observed, statistics, n_bootstrap):
    """Summarize a centered bootstrap-t distribution without tail extrapolation."""
    extreme = int(np.count_nonzero(np.abs(statistics) >= abs(t_observed)))
    p_value = (extreme + 1) / (n_bootstrap + 1)
    critical = float(np.quantile(np.abs(statistics), .95, method="linear"))
    # These limits quantify Monte Carlo uncertainty in the bootstrap tail.
    mc_low = float(beta.ppf(.025, extreme, n_bootstrap - extreme + 1)) if extreme else 0.0
    mc_high = float(beta.ppf(.975, extreme + 1, n_bootstrap - extreme)) if extreme < n_bootstrap else 1.0
    return {
        "difference": float(difference),
        "cluster_standard_error": float(standard_error),
        "t_observed": float(t_observed),
        "studentized_ci_low": float(difference - critical * standard_error),
        "studentized_ci_high": float(difference + critical * standard_error),
        "p_value": p_value, "extreme_replicates": extreme,
        "bootstrap_replicates": n_bootstrap,
        "p_value_floor": 1 / (n_bootstrap + 1), "p_at_resolution_floor": extreme == 0,
        "p_monte_carlo_se": float(np.sqrt(p_value * (1 - p_value) / n_bootstrap)),
        "tail_probability_mc_ci_low": mc_low, "tail_probability_mc_ci_high": mc_high,
    }


def conditional_gap_tests(pairs, *, n_bootstrap=100_000, seed=42, batch_size=512):
    """Test pooled mean CVG differences with a paired cluster bootstrap-t.

    Resample participants, retaining all their pairs and all methods together.
    The null is zero PACE-minus-comparator difference in pooled positive-gap
    means. Inference conditions on the fitted models and training rank maps.
    An undefined observed statistic or resample raises instead of being omitted.
    """
    n_bootstrap, seed, batch_size = _bootstrap_settings(n_bootstrap, seed, batch_size)
    columns = [f"{method}_rank_decline" for method in METHODS]
    missing = [column for column in ["RID", *columns] if column not in pairs]
    if missing:
        raise ValueError(f"Conditional-gap tests require columns: {', '.join(missing)}.")
    if pairs.RID.isna().any() or any(
        (isinstance(value, str) and not value.strip())
        or (isinstance(value, (int, float, np.number)) and not np.isfinite(value))
        for value in pairs.RID
    ):
        raise ValueError("Conditional-gap tests require nonmissing, finite RID values.")
    ids, positions = _participant_indices(pairs.RID.to_numpy())
    n_subjects = len(ids)
    if n_subjects < 2:
        raise ValueError("Conditional-gap tests require at least two participants.")
    decline = pairs[columns].to_numpy(dtype=float)
    if not np.isfinite(decline).all():
        raise ValueError("Conditional-gap tests require finite declines.")

    gap_sums = np.zeros((n_subjects, len(METHODS)), dtype=float)
    violation_counts = np.zeros((n_subjects, len(METHODS)), dtype=np.int64)
    np.add.at(gap_sums, positions, np.maximum(decline, 0))
    np.add.at(violation_counts, positions, (decline > 0).astype(np.int64))
    totals = violation_counts.sum(axis=0)
    for j, method in enumerate(METHODS):
        if totals[j] == 0:
            raise ValueError(f"No positive gaps for {LABELS[method]}.")
    means, differences, standard_errors, valid = _cluster_ratio_contrasts(
        gap_sums, violation_counts
    )
    if not valid:
        raise ValueError("Observed CVG contrasts require finite, positive cluster standard errors.")
    t_observed = differences / standard_errors
    if not np.isfinite(t_observed).all():
        raise ValueError("Observed CVG contrasts require finite t statistics.")

    statistics = _cluster_bootstrap_statistics(
        gap_sums, violation_counts, differences, n_bootstrap=n_bootstrap,
        seed=seed, batch_size=batch_size,
        empty_group_message="a method has no positive gaps",
    )

    rows = []
    for method in ("SLOPE", "Centiloids", "dEBM_upstream"):
        j = METHODS.index(method) - 1
        rows.append({
            "reference": LABELS[METHODS[0]], "comparator": LABELS[method],
            "n_reference": int(totals[0]), "n_comparator": int(totals[j + 1]),
            "reference_cvg": float(means[0]), "comparator_cvg": float(means[j + 1]),
            **_bootstrap_comparison(
                differences[j], standard_errors[j], t_observed[j], statistics[:, j], n_bootstrap
            ),
        })
    return {
        "test": "Paired subject-cluster bootstrap-t", "alternative": "two-sided",
        "estimand": "PACE-minus-comparator difference in pooled conditional violation gap means",
        "null_difference": 0., "adjustment": "none", "n_comparisons": 3,
        "n_subjects": n_subjects, "n_pairs": len(pairs),
        "sample": "strictly positive adjacent-pair declines; all participants retained for resampling",
        "interpretation": "subject-level inference conditional on fitted models and training rank maps",
        "resampling": "participants sampled with replacement; all pairs and methods kept together",
        "bootstrap_replicates": n_bootstrap, "seed": seed, "batch_size": batch_size,
        "rng": "numpy.random.default_rng (PCG64)", "invalid_replicates": 0,
        "standard_error": "paired participant influence-function sandwich for differences of pooled ratios",
        "p_value_method": "(1 + count(abs(centered bootstrap t) >= abs(observed t))) / (B + 1)",
        "confidence_level": .95,
        "confidence_interval": "symmetric bootstrap-t: difference +/- quantile(abs(centered t), .95) * observed SE",
        "methods": [
            {"method": LABELS[method], "cvg": float(means[j]),
             "positive_gaps": int(totals[j]),
             "subjects_with_violations": int((violation_counts[:, j] > 0).sum()),
             "subjects_without_violations": int((violation_counts[:, j] == 0).sum())}
            for j, method in enumerate(METHODS)
        ],
        "comparisons": rows,
    }


def diagnosis_tests(data, *, n_bootstrap=100_000, seed=42, batch_size=512):
    """Test pooled diagnosis-score means with the subject-cluster bootstrap-t.

    Participants are sampled jointly, retaining all eligible visits and their
    recorded diagnoses. A participant may contribute to several diagnosis
    groups; the influence-function contrasts retain that covariance. The two
    nulls are zero CN-minus-MCI and MCI-minus-AD pooled mean-score differences.
    """
    n_bootstrap, seed, batch_size = _bootstrap_settings(n_bootstrap, seed, batch_size)
    required = ["RID", "PHC_Diagnosis", "Pseudotime_Normalized"]
    missing = [column for column in required if column not in data]
    if missing:
        raise ValueError(f"Diagnosis tests require columns: {', '.join(missing)}.")
    selected = data.loc[data.PHC_Diagnosis.isin([1, 2, 3]), required].copy()
    if selected.RID.isna().any() or any(
        (isinstance(value, str) and not value.strip())
        or (isinstance(value, (int, float, np.number)) and not np.isfinite(value))
        for value in selected.RID
    ):
        raise ValueError("Diagnosis tests require nonmissing, finite RID values.")
    ids, positions = _participant_indices(selected.RID.to_numpy())
    n_subjects = len(ids)
    if n_subjects < 2:
        raise ValueError("Diagnosis tests require at least two participants.")
    try:
        values = selected.Pseudotime_Normalized.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Diagnosis tests require finite numeric scores.") from error
    if not np.isfinite(values).all():
        raise ValueError("Diagnosis tests require finite scores.")

    labels = ("CN", "MCI", "AD")
    groups = selected.PHC_Diagnosis.to_numpy(dtype=int) - 1
    score_sums = np.zeros((n_subjects, len(labels)), dtype=float)
    visit_counts = np.zeros((n_subjects, len(labels)), dtype=np.int64)
    np.add.at(score_sums, (positions, groups), values)
    np.add.at(visit_counts, (positions, groups), 1)
    totals = visit_counts.sum(axis=0)
    for j, label in enumerate(labels):
        if totals[j] == 0:
            raise ValueError(f"No eligible diagnosis scores for {label}.")
    contrasts = ((0, 1), (1, 2))
    means, differences, standard_errors, valid = _cluster_ratio_contrasts(
        score_sums, visit_counts, contrasts
    )
    if not valid:
        raise ValueError("Observed diagnosis contrasts require finite, positive cluster standard errors.")
    t_observed = differences / standard_errors
    if not np.isfinite(t_observed).all():
        raise ValueError("Observed diagnosis contrasts require finite t statistics.")
    statistics = _cluster_bootstrap_statistics(
        score_sums, visit_counts, differences, n_bootstrap=n_bootstrap,
        seed=seed, batch_size=batch_size, contrasts=contrasts,
        empty_group_message="a diagnosis group has no eligible scores",
    )
    rows = []
    for j, (left, right) in enumerate(contrasts):
        rows.append({
            "comparison": f"{labels[left]}_vs_{labels[right]}",
            "reference": labels[left], "comparator": labels[right],
            "n_reference": int(totals[left]), "n_comparator": int(totals[right]),
            "reference_mean": float(means[left]), "comparator_mean": float(means[right]),
            "n_reference_subjects": int((visit_counts[:, left] > 0).sum()),
            "n_comparator_subjects": int((visit_counts[:, right] > 0).sum()),
            "n_shared_subjects": int(((visit_counts[:, left] > 0) & (visit_counts[:, right] > 0)).sum()),
            **_bootstrap_comparison(
                differences[j], standard_errors[j], t_observed[j], statistics[:, j], n_bootstrap
            ),
        })
    return {
        "test": "Paired subject-cluster bootstrap-t", "alternative": "two-sided",
        "estimand": "CN-minus-MCI and MCI-minus-AD differences in pooled diagnosis-score means",
        "null_difference": 0., "adjustment": "none", "n_comparisons": len(contrasts),
        "n_subjects": n_subjects, "n_visits": len(selected),
        "excluded_visits": len(data) - len(selected),
        "sample": "all visits with diagnosis 1 (CN), 2 (MCI), or 3 (AD)",
        "interpretation": "subject-level inference conditional on fitted models and training rank maps",
        "resampling": "participants sampled with replacement; all eligible visits and diagnoses kept together",
        "bootstrap_replicates": n_bootstrap, "seed": seed, "batch_size": batch_size,
        "rng": "numpy.random.default_rng (PCG64)", "invalid_replicates": 0,
        "standard_error": "paired participant influence-function sandwich for differences of pooled ratios",
        "p_value_method": "(1 + count(abs(centered bootstrap t) >= abs(observed t))) / (B + 1)",
        "confidence_level": .95,
        "confidence_interval": "symmetric bootstrap-t: difference +/- quantile(abs(centered t), .95) * observed SE",
        "groups": {
            label: {"visits": int(totals[j]), "mean": float(means[j]),
                    "median": float(np.median(values[groups == j])),
                    "subjects": int((visit_counts[:, j] > 0).sum())}
            for j, label in enumerate(labels)
        },
        "comparisons": rows,
    }
