"""Participant-weighted empirical midranks with monotone extrapolation."""

import numpy as np


def fit_rank_map(scores, subjects):
    """Fit only on training visits, giving every participant total weight one."""
    scores = np.asarray(scores, dtype=float)
    subjects = np.asarray(subjects)
    if scores.ndim != 1 or subjects.shape != scores.shape or not len(scores):
        raise ValueError("Scores and subjects must be equally sized nonempty vectors.")
    if not np.isfinite(scores).all():
        raise ValueError("Training scores must be finite.")
    _, subject_index, counts = np.unique(subjects, return_inverse=True, return_counts=True)
    weight = 1 / counts[subject_index]
    x, index = np.unique(scores, return_inverse=True)
    if len(x) < 2:
        raise ValueError("Rank calibration requires at least two distinct training scores.")
    mass = np.bincount(index, weights=weight, minlength=len(x))
    quantile = (np.cumsum(mass) - mass / 2) / mass.sum()
    return {
        "x": x.tolist(), "quantile": quantile.tolist(),
        "minimum": float(quantile[0]), "maximum": float(quantile[-1]),
        "weighting": "equal subjects, equal visits within subject",
    }


def apply_rank_map(scores, mapping):
    """Interpolate fixed training midranks; never clip held-out scores."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Scores must be a finite vector.")
    x = np.asarray(mapping["x"])
    q = np.asarray(mapping["quantile"])
    values = np.interp(scores, x, q)
    k = max(1, int(np.ceil(.05 * (len(x) - 1))))
    low, high = scores < x[0], scores > x[-1]
    values[low] = q[0] + (scores[low] - x[0]) * (q[k] - q[0]) / (x[k] - x[0])
    values[high] = q[-1] + (scores[high] - x[-1]) * (q[-1] - q[-1-k]) / (x[-1] - x[-1-k])
    return (values - mapping["minimum"]) / (mapping["maximum"] - mapping["minimum"])
