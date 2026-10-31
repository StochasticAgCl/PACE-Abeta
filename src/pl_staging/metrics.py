"""Outcome-blind longitudinal order metrics; exact ties are violations."""
from __future__ import annotations

import numpy as np
import pandas as pd


def counts(frame: pd.DataFrame, scores) -> dict:
    """Count adjacent nonincreases, pooling pairs rather than people.

    No numerical tolerance is used: a zero difference is an exact tie. A
    population without adjacent pairs has an undefined ratio, represented by
    ``None`` so the result remains valid JSON.
    """
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(values) != len(frame):
        raise ValueError("One finite score is required for each scan")
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite staging scores cannot be evaluated")
    if not {"RID", "PHC_Age_PET"}.issubset(frame.columns):
        raise ValueError("Scan index requires RID and PHC_Age_PET")
    ages = pd.to_numeric(frame["PHC_Age_PET"], errors="raise").to_numpy(float)
    if not np.isfinite(ages).all() or frame["RID"].isna().any():
        raise ValueError("Subject identifiers and scan ages must be observed")
    ordered = frame[["RID", "PHC_Age_PET"]].copy()
    ordered["score"] = values
    ordered = ordered.sort_values(["RID", "PHC_Age_PET"], kind="mergesort")
    delta = ordered.groupby("RID", sort=False).score.diff().dropna().to_numpy()
    pairs = int(len(delta))
    violations = int(np.count_nonzero(delta <= 0))
    return {
        "violations": violations,
        "strict_decreases": int(np.count_nonzero(delta < 0)),
        "ties": int(np.count_nonzero(delta == 0)),
        "pairs": pairs,
        "violation_ratio": violations / pairs if pairs else None,
    }


def pooled_counts(records) -> dict:
    """Pool independently held-out folds using their pair denominators."""
    rows = list(records)
    result = {
        key: sum(int(row[key]) for row in rows)
        for key in ("violations", "strict_decreases", "ties", "pairs")
    }
    if result["violations"] != result["strict_decreases"] + result["ties"]:
        raise ValueError("Nonincrease counts must equal declines plus ties")
    if not 0 <= result["violations"] <= result["pairs"]:
        raise ValueError("Invalid pooled pair counts")
    result["violation_ratio"] = (
        result["violations"] / result["pairs"] if result["pairs"] else None
    )
    return result


def gaps(frame: pd.DataFrame, scores) -> dict:
    """Average strictly positive declines over all adjacent visit pairs.

    Exact ties contribute zero. The conditional gap is undefined when there
    are no declines, while the all-pair gap is then zero.
    """
    counts(frame, scores)
    if frame.duplicated(["RID", "PHC_Age_PET"]).any():
        raise ValueError("Duplicate participant/age rows make visit order ambiguous")
    ordered = frame[["RID", "PHC_Age_PET"]].copy()
    ordered["score"] = np.asarray(scores, dtype=np.float64).reshape(-1)
    ordered = ordered.sort_values(["RID", "PHC_Age_PET"], kind="mergesort")
    decline = -ordered.groupby("RID", sort=False).score.diff().dropna().to_numpy()
    positive = decline[decline > 0]
    pairs, violations = int(len(decline)), int(len(positive))
    total = float(positive.sum())
    return dict(n_pairs=pairs, n_violations=violations,
                violation_ratio=violations / pairs if pairs else None,
                conditional_violation_gap=total / violations if violations else None,
                total_violation_gap=total, all_violation_gap=total / pairs if pairs else None,
                ties=int(np.count_nonzero(decline == 0)))


def pooled_gaps(records) -> dict:
    """Pool fold gap sums and pair counts, rather than averaging fold means."""
    import math
    rows = list(records)
    for row in rows:
        if not 0 <= row["n_violations"] <= row["n_pairs"]:
            raise ValueError("Invalid strict-decline counts")
        if not np.isfinite(row["total_violation_gap"]) or row["total_violation_gap"] < 0:
            raise ValueError("Gap totals must be finite and nonnegative")
        if row["n_violations"] == 0 and row["total_violation_gap"] != 0:
            raise ValueError("No violations must imply a zero gap total")
    pairs = sum(int(row["n_pairs"]) for row in rows)
    violations = sum(int(row["n_violations"]) for row in rows)
    total = math.fsum(float(row["total_violation_gap"]) for row in rows)
    return dict(n_pairs=pairs, n_violations=violations,
                violation_ratio=violations / pairs if pairs else None,
                conditional_violation_gap=total / violations if violations else None,
                total_violation_gap=total, all_violation_gap=total / pairs if pairs else None,
                ties=sum(int(row["ties"]) for row in rows))


def violation_ratio_times_conditional_gap(raw_counts, rank_gaps):
    """Multiply the original nonincrease VR by the positive rank-gap mean.

    Unlike all_violation_gap, the first factor counts exact raw-score ties.
    No raw violations defines a zero product. A nonzero ratio with no positive
    rank gaps leaves the conditional mean, and therefore the product, undefined.
    """
    if raw_counts['pairs'] != rank_gaps['n_pairs']:
        raise ValueError('The product factors must use the same adjacent pairs')
    ratio = raw_counts['violation_ratio']
    conditional = rank_gaps['conditional_violation_gap']
    if ratio is None:
        return None
    if not np.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError('Violation ratio must be finite and between zero and one')
    if conditional is None:
        return 0.0 if ratio == 0 else None
    if not np.isfinite(conditional) or conditional < 0:
        raise ValueError('Conditional gap must be finite and nonnegative')
    return float(ratio * conditional)
