import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.stats import beta

from pace_abeta.analysis.layout import export
from pace_abeta.analysis.metrics import METHODS, conditional_gap_tests, diagnosis_tests
from pace_abeta.analysis.plot_blocks import diagnosis_plot, format_bootstrap_p_value


@pytest.fixture
def visits():
    rng = np.random.default_rng(845)
    rows = []
    for rid in range(1, 25):
        codes = [1, 2, 3] if rid <= 12 else ([1, 2] if rid <= 18 else [2, 3])
        if rid % 3 == 0:
            codes = [*codes, codes[0]]
        subject_effect = rng.uniform(.05, .6)
        for code in codes:
            score = subject_effect + .08 * (code - 1) + rng.uniform(-.06, .06)
            rows.append((rid, code, score))
    return pd.DataFrame(rows, columns=["RID", "PHC_Diagnosis", "Pseudotime_Normalized"])


def scalar_statistics(frame, drawn_subjects):
    """Replay pooled means and joint participant contrasts using scalar sums."""
    sums, counts = [], []
    for rid in drawn_subjects:
        subject = frame.loc[frame.RID == rid]
        scores = [subject.loc[subject.PHC_Diagnosis == code, "Pseudotime_Normalized"].tolist()
                  for code in (1, 2, 3)]
        sums.append([sum(group) for group in scores])
        counts.append([len(group) for group in scores])
    n = len(drawn_subjects)
    totals = [sum(row[j] for row in counts) for j in range(3)]
    means = [sum(row[j] for row in sums) / totals[j] for j in range(3)]
    influence = [[(sums[i][j] - means[j] * counts[i][j]) / (totals[j] / n)
                  for j in range(3)] for i in range(n)]
    differences = [means[left] - means[right] for left, right in ((0, 1), (1, 2))]
    errors = [math.sqrt(sum((row[left] - row[right]) ** 2 for row in influence) / (n * (n - 1)))
              for left, right in ((0, 1), (1, 2))]
    return np.asarray(means), np.asarray(differences), np.asarray(errors), np.asarray(influence)


def test_scalar_joint_bootstrap_replays_statistics_tail_and_intervals(visits):
    original = visits.copy(deep=True)
    repetitions = 79
    result = diagnosis_tests(visits, n_bootstrap=repetitions, seed=42, batch_size=13)
    ids = sorted(visits.RID.unique())
    means, delta, se, influence = scalar_statistics(visits, ids)
    rng = np.random.default_rng(42)
    samples = []
    for indices in rng.integers(0, len(ids), size=(repetitions, len(ids)), dtype=np.int64):
        _, sample_delta, sample_se, _ = scalar_statistics(visits, [ids[index] for index in indices])
        samples.append((sample_delta - delta) / sample_se)
    samples = np.asarray(samples)
    assert [row["comparison"] for row in result["comparisons"]] == ["CN_vs_MCI", "MCI_vs_AD"]
    assert [(row["reference"], row["comparator"]) for row in result["comparisons"]] == [("CN", "MCI"), ("MCI", "AD")]
    for j, row in enumerate(result["comparisons"]):
        left, right = ((0, 1), (1, 2))[j]
        extreme = int(np.count_nonzero(np.abs(samples[:, j]) >= abs(delta[j] / se[j])))
        critical = np.quantile(np.abs(samples[:, j]), .95)
        p = (extreme + 1) / (repetitions + 1)
        assert row["reference_mean"] == pytest.approx(means[left], rel=1e-13)
        assert row["comparator_mean"] == pytest.approx(means[right], rel=1e-13)
        assert row["difference"] == pytest.approx(delta[j], rel=1e-13)
        assert row["cluster_standard_error"] == pytest.approx(se[j], rel=1e-13)
        assert row["t_observed"] == pytest.approx(delta[j] / se[j], rel=1e-13)
        assert row["extreme_replicates"] == extreme
        assert row["p_value"] == p
        assert row["studentized_ci_low"] == pytest.approx(delta[j] - critical * se[j], rel=1e-13)
        assert row["studentized_ci_high"] == pytest.approx(delta[j] + critical * se[j], rel=1e-13)
        assert row["p_value_floor"] == 1 / (repetitions + 1)
        assert row["p_at_resolution_floor"] == (extreme == 0)
        assert row["p_monte_carlo_se"] == pytest.approx(math.sqrt(p * (1 - p) / repetitions))
        low = beta.ppf(.025, extreme, repetitions - extreme + 1) if extreme else 0.
        high = beta.ppf(.975, extreme + 1, repetitions - extreme) if extreme < repetitions else 1.
        assert row["tail_probability_mc_ci_low"] == pytest.approx(low)
        assert row["tail_probability_mc_ci_high"] == pytest.approx(high)
        n = len(influence)
        unpaired = math.sqrt(sum(influence[:, left] ** 2 + influence[:, right] ** 2) / (n * (n - 1)))
        assert row["cluster_standard_error"] < unpaired
    assert result["invalid_replicates"] == 0
    pd.testing.assert_frame_equal(visits, original)


def test_membership_uses_visit_diagnoses_and_pooled_scores(visits):
    visits.loc[0, "Pseudotime_Normalized"] = 0.
    visits.loc[1, "Pseudotime_Normalized"] = 1000.
    excluded = pd.DataFrame({"RID": [999, None, 999], "PHC_Diagnosis": [0, np.nan, 4],
                             "Pseudotime_Normalized": [np.inf, np.nan, -100.]})
    frame = pd.concat([visits, excluded], ignore_index=True)
    result = diagnosis_tests(frame, n_bootstrap=79)
    assert result["n_visits"] == len(visits)
    assert result["n_subjects"] == visits.RID.nunique()
    assert result["excluded_visits"] == len(excluded)
    for code, label in enumerate(("CN", "MCI", "AD"), start=1):
        group = visits.loc[visits.PHC_Diagnosis == code]
        summary = result["groups"][label]
        assert summary["visits"] == len(group)
        assert summary["subjects"] == group.RID.nunique()
        assert summary["mean"] == pytest.approx(group.Pseudotime_Normalized.mean())
        assert summary["median"] == pytest.approx(group.Pseudotime_Normalized.median())
    for row, (left, right) in zip(result["comparisons"], ((1, 2), (2, 3)), strict=True):
        left_ids = set(visits.loc[visits.PHC_Diagnosis == left, "RID"])
        right_ids = set(visits.loc[visits.PHC_Diagnosis == right, "RID"])
        assert row["n_shared_subjects"] == len(left_ids & right_ids)
        assert row["n_reference_subjects"] == len(left_ids)
        assert row["n_comparator_subjects"] == len(right_ids)
    assert result == diagnosis_tests(visits, n_bootstrap=79) | {"excluded_visits": 3}


def test_batch_sizes_preserve_bootstrap_stream(visits):
    first = diagnosis_tests(visits, n_bootstrap=79, batch_size=1)
    second = diagnosis_tests(visits, n_bootstrap=79, batch_size=512)
    first.pop("batch_size")
    second.pop("batch_size")
    assert first == second


def test_numeric_and_string_participant_ids_have_the_same_draw_order(visits):
    numeric = diagnosis_tests(visits, n_bootstrap=79)
    strings = diagnosis_tests(visits.assign(RID=visits.RID.astype(str)), n_bootstrap=79)
    assert strings == numeric


def test_integer_like_string_ids_preserve_distinct_leading_zero_labels(visits):
    labels = visits.RID.astype(str)
    labels.loc[visits.RID == 2] = "01"
    frame = visits.assign(RID=labels)
    result = diagnosis_tests(frame, n_bootstrap=79)
    assert result["n_subjects"] == visits.RID.nunique()
    assert result["n_visits"] == len(visits)
    assert result["groups"] == diagnosis_tests(visits, n_bootstrap=79)["groups"]


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_eligible_scores_are_rejected(visits, invalid):
    visits.loc[0, "Pseudotime_Normalized"] = invalid
    with pytest.raises(ValueError, match="finite scores"):
        diagnosis_tests(visits, n_bootstrap=3)


@pytest.mark.parametrize("invalid", [None, np.nan, np.inf, -np.inf, "", "  "])
def test_invalid_eligible_participant_ids_are_rejected(visits, invalid):
    visits["RID"] = visits.RID.astype(object)
    visits.loc[0, "RID"] = invalid
    with pytest.raises(ValueError, match="RID"):
        diagnosis_tests(visits, n_bootstrap=3)


@pytest.mark.parametrize("column", ["RID", "PHC_Diagnosis", "Pseudotime_Normalized"])
def test_missing_columns_are_rejected(visits, column):
    with pytest.raises(ValueError, match=column):
        diagnosis_tests(visits.drop(columns=column), n_bootstrap=3)


def test_insufficient_participants_or_missing_diagnosis_group_are_rejected(visits):
    single = visits.assign(RID=1)
    with pytest.raises(ValueError, match="at least two participants"):
        diagnosis_tests(single, n_bootstrap=3)
    with pytest.raises(ValueError, match="No eligible diagnosis scores for AD"):
        diagnosis_tests(visits.loc[visits.PHC_Diagnosis != 3], n_bootstrap=3)


def test_constant_contrasts_are_rejected(visits):
    visits["Pseudotime_Normalized"] = .5
    with pytest.raises(ValueError, match="positive cluster standard errors"):
        diagnosis_tests(visits, n_bootstrap=3)


def test_missing_group_in_resample_stops_without_redrawing(monkeypatch, visits):
    class OneSubject:
        def integers(self, low, high, size, dtype):
            return np.full(size, 23, dtype=dtype)

    monkeypatch.setattr(np.random, "default_rng", lambda seed: OneSubject())
    with pytest.raises(ValueError, match="Bootstrap replicate 0 is undefined.*no resamples are dropped or redrawn"):
        diagnosis_tests(visits, n_bootstrap=3)


@pytest.mark.parametrize("key,value", [("n_bootstrap", 0), ("n_bootstrap", 2.5),
                                       ("batch_size", 0), ("batch_size", True),
                                       ("seed", -1), ("seed", None)])
def test_invalid_bootstrap_settings_are_rejected(visits, key, value):
    with pytest.raises(ValueError, match=key):
        diagnosis_tests(visits, **{key: value})


def test_diagnosis_and_gap_reports_share_statistical_metadata(visits):
    rng = np.random.default_rng(731)
    pairs = pd.DataFrame(rng.uniform(.01, .3, size=(32, 4)),
                         columns=[f"{method}_rank_decline" for method in METHODS])
    pairs.insert(0, "RID", np.repeat(np.arange(1, 17), 2))
    gap = conditional_gap_tests(pairs, n_bootstrap=79)
    diagnosis = diagnosis_tests(visits, n_bootstrap=79)
    for key in ("test", "alternative", "null_difference", "adjustment", "standard_error",
                "p_value_method", "confidence_level", "confidence_interval", "rng",
                "bootstrap_replicates", "seed", "batch_size", "invalid_replicates"):
        assert diagnosis[key] == gap[key]
    keys = set(gap["comparisons"][0]) - {"reference_cvg", "comparator_cvg"}
    assert keys <= set(diagnosis["comparisons"][0])


@pytest.mark.parametrize("p,label", [(1e-5, "p < 0.0001"), (3e-5, "p < 0.0001"),
                                     (1e-4, "p = 0.0001"), (.0325, "p = 0.0325")])
def test_common_p_display_cutoff(p, label):
    assert format_bootstrap_p_value(p) == label


def test_diagnosis_plot_uses_shared_labels_and_annotation_size(visits, tmp_path):
    comparisons = [
        {"comparison": "CN_vs_MCI", "reference": "CN", "comparator": "MCI",
         "p_value": .00003, "p_at_resolution_floor": False},
        {"comparison": "MCI_vs_AD", "reference": "MCI", "comparator": "AD",
         "p_value": .0001, "p_at_resolution_floor": True},
    ]
    figure = diagnosis_plot(visits, comparisons)
    try:
        annotations = [text for text in figure.axes[0].texts if text.get_text().startswith("p ")]
        assert {text.get_text() for text in annotations} == {"p < 0.0001", "p = 0.0001"}
        assert all(text.get_fontsize() == 6.5 and text.get_ha() == "center" for text in annotations)
        export(figure, "figure2a", tmp_path)
        assert all(text.get_fontsize() == 6.5 for text in annotations)
    finally:
        plt.close(figure)
