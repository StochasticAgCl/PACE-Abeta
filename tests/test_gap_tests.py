import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pace_abeta.analysis.layout import export
from pace_abeta.analysis.metrics import METHODS, conditional_gap_tests
from pace_abeta.analysis.plot_blocks import gap_plot


@pytest.fixture
def declines():
    rng = np.random.default_rng(731)
    values = rng.uniform(-.08, .3, size=(32, 4))
    values[::2] = rng.uniform(.02, .2, size=(16, 4))
    values[1, 0] = 0
    values[3, 1] = 0
    frame = pd.DataFrame(values, columns=[f"{method}_rank_decline" for method in METHODS])
    frame.insert(0, "RID", np.repeat(np.arange(1, 17), 2))
    return frame


def small_test(frame, **kwargs):
    return conditional_gap_tests(frame, n_bootstrap=79, **kwargs)


def direct_statistics(frame, drawn_subjects):
    """Replay the ratio and paired sandwich using scalar participant sums."""
    sums, counts = [], []
    for subject in drawn_subjects:
        visits = frame.loc[frame.RID == subject]
        method_values = [visits[f"{method}_rank_decline"].tolist() for method in METHODS]
        sums.append([sum(value for value in values if value > 0) for values in method_values])
        counts.append([sum(value > 0 for value in values) for values in method_values])
    n = len(drawn_subjects)
    totals = [sum(row[j] for row in counts) for j in range(4)]
    means = [sum(row[j] for row in sums) / totals[j] for j in range(4)]
    influence = [[(sums[i][j] - means[j] * counts[i][j]) * n / totals[j]
                  for j in range(4)] for i in range(n)]
    difference = [means[0] - means[j] for j in range(1, 4)]
    se = [math.sqrt(sum((row[0] - row[j]) ** 2 for row in influence) / (n * (n - 1)))
          for j in range(1, 4)]
    return np.asarray(means), np.asarray(difference), np.asarray(se), np.asarray(influence)


def test_gap_samples_exclude_zero_and_negative_declines(declines):
    original = declines.copy(deep=True)
    result = small_test(declines)
    assert [row["comparator"] for row in result["comparisons"]] == ["SLOPE", "Centiloids", "dEBM"]
    assert result["n_subjects"] == 16
    assert result["n_pairs"] == 32
    for row in result["methods"]:
        key = {"PACE-Abeta": "PL_DAPS", "dEBM": "dEBM_upstream"}.get(row["method"], row["method"])
        values = declines[f"{key}_rank_decline"]
        positive = values[values > 0]
        assert row["positive_gaps"] == len(positive)
        assert row["cvg"] == pytest.approx(positive.mean())
    assert result["alternative"] == "two-sided"
    assert result["adjustment"] == "none"
    assert result["n_comparisons"] == 3
    assert all("p_value" in row and "p_adj" not in row and "u_statistic" not in row
               for row in result["comparisons"])
    pd.testing.assert_frame_equal(declines, original)


def test_paired_covariance_is_included_in_standard_error(declines):
    reference = declines.PL_DAPS_rank_decline.to_numpy()
    positive = np.maximum(reference, .01)
    declines["PL_DAPS_rank_decline"] = positive
    declines["Centiloids_rank_decline"] = positive * 1.15 + np.linspace(.001, .003, len(positive))
    result = small_test(declines)
    _, _, expected_se, influence = direct_statistics(declines, sorted(declines.RID.unique()))
    row = next(row for row in result["comparisons"] if row["comparator"] == "Centiloids")
    n = len(influence)
    unpaired_se = math.sqrt(sum(influence[:, 0] ** 2 + influence[:, 1] ** 2) / (n * (n - 1)))
    assert row["cluster_standard_error"] == pytest.approx(expected_se[0], rel=1e-13)
    assert row["cluster_standard_error"] < unpaired_se / 5


def test_scalar_replay_matches_centered_studentization_p_and_interval(declines):
    zero_subject = pd.DataFrame({"RID": [99], **{
        f"{method}_rank_decline": [-.2] for method in METHODS
    }})
    frame = pd.concat([declines, zero_subject], ignore_index=True)
    repetitions = 79
    result = conditional_gap_tests(frame, n_bootstrap=repetitions, seed=42, batch_size=13)
    ids = sorted(frame.RID.unique())
    means, delta, se, _ = direct_statistics(frame, ids)
    rng = np.random.default_rng(42)
    t_star = []
    for indices in rng.integers(0, len(ids), size=(repetitions, len(ids)), dtype=np.int64):
        _, sampled_delta, sampled_se, _ = direct_statistics(frame, [ids[index] for index in indices])
        t_star.append((sampled_delta - delta) / sampled_se)
    t_star = np.asarray(t_star)
    keys = {"Centiloids": 0, "SLOPE": 1, "dEBM": 2}
    for row in result["comparisons"]:
        j = keys[row["comparator"]]
        extreme = int(np.sum(np.abs(t_star[:, j]) >= abs(delta[j] / se[j])))
        critical = np.quantile(np.abs(t_star[:, j]), .95)
        assert row["difference"] == pytest.approx(delta[j], rel=1e-13)
        assert row["cluster_standard_error"] == pytest.approx(se[j], rel=1e-13)
        assert row["t_observed"] == pytest.approx(delta[j] / se[j], rel=1e-13)
        assert row["extreme_replicates"] == extreme
        assert row["p_value"] == (extreme + 1) / (repetitions + 1)
        assert row["studentized_ci_low"] == pytest.approx(delta[j] - critical * se[j], rel=1e-13)
        assert row["studentized_ci_high"] == pytest.approx(delta[j] + critical * se[j], rel=1e-13)
        assert row["p_value_floor"] == 1 / (repetitions + 1)
        assert row["p_at_resolution_floor"] == (extreme == 0)
    assert result["n_subjects"] == 17
    assert all(row["subjects_without_violations"] == 1 for row in result["methods"])
    np.testing.assert_allclose([row["cvg"] for row in result["methods"]], means)


def test_batch_size_does_not_change_resamples_or_inference(declines):
    first = small_test(declines, batch_size=1)
    second = small_test(declines, batch_size=512)
    first.pop("batch_size")
    second.pop("batch_size")
    assert first == second


def test_numeric_and_string_participant_ids_have_the_same_draw_order(declines):
    numeric = small_test(declines)
    strings = small_test(declines.assign(RID=declines.RID.astype(str)))
    assert strings == numeric


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_declines_are_rejected(declines, invalid):
    declines.loc[0, "Centiloids_rank_decline"] = invalid
    with pytest.raises(ValueError, match="finite declines"):
        small_test(declines)


@pytest.mark.parametrize("invalid", [None, np.nan, np.inf, -np.inf, "", "  "])
def test_missing_or_invalid_participant_ids_are_rejected(declines, invalid):
    declines["RID"] = declines.RID.astype(object)
    declines.loc[0, "RID"] = invalid
    with pytest.raises(ValueError, match="RID"):
        small_test(declines)


def test_missing_rid_column_and_single_participant_are_rejected(declines):
    with pytest.raises(ValueError, match="RID"):
        small_test(declines.drop(columns="RID"))
    declines["RID"] = 1
    with pytest.raises(ValueError, match="at least two participants"):
        small_test(declines)


@pytest.mark.parametrize("method", METHODS)
def test_empty_positive_gap_samples_are_rejected(declines, method):
    declines[f"{method}_rank_decline"] = 0.
    with pytest.raises(ValueError, match="No positive gaps"):
        small_test(declines)


def test_zero_observed_standard_error_is_rejected(declines):
    declines["SLOPE_rank_decline"] = declines.PL_DAPS_rank_decline
    with pytest.raises(ValueError, match="positive cluster standard errors"):
        small_test(declines)


def test_degenerate_resample_stops_inference_without_redrawing(monkeypatch, declines):
    class RepeatedSubject:
        def integers(self, low, high, size, dtype):
            return np.zeros(size, dtype=dtype)

    monkeypatch.setattr(np.random, "default_rng", lambda seed: RepeatedSubject())
    with pytest.raises(ValueError, match="Bootstrap replicate 0 is undefined"):
        small_test(declines)


def test_zero_violation_resample_is_not_assigned_zero_cvg(monkeypatch, declines):
    declines.loc[declines.RID == 1, [f"{method}_rank_decline" for method in METHODS]] = -.1

    class ZeroViolationSubjects:
        def integers(self, low, high, size, dtype):
            return np.zeros(size, dtype=dtype)

    monkeypatch.setattr(np.random, "default_rng", lambda seed: ZeroViolationSubjects())
    with pytest.raises(ValueError, match="no resamples are dropped or redrawn"):
        small_test(declines)


@pytest.mark.parametrize("key,value", [("n_bootstrap", 0), ("n_bootstrap", 2.5),
                                       ("batch_size", 0), ("batch_size", True),
                                       ("seed", -1), ("seed", None)])
def test_invalid_bootstrap_settings_are_rejected(declines, key, value):
    with pytest.raises(ValueError, match=key):
        conditional_gap_tests(declines, **{key: value})


def test_hidden_outliers_remain_in_tests_and_do_not_set_plot_range(tmp_path):
    base = np.r_[np.arange(1, 12), 1000.] / 1000
    pairs = pd.DataFrame({
        "RID": np.arange(1, 13),
        "PL_DAPS_rank_decline": .7 * base,
        "SLOPE_rank_decline": base,
        "Centiloids_rank_decline": .9 * base,
        "dEBM_upstream_rank_decline": .8 * base,
    })
    result = small_test(pairs)
    comparisons = [{**row, "reference": "DAPS"} for row in result["comparisons"]]
    assert all(row["n_reference"] == row["n_comparator"] == 12 for row in comparisons)
    assert comparisons[0]["difference"] == pytest.approx((pairs.PL_DAPS_rank_decline - pairs.SLOPE_rank_decline).mean())
    # Apply the common display cutoff regardless of the bootstrap floor flag.
    comparisons[0].update(p_value=.00003, p_at_resolution_floor=False)
    comparisons[1].update(p_value=.00001, p_value_floor=.00001, p_at_resolution_floor=True)
    comparisons[2].update(p_value=.0001, p_at_resolution_floor=False)

    methods = {"SLOPE": "SLOPE", "Centiloids": "Centiloids",
               "dEBM": "dEBM_upstream", "DAPS": "PL_DAPS"}
    palette = dict(zip(methods, ["#B8E0D2", "#A2D2FF", "#F7C58A", "#FFD1DC"]))
    gaps = pd.concat([pd.DataFrame({"Method": method,
                                   "Violation_Gap": pairs[f"{key}_rank_decline"]})
                      for method, key in methods.items()], ignore_index=True)
    original = gaps.copy(deep=True)
    figure = gap_plot(gaps, list(methods), palette, comparisons)
    try:
        axes = figure.axes[0]
        fliers = [line for line in axes.lines
                  if len(line.get_ydata()) and np.max(line.get_ydata()) >= .7]
        assert len(fliers) == 4
        assert all(line.get_alpha() == 0 for line in fliers)
        np.testing.assert_allclose(sorted(np.max(line.get_ydata()) for line in fliers),
                                   [.7, .8, .9, 1.])
        assert axes.get_ylim()[1] < .07
        whiskers = [line for line in axes.lines
                    if len(line.get_xdata()) == len(line.get_ydata()) == 2
                    and np.ptp(line.get_xdata()) == 0
                    and np.ptp(line.get_ydata()) > 0]
        assert len(whiskers) == 8
        assert all(line.get_visible() and line.get_alpha() != 0 for line in whiskers)
        annotations = [text for text in axes.texts if text.get_text().startswith("p ")]
        assert len(annotations) == 3
        expected_labels = set()
        for row in comparisons:
            p = row["p_value"]
            expected_labels.add("p < 0.0001" if p < .0001 else f"p = {p:.4f}")
        assert {text.get_text() for text in annotations} == expected_labels
        assert all(text.get_ha() == "center" and text.get_fontsize() == 6.5 for text in annotations)
        assert all("*" not in text.get_text() for text in annotations)
        export(figure, "figure2c", tmp_path)
        assert all(text.get_fontsize() == 6.5 for text in annotations)
        pd.testing.assert_frame_equal(gaps, original)
    finally:
        plt.close(figure)
