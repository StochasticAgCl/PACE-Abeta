import numpy as np
import pandas as pd
import pytest

from pace_abeta.analysis.calibration import apply_rank_map, fit_rank_map
from pace_abeta.analysis.metrics import adjacent_pairs, summarize_declines
from pace_abeta.analysis.piecewise import best_breakpoint, describe_piecewise


def test_each_participant_has_equal_total_mass():
    mapping = fit_rank_map([0., 0., 0., 2.], ["a", "a", "a", "b"])
    np.testing.assert_allclose(mapping["quantile"], [.25, .75])
    np.testing.assert_allclose(apply_rank_map([0., 1., 2.], mapping), [0., .5, 1.])


def test_fixed_mapping_preserves_ties_and_extrapolates_without_test_refit():
    mapping = fit_rank_map([0., 1., 1., 3.], [1, 1, 2, 3])
    test = np.array([-2., 0., 1., 1., 3., 7.])
    calibrated = apply_rank_map(test, mapping)
    assert calibrated[0] < 0 and calibrated[-1] > 1
    np.testing.assert_array_equal(np.sign(np.diff(test)), np.sign(np.diff(calibrated)))
    np.testing.assert_array_equal(apply_rank_map(np.r_[test, -1e5, 1e5], mapping)[:len(test)], calibrated)


def test_threshold_boundary_and_conditional_denominator():
    stats = summarize_declines(np.array([.125, .25, 0., -.25]), .125)
    assert stats["n_violations"] == 1
    assert stats["violation_ratio"] == .25
    assert stats["conditional_violation_gap"] == .25
    assert stats["all_violation_gap"] == .0625
    assert summarize_declines([-1., 0.])["conditional_violation_gap"] is None


def test_adjacent_pairs_are_within_subject_and_chronological():
    frame = pd.DataFrame({"RID": ["a", "b", "a", "a", "b"], "PHC_Age_PET": [72., 65., 70., 73., 66.]})
    a, b = adjacent_pairs(frame)
    assert set(zip(a, b)) == {(2, 0), (0, 3), (1, 4)}
    with pytest.raises(ValueError, match="ambiguous"):
        adjacent_pairs(pd.concat([frame, frame.iloc[[0]]], ignore_index=True))


def test_minimum_sse_knot_and_fixed_test_knot():
    x = np.linspace(0., 1., 201)
    y = 2. + .2*x + 12.*np.maximum(x-.31, 0)
    knot, _, _ = best_breakpoint(x, y)
    assert abs(knot-.31) < 1e-6
    test = describe_piecewise(x, 5.+3*x+8*np.maximum(x-knot, 0), knot)
    assert test["knot"] == knot
    np.testing.assert_allclose(test["coefficients"], [5., 3., 8.], atol=1e-9)


def test_collapsed_training_score_is_not_silently_calibrated():
    with pytest.raises(ValueError, match="distinct"):
        fit_rank_map([1., 1.], [1, 2])
