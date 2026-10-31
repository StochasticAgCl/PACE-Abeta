"""Baseline-row selection, participant counts, and demographic summaries."""

import json

import numpy as np
import pandas as pd
import pytest

from pace_abeta.analysis.demographics import summarize_demographics, write_demographics


def visits(rows):
    return pd.DataFrame(rows, columns=["RID", "PHC_Age_PET", "PHC_Diagnosis",
                                      "PHC_Sex", "PHC_Education"])


def cohort():
    return {
        "train": visits([("a", 70., 1., 1., 12.), ("a", 72., 2., 1., 12.),
                         ("b", 74., 2., 2., 16.)]),
        "test": visits([("c", 76., 3., 1., 18.), ("c", 77., 3., 1., 18.)]),
    }


def test_earliest_row_is_not_backfilled_and_later_diagnosis_does_not_reclassify():
    metadata = cohort()
    metadata["train"] = visits([
        ("a", 72., 2., 1., 12.), ("a", 70., np.nan, np.nan, np.nan),
        ("b", 74., 2., 2., 16.), ("b", 75., 3., 2., 16.),
    ])
    summary = summarize_demographics(metadata)
    groups = summary["cohorts"]["pooled"]
    assert groups["All"]["participants"] == 3
    assert groups["All"]["education"]["n"] == 2
    assert groups["All"]["sex_missing"] == 1
    assert groups["CN"]["participants"] == 0
    assert groups["MCI"]["participants"] == 1
    assert groups["AD"]["participants"] == 1
    assert groups["Unclassified"]["age"]["mean"] == 70.
    assert groups["Unclassified"]["education"]["mean"] is None
    assert summary["unclassified_baseline_diagnosis"]["missing"] == 1


def test_split_counts_and_sample_standard_deviation_use_one_row_per_person():
    summary = summarize_demographics(cohort())
    pooled = summary["cohorts"]["pooled"]["All"]
    assert pooled["participants"] == pooled["train"] + pooled["test"] == 3
    assert pooled["visits"] == 5
    assert pooled["single_visit_participants"] == 1
    assert pooled["multiple_visit_participants"] == 2
    assert pooled["age"]["mean"] == pytest.approx(np.mean([70., 74., 76.]))
    assert pooled["age"]["sd"] == pytest.approx(np.std([70., 74., 76.], ddof=1))
    assert pooled["education"]["sd"] == pytest.approx(np.std([12., 16., 18.], ddof=1))
    assert summary["cohorts"]["train"]["AD"]["age"]["mean"] is None
    assert summary["cohorts"]["test"]["AD"]["age"]["sd"] is None


def test_all_retains_all_five_unclassified_baseline_diagnoses():
    metadata = cohort()
    metadata["train"] = pd.concat([
        metadata["train"],
        visits([(f"unknown-{index}", 65. + index, diagnosis, 1., 16.)
                for index, diagnosis in enumerate([np.nan, np.nan, np.nan, 4., 4.])]),
    ], ignore_index=True)
    summary = summarize_demographics(metadata)
    groups = summary["cohorts"]["pooled"]
    assert groups["All"]["participants"] == 8
    assert sum(groups[label]["participants"] for label in ("CN", "MCI", "AD", "Unclassified")) == 8
    assert summary["unclassified_baseline_diagnosis"] == {
        "participants": 5, "missing": 3, "other": 2, "other_codes": {"4.0": 2}}


@pytest.mark.parametrize("column,value,error", [
    ("PHC_Age_PET", np.nan, "ages must be finite"),
    ("PHC_Age_PET", np.inf, "ages must be finite"),
    ("PHC_Sex", 0., "PHC_Sex must be"),
    ("PHC_Education", -1., "nonnegative"),
    ("PHC_Education", np.inf, "finite or missing"),
    ("PHC_Diagnosis", np.inf, "finite or missing"),
    ("RID", None, "IDs may not be missing"),
])
def test_invalid_values_fail_explicitly(column, value, error):
    metadata = cohort()
    metadata["train"].loc[0, column] = value
    with pytest.raises(ValueError, match=error):
        summarize_demographics(metadata)


def test_duplicate_keys_overlapping_people_and_missing_columns_fail():
    metadata = cohort()
    metadata["train"] = pd.concat([metadata["train"], metadata["train"].iloc[[0]]])
    with pytest.raises(ValueError, match="keys must be unique"):
        summarize_demographics(metadata)
    metadata = cohort()
    metadata["test"]["RID"] = "a"
    with pytest.raises(ValueError, match="must be disjoint"):
        summarize_demographics(metadata)
    metadata = cohort()
    metadata["test"] = metadata["test"].drop(columns="PHC_Education")
    with pytest.raises(ValueError, match="missing demographic columns"):
        summarize_demographics(metadata)


def test_aggregate_outputs_include_counts_and_observed_value_notes(tmp_path):
    metadata = cohort()
    metadata["train"].loc[0, "PHC_Education"] = np.nan
    summary = write_demographics(metadata, tmp_path)
    assert json.loads((tmp_path / "demographics.json").read_text()) == summary
    csv = pd.read_csv(tmp_path / "demographics.csv")
    assert set(csv.split) == {"train", "test", "pooled"}
    assert set(csv.diagnosis) == {"All", "CN", "MCI", "AD", "Unclassified"}
    assert "RID" not in csv.columns
    assert csv.loc[csv.split.eq("pooled") & csv.diagnosis.eq("All"), "education_n"].item() == 2
    tex = (tmp_path / "table1_demographics.tex").read_text()
    assert " & CN & MCI & AD" in tex
    assert " & All &" not in tex
    assert r"@{\extracolsep{\fill}}lccc@{}" in tex
    assert "Training/test" not in tex
    assert "Education is available" not in tex
    assert "Standard deviation is unavailable for a single observation" in tex
    assert r"73.33\(\pm\)3.06" not in tex
    assert r"Age (years) & 70.00\(\pm\)-- & 74.00\(\pm\)-- & 76.00\(\pm\)--" in tex
    assert tex.count(r"\hrule height 0.8pt") == 2
    assert tex.count(r"\hrule height 0.4pt") == 1
    assert r"\begin{table}" not in tex
