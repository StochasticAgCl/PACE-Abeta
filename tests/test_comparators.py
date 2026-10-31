"""Synthetic checks for comparator interfaces and original-package adapters."""

import json

import numpy as np
import pandas as pd
import pytest

from pace_abeta.comparators.common import load_inputs
from pace_abeta.comparators.debm import continuous_scores, diagnosis_labels, verify_pyebm
from pace_abeta.comparators.slope import verify_original_source
from pace_abeta.comparators import runner


@pytest.fixture
def prepared_work(tmp_path):
    prepared = tmp_path / "data/prepared"
    prepared.mkdir(parents=True)
    features = [f"biomarker_{i}" for i in range(68)]
    (prepared / "metadata.json").write_text(json.dumps({"features": features}))
    matrices = {}
    for split, ids in (("train", ["a", "a", "b", "b"]), ("test", ["c", "c", "d", "d"])):
        frame = pd.DataFrame({"RID": ids, "PHC_Age_PET": [65., 66., 70., 71.],
                              "PHC_CENTILOIDS": [-4., 10., 20., 32.],
                              "PHC_Diagnosis": [1, 2, 3, np.nan] if split == "train"
                              else ["not read"] * 4})
        frame.to_csv(prepared / f"{split}_input.csv", index=False)
        frame[["RID", "PHC_Age_PET"]].to_csv(prepared / f"{split}_index.csv", index=False)
        matrices[split] = np.zeros((4, 68))
        folder = tmp_path / "data" / ("refit" if split == "train" else "test")
        folder.mkdir()
        np.savez(folder / "scores.npz", stage_mean=np.array([-2., -1., 1., 2.]))
    np.savez(prepared / "features.npz", **matrices)
    return tmp_path


def test_input_contract_and_no_test_diagnosis(prepared_work):
    inputs = load_inputs(prepared_work)
    assert "PHC_Diagnosis" not in inputs.frames["test"]
    assert inputs.features == [f"biomarker_{i}" for i in range(68)]
    np.testing.assert_array_equal(inputs.pace_scores["test"], [-2., -1., 1., 2.])


def test_rejects_misaligned_visit_index(prepared_work):
    path = prepared_work / "data/prepared/test_index.csv"
    frame = pd.read_csv(path).iloc[::-1]
    frame.to_csv(path, index=False)
    with pytest.raises(AssertionError):
        load_inputs(prepared_work)


def test_rejects_subject_leakage(prepared_work):
    for name in ("test_input.csv", "test_index.csv"):
        path = prepared_work / "data/prepared" / name
        frame = pd.read_csv(path)
        frame["RID"] = ["a", "a", "d", "d"]
        frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="overlap"):
        load_inputs(prepared_work)


def test_unknown_diagnoses_preserve_original_intermediate_pool():
    assert diagnosis_labels([1., 2., 3., 4., np.nan]).tolist() == ["CN", "MCI", "AD", "Unknown", "Unknown"]


def test_unchanged_source_files_and_original_continuous_staging():
    assert len(verify_original_source()) == 2
    assert verify_pyebm()["version"] == "2.0.3"
    from pyebm import core_utilities
    from pyebm.mixture_model import do_classification
    from types import SimpleNamespace
    matrix = np.array([[0., 0.2, 0.5], [0.7, 1.1, 1.2], [1.8, 2., 1.9]])
    arrays = {"control": np.array([[0., .3]] * 3), "disease": np.array([[2., .4]] * 3),
              "mixing": np.array([.5, .5, .5]), "order": np.array([0, 1, 2]),
              "centers": np.array([.2, .5, .8])}
    options = SimpleNamespace(MixtureModel="GMMvv2", PatientStaging=["exp", "p"])
    parameters = SimpleNamespace(Control=arrays["control"], Disease=arrays["disease"], Mixing=arrays["mixing"])
    p = do_classification.Classify(matrix[:, :, None], parameters, options)[0]
    expected = core_utilities.patient_staging([0, 1, 2], arrays["centers"], p, 1-p, ["exp", "p"])
    np.testing.assert_array_equal(continuous_scores(matrix, arrays), expected)


def test_four_score_output_and_frozen_input_guard(prepared_work, monkeypatch):
    def fixed_scores(inputs, folder):
        return {s: np.arange(len(inputs.frames[s])) / 3 for s in ("train", "test")}, {"implementation": {"synthetic": True}}
    monkeypatch.setattr(runner, "fit_slope", fixed_scores)
    monkeypatch.setattr(runner, "fit_debm", fixed_scores)
    runner.run_comparison(prepared_work)
    output = pd.read_csv(prepared_work / "data/evaluation/test_scores.csv")
    assert output.columns.tolist() == ["RID", "PHC_Age_PET", "PL_DAPS", "Centiloids", "SLOPE", "dEBM_upstream"]
    np.testing.assert_array_equal(output.PL_DAPS, [-2., -1., 1., 2.])
    np.testing.assert_array_equal(output.Centiloids, [-4., 10., 20., 32.])
    runner.run_comparison(prepared_work)
    score_file = prepared_work / "data/test/scores.npz"
    np.savez(score_file, stage_mean=np.ones(4))
    with pytest.raises(RuntimeError, match="changed"):
        runner.run_comparison(prepared_work)
