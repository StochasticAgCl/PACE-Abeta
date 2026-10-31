"""Direct pyebm2.0.3 fitting and original continuous patient staging."""

from contextlib import redirect_stdout
from importlib.metadata import version
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .common import cached_run, complete_run, digest, read_json, write_json


def verify_pyebm():
    if version("pyebm") != "2.0.3":
        raise RuntimeError("The fixed comparator requires pyebm==2.0.3")
    import pyebm
    root = Path(pyebm.__file__).resolve().parent.parent
    provenance = read_json(Path(__file__).parent / "vendor/pyebm_provenance.json")
    for name, expected in provenance["files"].items():
        if digest(root / name) != expected:
            raise RuntimeError(f"Installed pyebm source differs from the official2.0.3 wheel: {name}")
    return provenance


def diagnosis_labels(values):
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    return numeric.map({1.0: "CN", 2.0: "MCI", 3.0: "AD"}).fillna("Unknown")


def classify(matrix, arrays):
    from pyebm.mixture_model import do_classification
    options = SimpleNamespace(MixtureModel="GMMvv2", PatientStaging=["exp", "p"])
    parameters = SimpleNamespace(Control=arrays["control"], Disease=arrays["disease"], Mixing=arrays["mixing"])
    return do_classification.Classify(np.asarray(matrix, dtype=np.float64)[:, :, None], parameters, options)[0]


def continuous_scores(matrix, arrays):
    from pyebm import core_utilities
    probabilities = classify(matrix, arrays)
    return np.asarray(core_utilities.patient_staging(
        arrays["order"].tolist(), arrays["centers"], probabilities, 1 - probabilities, ["exp", "p"]), dtype=float)


def fit_debm(inputs, folder):
    provenance = verify_pyebm()
    implementation = {"name": "pyebm", "version": "2.0.3", "wheel_sha256": provenance["wheel_sha256"],
                      "source_sha256": provenance["files"], "wrapper_sha256": digest(__file__),
                      "seed": 42, "Factors": [], "Labels": ["CN", "MCI", "AD"],
                      "MixtureModel": "GMMvv2", "PatientStaging": ["exp", "p"],
                      "Bootstrap": 0, "runtime_patches": False}
    cached = cached_run(folder, inputs.hashes, implementation)
    if cached:
        with np.load(Path(folder) / "scores.npz", allow_pickle=False) as archive:
            return {split: archive[split].copy() for split in ("train", "test")}, cached
    from pyebm import debm
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "protocol.json", {"implementation": implementation, "input_hashes": inputs.hashes,
               "training_rows": "Every training visit is a cross-sectional row",
               "test_input_to_fit": False, "test_diagnosis_used": False})
    np.random.seed(42)
    random.seed(42)
    data = pd.DataFrame(inputs.matrices["train"], columns=inputs.features)
    labels = diagnosis_labels(inputs.frames["train"].PHC_Diagnosis)
    data.insert(0, "Diagnosis", labels.to_numpy())
    data.insert(0, "PTID", [f"train_visit_{i:04d}" for i in range(len(data))])
    started = time.perf_counter()
    with (folder / "fit.log").open("w") as log, redirect_stdout(log):
        model, outputs, test_outputs = debm.fit(data, Factors=[], Labels=["CN", "MCI", "AD"], DataTest=[])
    if model.BiomarkerList != inputs.features or test_outputs != [[]]:
        raise RuntimeError("Unexpected original dEBM output schema")
    if outputs[0].PTID.tolist() != data.PTID.tolist():
        raise RuntimeError("Original dEBM changed training row order")
    parameters = model.BiomarkerParameters[0]
    arrays = {"control": parameters.Control, "disease": parameters.Disease, "mixing": parameters.Mixing,
              "order": np.asarray(model.CentralOrderings[0], dtype=int),
              "centers": np.asarray(model.EventCenters[0], dtype=float),
              "train_stages": outputs[0].Stages.to_numpy(dtype=float)}
    arrays["train_p_yes"] = classify(inputs.matrices["train"], arrays)
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise FloatingPointError("Original dEBM produced nonfinite model arrays")
    np.savez(folder / "model.npz", **arrays)
    # Test staging uses only its feature matrix and the frozen training estimates.
    scores = {split: continuous_scores(inputs.matrices[split], arrays) for split in ("train", "test")}
    np.testing.assert_array_equal(scores["train"], arrays["train_stages"])
    if not all(np.isfinite(a).all() for a in scores.values()):
        raise FloatingPointError("Original dEBM produced nonfinite scores")
    np.savez(folder / "scores.npz", **scores)
    pd.DataFrame({"position": np.arange(1, 69), "biomarker": np.asarray(inputs.features)[arrays["order"]],
                  "event_center": arrays["centers"]}).to_csv(folder / "event_order.csv", index=False)
    record = complete_run(folder, inputs, implementation,
                          {"fit_seconds": time.perf_counter() - started,
                           "training_label_counts": {str(k): int(v) for k, v in labels.value_counts().items()},
                           "test_diagnosis_used": False, "hyperparameter_search": False})
    return scores, record
