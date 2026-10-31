"""Generate aligned raw-score tables for the four fixed manuscript methods."""

from pathlib import Path

import numpy as np

from .common import KEYS, METHODS, cached_run, complete_run, digest, load_inputs, write_json
from .debm import fit_debm
from .slope import fit_slope


def run_comparison(work_dir):
    inputs = load_inputs(work_dir)
    root = inputs.work_dir
    folder = root / "data/evaluation"
    source_dir = Path(__file__).parent
    implementation = {"pipeline": "fixed four-method comparison", "seed": 42,
                      "source_sha256": {p.name: digest(p) for p in sorted(source_dir.glob("*.py"))},
                      "normalization": "None here: raw encoder means, raw Centiloids, original SLOPE and dEBM scores"}
    # Comparator checks run on resume too, so package/source/input changes cannot hide behind a table cache.
    print("SLOPE: fixed original seed42 benchmark", flush=True)
    slope_scores, slope_record = fit_slope(inputs, root / "data/comparators/slope")
    print("dEBM: original pyebm2.0.3 defaults", flush=True)
    debm_scores, debm_record = fit_debm(inputs, root / "data/comparators/debm")
    implementation["comparators"] = {"SLOPE": slope_record["implementation"], "dEBM": debm_record["implementation"]}
    cached = cached_run(folder, inputs.hashes, implementation)
    if cached:
        return cached
    folder.mkdir(parents=True, exist_ok=True)
    inventory = {}
    for split in ("train", "test"):
        frame = inputs.frames[split][KEYS].copy()
        frame["PL_DAPS"] = inputs.pace_scores[split]
        frame["Centiloids"] = inputs.frames[split].PHC_CENTILOIDS.to_numpy()
        frame["SLOPE"] = slope_scores[split]
        frame["dEBM_upstream"] = debm_scores[split]
        if not np.isfinite(frame[METHODS].to_numpy()).all():
            raise FloatingPointError(f"Nonfinite {split} comparison score")
        frame.to_csv(folder / f"{split}_scores.csv", index=False)
        inventory[split] = {"participants": int(frame.RID.nunique()), "visits": len(frame)}
    write_json(folder / "score_definitions.json", {
        "PL_DAPS": "Frozen single-scan PACE encoder mean; legacy column name preserved for downstream compatibility",
        "Centiloids": "Unchanged PHC_CENTILOIDS",
        "SLOPE": "Original training-fitted UMAP/spline arc-length score with train-diagnosis orientation",
        "dEBM_upstream": "Original pyebm patient_staging(order,centers,p,1-p,['exp','p'])",
        "rank_normalization": "Fit training-only rank maps separately in the downstream analysis step",
    })
    return complete_run(folder, inputs, implementation, {"inventory": inventory, "test_diagnosis_used": False})
