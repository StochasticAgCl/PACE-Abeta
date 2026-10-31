"""Run calibration, temporal evaluation, and the prespecified memory workflow."""

from pathlib import Path
import json
import subprocess

import numpy as np
import pandas as pd
from matplotlib.cbook import boxplot_stats

from .calibration import apply_rank_map, fit_rank_map
from .demographics import write_demographics
from .metrics import (METHODS, LABELS, THRESHOLDS, adjacent_pairs,
                      summarize_declines, diagnosis_tests, conditional_gap_tests)
from .piecewise import best_breakpoint, describe_piecewise

KEYS = ["RID", "PHC_Age_PET"]
COGNITION_COLUMNS = ["RID", "PHC_Age_Cognition", "PHC_MEM", "PHC_MEM_SE",
                     "PHC_Diagnosis", "PHC_Sex", "PHC_Education"]


def read_frame(path, required=()):
    frame = pd.read_csv(path, dtype={"RID": str}, float_precision="round_trip")
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if frame.RID.isna().any():
        raise ValueError(f"{path}: participant IDs may not be missing")
    return frame


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def r_directory():
    """Locate R programs distributed as package resources."""
    return Path(__file__).resolve().parent


def run_analysis(work_dir, cognition, seed=42, centiloid_max=30., rscript="Rscript"):
    work = Path(work_dir).resolve()
    data = work / "data" / "analysis"
    report = work / "reports" / "analysis"
    data.mkdir(parents=True, exist_ok=True); report.mkdir(parents=True, exist_ok=True)
    cognition = Path(cognition).resolve()
    cog = read_frame(cognition, COGNITION_COLUMNS)
    if cog.dropna(subset=["PHC_Age_Cognition"]).duplicated(["RID", "PHC_Age_Cognition"]).any():
        raise ValueError("Cognition rows must have unique participant/age keys.")
    scores, metadata = {}, {}
    for split in ("train", "test"):
        scores[split] = read_frame(work / "data" / "evaluation" / f"{split}_scores.csv", KEYS + list(METHODS))
        metadata[split] = read_frame(work / "data" / "prepared" / f"{split}_input.csv", KEYS + ["PHC_Diagnosis"])
        if scores[split].duplicated(KEYS).any() or metadata[split].duplicated(KEYS).any():
            raise ValueError("PET participant/age keys must be unique.")
        if not np.isfinite(scores[split][["PHC_Age_PET", *METHODS]].to_numpy(float)).all():
            raise ValueError("Evaluation ages and method scores must be finite.")
        if set(map(tuple, scores[split][KEYS].to_numpy())) != set(map(tuple, metadata[split][KEYS].to_numpy())):
            raise ValueError(f"{split}: prepared metadata and evaluation visit keys differ.")
        if "PHC_CENTILOIDS" in metadata[split]:
            joined = scores[split][KEYS + ["Centiloids"]].merge(
                metadata[split][KEYS + ["PHC_CENTILOIDS"]], on=KEYS, validate="one_to_one")
            if not np.array_equal(joined.Centiloids.to_numpy(), joined.PHC_CENTILOIDS.to_numpy()):
                raise ValueError(f"{split}: Centiloids differ between prepared metadata and evaluation.")
    if set(scores["train"].RID) & set(scores["test"].RID):
        raise ValueError("Training and test participants must be disjoint.")
    demographics = write_demographics(metadata, report)
    rank, maps = {}, {}
    for split in scores:
        rank[split] = scores[split][KEYS].copy()
    for method in METHODS:
        maps[method] = fit_rank_map(scores["train"][method], scores["train"].RID)
        for split in scores:
            rank[split][method] = apply_rank_map(scores[split][method], maps[method])
    write_json(data / "training_rank_maps.json", maps)
    for split in rank:
        rank[split].to_csv(data / f"rank_{split}_scores.csv", index=False)

    a, b = adjacent_pairs(rank["test"])
    pairs = pd.DataFrame({"RID": rank["test"].RID.to_numpy()[a], "earlier_row": a, "later_row": b,
                          "earlier_age": rank["test"].PHC_Age_PET.to_numpy()[a],
                          "later_age": rank["test"].PHC_Age_PET.to_numpy()[b]})
    rows, zero, boxes = [], [], {}
    for method in METHODS:
        decline = rank["test"][method].to_numpy()[a] - rank["test"][method].to_numpy()[b]
        pairs[f"{method}_rank_decline"] = decline
        for threshold in THRESHOLDS:
            rows.append({"method": method, "label": LABELS[method], "threshold": threshold,
                         **summarize_declines(decline, threshold)})
        stats = summarize_declines(decline)
        zero.append({"Method": LABELS[method], "Pairs": len(decline), "Violations": stats["n_violations"],
                     "Violation_Ratio": stats["violation_ratio"], "Conditional_Violation_Gap": stats["conditional_violation_gap"]})
        positive = decline[decline > 0]
        if len(positive):
            box = boxplot_stats(positive, whis=1.5)[0]
            boxes[method] = {key: float(box[key]) for key in ("q1", "med", "q3", "whislo", "whishi")}
            boxes[method].update(n=len(positive), mean=float(positive.mean()), outliers=len(box["fliers"]))
        else:
            boxes[method] = {"n": 0, "mean": None}
    pairs.to_csv(data / "adjacent_pairs.csv", index=False)
    pd.DataFrame(rows).to_csv(data / "threshold_metrics.csv", index=False)
    pd.DataFrame(zero).to_csv(data / "zero_threshold_metrics.csv", index=False)

    diagnosis = rank["test"][KEYS + ["PL_DAPS"]].merge(metadata["test"][KEYS + ["PHC_Diagnosis"]], on=KEYS, validate="one_to_one")
    diagnosis = diagnosis.rename(columns={"PL_DAPS": "Pseudotime_Normalized"})
    diagnosis[~diagnosis.PHC_Diagnosis.isin([1, 2, 3])].to_csv(data / "diagnosis_excluded.csv", index=False)
    diagnosis = diagnosis[diagnosis.PHC_Diagnosis.isin([1, 2, 3])].copy()
    diagnosis.to_csv(data / "diagnosis_visits.csv", index=False)
    groups = {}
    for code, label in [(1, "CN"), (2, "MCI"), (3, "AD")]:
        values = diagnosis.loc[diagnosis.PHC_Diagnosis.eq(code), "Pseudotime_Normalized"]
        groups[label] = {"visits": len(values), "median": float(values.median()) if len(values) else None}
    tests = diagnosis_tests(diagnosis)
    gap_tests = conditional_gap_tests(pairs)
    write_json(report / "diagnosis_tests.json", tests)
    write_json(report / "conditional_gap_tests.json", gap_tests)

    knot, profile, bounds = best_breakpoint(rank["train"].PL_DAPS, scores["train"].Centiloids)
    profile.to_csv(data / "training_breakpoint_profile.csv", index=False)
    piecewise = {}
    for split in scores:
        piecewise[split] = describe_piecewise(rank[split].PL_DAPS, scores[split].Centiloids, knot)
        pd.DataFrame({"RID": rank[split].RID, "Pseudotime_Normalized": rank[split].PL_DAPS,
                      "PHC_CENTILOIDS": scores[split].Centiloids}).to_csv(data / f"piecewise_{split}.csv", index=False)
    write_json(data / "piecewise.json", piecewise)

    pet = scores["test"][KEYS + ["Centiloids"]].merge(metadata["test"][KEYS + ["PHC_Diagnosis"]], on=KEYS, validate="one_to_one")
    pet = pet.rename(columns={"Centiloids": "PHC_CENTILOIDS"})
    pet["Pseudotime_Normalized"] = rank["test"].PL_DAPS.to_numpy()
    pet_path = data / "memory_input_pet.csv"
    pet.to_csv(pet_path, index=False)
    memory = data / "memory"
    memory.mkdir(exist_ok=True)
    log_path = report / "memory_analysis.log"
    try:
        with log_path.open("w") as log:
            subprocess.run([str(rscript), str(r_directory() / "memory_analysis.R"), str(pet_path), str(cognition),
                            str(memory), repr(knot), str(seed), str(centiloid_max)], check=True, stdout=log, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Memory analysis failed; inspect {log_path}") from error
    matched = read_frame(memory / "matched_subjects.csv")
    modeled = read_frame(memory / "dat_lmm.csv")
    mean = matched.groupby("pt_group").baseline_centiloids.mean()
    var = matched.groupby("pt_group").baseline_centiloids.var()
    smd = float((mean["late"] - mean["early"]) / np.sqrt((var["late"] + var["early"]) / 2))
    pace = next(row for row in zero if row["Method"] == "PACE-Abeta")
    comparators = [row for row in zero if row["Method"] != "PACE-Abeta"]
    improvements = {}
    for endpoint in ("Violation_Ratio", "Conditional_Violation_Gap"):
        eligible = [row for row in comparators if row[endpoint] is not None]
        if eligible:
            best = min(eligible, key=lambda row: row[endpoint])
            reduction = (100 * (1 - pace[endpoint] / best[endpoint])
                         if pace[endpoint] is not None and best[endpoint] > 0 else None)
            improvements[endpoint] = {"comparator": best["Method"], "relative_reduction_percent": reduction}
    summary = {
        "schema_version": 1, "seed": seed, "centiloid_max": centiloid_max,
        "cohort": {split: {"participants": int(scores[split].RID.nunique()), "visits": len(scores[split])} for split in scores},
        "demographics": demographics,
        "calibration": "training-only participant-weighted midranks; monotone tails; no test clipping",
        "temporal": zero, "conditional_gap_distributions": boxes,
        "improvement_vs_best_comparator": improvements,
        "diagnosis": {"groups": groups, "cluster_bootstrap_tests": tests,
                      "excluded_visits": len(scores["test"]) - len(diagnosis)},
        "conditional_gap_tests": gap_tests,
        "piecewise": piecewise, "breakpoint_search_bounds": list(map(float, bounds)),
        "memory": {"eligible_people": len(read_frame(memory / "adni_pt_selected.csv")),
                   "matched_people": int(matched.RID.nunique()), "matched_per_group": matched.pt_group.value_counts().to_dict(),
                   "observations": len(modeled), "absolute_centiloid_smd": abs(smd),
                   "slopes": pd.read_csv(memory / "wald_slope_results.csv").to_dict("records"),
                   "run_statistics": json.loads((memory / "run_stats.json").read_text())},
        "inference": "Diagnosis and mean conditional-gap comparisons use subject-cluster bootstrap-t tests. Segment OLS annotations are descriptive. Memory tests are nominal Wald tests conditional on the training knot and realized matching.",
    }
    write_json(report / "summary.json", summary)
    return summary
