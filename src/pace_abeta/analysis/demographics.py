"""Participant-level demographics from the eligible, unadjusted PET rows."""

from pathlib import Path
import json

import numpy as np
import pandas as pd


KEYS = ["RID", "PHC_Age_PET"]
REQUIRED_COLUMNS = KEYS + ["PHC_Diagnosis", "PHC_Sex", "PHC_Education"]
DIAGNOSES = {"CN": 1, "MCI": 2, "AD": 3}
TABLE_GROUPS = ("CN", "MCI", "AD")


def _validate(frame, split):
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{split}: missing demographic columns {sorted(missing)}")
    result = frame[REQUIRED_COLUMNS].copy()
    if result.RID.isna().any() or result.RID.astype(str).str.strip().eq("").any():
        raise ValueError(f"{split}: participant IDs may not be missing or empty")
    result["RID"] = result.RID.astype(str)
    for column in REQUIRED_COLUMNS[1:]:
        result[column] = pd.to_numeric(result[column], errors="raise")
    if not np.isfinite(result.PHC_Age_PET.to_numpy(float)).all():
        raise ValueError(f"{split}: PET ages must be finite")
    if result.duplicated(KEYS).any():
        raise ValueError(f"{split}: PET participant/age keys must be unique")
    for column in ("PHC_Diagnosis", "PHC_Sex", "PHC_Education"):
        if np.isinf(result[column].to_numpy(float)).any():
            raise ValueError(f"{split}: {column} must be finite or missing")
    if not result.PHC_Sex.dropna().isin([1, 2]).all():
        raise ValueError(f"{split}: PHC_Sex must be 1 (male), 2 (female), or missing")
    if result.PHC_Education.dropna().lt(0).any():
        raise ValueError(f"{split}: education years must be nonnegative or missing")
    return result


def _continuous(values):
    observed = values.dropna()
    n = len(observed)
    return {"n": n, "missing": len(values) - n,
            "mean": float(observed.mean()) if n else None,
            "sd": float(observed.std(ddof=1)) if n > 1 else None}


def _summarize(baseline, visits):
    groups = {}
    for label in ("All", *TABLE_GROUPS, "Unclassified"):
        if label == "All":
            subset = baseline
        elif label == "Unclassified":
            subset = baseline.loc[~baseline.PHC_Diagnosis.isin(DIAGNOSES.values())]
        else:
            subset = baseline.loc[baseline.PHC_Diagnosis.eq(DIAGNOSES[label])]
        sizes = visits.loc[visits.RID.isin(subset.RID)].groupby("RID").size()
        groups[label] = {
            "participants": len(subset), "visits": int(sizes.sum()),
            "train": int(subset.partition.eq("train").sum()),
            "test": int(subset.partition.eq("test").sum()),
            "single_visit_participants": int(sizes.eq(1).sum()),
            "multiple_visit_participants": int(sizes.gt(1).sum()),
            "male": int(subset.PHC_Sex.eq(1).sum()),
            "female": int(subset.PHC_Sex.eq(2).sum()),
            "sex_missing": int(subset.PHC_Sex.isna().sum()),
            "age": _continuous(subset.PHC_Age_PET),
            "education": _continuous(subset.PHC_Education),
        }
    return groups


def summarize_demographics(metadata):
    """Summarize each participant's first eligible PET row, without imputation."""
    if set(metadata) != {"train", "test"}:
        raise ValueError("Demographics require both train and test metadata")
    frames = {split: _validate(metadata[split], split) for split in ("train", "test")}
    if set(frames["train"].RID) & set(frames["test"].RID):
        raise ValueError("Training and test participants must be disjoint")
    visits = pd.concat([frame.assign(partition=split) for split, frame in frames.items()],
                       ignore_index=True)
    # Keep the entire earliest row: groupby.first would fill missing baseline fields.
    baseline = visits.sort_values(KEYS).drop_duplicates("RID", keep="first")
    cohorts = {split: _summarize(baseline.loc[baseline.partition.eq(split)], frames[split])
               for split in frames}
    cohorts["pooled"] = _summarize(baseline, visits)
    unknown = baseline.loc[~baseline.PHC_Diagnosis.isin(DIAGNOSES.values())]
    return {
        "schema_version": 1,
        "baseline": "Earliest eligible PET row per participant, ordered by PHC_Age_PET; no backfilling or imputation.",
        "diagnosis_codes": DIAGNOSES,
        "standard_deviation_ddof": 1,
        "continuous_statistics": "Available baseline observations; SD is null when fewer than two observations are available.",
        "cohorts": cohorts,
        "unclassified_baseline_diagnosis": {
            "participants": len(unknown), "missing": int(unknown.PHC_Diagnosis.isna().sum()),
            "other": int(unknown.PHC_Diagnosis.notna().sum()),
            "other_codes": {str(code): int(count) for code, count in
                            unknown.PHC_Diagnosis.dropna().value_counts().sort_index().items()},
        },
    }


def _mean_sd(stats):
    if stats["mean"] is None:
        return "--"
    sd = f'{stats["sd"]:.2f}' if stats["sd"] is not None else "--"
    return rf'{stats["mean"]:.2f}\(\pm\){sd}'


def demographic_table_tex(summary):
    """Return a three-rule tabular and notes for inclusion inside a table float."""
    groups = summary["cohorts"]["pooled"]
    rows = [
        ("Participants", lambda group: f'{group["participants"]:,}'),
        ("Male/female", lambda group: f'{group["male"]:,}/{group["female"]:,}'),
        ("Age (years)", lambda group: _mean_sd(group["age"])),
        ("Education (years)", lambda group: _mean_sd(group["education"])),
    ]
    lines = [r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lccc@{}}",
             r"\noalign{\hrule height 0.8pt\vskip 3pt}",
             r" & CN & MCI & AD \\",
             r"\noalign{\vskip 2pt\hrule height 0.4pt\vskip 3pt}"]
    for label, render in rows:
        lines.append(" & ".join([label, *(render(groups[group]) for group in TABLE_GROUPS)]) + r" \\")
    lines.extend([r"\noalign{\vskip 3pt\hrule height 0.8pt}", r"\end{tabular*}"])
    notes = ["Diagnosis and demographics are taken from each participant's first eligible PET visit.",
             "Continuous values are means with sample standard deviations."]
    unknown = summary["unclassified_baseline_diagnosis"]
    if unknown["participants"]:
        notes.append(f'Not shown: {unknown["participants"]:,} participants with unclassified baseline diagnosis '
                     f'({unknown["missing"]:,} missing; {unknown["other"]:,} other).')
    if groups["All"]["sex_missing"]:
        notes.append(f'Sex is missing for {groups["All"]["sex_missing"]:,} participants.')
    if any(groups[label][key]["n"] == 1 for label in TABLE_GROUPS for key in ("age", "education")):
        notes.append("Standard deviation is unavailable for a single observation.")
    lines.extend([r"\par\smallskip", r"\begin{minipage}{\linewidth}\fontsize{9}{10.45}\selectfont",
                  " ".join(notes), r"\end{minipage}"])
    return "\n".join(lines) + "\n"


def write_demographics(metadata, report_dir):
    """Write aggregate JSON, CSV, and the manuscript table; return the summary."""
    summary = summarize_demographics(metadata)
    destination = Path(report_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "demographics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    rows = []
    for split, groups in summary["cohorts"].items():
        for label, statistics in groups.items():
            row = {"split": split, "diagnosis": label}
            for key, value in statistics.items():
                if isinstance(value, dict):
                    row.update({f"{key}_{field}": number for field, number in value.items()})
                else:
                    row[key] = value
            rows.append(row)
    pd.DataFrame(rows).to_csv(destination / "demographics.csv", index=False)
    (destination / "table1_demographics.tex").write_text(demographic_table_tex(summary))
    return summary
