"""Input validation and frozen-artifact handling for the comparator run."""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["RID", "PHC_Age_PET"]
METHODS = ["PL_DAPS", "Centiloids", "SLOPE", "dEBM_upstream"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def now():
    return datetime.now(timezone.utc).isoformat()


def resolve_work_dir(work_dir):
    """Resolve prepared runtime data without writing into source directories."""
    work_dir = Path(work_dir).expanduser().resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src/pace_abeta").is_dir():
            if work_dir == parent or (parent in work_dir.parents and parent / "runs" not in work_dir.parents):
                raise ValueError("Inside the repository, --work-dir must be a subdirectory of runs/")
            break
    if not work_dir.is_dir():
        raise FileNotFoundError(f"Prepared work directory does not exist: {work_dir}")
    return work_dir


@dataclass
class Inputs:
    work_dir: Path
    features: list
    frames: dict
    matrices: dict
    pace_scores: dict
    hashes: dict


def load_inputs(work_dir):
    work_dir = resolve_work_dir(work_dir)
    prepared = work_dir / "data/prepared"
    metadata = read_json(prepared / "metadata.json")
    features = metadata["features"]
    if len(features) != 68 or len(set(features)) != 68:
        raise ValueError("The fixed benchmark requires 68 distinct regional biomarkers")
    paths = [prepared / "metadata.json", prepared / "features.npz"]
    frames, matrices, pace_scores = {}, {}, {}
    with np.load(prepared / "features.npz", allow_pickle=False) as archive:
        for split in ("train", "test"):
            path = prepared / f"{split}_input.csv"
            columns = KEYS + ["PHC_CENTILOIDS"]
            if split == "train":
                columns += ["PHC_Diagnosis"]
            frame = pd.read_csv(path, usecols=columns, dtype={"RID": str},
                                float_precision="round_trip")
            if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
                raise ValueError(f"Missing or duplicate {split} visit identifiers")
            if not np.isfinite(frame[["PHC_Age_PET", "PHC_CENTILOIDS"]].to_numpy(float)).all():
                raise ValueError(f"Nonfinite {split} visit age or Centiloids")
            matrix = np.asarray(archive[split], dtype=np.float64)
            if matrix.shape != (len(frame), 68) or not np.isfinite(matrix).all():
                raise ValueError(f"Invalid {split} feature matrix")
            index_path = prepared / f"{split}_index.csv"
            if index_path.exists():
                index = pd.read_csv(index_path, dtype={"RID": str}, float_precision="round_trip")
                pd.testing.assert_frame_equal(frame[KEYS], index[KEYS], check_dtype=False)
                paths.append(index_path)
            score_path = work_dir / "data" / ("refit" if split == "train" else "test") / "scores.npz"
            with np.load(score_path, allow_pickle=False) as scores:
                means = np.asarray(scores["stage_mean"], dtype=np.float64).reshape(-1)
            if len(means) != len(frame) or not np.isfinite(means).all():
                raise ValueError(f"Invalid frozen PACE encoder means for {split}")
            frames[split], matrices[split], pace_scores[split] = frame, matrix, means
            paths.extend([path, score_path])
    if set(frames["train"].RID) & set(frames["test"].RID):
        raise ValueError("Training and test participants overlap")
    hashes = {str(path.relative_to(work_dir)): digest(path) for path in paths}
    return Inputs(work_dir, features, frames, matrices, pace_scores, hashes)


def cached_run(folder, input_hashes, implementation):
    """Reuse a completed run only when inputs, implementation and outputs match."""
    folder = Path(folder)
    path = folder / "complete.json"
    if not path.exists():
        if folder.exists() and any(folder.iterdir()):
            raise RuntimeError(f"Incomplete comparator output exists; use a fresh work directory: {folder}")
        return None
    record = read_json(path)
    if record["input_hashes"] != input_hashes or record["implementation"] != implementation:
        raise RuntimeError(f"Frozen comparator inputs or implementation changed: {folder}")
    for relative, expected in record["output_hashes"].items():
        if digest(folder / relative) != expected:
            raise RuntimeError(f"Frozen comparator artifact changed: {folder / relative}")
    return record


def complete_run(folder, inputs, implementation, details):
    folder = Path(folder)
    outputs = sorted(p for p in folder.iterdir() if p.is_file() and p.name != "complete.json")
    record = {"completed_at": now(), "input_hashes": inputs.hashes,
              "implementation": implementation, **details,
              "output_hashes": {p.name: digest(p) for p in outputs}}
    write_json(folder / "complete.json", record)
    return record
