"""Fixed seed42 wrapper around the unchanged group-supplied SLOPE code."""

from contextlib import redirect_stdout
from importlib.metadata import version
import os
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import torch

from .common import cached_run, complete_run, digest, read_json, write_json

CONFIG = {
    "dimensions": [68, 570, 23, 570, 68], "activation": "GELU", "seed": 42,
    "epochs": 850, "learning_rate": 8.325158565947957e-6,
    "lambda_reconstruction": 0.42939811886786894, "lambda_directional": 1.0,
    "reconstruction_loss": "SmoothL1Loss", "optimizer": "Adam", "device": "cpu",
    "threads": 4, "umap_components": 2, "umap_random_state": 42,
    "umap_transform_seed": 42, "spline_train_k": 30, "spline_smooth": 0.5,
    "spline_n_eval": 2000,
}


def verify_original_source():
    vendor = Path(__file__).parent / "vendor"
    provenance = read_json(vendor / "slope_provenance.json")
    for name, expected in provenance["files"].items():
        if digest(vendor / "slope" / name) != expected:
            raise RuntimeError(f"Group-supplied SLOPE source was modified: {name}")
    return provenance["files"]


def fit_slope(inputs, folder):
    source_hashes = verify_original_source()
    implementation = {"name": "SLOPE group original seed42", "source_sha256": source_hashes,
                      "wrapper_sha256": digest(__file__), "config": CONFIG,
                      "versions": {p: version(p) for p in
                                   ("torch", "numba", "umap-learn", "numpy", "pandas", "scipy", "scikit-learn", "networkx")}}
    cached = cached_run(folder, inputs.hashes, implementation)
    if cached:
        with np.load(Path(folder) / "scores.npz", allow_pickle=False) as archive:
            return {split: archive[split].copy() for split in ("train", "test")}, cached
    # UMAP restores the NumBa thread count after fitting. Its OpenMP runtime is
    # shared with PyTorch, so both must match the original four-thread wrapper.
    for variable, relative in (("NUMBA_CACHE_DIR", "numba"), ("MPLCONFIGDIR", "matplotlib")):
        cache = inputs.work_dir / "cache" / relative
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(variable, str(cache))
    os.environ.setdefault("NUMBA_NUM_THREADS", str(CONFIG["threads"]))
    import numba
    numba.set_num_threads(CONFIG["threads"])
    torch.set_num_threads(CONFIG["threads"])
    # Imports preserve the original files, including their seed42 defaults.
    from .vendor.slope.Train import AE, set_global_seed, train_AE, test_AE
    from .vendor.slope.spline import compute_spline_pseudotime

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "protocol.json", {"implementation": implementation,
               "training_diagnosis_usage": "Train-only spline direction, after unsupervised representation fitting",
               "test_diagnosis_used": False, "input_hashes": inputs.hashes})
    torch.set_num_threads(CONFIG["threads"])
    set_global_seed(42)
    model = AE(dims=CONFIG["dimensions"], activation=torch.nn.GELU).cpu()
    matrices = {s: torch.tensor(inputs.matrices[s], dtype=torch.float32).contiguous()
                for s in ("train", "test")}
    identifiers = {s: inputs.frames[s][["RID", "PHC_Age_PET"]].reset_index(drop=True)
                   for s in ("train", "test")}
    started = time.perf_counter()
    with (folder / "fit.log").open("w") as log, redirect_stdout(log):
        model, projection, train_umap, train_latent, history = train_AE(
            model, matrices["train"], matrices["test"], identifiers["train"], identifiers["test"],
            torch.nn.SmoothL1Loss(), torch.optim.Adam,
            lr=CONFIG["learning_rate"], epochs=CONFIG["epochs"],
            lambda_reconstruction=CONFIG["lambda_reconstruction"],
            lambda_directional=CONFIG["lambda_directional"],
            directional_loss_enabled=True, return_losses=True)
        if projection.random_state != 42 or projection.transform_seed != 42:
            raise RuntimeError("Original SLOPE UMAP defaults no longer reproduce seed42")
        if torch.get_num_threads() != CONFIG["threads"] or numba.get_num_threads() != CONFIG["threads"]:
            raise RuntimeError("SLOPE thread context changed during UMAP fitting")
        test_umap, test_latent = test_AE(model, matrices["test"], identifiers["test"], projection)
        train_score, spline, flipped = compute_spline_pseudotime(
            train_umap[["UMAP1", "UMAP2"]].to_numpy(),
            ref_vector=inputs.frames["train"].PHC_Diagnosis.to_numpy(float),
            k=CONFIG["spline_train_k"], smooth=CONFIG["spline_smooth"], n_eval=CONFIG["spline_n_eval"])
        test_score, _, _ = compute_spline_pseudotime(
            test_umap[["UMAP1", "UMAP2"]].to_numpy(), k=10,
            smooth=CONFIG["spline_smooth"], n_eval=CONFIG["spline_n_eval"], saved_tck=spline)
    if flipped:
        test_score = 1 - test_score
    scores = {"train": np.asarray(train_score), "test": np.asarray(test_score)}
    if not all(np.isfinite(a).all() for a in scores.values()) or np.ptp(scores["train"]) <= 0:
        raise FloatingPointError("Original SLOPE produced invalid scores")
    np.savez(folder / "scores.npz", **scores)
    np.savez(folder / "latent.npz", train=train_latent, test=test_latent)
    train_umap.to_csv(folder / "train_umap.csv", index=False)
    test_umap.to_csv(folder / "test_umap.csv", index=False)
    pd.DataFrame(history).to_csv(folder / "losses.csv", index=False)
    torch.save(model.state_dict(), folder / "model.pt")
    joblib.dump({"umap": projection, "spline": spline, "flipped": bool(flipped)}, folder / "projection.joblib")
    record = complete_run(folder, inputs, implementation,
                          {"fit_seconds": time.perf_counter() - started,
                           "orientation_flipped": bool(flipped), "hyperparameter_search": False,
                           "test_diagnosis_used": False,
                           "original_training_embedding_behavior_preserved": True})
    return scores, record
