# Pretrained PACE-Aβ inference

The pretrained bundle contains the fitted neural-network parameters and regional covariate-adjustment parameters:

- `models/pace_abeta.pt`: model state, architecture/training configuration, and input dimension.
- `models/preprocessing.json`: ordered features, regression coefficients, covariate-centering means, and covariate-imputation means.

The model returns a posterior mean and variance for each individual PET scan. The reported **PACE-Aβ score** is the posterior mean after participant-weighted rank calibration against a supplied training reference. No participant identifiers, individual scans, individual scores, or empirical rank knots are included in the bundle.

## Score a table

```bash
python scripts/predict.py \
  --input /path/to/new_visits.csv \
  --reference /path/to/authorized_training_reference.csv \
  --output /path/to/pace_scores.csv
```

Use `--model` and `--preprocessing` to select another compatible model and its corresponding adjustment parameters. The reference and input tables contain raw, unadjusted SUVRs in the 68 named columns listed in `configs/features.json`, plus `PHC_Sex` and `PHC_Education`. Column order in the CSV is immaterial; the stored feature list fixes model input order. SUVRs must be finite and observed. Missing sex or education is filled using the pretrained covariate means. PET acquisition and SUVR processing should match the intended use of the model.

The reference additionally requires a nonmissing `RID` for every scan. Numeric and string identifiers are supported; use one consistent identifier encoding within the reference so repeated visits receive the correct participant weight. The input table does not need a participant identifier or visit history for model scoring. If present, `RID`, `PHC_Age_PET`, and `PHC_SCANDATE` are retained in the output.

The command returns:

| Column | Meaning |
| --- | --- |
| `PACE_Abeta_score` | Participant-weighted rank calibration of the posterior mean. |
| `posterior_mean` | Uncalibrated single-scan variational expectation, in latent units. |
| `posterior_variance` | Encoder variance in squared latent units, before rank calibration. |

The variance is not a variance estimate for the rank-calibrated score. Scores outside the reference's observed mean range use monotone linear extrapolation and may fall below 0 or above 1; they are not clipped.

## Choose the reference once

Use an authorized training/reference cohort that represents the intended comparison population. Keep that reference fixed when scoring subsequent cohorts or visits. Each participant receives total weight one, distributed equally across their reference visits. This prevents participants with more scans from dominating calibration.

Reference and target scans use the same pretrained adjustment and model. No encoder, decoder, or adjustment coefficient is refitted during prediction. The reference only defines the output rank scale. Changing the reference changes that scale, so numerical thresholds from a different reference should not be transferred without recalibration. To reproduce published rank scores, use the corresponding original training reference. Do not fit a separate rank map to each test cohort.

Diagnosis labels and visit timing are unnecessary for pretrained inference. Training-time CN covariate adjustment has already been fitted and is part of the bundle; applying these learned parameters is model inference, not new parameter tuning. Input tables should be selected according to the intended analysis before prediction; the prediction command does not impose cohort-specific demographic filters or exclude rows silently.

The reference rank map is constructed locally in memory. Only the requested output table is written. For model fitting, independent preparation, and a fresh parameter search, see [training.md](training.md).

## Posterior means without a reference

The command above requires a reference because it returns the PACE-Aβ score. It also includes the posterior mean and variance in its output. For Python inference that returns only those unscaled quantities, use the model-loading and scoring example in [training.md](training.md#outputs-and-single-scan-scoring); that API does not need a reference. An unscaled posterior mean should not be labeled `PACE_Abeta_score`.
