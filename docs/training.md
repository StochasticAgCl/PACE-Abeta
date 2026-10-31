# Training PACE-Aβ

PACE-Aβ maps one 68-region amyloid PET scan to a continuous score. The encoder estimates a Gaussian latent-stage distribution. Its mean is the unscaled stage estimate; the reported PACE-Aβ score applies the fixed participant-weighted training-reference rank transformation to that mean. Scoring a new visit requires neither other visits from that participant nor visit dates. Training uses each participant's complete observed visit order.

## Input table

Supply a CSV containing joined, **unadjusted** regional PET values and visit-level demographics. The conventional filename is `data/raw/adni_pet.csv`; the command-line path is explicit and may be outside this repository. Obtaining and joining the source PHC tables is a separate data-access step. Do not substitute latest diagnosis for the diagnosis associated with the PET visit.

| Column | Meaning and encoding |
| --- | --- |
| `RID` | Numeric participant identifier, consistent across visits. |
| `PHC_Age_PET` | Age at PET in years; must increase strictly within participant. |
| The 68 columns in [`../configs/features.json`](../configs/features.json) | Dimensionless regional SUVRs, in the specified left-then-right cortical order. |
| `PHC_Diagnosis` | Visit diagnosis: 1 = CN, 2 = MCI, 3 = AD. Baseline must have one of these codes; missing diagnosis at later visits is allowed. Only observed CN visits fit covariate adjustment. |
| `PHC_Sex` | PHC numeric sex coding, 1 = male and 2 = female. Missing values are allowed. |
| `PHC_Education` | Years of education. Missing values are allowed. |
| `PHC_TRACER` | Amyloid tracer; the reference cohort uses `FBP` (florbetapir). |
| `PHC_Race` | PHC race code; the reference cohort uses code 5. |
| `PHC_Ethnicity` | PHC ethnicity code; the reference cohort uses code 2. |

The reference eligibility mask is `PHC_TRACER == 'FBP'`, `PHC_Race == 5`, and `PHC_Ethnicity == 2`, with all 68 SUVRs and PET age observed. After this visit screening, baseline is each participant's earliest remaining PET row, ordered by PET age. Participants whose baseline diagnosis is missing or is not CN, MCI, or AD are excluded with all their visits before splitting or cross-validation; later diagnosis does not replace missing baseline diagnosis. These are cohort-reproduction criteria, not mathematical requirements of the model. Set `cohort_ready` to `true` in a copied configuration only when the supplied table has already undergone the intended cohort selection; this explicitly bypasses the three eligibility-column filters. Complete-feature screening, baseline-diagnosis eligibility, and duplicate-visit checks still apply.

Keep `PHC_SCANDATE` (an ISO date), `PHC_CENTILOIDS`, and any other downstream clinical metadata in the input. All columns are preserved in the prepared `train_input.csv` and `test_input.csv`. These metadata do not enter the encoder. Clinical analysis additionally needs its endpoint-specific longitudinal tables. The historical PHC join matched diagnosis on participant, phase, and visit code, with nearest observed diagnosis within 0.5 years when the exact visit match was unavailable.

`configs/features.json` is the authoritative feature order. The training configuration refers to it through `features_file`, resolved relative to the configuration file. If moving a configuration elsewhere, copy the feature list with it or set that path accordingly.

## Preparation and split

Visits are sorted by participant and PET age. Half of the participants with multiple scans are sampled into the test set using pandas `sample(frac=0.5, random_state=42)`. Every participant with one scan stays in training. The longitudinal training participants are partitioned with shuffled `KFold(random_state=42)`, using 10 folds by default; single-visit participants have fold `-1` and remain in every training fold.

Each regional SUVR is adjusted for sex and education using ordinary least squares fitted on training CN visits. Missing covariates are imputed with the corresponding CN means. Adjustment subtracts the fitted covariate contribution centered at that region's CN covariate mean. There is no age adjustment, feature standardization, or SUVR imputation. Full-training adjustment and each fold's adjustment are fitted separately. Validation and test observations never fit these coefficients.

For the reference cohort, this yields 846 training participants and 1,720 scans, including 456 single-visit participants; 390 test participants and 1,250 scans; 874 cross-validation adjacent pairs and 860 test adjacent pairs. Other eligible inputs may yield different counts. A new input table requires a new work directory.

## Full parameter search

After installing the project, run:

```bash
python scripts/train.py \
  --raw-input /path/to/adni_pet.csv \
  --work-dir /path/to/pace_run \
  --stage all
```

The run directory holds all generated data and models and may be outside the repository. `--config` defaults to `configs/training.json`. Stages can also be executed separately: `prepare`, `search`, `freeze`, `refit`, `test`, and `audit`. Repeating a completed stage verifies its recorded inputs and reuses its completed outputs. An interrupted search resumes its existing study and sampler.

Both `scripts/train.py` and `scripts/run_pipeline.py` accept `--folds` (at least 2) and `--parallel` (at least 1). These override `folds` and `fold_workers` in the configuration for that run; the configuration file itself is not changed. For example, append `--folds 10 --parallel 5` to evaluate the ten folds in two batches of five. The actual worker count is capped by the available CPUs and the fold count. Every batch finishes before the next is submitted; a final shorter batch is allowed. Supply the same overrides to each stage and when resuming. Use a new work directory after changing either setting.

The default search has 128 sequential Optuna attempts, including numerical failures, with 10 validation folds per attempt and at most 10 workers. Each worker uses one CPU thread. Parallelism changes scheduling only; fold count changes the validation split. There are 16 TPE startup trials, no pruning, and no early stopping. Four prespecified initial configurations share β = 0.01, τ = 0.1, learning rate = 0.001, and 300 epochs, with ρ = 0.2, 0.4, 0.6, and 0.8. The ρ search domain is the discrete set {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}, represented as categorical choices. These four trials count toward the 128-attempt budget.

The fixed encoder widths are 64, 32, and 16 with GELU activations and a mirrored decoder. Dropout and weight decay are zero. Search domains are recorded in `configs/training.json`. Selection minimizes the literal product of two pooled CV quantities:

\[
\mathrm{VR}_{\leq}=\frac{\#\{\mu_{\mathrm{later}}\leq\mu_{\mathrm{earlier}}\}}{\#\{\text{all adjacent validation pairs}\}},
\qquad
\mathrm{CVG}_{>}=\operatorname{mean}\{s_{\mathrm{earlier}}-s_{\mathrm{later}}:s_{\mathrm{earlier}}>s_{\mathrm{later}}\},
\qquad
\mathrm{objective}=\mathrm{VR}_{\leq}\times\mathrm{CVG}_{>}.
\]

The first factor counts nonincreasing raw encoder means, including exact ties, as violations. Counting ties discourages score collapse by preventing identical consecutive scores from being treated as successful progression. The second factor is the conditional mean of strictly positive declines in rank-calibrated scores. Each fitted fold model has its own participant-weighted rank map, fitted only to that fold's training scans and applied unchanged to its validation scans, without clipping. Counts and positive-gap totals are pooled across folds before computing the two factors. Fold means and fold products are not averaged.

The configured metric is `violation_ratio_times_conditional_gap`. This product is stored separately from the conventional `rank.all_violation_gap`, which divides the positive-gap sum by all pairs. With ties these two quantities differ; the conventional quantity is retained only as a diagnostic. `selection_metric: "violation_ratio"` selects by VR alone.

A model with constant training encoder means is rejected before rank calibration. If the pooled VR is zero and there are no positive declines, the product is defined as zero. If VR is positive but there are no positive rank-score declines, the conditional mean is undefined. That candidate retains all fold results, is recorded as `objective_undefined`, and receives Optuna `FAIL` status without a numerical penalty; it counts toward the 128 attempts and cannot be selected. An individual fold with no positive declines remains valid when other folds supply a defined pooled conditional mean. Undefined train or test products are reported as `null`. Objective ties are broken by fewer epochs and then lower trial number, without another metric as a secondary objective. Test evaluation happens after selection and one full-training refit.

The optimized loss is

\[
J=\frac{1}{Nd}\sum_i\mathbb E_q\sum_j\|x_{ij}-g_\theta(Z_{ij})\|^2
+\frac{\beta}{N}\sum_i\mathbb E_q[-\log P_\tau(\pi_i\mid Z_i)]
+\frac{\beta}{N}\sum_i\operatorname{KL}(q_i\|\mathcal N(0,K_i)),
\]

where \(d=68\), \(K_i=(1-\rho)I+\rho\mathbf1\mathbf1^T\), and \(P_\tau\) is the complete ascending Plackett–Luce order probability. The same β multiplies the order and KL terms. There is no additional ordering weight or annealing schedule. One Gaussian draw estimates reconstruction and 16 antithetic Gaussian draws estimate the expected order loss; the joint Gaussian KL is analytic.

Each batch contains complete participants. Summed contributions are divided by \(bN/M\), where \(b\) is the number of participants in the batch, \(N\) is the training scan count, and \(M\) is the training participant count. This preserves the visit-normalized objective despite different numbers of scans per participant.

## Refit the selected configuration

For a new refit without hyperparameter search:

```bash
python scripts/train.py \
  --raw-input /path/to/adni_pet.csv \
  --work-dir /path/to/pace_fixed_run \
  --stage refit-selected
```

This uses `selected_model`, the configuration selected by the reference 10-fold search: β = 0.10319626890596051, τ = 0.020366074203530617, ρ = 0.8, learning rate = 0.00002456503433639552, and 1,518 epochs, with seed 42 and 32 participants per batch. The pretrained checkpoint uses this same configuration. Use a separate directory from a parameter search. Its report records `mode: "fixed"`, zero search attempts, and no cross-validation result.

For preparation only, use `--stage prepare`. `--stage smoke` runs a separate 20-epoch configuration over the requested folds and does not select a model.

## Outputs and single-scan scoring

Important files under the work directory are:

- `data/prepared/train_input.csv`, `test_input.csv`: eligible unadjusted visits and clinical metadata, with subject partitions.
- `data/prepared/features.json`: ordered input columns.
- `data/prepared/preprocessing.npz`: learned full-training adjustment coefficients and covariate means.
- `data/prepared/features.npz`: adjusted train/test matrices plus adjustment parameters.
- `data/prepared/folds/fold{k}_preprocessing.npz`: fold-local matrices, masks, and adjustment parameters.
- `data/search/selection.json`: frozen configuration, selection metric, literal VR-times-CVG product, and both raw-count and strict rank-gap CV diagnostics.
- `data/runs/trial{n}/fold{k}/rank_map.json`: the map fitted only to that fold's training scans. Its hash and the score archive hash are recorded in `complete.json`.
- `data/refit/model.pt`: weights, model configuration, and input dimension.
- `data/refit/scores.npz`, `data/test/scores.npz`: `stage_mean`, `stage_logvar`, `stage_variance`, and `rank_stage`.
- `data/refit/rank_map.json`: the full-training map applied unchanged to test scores.
- `reports/test_summary.json`: `train_rank` and `test_rank` report strict rank-score diagnostics; original nonincrease counts and ratios remain in `train` and `test`. The literal products are stored separately as `train_violation_ratio_times_conditional_gap` and `test_violation_ratio_times_conditional_gap`.
- `reports/test_summary.json`, `reports/audit.json`: numerical results and checkpoint/score verification.

A checkpoint alone does not replace preprocessing. Apply the checkpoint's associated training adjustment to new raw scans before scoring. The model API is:

```python
import numpy as np
import torch
from pl_staging.model import ProbabilisticAE, score_model
from pl_staging.preprocessing import apply_preadjust_params_simple

checkpoint = torch.load("/path/to/model.pt", map_location="cpu", weights_only=True)
model = ProbabilisticAE(checkpoint["input_dim"], checkpoint["config"])
model.load_state_dict(checkpoint["model_state_dict"])

with np.load("/path/to/preprocessing.npz", allow_pickle=False) as prep:
    x = apply_preadjust_params_simple(
        raw_suvr, sex_education, prep["betas"], prep["means"],
        cov_impute_means=prep["imputation"],
    )
scores = score_model(model, x)
stage = scores["stage_mean"]
```

`raw_suvr` must follow the stored feature order; `sex_education` has two columns in that order. Scoring uses an evaluation-mode float64 copy of the encoder and reports its variational mean. The reported PACE-Aβ score applies the fixed training-reference rank transformation to this mean. The [prediction command](inference.md) performs that transformation; raw means are sufficient to compute adjacent-order counts because the transformation is strictly increasing.

Reproducing exact weights requires the same prepared matrices, software versions, CPU numerical behavior, seeds, and operation order. The pinned environment and deterministic execution preserve these choices. Input CSV floating-point parsing can affect the last bits; retain source values rather than repeatedly reserializing already parsed numeric tables.
