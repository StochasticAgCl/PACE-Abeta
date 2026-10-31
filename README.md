# PACE-Aβ

Progression-Aware Continuous Embedding of Amyloid PET.

This repository contains the preprocessing, model training, cross-validation,
comparator fitting, longitudinal analyses, and result-figure code for
**Progression-Aware Variational Learning for Continuous Alzheimer Staging**.

## Data access

Participant-level data are not distributed with this repository. Obtain access
to the relevant ADNI and ADSP-PHC data releases through their authorized
repositories and comply with the applicable data-use agreements:

- [ADNI data access](https://adni.loni.usc.edu/data-samples/adni-data/)
- [ADSP-PHC resources](https://adsp.niagads.org/funded-programs/phenotype-harmonization/)

These controlled-access participant records and participant-level derivatives
must not be uploaded to a public repository. Each user must obtain authorized
access and prepare the inputs locally.

Create `data/raw/` and place your harmonized regional PET and cognition tables there, named
`adni_pet.csv` and `adni_cognition.csv`. Data and run directories are ignored by
Git. Alternative input and output directories can be supplied to the commands.

The inputs are regional SUVR and harmonized clinical tables, not raw PET images.
Prepare the joins under your data-use agreement before running the pipeline.
Use visit-specific diagnosis; do not fill earlier visits with the participant's
latest diagnosis. The reference PHC join used participant, phase, and visit code,
then the nearest diagnosis within 0.5 years when an exact match was unavailable.

### PET table: `data/raw/adni_pet.csv`

One row represents one PET visit. Use a header row and the following columns:

| Column | Required contents |
| --- | --- |
| `RID` | Numeric participant identifier, shared with the cognition table. |
| `PHC_Age_PET` | Age at PET in years; unique and strictly increasing across a participant's visits. |
| All 68 names in [`configs/features.json`](configs/features.json) | Unadjusted regional SUVRs. The file specifies the exact names and feature order. |
| `PHC_CENTILOIDS` | Finite global Centiloid value for each eligible scan. |
| `PHC_Diagnosis` | Visit diagnosis: 1 = CN, 2 = MCI, 3 = AD. The first eligible PET visit must have one of these codes; missing values at later visits are allowed. |
| `PHC_Sex` | 1 = male, 2 = female. Missing values are allowed and handled by training-only imputation. |
| `PHC_Education` | Years of education; missing values are allowed. |
| `PHC_TRACER` | Tracer string; the manuscript cohort uses `FBP`. |
| `PHC_Race` | PHC race code; the manuscript cohort uses 5. |
| `PHC_Ethnicity` | PHC ethnicity code; the manuscript cohort uses 2. |

`PHC_SCANDATE` may be retained as an ISO date but is not used to fit the model.
The default configuration selects the tracer/race/ethnicity values above and
requires observed PET age and all 68 SUVRs. It performs covariate adjustment
itself; do not supply already adjusted or standardized SUVRs. Participants whose
first otherwise eligible PET visit has missing or other diagnosis are excluded
with all their visits before the participant split and cross-validation. Training CN
observations must be available to estimate that adjustment. Further input and
split details are in [the training guide](docs/training.md).

### Cognition table: `data/raw/adni_cognition.csv`

One row represents one cognitive visit. PET and cognition need not share dates.

| Column | Required contents |
| --- | --- |
| `RID` | The same participant coding as the PET table. |
| `PHC_Age_Cognition` | Age at cognitive assessment in years; unique within participant. |
| `PHC_MEM` | Harmonized memory composite score. |
| `PHC_MEM_SE` | Positive measurement standard error of the memory score. |
| `PHC_Diagnosis` | Visit diagnosis: 1 = CN, 2 = MCI, 3 = AD. |
| `PHC_Sex` | 1 / `M` / `male`, or 2 / `F` / `female`. |
| `PHC_Education` | Years of education. |

Rows without valid age, memory score, or positive standard error are excluded
from the memory analysis. The table must already contain visit-level diagnosis
and demographic fields. See [the analysis guide](docs/analysis.md) for anchoring,
matching, and the mixed-model specification.

## Dependencies

Use Python 3.12.13 and R 4.5.1. Install the dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
Rscript scripts/install_r_dependencies.R
```

Python package versions are pinned in `requirements.txt`; R package versions
are pinned in `renv.lock`. The model and cross-validation
run on CPU; the full search uses at most 10 single-threaded fold workers
for its ten folds, limited by the available CPU count.
A working C/C++/Fortran toolchain is needed when R installs dependencies from source.

Use an R installation with Cairo graphics support for PDF and PNG export. Arial
is used when available, with DejaVu Sans as the Python fallback; R uses the
system sans-serif family. Fonts affect rendered images, not numerical results.

## Reproduce the experiments

### Full search and all paper analyses

After preparing the two input files, run from the repository root:

```bash
python scripts/run_pipeline.py
```

This command performs, in order:

1. Cohort selection, the seed-42 participant split, and training-only covariate
   adjustment, including separate adjustment inside every validation fold.
2. All 128 parameter-search attempts with 10 validation folds per attempt. Encoder
   architecture is fixed; the search varies β, τ, ρ, learning rate, and epochs.
3. Selection by the pooled nonincrease violation ratio (including exact ties)
   multiplied by the conditional mean of positive rank-score declines. Each
   fold's rank map is fitted only on its training visits. Then one full-training
   refit and test scoring are performed.
4. Fixed SLOPE fitting, official dEBM fitting, and the Centiloid comparison.
5. Baseline demographic summaries of all analyzed participants,
   training-reference rank normalization for each method, temporal metrics at
   thresholds 0, 0.025, 0.05, 0.075, and 0.1, diagnosis comparisons, and piecewise
   fits with the training-selected breakpoint applied unchanged to testing.
6. The ADNI test memory analysis, using baseline CN/MCI and Centiloids ≤30,
   Centiloid matching, and the original known-standard-error mixed model.
7. All individual panels and assembled result figures for Figures 2–3.

Use `--folds` to set the cross-validation fold count and `--parallel` to set the
maximum number of folds evaluated simultaneously. For example, this retains
the ten-fold design while running five folds at a time:

```bash
python scripts/run_pipeline.py --folds 10 --parallel 5
```

Both options default to `configs/training.json` (`folds: 10`, `fold_workers: 12`).
The effective worker count cannot exceed the available CPUs or the fold count.
Folds run in consecutive batches; each batch finishes before the next starts.
Changing only parallelism preserves the fold assignments, seeds, and selection
rule. Changing the fold count changes the validation experiment and can change
the selected model. Supply the same overrides when resuming a run, and use a
new work directory when changing either setting.

Outputs default to `runs/paper/`. To keep inputs and generated results outside
the source checkout:

```bash
python scripts/run_pipeline.py \
  --pet /path/to/adni_pet.csv \
  --cognition /path/to/adni_cognition.csv \
  --work-dir /path/to/pace_run \
  --rscript /path/to/Rscript
```

Rerun the same command to resume an interrupted training search. Input and source
hashes are checked before completed training or comparator artifacts are reused.
An interrupted comparator fit must restart: move only the incomplete folder
named by the error to a backup location, then rerun the comparison stage.
Completed training and comparator fits can remain in place. Use a new work
directory after changing inputs or experimental settings. To run one downstream
stage, use `--stage comparison`, `--stage analysis`, or `--stage figures` with
the existing work directory.

### Refit the selected configuration

To retrain with the selected parameters and generate all results without
repeating parameter selection:

```bash
python scripts/run_pipeline.py --mode refit --work-dir runs/refit
```

This mode uses the configuration selected by the reference search (trial 125,
zero-based): β = 0.10319626890596051, τ = 0.020366074203530617, ρ = 0.8,
learning rate = 0.00002456503433639552, and 1,518 epochs. These parameters are
stored in `selected_model` in `configs/training.json`. This mode records a
fixed-configuration run with zero search attempts. Full search and fixed refit
must use separate work directories.

### Apply the pretrained model

The `models/` directory contains the PACE-Aβ checkpoint and its fitted
preprocessing parameters. Supply new raw PET rows and an authorized training
reference to obtain the rank-normalized PACE-Aβ score:

```bash
python scripts/predict.py \
  --input /path/to/new_pet.csv \
  --reference /path/to/training_reference.csv \
  --output /path/to/pace_scores.csv
```

The reference must contain training participants only; do not use the new
evaluation cohort to fit the rank map. The original training reference is not
distributed. Exact reproduction of the manuscript's calibrated scores requires
that same reference. See [the inference guide](docs/inference.md) for input
requirements and the interpretation of the raw model outputs.

## Generated results

All paths below are relative to the selected work directory:

| Path | Contents |
| --- | --- |
| `data/prepared/` | Eligible visits, participant split, adjusted features, and fold-specific preprocessing. |
| `data/search/` | Trial configurations, fold results, search state, and frozen selection. |
| `data/refit/`, `data/test/` | Fitted PACE checkpoint, raw encoder outputs, calibrated scores, and the training rank map. |
| `data/comparators/` | Fitted SLOPE and dEBM objects and provenance records. |
| `data/evaluation/` | Four-method raw training and test scores. |
| `data/analysis/` | Training rank maps, calibrated scores, temporal metrics, diagnosis and piecewise results. |
| `data/analysis/memory/` | Selected and matched samples, mixed models, contrasts, and predicted trajectories. |
| `reports/test_summary.json` | Frozen PACE test evaluation. |
| `reports/analysis/summary.json` | Numerical summaries used in the manuscript. |
| `reports/analysis/demographics.json`, `demographics.csv` | Participant demographics by split and first-eligible-visit diagnosis. |
| `reports/analysis/table1_demographics.tex` | Three-rule demographic table and notes, generated from selected PET inputs. |
| `figures/` | Figures 2–3 and their individual panels in PDF and 600 dpi PNG. Use the PNG files in the manuscript. |

Figure 1 is the supplied model diagram; it is not an experimental output and is
not included. Result figures are generated entirely by Python and R. No external
image assets are needed.

## Verification and data handling

Run the synthetic unit tests with:

```bash
python -B -m pytest -q -p no:cacheprovider
```

The source distribution contains no participant-level inputs, fitted comparator
objects, per-visit scores, or generated result figures. It includes only PACE
weights and learned preprocessing parameters. Generated tables, plots, rank maps,
checkpoints containing participant objects, and logs remain local to each run
and must be handled under the relevant data-use agreements.

Before sharing a copy of the source, remove local data, run outputs, and caches,
then run:

```bash
python -B scripts/check_release.py
```

The release check inspects ignored files as well as tracked files. `.gitignore`
alone does not remove private content from a checkout or from earlier commits.
Reference-cohort verification results are documented in
[`docs/validation.md`](docs/validation.md). Different data releases, participant
sets, or numerical libraries can produce different fitted values.

## Repository contents

- `src/pl_staging/`: preprocessing, probabilistic model, and training.
- `src/pace_abeta/comparators/`: SLOPE, Centiloids, and dEBM comparison.
- `src/pace_abeta/analysis/`: score calibration, statistical analysis, and plots.
- `configs/`: feature order and experimental settings.
- `scripts/`: command-line entry points.
- `tests/`: model, data-processing, and analysis checks.

The SLOPE implementation is supplied by its authors and retained without
algorithm changes. dEBM uses the published `pyebm==2.0.3` package. See
`docs/comparators.md` for source attribution and package licensing.
