# Fixed manuscript comparators

Run the comparator stage after preparing data and producing frozen PACE encoder
means:

```bash
python scripts/compare.py --work-dir runs/paper
```

An external work directory is also supported. Participant data, fitted objects,
and generated tables belong in the work directory and are not distributed with
the source repository. The comparator stage does not tune any method or change
the PACE checkpoint.

## Input and output contract

The work directory must contain:

- `data/prepared/train_input.csv` and `test_input.csv`, with `RID`,
  `PHC_Age_PET`, and `PHC_CENTILOIDS`; training also requires numeric
  `PHC_Diagnosis` codes 1=CN, 2=MCI, and 3=AD.
- `data/prepared/metadata.json` with the ordered list `features` of 68 biomarkers.
- `data/prepared/features.npz`, with adjusted numeric matrices `train` and `test`
  in the same row order as the input tables.
- `data/refit/scores.npz` and `data/test/scores.npz`, each containing the frozen
  single-scan encoder means under `stage_mean`.

If prepared visit-index files exist, their identifiers and ages must match the
input rows exactly. Training and test participants must be disjoint. The test
diagnosis column is not read by this stage.

Outputs are `data/evaluation/train_scores.csv` and `test_scores.csv`, with
columns `RID`, `PHC_Age_PET`, `PL_DAPS`, `Centiloids`, `SLOPE`, and
`dEBM_upstream`. `PL_DAPS` is a retained internal column name for the raw PACE
encoder mean. This stage preserves raw scores; the downstream analysis fits
each method's rank map on training participants only.

Source, input, dependency and output hashes accompany completed fits. A rerun
reuses artifacts only when these match; incomplete or changed runs are not
silently overwritten. Comparator outputs under `data/comparators/` include
private fitted objects and may retain participant embeddings. They must stay
out of the distributed repository.

## SLOPE

The two files under `pace_abeta/comparators/vendor/slope/` are exact copies of
the group-supplied original `Train.py` and `spline.py`. Their bytes and original
seed42 defaults are checked against `vendor/slope_provenance.json` at runtime.
The wrapper does not replace the loss, training loop, UMAP operation, spline
construction or scoring functions. The group-supplied files contain no separate
license notice. Their source attribution and hashes are recorded in
`vendor/slope_provenance.json`.

The fixed benchmark settings are the settings used in the manuscript comparison,
not every function-signature default: dimensions `[68,570,23,570,68]`, GELU,
850 epochs, Adam with learning rate `8.325158565947957e-6`, SmoothL1
reconstruction weight `0.42939811886786894`, and directional weight 1.
The wrapper uses CPU with four PyTorch and NumBa threads, initialization/training seed42,
two-dimensional UMAP with fit/transform seed42, and spline parameters
`k=30`, `smooth=0.5`, `n_eval=2000`. Test projection reuses the fitted UMAP and
spline. No hyperparameter or epoch selection is performed.

Training diagnosis codes orient the spline once, using the original correlation
rule. The same orientation is applied to testing. The original `train_AE`
function receives test features and visit identifiers to print post-training
diagnostics; those diagnostics do not update weights or choose a model. The
original cached training embedding from before the final optimizer step is
retained when fitting UMAP. Test encoding uses the returned model. This inherited
behavior is preserved rather than silently repaired.

## Discriminative event-based model

The dependency is pinned to [`pyebm==2.0.3`](https://pypi.org/project/pyebm/2.0.3/).
The official wheel SHA256 is
`246e1bb93db4ad0ed0c701f3e241b9c7ceda112599acff1c86291abaf6cb8df4`.
The wrapper checks the installed package's Python sources against the hashes
from that wheel before use. The wheel's license classifier is GPLv3+, and its
included license text is GNU GPL Version3. pyebm is an external dependency;
its algorithm source is not copied or patched here. See the
[author repository](https://github.com/88vikram/pyebm) for its documentation.

The wrapper directly calls:

```python
model, train_outputs, _ = pyebm.debm.fit(
    training_rows, Factors=[], Labels=["CN", "MCI", "AD"], DataTest=[]
)
```

The method uses the original defaults `GMMvv2`, no bootstrap, and
`PatientStaging=['exp','p']`. `Factors=[]` is an explicit input choice that
avoids a second covariate correction after the shared training-CN adjustment.
Every training visit is treated as a cross-sectional row, including repeated
visits from the same participant. Unknown diagnoses retain the original
package's intermediate-pool handling. All 68 regional SUVRs are biomarkers.

Test staging passes only the feature matrix through original
`do_classification.Classify` and `core_utilities.patient_staging`. The continuous
score is the posterior expectation of learned event centers. Test diagnoses,
test outcomes, and test longitudinal order are not used to fit the mixtures or
event sequence. The wrapper contains no substitute ordering routine, stopping
rule or alternative score formula.

## Centiloids and PACE

Centiloids are copied unchanged from `PHC_CENTILOIDS`. PACE is the frozen encoder
mean already produced by the training stage. Neither comparator fitting nor
this export applies a new transformation to either column.
