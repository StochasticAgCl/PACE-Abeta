# Reproduction checks

The reference experiment uses the authorized ADNI regional PET and PHC cognition
tables described in the README. Participant-level validation artifacts remain
outside the public source distribution.

## Training and selection audit

On 30 October 2026, an independent audit checked all 128 completed trials and
1,280 fold fits. The audit verified 6,562 recorded file hashes and recomputed
validation counts, training-reference rank calibration, positive-gap summaries,
and the selection objective from saved fold outputs. Pooling all folds selected
trial 125 (zero-based), matching the frozen search result. The audit confirmed
selection by objective, then epochs and trial number, and verified the ordering
of selection, full-training refitting, and test evaluation.

The selected configuration uses β = 0.10319626890596051,
τ = 0.020366074203530617, ρ = 0.8, learning rate = 0.00002456503433639552,
and 1,518 epochs. Encoder widths are 64, 32, and 16, with GELU activations.
The selected configuration, full-training refit, and pretrained bundle agree.
The bundled preprocessing coefficients and feature order match full-training
preprocessing exactly. These coefficients are unchanged because the excluded
participants supplied no CN observations for covariate adjustment.

Across 874 pooled validation pairs, the selected trial has 223 nonincrease
violations, VR = 0.2551487414, and conditional rank-score gap = 0.0433303918.
Their product, 0.0110556949, is the selection objective. Exact ties count in VR
during selection; the gap is the conditional mean of strictly positive declines.
None of the 128 completed trials has an adjacent validation tie.

## Full pipeline and pretrained model checks

A fresh run completed cohort preparation, all 128 ten-fold search attempts,
full-training refitting, comparator fitting, score calibration, statistical
analyses, and figure generation. The search used categorical ρ values
{0.0, 0.1, ..., 0.9}. No prior trial results or fitted comparator outputs were
reused for this run.

The final checkpoint was independently evaluated from its saved linear-layer
weights and activation functions. All posterior means, log variances, and
variances matched the saved train and test arrays bitwise. The public prediction
code, using the bundled model and preprocessing, also reproduced all 1,720
training and 1,250 test posterior outputs and calibrated scores bitwise from
the original input values and authorized training reference.

Both Fig.2A and Fig.2C use subject-cluster bootstrap-t mean comparisons with
100,000 resamples and seed 42. Diagnosis comparisons yielded p = 0.0004
for CN versus MCI and p < 0.0001 for MCI versus AD. Gap comparisons yielded
p < 0.0001 for Centiloids and SLOPE, and p = 0.0005 for dEBM. All values below
0.0001 use the same reporting threshold.

Independent calculations reproduced all temporal metrics and all five
bootstrap exceedance counts. A separate piecewise fit reproduced the training
breakpoint within 2.1e-14 and verified its unchanged application to the test set.
Baseline demographic counts and summary statistics were independently checked
against the earliest eligible PET observations.

The memory model includes a baseline age-by-time interaction. Repeating the
cognition analysis from the raw input reproduced all 18 output tables exactly.
Independent calculations verified the slope contrasts and all 200 plotted
predictions and confidence intervals.

## Reference cohort

After excluding participants with missing or other baseline diagnoses, the
cohort contains 1,236 participants and 2,970 scans. The seed-42 split contains
846 training participants with 1,720 PET scans and
390 test participants with 1,250 scans. The 390 participants with multiple
training visits form ten subject-level validation folds. All 456 single-visit
training participants remain in every fitting split. There are 874 adjacent
pairs in pooled cross-validation and 860 in testing.

## Reference manuscript results

Each method uses its own participant-weighted training-reference rank map.
At threshold zero, the test results are:

| Method | Violations / pairs | Violation ratio | Mean conditional violation gap |
| --- | --- | --- | --- |
| PACE-Aβ | 191 / 860 | 0.2220930233 | 0.0406780670 |
| Centiloids | 348 / 860 | 0.4046511628 | 0.0632826478 |
| SLOPE | 346 / 860 | 0.4023255814 | 0.0584805377 |
| dEBM | 317 / 860 | 0.3686046512 | 0.0552360990 |

Full-training refitting produces 189/874 training violations and 191/860 test
violations. The test VR-times-CVG product is 0.0090343149.

The training-selected piecewise knot is 0.3287530424463339. This same knot is
applied to test curves and memory grouping. The diagnosis panel includes 590 CN,
507 MCI, and 134 AD visits; 19 visits are excluded only from that panel.

The memory analysis starts with 232 baseline CN/MCI participants with
Centiloids ≤30. Centiloid matching retains 54 participants per group and 664
cognitive observations. At the common reference age, estimated annual PHC_MEM
slopes are −0.0063778315 in the early group and −0.0511445671 in the late group.
The late-minus-early contrast is −0.0447667355, with 95% CI
[−0.0625297131, −0.0270037580] and two-sided Wald p = 0.0000007830140.

## Numerical environment

Python dependencies are pinned in `requirements.txt`; `renv.lock` records the
R package versions.

Validation uses Python 3.12.13 and R 4.5.1. PACE folds use one CPU thread each.
SLOPE uses its original four-thread PyTorch/Numba context, including test
inference after UMAP fitting. The `--parallel` option changes the number of
simultaneous fold workers; `--folds` changes the validation experiment.

Exact numerical reproduction requires the same inputs, row order, seeds,
numerical libraries, and execution context. Figure pixels additionally depend
on fonts and rendering libraries. Different authorized data releases can change
cohort membership and downstream estimates.
