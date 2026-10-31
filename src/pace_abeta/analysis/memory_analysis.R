# Longitudinal memory analysis with known measurement variances.
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 6) stop("Usage: memory_analysis.R PET.csv COGNITION.csv OUTPUT KNOT SEED CL_MAX")
pet_path <- args[1]; cognition_path <- args[2]; output_path <- args[3]
knot_value <- as.numeric(args[4]); seed_value <- as.integer(args[5]); centiloid_max <- as.numeric(args[6])
if (!is.finite(knot_value) || !is.finite(centiloid_max) || is.na(seed_value)) stop("Invalid analysis settings")
library(dplyr)
library(tidyr)
library(metafor)
clean_sex <- function(x) {
  sx <- tolower(trimws(as.character(x)))
  sx[sx %in% c("m", "male", "1")] <- "M"
  sx[sx %in% c("f", "female", "2")] <- "F"
  factor(sx, levels=c("F", "M"))
}
init_summary <- function(out_dir, run_tag, header_info) {
  path <- file.path(out_dir, "summary.txt")
  writeLines(paste(run_tag, header_info, sep="\n"), path)
  function(text) cat(text, file=path, append=TRUE)
}


data_adni_pt <- read.csv(pet_path, header = TRUE)
mem <- read.csv(cognition_path, header = TRUE)

# =========================================================
#                PHC_Diagnosis
#                PHC_Sex, PHC_Education
# =========================================================

best_bp_train <- knot_value
set.seed(seed_value)

# =========================================================
# =========================================================
adni_pt_base <- data_adni_pt %>%
  mutate(
    RID = as.character(RID),
    PHC_Age_PET = as.numeric(PHC_Age_PET),
    PHC_CENTILOIDS = as.numeric(PHC_CENTILOIDS),
    Pseudotime_Normalized = as.numeric(Pseudotime_Normalized),
    PHC_Diagnosis = as.numeric(PHC_Diagnosis)
  ) %>%
  filter(
    !is.na(RID),
    !is.na(PHC_Age_PET),
    !is.na(PHC_CENTILOIDS),
    !is.na(Pseudotime_Normalized),
    !is.na(PHC_Diagnosis)
  ) %>%
  arrange(RID, PHC_Age_PET) %>%
  group_by(RID) %>%
  slice(1) %>%
  ungroup()

# =========================================================
#      - baseline PHC_Diagnosis %in% c(1, 2)
#      - baseline PHC_CENTILOIDS <= centiloid_max
# =========================================================
adni_pt_selected <- adni_pt_base %>%
  filter(
    PHC_Diagnosis %in% c(1, 2),
    PHC_CENTILOIDS <= centiloid_max
  ) %>%
  mutate(
    pt_group = ifelse(Pseudotime_Normalized <= best_bp_train, "early", "late")
  ) %>%
  transmute(
    RID,
    PHC_Age_PET,
    baseline_centiloids = PHC_CENTILOIDS,
    baseline_diagnosis = PHC_Diagnosis,
    Pseudotime_Normalized,
    pt_group
  )

# =========================================================
# =========================================================
mem2 <- mem %>%
  mutate(
    RID = as.character(RID),
    PHC_Age_Cognition = as.numeric(PHC_Age_Cognition),
    PHC_MEM = as.numeric(PHC_MEM),
    PHC_MEM_SE = as.numeric(PHC_MEM_SE),
    PHC_Diagnosis = as.numeric(PHC_Diagnosis),
    PHC_Sex = as.character(PHC_Sex),
    PHC_Education = as.numeric(PHC_Education)
  ) %>%
  filter(
    !is.na(RID),
    !is.na(PHC_Age_Cognition),
    !is.na(PHC_MEM),
    !is.na(PHC_MEM_SE),
    PHC_MEM_SE > 0
  )

# =========================================================
# =========================================================
mem_anchor <- mem2 %>%
  inner_join(
    adni_pt_selected %>% select(
      RID, PHC_Age_PET, baseline_centiloids,
      Pseudotime_Normalized, pt_group, baseline_diagnosis
    ),
    by = "RID"
  ) %>%
  mutate(
    age_diff_abs = abs(PHC_Age_Cognition - PHC_Age_PET)
  ) %>%
  arrange(RID, age_diff_abs, PHC_Age_Cognition) %>%
  group_by(RID) %>%
  slice(1) %>%
  ungroup() %>%
  select(
    RID,
    baseline_age_pet = PHC_Age_PET,
    baseline_centiloids,
    baseline_pseudotime = Pseudotime_Normalized,
    baseline_age_mem = PHC_Age_Cognition,
    baseline_sex = PHC_Sex,
    baseline_education = PHC_Education,
    baseline_diagnosis,
    pt_group
  )

# =========================================================
# =========================================================
mem_sub <- mem2 %>%
  inner_join(mem_anchor, by = "RID") %>%
  filter(PHC_Age_Cognition >= baseline_age_mem) %>%
  mutate(
    time_year = PHC_Age_Cognition - baseline_age_mem
  ) %>%
  arrange(RID, PHC_Age_Cognition)

# =========================================================
# =========================================================
rm(mem)
gc()

# =========================================================
# =========================================================
table(mem_sub$pt_group)

check_baseline <- mem_sub %>%
  group_by(RID) %>%
  summarise(
    min_time_year = min(time_year, na.rm = TRUE),
    first_age = min(PHC_Age_Cognition, na.rm = TRUE),
    baseline_age_mem = first(baseline_age_mem),
    n_unique_baseline_age_mem = n_distinct(baseline_age_mem),
    .groups = "drop"
  )

print(head(check_baseline))
summary(check_baseline$n_unique_baseline_age_mem)
summary(mem_sub$time_year)

mem_sub %>%
  select(
    RID, pt_group, baseline_diagnosis, baseline_centiloids, baseline_pseudotime,
    PHC_Age_Cognition, baseline_age_pet, baseline_age_mem,
    baseline_sex, baseline_education, time_year, PHC_MEM, PHC_MEM_SE
  ) %>%
  arrange(RID, PHC_Age_Cognition) %>%
  head(30)

# =========================================================
# =========================================================


# =========================================================
# =========================================================
dat_lmm_all <- mem_sub %>%
  dplyr::mutate(
    RID = as.character(RID),
    pt_group = factor(pt_group, levels = c("early", "late")),
    PHC_MEM = as.numeric(PHC_MEM),
    PHC_MEM_SE = as.numeric(PHC_MEM_SE),
    time_year = as.numeric(time_year),
    baseline_centiloids = as.numeric(baseline_centiloids),
    baseline_pseudotime = as.numeric(baseline_pseudotime),
    baseline_diagnosis = as.numeric(baseline_diagnosis),
    age_baseline = as.numeric(baseline_age_mem),
    sex = clean_sex(baseline_sex),
    education = as.numeric(baseline_education)
  ) %>%
  dplyr::filter(
    !is.na(RID),
    !is.na(time_year),
    !is.na(pt_group),
    !is.na(PHC_MEM),
    !is.na(PHC_MEM_SE),
    PHC_MEM_SE > 0,
    !is.na(baseline_centiloids),
    !is.na(baseline_pseudotime),
    !is.na(baseline_diagnosis),
    !is.na(age_baseline),
    !is.na(sex),
    !is.na(education)
  ) %>%
  mutate(
    age_baseline_c = age_baseline - mean(age_baseline),
    education_c = education - mean(education)
  ) %>%
  arrange(RID, time_year, PHC_Age_Cognition)

# Collect the analyzable participant pool.
subject_pool <- dat_lmm_all %>%
  group_by(RID) %>%
  summarise(
    pt_group = first(as.character(pt_group)),
    baseline_centiloids = first(baseline_centiloids),
    baseline_pseudotime = first(baseline_pseudotime),
    baseline_diagnosis = first(baseline_diagnosis),
    n_visits = n(),
    .groups = "drop"
  )

target_subject_pool <- subject_pool

# Match equal numbers of early and late participants within Centiloid bins.
n_bins_target <- 10
centiloid_breaks <- unique(quantile(
  target_subject_pool$baseline_centiloids,
  probs = seq(0, 1, length.out = n_bins_target + 1),
  na.rm = TRUE,
  type = 8
))

if (length(centiloid_breaks) < 3) {
  stop("Not enough unique centiloid breakpoints to create matching bins.")
}

subject_pool_binned <- target_subject_pool %>%
  mutate(
    centiloid_bin = cut(
      baseline_centiloids,
      breaks = centiloid_breaks,
      include.lowest = TRUE,
      ordered_result = TRUE
    )
  ) %>%
  filter(!is.na(centiloid_bin))

bin_keep <- subject_pool_binned %>%
  count(centiloid_bin, pt_group, name = "n") %>%
  tidyr::pivot_wider(
    names_from = pt_group,
    values_from = n,
    values_fill = 0
  ) %>%
  mutate(
    n_keep = pmin(early, late)
  ) %>%
  filter(n_keep > 0)

# Sample the same matched participants for a fixed seed.
set.seed(seed_value)
matched_subjects <- subject_pool_binned %>%
  inner_join(bin_keep %>% select(centiloid_bin, n_keep), by = "centiloid_bin") %>%
  group_by(centiloid_bin, pt_group) %>%
  group_modify(function(.x, .y) {
    n_to_keep <- unique(.x$n_keep)
    .x[sample(seq_len(nrow(.x)), n_to_keep), , drop = FALSE]
  }) %>%
  ungroup() %>%
  select(RID, pt_group, baseline_centiloids, centiloid_bin, n_visits)

dat_lmm <- dat_lmm_all %>%
  semi_join(matched_subjects, by = "RID") %>%
  mutate(
    pt_group = factor(pt_group, levels = c("early", "late")),
    obs_id = paste0("outcome_", row_number())
  )

# =========================================================
# =========================================================
out_dir <- output_path
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)


balance_by_group <- function(df) {
  df %>%
    group_by(pt_group) %>%
    summarise(
      n_subjects = n_distinct(RID),
      mean_centiloids = mean(baseline_centiloids, na.rm = TRUE),
      sd_centiloids = sd(baseline_centiloids, na.rm = TRUE),
      median_centiloids = median(baseline_centiloids, na.rm = TRUE),
      q25_centiloids = quantile(baseline_centiloids, 0.25, na.rm = TRUE),
      q75_centiloids = quantile(baseline_centiloids, 0.75, na.rm = TRUE),
      .groups = "drop"
    )
}

subject_balance_before <- target_subject_pool %>%
  mutate(pt_group = factor(pt_group, levels = c("early", "late"))) %>%
  balance_by_group()

subject_balance_after <- matched_subjects %>%
  mutate(pt_group = factor(pt_group, levels = c("early", "late"))) %>%
  balance_by_group()

bin_balance_after <- matched_subjects %>%
  count(centiloid_bin, pt_group, name = "n") %>%
  tidyr::pivot_wider(
    names_from = pt_group,
    values_from = n,
    values_fill = 0
  )

write.csv(subject_balance_before, file.path(out_dir, "centiloid_balance_before.csv"), row.names = FALSE)
write.csv(subject_balance_after, file.path(out_dir, "centiloid_balance_after.csv"), row.names = FALSE)
write.csv(bin_keep, file.path(out_dir, "centiloid_bin_matching_plan.csv"), row.names = FALSE)
write.csv(bin_balance_after, file.path(out_dir, "centiloid_bin_balance_after.csv"), row.names = FALSE)
write.csv(matched_subjects, file.path(out_dir, "matched_subjects.csv"), row.names = FALSE)

ks_before <- stats::ks.test(
  target_subject_pool$baseline_centiloids[target_subject_pool$pt_group == "early"],
  target_subject_pool$baseline_centiloids[target_subject_pool$pt_group == "late"]
)

ks_after <- stats::ks.test(
  matched_subjects$baseline_centiloids[matched_subjects$pt_group == "early"],
  matched_subjects$baseline_centiloids[matched_subjects$pt_group == "late"]
)

append_summary <- init_summary(
  out_dir   = out_dir,
  run_tag   = "ADNI_MEMORY_KNOWN_SE",
  header_info = paste0(
    "Outcome: PHC_MEM\n",
    "Known observation uncertainty: PHC_MEM_SE; total variance includes PHC_MEM_SE^2, a subject random intercept, and an observation-level residual variance.\n",
    "Time: time_year\n",
    "Group: pt_group (early vs late)\n",
    "Baseline covariates: PHC_Age_Cognition, PHC_Sex, PHC_Education\n",
    "Selection: baseline CN/MCI; configured Centiloid upper bound\n",
    "Matching: pooled baseline centiloid quantile bins; equal early/late subjects per bin.\n",
    "Target subjects: ", nrow(target_subject_pool), "\n",
    "\n\nSubjects before matching:\n",
    paste(capture.output(print(subject_balance_before)), collapse = "\n"),
    "\n\nSubjects after matching:\n",
    paste(capture.output(print(subject_balance_after)), collapse = "\n"),
    "\n\nKS test before matching:\n",
    paste(capture.output(print(ks_before)), collapse = "\n"),
    "\n\nKS test after matching:\n",
    paste(capture.output(print(ks_after)), collapse = "\n"),
    "\n\n"
  )
)

# Known-measurement-error frequentist comparative LMM.
fit_compare <- metafor::rma.mv(
  yi = PHC_MEM,
  V = PHC_MEM_SE^2,
  mods = ~ time_year * pt_group + age_baseline_c + sex + education_c +
    time_year:age_baseline_c,
  random = list(~ 1 | RID, ~ 1 | obs_id),
  data = dat_lmm,
  method = "REML",
  test = "z",
  sparse = TRUE
)

summarise_wald_contrast <- function(fit, contrast, estimand) {
  beta_hat <- stats::coef(fit)
  beta_vcov <- stats::vcov(fit)
  weights <- stats::setNames(rep(0, length(beta_hat)), names(beta_hat))

  if (!all(names(contrast) %in% names(weights))) {
    stop("Contrast contains coefficients that are absent from the fitted model.")
  }

  weights[names(contrast)] <- contrast
  estimate <- sum(weights * beta_hat)
  standard_error <- sqrt(as.numeric(t(weights) %*% beta_vcov %*% weights))
  z_value <- estimate / standard_error
  p_value <- 2 * stats::pnorm(abs(z_value), lower.tail = FALSE)

  tibble::tibble(
    estimand = estimand,
    beta = estimate,
    standard_error = standard_error,
    ci_low = estimate - stats::qnorm(0.975) * standard_error,
    ci_high = estimate + stats::qnorm(0.975) * standard_error,
    z_value = z_value,
    p_value = p_value
  )
}

slope_results <- dplyr::bind_rows(
  summarise_wald_contrast(
    fit_compare,
    c(time_year = 1),
    "early_slope"
  ),
  summarise_wald_contrast(
    fit_compare,
    c(time_year = 1, "time_year:pt_grouplate" = 1),
    "late_slope"
  ),
  summarise_wald_contrast(
    fit_compare,
    c("time_year:pt_grouplate" = 1),
    "late_minus_early_slope"
  )
)

model_diagnostics <- tibble::tibble(
  model = "comparative",
  n_observations = fit_compare$k,
  n_subjects = n_distinct(dat_lmm$RID),
  subject_variance = fit_compare$sigma2[1],
  residual_variance = fit_compare$sigma2[2],
  log_likelihood = as.numeric(stats::logLik(fit_compare))
)

# Save comparative model results.
out_file <- file.path(out_dir, "frequentist_results.txt")
cat("", file = out_file)

cat("\n===== KNOWN-SE FREQUENTIST COMPARATIVE LMM =====\n", file = out_file, append = TRUE)
cat(paste(capture.output(summary(fit_compare)), collapse = "\n"), file = out_file, append = TRUE)
cat("\n\n===== WALD SLOPE TESTS =====\n", file = out_file, append = TRUE)
cat(paste(capture.output(print(slope_results)), collapse = "\n"), file = out_file, append = TRUE)
cat("\n\n===== MODEL DIAGNOSTICS =====\n", file = out_file, append = TRUE)
cat(paste(capture.output(print(model_diagnostics)), collapse = "\n"), file = out_file, append = TRUE)

write.csv(slope_results, file.path(out_dir, "wald_slope_results.csv"), row.names = FALSE)
write.csv(model_diagnostics, file.path(out_dir, "model_diagnostics.csv"), row.names = FALSE)
saveRDS(fit_compare, file.path(out_dir, "comparative_model_rma_mv.rds"))

append_summary("\nKnown-SE frequentist comparative model:\n")
append_summary(paste(capture.output(summary(fit_compare)), collapse = "\n"))
append_summary("\nWald slope tests:\n")
append_summary(paste(capture.output(print(slope_results)), collapse = "\n"))
append_summary("\nModel diagnostics:\n")
append_summary(paste(capture.output(print(model_diagnostics)), collapse = "\n"))

# =========================================================
# =========================================================

prediction_grid <- tidyr::expand_grid(
  time_year = seq(
    min(dat_lmm$time_year, na.rm = TRUE),
    max(dat_lmm$time_year, na.rm = TRUE),
    length.out = 100
  ),
  pt_group = factor(c("early", "late"), levels = c("early", "late"))
) %>%
  mutate(
    age_baseline_c = 0,
    education_c = 0,
    sex = factor(levels(dat_lmm$sex)[1], levels = levels(dat_lmm$sex))
  )

prediction_matrix <- stats::model.matrix(
  ~ time_year * pt_group + age_baseline_c + sex + education_c +
    time_year:age_baseline_c,
  data = prediction_grid
)

colnames(prediction_matrix)[colnames(prediction_matrix) == "(Intercept)"] <- "intrcpt"
prediction_matrix <- prediction_matrix[, names(stats::coef(fit_compare)), drop = FALSE]

prediction_beta <- stats::coef(fit_compare)
prediction_vcov <- stats::vcov(fit_compare)
prediction_mean <- as.numeric(prediction_matrix %*% prediction_beta)
prediction_se <- sqrt(rowSums((prediction_matrix %*% prediction_vcov) * prediction_matrix))

pred_adni <- prediction_grid %>%
  transmute(
    x = time_year,
    group = as.character(pt_group),
    predicted = prediction_mean,
    conf.low = prediction_mean - stats::qnorm(0.975) * prediction_se,
    conf.high = prediction_mean + stats::qnorm(0.975) * prediction_se
  )



# Machine-readable intermediate data and fitted-model summaries.
for (nm in c("adni_pt_base", "adni_pt_selected", "mem_anchor", "dat_lmm_all",
             "subject_pool", "target_subject_pool", "subject_pool_binned",
             "dat_lmm")) {
  write.csv(get(nm), file.path(out_dir, paste0(nm, ".csv")), row.names=FALSE)
}
stopifnot(isTRUE(all.equal(as.numeric(diag(fit_compare$V)), dat_lmm$PHC_MEM_SE^2)))
stats <- list(breakpoint=best_bp_train,
              ks_before=list(D=unname(ks_before$statistic), p=ks_before$p.value),
              ks_after=list(D=unname(ks_after$statistic), p=ks_after$p.value),
              known_variances_verified=TRUE,
              packages=sapply(c("metafor", "dplyr", "tidyr"), function(p) as.character(packageVersion(p))))
jsonlite::write_json(stats, file.path(out_dir,"run_stats.json"), auto_unbox=TRUE, pretty=TRUE, digits=17)
write.csv(pred_adni, file.path(out_dir,"predicted_trajectories.csv"), row.names=FALSE)
write.csv(data.frame(term=names(coef(fit_compare)), beta=coef(fit_compare)),
          file.path(out_dir,"coefficients.csv"), row.names=FALSE)
write.csv(vcov(fit_compare), file.path(out_dir,"coefficient_covariance.csv"), row.names=TRUE)
writeLines(capture.output(sessionInfo()),file.path(out_dir,"sessionInfo.txt"))
