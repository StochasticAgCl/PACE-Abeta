# Memory plot objects: histograms, densities, fixed-effect trajectories, and confidence bands.
if (!exists("root")) stop("Source this file from figure4.R")
library(dplyr)
library(tidyr)
library(metafor)
dat_lmm <- read.csv(file.path(root,"data/analysis/memory/dat_lmm.csv"))
dat_lmm$pt_group <- factor(dat_lmm$pt_group, levels=c("early", "late"))
dat_lmm$sex <- factor(dat_lmm$sex, levels=c("F", "M"))
matched_subjects <- read.csv(file.path(root,"data/analysis/memory/matched_subjects.csv"))
slope_results <- read.csv(file.path(root,"data/analysis/memory/wald_slope_results.csv"))
fit_compare <- readRDS(file.path(root,"data/analysis/memory/comparative_model_rma_mv.rds"))
ks_after <- stats::ks.test(
  matched_subjects$baseline_centiloids[matched_subjects$pt_group == "early"],
  matched_subjects$baseline_centiloids[matched_subjects$pt_group == "late"]
)
img_dir <- file.path(root,"figures/original_style")
library(ggplot2)

theme_paper <- function(base_size = 12, base_family = "sans") {
  theme_classic(base_size = base_size, base_family = base_family) +
    theme(
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      axis.line = element_line(color = "black", linewidth = 0.4),
      axis.ticks = element_line(color = "black", linewidth = 0.4),
      axis.ticks.length = grid::unit(2.5, "pt"),
      axis.title = element_text(size = 12, color = "black"),
      axis.text = element_text(size = 10, color = "black"),
      legend.title = element_text(size = 11, color = "black"),
      legend.text = element_text(size = 10, color = "black"),
      plot.title = element_text(size = 12, color = "black", hjust = 0.5, face = "plain"),
      plot.subtitle = element_text(size = 11, color = "black", hjust = 0.5),
      strip.background = element_blank(),
      strip.text = element_text(size = 11, color = "black"),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank()
    )
}

save_paper_plot <- function(filename_png, plot, width, height) {
  ggsave(
    filename = filename_png,
    plot = plot,
    width = width,
    height = height,
    units = "in",
    dpi = 300,
    bg = "white"
  )
}

format_p_value <- function(p_value) {
  if (p_value < 0.001) {
    formatC(p_value, format = "e", digits = 2)
  } else {
    sprintf("%.4f", p_value)
  }
}

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

time_ranges_adni <- dat_lmm %>%
  group_by(pt_group) %>%
  summarise(
    xmin = min(time_year, na.rm = TRUE),
    xmax = max(time_year, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    ymin = -Inf,
    ymax = Inf
  )

interaction_result <- slope_results %>%
  filter(estimand == "late_minus_early_slope")

label_diff <- sprintf(
  "p-value = %s",
  format_p_value(interaction_result$p_value)
)

x_span <- diff(range(pred_adni$x, na.rm = TRUE))
y_span <- diff(range(c(pred_adni$conf.low, pred_adni$conf.high), na.rm = TRUE))

if (is.na(y_span) || y_span == 0) {
  y_span <- 1
}

x_diff <- min(pred_adni$x, na.rm = TRUE) + 0.03 * x_span
y_diff <- min(pred_adni$conf.low, na.rm = TRUE) + 0.08 * y_span

p_trend_adni <- ggplot() +
  geom_rect(
    data = time_ranges_adni,
    aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = pt_group),
    inherit.aes = FALSE,
    alpha = 0.04
  ) +
  geom_ribbon(
    data = pred_adni,
    aes(x = x, ymin = conf.low, ymax = conf.high, fill = group),
    alpha = 0.20,
    color = NA
  ) +
  geom_line(
    data = pred_adni,
    aes(x = x, y = predicted, color = group),
    linewidth = 1.4
  ) +
  annotate(
    "text",
    x = x_diff,
    y = y_diff,
    label = label_diff,
    hjust = 0,
    vjust = 0,
    size = 4.0,
    fontface = "bold"
  ) +
  labs(
    title = "ADNI test (CN/MCI)",
    x = "Follow-up (years)",
    y = "PHC_MEM",
    color = "PACE-A\u03b2 Group",
    fill = "PACE-A\u03b2 Group"
  ) +
  scale_color_discrete(labels = c(early = "lower", late = "higher")) +
  scale_fill_discrete(labels = c(early = "lower", late = "higher")) +
  theme_paper() +
  theme(
    legend.position = "right"
  )

save_paper_plot(
  filename_png = file.path(img_dir, "adni_phc_mem.png"),
  plot = p_trend_adni,
  width = 4.8,
  height = 3.8
)


p_balance <- ggplot(
  matched_subjects,
  aes(x = baseline_centiloids, fill = pt_group, color = pt_group)
) +
  geom_histogram(position = "identity", bins = 30, alpha = 0.35) +
  geom_density(aes(y = after_stat(count)), linewidth = 1.0, alpha = 0.15) +
  labs(
    title = "ADNI test (CN/MCI)",
    x = "Baseline centiloids",
    y = "Subject count",
    fill = "PACE-A\u03b2 Group",
    color = "PACE-A\u03b2 Group"
  ) +
  scale_color_discrete(labels = c(early = "lower", late = "higher")) +
  scale_fill_discrete(labels = c(early = "lower", late = "higher")) +
  theme_paper() +
  theme(legend.position = "right") +
  annotate(
    "text",
    x = Inf,
    y = Inf,
    label = sprintf("p-value = %.2e", ks_after$p.value),
    hjust = 1.02,
    vjust = 1.4,
    size = 4.0,
    fontface = "bold"
  )

save_paper_plot(
  filename_png = file.path(img_dir, "adni_matching.png"),
  plot = p_balance,
  width = 4.8,
  height = 3.8
)



# Additional SVG exports use the same original ggplot objects and dimensions.
ggsave(file.path(img_dir, "adni_phc_mem.svg"), p_trend_adni, width=4.8, height=3.8, units="in", bg="white")
ggsave(file.path(img_dir, "adni_matching.svg"), p_balance, width=4.8, height=3.8, units="in", bg="white")
write.csv(pred_adni, file.path(root,"data/analysis/original_plot_predictions.csv"), row.names=FALSE)
jsonlite::write_json(list(ks_p=ks_after$p.value, interaction_p=interaction_result$p_value,
  width=4.8, height=3.8, dpi=300), file.path(root,"data/analysis/original_plot_checks.json"),
  auto_unbox=TRUE, pretty=TRUE, digits=17)

ggsave(file.path(img_dir,"figure4a.pdf"),p_balance,width=4.8,height=3.8,units="in",device=cairo_pdf)
ggsave(file.path(img_dir,"figure4b.pdf"),p_trend_adni,width=4.8,height=3.8,units="in",device=cairo_pdf)
