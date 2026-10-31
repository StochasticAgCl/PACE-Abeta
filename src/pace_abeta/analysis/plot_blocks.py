"""Manuscript plot primitives with data and output locations supplied at runtime."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .fonts import FONT_FAMILY
import seaborn as sns
import statsmodels.api as sm
from .metrics import diagnosis_tests

def format_bootstrap_p_value(p):
    """Apply one reporting threshold to every bootstrap comparison."""
    if not np.isfinite(p) or not 0 < p <= 1:
        raise ValueError("A displayed bootstrap p-value must be finite and positive.")
    return "p < 0.0001" if p < .0001 else f"p = {p:.4f}"


def diagnosis_plot(data, comparisons=None):
    sns.set_theme(style="white", context="paper", font_scale=1.2)
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "axes.linewidth": 1.0,
        "axes.edgecolor": "black",
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    diag_map = {
        1: "CN",
        2: "MCI",
        3: "AD"
    }
    diag_order = ["CN", "MCI", "AD"]
    diag_palette = {
        "CN": "#AFCBE8",
        "MCI": "#F7C58A",
        "AD": "#F2A6A2",
    }
    method_key_map = {
        "SLOPE_test": "SLOPE",
        "AE_test": "AE-UMAP",
        "1d_AE_test": "DAPS",
    }
    method_order = ["SLOPE", "AE-UMAP", "DAPS"]
    method_palette = {
        "SLOPE": "#B8E0D2",
        "AE-UMAP": "#A2D2FF",
        "DAPS": "#FFD1DC",
    }
    def add_sig_annotation(ax, x1, x2, y, h, p, fontsize=6.5):
        """
        x1, x2: categorical positions, e.g. 0 and 1
        y: base y position
        h: bracket height
        """
        ax.plot(
            [x1, x1, x2, x2],
            [y, y + h, y + h, y],
            lw=1.0,
            c="black"
        )
        text = format_bootstrap_p_value(p)
        ax.text(
            (x1 + x2) / 2,
            y + h,
            text,
            ha="center",
            va="bottom",
            fontsize=fontsize
        )
    df_pt = data.copy()
    required_cols = ["PHC_Diagnosis", "Pseudotime_Normalized"]
    missing_cols = [c for c in required_cols if c not in df_pt.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in data: {missing_cols}")
    df_pt = df_pt.dropna(subset=["PHC_Diagnosis", "Pseudotime_Normalized"]).copy()
    df_pt = df_pt[df_pt["PHC_Diagnosis"].isin([1, 2, 3])].copy()
    df_pt["Diagnosis"] = df_pt["PHC_Diagnosis"].map(diag_map)
    df_pt["Diagnosis"] = pd.Categorical(df_pt["Diagnosis"], categories=diag_order, ordered=True)
    if comparisons is None:
        comparisons = diagnosis_tests(data)["comparisons"]
    p_values = {row["comparison"]: row["p_value"] for row in comparisons}
    p_cn_mci = p_values["CN_vs_MCI"]
    p_mci_ad = p_values["MCI_vs_AD"]
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    sns.boxplot(
        data=df_pt,
        x="Diagnosis",
        y="Pseudotime_Normalized",
        order=diag_order,
        palette=diag_palette,
        width=0.55,
        linewidth=1.0,
        fliersize=2.5,
        flierprops={"alpha": 0.0},
        ax=ax
    )
    ax.set_xlabel("Diagnosis")
    ax.set_ylabel("Rank-calibrated DAPS score")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    y_min = df_pt["Pseudotime_Normalized"].min()
    y_max = df_pt["Pseudotime_Normalized"].max()
    y_range = y_max - y_min if y_max > y_min else 1.0
    base_y_1 = y_max + 0.08 * y_range
    base_y_2 = y_max + 0.25 * y_range
    h = 0.035 * y_range
    add_sig_annotation(ax, 0, 1, base_y_1, h, p_cn_mci)
    add_sig_annotation(ax, 1, 2, base_y_2, h, p_mci_ad)
    ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.45 * y_range)
    plt.tight_layout()
    return fig

def ratio_plot(ratio_df, method_palette):
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    bars = ax.bar(
        ratio_df["Method"],
        ratio_df["Violation_Ratio"],
        color=[method_palette[m] for m in ratio_df["Method"]],
        edgecolor="black",
        linewidth=0.8
    )
    ax.set_xlabel("Method")
    ax.set_ylabel("Violation ratio")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    max_ratio = ratio_df["Violation_Ratio"].max()
    ax.set_ylim(0, max_ratio * 1.20 if max_ratio > 0 else 1)
    for bar, val in zip(bars, ratio_df["Violation_Ratio"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=10
        )
    plt.tight_layout()
    return fig

def gap_plot(gap_df, method_order, method_palette, comparisons=None):
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    sns.boxplot(
        data=gap_df,
        x="Method",
        y="Violation_Gap",
        order=method_order,
        palette=method_palette,
        width=0.55,
        linewidth=1.0,
        fliersize=2.5,
        flierprops={"alpha": 0.0},
        ax=ax
    )
    ax.set_xlabel("Method")
    ax.set_ylabel("Conditional violation gap")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Preserve all observations in the boxplot statistics; only the outlier
    # markers are transparent. Set the display range from the visible whiskers.
    visible_y = np.concatenate([np.asarray(line.get_ydata(), dtype=float)
                                for line in ax.lines if line.get_alpha() != 0])
    maximum = float(visible_y.max())
    span = max(maximum, np.finfo(float).eps)
    ax.set_ylim(-.05 * span, maximum + .08 * span)
    if comparisons:
        positions = {method: i for i, method in enumerate(method_order)}
        ordered = sorted(comparisons,
                         key=lambda row: abs(positions[row["reference"]] -
                                             positions[row["comparator"]]))
        for level, row in enumerate(ordered):
            x1, x2 = sorted((positions[row["reference"]], positions[row["comparator"]]))
            y = maximum + (.06 + .15 * level) * span
            h = .022 * span
            ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
                    color="black", linewidth=1.0)
            label = format_bootstrap_p_value(row["p_value"])
            ax.text((x1 + x2) / 2, y + h, label,
                    ha="center", va="bottom", fontsize=6.5)
        ax.set_ylim(-.05 * span, maximum + .54 * span)
    plt.tight_layout()
    return fig

def piecewise_plot(df, title, *, manual_bp,
                                           xcol="Pseudotime_Normalized", ycol="PHC_CENTILOIDS"):
    """Plot descriptive piecewise fits at a required, externally supplied knot.
    This function performs no breakpoint selection or significance filtering.
    Segment p-values are descriptive annotations and do not affect the knot.
    """
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "axes.linewidth": 1.0,
        "axes.edgecolor": "black",
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    d = df[[xcol, ycol]].dropna().copy()
    x = d[xcol].to_numpy().astype(float)
    y = d[ycol].to_numpy().astype(float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    n = len(x)
    if n < 10:
        raise ValueError(f"Too few samples (n={n}); fixed-knot piecewise regression requires at least 10 samples.")
    best_bp = float(manual_bp)
    if not (x.min() <= best_bp <= x.max()):
        raise ValueError(
            f"manual_bp={best_bp} is outside the x range [{x.min():.4g}, {x.max():.4g}]"
        )
    left_mask = x <= best_bp
    right_mask = x > best_bp
    if left_mask.sum() < 3 or right_mask.sum() < 3:
        raise ValueError(
            f"manual_bp={best_bp:.4g} leaves too few samples on one side: "
            f"left={left_mask.sum()}, right={right_mask.sum()}."
        )
    hinge = np.maximum(0.0, x - best_bp)
    X = np.column_stack([np.ones_like(x), x, hinge])
    best_coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ best_coef
    best_sse = float(np.sum((y - yhat) ** 2))
    a, b, c = best_coef
    slope_left = b
    slope_right = b + c
    left_mask = x <= best_bp
    right_mask = x > best_bp
    x1, y1 = x[left_mask], y[left_mask]
    x2, y2 = x[right_mask], y[right_mask]
    if len(x1) < 3 or len(x2) < 3:
        raise ValueError(
            f"The final breakpoint leaves too few samples on one side: left={len(x1)}, right={len(x2)}"
        )
    model1 = sm.OLS(y1, sm.add_constant(x1)).fit()
    p1 = float(model1.pvalues[1])
    r2_1 = float(model1.rsquared)
    model2 = sm.OLS(y2, sm.add_constant(x2)).fit()
    p2 = float(model2.pvalues[1])
    r2_2 = float(model2.rsquared)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    scatter_color = "#9ECDF2"
    left_line_color = "#4C78A8"
    right_line_color = "#E45756"
    bp_color = "#6E6E6E"
    ax.scatter(
        x,
        y,
        s=28,
        alpha=0.55,
        c=scatter_color,
        edgecolors="none",
        rasterized=True
    )
    x1_line = np.linspace(x1.min(), x1.max(), 100)
    y1_fit = a + b * x1_line
    ax.plot(
        x1_line,
        y1_fit,
        linewidth=2.0,
        color=left_line_color
    )
    x2_line = np.linspace(x2.min(), x2.max(), 100)
    y2_fit = a + b * x2_line + c * (x2_line - best_bp)
    ax.plot(
        x2_line,
        y2_fit,
        linewidth=2.0,
        color=right_line_color
    )
    ax.axvline(
        best_bp,
        linestyle="--",
        linewidth=1.2,
        color=bp_color
    )
    mid1 = np.median(x1)
    mid2 = np.median(x2)
    y_mid1 = a + b * mid1
    y_mid2 = a + b * mid2 + c * (mid2 - best_bp)
    y_range = y.max() - y.min() if y.max() > y.min() else 1.0
    text_offset = 0.035 * y_range
    ax.text(
        mid1,
        y_mid1 + text_offset,
        f"p = {p1:.2e}\nR² = {r2_1:.3f}",
        fontsize=9.5,
        ha="center",
        va="bottom",
        color="black"
    )
    ax.text(
        mid2,
        y_mid2 + text_offset,
        f"p = {p2:.2e}\nR² = {r2_2:.3f}",
        fontsize=9.5,
        ha="center",
        va="bottom",
        color="black"
    )
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("Pseudotime")
    ax.set_ylabel(ycol)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)
    ax.set_xlim(x.min() - 0.03 * (x.max() - x.min()),
                x.max() + 0.03 * (x.max() - x.min()))
    ax.set_ylim(y.min() - 0.07 * y_range,
                y.max() + 0.12 * y_range)
    plt.tight_layout()
    return fig
