"""Phase 6 Step 8: Generate figures for the paper.

Updated to include red flags analysis results:
1. Forest plot: coefficient estimates ± CI per model (Fig 1)
2. S-index bar chart: S_original vs S_noO side-by-side (Fig 2) — UPDATED
3. S-index × temperature trajectory with S_noO (Fig 3) — UPDATED
4. Prompt ablation comparison (Fig 4)
5. V4 rationality check bar chart (supplementary Fig 5)
6. R² decomposition: O contribution to variance (Fig 6) — NEW
7. Income × Involvement: per-scenario β_Inc_L (Fig 7) — NEW
8. B5 × B5 interaction heatmap (Fig 8) — NEW

Output: figures/*.pdf

Usage:
    python scripts/05_analysis/generate_figures.py
"""

import pathlib

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

# Color palette for models
MODEL_COLORS = {
    "phi4": "#1f77b4",
    "mistral-small3.2": "#ff7f0e",
    "gemma3-27b": "#2ca02c",
    "qwen3-32b": "#d62728",
}

MODEL_ORDER = ["phi4", "mistral-small3.2", "gemma3-27b", "qwen3-32b"]

# Predictor display names
PREDICTOR_LABELS = {
    "Inc_L": "Income (linear)",
    "Inc_Q": "Income (quadratic)",
    "Age_L": "Age",
    "Gender_code": "Gender",
    "Region_code": "Region",
    "O": "Openness",
    "C": "Conscientiousness",
    "E": "Extraversion",
    "A": "Agreeableness",
    "N": "Neuroticism",
}

PREDICTOR_ORDER = [
    "Inc_L", "Inc_Q", "Age_L", "Gender_code", "Region_code",
    "O", "C", "E", "A", "N",
]

# Matplotlib defaults for paper
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


def fig1_forest_plot(df_coefs: pd.DataFrame):
    """Forest plot: coefficient estimates ± 95% CI per model at temp=1.0."""
    df = df_coefs[df_coefs["temperature"] == 1.0].copy()
    if len(df) == 0:
        print("  SKIP fig1: no temp=1.0 data")
        return

    models = sorted(df["model"].unique())
    n_models = len(models)
    n_preds = len(PREDICTOR_ORDER)

    fig, ax = plt.subplots(figsize=(8, 6))

    y_positions = np.arange(n_preds)
    bar_width = 0.8 / n_models

    for i, model_name in enumerate(models):
        df_m = df[df["model"] == model_name]
        color = MODEL_COLORS.get(model_name, f"C{i}")

        betas = []
        ci_lows = []
        ci_highs = []
        sigs = []

        for pred in PREDICTOR_ORDER:
            row = df_m[df_m["predictor"] == pred]
            if len(row) > 0:
                betas.append(row["beta"].iloc[0])
                ci_lows.append(row["ci_low"].iloc[0])
                ci_highs.append(row["ci_high"].iloc[0])
                sigs.append(row["sig_fdr"].iloc[0] if "sig_fdr" in row.columns else False)
            else:
                betas.append(np.nan)
                ci_lows.append(np.nan)
                ci_highs.append(np.nan)
                sigs.append(False)

        betas = np.array(betas)
        ci_lows = np.array(ci_lows)
        ci_highs = np.array(ci_highs)

        y_pos = y_positions + i * bar_width - (n_models - 1) * bar_width / 2

        for j in range(n_preds):
            if np.isnan(betas[j]):
                continue
            marker = "D" if sigs[j] else "o"
            markersize = 7 if sigs[j] else 5
            ax.errorbar(
                betas[j], y_pos[j],
                xerr=[[betas[j] - ci_lows[j]], [ci_highs[j] - betas[j]]],
                fmt=marker, color=color, markersize=markersize,
                capsize=2, linewidth=1, label=model_name if j == 0 else None,
            )

    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    # Separator between demographics and personality
    ax.axhline(y=4.5, color="gray", linestyle=":", linewidth=0.5, alpha=0.7)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([PREDICTOR_LABELS.get(p, p) for p in PREDICTOR_ORDER])
    ax.set_xlabel("Coefficient (log-odds)")
    ax.set_title("Per-Model Coefficient Estimates (temp = 1.0)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.invert_yaxis()

    ax.text(0.02, 0.02, "Filled diamonds = FDR-significant (q < 0.05)",
            transform=ax.transAxes, fontsize=8, alpha=0.7)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_forest_plot.pdf")
    plt.close(fig)
    print("  Saved: figures/fig1_forest_plot.pdf")


def fig2_s_index_comparison(df_s: pd.DataFrame, df_sens: pd.DataFrame):
    """S-index comparison: S_original vs S_noO side-by-side at temp=1.0.

    This is the key figure showing how the conclusion flips when O is excluded.
    """
    df_orig = df_s[df_s["temperature"] == 1.0].copy()
    df_no_o = df_sens[df_sens["temperature"] == 1.0].copy()

    if len(df_orig) == 0 or len(df_no_o) == 0:
        print("  SKIP fig2: missing data")
        return

    # Merge on model
    merged = df_orig[["model", "s_index"]].merge(
        df_no_o[["model", "s_noO"]], on="model"
    )
    # Sort by model order
    merged["sort_key"] = merged["model"].map(
        {m: i for i, m in enumerate(MODEL_ORDER)}
    )
    merged = merged.sort_values("sort_key")

    models = merged["model"].tolist()
    s_orig = merged["s_index"].values
    s_no_o = merged["s_noO"].values

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(models))
    width = 0.35

    colors_orig = [MODEL_COLORS.get(m, "gray") for m in models]
    colors_noo = [MODEL_COLORS.get(m, "gray") for m in models]

    bars1 = ax.bar(x - width/2, s_orig, width,
                   color=colors_orig, alpha=0.85, label="S (with Openness)")
    bars2 = ax.bar(x + width/2, s_no_o, width,
                   color=colors_noo, alpha=0.45, hatch="//",
                   edgecolor=[MODEL_COLORS.get(m, "gray") for m in models],
                   label="S (without Openness)")

    # Reference lines
    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1, alpha=0.7,
               label="S = 1 (balanced)")
    ax.axhspan(1.5, 2.5, alpha=0.08, color="green", label="S_literature (1.5-2.5)")
    ax.axhline(y=2.0, color="green", linestyle="--", linewidth=1, alpha=0.5)

    # Value labels
    for bar, val in zip(bars1, s_orig):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.08,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for bar, val in zip(bars2, s_no_o):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.08,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("S-Index (log scale)")
    ax.set_title("Stereotype Index: With vs. Without Openness (temp = 1.0)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # Use log scale to show both S<1 and S>1 fairly — S=1 at center
    ax.set_yscale("log")
    ax.set_ylim(0.3, max(s_no_o) * 1.5)
    # Add minor gridlines for readability
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.grid(axis="y", alpha=0.3, which="both")

    # Annotation
    ax.annotate("Key: S = 1 means income = personality.\n"
                "S < 1 → personality dominates.\n"
                "S > 1 → income dominates.",
                xy=(0.98, 0.95), xycoords="axes fraction",
                fontsize=8, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                          edgecolor="orange", alpha=0.9))

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_s_index.pdf")
    plt.close(fig)
    print("  Saved: figures/fig2_s_index.pdf")


def fig3_s_index_temperature(df_s: pd.DataFrame, df_sens: pd.DataFrame):
    """S-index × temperature trajectory per model, with S_noO panel."""
    if df_s["temperature"].nunique() < 2:
        print("  SKIP fig3: only 1 temperature")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    # Left panel: S_original
    for model_name in MODEL_ORDER:
        df_m = df_s[df_s["model"] == model_name].sort_values("temperature")
        if len(df_m) == 0:
            continue
        color = MODEL_COLORS.get(model_name, "gray")
        ax1.plot(df_m["temperature"], df_m["s_index"],
                 marker="o", color=color, label=model_name, linewidth=2, markersize=6)

    ax1.axhline(y=2.0, color="green", linestyle="--", linewidth=1, alpha=0.5,
                label="S_literature = 2.0")
    ax1.axhline(y=1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax1.set_xlabel("Temperature")
    ax1.set_ylabel("S-Index (with Openness)")
    ax1.set_title("S_original Across Temperatures")
    ax1.set_xticks([0.0, 0.5, 1.0])
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Right panel: S_noO
    for model_name in MODEL_ORDER:
        df_m = df_sens[df_sens["model"] == model_name].sort_values("temperature")
        if len(df_m) == 0:
            continue
        color = MODEL_COLORS.get(model_name, "gray")
        ax2.plot(df_m["temperature"], df_m["s_noO"],
                 marker="s", color=color, label=model_name, linewidth=2, markersize=6,
                 linestyle="--")

    ax2.axhline(y=2.0, color="green", linestyle="--", linewidth=1, alpha=0.5,
                label="S_literature = 2.0")
    ax2.axhline(y=1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax2.set_xlabel("Temperature")
    ax2.set_ylabel("S-Index (without Openness)")
    ax2.set_title("S_noO Across Temperatures")
    ax2.set_xticks([0.0, 0.5, 1.0])
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_s_temperature.pdf")
    plt.close(fig)
    print("  Saved: figures/fig3_s_temperature.pdf")


def fig4_ablation_comparison(df_ablation: pd.DataFrame):
    """Prompt ablation: coefficient comparison across conditions for key predictors."""
    if len(df_ablation) == 0:
        print("  SKIP fig4: no ablation data")
        return

    models = sorted(df_ablation["model"].unique())
    key_preds = ["Inc_L", "O", "C", "E", "A", "N"]

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    condition_colors = {
        "A_nl_consumer": "#1f77b4",
        "B_structured": "#ff7f0e",
        "C_coded": "#2ca02c",
        "D_general_psych": "#d62728",
    }
    condition_labels = {
        "A_nl_consumer": "A: NL consumer",
        "B_structured": "B: Structured",
        "C_coded": "C: Coded",
        "D_general_psych": "D: General psych",
    }

    for ax_idx, model_name in enumerate(models):
        ax = axes[ax_idx]
        df_m = df_ablation[df_ablation["model"] == model_name]
        conditions = sorted(df_m["condition"].unique())

        y_pos = np.arange(len(key_preds))
        bar_width = 0.8 / len(conditions)

        for i, condition in enumerate(conditions):
            df_c = df_m[df_m["condition"] == condition]
            betas = []
            ci_lows = []
            ci_highs = []

            for pred in key_preds:
                row = df_c[df_c["predictor"] == pred]
                if len(row) > 0:
                    betas.append(row["beta"].iloc[0])
                    ci_lows.append(row["ci_low"].iloc[0])
                    ci_highs.append(row["ci_high"].iloc[0])
                else:
                    betas.append(np.nan)
                    ci_lows.append(np.nan)
                    ci_highs.append(np.nan)

            betas = np.array(betas)
            ci_lows = np.array(ci_lows)
            ci_highs = np.array(ci_highs)

            y = y_pos + i * bar_width - (len(conditions) - 1) * bar_width / 2
            color = condition_colors.get(condition, f"C{i}")

            for j in range(len(key_preds)):
                if np.isnan(betas[j]):
                    continue
                ax.errorbar(
                    betas[j], y[j],
                    xerr=[[betas[j] - ci_lows[j]], [ci_highs[j] - betas[j]]],
                    fmt="o", color=color, markersize=5, capsize=2, linewidth=1,
                    label=condition_labels.get(condition, condition) if j == 0 else None,
                )

        ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([PREDICTOR_LABELS.get(p, p) for p in key_preds])
        ax.set_xlabel("Coefficient (log-odds)")
        ax.set_title(model_name)
        ax.legend(loc="upper right", fontsize=7)
        ax.invert_yaxis()

    fig.suptitle("Prompt Ablation: Coefficient Comparison", y=1.02, fontsize=13)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_ablation.pdf")
    plt.close(fig)
    print("  Saved: figures/fig4_ablation.pdf")


def fig5_v4_check(df: pd.DataFrame):
    """V4 rationality check: pass rate per model per scenario (supplementary)."""
    df_v4 = df[
        (df["is_v4"] == 1) &
        (df["temperature"] == 1.0) &
        (df["dir_condition"] == "A_nl_consumer")
    ].copy()
    if len(df_v4) == 0:
        print("  SKIP fig5: no V4 data")
        return

    v4_summary = df_v4.groupby(["dir_model", "scenario_id"]).agg(
        pass_rate=("Choice_A", "mean"),
        n=("Choice_A", "count"),
    ).reset_index()

    models = sorted(v4_summary["dir_model"].unique())
    scenarios = sorted(v4_summary["scenario_id"].unique())

    fig, ax = plt.subplots(figsize=(8, 4))

    x = np.arange(len(scenarios))
    bar_width = 0.8 / len(models)

    for i, model_name in enumerate(models):
        df_m = v4_summary[v4_summary["dir_model"] == model_name]
        rates = []
        for sc in scenarios:
            row = df_m[df_m["scenario_id"] == sc]
            rates.append(row["pass_rate"].iloc[0] if len(row) > 0 else 0)

        color = MODEL_COLORS.get(model_name, f"C{i}")
        offset = i * bar_width - (len(models) - 1) * bar_width / 2
        ax.bar(x + offset, rates, bar_width, label=model_name, color=color, alpha=0.85)

    ax.axhline(y=0.80, color="red", linestyle="--", linewidth=1, alpha=0.7,
               label="80% threshold")

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("V4 Pass Rate (chose A)")
    ax.set_title("V4 Dominance Check: Rationality Pass Rate")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_v4_check.pdf")
    plt.close(fig)
    print("  Saved: figures/fig5_v4_check.pdf")


def fig6_r2_decomposition():
    """R² decomposition showing O's dominance of explained variance.

    Data from red_flags_report.txt (hardcoded from analysis results).
    """
    # R² values from red flags analysis
    models = ["phi4", "mistral-small3.2", "gemma3-27b", "qwen3-32b"]
    r2_data = {
        "phi4": {
            "Full model": 0.5111, "Without O": 0.1833, "O alone": 0.4216,
            "Income (L+Q)": 0.1723, "B5 (all 5)": 0.4360, "B5 (no O)": 0.1416,
        },
        "mistral-small3.2": {
            "Full model": 0.5467, "Without O": 0.2163, "O alone": 0.4164,
            "Income (L+Q)": 0.1976, "B5 (all 5)": 0.4411, "B5 (no O)": 0.1600,
        },
        "gemma3-27b": {
            "Full model": 0.5050, "Without O": 0.1699, "O alone": 0.3599,
            "Income (L+Q)": 0.1564, "B5 (all 5)": 0.3722, "B5 (no O)": 0.0964,
        },
        "qwen3-32b": {
            "Full model": 0.5685, "Without O": 0.2810, "O alone": 0.4069,
            "Income (L+Q)": 0.2598, "B5 (all 5)": 0.4261, "B5 (no O)": 0.1932,
        },
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left panel: Stacked bar — O contribution vs rest
    x = np.arange(len(models))
    width = 0.5

    r2_without_o = [r2_data[m]["Without O"] for m in models]
    r2_o_contrib = [r2_data[m]["Full model"] - r2_data[m]["Without O"] for m in models]
    colors_m = [MODEL_COLORS[m] for m in models]

    bars_rest = ax1.bar(x, r2_without_o, width, color=colors_m, alpha=0.4,
                        label="Other 9 predictors")
    bars_o = ax1.bar(x, r2_o_contrib, width, bottom=r2_without_o,
                     color=colors_m, alpha=0.85, label="Openness contribution")

    # Percentage labels
    for i, m in enumerate(models):
        total = r2_data[m]["Full model"]
        o_pct = r2_o_contrib[i] / total * 100
        ax1.text(i, total + 0.015, f"{o_pct:.0f}%",
                 ha="center", fontsize=9, fontweight="bold")
        ax1.text(i, total + 0.04, f"R²={total:.3f}",
                 ha="center", fontsize=8, alpha=0.7)

    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=9)
    ax1.set_ylabel("McFadden pseudo-R²")
    ax1.set_title("R² Decomposition: Openness Contribution")
    ax1.set_ylim(0, 0.7)

    # Custom legend
    patch_o = mpatches.Patch(color="gray", alpha=0.85, label="Openness alone")
    patch_rest = mpatches.Patch(color="gray", alpha=0.4, label="Other 9 predictors")
    ax1.legend(handles=[patch_o, patch_rest], loc="upper left", fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # Right panel: Grouped bars — comparing predictor groups
    groups = ["O alone", "Income (L+Q)", "B5 (no O)"]
    group_colors = ["#e74c3c", "#3498db", "#95a5a6"]
    n_groups = len(groups)
    bar_w = 0.8 / n_groups

    for i, group in enumerate(groups):
        vals = [r2_data[m][group] for m in models]
        offset = i * bar_w - (n_groups - 1) * bar_w / 2
        bars = ax2.bar(x + offset, vals, bar_w, color=group_colors[i],
                       alpha=0.85, label=group)
        for j, val in enumerate(vals):
            ax2.text(x[j] + offset, val + 0.008, f"{val:.2f}",
                     ha="center", fontsize=7)

    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=9)
    ax2.set_ylabel("McFadden pseudo-R²")
    ax2.set_title("R² by Predictor Group: O vs Income vs B5\\O")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.set_ylim(0, 0.5)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig6_r2_decomposition.pdf")
    plt.close(fig)
    print("  Saved: figures/fig6_r2_decomposition.pdf")


def fig7_income_involvement(df_inc: pd.DataFrame):
    """Income × Involvement: per-scenario β_Inc_L by involvement level.

    Shows whether income sensitivity is calibrated to product price tier.
    """
    # Filter per-scenario rows
    df_scen = df_inc[df_inc["involvement"].notna()].copy()
    if len(df_scen) == 0:
        print("  SKIP fig7: no per-scenario data")
        return

    # Interaction significance
    df_inter = df_inc[df_inc["predictor"] == "Inc_L_x_Involvement"].copy()

    scenario_labels = {
        -2.0: "S5\n(cheapest)",
        -1.0: "S4",
        0.0: "S3\n(mid)",
        1.0: "S1",
        2.0: "S2\n(priciest)",
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name in MODEL_ORDER:
        df_m = df_scen[df_scen["model"] == model_name].sort_values("involvement")
        if len(df_m) == 0:
            continue
        color = MODEL_COLORS.get(model_name, "gray")

        # Check if interaction is significant
        inter_row = df_inter[df_inter["model"] == model_name]
        is_sig = inter_row["sig"].iloc[0] if len(inter_row) > 0 else False
        linestyle = "-" if is_sig else "--"
        marker = "D" if is_sig else "o"
        label_suffix = " *" if is_sig else ""

        ax.plot(df_m["involvement"], df_m["beta"],
                marker=marker, color=color, linewidth=2, markersize=7,
                linestyle=linestyle, label=f"{model_name}{label_suffix}")

        # Error bars (±1 SE)
        ax.fill_between(
            df_m["involvement"],
            df_m["beta"] - df_m["se"],
            df_m["beta"] + df_m["se"],
            color=color, alpha=0.1,
        )

    ax.set_xlabel("Involvement Level (scenario price tier)")
    ax.set_ylabel("β_Income (log-odds)")
    ax.set_title("Income Sensitivity by Product Involvement Level")

    inv_levels = sorted(df_scen["involvement"].unique())
    ax.set_xticks(inv_levels)
    ax.set_xticklabels([scenario_labels.get(v, f"{v:.0f}") for v in inv_levels])

    ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    ax.text(0.98, 0.02, "* = significant Inc_L × Involvement interaction\n"
            "Shaded bands = ±1 SE (not 95% CI)",
            transform=ax.transAxes, fontsize=8, ha="right", alpha=0.7)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig7_income_involvement.pdf")
    plt.close(fig)
    print("  Saved: figures/fig7_income_involvement.pdf")


def fig8_b5_interactions_heatmap(df_inter: pd.DataFrame):
    """B5 × B5 interaction heatmap: signed β values across models.

    Shows which trait interactions are consistent across models.
    """
    b5_pairs = ["OxC", "OxE", "OxA", "OxN", "CxE", "CxA", "CxN", "ExA", "ExN", "AxN"]
    b5_pair_labels = [
        "O×C", "O×E", "O×A", "O×N",
        "C×E", "C×A", "C×N",
        "E×A", "E×N",
        "A×N",
    ]

    # Prefer FDR-corrected file if available
    fdr_path = RESULTS_DIR / "b5_interactions_per_model_fdr.csv"
    if fdr_path.exists():
        df_inter_fdr = pd.read_csv(fdr_path)
        df_b5 = df_inter_fdr[df_inter_fdr["type"] == "B5xB5"].copy()
        use_fdr = True
    else:
        df_b5 = df_inter[df_inter["type"] == "B5xB5"].copy()
        use_fdr = False

    if len(df_b5) == 0:
        print("  SKIP fig8: no B5×B5 data")
        return

    # Build matrix: rows = pairs, columns = models
    matrix = np.full((len(b5_pairs), len(MODEL_ORDER)), np.nan)
    sig_001 = np.zeros((len(b5_pairs), len(MODEL_ORDER)), dtype=bool)
    sig_01 = np.zeros((len(b5_pairs), len(MODEL_ORDER)), dtype=bool)
    sig_05 = np.zeros((len(b5_pairs), len(MODEL_ORDER)), dtype=bool)

    p_col = "p_fdr" if use_fdr else "p"

    for i, pair in enumerate(b5_pairs):
        for j, model in enumerate(MODEL_ORDER):
            row = df_b5[(df_b5["predictor"] == pair) & (df_b5["model"] == model)]
            if len(row) > 0:
                matrix[i, j] = row["beta"].iloc[0]
                pval = row[p_col].iloc[0]
                sig_001[i, j] = pval < 0.001
                sig_01[i, j] = pval < 0.01
                sig_05[i, j] = pval < 0.05

    fig, ax = plt.subplots(figsize=(8, 7))

    # Diverging colormap centered at 0
    vmax = np.nanmax(np.abs(matrix))
    im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")

    # Add text annotations with standard significance notation
    for i in range(len(b5_pairs)):
        for j in range(len(MODEL_ORDER)):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            if sig_001[i, j]:
                sig_star = "***"
            elif sig_01[i, j]:
                sig_star = "**"
            elif sig_05[i, j]:
                sig_star = "*"
            else:
                sig_star = ""
            text_color = "white" if abs(val) > vmax * 0.6 else "black"
            is_sig = sig_05[i, j]
            ax.text(j, i, f"{val:.2f}{sig_star}",
                    ha="center", va="center", fontsize=7,
                    color=text_color, fontweight="bold" if is_sig else "normal")

    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels(MODEL_ORDER, fontsize=9, rotation=20, ha="right")
    ax.set_yticks(range(len(b5_pairs)))
    ax.set_yticklabels(b5_pair_labels, fontsize=9)
    p_label = "FDR-corrected" if use_fdr else "uncorrected"
    ax.set_title(f"Big Five Trait Interactions (β, temp = 1.0, {p_label} p)")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, label="β (log-odds)")

    ax.text(0.5, -0.12, f"* p < 0.05, ** p < 0.01, *** p < 0.001 ({p_label})",
            transform=ax.transAxes, fontsize=8, ha="center", alpha=0.7)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig8_b5_interactions.pdf")
    plt.close(fig)
    print("  Saved: figures/fig8_b5_interactions.pdf")


def main():
    print("=" * 60)
    print("Phase 6 Step 8: Generate Figures (Updated with Red Flags)")
    print("=" * 60)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data files
    coefs_path = RESULTS_DIR / "per_model_coefficients.csv"
    s_index_path = RESULTS_DIR / "s_index.csv"
    s_sens_path = RESULTS_DIR / "s_index_sensitivity.csv"
    ablation_path = RESULTS_DIR / "ablation_effects.csv"
    analysis_path = RESULTS_DIR / "analysis_df.parquet"
    inc_inv_path = RESULTS_DIR / "income_involvement_interaction.csv"
    b5_inter_path = RESULTS_DIR / "b5_interactions_per_model.csv"

    # Fig 1: Forest plot
    print("\nFig 1: Forest plot...")
    if coefs_path.exists():
        fig1_forest_plot(pd.read_csv(coefs_path))
    else:
        print("  SKIP: per_model_coefficients.csv not found")

    # Fig 2: S-index comparison (S_original vs S_noO)
    print("\nFig 2: S-index comparison (with/without O)...")
    if s_index_path.exists() and s_sens_path.exists():
        fig2_s_index_comparison(pd.read_csv(s_index_path), pd.read_csv(s_sens_path))
    else:
        print("  SKIP: s_index.csv or s_index_sensitivity.csv not found")

    # Fig 3: S-index temperature trajectory (with S_noO panel)
    print("\nFig 3: S-index x temperature (dual panel)...")
    if s_index_path.exists() and s_sens_path.exists():
        fig3_s_index_temperature(pd.read_csv(s_index_path), pd.read_csv(s_sens_path))
    else:
        print("  SKIP: s_index.csv or s_index_sensitivity.csv not found")

    # Fig 4: Ablation comparison
    print("\nFig 4: Prompt ablation...")
    if ablation_path.exists():
        fig4_ablation_comparison(pd.read_csv(ablation_path))
    else:
        print("  SKIP: ablation_effects.csv not found")

    # Fig 5: V4 check (supplementary)
    print("\nFig 5: V4 rationality check...")
    if analysis_path.exists():
        fig5_v4_check(pd.read_parquet(analysis_path))
    else:
        print("  SKIP: analysis_df.parquet not found")

    # Fig 6: R² decomposition (NEW)
    print("\nFig 6: R² decomposition...")
    fig6_r2_decomposition()

    # Fig 7: Income × Involvement (NEW)
    print("\nFig 7: Income × Involvement...")
    if inc_inv_path.exists():
        fig7_income_involvement(pd.read_csv(inc_inv_path))
    else:
        print("  SKIP: income_involvement_interaction.csv not found")

    # Fig 8: B5 × B5 interaction heatmap (NEW)
    print("\nFig 8: B5 × B5 interactions...")
    if b5_inter_path.exists():
        fig8_b5_interactions_heatmap(pd.read_csv(b5_inter_path))
    else:
        print("  SKIP: b5_interactions_per_model.csv not found")

    print(f"\nDone. 8 figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
