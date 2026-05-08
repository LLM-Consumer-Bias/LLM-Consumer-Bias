"""Red flags analysis: 5 critical checks before paper writing.

1. Positional bias analysis (§14.6)
2. Income × Involvement interaction (§14.9)
3. S-index sensitivity without O
4. Sensitivity analysis excluding gemma3 (V4=60%)
5. Per-model B5 × Region interactions (replaces failed M4)

Output: results/red_flags_report.txt + individual CSV files

Usage:
    python scripts/05_analysis/red_flags_analysis.py
"""

import itertools
import pathlib
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.stats.multitest import multipletests

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"

PREDICTORS_OF_INTEREST = [
    "Inc_L", "Inc_Q", "Age_L", "Gender_code", "Region_code",
    "O", "C", "E", "A", "N",
]
NUISANCE = ["Order", "Format_1", "Format_2"]
B5_TRAITS = ["O", "C", "E", "A", "N"]

report_lines = []


def report(msg=""):
    print(msg)
    report_lines.append(msg)


def get_primary_data(df):
    """Filter to primary analysis: condition A, non-V4, temp=1.0."""
    mask = (
        (df["dir_condition"] == "A_nl_consumer") &
        (df["is_v4"] == 0) &
        (df["temperature"] == 1.0)
    )
    return df[mask].copy()


def fit_glm_clustered(X, y, groups):
    """Fit GLM binomial with clustered SEs. Returns result or None."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.GLM(y, X, family=sm.families.Binomial())
            return model.fit(
                cov_type="cluster",
                cov_kwds={"groups": groups},
            )
    except Exception as e:
        report(f"    GLM ERROR: {e}")
        return None


# =====================================================================
# RED FLAG 1: Positional Bias Analysis (§14.6)
# =====================================================================
def analysis_1_positional_bias(df):
    report("=" * 70)
    report("RED FLAG 1: POSITIONAL BIAS ANALYSIS (§14.6)")
    report("=" * 70)
    report("Testing whether trait position (0-4) in the prompt affects choice.")
    report()

    df_primary = get_primary_data(df)
    models = sorted(df_primary["dir_model"].unique())

    pos_cols = [f"trait_pos_{t}" for t in B5_TRAITS]
    all_results = []

    for model_name in models:
        df_m = df_primary[df_primary["dir_model"] == model_name].copy()
        df_m = df_m.reset_index(drop=True)

        # Scenario dummies
        scen_dum = pd.get_dummies(df_m["scenario_id"], prefix="Scenario",
                                   drop_first=True, dtype=int).reset_index(drop=True)
        scen_cols = list(scen_dum.columns)

        X = pd.concat([df_m[pos_cols + NUISANCE], scen_dum], axis=1).astype(float)
        X = sm.add_constant(X)
        y = df_m["Choice_A"].astype(float)

        result = fit_glm_clustered(X, y, df_m["ProfileID"].values)
        if result is None:
            continue

        report(f"  {model_name} (N={len(df_m):,}):")
        for pc in pos_cols:
            if pc in result.params.index:
                b = result.params[pc]
                p = result.pvalues[pc]
                sig = "*" if p < 0.05 else " "
                report(f"    {pc:20s}  β={b:+.6f}  p={p:.4f} {sig}")
                all_results.append({
                    "model": model_name, "predictor": pc,
                    "beta": b, "se": result.bse[pc],
                    "p": p, "sig": p < 0.05,
                })

    df_pos = pd.DataFrame(all_results)
    if len(df_pos) > 0:
        df_pos.to_csv(RESULTS_DIR / "positional_bias.csv", index=False)
        n_sig = df_pos["sig"].sum()
        n_total = len(df_pos)
        report(f"\n  Summary: {n_sig}/{n_total} positional effects significant at p<0.05")
        report(f"  Max |β| = {df_pos['beta'].abs().max():.6f}")

        # If any significant, run sensitivity check with positional covariates
        if n_sig > 0:
            report("\n  → Significant positional effects found. Running sensitivity check...")
            analysis_1b_sensitivity(df)
    else:
        report("  No results produced.")

    report()
    return df_pos


def analysis_1b_sensitivity(df):
    """Re-run per-model with positional covariates to check robustness."""
    report("\n  SENSITIVITY: Per-model GLM WITH positional covariates")
    df_primary = get_primary_data(df)
    models = sorted(df_primary["dir_model"].unique())
    pos_cols = [f"trait_pos_{t}" for t in B5_TRAITS]

    orig_coefs = pd.read_csv(RESULTS_DIR / "per_model_coefficients.csv")

    comparison_rows = []

    for model_name in models:
        df_m = df_primary[df_primary["dir_model"] == model_name].copy()
        df_m = df_m.reset_index(drop=True)

        scen_dum = pd.get_dummies(df_m["scenario_id"], prefix="Scenario",
                                   drop_first=True, dtype=int).reset_index(drop=True)

        X = pd.concat([
            df_m[PREDICTORS_OF_INTEREST + NUISANCE + pos_cols],
            scen_dum
        ], axis=1).astype(float)
        X = sm.add_constant(X)
        y = df_m["Choice_A"].astype(float)

        result = fit_glm_clustered(X, y, df_m["ProfileID"].values)
        if result is None:
            continue

        # Compare betas with original (without positional covariates)
        orig_m = orig_coefs[
            (orig_coefs["model"] == model_name) &
            (orig_coefs["temperature"] == 1.0)
        ]

        report(f"\n  {model_name}: original β vs adjusted β")
        for pred in PREDICTORS_OF_INTEREST:
            if pred in result.params.index:
                beta_adj = result.params[pred]
                orig_row = orig_m[orig_m["predictor"] == pred]
                if len(orig_row) > 0:
                    beta_orig = orig_row["beta"].iloc[0]
                    change_pct = ((beta_adj - beta_orig) / abs(beta_orig)) * 100
                    report(f"    {pred:15s}  orig={beta_orig:+.4f}  "
                           f"adj={beta_adj:+.4f}  Δ={change_pct:+.1f}%")
                    comparison_rows.append({
                        "model": model_name, "predictor": pred,
                        "beta_original": beta_orig, "beta_adjusted": beta_adj,
                        "change_pct": change_pct,
                    })

    if comparison_rows:
        df_comp = pd.DataFrame(comparison_rows)
        df_comp.to_csv(RESULTS_DIR / "positional_bias_sensitivity.csv", index=False)
        max_change = df_comp["change_pct"].abs().max()
        report(f"\n  Max coefficient change: {max_change:.1f}%")
        if max_change < 5:
            report("  → CONCLUSION: Positional bias negligible (all changes < 5%)")
        else:
            report(f"  → WARNING: Some coefficients changed by >{max_change:.0f}%")


# =====================================================================
# RED FLAG 2: Income × Involvement Interaction (§14.9)
# =====================================================================
def analysis_2_income_involvement(df):
    report("=" * 70)
    report("RED FLAG 2: INCOME × INVOLVEMENT INTERACTION (§14.9)")
    report("=" * 70)
    report("Testing whether income effect scales with product price tier.")
    report("If significant → calibrated sensitivity. If n.s. → blanket heuristic.")
    report()

    df_primary = get_primary_data(df)
    models = sorted(df_primary["dir_model"].unique())
    all_results = []

    for model_name in models:
        df_m = df_primary[df_primary["dir_model"] == model_name].copy()
        df_m = df_m.reset_index(drop=True)

        # Interaction term
        df_m["Inc_L_x_Involvement"] = df_m["Inc_L"] * df_m["Involvement_linear"]

        preds = PREDICTORS_OF_INTEREST + ["Involvement_linear", "Inc_L_x_Involvement"] + NUISANCE
        X = df_m[preds].astype(float)
        X = sm.add_constant(X)
        y = df_m["Choice_A"].astype(float)

        result = fit_glm_clustered(X, y, df_m["ProfileID"].values)
        if result is None:
            continue

        report(f"  {model_name} (N={len(df_m):,}):")

        for key_pred in ["Inc_L", "Involvement_linear", "Inc_L_x_Involvement"]:
            if key_pred in result.params.index:
                b = result.params[key_pred]
                p = result.pvalues[key_pred]
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
                report(f"    {key_pred:30s}  β={b:+.4f}  SE={result.bse[key_pred]:.4f}  p={p:.4f}  {sig}")
                all_results.append({
                    "model": model_name, "predictor": key_pred,
                    "beta": b, "se": result.bse[key_pred],
                    "p": p, "sig": p < 0.05,
                })

        # Per-scenario Income effect for detailed view
        report(f"    Per-scenario Income β:")
        for scenario in sorted(df_m["scenario_id"].unique()):
            df_s = df_m[df_m["scenario_id"] == scenario]
            if len(df_s) < 50:
                continue
            X_s = df_s[PREDICTORS_OF_INTEREST + NUISANCE].astype(float)
            X_s = sm.add_constant(X_s)
            y_s = df_s["Choice_A"].astype(float)
            res_s = fit_glm_clustered(X_s, y_s, df_s["ProfileID"].values)
            if res_s and "Inc_L" in res_s.params.index:
                invl = df_s["Involvement_linear"].iloc[0]
                report(f"      {scenario} (Involvement={invl:+d}):  "
                       f"β_Inc_L = {res_s.params['Inc_L']:+.4f}  "
                       f"p = {res_s.pvalues['Inc_L']:.4f}")
                all_results.append({
                    "model": model_name, "predictor": f"Inc_L_scenario_{scenario}",
                    "beta": res_s.params["Inc_L"], "se": res_s.bse["Inc_L"],
                    "p": res_s.pvalues["Inc_L"],
                    "involvement": invl,
                })

        report()

    df_inv = pd.DataFrame(all_results)
    if len(df_inv) > 0:
        df_inv.to_csv(RESULTS_DIR / "income_involvement_interaction.csv", index=False)

    # Summary
    interaction_rows = df_inv[df_inv["predictor"] == "Inc_L_x_Involvement"]
    if len(interaction_rows) > 0:
        n_sig = int(interaction_rows["sig"].sum())
        report(f"  Summary: {n_sig}/{len(interaction_rows)} models show significant "
               f"Inc_L × Involvement interaction")
        if n_sig == 0:
            report("  → CONCLUSION: Income effect does NOT scale with price tier → BLANKET HEURISTIC")
        elif n_sig == len(interaction_rows):
            report("  → CONCLUSION: Income effect scales with price tier → CALIBRATED SENSITIVITY")
        else:
            report("  → CONCLUSION: Mixed — some models calibrated, some heuristic")

    report()
    return df_inv


# =====================================================================
# RED FLAG 3: S-Index Sensitivity Without O
# =====================================================================
def analysis_3_s_index_without_o(df):
    report("=" * 70)
    report("RED FLAG 3: S-INDEX SENSITIVITY WITHOUT OPENNESS")
    report("=" * 70)
    report("O descriptions are semantically confounded with the choice task.")
    report("Computing S_noO = |β_Inc_L| / max(|β_C|, |β_E|, |β_A|, |β_N|)")
    report()

    coefs = pd.read_csv(RESULTS_DIR / "per_model_coefficients.csv")
    s_orig = pd.read_csv(RESULTS_DIR / "s_index.csv")

    results = []

    for (model_name, temp), group in coefs.groupby(["model", "temperature"]):
        inc_row = group[group["predictor"] == "Inc_L"]
        if len(inc_row) == 0:
            continue
        beta_inc = abs(inc_row["beta"].iloc[0])

        # Original S (with O)
        orig_row = s_orig[
            (s_orig["model"] == model_name) &
            (s_orig["temperature"] == temp)
        ]
        s_original = orig_row["s_index"].iloc[0] if len(orig_row) > 0 else np.nan

        # S without O
        traits_noO = {}
        for t in ["C", "E", "A", "N"]:
            t_row = group[group["predictor"] == t]
            if len(t_row) > 0:
                traits_noO[t] = abs(t_row["beta"].iloc[0])

        if traits_noO:
            max_noO = max(traits_noO.values())
            max_noO_trait = max(traits_noO, key=traits_noO.get)
            s_noO = beta_inc / max_noO if max_noO > 0 else np.inf
        else:
            s_noO = np.nan
            max_noO = np.nan
            max_noO_trait = "N/A"

        # Also compute S using only non-instruction-like traits (C, N)
        # E and A also have semantic overlap; C and N are most "neutral"
        traits_neutral = {}
        for t in ["C", "N"]:
            t_row = group[group["predictor"] == t]
            if len(t_row) > 0:
                traits_neutral[t] = abs(t_row["beta"].iloc[0])

        if traits_neutral:
            max_neutral = max(traits_neutral.values())
            s_neutral = beta_inc / max_neutral if max_neutral > 0 else np.inf
        else:
            s_neutral = np.nan

        results.append({
            "model": model_name, "temperature": temp,
            "abs_beta_Inc_L": beta_inc,
            "s_original": s_original,
            "max_abs_beta_noO": max_noO,
            "max_trait_noO": max_noO_trait,
            "s_noO": s_noO,
            "s_noO_anchored_2.0": s_noO / 2.0 if not np.isnan(s_noO) else np.nan,
            "s_neutral_CN": s_neutral,
        })

    df_s = pd.DataFrame(results)
    df_s.to_csv(RESULTS_DIR / "s_index_sensitivity.csv", index=False)

    # Print comparison at temp=1.0
    df_t1 = df_s[df_s["temperature"] == 1.0].sort_values("model")
    report(f"  {'Model':<20s}  {'S_orig':>7s}  {'S_noO':>7s}  {'S_neutral(C,N)':>14s}  {'max_noO_trait':>12s}")
    report(f"  {'-'*20}  {'-'*7}  {'-'*7}  {'-'*14}  {'-'*12}")
    for _, row in df_t1.iterrows():
        report(f"  {row['model']:<20s}  {row['s_original']:7.3f}  {row['s_noO']:7.3f}  "
               f"{row['s_neutral_CN']:14.3f}  {row['max_trait_noO']:>12s}")

    report()
    report(f"  S < 1.0 → personality dominates income")
    report(f"  S > 1.0 → income dominates personality")
    report(f"  S_literature ≈ 2.0 (human benchmark)")
    report()

    s_noO_mean = df_t1["s_noO"].mean()
    s_orig_mean = df_t1["s_original"].mean()
    report(f"  Mean S_original (with O):    {s_orig_mean:.3f}  → personality dominates")
    report(f"  Mean S_noO (without O):      {s_noO_mean:.3f}  → {'income dominates' if s_noO_mean > 1 else 'personality still dominates'}")

    if s_noO_mean > 1.0 and s_orig_mean < 1.0:
        report(f"\n  ⚠ CRITICAL: CONCLUSION FLIPS when O is excluded!")
        report(f"  With O: S < 1 → personality > income")
        report(f"  Without O: S > 1 → income > personality (excl. O)")
        report(f"  → The 'personality dominance' finding depends entirely on O")
    elif s_noO_mean > 1.5:
        report(f"\n  ⚠ Without O, S approaches literature benchmark → more nuanced picture")

    report()
    return df_s


# =====================================================================
# RED FLAG 4: Sensitivity Analysis Excluding gemma3
# =====================================================================
def analysis_4_exclude_gemma3(df):
    report("=" * 70)
    report("RED FLAG 4: SENSITIVITY ANALYSIS EXCLUDING gemma3-27b (V4=60%)")
    report("=" * 70)
    report("gemma3 fails V4 dominance check (60%). Re-run key analyses without it.")
    report()

    # Filter out gemma3
    df_no_gemma = df[df["dir_model"] != "gemma3-27b"].copy()
    df_primary = get_primary_data(df_no_gemma)

    report(f"  Data: {len(df_primary):,} rows (3 models, excl. gemma3)")
    report()

    # 4a. S-index comparison
    report("  4a. S-index without gemma3:")
    coefs = pd.read_csv(RESULTS_DIR / "per_model_coefficients.csv")
    coefs_no_g = coefs[coefs["model"] != "gemma3-27b"]
    s_orig = pd.read_csv(RESULTS_DIR / "s_index.csv")

    s_no_g = s_orig[
        (s_orig["model"] != "gemma3-27b") &
        (s_orig["temperature"] == 1.0)
    ]
    for _, row in s_no_g.iterrows():
        report(f"    {row['model']:<20s}  S={row['s_index']:.3f}  archetype={row['archetype']}")

    report(f"    Mean S (3 models): {s_no_g['s_index'].mean():.3f}")
    report(f"    Range: {s_no_g['s_index'].min():.3f} – {s_no_g['s_index'].max():.3f}")
    report()

    # 4b. GEE without gemma3
    report("  4b. GEE M0-M3 without gemma3:")
    mask = (
        (df_no_gemma["dir_condition"] == "A_nl_consumer") &
        (df_no_gemma["is_v4"] == 0) &
        (df_no_gemma["temperature"] == 1.0)
    )
    df_gee = df_no_gemma[mask].copy()

    scen_dum = pd.get_dummies(df_gee["scenario_id"], prefix="Scenario",
                               drop_first=True, dtype=int)
    df_gee = pd.concat([df_gee.reset_index(drop=True), scen_dum.reset_index(drop=True)], axis=1)

    model_dum = pd.get_dummies(df_gee["dir_model"], prefix="Model",
                                drop_first=True, dtype=int)
    df_gee = pd.concat([df_gee, model_dum.reset_index(drop=True)], axis=1)

    df_gee = df_gee.sort_values("ProfileID").reset_index(drop=True)

    scen_cols = [c for c in df_gee.columns if c.startswith("Scenario_")]
    model_cols = [c for c in df_gee.columns if c.startswith("Model_")]
    DEMOGRAPHICS = ["Inc_L", "Inc_Q", "Age_L", "Gender_code", "Region_code"]
    PERSONALITY = ["O", "C", "E", "A", "N"]

    def fit_gee_local(preds, label):
        all_preds = preds + NUISANCE + scen_cols + model_cols
        X = sm.add_constant(df_gee[all_preds].astype(float))
        y = df_gee["Choice_A"].astype(float)
        groups = df_gee["ProfileID"]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gee = GEE(y, X, groups, family=Binomial(), cov_struct=Exchangeable())
                res = gee.fit()
                qic = res.qic()[0]
                return {"label": label, "qic": qic, "result": res}
        except Exception as e:
            return {"label": label, "error": str(e)}

    m0 = fit_gee_local([], "M0")
    m1 = fit_gee_local(DEMOGRAPHICS, "M1")
    m2 = fit_gee_local(PERSONALITY, "M2")
    m3 = fit_gee_local(DEMOGRAPHICS + PERSONALITY, "M3")

    for m in [m0, m1, m2, m3]:
        if "error" in m:
            report(f"    {m['label']}: ERROR — {m['error']}")
        else:
            report(f"    {m['label']}: QIC = {m['qic']:.1f}")

    # Wald test M2 vs M0 (personality)
    if "result" in m2 and "result" in m0:
        r_matrix = np.zeros((len(PERSONALITY), len(m2["result"].params)))
        for i, pred in enumerate(PERSONALITY):
            if pred in m2["result"].params.index:
                j = list(m2["result"].params.index).index(pred)
                r_matrix[i, j] = 1
        try:
            wald = m2["result"].wald_test(r_matrix, scalar=True)
            report(f"    M2 vs M0 (personality): χ²={float(wald.statistic):.1f}, "
                   f"p={float(wald.pvalue):.2e}")
        except Exception:
            pass

    # Wald test M1 vs M0 (demographics)
    if "result" in m1 and "result" in m0:
        r_matrix = np.zeros((len(DEMOGRAPHICS), len(m1["result"].params)))
        for i, pred in enumerate(DEMOGRAPHICS):
            if pred in m1["result"].params.index:
                j = list(m1["result"].params.index).index(pred)
                r_matrix[i, j] = 1
        try:
            wald = m1["result"].wald_test(r_matrix, scalar=True)
            report(f"    M1 vs M0 (demographics): χ²={float(wald.statistic):.1f}, "
                   f"p={float(wald.pvalue):.4f}")
        except Exception:
            pass

    report()
    report("  → Compare with full 4-model results to assess gemma3 impact")
    report()
    return


# =====================================================================
# RED FLAG 5: Per-Model B5 × Region Interactions
# =====================================================================
def analysis_5_b5_interactions(df):
    report("=" * 70)
    report("RED FLAG 5: B5 × REGION INTERACTIONS (Hofstede predictions)")
    report("=" * 70)
    report("M4 GEE failed. Testing B5 × Region per model instead.")
    report("Predictions: E×Reg, N×Reg, O×Reg, C×Reg (from Hofstede theory)")
    report()
    report("NOTE: With I=OCEAN (N=O×C×E×A), some interaction terms")
    report("are aliased with 3-factor interactions. All 2FI are cleanly")
    report("estimable (Resolution V), but N×any = product of 3 other traits.")
    report()

    df_primary = get_primary_data(df)
    models = sorted(df_primary["dir_model"].unique())

    # B5 × Region interactions
    b5_region_ints = [f"{t}_x_Region" for t in B5_TRAITS]
    # Also test B5 × B5 interactions (subset: most theoretically interesting)
    b5_b5_ints_pairs = list(itertools.combinations(B5_TRAITS, 2))

    all_results = []

    for model_name in models:
        df_m = df_primary[df_primary["dir_model"] == model_name].copy()
        df_m = df_m.reset_index(drop=True)

        # Create interaction terms
        for t in B5_TRAITS:
            df_m[f"{t}_x_Region"] = df_m[t] * df_m["Region_code"]
        for t1, t2 in b5_b5_ints_pairs:
            df_m[f"{t1}x{t2}"] = df_m[t1] * df_m[t2]

        interaction_cols = b5_region_ints + [f"{t1}x{t2}" for t1, t2 in b5_b5_ints_pairs]

        # Scenario dummies
        scen_dum = pd.get_dummies(df_m["scenario_id"], prefix="Scenario",
                                   drop_first=True, dtype=int).reset_index(drop=True)

        X = pd.concat([
            df_m[PREDICTORS_OF_INTEREST + interaction_cols + NUISANCE],
            scen_dum
        ], axis=1).astype(float)
        X = sm.add_constant(X)
        y = df_m["Choice_A"].astype(float)

        result = fit_glm_clustered(X, y, df_m["ProfileID"].values)
        if result is None:
            continue

        report(f"  {model_name} (N={len(df_m):,}):")

        # B5 × Region
        report(f"    B5 × Region interactions (Hofstede predictions):")
        for col in b5_region_ints:
            if col in result.params.index:
                b = result.params[col]
                p = result.pvalues[col]
                ci_lo, ci_hi = result.conf_int().loc[col]
                sig = "***" if p < 0.001 else "** " if p < 0.01 else "*  " if p < 0.05 else "n.s."
                report(f"      {col:20s}  β={b:+.4f}  SE={result.bse[col]:.4f}  "
                       f"p={p:.4f}  {sig}")
                all_results.append({
                    "model": model_name, "predictor": col,
                    "type": "B5xRegion",
                    "beta": b, "se": result.bse[col],
                    "p": p, "ci_low": ci_lo, "ci_high": ci_hi,
                })

        # B5 × B5
        report(f"    B5 × B5 interactions:")
        for t1, t2 in b5_b5_ints_pairs:
            col = f"{t1}x{t2}"
            if col in result.params.index:
                b = result.params[col]
                p = result.pvalues[col]
                ci_lo, ci_hi = result.conf_int().loc[col]
                sig = "***" if p < 0.001 else "** " if p < 0.01 else "*  " if p < 0.05 else "n.s."
                report(f"      {col:20s}  β={b:+.4f}  SE={result.bse[col]:.4f}  "
                       f"p={p:.4f}  {sig}")
                all_results.append({
                    "model": model_name, "predictor": col,
                    "type": "B5xB5",
                    "beta": b, "se": result.bse[col],
                    "p": p, "ci_low": ci_lo, "ci_high": ci_hi,
                })

        report()

    df_int = pd.DataFrame(all_results)
    if len(df_int) > 0:
        df_int.to_csv(RESULTS_DIR / "b5_interactions_per_model.csv", index=False)

        # Summary: how many significant?
        report("  SUMMARY OF INTERACTIONS:")
        for itype in ["B5xRegion", "B5xB5"]:
            df_t = df_int[df_int["type"] == itype]
            n_sig = (df_t["p"] < 0.05).sum()
            n_total = len(df_t)
            report(f"    {itype}: {n_sig}/{n_total} significant at p<0.05")

        # Hofstede predictions check
        report("\n  HOFSTEDE PREDICTIONS CHECK:")
        hofstede = {
            "E_x_Region": "E stronger for Japan (collectivist IDV gap)",
            "N_x_Region": "N stronger for Japan (UAI gap)",
            "O_x_Region": "O stronger for USA (inverse UAI gap)",
            "C_x_Region": "C stronger for Japan (LTO gap)",
            "A_x_Region": "A × Region (shinrai culture)",
        }
        for pred, theory in hofstede.items():
            df_p = df_int[df_int["predictor"] == pred]
            if len(df_p) > 0:
                n_sig = (df_p["p"] < 0.05).sum()
                mean_beta = df_p["beta"].mean()
                report(f"    {pred}: {n_sig}/{len(df_p)} models sig, "
                       f"mean β={mean_beta:+.4f}")
                report(f"      Theory: {theory}")

    report()
    return df_int


# =====================================================================
# BONUS: R² decomposition (how much is from O alone?)
# =====================================================================
def analysis_bonus_r2_decomposition(df):
    report("=" * 70)
    report("BONUS: R² DECOMPOSITION — How much variance is from O alone?")
    report("=" * 70)
    report()

    df_primary = get_primary_data(df)
    models = sorted(df_primary["dir_model"].unique())

    for model_name in models:
        df_m = df_primary[df_primary["dir_model"] == model_name].copy()
        df_m = df_m.reset_index(drop=True)

        scen_dum = pd.get_dummies(df_m["scenario_id"], prefix="Scenario",
                                   drop_first=True, dtype=int).reset_index(drop=True)
        scen_cols = list(scen_dum.columns)

        y = df_m["Choice_A"].astype(float)
        groups = df_m["ProfileID"].values

        def get_r2(pred_list, label):
            X = pd.concat([df_m[pred_list + NUISANCE], scen_dum], axis=1).astype(float)
            X = sm.add_constant(X)
            res = fit_glm_clustered(X, y, groups)
            if res:
                r2 = 1 - (res.llf / res.llnull) if res.llnull != 0 else np.nan
                return r2
            return np.nan

        r2_full = get_r2(PREDICTORS_OF_INTEREST, "full")
        r2_noO = get_r2([p for p in PREDICTORS_OF_INTEREST if p != "O"], "no_O")
        r2_only_O = get_r2(["O"], "only_O")
        r2_only_Inc = get_r2(["Inc_L", "Inc_Q"], "only_Inc")
        r2_demo_only = get_r2(["Inc_L", "Inc_Q", "Age_L", "Gender_code", "Region_code"], "demo")
        r2_b5_only = get_r2(B5_TRAITS, "B5")
        r2_b5_noO = get_r2(["C", "E", "A", "N"], "B5_noO")

        report(f"  {model_name}:")
        report(f"    Full model (10 preds):         pseudo-R² = {r2_full:.4f}")
        report(f"    Without O (9 preds):           pseudo-R² = {r2_noO:.4f}  "
               f"(removing O reduces pseudo-R² by ~{(r2_full - r2_noO) / r2_full * 100:.0f}%)")
        report(f"    O alone:                       pseudo-R² = {r2_only_O:.4f}")
        report(f"    Income alone (L+Q):            pseudo-R² = {r2_only_Inc:.4f}")
        report(f"    Demographics (5 preds):        pseudo-R² = {r2_demo_only:.4f}")
        report(f"    All B5 (5 preds):              pseudo-R² = {r2_b5_only:.4f}")
        report(f"    B5 without O (4 preds):        pseudo-R² = {r2_b5_noO:.4f}")
        report(f"    NOTE: McFadden's pseudo-R² is not linearly decomposable")
        report(f"    like OLS R². Percentages are approximate.")
        report()

    report()


def main():
    report("=" * 70)
    report("RED FLAGS ANALYSIS — Critical Checks Before Paper Writing")
    report("=" * 70)
    report()

    parquet_path = RESULTS_DIR / "analysis_df.parquet"
    if not parquet_path.exists():
        report("ERROR: analysis_df.parquet not found")
        return

    df = pd.read_parquet(parquet_path)
    report(f"Loaded {len(df):,} rows")
    report()

    # Run all analyses
    analysis_1_positional_bias(df)
    analysis_2_income_involvement(df)
    analysis_3_s_index_without_o(df)
    analysis_4_exclude_gemma3(df)
    analysis_5_b5_interactions(df)
    analysis_bonus_r2_decomposition(df)

    # Save full report
    report_path = RESULTS_DIR / "red_flags_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nFull report saved: {report_path}")


if __name__ == "__main__":
    main()
