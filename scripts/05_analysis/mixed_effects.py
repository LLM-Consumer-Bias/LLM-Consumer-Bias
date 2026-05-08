"""Phase 6 Step 4: Mixed-effects analysis using GEE (design.md §14.3).

METHODOLOGICAL NOTE:
    The design document (§14.3) specifies glmer with (1|Model) + (1|ProfileID).
    We use GEE instead for the following reasons:
    1. With K=4 models, a random intercept (1|Model) has only 2 df for variance
       estimation — likely singular. Model is treated as a fixed effect instead.
    2. GEE with exchangeable correlation within ProfileID clusters accounts for
       within-profile dependence without requiring random effects convergence.
    3. GEE estimates MARGINAL (population-averaged) coefficients, which are
       attenuated relative to conditional (subject-specific) glmer coefficients
       by a factor of 0.67–0.88 (see must_do_4_7_analysis.py §5 for estimation).
    4. All qualitative conclusions are conservative under GEE (marginal < conditional).

Model sequence:
    M0: Choice_A ~ 1 + C(ScenarioID) + Order + Format_1 + Format_2 + C(Model)
    M1: M0 + Inc_L + Inc_Q + Age_L + Gender_code + Region_code
    M2: M0 + O + C + E + A + N
    M3: M0 + demographics + personality (full additive)
    M4: M3 + B5 two-way interactions (10 terms)

Model comparisons: Wald tests (not LRTs — LRT is not well-defined for GEE).
FDR correction: BH-FDR on 10 main effects per model (Family F1, design §14.5).

Output:
    results/mixed_effects_summary.csv
    results/wald_test_results.csv  (renamed from lrt_results.csv)

Usage:
    python scripts/05_analysis/mixed_effects.py
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
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"

# Variable groups
NUISANCE = ["Order", "Format_1", "Format_2"]
DEMOGRAPHICS = ["Inc_L", "Inc_Q", "Age_L", "Gender_code", "Region_code"]
PERSONALITY = ["O", "C", "E", "A", "N"]

# B5 two-way interactions
B5_INTERACTIONS = list(itertools.combinations(PERSONALITY, 2))


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for GEE: condition A, non-V4, temp=1.0, all models."""
    mask = (
        (df["dir_condition"] == "A_nl_consumer") &
        (df["is_v4"] == 0) &
        (df["temperature"] == 1.0)
    )
    df_gee = df[mask].copy()

    # Create scenario dummies (reference = S1)
    scenario_dummies = pd.get_dummies(df_gee["scenario_id"], prefix="Scenario",
                                      drop_first=True, dtype=int)
    df_gee = pd.concat([df_gee, scenario_dummies], axis=1)

    # Create model dummies (treat as fixed effect since K=4)
    model_dummies = pd.get_dummies(df_gee["dir_model"], prefix="Model",
                                   drop_first=True, dtype=int)
    df_gee = pd.concat([df_gee, model_dummies], axis=1)

    # Create B5 interaction terms
    for t1, t2 in B5_INTERACTIONS:
        df_gee[f"{t1}x{t2}"] = df_gee[t1] * df_gee[t2]

    # Sort by ProfileID for GEE clustering
    df_gee = df_gee.sort_values("ProfileID").reset_index(drop=True)

    return df_gee


def fit_gee(df: pd.DataFrame, predictors: list, model_label: str) -> dict:
    """Fit a GEE model with exchangeable correlation within ProfileID."""
    # Get scenario and model dummy columns
    scenario_cols = [c for c in df.columns if c.startswith("Scenario_")]
    model_cols = [c for c in df.columns if c.startswith("Model_")]

    all_predictors = predictors + NUISANCE + scenario_cols + model_cols
    X = sm.add_constant(df[all_predictors].astype(float))
    y = df["Choice_A"].astype(float)
    groups = df["ProfileID"]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gee = GEE(
                endog=y,
                exog=X,
                groups=groups,
                family=Binomial(),
                cov_struct=Exchangeable(),
            )
            result = gee.fit()
    except Exception as e:
        print(f"    ERROR fitting {model_label}: {e}")
        return {"label": model_label, "error": str(e)}

    # Quasi-likelihood information criterion
    qic = result.qic() if hasattr(result, "qic") else (np.nan, np.nan)

    # Extract coefficients
    coefs = []
    for pred in predictors:
        if pred in result.params.index:
            coefs.append({
                "predictor": pred,
                "beta": result.params[pred],
                "se": result.bse[pred],
                "z": result.tvalues[pred],
                "p": result.pvalues[pred],
                "ci_low": result.conf_int().loc[pred, 0],
                "ci_high": result.conf_int().loc[pred, 1],
            })

    return {
        "label": model_label,
        "n_obs": len(df),
        "n_clusters": groups.nunique(),
        "qic": qic[0] if isinstance(qic, tuple) else qic,
        "n_predictors": len(all_predictors) + 1,  # +1 for constant
        "result": result,
        "coefs": coefs,
    }


def compare_models(fit1: dict, fit2: dict, df1: int, label: str) -> dict:
    """Compare nested models using Wald-type test (QIC difference).

    Since GEE doesn't support LRT directly, we use:
    1. QIC difference for model selection
    2. Wald test for the added predictors
    """
    if "error" in fit1 or "error" in fit2:
        return {"comparison": label, "error": "model fitting failed"}

    qic_diff = fit1["qic"] - fit2["qic"]

    # Wald test for the added block of predictors
    result_full = fit1["result"]
    added_preds = [c["predictor"] for c in fit1["coefs"]
                   if c["predictor"] not in [cc["predictor"] for cc in fit2["coefs"]]]

    if added_preds:
        # Joint Wald test for the added predictors
        r_matrix = np.zeros((len(added_preds), len(result_full.params)))
        found = 0
        for i, pred in enumerate(added_preds):
            if pred in result_full.params.index:
                j = list(result_full.params.index).index(pred)
                r_matrix[i, j] = 1
                found += 1

        try:
            if found > 0:
                wald_stat = result_full.wald_test(r_matrix, scalar=True)
                chi2 = float(wald_stat.statistic)
                p_val = float(wald_stat.pvalue)
            else:
                chi2, p_val = np.nan, np.nan
            df_test = len(added_preds)
        except Exception:
            chi2, p_val, df_test = np.nan, np.nan, len(added_preds)
    else:
        chi2, p_val, df_test = np.nan, np.nan, 0

    return {
        "comparison": label,
        "full_model": fit1["label"],
        "null_model": fit2["label"],
        "qic_full": fit1["qic"],
        "qic_null": fit2["qic"],
        "qic_diff": qic_diff,
        "wald_chi2": chi2,
        "df": df_test,
        "p_value": p_val,
        "sig": p_val < 0.05 if not np.isnan(p_val) else False,
    }


def main():
    print("=" * 60)
    print("Phase 6 Step 4: Mixed-Effects Analysis (GEE)")
    print("=" * 60)

    # Load preprocessed data
    parquet_path = RESULTS_DIR / "analysis_df.parquet"
    if not parquet_path.exists():
        print("ERROR: results/analysis_df.parquet not found. Run preprocessing.py first.")
        return

    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df):,} rows")

    # Prepare data
    print("\nPreparing data for GEE...")
    df_gee = prepare_data(df)
    print(f"  Analysis sample: {len(df_gee):,} rows, "
          f"{df_gee['ProfileID'].nunique()} profiles, "
          f"{df_gee['dir_model'].nunique()} models")

    # Fit models M0–M4
    print("\nFitting models...")
    interaction_terms = [f"{t1}x{t2}" for t1, t2 in B5_INTERACTIONS]

    m0 = fit_gee(df_gee, [], "M0")
    print(f"  M0 (nuisance only): QIC={m0.get('qic', 'N/A')}")

    m1 = fit_gee(df_gee, DEMOGRAPHICS, "M1")
    print(f"  M1 (+ demographics): QIC={m1.get('qic', 'N/A')}")

    m2 = fit_gee(df_gee, PERSONALITY, "M2")
    print(f"  M2 (+ personality): QIC={m2.get('qic', 'N/A')}")

    m3 = fit_gee(df_gee, DEMOGRAPHICS + PERSONALITY, "M3")
    print(f"  M3 (demographics + personality): QIC={m3.get('qic', 'N/A')}")

    m4 = fit_gee(df_gee, DEMOGRAPHICS + PERSONALITY + interaction_terms, "M4")
    if "error" in m4:
        print(f"  M4 (+ B5 interactions): FAILED TO CONVERGE — {m4['error']}")
        print(f"  NOTE: M4 failure is expected — 10 additional B5 interaction terms")
        print(f"  create multicollinearity with the Resolution V fractional factorial.")
        print(f"  B5 interaction evidence relies on per-model GLMs (primary analysis).")
    else:
        print(f"  M4 (+ B5 interactions): QIC={m4.get('qic', 'N/A')}")

    # Model comparisons
    print("\nModel comparisons (Wald tests)...")
    comparisons = [
        compare_models(m1, m0, len(DEMOGRAPHICS), "M1 vs M0 (demographics)"),
        compare_models(m2, m0, len(PERSONALITY), "M2 vs M0 (personality)"),
        compare_models(m3, m1, len(PERSONALITY), "M3 vs M1 (personality | demographics)"),
        compare_models(m4, m3, len(interaction_terms), "M4 vs M3 (B5 interactions)"),
    ]

    for comp in comparisons:
        if "error" in comp:
            print(f"  {comp['comparison']}: ERROR")
        else:
            sig = "***" if comp.get("p_value", 1) < 0.001 else \
                  "** " if comp.get("p_value", 1) < 0.01 else \
                  "*  " if comp.get("p_value", 1) < 0.05 else "n.s."
            print(f"  {comp['comparison']}: χ²={comp.get('wald_chi2', 'N/A'):.2f}, "
                  f"df={comp.get('df', 'N/A')}, p={comp.get('p_value', 'N/A'):.4f} {sig}")

    # Save Wald test results (renamed from lrt_results.csv — these are Wald tests, not LRTs)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_wald = pd.DataFrame(comparisons)
    df_wald.to_csv(RESULTS_DIR / "wald_test_results.csv", index=False)
    # Also save with old name for backward compatibility
    df_wald.to_csv(RESULTS_DIR / "lrt_results.csv", index=False)
    print(f"\nSaved: results/wald_test_results.csv")

    # Save coefficient summaries for all models
    all_coefs = []
    for model_fit in [m0, m1, m2, m3, m4]:
        if "error" in model_fit:
            continue
        for coef in model_fit.get("coefs", []):
            coef["model_label"] = model_fit["label"]
            coef["n_obs"] = model_fit["n_obs"]
            coef["qic"] = model_fit["qic"]
            all_coefs.append(coef)

    df_coefs = pd.DataFrame(all_coefs)

    # Apply FDR correction (Family F1: 10 main effects per model, design §14.5)
    # FDR applied within each model (M1, M2, M3 separately)
    main_effects = set(DEMOGRAPHICS + PERSONALITY)
    for model_label in df_coefs["model_label"].unique():
        mask = (df_coefs["model_label"] == model_label) & \
               (df_coefs["predictor"].isin(main_effects))
        if mask.sum() > 0:
            p_raw = df_coefs.loc[mask, "p"].values
            reject, p_fdr, _, _ = multipletests(p_raw, method="fdr_bh", alpha=0.05)
            df_coefs.loc[mask, "p_fdr"] = p_fdr
            df_coefs.loc[mask, "sig_fdr"] = reject
        else:
            continue

    # For non-main-effects (interaction terms in M4), mark as exploratory (no FDR)
    no_fdr_mask = df_coefs["p_fdr"].isna()
    df_coefs.loc[no_fdr_mask, "p_fdr"] = np.nan
    df_coefs.loc[no_fdr_mask, "sig_fdr"] = np.nan

    df_coefs.to_csv(RESULTS_DIR / "mixed_effects_summary.csv", index=False)
    print(f"Saved: results/mixed_effects_summary.csv ({len(df_coefs)} rows, with FDR)")

    # Print M3 coefficients (full additive model)
    if "error" not in m3:
        print(f"\n{'=' * 60}")
        print("M3 Coefficients (Full Additive)")
        print(f"{'=' * 60}")
        for coef in m3["coefs"]:
            sig = "*" if coef["p"] < 0.05 else " "
            print(f"  {coef['predictor']:15s}  β={coef['beta']:+.4f}  "
                  f"SE={coef['se']:.4f}  z={coef['z']:.2f}  "
                  f"p={coef['p']:.4f} {sig}")


if __name__ == "__main__":
    main()
