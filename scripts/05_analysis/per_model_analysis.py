"""Phase 6 Step 3: Per-model logistic regression with clustered SEs.

Primary analysis (design.md §14.2).
For each model at temp=1.0 (primary) + all temps:
    GLM(Choice_A ~ Inc_L + Inc_Q + Age_L + Gender + Region +
                   O + C + E + A_trait + N +
                   C(ScenarioID) + Order + Format_1 + Format_2,
        family=Binomial)
    with clustered standard errors on ProfileID.

FDR correction within each model (Family F3: 10 tests per model).

Output:
    results/per_model_coefficients.csv (β, SE, CI, p_raw, p_fdr, sig)

Usage:
    python scripts/05_analysis/per_model_analysis.py
"""

import pathlib
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"

# Predictors of interest (10, for FDR correction)
PREDICTORS_OF_INTEREST = [
    "Inc_L", "Inc_Q", "Age_L", "Gender_code", "Region_code",
    "O", "C", "E", "A", "N",
]

# Nuisance variables (not tested)
NUISANCE_VARS = ["Order", "Format_1", "Format_2"]

# Scenario dummies will be added dynamically


def fit_per_model_glm(df: pd.DataFrame, model_name: str, temp: float) -> pd.DataFrame:
    """Fit logistic regression for one model at one temperature.

    Uses GLM + clustered SEs (sandwich estimator) on ProfileID.
    This is the design.md §14.2 fallback (used since R/lme4 not available).
    """
    # Filter to this model + temperature, condition A, non-V4
    mask = (
        (df["dir_model"] == model_name) &
        (df["temperature"] == temp) &
        (df["dir_condition"] == "A_nl_consumer") &
        (df["is_v4"] == 0)
    )
    df_model = df[mask].copy()

    if len(df_model) < 100:
        print(f"    SKIP: {model_name} t={temp} — only {len(df_model)} rows")
        return pd.DataFrame()

    print(f"    {model_name} t={temp}: {len(df_model):,} rows, "
          f"Choice_A rate = {df_model['Choice_A'].mean():.3f}")

    # Create scenario dummies (reference = S1)
    scenario_dummies = pd.get_dummies(df_model["scenario_id"], prefix="Scenario",
                                      drop_first=True, dtype=int)
    scenario_cols = list(scenario_dummies.columns)

    # Build design matrix — reset index to align properly
    df_model = df_model.reset_index(drop=True)
    scenario_dummies = scenario_dummies.reset_index(drop=True)

    X = pd.concat([df_model[PREDICTORS_OF_INTEREST + NUISANCE_VARS], scenario_dummies], axis=1)
    X = X.astype(float)
    X = sm.add_constant(X)
    y = df_model["Choice_A"].astype(float)

    # Fit GLM with binomial family
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.GLM(y, X, family=sm.families.Binomial())
            result = model.fit(
                cov_type="cluster",
                cov_kwds={"groups": df_model["ProfileID"].values},
            )
    except Exception as e:
        print(f"    ERROR fitting {model_name} t={temp}: {e}")
        return pd.DataFrame()

    # Extract results for predictors of interest
    rows = []
    for pred in PREDICTORS_OF_INTEREST:
        if pred not in result.params.index:
            continue

        beta = result.params[pred]
        se = result.bse[pred]
        ci_low, ci_high = result.conf_int().loc[pred]
        p_raw = result.pvalues[pred]
        z = result.tvalues[pred]
        or_val = np.exp(beta)
        or_ci_low = np.exp(ci_low)
        or_ci_high = np.exp(ci_high)

        rows.append({
            "model": model_name,
            "temperature": temp,
            "predictor": pred,
            "beta": beta,
            "se": se,
            "z": z,
            "p_raw": p_raw,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "odds_ratio": or_val,
            "or_ci_low": or_ci_low,
            "or_ci_high": or_ci_high,
        })

    df_results = pd.DataFrame(rows)

    # FDR correction within this model (Family F3: 10 tests)
    if len(df_results) > 0:
        reject, p_fdr, _, _ = multipletests(df_results["p_raw"], method="fdr_bh", alpha=0.05)
        df_results["p_fdr"] = p_fdr
        df_results["sig_raw"] = df_results["p_raw"] < 0.05
        df_results["sig_fdr"] = reject

    # Also extract nuisance + scenario coefficients (for completeness, no FDR)
    for pred in NUISANCE_VARS + scenario_cols + ["const"]:
        if pred not in result.params.index:
            continue
        beta = result.params[pred]
        se = result.bse[pred]
        ci_low, ci_high = result.conf_int().loc[pred]
        p_raw = result.pvalues[pred]

    # Model fit statistics
    n = len(df_model)
    ll = result.llf
    ll_null = result.llnull
    pseudo_r2 = 1 - (ll / ll_null) if ll_null != 0 else np.nan
    aic = result.aic
    bic = result.bic_llf if hasattr(result, "bic_llf") else np.nan

    df_results["n_obs"] = n
    df_results["pseudo_r2_mcfadden"] = pseudo_r2
    df_results["aic"] = aic
    df_results["ll"] = ll

    return df_results


def main():
    print("=" * 60)
    print("Phase 6 Step 3: Per-Model Analysis")
    print("=" * 60)

    # Load preprocessed data
    parquet_path = RESULTS_DIR / "analysis_df.parquet"
    if not parquet_path.exists():
        print("ERROR: results/analysis_df.parquet not found. Run preprocessing.py first.")
        return

    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df):,} rows")

    # Get unique models and temperatures
    models = sorted(df["dir_model"].unique())
    temps = sorted(df["temperature"].unique())
    print(f"Models: {models}")
    print(f"Temperatures: {temps}")

    # Primary analysis: temp=1.0 for all models
    print(f"\n--- Primary Analysis (temp=1.0) ---")
    all_results = []
    for model_name in models:
        result = fit_per_model_glm(df, model_name, temp=1.0)
        if len(result) > 0:
            all_results.append(result)

    # Secondary: all temperatures
    print(f"\n--- All Temperatures ---")
    for temp in temps:
        if temp == 1.0:
            continue  # Already done
        for model_name in models:
            result = fit_per_model_glm(df, model_name, temp)
            if len(result) > 0:
                all_results.append(result)

    if not all_results:
        print("No results produced!")
        return

    # Combine and save
    df_all = pd.concat(all_results, ignore_index=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(RESULTS_DIR / "per_model_coefficients.csv", index=False)
    print(f"\nSaved: results/per_model_coefficients.csv ({len(df_all)} rows)")

    # Print primary (temp=1.0) results table
    print(f"\n{'=' * 60}")
    print("PRIMARY RESULTS (temp=1.0)")
    print(f"{'=' * 60}")

    df_primary = df_all[df_all["temperature"] == 1.0].copy()
    for model_name in models:
        df_m = df_primary[df_primary["model"] == model_name]
        if len(df_m) == 0:
            continue
        print(f"\n--- {model_name} (N={df_m['n_obs'].iloc[0]:,}, "
              f"McFadden R²={df_m['pseudo_r2_mcfadden'].iloc[0]:.4f}) ---")
        for _, row in df_m.iterrows():
            sig_marker = "***" if row["p_fdr"] < 0.001 else \
                         "** " if row["p_fdr"] < 0.01 else \
                         "*  " if row["p_fdr"] < 0.05 else "   "
            print(f"  {row['predictor']:15s}  β={row['beta']:+.4f}  "
                  f"SE={row['se']:.4f}  "
                  f"OR={row['odds_ratio']:.3f} [{row['or_ci_low']:.3f}, {row['or_ci_high']:.3f}]  "
                  f"p_raw={row['p_raw']:.4f}  p_fdr={row['p_fdr']:.4f} {sig_marker}")


if __name__ == "__main__":
    main()
