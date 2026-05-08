"""Phase 6 Step 7: Prompt ablation analysis (design.md §10).

Compare conditions A vs B vs C vs D for mistral-small3.2 and qwen3-32b:
- Same GLM framework, condition as factor
- Test: condition × predictor interactions
- Does prompt format change sensitivity patterns?

Output: results/ablation_effects.csv

Usage:
    python scripts/05_analysis/prompt_ablation_analysis.py
"""

import pathlib
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"

ABLATION_MODELS = ["mistral-small3.2", "qwen3-32b"]

PREDICTORS_OF_INTEREST = [
    "Inc_L", "Inc_Q", "Age_L", "Gender_code", "Region_code",
    "O", "C", "E", "A", "N",
]
NUISANCE = ["Order", "Format_1", "Format_2"]


def fit_per_condition(df: pd.DataFrame, model_name: str, condition: str) -> pd.DataFrame:
    """Fit GLM for a specific model × condition combination."""
    mask = (
        (df["dir_model"] == model_name) &
        (df["dir_condition"] == condition) &
        (df["temperature"] == 1.0) &
        (df["is_v4"] == 0)
    )
    df_sub = df[mask].copy()

    if len(df_sub) < 100:
        print(f"    SKIP: {model_name}/{condition} — only {len(df_sub)} rows")
        return pd.DataFrame()

    # Create scenario dummies
    df_sub = df_sub.reset_index(drop=True)
    scenario_dummies = pd.get_dummies(df_sub["scenario_id"], prefix="Scenario",
                                      drop_first=True, dtype=int)
    scenario_dummies = scenario_dummies.reset_index(drop=True)
    scenario_cols = list(scenario_dummies.columns)

    # Ablation conditions B/C/D use only AB forms → Order/Format may be constant
    # Check and drop constant columns
    available_nuisance = []
    for n in NUISANCE:
        if df_sub[n].nunique() > 1:
            available_nuisance.append(n)

    X = pd.concat([df_sub[PREDICTORS_OF_INTEREST + available_nuisance], scenario_dummies], axis=1)
    X = X.astype(float)
    X = sm.add_constant(X)
    y = df_sub["Choice_A"].astype(float)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.GLM(y, X, family=sm.families.Binomial())
            result = model.fit(
                cov_type="cluster",
                cov_kwds={"groups": df_sub["ProfileID"].values},
            )
    except Exception as e:
        print(f"    ERROR: {e}")
        return pd.DataFrame()

    rows = []
    for pred in PREDICTORS_OF_INTEREST:
        if pred not in result.params.index:
            continue
        rows.append({
            "model": model_name,
            "condition": condition,
            "predictor": pred,
            "beta": result.params[pred],
            "se": result.bse[pred],
            "z": result.tvalues[pred],
            "p_raw": result.pvalues[pred],
            "ci_low": result.conf_int().loc[pred, 0],
            "ci_high": result.conf_int().loc[pred, 1],
            "n_obs": len(df_sub),
            "choice_a_rate": df_sub["Choice_A"].mean(),
        })

    df_result = pd.DataFrame(rows)
    if len(df_result) > 0:
        reject, p_fdr, _, _ = multipletests(df_result["p_raw"], method="fdr_bh", alpha=0.05)
        df_result["p_fdr"] = p_fdr
        df_result["sig_fdr"] = reject

    return df_result


def fit_interaction_model(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Fit condition × predictor interaction model for one model.

    Tests whether prompt format changes predictor coefficients.
    """
    mask = (
        (df["dir_model"] == model_name) &
        (df["temperature"] == 1.0) &
        (df["is_v4"] == 0)
    )
    df_sub = df[mask].copy()

    conditions = sorted(df_sub["dir_condition"].unique())
    if len(conditions) < 2:
        print(f"    SKIP interaction: {model_name} — only {len(conditions)} condition(s)")
        return pd.DataFrame()

    print(f"  {model_name}: {len(df_sub):,} rows, conditions={conditions}")

    df_sub = df_sub.reset_index(drop=True)

    # Create condition dummies (reference = A_nl_consumer)
    cond_dummies = pd.get_dummies(df_sub["dir_condition"], prefix="Cond",
                                  drop_first=True, dtype=int)
    cond_cols = list(cond_dummies.columns)
    df_sub = pd.concat([df_sub, cond_dummies], axis=1)

    # Create scenario dummies
    scenario_dummies = pd.get_dummies(df_sub["scenario_id"], prefix="Scenario",
                                      drop_first=True, dtype=int)
    scenario_cols = list(scenario_dummies.columns)
    df_sub = pd.concat([df_sub, scenario_dummies], axis=1)

    # Create condition × predictor interactions
    interaction_cols = []
    for cond_col in cond_cols:
        for pred in PREDICTORS_OF_INTEREST:
            col_name = f"{cond_col}_x_{pred}"
            df_sub[col_name] = df_sub[cond_col] * df_sub[pred]
            interaction_cols.append(col_name)

    # Check which nuisance vars are usable
    available_nuisance = [n for n in NUISANCE if df_sub[n].nunique() > 1]

    all_preds = (PREDICTORS_OF_INTEREST + cond_cols + interaction_cols +
                 available_nuisance + scenario_cols)
    X = df_sub[all_preds].astype(float)
    X = sm.add_constant(X)
    y = df_sub["Choice_A"].astype(float)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.GLM(y, X, family=sm.families.Binomial())
            result = model.fit(
                cov_type="cluster",
                cov_kwds={"groups": df_sub["ProfileID"].values},
            )
    except Exception as e:
        print(f"    ERROR: {e}")
        return pd.DataFrame()

    # Extract interaction coefficients
    rows = []
    for col in interaction_cols:
        if col not in result.params.index:
            continue
        rows.append({
            "model": model_name,
            "predictor": col,
            "beta": result.params[col],
            "se": result.bse[col],
            "z": result.tvalues[col],
            "p": result.pvalues[col],
            "ci_low": result.conf_int().loc[col, 0],
            "ci_high": result.conf_int().loc[col, 1],
        })

    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("Phase 6 Step 7: Prompt Ablation Analysis")
    print("=" * 60)

    # Load preprocessed data
    parquet_path = RESULTS_DIR / "analysis_df.parquet"
    if not parquet_path.exists():
        print("ERROR: results/analysis_df.parquet not found. Run preprocessing.py first.")
        return

    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df):,} rows")

    # Check which ablation data is available
    ablation_data = df[df["dir_condition"] != "A_nl_consumer"]
    if len(ablation_data) == 0:
        print("No ablation data found. Skipping.")
        return

    print(f"Ablation data: {len(ablation_data):,} rows")
    print("Conditions × Models:")
    print(ablation_data.groupby(["dir_model", "dir_condition"]).size().to_string())

    # 1. Per-condition fits
    print(f"\n--- Per-Condition Analysis ---")
    all_per_cond = []
    for model_name in ABLATION_MODELS:
        conditions = sorted(df[df["dir_model"] == model_name]["dir_condition"].unique())
        for condition in conditions:
            print(f"\n  {model_name} / {condition}:")
            result = fit_per_condition(df, model_name, condition)
            if len(result) > 0:
                all_per_cond.append(result)

    if all_per_cond:
        df_per_cond = pd.concat(all_per_cond, ignore_index=True)
    else:
        df_per_cond = pd.DataFrame()

    # 2. Condition × predictor interaction models
    print(f"\n--- Condition × Predictor Interactions ---")
    all_interactions = []
    for model_name in ABLATION_MODELS:
        result = fit_interaction_model(df, model_name)
        if len(result) > 0:
            all_interactions.append(result)

    if all_interactions:
        df_interactions = pd.concat(all_interactions, ignore_index=True)
    else:
        df_interactions = pd.DataFrame()

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if len(df_per_cond) > 0:
        df_per_cond.to_csv(RESULTS_DIR / "ablation_effects.csv", index=False)
        print(f"\nSaved: results/ablation_effects.csv ({len(df_per_cond)} rows)")

    if len(df_interactions) > 0:
        # Apply FDR correction within each model for interaction terms
        for model_name in df_interactions["model"].unique():
            mask = df_interactions["model"] == model_name
            p_raw = df_interactions.loc[mask, "p"].values
            if len(p_raw) > 0 and not np.all(np.isnan(p_raw)):
                valid = ~np.isnan(p_raw)
                p_fdr = np.full_like(p_raw, np.nan)
                if valid.sum() > 0:
                    _, p_fdr[valid], _, _ = multipletests(
                        p_raw[valid], method="fdr_bh", alpha=0.05)
                df_interactions.loc[mask, "p_fdr"] = p_fdr
                df_interactions.loc[mask, "sig_fdr"] = p_fdr < 0.05

        df_interactions.to_csv(RESULTS_DIR / "ablation_interactions.csv", index=False)
        print(f"Saved: results/ablation_interactions.csv ({len(df_interactions)} rows, with FDR)")

    # Print comparison table
    if len(df_per_cond) > 0:
        print(f"\n{'=' * 60}")
        print("Ablation: β comparison across conditions")
        print(f"{'=' * 60}")
        for model_name in ABLATION_MODELS:
            df_m = df_per_cond[df_per_cond["model"] == model_name]
            if len(df_m) == 0:
                continue
            print(f"\n  {model_name}:")
            pivot = df_m.pivot_table(
                index="predictor",
                columns="condition",
                values="beta",
            )
            if len(pivot) > 0:
                print(pivot.to_string())

    # Print significant interactions
    if len(df_interactions) > 0:
        sig_interactions = df_interactions[df_interactions["p"] < 0.05]
        if len(sig_interactions) > 0:
            print(f"\n{'=' * 60}")
            print("Significant Condition × Predictor Interactions (p < 0.05)")
            print(f"{'=' * 60}")
            for _, row in sig_interactions.iterrows():
                print(f"  {row['model']}: {row['predictor']}  "
                      f"β={row['beta']:+.4f}  p={row['p']:.4f}")


if __name__ == "__main__":
    main()
