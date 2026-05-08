"""Phase 6 Step 5: Temperature comparison (design.md §14.7).

- Per-model analysis at each temperature (3 × 4 = 12 model fits)
- Temperature interaction model: Temp_L/Temp_Q × predictors
- Fleiss' κ per model per temperature (determinism assessment)
- S-index trajectory across temperatures

Output:
    results/temperature_effects.csv
    results/fleiss_kappa.csv

Usage:
    python scripts/05_analysis/temperature_analysis.py
"""

import pathlib
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as scipy_stats
from statsmodels.stats.multitest import multipletests

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"

PREDICTORS_OF_INTEREST = [
    "Inc_L", "Inc_Q", "Age_L", "Gender_code", "Region_code",
    "O", "C", "E", "A", "N",
]
NUISANCE = ["Order", "Format_1", "Format_2"]


def fleiss_kappa(ratings_matrix: np.ndarray) -> float:
    """Compute Fleiss' kappa for inter-rater agreement.

    Args:
        ratings_matrix: n_subjects × n_categories matrix where each cell
            is the count of raters who assigned that category.

    Returns:
        Fleiss' kappa statistic.
    """
    n, k = ratings_matrix.shape
    n_raters = ratings_matrix.sum(axis=1)
    if n_raters.std() > 0.01:
        # Variable number of raters — use mean
        N = n_raters.mean()
    else:
        N = n_raters[0]

    if N <= 1 or n == 0:
        return np.nan

    # P_i for each subject
    p_i = (np.sum(ratings_matrix ** 2, axis=1) - N) / (N * (N - 1))
    P_bar = np.mean(p_i)

    # P_j for each category
    p_j = np.sum(ratings_matrix, axis=0) / (n * N)
    P_e = np.sum(p_j ** 2)

    if P_e == 1.0:
        return 1.0  # Perfect agreement

    kappa = (P_bar - P_e) / (1 - P_e)
    return kappa


def compute_fleiss_kappa_per_model(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Fleiss' κ per model per temperature.

    For each (model, temp, profile, scenario, form), we have M=3 replications.
    Each replication produces a choice (A or B).
    Fleiss' κ measures agreement across replications.
    """
    results = []

    for model_name in sorted(df["dir_model"].unique()):
        for temp in sorted(df["temperature"].unique()):
            mask = (
                (df["dir_model"] == model_name) &
                (df["temperature"] == temp) &
                (df["dir_condition"] == "A_nl_consumer") &
                (df["is_v4"] == 0)
            )
            df_sub = df[mask].copy()

            if len(df_sub) < 10:
                continue

            # Group by (profile, scenario, form) — each group should have M=3 reps
            grouped = df_sub.groupby(["profile_id", "scenario_id", "form_id"])

            ratings_list = []
            for _, group in grouped:
                n_a = (group["Choice_A"] == 1).sum()
                n_b = (group["Choice_A"] == 0).sum()
                n_total = n_a + n_b
                if n_total >= 2:  # Need at least 2 raters
                    ratings_list.append([n_a, n_b])

            if len(ratings_list) < 10:
                continue

            ratings_matrix = np.array(ratings_list)
            kappa = fleiss_kappa(ratings_matrix)

            results.append({
                "model": model_name,
                "temperature": temp,
                "fleiss_kappa": kappa,
                "n_items": len(ratings_list),
                "mean_n_raters": ratings_matrix.sum(axis=1).mean(),
            })

            print(f"  {model_name} t={temp}: κ={kappa:.4f} ({len(ratings_list):,} items)")

    return pd.DataFrame(results)


def fit_temperature_interaction_model(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Fit temperature interaction model for one model across all temperatures.

    Model: Choice_A ~ (Temp_L + Temp_Q) × predictors + nuisance + scenario
    """
    mask = (
        (df["dir_model"] == model_name) &
        (df["dir_condition"] == "A_nl_consumer") &
        (df["is_v4"] == 0) &
        (df["Temp_L"].notna())
    )
    df_model = df[mask].copy()

    if len(df_model) < 100:
        return pd.DataFrame()

    print(f"  {model_name}: {len(df_model):,} rows across {df_model['temperature'].nunique()} temperatures")

    # Create scenario dummies
    df_model = df_model.reset_index(drop=True)
    scenario_dummies = pd.get_dummies(df_model["scenario_id"], prefix="Scenario",
                                      drop_first=True, dtype=int)
    scenario_dummies = scenario_dummies.reset_index(drop=True)
    scenario_cols = list(scenario_dummies.columns)

    # Create interaction terms: Temp_L × predictor, Temp_Q × predictor
    for pred in PREDICTORS_OF_INTEREST:
        df_model[f"TempL_x_{pred}"] = df_model["Temp_L"] * df_model[pred]
        df_model[f"TempQ_x_{pred}"] = df_model["Temp_Q"] * df_model[pred]

    temp_interaction_cols = [f"TempL_x_{p}" for p in PREDICTORS_OF_INTEREST] + \
                           [f"TempQ_x_{p}" for p in PREDICTORS_OF_INTEREST]

    # Full predictor list
    non_scenario_preds = (["Temp_L", "Temp_Q"] + PREDICTORS_OF_INTEREST +
                          temp_interaction_cols + NUISANCE)

    X = pd.concat([df_model[non_scenario_preds], scenario_dummies], axis=1)
    X = X.astype(float)
    X = sm.add_constant(X)
    y = df_model["Choice_A"].astype(float)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.GLM(y, X, family=sm.families.Binomial())
            result = model.fit(
                cov_type="cluster",
                cov_kwds={"groups": df_model["ProfileID"].values},
            )
    except Exception as e:
        print(f"    ERROR: {e}")
        return pd.DataFrame()

    # Extract temperature interaction coefficients
    rows = []
    test_preds = ["Temp_L", "Temp_Q"] + temp_interaction_cols
    for pred in test_preds:
        if pred not in result.params.index:
            continue
        rows.append({
            "model": model_name,
            "predictor": pred,
            "beta": result.params[pred],
            "se": result.bse[pred],
            "z": result.tvalues[pred],
            "p": result.pvalues[pred],
            "ci_low": result.conf_int().loc[pred, 0],
            "ci_high": result.conf_int().loc[pred, 1],
        })

    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("Phase 6 Step 5: Temperature Analysis")
    print("=" * 60)

    # Load preprocessed data
    parquet_path = RESULTS_DIR / "analysis_df.parquet"
    if not parquet_path.exists():
        print("ERROR: results/analysis_df.parquet not found. Run preprocessing.py first.")
        return

    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df):,} rows")

    models = sorted(df["dir_model"].unique())
    temps = sorted(df["temperature"].unique())
    print(f"Models: {models}")
    print(f"Temperatures: {temps}")

    # 1. Fleiss' kappa
    print(f"\n--- Fleiss' κ (determinism assessment) ---")
    df_kappa = compute_fleiss_kappa_per_model(df)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if len(df_kappa) > 0:
        df_kappa.to_csv(RESULTS_DIR / "fleiss_kappa.csv", index=False)
        print(f"\nSaved: results/fleiss_kappa.csv ({len(df_kappa)} rows)")
    else:
        print("\nNo kappa results computed.")

    # 2. Temperature interaction models
    print(f"\n--- Temperature Interaction Models ---")
    all_temp_effects = []
    for model_name in models:
        result = fit_temperature_interaction_model(df, model_name)
        if len(result) > 0:
            all_temp_effects.append(result)

    if all_temp_effects:
        df_temp = pd.concat(all_temp_effects, ignore_index=True)

        # Apply FDR correction within each model (all temperature predictors)
        for model_name in df_temp["model"].unique():
            mask = df_temp["model"] == model_name
            p_raw = df_temp.loc[mask, "p"].values
            if len(p_raw) > 0 and not np.all(np.isnan(p_raw)):
                valid = ~np.isnan(p_raw)
                p_fdr = np.full_like(p_raw, np.nan)
                if valid.sum() > 0:
                    _, p_fdr[valid], _, _ = multipletests(
                        p_raw[valid], method="fdr_bh", alpha=0.05)
                df_temp.loc[mask, "p_fdr"] = p_fdr
                df_temp.loc[mask, "sig_fdr"] = p_fdr < 0.05

        df_temp.to_csv(RESULTS_DIR / "temperature_effects.csv", index=False)
        print(f"\nSaved: results/temperature_effects.csv ({len(df_temp)} rows, with FDR)")

        # Print key interactions
        print(f"\n{'=' * 60}")
        print("Key Temperature × Predictor Interactions")
        print(f"{'=' * 60}")
        key_preds = ["TempL_x_Inc_L", "TempQ_x_Inc_L",
                     "TempL_x_O", "TempL_x_C", "TempL_x_E", "TempL_x_A", "TempL_x_N"]
        for model_name in models:
            df_m = df_temp[df_temp["model"] == model_name]
            print(f"\n  {model_name}:")
            for _, row in df_m[df_m["predictor"].isin(key_preds)].iterrows():
                sig = "*" if row["p"] < 0.05 else " "
                print(f"    {row['predictor']:20s}  β={row['beta']:+.4f}  "
                      f"p={row['p']:.4f} {sig}")
    else:
        print("\nNo temperature interaction results.")

    # 3. Choice_A rate by model × temperature
    print(f"\n--- Choice_A Rate by Model × Temperature ---")
    summary = df[
        (df["dir_condition"] == "A_nl_consumer") & (df["is_v4"] == 0)
    ].groupby(["dir_model", "temperature"]).agg(
        n=("Choice_A", "count"),
        choice_a_rate=("Choice_A", "mean"),
    ).reset_index()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
