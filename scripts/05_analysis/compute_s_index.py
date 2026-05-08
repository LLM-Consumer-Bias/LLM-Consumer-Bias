"""Phase 6 Step 6: Compute S-index (Stereotype Index) — design.md §15.

S(m) = |β_Inc_L(m)| / max(|β_O|, |β_C|, |β_E|, |β_A|, |β_N|)

S_anchored(m) = S(m) / S_literature  (at 1.5, 2.0, 2.5)

Input:  results/per_model_coefficients.csv
Output: results/s_index.csv (per model, per temperature)

Usage:
    python scripts/05_analysis/compute_s_index.py
"""

import pathlib

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"

S_LITERATURE_VALUES = [1.5, 2.0, 2.5]

B5_TRAITS = ["O", "C", "E", "A", "N"]


def compute_s_index(df_coefs: pd.DataFrame) -> pd.DataFrame:
    """Compute S-index per model per temperature from per-model coefficients."""
    results = []

    groups = df_coefs.groupby(["model", "temperature"])
    for (model_name, temp), group in groups:
        # Get |β_Inc_L|
        inc_l_row = group[group["predictor"] == "Inc_L"]
        if len(inc_l_row) == 0:
            continue
        beta_inc_l = abs(inc_l_row["beta"].iloc[0])
        p_inc_l = inc_l_row["p_raw"].iloc[0]

        # Get max(|β_trait|) for B5 traits
        trait_betas = {}
        for trait in B5_TRAITS:
            trait_row = group[group["predictor"] == trait]
            if len(trait_row) > 0:
                trait_betas[trait] = abs(trait_row["beta"].iloc[0])

        if not trait_betas:
            continue

        max_trait = max(trait_betas.values())
        max_trait_name = max(trait_betas, key=trait_betas.get)

        # S-index
        if max_trait > 0:
            s_index = beta_inc_l / max_trait
        else:
            s_index = np.inf if beta_inc_l > 0 else 1.0

        # S_anchored at multiple literature benchmarks
        result = {
            "model": model_name,
            "temperature": temp,
            "beta_Inc_L": inc_l_row["beta"].iloc[0],
            "abs_beta_Inc_L": beta_inc_l,
            "p_Inc_L": p_inc_l,
            "max_abs_beta_B5": max_trait,
            "max_B5_trait": max_trait_name,
            "s_index": s_index,
        }

        # Individual trait betas
        for trait in B5_TRAITS:
            t_row = group[group["predictor"] == trait]
            if len(t_row) > 0:
                result[f"beta_{trait}"] = t_row["beta"].iloc[0]
                result[f"abs_beta_{trait}"] = abs(t_row["beta"].iloc[0])
                result[f"p_{trait}"] = t_row["p_raw"].iloc[0]

        # S_anchored at each literature benchmark
        for s_lit in S_LITERATURE_VALUES:
            result[f"s_anchored_{s_lit}"] = s_index / s_lit

        # Demographic betas for context
        for pred in ["Inc_Q", "Age_L", "Gender_code", "Region_code"]:
            pred_row = group[group["predictor"] == pred]
            if len(pred_row) > 0:
                result[f"beta_{pred}"] = pred_row["beta"].iloc[0]

        results.append(result)

    return pd.DataFrame(results)


def classify_archetypes(df_s: pd.DataFrame) -> pd.DataFrame:
    """Classify models into archetypes via k-means (k=3) on S-index.

    Only use primary temperature (t=1.0) for classification.
    """
    df_primary = df_s[df_s["temperature"] == 1.0].copy()

    if len(df_primary) < 3:
        print("  Too few models for k-means (need ≥3). Skipping archetype classification.")
        df_s["archetype"] = "insufficient_data"
        return df_s

    # k-means on S-index
    s_values = df_primary[["s_index"]].values
    n_clusters = min(3, len(df_primary))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_primary["cluster"] = kmeans.fit_predict(s_values)

    # Label clusters by mean S-index (lower = balanced, higher = stereotyping)
    cluster_means = df_primary.groupby("cluster")["s_index"].mean().sort_values()
    label_map = {}
    labels = ["balanced", "moderate", "stereotyping"][:n_clusters]
    for i, (cluster_id, _) in enumerate(cluster_means.items()):
        label_map[cluster_id] = labels[i]

    df_primary["archetype"] = df_primary["cluster"].map(label_map)
    df_primary = df_primary.drop(columns=["cluster"])

    # Propagate archetypes to all temperatures
    archetype_map = dict(zip(df_primary["model"], df_primary["archetype"]))
    df_s["archetype"] = df_s["model"].map(archetype_map).fillna("unclassified")

    return df_s


def main():
    print("=" * 60)
    print("Phase 6 Step 6: S-Index Computation")
    print("=" * 60)

    # Load per-model coefficients
    coefs_path = RESULTS_DIR / "per_model_coefficients.csv"
    if not coefs_path.exists():
        print("ERROR: results/per_model_coefficients.csv not found. "
              "Run per_model_analysis.py first.")
        return

    df_coefs = pd.read_csv(coefs_path)
    print(f"Loaded {len(df_coefs)} coefficient rows")

    # Compute S-index
    print("\nComputing S-index...")
    df_s = compute_s_index(df_coefs)

    if len(df_s) == 0:
        print("No S-index values computed!")
        return

    # Archetype classification
    print("\nClassifying archetypes (k-means)...")
    df_s = classify_archetypes(df_s)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_s.to_csv(RESULTS_DIR / "s_index.csv", index=False)
    print(f"\nSaved: results/s_index.csv ({len(df_s)} rows)")

    # Print results
    print(f"\n{'=' * 60}")
    print("S-Index Results")
    print(f"{'=' * 60}")

    for temp in sorted(df_s["temperature"].unique()):
        print(f"\n--- Temperature = {temp} ---")
        df_t = df_s[df_s["temperature"] == temp].sort_values("s_index", ascending=False)
        for _, row in df_t.iterrows():
            arch = f" [{row['archetype']}]" if row.get("archetype") else ""
            print(f"  {row['model']:25s}  S={row['s_index']:.2f}  "
                  f"|β_Inc_L|={row['abs_beta_Inc_L']:.4f}  "
                  f"max|β_B5|={row['max_abs_beta_B5']:.4f} ({row['max_B5_trait']})"
                  f"{arch}")

            # S_anchored
            for s_lit in S_LITERATURE_VALUES:
                s_anch = row[f"s_anchored_{s_lit}"]
                interpretation = "≈ human" if 0.7 < s_anch < 1.3 else \
                                 "over-stereotyping" if s_anch > 1.3 else "under-stereotyping"
                print(f"    S_anchored(S_lit={s_lit}): {s_anch:.2f} ({interpretation})")

    # Temperature trajectory
    if df_s["temperature"].nunique() > 1:
        print(f"\n{'=' * 60}")
        print("S-Index Temperature Trajectory")
        print(f"{'=' * 60}")
        pivot = df_s.pivot_table(index="model", columns="temperature", values="s_index")
        print(pivot.to_string())


if __name__ == "__main__":
    main()
