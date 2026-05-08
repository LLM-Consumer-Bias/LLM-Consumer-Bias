"""Phase 6 Step 1: Preprocess raw JSONL files into analysis-ready DataFrame.

Input:  data/raw/*/responses.jsonl + design matrices + stimuli files
Output: results/analysis_df.parquet (one row per valid response)

Processing:
1. Read all JSONL files, extract model/temperature/condition from directory name
2. Parse responses via parse_response() → choice column ('A'/'B'/'invalid')
3. Create Choice_A binary (1 if A, 0 if B; drop invalid)
4. Merge with personas_384.csv on ProfileID → demographics + Big Five
5. Merge with assignment.csv on ProfileID+ScenarioID → Version
6. Code form variables: Order (−1/+1), Format_1, Format_2 (Helmert)
7. Code temperature: Temp_L (linear), Temp_Q (quadratic) orthogonal polynomials
8. Code Scenario as factor + Involvement_linear
9. Compute trait positions from hash(task_id)
10. Filter: remove invalid/error, flag V4 rows

Usage:
    python scripts/05_analysis/preprocessing.py
"""

import pathlib
import sys

import jsonlines
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "04_parse"))
sys.path.insert(0, str(ROOT / "scripts" / "02_render"))

from parse_responses import parse_response
from render_prompts import get_trait_positions

RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"


def parse_directory_name(dirname: str) -> dict:
    """Extract model, condition, temperature from directory name.

    Format: {model_name}[_{condition}]_t{temp}
    Examples:
        phi4_t1.0              → model=phi4, condition=A_nl_consumer, temp=1.0
        mistral-small3.2_t0.5  → model=mistral-small3.2, condition=A_nl_consumer, temp=0.5
        mistral-small3.2_B_structured_t1.0 → model=mistral-small3.2, condition=B_structured, temp=1.0
        qwen3-32b_D_general_psych_t1.0     → model=qwen3-32b, condition=D_general_psych, temp=1.0
    """
    # Temperature is always the last part after _t
    parts = dirname.rsplit("_t", 1)
    if len(parts) != 2:
        raise ValueError(f"Cannot parse directory name: {dirname}")

    prefix = parts[0]
    temp = float(parts[1])

    # Known conditions (non-default)
    conditions = ["B_structured", "C_coded", "D_general_psych"]
    condition = "A_nl_consumer"
    model = prefix

    for cond in conditions:
        if f"_{cond}" in prefix:
            condition = cond
            model = prefix.replace(f"_{cond}", "")
            break

    return {"model": model, "condition": condition, "temperature": temp}


def load_all_jsonl() -> pd.DataFrame:
    """Load all JSONL files from data/raw/ directories."""
    records = []
    raw_dirs = sorted(RAW_DIR.iterdir()) if RAW_DIR.exists() else []

    for d in raw_dirs:
        if not d.is_dir():
            continue
        jsonl_file = d / "responses.jsonl"
        if not jsonl_file.exists():
            print(f"  SKIP: {d.name} (no responses.jsonl)")
            continue

        # Parse directory metadata
        dir_meta = parse_directory_name(d.name)
        print(f"  Loading {d.name} ... ", end="", flush=True)

        count = 0
        with jsonlines.open(jsonl_file) as reader:
            for record in reader:
                record["dir_model"] = dir_meta["model"]
                record["dir_condition"] = dir_meta["condition"]
                record["dir_temperature"] = dir_meta["temperature"]
                records.append(record)
                count += 1

        print(f"{count:,} records")

    if not records:
        raise RuntimeError("No JSONL records found in data/raw/")

    return pd.DataFrame(records)


def parse_choices(df: pd.DataFrame) -> pd.DataFrame:
    """Parse response text into choice column."""
    choices = []
    for _, row in df.iterrows():
        if row.get("status") == "error":
            choices.append("error")
        else:
            response_text = row.get("response") or ""
            form_id = row.get("form_id", "")
            choice = parse_response(response_text, form_id)
            choices.append(choice)
    df["choice"] = choices
    return df


def code_form_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Code Order and Format (Helmert) from form_id.

    Order: A-first = -1, B-first = +1
    Format (Helmert coding):
        AB:  Format_1 = -1, Format_2 = -1
        REP: Format_1 = +1, Format_2 = -1
        CMP: Format_1 =  0, Format_2 = +2
    """
    order_map = {
        "AB_Afirst": -1, "AB_Bfirst": 1,
        "REP_Afirst": -1, "REP_Bfirst": 1,
        "CMP_AoverB": -1, "CMP_BoverA": 1,
    }
    format_helmert = {
        "AB_Afirst": (-1, -1), "AB_Bfirst": (-1, -1),
        "REP_Afirst": (1, -1), "REP_Bfirst": (1, -1),
        "CMP_AoverB": (0, 2), "CMP_BoverA": (0, 2),
    }

    df["Order"] = df["form_id"].map(order_map)
    df["Format_1"] = df["form_id"].map(lambda x: format_helmert.get(x, (0, 0))[0])
    df["Format_2"] = df["form_id"].map(lambda x: format_helmert.get(x, (0, 0))[1])
    return df


def code_temperature(df: pd.DataFrame) -> pd.DataFrame:
    """Code temperature with orthogonal polynomial contrasts.

    Temp_L (linear):    0.0 = -1, 0.5 = 0, 1.0 = +1
    Temp_Q (quadratic): 0.0 = +1, 0.5 = -2, 1.0 = +1
    """
    temp_l_map = {0.0: -1, 0.5: 0, 1.0: 1}
    temp_q_map = {0.0: 1, 0.5: -2, 1.0: 1}

    df["Temp_L"] = df["temperature"].map(temp_l_map)
    df["Temp_Q"] = df["temperature"].map(temp_q_map)
    return df


def code_scenario(df: pd.DataFrame) -> pd.DataFrame:
    """Code scenario involvement.

    S5 (toothbrush) = -2, S4 (jacket) = -1, S3 (headphones) = 0,
    S1 (smartphone) = +1, S2 (laptop) = +2
    """
    involvement_map = {"S5": -2, "S4": -1, "S3": 0, "S1": 1, "S2": 2}
    df["Involvement_linear"] = df["scenario_id"].map(involvement_map)
    return df


def compute_trait_positions_col(df: pd.DataFrame) -> pd.DataFrame:
    """Compute trait position (0-4) from hash(task_id) for positional bias analysis.

    Uses the same trait_id format as render_prompts: ProfileID:ScenarioID:FormID:Rep
    """
    positions = {"trait_pos_O": [], "trait_pos_C": [], "trait_pos_E": [],
                 "trait_pos_A": [], "trait_pos_N": []}

    for _, row in df.iterrows():
        # Reconstruct the trait-order task_id (same as used in render_prompts)
        trait_task_id = f"{row['profile_id']}:{row['scenario_id']}:{row['form_id']}:{row['replication']}"
        pos = get_trait_positions(trait_task_id)
        for trait in ["O", "C", "E", "A", "N"]:
            positions[f"trait_pos_{trait}"].append(pos[trait])

    for col, vals in positions.items():
        df[col] = vals

    return df


def main():
    print("=" * 60)
    print("Phase 6 Step 1: Preprocessing")
    print("=" * 60)

    # 1. Load all JSONL
    print("\n1. Loading JSONL files...")
    df = load_all_jsonl()
    print(f"   Total records: {len(df):,}")

    # 2. Parse choices
    print("\n2. Parsing responses...")
    df = parse_choices(df)
    status_counts = df["choice"].value_counts()
    print(f"   Choice distribution:\n{status_counts.to_string()}")

    # 3. Load design matrices
    print("\n3. Merging with design matrices...")
    personas = pd.read_csv(ROOT / "data" / "design" / "personas_384.csv")
    assignment = pd.read_csv(ROOT / "data" / "stimuli" / "assignment.csv")

    # Merge personas (demographics + Big Five)
    persona_cols = ["ProfileID", "DemoID", "Age", "Gender", "Income", "Region",
                    "Age_L", "Gender_code", "Inc_L", "Inc_Q", "Region_code",
                    "Region_text", "B5ID", "O", "C", "E", "A", "N"]
    df = df.merge(personas[persona_cols], left_on="profile_id", right_on="ProfileID", how="left")

    # Merge assignment (Version per ProfileID × ScenarioID)
    df = df.merge(
        assignment[["ProfileID", "ScenarioID", "Version"]],
        left_on=["profile_id", "scenario_id"],
        right_on=["ProfileID", "ScenarioID"],
        how="left",
        suffixes=("", "_assign"),
    )
    # Use assignment version (more reliable than parsing), but keep original for QC
    if "version" in df.columns and "Version" in df.columns:
        df["version_from_record"] = df["version"]
        df["version"] = df["Version"]

    print(f"   After merge: {len(df):,} rows, {df.columns.size} columns")

    # 4. Code form variables
    print("\n4. Coding form variables (Order, Format_1, Format_2)...")
    df = code_form_variables(df)

    # 5. Code temperature
    print("\n5. Coding temperature (Temp_L, Temp_Q)...")
    df = code_temperature(df)

    # 6. Code scenario
    print("\n6. Coding scenario (Involvement_linear)...")
    df = code_scenario(df)

    # 7. Compute trait positions
    print("\n7. Computing trait positions for positional bias analysis...")
    # Only compute for valid rows with required fields
    valid_mask = df["profile_id"].notna() & df["scenario_id"].notna() & df["form_id"].notna()
    df_valid = df[valid_mask].copy()
    df_valid = compute_trait_positions_col(df_valid)
    # Merge back
    for col in ["trait_pos_O", "trait_pos_C", "trait_pos_E", "trait_pos_A", "trait_pos_N"]:
        df[col] = np.nan
    df.loc[valid_mask, ["trait_pos_O", "trait_pos_C", "trait_pos_E",
                        "trait_pos_A", "trait_pos_N"]] = df_valid[
        ["trait_pos_O", "trait_pos_C", "trait_pos_E",
         "trait_pos_A", "trait_pos_N"]].values

    # 8. Create Choice_A binary (1 if A, 0 if B)
    print("\n8. Creating Choice_A binary variable...")
    df["Choice_A"] = np.where(df["choice"] == "A", 1,
                              np.where(df["choice"] == "B", 0, np.nan))

    # 9. Flag V4 (dominance check — keep but mark for exclusion)
    df["is_v4"] = (df["version"] == "V4").astype(int)

    # 10. Filter
    print("\n9. Filtering...")
    n_before = len(df)
    n_errors = (df["choice"] == "error").sum()
    n_invalid = (df["choice"] == "invalid").sum()
    n_v4 = df["is_v4"].sum()

    # Keep only valid choices (A or B)
    df_analysis = df[df["Choice_A"].notna()].copy()
    df_analysis["Choice_A"] = df_analysis["Choice_A"].astype(int)
    n_after = len(df_analysis)

    # Separate V4 for QC reporting
    df_v4 = df_analysis[df_analysis["is_v4"] == 1].copy()
    df_main = df_analysis[df_analysis["is_v4"] == 0].copy()

    print(f"   Before filter: {n_before:,}")
    print(f"   Errors removed: {n_errors:,}")
    print(f"   Invalid removed: {n_invalid:,}")
    print(f"   Valid responses: {n_after:,}")
    print(f"   V4 (kept but flagged): {len(df_v4):,}")
    print(f"   Primary analysis rows (non-V4): {len(df_main):,}")

    # 11. Save
    print("\n10. Saving outputs...")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Full dataset (including V4, for QC)
    df_analysis.to_parquet(RESULTS_DIR / "analysis_df.parquet", index=False)
    print(f"   → results/analysis_df.parquet ({len(df_analysis):,} rows)")

    # Primary analysis dataset (excluding V4)
    df_main.to_parquet(RESULTS_DIR / "analysis_df_main.parquet", index=False)
    print(f"   → results/analysis_df_main.parquet ({len(df_main):,} rows)")

    # Summary stats
    print("\n" + "=" * 60)
    print("Summary by model × temperature × condition:")
    print("=" * 60)
    summary = df_analysis.groupby(["dir_model", "dir_temperature", "dir_condition"]).agg(
        n_rows=("Choice_A", "count"),
        choice_a_rate=("Choice_A", "mean"),
        n_profiles=("profile_id", "nunique"),
    ).reset_index()
    print(summary.to_string(index=False))

    # Column inventory
    print("\n" + "=" * 60)
    print(f"Final columns ({len(df_analysis.columns)}):")
    print("=" * 60)
    for col in sorted(df_analysis.columns):
        print(f"  {col}: {df_analysis[col].dtype}")

    print("\nPreprocessing complete.")
    return df_analysis


if __name__ == "__main__":
    main()
