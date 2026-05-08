"""Generate assignment.csv: ProfileID × ScenarioID → Version (V1–V4).

Design.md section 4.5:
  - 96 profiles per version per scenario (384 / 4 = 96)
  - Stratified by DemoID: each of 24 demo groups evenly split across 4 versions
    (16 B5 profiles per demo group → 4 per version)
  - Deterministic via seeded shuffle
"""

import numpy as np
import pandas as pd
import pathlib

DATA_DIR = pathlib.Path(__file__).resolve().parents[2] / "data"
SCENARIOS = ["S1", "S2", "S3", "S4", "S5"]
VERSIONS = ["V1", "V2", "V3", "V4"]
SEED = 20250101


def make_assignment() -> pd.DataFrame:
    personas = pd.read_csv(DATA_DIR / "design" / "personas_384.csv")
    rng = np.random.default_rng(SEED)

    rows = []
    for s_idx, s in enumerate(SCENARIOS):
        # For each scenario, assign versions stratified by DemoID
        for demo_id in personas["DemoID"].unique():
            demo_profiles = personas[personas["DemoID"] == demo_id]["ProfileID"].tolist()
            assert len(demo_profiles) == 16, f"{demo_id}: expected 16 profiles"

            # Shuffle and split into 4 groups of 4
            shuffled = demo_profiles.copy()
            rng.shuffle(shuffled)
            for v_idx, v in enumerate(VERSIONS):
                for pid in shuffled[v_idx * 4 : (v_idx + 1) * 4]:
                    rows.append({
                        "ProfileID": pid,
                        "ScenarioID": s,
                        "Version": v,
                    })

    df = pd.DataFrame(rows)

    # Merge DemoID and B5ID for convenience
    personas_slim = personas[["ProfileID", "DemoID", "B5ID"]]
    df = df.merge(personas_slim, on="ProfileID")
    df = df[["ProfileID", "DemoID", "B5ID", "ScenarioID", "Version"]]

    return df


if __name__ == "__main__":
    df = make_assignment()
    out = DATA_DIR / "stimuli" / "assignment.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} assignments to {out}")

    # Balance check
    for s in SCENARIOS:
        subset = df[df["ScenarioID"] == s]
        counts = subset["Version"].value_counts().sort_index()
        print(f"  {s}: {dict(counts)}")
