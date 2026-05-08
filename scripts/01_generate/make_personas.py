"""Generate 384 persona profiles: 24 demographics × 16 Big Five = 384."""

import pandas as pd
import pathlib

DATA_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "design"


def make_personas() -> pd.DataFrame:
    demo = pd.read_csv(DATA_DIR / "demographics_24.csv")
    b5 = pd.read_csv(DATA_DIR / "bigfive_16.csv")

    assert len(demo) == 24, f"Expected 24 demographics, got {len(demo)}"
    assert len(b5) == 16, f"Expected 16 B5 profiles, got {len(b5)}"

    # Cross join
    demo["_key"] = 1
    b5["_key"] = 1
    personas = demo.merge(b5, on="_key").drop("_key", axis=1)

    # Create ProfileID: P001–P384
    personas.insert(0, "ProfileID", [f"P{i:03d}" for i in range(1, len(personas) + 1)])

    assert len(personas) == 384, f"Expected 384 personas, got {len(personas)}"
    return personas


if __name__ == "__main__":
    df = make_personas()
    out = DATA_DIR / "personas_384.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} personas to {out}")
    print(f"Columns: {list(df.columns)}")
