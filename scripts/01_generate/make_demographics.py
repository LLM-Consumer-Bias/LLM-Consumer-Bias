"""Generate 24 demographic profiles: 2 Age × 2 Gender × 3 Income × 2 Region."""

import itertools
import pandas as pd

AGE_LEVELS = [25, 55]
GENDER_LEVELS = ["Male", "Female"]
INCOME_LEVELS = [25000, 65000, 120000]
REGION_LEVELS = ["USA", "Japan"]

# Effect codes (from design.md section 3.1)
AGE_CODE = {25: -1, 55: +1}
GENDER_CODE = {"Male": -1, "Female": +1}
INCOME_CODE_L = {25000: -1, 65000: 0, 120000: +1}   # linear
INCOME_CODE_Q = {25000: +1, 65000: -2, 120000: +1}   # quadratic
REGION_CODE = {"USA": -1, "Japan": +1}

# Region prompt text
REGION_TEXT = {"USA": "the United States", "Japan": "Japan"}


def make_demographics() -> pd.DataFrame:
    rows = []
    for i, (age, gender, income, region) in enumerate(
        itertools.product(AGE_LEVELS, GENDER_LEVELS, INCOME_LEVELS, REGION_LEVELS),
        start=1,
    ):
        rows.append({
            "DemoID": f"D{i:02d}",
            "Age": age,
            "Gender": gender,
            "Income": income,
            "Region": region,
            "Age_L": AGE_CODE[age],
            "Gender_code": GENDER_CODE[gender],
            "Inc_L": INCOME_CODE_L[income],
            "Inc_Q": INCOME_CODE_Q[income],
            "Region_code": REGION_CODE[region],
            "Region_text": REGION_TEXT[region],
        })
    df = pd.DataFrame(rows)
    assert len(df) == 24, f"Expected 24 rows, got {len(df)}"
    return df


if __name__ == "__main__":
    import pathlib

    out = pathlib.Path(__file__).resolve().parents[2] / "data" / "design" / "demographics_24.csv"
    df = make_demographics()
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} demographic profiles to {out}")
