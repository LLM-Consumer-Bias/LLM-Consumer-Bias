"""Generate 16 Big Five profiles: 2^(5-1) Resolution V fractional factorial.

Defining relation: I = OCEAN  →  N = O × C × E × A
"""

import pandas as pd
import itertools


def make_bigfive() -> pd.DataFrame:
    """Generate the 16-run Resolution V design from design.md section 3.3."""
    rows = []
    # Full 2^4 for O, C, E, A; N = O*C*E*A
    for i, (o, c, e, a) in enumerate(
        itertools.product([-1, +1], repeat=4), start=1
    ):
        n = o * c * e * a  # defining relation I = OCEAN
        rows.append({
            "B5ID": f"B{i:02d}",
            "O": o,
            "C": c,
            "E": e,
            "A": a,
            "N": n,
        })
    df = pd.DataFrame(rows)
    assert len(df) == 16, f"Expected 16 rows, got {len(df)}"
    return df


def verify_balance(df: pd.DataFrame) -> None:
    """Verify Resolution V properties: column balance and pairwise balance."""
    traits = ["O", "C", "E", "A", "N"]

    # 1. Column balance: each trait has 8 × (+1) and 8 × (-1)
    for t in traits:
        counts = df[t].value_counts()
        assert counts.get(+1, 0) == 8, f"{t}: expected 8 × (+1), got {counts.get(+1, 0)}"
        assert counts.get(-1, 0) == 8, f"{t}: expected 8 × (-1), got {counts.get(-1, 0)}"

    # 2. Pairwise balance: all 10 pairs have 4-4-4-4
    from itertools import combinations
    for t1, t2 in combinations(traits, 2):
        for v1 in [-1, +1]:
            for v2 in [-1, +1]:
                count = len(df[(df[t1] == v1) & (df[t2] == v2)])
                assert count == 4, (
                    f"Pair ({t1}={v1}, {t2}={v2}): expected 4, got {count}"
                )

    # 3. Defining relation: N = O*C*E*A for all rows
    computed_n = df["O"] * df["C"] * df["E"] * df["A"]
    assert (df["N"] == computed_n).all(), "Defining relation N = O*C*E*A violated"

    print("All balance checks passed:")
    print("  - 5 columns: 8/8 each")
    print("  - 10 pairs: 4-4-4-4 each")
    print("  - Defining relation I = OCEAN: verified")


if __name__ == "__main__":
    import pathlib

    df = make_bigfive()
    verify_balance(df)

    out = pathlib.Path(__file__).resolve().parents[2] / "data" / "design" / "bigfive_16.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} B5 profiles to {out}")
