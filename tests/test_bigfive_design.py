"""Tests for Big Five 2^(5-1) Resolution V fractional factorial design."""

import sys
import pathlib
from itertools import combinations

import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "01_generate"))

from make_bigfive import make_bigfive

TRAITS = ["O", "C", "E", "A", "N"]


@pytest.fixture
def df():
    return make_bigfive()


def test_row_count(df):
    assert len(df) == 16


def test_column_balance(df):
    """Each trait has exactly 8 × (+1) and 8 × (-1)."""
    for t in TRAITS:
        assert df[t].value_counts()[+1] == 8, f"{t} +1 count != 8"
        assert df[t].value_counts()[-1] == 8, f"{t} -1 count != 8"


def test_pairwise_balance(df):
    """All 10 pairs have 4-4-4-4 distribution."""
    for t1, t2 in combinations(TRAITS, 2):
        for v1 in [-1, +1]:
            for v2 in [-1, +1]:
                count = len(df[(df[t1] == v1) & (df[t2] == v2)])
                assert count == 4, f"({t1}={v1}, {t2}={v2}): {count} != 4"


def test_defining_relation(df):
    """N = O × C × E × A for all rows."""
    computed = df["O"] * df["C"] * df["E"] * df["A"]
    assert (df["N"] == computed).all()


def test_unique_ids(df):
    assert df["B5ID"].nunique() == 16


def test_values_binary(df):
    """All trait values are exactly -1 or +1."""
    for t in TRAITS:
        assert set(df[t].unique()) == {-1, +1}, f"{t} has non-binary values"
