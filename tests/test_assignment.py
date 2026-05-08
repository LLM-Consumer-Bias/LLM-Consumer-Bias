"""Tests for version assignment balance."""

import sys
import pathlib

import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "01_generate"))

from make_assignment import make_assignment

SCENARIOS = ["S1", "S2", "S3", "S4", "S5"]
VERSIONS = ["V1", "V2", "V3", "V4"]


@pytest.fixture
def df():
    return make_assignment()


def test_total_rows(df):
    """384 profiles × 5 scenarios = 1920."""
    assert len(df) == 1920


def test_96_per_version_per_scenario(df):
    """Each scenario has exactly 96 profiles per version."""
    for s in SCENARIOS:
        subset = df[df["ScenarioID"] == s]
        counts = subset["Version"].value_counts()
        for v in VERSIONS:
            assert counts.get(v, 0) == 96, f"{s}/{v}: {counts.get(v, 0)} != 96"


def test_each_profile_one_version_per_scenario(df):
    """Each profile appears exactly once per scenario."""
    for s in SCENARIOS:
        subset = df[df["ScenarioID"] == s]
        assert subset["ProfileID"].nunique() == 384
        assert len(subset) == 384


def test_stratified_by_demo(df):
    """Each DemoID × Scenario has 4 profiles per version."""
    for s in SCENARIOS:
        subset = df[df["ScenarioID"] == s]
        for demo_id in subset["DemoID"].unique():
            demo_sub = subset[subset["DemoID"] == demo_id]
            counts = demo_sub["Version"].value_counts()
            for v in VERSIONS:
                assert counts.get(v, 0) == 4, (
                    f"{s}/{demo_id}/{v}: {counts.get(v, 0)} != 4"
                )


def test_all_profiles_present(df):
    """All 384 ProfileIDs appear."""
    assert df["ProfileID"].nunique() == 384


def test_deterministic(df):
    """Running twice produces identical results."""
    df2 = make_assignment()
    pd.testing.assert_frame_equal(df, df2)
