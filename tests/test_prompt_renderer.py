"""Tests for prompt renderer."""

import sys
import pathlib

import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "02_render"))
sys.path.insert(0, str(ROOT / "scripts" / "01_generate"))

from render_prompts import render_prompt, get_trait_positions, SYSTEM_HEADER


@pytest.fixture
def sample_profile():
    """D01 × B08 from design.md section 8."""
    return {
        "ProfileID": "P008",
        "DemoID": "D01",
        "Age": 25,
        "Gender": "Male",
        "Income": 25000,
        "Region": "USA",
        "Region_text": "the United States",
        "Age_L": -1,
        "Gender_code": -1,
        "Inc_L": -1,
        "Inc_Q": 1,
        "Region_code": -1,
        "B5ID": "B08",
        "O": 1, "C": 1, "E": 1, "A": -1, "N": -1,
    }


def test_system_header_no_cueing():
    """System header must not mention income, personality, financial, etc."""
    forbidden = ["income", "financial", "personality", "money", "budget", "price"]
    lower = SYSTEM_HEADER.lower()
    for word in forbidden:
        assert word not in lower, f"System header contains '{word}'"


def test_system_header_has_key_phrase():
    assert "priorities and tendencies" in SYSTEM_HEADER


def test_render_basic(sample_profile):
    system, user = render_prompt(sample_profile, "S1", "V2", "AB_Afirst", 1)
    assert "25-year-old man" in user
    assert "$25,000" in user
    assert "the United States" in user
    assert "=== YOUR PROFILE ===" in user
    assert "=== SCENARIO S1:" in user
    assert "Reply 'A' or 'B'" in user


def test_no_effect_codes_in_prompt(sample_profile):
    """Condition A (primary): no numeric codes in prompt."""
    _, user = render_prompt(sample_profile, "S1", "V2", "AB_Afirst", 1)
    assert "O=" not in user
    assert "C=" not in user
    assert "Inc_L" not in user
    assert "Age_L" not in user


def test_trait_order_varies(sample_profile):
    """Different task_ids should produce different trait orders."""
    orders = set()
    for rep in range(1, 10):
        positions = get_trait_positions(f"P008:S1:AB_Afirst:{rep}")
        orders.add(tuple(positions[t] for t in "OCEAN"))
    assert len(orders) > 1, "All replications have same trait order"


def test_trait_order_deterministic(sample_profile):
    """Same task_id always produces same order."""
    p1 = get_trait_positions("P008:S1:AB_Afirst:1")
    p2 = get_trait_positions("P008:S1:AB_Afirst:1")
    assert p1 == p2


def test_option_order_bfirst(sample_profile):
    """AB_Bfirst should show Option B before Option A."""
    _, user = render_prompt(sample_profile, "S1", "V2", "AB_Bfirst", 1)
    pos_b = user.index("Option B:")
    pos_a = user.index("Option A:")
    assert pos_b < pos_a, "B-first form should show Option B before A"


def test_condition_b_structured(sample_profile):
    """Condition B uses labels, not NL."""
    _, user = render_prompt(sample_profile, "S1", "V2", "AB_Afirst", 1,
                            condition="B_structured")
    assert "Age: 25" in user
    assert "O: High" in user
    assert "A: Low" in user


def test_condition_c_coded(sample_profile):
    """Condition C uses effect codes."""
    _, user = render_prompt(sample_profile, "S1", "V2", "AB_Afirst", 1,
                            condition="C_coded")
    assert "Inc_L: -1" in user
    assert "O: 1" in user


def test_condition_d_general_psych(sample_profile):
    """Condition D uses BFI-2-S wording, not consumer framing."""
    _, user = render_prompt(sample_profile, "S1", "V2", "AB_Afirst", 1,
                            condition="D_general_psych")
    # BFI-2-S wording should contain general personality words
    assert "original" in user or "curious" in user or "talkative" in user
    # Should NOT contain consumer-specific words
    assert "purchases" not in user.split("SCENARIO")[0]


def test_all_5_traits_present(sample_profile):
    """All 5 trait descriptions should appear in the prompt."""
    _, user = render_prompt(sample_profile, "S1", "V2", "AB_Afirst", 1)
    profile_section = user.split("=== END PROFILE ===")[0]
    # Count bullet points (trait lines start with "- ")
    trait_lines = [l for l in profile_section.split("\n") if l.strip().startswith("- ")]
    assert len(trait_lines) == 5, f"Expected 5 trait lines, got {len(trait_lines)}"
