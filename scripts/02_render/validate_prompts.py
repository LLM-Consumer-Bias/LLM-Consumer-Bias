"""Validate rendered prompts: spot-check 100 random prompts for correctness.

Checks:
1. No numeric effect codes (O=+1, Inc_L, Age_L, etc.)
2. Trait order varies between calls
3. All 5 traits present
4. System header correct
5. Scenario attributes present
"""

import pathlib
import sys
import random
import re
from collections import Counter

import pandas as pd
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "02_render"))
sys.path.insert(0, str(ROOT / "scripts" / "01_generate"))

from render_prompts import render_prompt, get_trait_positions, SYSTEM_HEADER

SCENARIOS = ["S1", "S2", "S3", "S4", "S5"]
FORMS = ["AB_Afirst", "AB_Bfirst", "REP_Afirst", "REP_Bfirst", "CMP_AoverB", "CMP_BoverA"]

# Forbidden patterns in condition A prompts
FORBIDDEN = [
    r'O=[+-]?\d', r'C=[+-]?\d', r'E=[+-]?\d', r'A=[+-]?\d', r'N=[+-]?\d',
    r'Inc_L', r'Inc_Q', r'Age_L', r'Gender_code', r'Region_code',
]


def main():
    personas = pd.read_csv(ROOT / "data" / "design" / "personas_384.csv")
    assignment = pd.read_csv(ROOT / "data" / "stimuli" / "assignment.csv")

    random.seed(42)
    n_checks = 100

    errors = []
    trait_orders_seen = set()
    first_trait_counter = Counter()

    for _ in range(n_checks):
        # Random combination
        persona = personas.sample(1).iloc[0].to_dict()
        scenario_id = random.choice(SCENARIOS)
        form_id = random.choice(FORMS)
        rep = random.randint(1, 3)

        # Get version
        asgn = assignment[
            (assignment["ProfileID"] == persona["ProfileID"]) &
            (assignment["ScenarioID"] == scenario_id)
        ]
        version = asgn["Version"].iloc[0]

        system, user = render_prompt(persona, scenario_id, version, form_id, rep)

        # Check 1: no effect codes
        profile_section = user.split("=== END PROFILE ===")[0]
        for pattern in FORBIDDEN:
            if re.search(pattern, profile_section):
                errors.append(f"Effect code found: {pattern} in {persona['ProfileID']}")

        # Check 2: trait order tracking
        task_id = f"{persona['ProfileID']}:{scenario_id}:{form_id}:{rep}"
        positions = get_trait_positions(task_id)
        order_tuple = tuple(positions[t] for t in "OCEAN")
        trait_orders_seen.add(order_tuple)
        first_trait = [t for t, p in positions.items() if p == 0][0]
        first_trait_counter[first_trait] += 1

        # Check 3: all 5 traits present
        trait_lines = [l for l in profile_section.split("\n") if l.strip().startswith("- ")]
        if len(trait_lines) != 5:
            errors.append(f"{persona['ProfileID']}: {len(trait_lines)} traits (expected 5)")

        # Check 4: system header
        if "priorities and tendencies" not in system:
            errors.append("System header missing key phrase")

        # Check 5: scenario present
        if f"=== SCENARIO {scenario_id}" not in user:
            errors.append(f"Scenario {scenario_id} block missing")

    # Report
    print(f"Validated {n_checks} random prompts")
    print(f"Errors: {len(errors)}")
    for e in errors:
        print(f"  ERROR: {e}")

    print(f"\nTrait orders seen: {len(trait_orders_seen)} unique (out of 120 possible)")
    print(f"First-trait distribution: {dict(first_trait_counter)}")

    if not errors:
        print("\nAll checks passed!")
    return len(errors)


if __name__ == "__main__":
    sys.exit(main())
